"""Pre-train the AllSet encoder with masked-node MAE and save its weights.

Thin driver: builds a store-backed encoder pinned to the DDI knowledge dataset,
streams Incidence batches directly from a populated knowledge cache (no live
retriever), runs MaskedNodeMAE, saves {"encoder": state_dict}. See
docs/superpowers/specs/2026-07-06-hnn-pretraining-design.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import torch

from knowhyper.pretrain.corpus import batched, cached_subhypergraphs
from knowhyper.pretrain.mae import MaskedNodeMAE

if TYPE_CHECKING:
    from knowhyper.encoders.allset import AllSetHyperEncoder


def _build_encoder() -> AllSetHyperEncoder:
    """Store-backed AllSet encoder pinned to the DDI knowledge dataset."""
    from primehkg import load_core_dataset

    from knowhyper.encoders.allset import AllSetHyperEncoder
    from knowhyper.ieforge.primehkg_backend import PrimeHKGFeatureStore
    from knowhyper.pins import CORE_FINGERPRINT, CORE_REPO, CORE_REVISION

    dataset = load_core_dataset(
        hub_repo_id=CORE_REPO,
        revision=CORE_REVISION,
        expected_fingerprint=CORE_FINGERPRINT,
    )
    store = PrimeHKGFeatureStore(dataset, dim=16)
    if store.dim != 768:
        msg = (
            f"feature store dim={store.dim}, expected 768 (vector_node_v1 sidecar). "
            "Sidecar likely missing → pre-training would run on all-zero features. "
            "Aborting."
        )
        raise RuntimeError(msg)
    return AllSetHyperEncoder(
        feature_store=store, hidden_size=768, num_layers=2, num_heads=4
    )


def _build_lever_b_encoder(revision: str, fingerprint: str) -> AllSetHyperEncoder:
    """Composite-store AllSet encoder (768d) pinned to a Lever B dataset revision."""
    from primehkg import load_core_dataset

    from knowhyper.encoders.allset import AllSetHyperEncoder
    from knowhyper.features.modality_index import ModalityIndex, resolve_features_dir
    from knowhyper.ieforge.composite_feature_store import (
        ATTR_MODALITY_TO_SOURCE,
        AttributeNodeFeatureStore,
        CompositeFeatureStore,
    )
    from knowhyper.ieforge.primehkg_backend import PrimeHKGFeatureStore
    from knowhyper.pins import CORE_REPO

    dataset = load_core_dataset(
        hub_repo_id=CORE_REPO,
        revision=revision,
        expected_fingerprint=fingerprint,
    )
    anchor = PrimeHKGFeatureStore(dataset, dim=16)
    if getattr(anchor, "_index", None) is None:
        msg = (
            "PrimeHKGFeatureStore has no vector_node_index sidecar → anchor nodes "
            "would carry all-zero features. This is invisible in the cross-modal "
            "loss (which scores only attribute nodes), so pre-training would silently "
            f"run on degraded anchors. Check the vector_node_v1 sidecar for {revision}. "
            "Aborting."
        )
        raise RuntimeError(msg)
    # FEATURES_REVISION, not `revision`. `revision` is the CORE pin (the argument
    # this function takes, used for load_core_dataset above); `resolve_features_dir`
    # resolves primehkg-FEATURES, a different repo. Passing the core revision worked
    # only while both repos sat at the same tag -- the moment core went to 0.12.13 and
    # features to 0.12.14 it asked the features repo for a revision that exists only
    # in core, and died with LocalEntryNotFoundError under HF_HUB_OFFLINE. This is the
    # same wrong-repo read #169 fixed in knowhyper_feature_store/composite.yaml; the
    # pretrain path still had it.
    from knowhyper.pins import FEATURES_REVISION

    attr = AttributeNodeFeatureStore(
        modality_index=ModalityIndex(
            features_dir=resolve_features_dir(FEATURES_REVISION)
        )
    )
    # REFUSE to pretrain on all-zero attribute vectors.
    #
    # CrossModalMAE masks attribute nodes and reconstructs their RAW MODALITY VECTORS,
    # so if the store returns zeros the objective degenerates to predicting zeros from
    # zeros and the encoder learns nothing. That is exactly what happened under features
    # 0.12.14: CORE emits attr ids ending `:synonyms` while 0.12.14 published those
    # vectors as `synonym_context`, and smiles/protein had no embeddings at all
    # (primehkg #470). Three encoders were pretrained, verified present with manifests,
    # and used for every ev_warm result -- all worthless, and nothing failed.
    #
    # Probe the store the way the objective will, and abort on a dead modality.
    _probe_modalities = ("synonyms", "smiles", "sequence")
    _dead = []
    for _mod in _probe_modalities:
        _source = ATTR_MODALITY_TO_SOURCE.get(_mod, _mod)
        _hits = [
            nid
            for (nid, sm) in attr._index._map  # noqa: SLF001 - diagnostic probe
            if sm == _source
        ]
        if not _hits:
            _dead.append(f"{_mod}->{_source} (no rows in the features build)")
    if _dead:
        msg = (
            "pre-training would run on DEAD attribute modalities: "
            + "; ".join(_dead)
            + f". features_revision={FEATURES_REVISION}. CrossModalMAE reconstructs "
            "raw modality vectors, so a zero vector makes the objective degenerate and "
            "the encoder useless -- silently. Fix the features pin or the "
            "ATTR_MODALITY_TO_SOURCE mapping before pre-training. Set "
            "KNOWHYPER_ALLOW_DEAD_MODALITIES=1 to override deliberately."
        )
        if os.environ.get("KNOWHYPER_ALLOW_DEAD_MODALITIES") != "1":
            raise RuntimeError(msg)
        print(f"WARNING (override active): {msg}")  # noqa: T201
    store = CompositeFeatureStore(anchor_store=anchor, attr_store=attr)
    return AllSetHyperEncoder(
        feature_store=store, hidden_size=768, num_layers=2, num_heads=4
    )


def save_encoder_artifact(
    encoder: AllSetHyperEncoder, out: str | Path, metadata: dict[str, object]
) -> Path:
    """Atomically save an encoder and a checksum-bearing provenance manifest."""
    out_path = Path(out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid4().hex}"
    weight_tmp = out_path.with_name(f"{out_path.name}.{token}.tmp")
    manifest_path = out_path.with_name(f"{out_path.name}.manifest.json")
    manifest_tmp = manifest_path.with_name(f"{manifest_path.name}.{token}.tmp")
    try:
        torch.save({"encoder": encoder.state_dict()}, weight_tmp)
        weight_tmp.replace(out_path)
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        # Record the FEATURES pin the encoder's attribute vectors came from.
        #
        # This was missing, and it is why a whole generation of encoders went unnoticed
        # as useless. `CrossModalMAE` masks attribute nodes and reconstructs their raw
        # modality vectors, so the encoder is only as good as those vectors -- and under
        # features 0.12.14 the synonyms/smiles/sequence vectors were ALL ZERO (primehkg
        # #470). The objective degenerated to predicting zeros from zeros. The manifest
        # recorded `revision: null` (the CORE arg, unset) and nothing about FEATURES, so
        # there was no field that could have flagged it.
        #
        # `features_revision` is the pin; `attr_vector_rows` is the live row count for
        # the multimodal sources, which is the value that actually distinguishes a
        # working build from a degenerate one (0.12.14 → 37,817; 0.12.15 → 64,986).
        from knowhyper.pins import FEATURES_REVISION as _FEAT_REV

        manifest = {
            "artifact_schema_version": 2,
            "kind": "allset_pretrained_encoder",
            "sha256": digest,
            "features_revision": _FEAT_REV,
            "metadata": metadata,
        }
        manifest_tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_tmp.replace(manifest_path)
    finally:
        weight_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
    return out_path


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    # lr=3e-4 (not 1e-3): the MAE objective diverges at 1e-3 on the full ~25k-graph
    # corpus (loss descends then runs away over many steps); 3e-4 + grad-clip is stable.
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--grad-clip", type=float, default=1.0, help="max grad norm (0=off)"
    )
    ap.add_argument("--mask-ratio", type=float, default=0.15)
    ap.add_argument(
        "--max-graphs",
        type=int,
        default=0,
        help="cap graphs streamed per epoch (0 = all)",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--objective",
        choices=["mae", "ssht", "cross_modal"],
        default="mae",
        help="self-supervised objective: masked-node MAE, SS-HT, or cross-modal MAE",
    )
    ap.add_argument(
        "--entity-view",
        action="store_true",
        help="augment cached sub-HGs with entity_attribute nodes (Lever B); "
        "required for --objective cross_modal",
    )
    ap.add_argument(
        "--revision",
        default=None,
        help=(
            "primehkg CORE revision for --entity-view. Defaults to the central pin "
            "in configs/knowledge/knowhyper.yaml; pass one only to deliberately "
            "pretrain against a different build than the end task will read."
        ),
    )
    ap.add_argument(
        "--fingerprint",
        default=None,
        help="expected fingerprint; defaults to the central pin alongside --revision",
    )
    ap.add_argument(
        "--cl-weight",
        type=float,
        default=1.0,
        help="ssht: hyperedge contrastive loss weight",
    )
    ap.add_argument(
        "--attr-weight",
        type=float,
        default=3.0,
        help="ssht: attr reconstruction loss weight",
    )
    ap.add_argument(
        "--replace-rate",
        type=float,
        default=0.05,
        help="ssht: replace-noise fraction of masked nodes",
    )
    ap.add_argument(
        "--skip-modalities",
        nargs="*",
        default=None,
        help="entity_view modalities to drop (e.g. 'name')",
    )
    ap.add_argument(
        "--incidence-cache",
        default=None,
        help="dir to precompute+reuse built Incidences (avoids per-epoch "
        "augment+build_incidence, the CPU bottleneck). Built once on first epoch.",
    )
    args = ap.parse_args()
    if args.objective == "cross_modal" and not args.entity_view:
        ap.error("--objective cross_modal requires --entity-view")
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("pretrain")

    if args.entity_view:
        # Falling back to the CENTRAL pin, not a literal. A stale default here built
        # an encoder on the new sub-HG cache while reading node features from the old
        # core -- mixed conditions that invalidated the ChemProt warm-start result.
        from knowhyper.pins import CORE_FINGERPRINT, CORE_REVISION

        revision = args.revision or CORE_REVISION
        fingerprint = args.fingerprint or CORE_FINGERPRINT
        # Logged, not silent: an encoder is only comparable to an end task that read
        # the SAME core, and the manifest is the only place that pairing is visible.
        log.info("core revision=%s fingerprint=%s…", revision, fingerprint[:12])
        encoder = _build_lever_b_encoder(revision, fingerprint).to(args.device)
    else:
        encoder = _build_encoder().to(args.device)

    if args.objective == "cross_modal":
        from knowhyper.pretrain.cross_modal_mae import CrossModalMAE

        obj = CrossModalMAE(encoder, mask_ratio=args.mask_ratio).to(args.device)
    elif args.objective == "ssht":
        from knowhyper.pretrain.ssht import SSHTObjective

        obj = SSHTObjective(
            encoder,
            mask_rate=args.mask_ratio,
            replace_rate=args.replace_rate,
            sce_alpha=3.0,
            cl_weight=args.cl_weight,
            attr_weight=args.attr_weight,
        ).to(args.device)
    else:
        obj = MaskedNodeMAE(encoder, mask_ratio=args.mask_ratio).to(args.device)
    opt = torch.optim.Adam(obj.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    augmenter = None
    if args.entity_view:
        from knowhyper.pins import CORE_REPO
        from knowhyper.pipeline.entity_view_augmenter import (
            EntityViewAugmenter,
            build_entity_view_index,
            load_core_nodes_iter,
        )

        # `revision`/`fingerprint`, NOT args.* -- those are None unless the caller
        # passed them explicitly, and the pin fallback happens above at the
        # `_build_lever_b_encoder` call. Passing the raw None here sent
        # revision=None to load_core_dataset, which under HF_HUB_OFFLINE fails with
        # `Couldn't reach 'nishide-dev/primehkg-core' (OfflineModeIsEnabled)` --
        # after the encoder had already been built. Worse, off-cluster it would have
        # silently resolved to the repo's default branch: a DIFFERENT core from the
        # one the encoder and the end task read, which is exactly the mixed-condition
        # confound the log line above exists to make visible.
        index = build_entity_view_index(
            load_core_nodes_iter(CORE_REPO, revision, fingerprint),
            source_fingerprint=fingerprint,
            skip_modalities=args.skip_modalities,
        )
        augmenter = EntityViewAugmenter(index)

    def _subhg_stream():
        subs = cached_subhypergraphs(args.cache_dir)
        if args.max_graphs:
            subs = islice(subs, args.max_graphs)
        if augmenter is not None:
            from knowhyper.pretrain.corpus import augment_subhypergraphs

            subs = augment_subhypergraphs(subs, augmenter)
        return subs

    # Precompute Incidences once (augment + build_incidence is the per-epoch CPU
    # bottleneck, ~20ms/graph, and is deterministic given the sub-HG stream +
    # encoder feature store). Build on first run; subsequent epochs/runs load.
    if args.incidence_cache:
        from knowhyper.pretrain.corpus import (
            build_incidence_cache,
            incidence_cache_ready,
        )

        if not incidence_cache_ready(args.incidence_cache):
            log.info("building incidence cache at %s ...", args.incidence_cache)
            build_incidence_cache(_subhg_stream(), encoder, args.incidence_cache)
        else:
            log.info("reusing incidence cache at %s", args.incidence_cache)

    for epoch in range(args.epochs):
        gen = torch.Generator().manual_seed(args.seed + epoch)
        total, nb = 0.0, 0
        if args.incidence_cache:
            from knowhyper.pretrain.corpus import cached_incidences

            stream = cached_incidences(args.incidence_cache)
        else:
            stream = (encoder.build_incidence(sub) for sub in _subhg_stream())
        for batch in batched(stream, args.batch_size):
            opt.zero_grad()
            out = obj(batch, generator=gen)
            if int(out["n_masked"]) == 0:
                continue
            out["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(obj.parameters(), args.grad_clip)
            opt.step()
            total += out["loss"].item()
            nb += 1
        sched.step()
        log.info("epoch %d mean_loss %.4f (%d batches)", epoch, total / max(nb, 1), nb)

    metadata = {key: value for key, value in vars(args).items() if key != "out"}
    metadata["cache_dir"] = str(Path(args.cache_dir).expanduser().resolve())
    if args.incidence_cache is not None:
        metadata["incidence_cache"] = str(
            Path(args.incidence_cache).expanduser().resolve()
        )
    saved_path = save_encoder_artifact(encoder, args.out, metadata)
    log.info("saved pre-trained encoder and manifest to %s", saved_path)


if __name__ == "__main__":
    main()
