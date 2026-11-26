#!/usr/bin/env python3
"""Extract cross-attention weights from KnowHyper checkpoints.

For each validation instance, records per-token attention weight from
CLS -> each knowledge token, classified as entity / attr:* / hyperedge.

Usage:
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run python3 scripts/analysis/extract_attention.py \
    --experiment khp_ddi_xattn_ev \
    --ckpt "$KNOWHYPER_HOME/runs/re/ddi/ev_bugfix/seed_0/checkpoints/best.ckpt" \
    --output "$KNOWHYPER_HOME/analysis/ddi/cold_seed0.jsonl" \
    [-- extra hydra overrides...]

Required Hydra overrides for current experiments (passed automatically unless
you provide custom ones after `--`):
  model.gate_init=1.0 model.gate_shape=vector +model.fusion_norm=true
  dataset.batch_size=64 dataset.num_workers=4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import torch
from _utils import extract_example_ids
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from ieforge.cli._hydra_entry import (
    _CONFIGS_DIR,
    _build_collator,
    _build_dataset,
    _build_metric,
    _build_task,
    _model_instantiation_cfg,
    _wrap_knowledge_collator,
)
from ieforge.data.datamodule import IEDataModule
from ieforge.engine.lightning_module import IELightningModule
from omegaconf import OmegaConf


def classify_node(nid: str, seed_ids: set[str]) -> dict:
    is_seed = nid in seed_ids
    if nid.startswith("attr:"):
        parts = nid.split(":")
        modality = parts[-1] if len(parts) >= 3 else "unknown"
        return {"category": f"attr:{modality}", "is_seed": False, "node_id": nid}
    return {"category": "entity", "is_seed": is_seed, "node_id": nid}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", required=True, help="Hydra experiment config name")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("extra", nargs="*", help="Additional Hydra overrides (after --)")
    return p.parse_args()


def _extract_label_dict(item, *, kind: str) -> dict[tuple[str, str], str]:
    """Build a (head, tail) -> label dict from a prediction or reference item.

    Handles two formats:
      - BioRED single-pair objects (``pred_label`` / ``gold_label`` + ``id1`` / ``id2``)
      - ChemProt / DDI iterable of relation objects (``.head``, ``.tail``, ``.label``)
    """
    if kind == "pred" and hasattr(item, "pred_label"):
        return {(item.id1, item.id2): item.pred_label}
    if kind == "gold" and hasattr(item, "gold_label"):
        return {(item.id1, item.id2): item.gold_label}
    if hasattr(item, "__iter__"):
        try:
            return {(r.head, r.tail): r.label for r in item}
        except (TypeError, AttributeError):
            return {}
    return {}


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from resolve_cache_fingerprint import resolve

    ckpt_path = args.ckpt
    out_jsonl = args.output

    overrides = [
        f"experiment={args.experiment}",
        "hydra.searchpath=[pkg://knowhyper.configs]",
    ]
    overrides.extend(args.extra)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="eval", overrides=overrides)

    # Resolve and validate the cache BEFORE building the dataset and running the
    # forward pass. Doing it afterwards throws away a GPU pass over the whole
    # validation split on what is a two-second configuration check, which is the
    # same "fail here rather than after ten epochs" rule the training drivers use.
    #
    # Resolve by FINGERPRINT, not by mtime: newest-mtime picks whichever arm was
    # written last, so with 13 DDI fingerprints present the arm whose attention we
    # report would be decided by write order, leaving no trace in the output. The
    # NER extractor already resolves this way.
    fingerprint, _profile, ds_dir = resolve(args.experiment)
    if not ds_dir:
        print(
            f"ERROR: resolve() returned no dataset name for {args.experiment}; the"
            " cache path would silently collapse a level",
            file=sys.stderr,
        )
        sys.exit(1)
    cfg_dataset_name = getattr(cfg.dataset, "name", None)
    if cfg_dataset_name and cfg_dataset_name != ds_dir:
        # The old code took the dataset segment from cfg. Keep that cross-check so a
        # fingerprint resolved for a different dataset than the checkpoint was
        # trained on cannot pass silently.
        print(
            f"ERROR: resolve() gave dataset {ds_dir!r} but the config says"
            f" {cfg_dataset_name!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    cache_root = Path(
        OmegaConf.to_container(cfg, resolve=True)["knowledge"]["pipeline"]["injector"][
            "retriever"
        ]["cache"]["root"]
    )
    val_cache_dir = cache_root / fingerprint / ds_dir / "validation"
    if not val_cache_dir.is_dir():
        print(f"ERROR: no cache at {val_cache_dir}", file=sys.stderr)
        sys.exit(1)
    cache_files = {f.stem: f for f in val_cache_dir.glob("*.json")}
    print(f"Using cache: {val_cache_dir} ({len(cache_files)} files)")

    val_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    val_cfg.split = "validation"
    dataset = _build_dataset(val_cfg, "validation", root_cfg=cfg)
    task = _build_task(cfg, dataset)
    collator, num_added = _build_collator(cfg, task)
    collator = _wrap_knowledge_collator(cfg, collator, split_role="validation")
    datamodule = IEDataModule(
        train_dataset=dataset,
        val_dataset=dataset,
        test_dataset=dataset,
        collator=collator,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
    )

    model = instantiate(
        _model_instantiation_cfg(cfg.model),
        label_list=task.label_list,
        extra_token_count=num_added,
    )
    metrics = _build_metric(cfg, dataset)
    loss_fn = instantiate(cfg.loss)
    lm = IELightningModule(
        model=model,
        task=task,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer_cfg={"_target_": "torch.optim.AdamW", "lr": 2e-5},
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lm.load_state_dict(ckpt["state_dict"])
    lm.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = lm.to(device)

    kh_model = lm.model
    pipeline = getattr(collator, "pipeline", None)
    injector = getattr(pipeline, "_injector", None) if pipeline else None
    ev_augmenter = (
        getattr(injector, "entity_view_augmenter", None) if injector else None
    )

    attn_weights_store = []
    orig_forward = kh_model.cross_attn.forward

    def hooked_forward(
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        **kwargs,
    ):
        out, weights = orig_forward(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=attn_mask,
            **kwargs,
        )
        attn_weights_store.append(weights.detach().cpu())
        return out, weights

    kh_model.cross_attn.forward = hooked_forward

    batch_store = []
    orig_test_step = lm.test_step

    def capturing_test_step(batch, *args, **kwargs):
        batch_store.append({"example_ids": extract_example_ids(batch)})
        return orig_test_step(batch, *args, **kwargs)

    lm.test_step = capturing_test_step

    captured_preds = []
    orig_update = lm.metrics.update

    def update_capturing(preds, refs, texts):
        captured_preds.append((preds, refs))
        return orig_update(preds, refs, texts)

    lm.metrics.update = update_capturing

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        limit_test_batches=1.0,
    )
    trainer.test(lm, datamodule=datamodule, ckpt_path=ckpt_path)

    def get_augmented_node_info(example_id: str) -> dict | None:
        if example_id not in cache_files:
            return None
        with open(cache_files[example_id]) as f:
            cached = json.load(f)

        base_node_ids = cached["node_ids"]
        seed_ids = set(cached.get("seed_node_ids", []))
        hyperedges = cached["hyperedges"]

        if ev_augmenter is not None:
            from knowhyper.pipeline.types import HyperedgeRef, SubHypergraph

            sub = SubHypergraph(
                seed_node_ids=tuple(cached.get("seed_node_ids", [])),
                node_ids=tuple(base_node_ids),
                hyperedges=tuple(
                    HyperedgeRef(
                        he_id=he["he_id"],
                        semantics=he["semantics"],
                        node_ids=tuple(he["node_ids"]),
                        roles=he.get("roles", {}),
                        polarity=he.get("polarity"),
                        score=he.get("score", 0.0),
                        score_breakdown=he.get("score_breakdown", {}),
                    )
                    for he in hyperedges
                ),
                provenance=cached.get("provenance", {}),
            )
            augmented = ev_augmenter.augment(sub)
            final_node_ids = list(augmented.node_ids)
            final_hyperedges = list(augmented.hyperedges)
        else:
            final_node_ids = base_node_ids
            final_hyperedges = [
                type("HE", (), {"semantics": he["semantics"]}) for he in hyperedges
            ]

        node_info = [classify_node(nid, seed_ids) for nid in final_node_ids]
        he_info = [
            {
                "category": "hyperedge",
                "semantics": he.semantics,
                "he_id": getattr(he, "he_id", ""),
            }
            for he in final_hyperedges
        ]
        return {
            "node_info": node_info,
            "he_info": he_info,
            "n_nodes": len(final_node_ids),
            "n_edges": len(final_hyperedges),
            "seed_ids": list(seed_ids),
        }

    rows = []
    n_batches = min(len(attn_weights_store), len(batch_store), len(captured_preds))

    for b_idx in range(n_batches):
        weights = attn_weights_store[b_idx]
        meta = batch_store[b_idx]
        preds, refs = captured_preds[b_idx]

        example_ids = meta["example_ids"]
        seen = set()
        order = []
        for eid in example_ids:
            if eid not in seen:
                seen.add(eid)
                order.append(eid)

        for j, ex_id in enumerate(order):
            if j >= weights.shape[0]:
                break
            w = weights[j, 0, :]
            info = get_augmented_node_info(ex_id)
            if info is None:
                continue

            n_nodes = info["n_nodes"]
            n_edges = info["n_edges"]
            total = n_nodes + n_edges
            w_valid = (
                w[:total].numpy().tolist()
                if total <= len(w)
                else w.numpy().tolist()[:total]
            )

            token_records = []
            for t_idx in range(min(n_nodes, len(w_valid))):
                ni = info["node_info"][t_idx]
                token_records.append(
                    {
                        "pos": t_idx,
                        "type": "node",
                        "category": ni["category"],
                        "is_seed": ni["is_seed"],
                        "attn_weight": w_valid[t_idx],
                    }
                )
            for t_idx in range(n_nodes, min(total, len(w_valid))):
                he_idx = t_idx - n_nodes
                hi = (
                    info["he_info"][he_idx]
                    if he_idx < len(info["he_info"])
                    else {"category": "hyperedge", "semantics": "unk"}
                )
                token_records.append(
                    {
                        "pos": t_idx,
                        "type": "hyperedge",
                        "category": hi["category"],
                        "semantics": hi.get("semantics", "unk"),
                        "attn_weight": w_valid[t_idx],
                    }
                )

            pr = _extract_label_dict(preds[j], kind="pred") if j < len(preds) else {}
            gd = _extract_label_dict(refs[j], kind="gold") if j < len(refs) else {}

            rows.append(
                {
                    "id": ex_id,
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "n_tokens": len(token_records),
                    "pred": [[h, t, lab] for (h, t), lab in pr.items()],
                    "gold": [[h, t, lab] for (h, t), lab in gd.items()],
                    "tokens": token_records,
                }
            )

    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} instances to {out_jsonl}")

    cat_weights = defaultdict(list)
    for r in rows:
        for t in r["tokens"]:
            cat_weights[t["category"]].append(t["attn_weight"])

    print("\n=== Attention Weight Summary by Category ===")
    for cat in sorted(cat_weights.keys()):
        ws = cat_weights[cat]
        mean_w = sum(ws) / len(ws)
        print(f"  {cat:20s}: mean={mean_w:.6f}  n_tokens={len(ws)}")


if __name__ == "__main__":
    main()
