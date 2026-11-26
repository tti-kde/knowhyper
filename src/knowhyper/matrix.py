"""The experiment matrix: backbone x dataset x method -> a runnable cell.

One place that knows how a (backbone, dataset, method, seed) tuple turns into a
Hydra invocation, so a new backbone or a new PrimeHKG snapshot does not mean
editing a dozen shell scripts. `scripts/run_matrix_cell.py` executes one cell.

Design constraints this encodes, each learned the hard way:

* **Test is always evaluated.** The BioRED EV runs passed
  `trainer.limit_test_batches=0` and left no test number to recover; the whole
  §4 main table then had to be redone. Nothing here may disable test.
* **`logger=none` + a `*_nolog` trainer.** ieforge builds the logger from the
  top-level group and passes it as a kwarg, so `++trainer.logger=false` is a
  silent no-op and the default CSVLogger crashes at a validation epoch end.
* **`save_top_k=1 save_last=false`.** The ieforge default keeps 4 ckpts/cell and
  exhausts the ABCI Lustre quota over ~100 cells.
* **One retriever profile per dataset.** `caching` is the *ChemProt* profile;
  BioRED needs `caching_biored`. Arms of one dataset that disagree here silently
  retrieve different graphs (see tests/ieforge/test_retriever_semantics_parity.py).

Adding a method means adding one `Method` entry. Adding a backbone means adding a
config to `configs/backbone/` and a name to `BACKBONES`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Backbone config-group names under configs/backbone/, and the HF ids they set.
#
# Swapping the backbone takes TWO different mechanisms, which is not obvious and
# silently breaks a sweep if you assume one:
#
#   * `*_baseline` configs list `/backbone: pubmedbert` in their defaults, so
#     `backbone=biolinkbert` overrides the group.
#   * The knowledge arms (`khp_*`) do NOT. They pin the encoder inline at
#     `model.inner.model_name`, and `backbone=...` fails with "Could not override
#     'backbone'. No match in the defaults list."
#
# So the model id has to be duplicated here. Keep it in step with
# `configs/backbone/*.yaml` -- `tests/test_matrix.py` pins that they agree.
BACKBONES = ("pubmedbert", "biolinkbert", "biobert", "scibert")

BACKBONE_MODEL_IDS = {
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "biolinkbert": "michiyasunaga/BioLinkBERT-base",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "scibert": "allenai/scibert_scivocab_uncased",
}

RE_DATASETS = ("chemprot", "ddi", "biored")

# NER datasets. Kept as a separate tuple, not merged into one flat list, because the
# two families differ in ways every consumer has to branch on anyway:
#
#   * NER cells need `knowledge.pipeline.query_builder.predictions_root` -- a
#     first-pass prediction directory. RE cells have no such input, and a NER cell
#     without it does not fail: `NERPredictionQueryBuilder` gets `predictions_root:
#     null` from the group config and the run reads no mentions at all.
#   * NER metric is entity_f1 over token spans; RE is relation_f1 over pairs. They are
#     not comparable numbers and must never share an aggregate table.
#   * Each NER dataset has its OWN retriever profile (chem/gene/disease), so "one
#     profile per dataset" is enforced across six values, not three.
#
# `DATASETS` stays the full list so existing callers and the `--datasets` default keep
# working.
# jnlpba and ncbi_disease are BASELINE-ONLY for now: `ner_baseline_*` configs exist
# for both, but no knowledge arm does. A knowledge arm additionally needs a
# first-pass prediction directory and a dataset-specific retriever profile, and
# JNLPBA's five entity types have no profile at all. `experiment_for()` returns
# None for every knowledge method on these two, so plan() refuses such a cell
# rather than composing one that would silently retrieve with the wrong profile.
NER_DATASETS = (
    "bc5cdr_chem",
    "bc2gm",
    "bc5cdr_disease",
    "jnlpba",
    "ncbi_disease",
)

DATASETS = RE_DATASETS + NER_DATASETS


@dataclass(frozen=True)
class Method:
    """One arm of the comparison, as a function of dataset."""

    name: str
    # experiment config per dataset; None = not available for that dataset
    experiments: dict[str, str | None]
    # extra Hydra overrides beyond the mandatory set
    overrides: tuple[str, ...] = ()
    # trainer group: knowledge-fusion arms need the differential-LR variant
    trainer: str = "nolog"
    # needs a pretrained encoder passed as +model.pretrained_encoder_path
    needs_encoder: bool = False
    # Controls this arm's score is only interpretable against. The FIRST is the
    # pre-registered primary contrast; the rest bound what it cannot separate.
    # R2b needs two: `shuffled` isolates the typed-structure signal from the
    # effect of the `[HG]` frame itself (R2a's sham arm showed the frame alone
    # moves the score as much as real text), and `off_short` bounds the total.
    controls: tuple[str, ...] = ()
    # True when the config lists `/backbone` in its defaults (the baselines);
    # False when the encoder is pinned inline at model.inner.model_name (khp_*).
    uses_backbone_group: bool = False
    note: str = ""

    def experiment_for(self, dataset: str) -> str | None:
        return self.experiments.get(dataset)


def _per_dataset(
    chemprot: str | None,
    ddi: str | None,
    biored: str | None,
    **ner: str | None,
):
    """Map dataset -> experiment config.

    The three RE datasets stay positional so the 18 existing call sites keep reading
    as a fixed triple. NER datasets arrive as keywords, so a method that has no NER
    variant simply omits them and `experiment_for()` returns None -- which is already
    how "not available for that dataset" is expressed.

    Deliberately NOT a 6-positional signature: that would have forced `None, None,
    None` onto every RE-only method, and a misplaced positional there is exactly the
    silent-wrong-config class this module exists to prevent.
    """
    unknown = set(ner) - set(NER_DATASETS)
    if unknown:
        msg = f"unknown NER dataset(s): {sorted(unknown)}; known: {NER_DATASETS}"
        raise ValueError(msg)
    # EVERY dataset key is always present, defaulting to None. A method that omits a
    # dataset must still say so explicitly in the mapping -- `experiment_for()` reads
    # it with `.get()`, so a missing key and an explicit None behave identically at
    # runtime, but only the explicit form is visible to
    # `test_methods_declare_a_config_for_every_dataset_or_none`, whose whole job is to
    # stop a half-populated method from silently dropping datasets out of a sweep.
    out: dict[str, str | None] = dict.fromkeys(DATASETS)
    out["chemprot"] = chemprot
    out["ddi"] = ddi
    out["biored"] = biored
    out.update(ner)
    return out


# Overrides that every knowledge-fusion arm needs. Omitting any of these does not
# error -- it silently trains a model whose knowledge branch never learns.
#
# The first three are now also the defaults in experiment/_run_defaults.yaml and the
# three khp_*_xattn roots, so these are belt-and-braces rather than load-bearing:
# a cell is correctly configured even if this tuple is bypassed. They use `++`, not
# `+`, precisely because the keys now exist -- plain `+` fails with "An item is
# already at 'model.fusion_norm'" once a default is present, and `++` sets the value
# whether or not the key is there.
FUSION_OVERRIDES = (
    "++model.gate_init=1.0",
    "++model.gate_shape=vector",
    "++model.fusion_norm=true",
    "trainer.callbacks.3.knowledge_lr_mult=10",
)

# Empty-batch note for the BioRED surface arms, kept because the id-only arms still
# hit it and a future linking change could reintroduce it.
#
# `require_knowledge: true` aborts when EVERY graph in a batch has zero nodes
# (`models.py:307`), and validation does not shuffle. Measured in TRUE DataLoader
# order (dataset row order) over the complete 6,659-instance validation split:
#
#                    empty    all-empty batches @bs64
#     id-only        46.2%    9      -> aborts, which is why those arms cannot run
#     surface         4.6%    0      -> runs at the standard bs=64
#
# So surface fallback alone is sufficient; no batch-size override is needed. An
# earlier version of this file raised the batch to 96 on the strength of a
# measurement taken over LEXICOGRAPHICALLY SORTED FILENAMES instead of row order.
# That ordering is not what the DataLoader sees: the one long empty run (66 rows,
# PMID 27840894, all SequenceVariant HGVS strings that PrimeHKG has no node type
# for) begins at row 646, and 646 % 64 = 6, so no aligned 64-window fits inside it.
# The override was therefore unnecessary AND harmful -- ieforge has no eval-only
# batch setting, so it moved training too and made these arms' absolute scores
# incomparable to every other arm. If you ever need to re-check this, iterate the
# dataset, not the cache directory.

METHODS: tuple[Method, ...] = (
    Method(
        name="base",
        uses_backbone_group=True,
        experiments=_per_dataset(
            "re_baseline",
            "ddi_baseline",
            "biored_baseline",
            bc5cdr_chem="ner_baseline",
            bc2gm="ner_baseline_bc2gm",
            bc5cdr_disease="ner_baseline_bc5cdr_disease",
            jnlpba="ner_baseline_jnlpba",
            ncbi_disease="ner_baseline_ncbi_disease",
        ),
        note="text-only backbone, no knowledge pipeline. The paper's OFF row.",
    ),
    Method(
        name="edge",
        experiments=_per_dataset(
            "khp_re_xattn",
            "khp_ddi_xattn",
            "khp_biored_xattn",
            # NER maps to the STRICT arms. Not because strict is expected to help --
            # its end-task effect on NER is unmeasured -- but because the campaign
            # fixed one linking setting so `edge`/`ev_cold`/`ev_warm` differ in ONE
            # variable. The non-strict configs still exist for the A/B in §3 of
            # docs/superpowers/plans/2026-07-28-ner-matrix-campaign.md.
            bc5cdr_chem="khp_ner_strict",
            bc2gm="khp_ner_strict_bc2gm",
            bc5cdr_disease="khp_ner_strict_bc5cdr_disease",
        ),
        overrides=FUSION_OVERRIDES,
        trainer="difflr_nolog",
        controls=("base",),
        note=(
            "structure only: hyperedges through the AllSet encoder. Controls "
            "against `base` -- this is the paper's headline knowledge-vs-no-knowledge "
            "contrast, and it was MISSING from the 2026-07-27 record because no "
            "control was declared, so the aggregate emitted only the strict/warm "
            "sub-contrasts and the headline delta had to be computed by hand."
        ),
    ),
    Method(
        name="ev_cold",
        experiments=_per_dataset(
            "khp_re_xattn_ev",
            "khp_ddi_xattn_ev",
            "khp_biored_xattn_ev",
            bc5cdr_chem="khp_ner_ev_strict",
            bc2gm="khp_ner_ev_strict_bc2gm",
            bc5cdr_disease="khp_ner_ev_strict_bc5cdr_disease",
        ),
        overrides=FUSION_OVERRIDES,
        trainer="difflr_nolog",
        controls=("base",),
        note=(
            "entity view (attr:text/smiles/sequence), randomly initialised encoder. "
            "Controls against `base` for the same reason as `edge`."
        ),
    ),
    Method(
        name="edge_strict",
        experiments=_per_dataset("khp_re_xattn_strict", "khp_ddi_xattn_strict", None),
        overrides=FUSION_OVERRIDES,
        trainer="difflr_nolog",
        controls=("edge",),
        note=(
            "`edge` with strict_type_match ON. SETTLED 2026-07-28 at n=10, and the "
            "answer differs BY ARM, so do not generalise from this one: on `edge` "
            "strict is neutral-to-negative (DDI val -0.0071 p=.052), but on the EV "
            "arms it HELPS (see ev_cold_strict). The often-quoted objection that "
            "strict triples the empty-sub-HG rate is not itself a reason -- measured "
            "on DDI test, the empty stratum shows no degradation from knowledge "
            "(`edge - base` +0.0024, `ev_cold - base` +0.0060), so an empty sub-HG "
            "is inert rather than harmful. Judge strict on end-task numbers only. "
            "No BioRED variant: BioRED already links strictly."
        ),
    ),
    Method(
        name="ev_cold_strict",
        experiments=_per_dataset(
            "khp_re_xattn_ev_strict", "khp_ddi_xattn_ev_strict", None
        ),
        overrides=FUSION_OVERRIDES,
        trainer="difflr_nolog",
        controls=("ev_cold",),
        note=(
            "`ev_cold` with strict_type_match ON. The prediction was that the "
            "entity-view arm would LOSE most from strict typing (fewer seeds means "
            "fewer attribute nodes). Measured at n=10 on DDI test, it GAINS: "
            "`ev_cold_strict - base` is +0.0139 p=.001 (9/10) where non-strict "
            "`ev_cold - base` is null (+0.0072 p=.289), and `ev_cold_strict - "
            "edge_strict` is +0.0059 where the non-strict pair is -0.0064. So "
            "strict is what makes the entity view beat structure-only on DDI -- the "
            "opposite of the prediction, and the reason strict is now the default "
            "direction for the EV arms."
        ),
    ),
    Method(
        name="ev_warm",
        experiments=_per_dataset(
            "khp_re_xattn_ev",
            "khp_ddi_xattn_ev",
            "khp_biored_xattn_ev",
            bc5cdr_chem="khp_ner_ev_strict",
            bc2gm="khp_ner_ev_strict_bc2gm",
            bc5cdr_disease="khp_ner_ev_strict_bc5cdr_disease",
        ),
        overrides=FUSION_OVERRIDES,
        trainer="difflr_nolog",
        needs_encoder=True,
        # Primary stays `ev_cold`: that isolates warm-start from the entity view.
        # `base` is secondary so the absolute knowledge-vs-none delta is recorded too.
        controls=("ev_cold", "base"),
        note=(
            "ev_cold warm-started from a cross-modal MAE encoder. The encoder must "
            "be pretrained on the SAME SubHG cache the end-task reads: warm-start "
            "only helped when the pretrain and end-task data matched (DDI +0.0091 "
            "p=0.002), and a shared encoder across differently-linked caches is a "
            "distribution mismatch. `--encoder` may therefore be a DIRECTORY laid "
            "out as <dir>/<dataset>/cross_modal/*_<fingerprint>/encoder.pt, and the "
            "right one is selected per cell; a plain file path is still accepted "
            "for a single-dataset run."
        ),
    ),
    Method(
        name="ev_warm_strict",
        experiments=_per_dataset(
            "khp_re_xattn_ev_strict", "khp_ddi_xattn_ev_strict", None
        ),
        overrides=FUSION_OVERRIDES,
        trainer="difflr_nolog",
        needs_encoder=True,
        controls=("ev_cold_strict",),
        note=(
            "ev_warm under strict typing. Needs its OWN encoder: the strict cache "
            "has different seeds, so an encoder pretrained on the non-strict cache "
            "would be the exact mismatch that made earlier warm-starts null."
        ),
    ),
    # --- injector top_k sweep (NER) -------------------------------------------
    #
    # Retrieval on NER runs AT THE CAP: median 64 hyperedges/instance on all three
    # datasets (measured 2026-07-29 from the strata files). So the graph is not too
    # sparse, it is truncated -- and cross-attention over 64 hyperedges may simply be
    # diluting the few relevant ones. This sweep is the cheapest remaining
    # experiment: `injector.top_k` is NOT part of the SubHG cache fingerprint (only
    # the RETRIEVER's own top_k is), so every value reads the existing cache and
    # needs no precompute. Verified by re-deriving the fingerprint under each
    # override -- injector 16/32 give `ac70b0f4ab594168`, unchanged.
    #
    # Each variant keeps its own shuffled control, because a top_k that "helps" only
    # by shrinking the branch would show the same gain with wrong content.
    *[
        Method(
            name=f"ev_cold_k{k}",
            experiments=_per_dataset(
                None,
                None,
                None,
                bc5cdr_chem="khp_ner_ev_strict",
                bc2gm="khp_ner_ev_strict_bc2gm",
                bc5cdr_disease="khp_ner_ev_strict_bc5cdr_disease",
            ),
            overrides=(
                *FUSION_OVERRIDES,
                f"knowledge.pipeline.injector.top_k={k}",
            ),
            trainer="difflr_nolog",
            controls=("ev_cold",),
            note=(
                f"ev_cold with the injector capped at {k} hyperedges instead of 64. "
                "Reads the SAME cache as ev_cold -- injector.top_k is outside the "
                "cache fingerprint -- so this costs training only."
            ),
        )
        for k in (4, 8, 16, 32)
    ],
    Method(
        name="ev_cold_shuffled",
        experiments=_per_dataset(
            None,
            None,
            None,
            bc5cdr_chem="khp_ner_ev_strict",
            bc2gm="khp_ner_ev_strict_bc2gm",
            bc5cdr_disease="khp_ner_ev_strict_bc5cdr_disease",
        ),
        overrides=(*FUSION_OVERRIDES, "+model.knowledge_control=shuffled"),
        trainer="difflr_nolog",
        controls=("ev_cold",),
        note=(
            "SAME config as ev_cold with the knowledge batch rolled by one "
            "(models.py:986/1164), so every instance gets another instance's "
            "sub-hypergraph: identical parameter count and compute, wrong content. "
            "This is the control every previous NER-KG result actually failed "
            "against -- candidate rerank measured `real - shuffled` = +0.00007 at "
            "n=3, i.e. indistinguishable, while the same arms did beat OFF. A gain "
            "over `base` alone is consistent with 'an extra branch regularises the "
            "model', so without this arm a positive NER result cannot be attributed "
            "to knowledge CONTENT. NER-only: the RE arms have their own controls."
        ),
    ),
    Method(
        name="text",
        uses_backbone_group=True,
        experiments=_per_dataset(
            "re_retrieved_definitions",
            "ddi_retrieved_definitions",
            "biored_retrieved_definitions",
            bc5cdr_chem="ner_retrieved_text",
            bc2gm="ner_retrieved_text_bc2gm",
            bc5cdr_disease="ner_retrieved_text_bc5cdr_disease",
        ),
        controls=("text_off",),
        note=(
            "R2a: retrieved KG text appended to the input, no graph encoder. The "
            "NER arms are NOT the same payload as the RE ones: they add "
            "`synonym_context` (features 0.12.14) because the NER question is "
            "whether a span is a NAME, for which other names are the evidence. "
            "They also cap `top_k` at 8 -- the text budget divides round-robin "
            "per item, so 64 items renders bare wrapper tags."
        ),
    ),
    Method(
        name="text_off",
        uses_backbone_group=True,
        experiments=_per_dataset(
            "re_off_short",
            "ddi_off_short",
            "biored_off_short",
            bc5cdr_chem="ner_text_off",
            bc2gm="ner_text_off_bc2gm",
            bc5cdr_disease="ner_text_off_bc5cdr_disease",
        ),
        note=(
            "truncation-matched control for `text`: identical retrieval and token "
            "budget, knowledge tokens dropped. NOT the same as `base` -- on BioRED "
            "87% of instances truncate, so `base` would confound truncation cost "
            "with the text effect."
        ),
    ),
    Method(
        name="text_sham",
        uses_backbone_group=True,
        experiments=_per_dataset(
            None,
            None,
            None,
            bc5cdr_chem="ner_text_sham",
            bc2gm="ner_text_sham_bc2gm",
            bc5cdr_disease="ner_text_sham_bc5cdr_disease",
        ),
        controls=("text",),
        note=(
            "CONTENT control for the NER text arm: mention surface and payload length "
            "kept, body replaced with [MASK]. Required because `text_off` is INERT on "
            "NER -- the source is 25 tokens at the median against a >=253-token floor, "
            "so it never truncates and off_short is byte-identical to `base` (same "
            "test_f1 to 6 dp on all 10 seeds). Only this arm separates 'the retrieved "
            "text helped' from 'an extra segment helped' or 'the entity NAME helped'. "
            "NER-only; the RE arms have their own sham."
        ),
    ),
    Method(
        name="serialized",
        uses_backbone_group=True,
        experiments=_per_dataset(
            "re_serialized_subhg",
            "ddi_serialized_subhg",
            "biored_serialized_subhg",
        ),
        # PRIMARY contrast is vs `shuffled`, not `off_short` (#150). R2a's sham arm
        # showed the frame alone moves DDI as much as the real text
        # (sham - off_short +0.0065 vs C1n - off_short +0.0073), so an off_short
        # contrast cannot separate the `[HG]` scaffold from typed structure.
        controls=("serialized_shuffled", "serialized_off_short"),
        note="R2b (#138): the retrieved typed sub-hypergraph serialised to text.",
    ),
    Method(
        name="serialized_shuffled",
        uses_backbone_group=True,
        experiments=_per_dataset(
            "re_serialized_shuffled",
            "ddi_serialized_shuffled",
            "biored_serialized_shuffled",
        ),
        note=(
            "Primary control for `serialized`: keeps the frame, field names, node "
            "ids and member roles, corrupting ONLY the semantics and polarity "
            "values, from an explicitly pinned pool with a deterministic hash."
        ),
    ),
    Method(
        name="serialized_off_short",
        uses_backbone_group=True,
        experiments=_per_dataset(
            "re_serialized_off_short",
            "ddi_serialized_off_short",
            "biored_serialized_off_short",
        ),
        note=(
            "Truncation-matched OFF control for `serialized`. Distinct from "
            "`text_off`: it selects the serialized payload before dropping it, so "
            "the token accounting matches R2b rather than R2a."
        ),
    ),
    Method(
        name="binary",
        experiments=_per_dataset(
            "khp_re_xattn_ev_binary",
            "khp_ddi_xattn_ev_binary",
            "khp_biored_xattn_ev_binary",
        ),
        overrides=FUSION_OVERRIDES,
        trainer="difflr_nolog",
        controls=("ev_cold",),
        note="N-ary hyperedges projected to 2-sections; isolates the N-ary claim.",
    ),
    Method(
        name="endpoint",
        experiments=_per_dataset(
            None,
            "khp_ddi_xattn_endpoint",
            "khp_biored_xattn_endpoint",
        ),
        overrides=FUSION_OVERRIDES,
        trainer="difflr_nolog",
        controls=("endpoint_shuffled", "base"),
        note=(
            "Head/tail vectors only -- no neighborhood, no message passing -- on "
            "the same retrieval and feature store as the EV arms. Reproduces the "
            "Asada / Sanger endpoint shape inside our protocol so `ev_cold - "
            "endpoint` isolates the neighborhood. PRIMARY control is "
            "`endpoint_shuffled` (same tokens, same types, same coverage, wrong "
            "identity); `base` answers the weaker question of whether endpoints "
            "beat text alone. ChemProt is absent on purpose: `ev_cold - base` "
            "there is +0.0004 (p=.866), so there is no effect to recover. BioRED "
            "inherits the SURFACE parent, matching its recorded comparator."
        ),
    ),
    Method(
        name="endpoint_shuffled",
        experiments=_per_dataset(
            None,
            "khp_ddi_xattn_endpoint_shuffled",
            "khp_biored_xattn_endpoint_shuffled",
        ),
        overrides=FUSION_OVERRIDES,
        trainer="difflr_nolog",
        controls=("endpoint",),
        note=(
            "`endpoint` with each seed vector replaced by a DIFFERENT entity of "
            "the SAME node type, chosen deterministically from the train-split "
            "seed pool. Not a batch roll: `model.knowledge_control=shuffled` "
            "exists only on the NER model, and rolling is not type- or "
            "coverage-matched -- DDI seeds span 11 node types, so a roll would "
            "confound wrong-identity with wrong-type."
        ),
    ),
    Method(
        name="ev_nosmiles",
        experiments=_per_dataset(
            "khp_re_xattn_ev_nosmiles",
            "khp_ddi_xattn_ev_nosmiles",
            None,
        ),
        overrides=FUSION_OVERRIDES,
        trainer="difflr_nolog",
        controls=("ev_cold",),
        note=(
            "`ev_cold` with attr:smiles masked out of the entity view. Masking "
            "happens at index build, after retrieval, so this shares ev_cold's "
            "SubHG fingerprint. BioRED is deliberately absent: its attr:smiles "
            "share is 0.03 (vs DDI 0.23), and every BioRED knowledge arm is "
            "already null, so the contrast has nothing to detect."
        ),
    ),
    Method(
        name="edge_surface",
        experiments=_per_dataset(None, None, "khp_biored_xattn_surface"),
        overrides=FUSION_OVERRIDES,
        trainer="difflr_nolog",
        controls=("base",),
        note=(
            "BioRED-only: `edge` with surface-fallback linking. The id-only arms "
            "cannot run at all -- 46.2% of BioRED sub-HGs are empty, rows are "
            "contiguous by document so 10 of 104 aligned 64-batches are entirely "
            "empty, and require_knowledge=true then aborts in validation_step. "
            "Surface fallback cuts that to ~8%. Controls against `base`, not "
            "`edge`, because `edge` produces no BioRED numbers to compare with."
        ),
    ),
    Method(
        name="ev_cold_surface",
        experiments=_per_dataset(None, None, "khp_biored_xattn_ev_surface"),
        overrides=FUSION_OVERRIDES,
        trainer="difflr_nolog",
        controls=("base", "edge_surface"),
        note=(
            "BioRED-only entity-view arm on the surface-fallback cache. Surface "
            "forms are appended unconditionally, so stratify by the id-only seed "
            "count (0/1/2) when reporting. Measured on 1,156 paired instances, the "
            "linked strata do NOT grow (top_k=64 was already saturated: 62.5->61.7 "
            "and 65.3->65.1 mean nodes), so the feared cancellation is not what "
            "retrieval does -- but a same-size sub-HG can still be a different one."
        ),
    ),
    Method(
        name="ev_warm_surface",
        experiments=_per_dataset(None, None, "khp_biored_xattn_ev_surface"),
        overrides=FUSION_OVERRIDES,
        trainer="difflr_nolog",
        needs_encoder=True,
        controls=("ev_cold_surface", "base"),
        note=(
            "BioRED's warm-start arm. `ev_warm` cannot cover BioRED: it resolves to "
            "the id-only cache (3a484c25253cf019), where BioRED cannot train at all "
            "-- 46.2% of its sub-HGs are empty, so require_knowledge=true aborts in "
            "validation_step. Every BioRED encoder is therefore pretrained on the "
            "SURFACE cache (cb831d8592bc38aa), and `_resolve_encoder` correctly "
            "found 0 matches for a BioRED `ev_warm` cell rather than silently "
            "pairing an encoder with a cache it was not pretrained on. This arm "
            "exists so the surface encoder has a matching end-task arm; without it "
            "BioRED simply has no warm-start cell and the matrix looks complete "
            "while missing one."
        ),
    ),
    Method(
        name="serialized_surface",
        experiments=_per_dataset(None, None, "biored_serialized_subhg_surface"),
        # Like every other text arm: these compose `/backbone` in their defaults,
        # so the backbone comes from the GROUP. Omitting this sends them down the
        # khp_* path (`model.inner.model_name=`) and Hydra rejects it with
        # "Key 'inner' is not in struct" -- 18 cells lost to exactly that.
        uses_backbone_group=True,
        trainer="nolog",
        controls=("serialized_shuffled_surface", "base"),
        note=(
            "R2b on the SURFACE cache, so `edge_surface` vs this is a "
            "single-variable structure-vs-text contrast on IDENTICAL graphs. "
            "Without it the only available reading compares a surface-linked "
            "fusion arm against an id-linked text arm, folding the linking change "
            "into the injection-format claim. Needs no extra precompute -- same "
            "fingerprint as the fusion surface arms."
        ),
    ),
    Method(
        name="graft_real",
        # These arms compose `/backbone` through the R2b -> R2a chain, so the
        # backbone comes from the GROUP. Omitting this sends them down the khp_*
        # path (`model.inner.model_name=`) and Hydra rejects it with "Key 'inner'
        # is not in struct" -- 18 cells were lost to exactly that on the surface arms.
        uses_backbone_group=True,
        # NOT difflr_nolog. `BACKBONE_PREFIX` is "model.inner.", so this model's
        # SECOND full BERT (at model.evidence_encoder.*) would be classified as
        # "knowledge branch" and 110M pretrained parameters would train at 10x the
        # backbone LR. That does not error; it destroys the evidence encoder and
        # reports a plausible score. Pinned by tests/test_matrix.py.
        trainer="nolog",
        experiments=_per_dataset(
            "re_graft_style_real",
            "ddi_graft_style_real",
            # SURFACE, matching khp_biored_xattn_surface_top3: the id-only cache
            # leaves 46.2% of sub-HGs empty, so the matched graph arm cannot run on
            # it and `edge_top3 - graft_real` would become a linking contrast.
            "biored_graft_style_real",
        ),
        controls=("graft_masked",),
        note=(
            "GRAFT-style controlled reconstruction (#189): the same top-3 PrimeHKG "
            "hyperedges the R2b arm serializes, encoded by an INDEPENDENT second "
            "encoder, mixed with source-only logits through a content-sensitive "
            "gate. Primary control is `graft_masked` (exact token count, content "
            "removed), NOT `base` -- a gain over base alone is consistent with 'a "
            "second encoder regularises the model'. This is NOT official GRAFT: "
            "different retrieval corpus, different candidate rows, encrypted "
            "official datasets. Report as `GRAFT-style` and keep official reported "
            "numbers in a separate table."
        ),
    ),
    Method(
        name="graft_masked",
        uses_backbone_group=True,
        trainer="nolog",
        experiments=_per_dataset(
            "re_graft_style_masked",
            "ddi_graft_style_masked",
            "biored_graft_style_masked",
        ),
        controls=(),
        note=(
            "Pre-registered primary control for `graft_real`: identical model, "
            "parameter names/shapes, source, retained block count and evidence "
            "sequence length -- every non-special evidence token replaced by "
            "[MASK]. Chosen over the R2b `shuffled` control because shuffled keeps "
            "connectivity and node ids and matches token length only in "
            "expectation; this arm is exact per instance."
        ),
    ),
    Method(
        name="edge_top3",
        experiments=_per_dataset(
            "khp_re_xattn_top3",
            "khp_ddi_xattn_top3",
            "khp_biored_xattn_surface_top3",
        ),
        overrides=FUSION_OVERRIDES,
        trainer="difflr_nolog",
        # `graft_real` is the primary: same evidence, different representation.
        # `base` is secondary so the absolute knowledge-vs-none delta is recorded.
        # `graft_masked` is listed too even though it is `graft_real`'s control, not
        # this arm's: `controls_for` is one level deep, so without it
        # `--methods edge_top3` would plan `graft_real` with NO content control and
        # the GRAFT arm's headline contrast would be absent from the sweep -- which
        # reads as a planning choice rather than a missing cell.
        controls=("graft_real", "base", "graft_masked"),
        note=(
            "Structure-only fusion restricted to the SAME 3 hyperedges as "
            "`graft_real`, so `edge_top3 - graft_real` is a single-variable "
            "text-parallel vs node/hyperedge-encoding contrast. Do NOT substitute "
            "`edge` here: it encodes up to 64 hyperedges, so `edge - graft_real` "
            "confounds representation with a 64-vs-3 evidence budget. The full-"
            "budget comparison `edge - graft_real` is a SYSTEM-level result and "
            "differs in parameters and compute as well; keep the two apart when "
            "writing it up."
        ),
    ),
    Method(
        name="graft_wide_real",
        uses_backbone_group=True,
        trainer="nolog",
        experiments=_per_dataset(
            "re_graft_style_wide_real",
            "ddi_graft_style_wide_real",
            "biored_graft_style_wide_real",
        ),
        controls=("graft_wide_masked",),
        note=(
            "GRAFT-style at 12 hyperedges / 512 evidence tokens, the follow-up to the "
            "n=3 screen's null (#189). The screen found `real - masked` ~= 0; this "
            "tests whether 3 of 64 edges (~118 tokens) was simply too thin a slice. "
            "The original marginal gate analysis was later superseded by a paired "
            "intervention. A NEW arm, not an edit: the "
            "pre-registration forbids changing the screened arms in response to their "
            "scores, and the serializer fingerprint differs so provenance is separate. "
            "Needs no new precompute -- the serializer is not in the retrieval cache "
            "key, so it reads the same top_k=64 cache. Its control must be "
            "`graft_wide_masked`, matched at the SAME budget; comparing against the "
            "3-edge masked arm would fold the budget change into the content effect."
        ),
    ),
    Method(
        name="graft_wide_masked",
        uses_backbone_group=True,
        trainer="nolog",
        experiments=_per_dataset(
            "re_graft_style_wide_masked",
            "ddi_graft_style_wide_masked",
            "biored_graft_style_wide_masked",
        ),
        controls=(),
        note=(
            "Exact-token-count control for `graft_wide_real` at the wide budget. "
            "Differs from its treated arm in `evidence_mode` and nothing else."
        ),
    ),
    Method(
        name="graft_text_real",
        uses_backbone_group=True,
        trainer="nolog",
        experiments=_per_dataset(
            "re_graft_style_text_real",
            "ddi_graft_style_text_real",
            "biored_graft_style_text_real",
        ),
        controls=("graft_text_masked",),
        note=(
            "GRAFT-style over PrimeHKG NATURAL TEXT (#189 follow-up B): the same "
            "parallel branch and gate, fed raw node_features_text instead of the "
            "[HG] serialization. The screened arms' null is confounded with OUR "
            "serialization format -- which is not what official GRAFT reads -- and "
            "the probe showed the encoder CAN separate real from masked, so the "
            "format is the remaining suspect alongside the fusion design. Null "
            "disappearing here means the format was the problem; null persisting "
            "means the fusion design is, independent of content type. NOT a re-run "
            "of R2a: that arm concatenated definitions into ONE sequence, this one "
            "uses a second encoder and a learned gate, and route has flipped a "
            "conclusion on this project before (NER text arms)."
        ),
    ),
    Method(
        name="graft_text_masked",
        uses_backbone_group=True,
        trainer="nolog",
        experiments=_per_dataset(
            "re_graft_style_text_masked",
            "ddi_graft_style_text_masked",
            "biored_graft_style_text_masked",
        ),
        controls=(),
        note=(
            "Exact-token-count control for `graft_text_real`. Differs in "
            "`evidence_mode` only."
        ),
    ),
    Method(
        name="serialized_shuffled_surface",
        experiments=_per_dataset(None, None, "biored_serialized_shuffled_surface"),
        # Like every other text arm: these compose `/backbone` in their defaults,
        # so the backbone comes from the GROUP. Omitting this sends them down the
        # khp_* path (`model.inner.model_name=`) and Hydra rejects it with
        # "Key 'inner' is not in struct" -- 18 cells lost to exactly that.
        uses_backbone_group=True,
        trainer="nolog",
        controls=(),
        note=(
            "Pre-registered primary control for `serialized_surface`: same [HG] "
            "frame and connectivity, semantics/polarity values replaced. The frame "
            "alone moves the score as much as real text does, so the contrast must "
            "be against shuffled, not against OFF."
        ),
    ),
)

METHODS_BY_NAME = {m.name: m for m in METHODS}

# Methods whose model exposes source-only logits and a learned gate, so
# `scripts/analysis/export_graft_diagnostics.py` can run on their checkpoints.
#
# DERIVED, not listed. The first version was a hand-written literal carrying the
# comment "so adding a GRAFT-style arm cannot silently ship without its
# diagnostics" -- and then four more parallel-model arms were added in the same PR
# (graft_wide_*, graft_text_*) and none was added to the set, so they ran with gate
# diagnostics silently disabled. The test asserted the literal's exact contents,
# which pinned the omission in place instead of catching it.
#
# `edge_top3` is correctly excluded by this rule rather than by hand: it is a
# CrossAttentionKnowledgeModel arm with no source-only branch, and it does not
# match the naming prefix.
GRAFT_DIAGNOSTIC_METHODS: frozenset[str] = frozenset(
    name for name in METHODS_BY_NAME if name.startswith("graft_")
)


@dataclass(frozen=True)
class Cell:
    """One unit of work: a single training run at one seed."""

    backbone: str
    dataset: str
    method: str
    seed: int
    encoder: str | None = None
    # First-pass prediction directory. REQUIRED for NER knowledge arms and rejected
    # for RE. See the guard in plan(): omitting it does not fail at runtime -- the
    # query builder takes `predictions_root: null` from the group config and the arm
    # reads no mentions at all, trains happily, and reports a knowledge score for a
    # model that received no knowledge.
    predictions_root: str | None = None

    @property
    def label(self) -> str:
        return f"{self.dataset}-{self.backbone}-{self.method}-s{self.seed}"

    @property
    def run_subdir(self) -> str:
        return f"{self.dataset}/{self.backbone}_{self.method}/seed_{self.seed}"


@dataclass
class CellPlan:
    """A cell resolved into the exact overrides to run."""

    cell: Cell
    experiment: str
    overrides: list[str] = field(default_factory=list)
    # How this arm accepts a backbone swap -- see BACKBONE_MODEL_IDS. A list
    # because the khp_* arms need TWO settings: the encoder and the tokenizer.
    backbone_overrides: list[str] = field(default_factory=list)

    def hydra_args(self, output_dir: str, *, epochs: int = 10) -> list[str]:
        # The mandatory settings come FIRST so a method's own overrides can raise
        # one deliberately -- Hydra takes the last occurrence. Putting
        # `dataset.batch_size=64` after `*self.overrides` silently reverted the
        # BioRED surface arms' eval batch size back to 64, i.e. re-armed the very
        # abort the override existed to avoid, while the plan still *listed* 128.
        # Anything that must not be overridable belongs after `*self.overrides`.
        return [
            f"experiment={self.experiment}",
            *self.backbone_overrides,
            "dataset.batch_size=64",
            "dataset.num_workers=8",
            *self.overrides,
            f"trainer.max_epochs={epochs}",
            # Never keep more than one checkpoint: 4/cell exhausts the ABCI quota.
            "trainer.callbacks.0.save_top_k=1",
            "trainer.callbacks.0.save_last=false",
            f"seed={self.cell.seed}",
            f"paths.output_dir={output_dir}",
            f"hydra.run.dir={output_dir}",
        ]


class MatrixError(ValueError):
    """A cell that cannot be built, with the reason stated."""


def plan(cell: Cell) -> CellPlan:
    """Resolve a cell, or raise MatrixError explaining why it is unavailable."""
    if cell.backbone not in BACKBONES:
        msg = f"unknown backbone {cell.backbone!r}; known: {', '.join(BACKBONES)}"
        raise MatrixError(msg)
    if cell.dataset not in DATASETS:
        msg = f"unknown dataset {cell.dataset!r}; known: {', '.join(DATASETS)}"
        raise MatrixError(msg)
    method = METHODS_BY_NAME.get(cell.method)
    if method is None:
        msg = (
            f"unknown method {cell.method!r}; "
            f"known: {', '.join(sorted(METHODS_BY_NAME))}"
        )
        raise MatrixError(msg)

    experiment = method.experiment_for(cell.dataset)
    if experiment is None:
        msg = (
            f"method {cell.method!r} has no experiment config for {cell.dataset!r}. "
            f"{method.note}"
        )
        raise MatrixError(msg)

    overrides = [
        # logger=none is load-bearing: the default CSVLogger crashes at a
        # validation epoch end and leaves no checkpoint. `++trainer.logger=false`
        # does NOT work -- ieforge overrides it with its own kwarg.
        "logger=none",
        f"trainer={method.trainer}",
        *method.overrides,
    ]

    is_ner = cell.dataset in NER_DATASETS
    needs_predictions = is_ner and cell.method != "base"
    if needs_predictions:
        if not cell.predictions_root:
            msg = (
                f"NER method {cell.method!r} on {cell.dataset!r} needs "
                "--predictions-root: a directory holding first-pass "
                "{train,validation,test}.jsonl. Without it the query builder "
                "resolves predictions_root=null and the arm silently trains with no "
                "mentions, i.e. no knowledge, while still reporting a score."
            )
            raise MatrixError(msg)
        overrides.append(
            f"knowledge.pipeline.query_builder.predictions_root={cell.predictions_root}"
        )
    elif cell.predictions_root:
        # Reject rather than ignore. A predictions dir on an RE cell (or on the NER
        # `base` arm, which has no knowledge pipeline) means the caller has confused
        # two families of cell, and silently dropping it would hide that.
        msg = (
            f"predictions_root was given for {cell.dataset!r}/{cell.method!r}, which "
            "has no prediction-backed query builder. RE cells and the NER `base` arm "
            "must not receive one."
        )
        raise MatrixError(msg)

    if method.needs_encoder:
        if not cell.encoder:
            msg = (
                f"method {cell.method!r} needs a pretrained encoder; pass "
                f"--encoder. {method.note}"
            )
            raise MatrixError(msg)
        overrides.append(f"+model.pretrained_encoder_path={cell.encoder}")
    elif cell.encoder:
        msg = f"method {cell.method!r} does not take an encoder"
        raise MatrixError(msg)

    if method.uses_backbone_group:
        # The group config sets encoder and tokenizer together.
        backbone_overrides = [f"backbone={cell.backbone}"]
    else:
        # khp_* arms have no /backbone in their defaults; the encoder is pinned
        # inline, so the group override raises "No match in the defaults list".
        #
        # BOTH fields must move. The khp_* roots hardcode `model.tokenizer_name`
        # to PubMedBERT, so overriding only `model.inner.model_name` leaves a
        # BioLinkBERT encoder reading PubMedBERT word pieces -- different
        # vocabularies (28895 vs 30522), no error, and every input silently
        # mistokenised. Caught on the first BioLinkBERT probe, before the cells
        # that would have carried it into the paper.
        model_id = BACKBONE_MODEL_IDS[cell.backbone]
        backbone_overrides = [
            f"model.inner.model_name={model_id}",
            f"model.tokenizer_name={model_id}",
        ]

    return CellPlan(
        cell=cell,
        experiment=experiment,
        overrides=overrides,
        backbone_overrides=backbone_overrides,
    )


def expand(
    backbones: list[str],
    datasets: list[str],
    methods: list[str],
    seeds: list[int],
    encoder: str | None = None,
    predictions: dict[str, str] | None = None,
) -> list[Cell]:
    """Cartesian product, skipping combinations with no config.

    Silently dropping unavailable cells would understate coverage, so callers
    should report what `unavailable()` returns alongside the plan.

    `predictions` maps dataset -> first-pass prediction directory, and is PER
    DATASET rather than a single path because each NER dataset has its own K=16
    lattice. One shared path would hand BC2GM the chem predictions, which is not a
    crash: the arm would train on mentions from the wrong corpus. It is attached only
    to cells that need it, so `plan()`'s two-way guard keeps working.
    """
    cells = []
    for dataset in datasets:
        for backbone in backbones:
            for method in methods:
                spec = METHODS_BY_NAME.get(method)
                if spec is None or spec.experiment_for(dataset) is None:
                    continue
                cells.extend(
                    Cell(
                        backbone=backbone,
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        encoder=encoder if spec.needs_encoder else None,
                        predictions_root=(
                            (predictions or {}).get(dataset)
                            if dataset in NER_DATASETS and method != "base"
                            else None
                        ),
                    )
                    for seed in seeds
                )
    return cells


def unavailable(datasets: list[str], methods: list[str]) -> list[str]:
    """Human-readable reasons for (dataset, method) pairs that cannot run."""
    out = []
    for dataset in datasets:
        for method in methods:
            spec = METHODS_BY_NAME.get(method)
            if spec is None:
                out.append(f"{method}: unknown method")
            elif spec.experiment_for(dataset) is None:
                out.append(f"{dataset}/{method}: no config ({spec.note})")
    return out


def controls_for(methods: list[str]) -> list[str]:
    """Control arms required to interpret the given methods, not already listed.

    Deliberately ONE level deep. Making it transitive is tempting -- `edge_top3`
    needs `graft_real`, which needs `graft_masked` -- but it silently widens every
    EXISTING sweep: `binary` declares `ev_cold`, which declares `base`, so
    `--methods binary` would grow from 2 arms to 3 and change what a re-run of a
    recorded experiment costs and contains. A method that needs a control's control
    must declare BOTH, which is visible in the registry rather than emergent from a
    graph walk. `edge_top3` does exactly that, and
    `test_transitively_needed_controls_are_declared_explicitly` fails if a future
    method forgets.
    """
    needed: list[str] = []
    for name in methods:
        spec = METHODS_BY_NAME.get(name)
        needed.extend(
            control
            for control in (spec.controls if spec else ())
            if control not in methods
        )
    return sorted(set(needed))
