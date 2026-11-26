"""KnowHyper knowledge feature channels (registered into ieforge)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from ieforge.knowledge.core.base import IEKnowledgeFeatureChannel
from ieforge.registry import register

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ieforge.knowledge.core.schema import KnowledgeContext


@register("knowledge_channel", name="ner_mention_spans")
class NERMentionSpanChannel(IEKnowledgeFeatureChannel):
    """Collate predicted mention character spans for span-local NER fusion.

    The base NER collator already emits token ``offset_mapping``. Keeping the
    first-pass character spans as a separate feature avoids a second tokenizer and
    guarantees that the model builds its mask against exactly the token sequence it
    classifies. Empty-prediction rows are represented explicitly with an all-false
    mention mask.
    """

    SPANS_KEY = "knowledge_ner_mention_spans"
    MASK_KEY = "knowledge_ner_mention_mask"
    CONFIDENCE_KEY = "knowledge_ner_mention_confidences"
    ENTROPY_KEY = "knowledge_ner_mention_entropies"
    DECODED_KEY = "knowledge_ner_mention_decoded"
    LABELS_KEY = "knowledge_ner_mention_labels"

    def collate(self, contexts: Sequence[KnowledgeContext]) -> Mapping[str, Any]:
        spans_by_context: list[list[tuple[int, int]]] = []
        confidences_by_context: list[list[float]] = []
        entropies_by_context: list[list[float]] = []
        decoded_by_context: list[list[bool]] = []
        labels_by_context: list[list[str]] = []
        for ctx in contexts:
            raw_spans = ctx.query.metadata.get("entity_spans", [])
            raw_confidences = ctx.query.metadata.get(
                "entity_confidences", [1.0] * len(raw_spans)
            )
            raw_entropies = ctx.query.metadata.get(
                "entity_entropies", [0.0] * len(raw_spans)
            )
            raw_sources = ctx.query.metadata.get(
                "entity_sources", ["decoded"] * len(raw_spans)
            )
            raw_labels = ctx.query.metadata.get("entity_types", [""] * len(raw_spans))
            lengths = {
                len(raw_spans),
                len(raw_confidences),
                len(raw_entropies),
                len(raw_sources),
                len(raw_labels),
            }
            if len(lengths) != 1:
                msg = "NER mention metadata fields must have identical lengths"
                raise ValueError(msg)
            spans: list[tuple[int, int]] = []
            for raw_span in raw_spans:
                if len(raw_span) != 2:
                    msg = f"NER mention span must contain start/end, got {raw_span!r}"
                    raise ValueError(msg)
                start, end = int(raw_span[0]), int(raw_span[1])
                if start < 0 or end <= start or end > len(ctx.query.text):
                    msg = (
                        "NER mention span is outside query text: "
                        f"span=({start}, {end}), text_length={len(ctx.query.text)}"
                    )
                    raise ValueError(msg)
                spans.append((start, end))
            confidences = [float(value) for value in raw_confidences]
            entropies = [float(value) for value in raw_entropies]
            if any(not 0.0 <= value <= 1.0 for value in confidences + entropies):
                msg = "NER mention confidence and entropy must be in [0, 1]"
                raise ValueError(msg)
            spans_by_context.append(spans)
            confidences_by_context.append(confidences)
            entropies_by_context.append(entropies)
            decoded_by_context.append(
                [str(source) == "decoded" for source in raw_sources]
            )
            labels_by_context.append([str(label) for label in raw_labels])

        max_mentions = max((len(spans) for spans in spans_by_context), default=0)
        width = max(max_mentions, 1)
        span_tensor = torch.zeros((len(contexts), width, 2), dtype=torch.long)
        span_mask = torch.zeros((len(contexts), width), dtype=torch.bool)
        confidence_tensor = torch.zeros((len(contexts), width), dtype=torch.float)
        entropy_tensor = torch.zeros((len(contexts), width), dtype=torch.float)
        decoded_tensor = torch.zeros((len(contexts), width), dtype=torch.bool)
        padded_labels: list[list[str]] = []
        for row, spans in enumerate(spans_by_context):
            labels = labels_by_context[row]
            padded_labels.append([*labels, *([""] * (width - len(labels)))])
            if not spans:
                continue
            span_tensor[row, : len(spans)] = torch.tensor(spans, dtype=torch.long)
            span_mask[row, : len(spans)] = True
            confidence_tensor[row, : len(spans)] = torch.tensor(
                confidences_by_context[row], dtype=torch.float
            )
            entropy_tensor[row, : len(spans)] = torch.tensor(
                entropies_by_context[row], dtype=torch.float
            )
            decoded_tensor[row, : len(spans)] = torch.tensor(
                decoded_by_context[row], dtype=torch.bool
            )
        return {
            self.SPANS_KEY: span_tensor,
            self.MASK_KEY: span_mask,
            self.CONFIDENCE_KEY: confidence_tensor,
            self.ENTROPY_KEY: entropy_tensor,
            self.DECODED_KEY: decoded_tensor,
            self.LABELS_KEY: padded_labels,
        }


@register("knowledge_channel", name="incidence")
class IncidenceChannel(IEKnowledgeFeatureChannel):
    """Batch per-graph sub-HG incidence into one offset-indexed (PyG-Batch) layout.

    Reads ``incidence_*`` leaf tensors that ``IncidenceIntegrator`` placed in each
    context's ``graph_features``, then concatenates them across the batch, offsetting
    ``node_idx``/``edge_idx`` by cumulative node/edge counts, and emits a
    ``graph_of_node`` id vector + per-graph ``num_nodes``/``num_edges`` so the model's
    ``encode_batch`` can run the AllSet layers once and split results per graph.
    Plain leaf tensors → picklable → DataLoader num_workers > 0.
    """

    def collate(self, contexts: Sequence[KnowledgeContext]) -> Mapping[str, Any]:
        """Offset-concatenate per-graph incidence tensors into a flat batch.

        Every context contributes exactly one entry to ``num_nodes``/``num_edges``
        so the downstream model always receives B graphs — matching its [B, H]
        cross-attention query.  A context that lacks ``incidence_node_feats``
        (i.e. never passed through IncidenceIntegrator) contributes a *zero-node*
        placeholder (num_nodes=0, num_edges=0, no membership rows).
        If *no* context has incidence at all, ``{}`` is returned so the model's
        absent-keys / require_knowledge path can handle it.
        """
        # First pass: discover feature dim from any context that has incidence.
        feat_dim: int | None = None
        for ctx in contexts:
            nf = ctx.graph_features.get("incidence_node_feats")
            if nf is not None:
                feat_dim = int(nf.shape[1])
                break
        if feat_dim is None:
            # No context in this batch has incidence — let the model decide.
            return {}

        node_feats, node_missing = [], []
        node_idx, edge_idx, role_id, sem_id, pol_id = [], [], [], [], []
        graph_of_node, num_nodes, num_edges = [], [], []
        modality_masks: list[torch.Tensor] = []
        slot_modality_list: list[torch.Tensor] = []
        semantics_list: list[torch.Tensor] = []
        seed_list: list[torch.Tensor] = []
        modality_m: int | None = None
        pad_dim: int | None = None
        node_off = edge_off = 0
        for g, ctx in enumerate(contexts):
            gf = ctx.graph_features
            nf = gf.get("incidence_node_feats")
            if nf is None:
                # Zero-node placeholder: keeps per-graph tensors length-B.
                num_nodes.append(0)
                num_edges.append(0)
                node_feats.append(torch.zeros(0, feat_dim, dtype=torch.float))
                node_missing.append(torch.zeros(0, dtype=torch.bool))
                # No membership rows; offsets unchanged.
                continue
            n = int(nf.shape[0])
            e = int(gf["incidence_num_edges"].item())
            node_feats.append(nf)
            node_missing.append(gf["incidence_node_missing"])
            node_idx.append(gf["incidence_node_idx"] + node_off)
            edge_idx.append(gf["incidence_edge_idx"] + edge_off)
            role_id.append(gf["incidence_role_id"])
            sem_id.append(gf["incidence_sem_id"])
            pol_id.append(gf["incidence_pol_id"])
            graph_of_node.append(torch.full((n,), g, dtype=torch.long))
            num_nodes.append(n)
            num_edges.append(e)
            node_off += n
            edge_off += e
            if "incidence_node_modality_mask" in gf:
                modality_masks.append(gf["incidence_node_modality_mask"])
                slot_modality_list.append(gf["incidence_node_slot_modality"])
                if "incidence_node_semantics" in gf:
                    semantics_list.append(gf["incidence_node_semantics"])
                if "incidence_node_is_seed" in gf:
                    seed_list.append(gf["incidence_node_is_seed"])
                if modality_m is None:
                    modality_m = int(gf["incidence_modality_m"])
                    pad_dim = int(gf["incidence_pad_dim"])
        cat = lambda xs: torch.cat(xs, dim=0)  # noqa: E731
        result: dict[str, Any] = {
            "knowledge_incidence_node_feats": cat(node_feats),
            "knowledge_incidence_node_missing": cat(node_missing),
            "knowledge_incidence_node_idx": cat(node_idx)
            if node_idx
            else torch.zeros(0, dtype=torch.long),
            "knowledge_incidence_edge_idx": cat(edge_idx)
            if edge_idx
            else torch.zeros(0, dtype=torch.long),
            "knowledge_incidence_role_id": cat(role_id)
            if role_id
            else torch.zeros(0, dtype=torch.long),
            "knowledge_incidence_sem_id": cat(sem_id)
            if sem_id
            else torch.zeros(0, dtype=torch.long),
            "knowledge_incidence_pol_id": cat(pol_id)
            if pol_id
            else torch.zeros(0, dtype=torch.long),
            "knowledge_incidence_graph_of_node": cat(graph_of_node)
            if graph_of_node
            else torch.zeros(0, dtype=torch.long),
            "knowledge_incidence_num_nodes": torch.tensor(num_nodes, dtype=torch.long),
            "knowledge_incidence_num_edges": torch.tensor(num_edges, dtype=torch.long),
        }
        if modality_masks and modality_m is not None and pad_dim is not None:
            # Back-fill zero-node placeholders and emit the four multimodal keys.
            all_masks: list[torch.Tensor] = []
            all_slots: list[torch.Tensor] = []
            all_semantics: list[torch.Tensor] = []
            all_seed: list[torch.Tensor] = []
            for ctx in contexts:
                gf = ctx.graph_features
                nf = gf.get("incidence_node_feats")
                if nf is None:
                    all_masks.append(torch.zeros(0, modality_m))
                    all_slots.append(torch.full((0, modality_m), -1, dtype=torch.long))
                    if semantics_list:
                        all_semantics.append(torch.zeros(0, dtype=torch.long))
                    if seed_list:
                        all_seed.append(torch.zeros(0, dtype=torch.bool))
                elif "incidence_node_modality_mask" in gf:
                    all_masks.append(gf["incidence_node_modality_mask"])
                    all_slots.append(gf["incidence_node_slot_modality"])
                    if semantics_list:
                        all_semantics.append(
                            gf.get(
                                "incidence_node_semantics",
                                torch.zeros(int(nf.shape[0]), dtype=torch.long),
                            )
                        )
                    if seed_list:
                        all_seed.append(
                            gf.get(
                                "incidence_node_is_seed",
                                torch.zeros(int(nf.shape[0]), dtype=torch.bool),
                            )
                        )
                else:
                    # Populated but single-vector graph mixed in — guard so we never crash.
                    n_sv = int(nf.shape[0])
                    all_masks.append(torch.zeros(n_sv, modality_m))
                    all_slots.append(
                        torch.full((n_sv, modality_m), -1, dtype=torch.long)
                    )
                    if semantics_list:
                        all_semantics.append(torch.zeros(n_sv, dtype=torch.long))
                    if seed_list:
                        all_seed.append(torch.zeros(n_sv, dtype=torch.bool))
            result["knowledge_incidence_node_modality_mask"] = torch.cat(
                all_masks, dim=0
            )
            result["knowledge_incidence_node_slot_modality"] = torch.cat(
                all_slots, dim=0
            )
            result["knowledge_incidence_modality_m"] = torch.tensor(
                modality_m, dtype=torch.long
            )
            result["knowledge_incidence_pad_dim"] = torch.tensor(
                pad_dim, dtype=torch.long
            )
            if semantics_list:
                result["knowledge_incidence_node_semantics"] = torch.cat(
                    all_semantics, dim=0
                )
            if seed_list:
                result["knowledge_incidence_node_is_seed"] = torch.cat(all_seed, dim=0)

        mention_width = max(
            (
                int(ctx.graph_features["ner_mention_link_scores"].shape[0])
                for ctx in contexts
                if "ner_mention_link_scores" in ctx.graph_features
            ),
            default=0,
        )
        if mention_width or any(
            "incidence_node_mention_mask" in ctx.graph_features for ctx in contexts
        ):
            width = max(mention_width, 1)
            ownership_rows: list[torch.Tensor] = []
            local_ownership_rows: list[torch.Tensor] = []
            hyperedge_ownership_rows: list[torch.Tensor] = []
            score_rows: list[torch.Tensor] = []
            link_rows: list[torch.Tensor] = []
            for row, (ctx, n) in enumerate(zip(contexts, num_nodes, strict=True)):
                gf = ctx.graph_features
                edge_count = gf.get("incidence_num_edges")
                e = int(edge_count.item()) if edge_count is not None else 0
                ownership = gf.get("incidence_node_mention_mask")
                local_ownership = gf.get("incidence_local_node_mention_mask")
                hyperedge_ownership = gf.get("incidence_hyperedge_mention_mask")
                scores = gf.get("ner_mention_link_scores")
                links = gf.get("ner_mention_link_mask")
                # #171: MISSING plumbing is a hard error, never a zero-filled row.
                #
                # A zero link score is numerically identical to "linked nothing", and on the
                # strict arms (`khp_ner_strict`, `khp_ner_ev_strict`) a high no-link rate is
                # EXPECTED -- `strict_type_match: true` turns type mismatches into misses. So
                # a broken pipe and a strict arm that legitimately linked nothing produced the
                # same all-zero tensor, and the experiment would report the plumbing failure
                # as the strict effect.
                #
                # The three states are now distinct:
                #   MISSING plumbing -> raise here (the injector emits all five keys together,
                #                       so a subset present means it never ran for this row)
                #   explicit NIL     -> keys present with correct shapes, link_mask all False;
                #                       an explicitly EMPTY tensor (m == 0) is also NIL
                #   linked evidence  -> keys present, ownership aligned, link_mask True
                #
                # Per ROW, not per batch: the dangerous case is a mixed batch where the other
                # instances carry their keys, so the batch completes and only this one is
                # wrong. `row` is named in the message because otherwise the reader has to
                # search thousands of instances for it.
                # A ZERO-NODE PLACEHOLDER is not missing plumbing. This method's docstring
                # documents that "a context that lacks `incidence_node_feats` (i.e. never
                # passed through IncidenceIntegrator) contributes a *zero-node* placeholder",
                # and such a context legitimately carries no mention provenance either. It is
                # also common rather than exotic: 46% of BioRED sub-HGs are empty, so mixed
                # batches are the norm. Raising on it would break every arm that tolerates
                # empty sub-HGs -- caught by testing the documented case directly, after the
                # first version of this guard did exactly that.
                #
                # The distinguishing signal is whether the context HAS incidence: a
                # placeholder has no nodes AND no incidence features, whereas a genuinely
                # broken row has incidence but lacks the provenance keys the injector emits
                # alongside it.
                has_incidence = "incidence_node_feats" in gf
                # Written as an explicit `is None` chain rather than only a comprehension: a
                # comprehension does not narrow the optional types, so every later `.shape`
                # access becomes an `unresolved-attribute` error under `ty`.
                absent = [
                    name
                    for name, value in (
                        ("incidence_node_mention_mask", ownership),
                        ("ner_mention_link_scores", scores),
                        ("ner_mention_link_mask", links),
                    )
                    if value is None
                ]
                if absent and not has_incidence:
                    # Documented placeholder: pad it and carry on.
                    ownership_rows.append(torch.zeros((n, width), dtype=torch.bool))
                    local_ownership_rows.append(
                        torch.zeros((n, width), dtype=torch.bool)
                    )
                    hyperedge_ownership_rows.append(
                        torch.zeros((e, width), dtype=torch.bool)
                    )
                    score_rows.append(torch.zeros(width, dtype=torch.float))
                    link_rows.append(torch.zeros(width, dtype=torch.bool))
                    continue
                if ownership is None or scores is None or links is None:
                    msg = (
                        f"mention provenance is incomplete for row {row}: missing "
                        f"{absent}. The injector emits ownership, scores and mask together, "
                        "so a partial set means mention provenance never ran for this "
                        "instance. Zero-filling it would be indistinguishable from a "
                        "legitimate no-link (explicit NIL), which is the state strict "
                        "linking produces on purpose -- see issue #171. To express 'nothing "
                        "linked', supply the keys with an all-False mask."
                    )
                    raise ValueError(msg)
                m = int(scores.shape[0])
                expected_node_shape = (n, m)
                if (
                    tuple(ownership.shape) != expected_node_shape
                    or tuple(links.shape) != (m,)
                    or (
                        local_ownership is not None
                        and tuple(local_ownership.shape) != expected_node_shape
                    )
                    or (
                        hyperedge_ownership is not None
                        and tuple(hyperedge_ownership.shape) != (e, m)
                    )
                ):
                    msg = (
                        "mention provenance feature shapes do not align: "
                        f"ownership={tuple(ownership.shape)}, scores={tuple(scores.shape)}, "
                        f"links={tuple(links.shape)}, nodes={n}, edges={e}"
                    )
                    raise ValueError(msg)
                padded_ownership = torch.zeros((n, width), dtype=torch.bool)
                padded_local = torch.zeros((n, width), dtype=torch.bool)
                padded_hyperedge = torch.zeros((e, width), dtype=torch.bool)
                padded_scores = torch.zeros(width, dtype=torch.float)
                padded_links = torch.zeros(width, dtype=torch.bool)
                if m:
                    padded_ownership[:, :m] = ownership
                    if local_ownership is not None:
                        padded_local[:, :m] = local_ownership
                    if hyperedge_ownership is not None:
                        padded_hyperedge[:, :m] = hyperedge_ownership
                    padded_scores[:m] = scores
                    padded_links[:m] = links
                ownership_rows.append(padded_ownership)
                local_ownership_rows.append(padded_local)
                hyperedge_ownership_rows.append(padded_hyperedge)
                score_rows.append(padded_scores)
                link_rows.append(padded_links)
            result["knowledge_incidence_node_mention_mask"] = torch.cat(
                ownership_rows, dim=0
            )
            result["knowledge_incidence_local_node_mention_mask"] = torch.cat(
                local_ownership_rows, dim=0
            )
            result["knowledge_incidence_hyperedge_mention_mask"] = torch.cat(
                hyperedge_ownership_rows, dim=0
            )
            result["knowledge_ner_mention_link_scores"] = torch.stack(score_rows)
            result["knowledge_ner_mention_link_mask"] = torch.stack(link_rows)
        return result
