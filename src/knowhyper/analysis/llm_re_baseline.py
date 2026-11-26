"""Protocol-matched instruction-tuned LLM baseline for DDI (issue #314, Agent B).

Two arms, fixed before any test instance is scored:

* **noKG** -- the LLM sees exactly the string PubMedBERT sees (the BLURB DDI
  sentence with both target drugs masked as ``@DRUG$``) plus the label schema;
* **KG** -- the same, plus the retrieved sub-hypergraph KnowHyper's fusion arm
  consumed for that instance, rendered as text by the project's own canonical
  serializer (``[HG] semantics=...; ... [/HG]`` blocks).  Retrieval itself is
  untouched: the evidence is read from the SubHG cache the fusion arm read.

Everything that could be tuned is pinned here as data: the label schema and its
one-line definitions, the demonstration selection rule (first train example
per label, in dataset order, under a length cap), the system prompt, the output
format (``{"label": "..."}``) and the invalid-output policy (predict the
negative label, count it).  The prompt fingerprint is recorded with every run so
a later run can prove it used the same prompt.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from knowhyper.analysis.re_prediction_export import PredictionRecord
from knowhyper.pipeline.types import SubHypergraph

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

PROMPT_VERSION = "ddi_llm_v1"

#: Prompt order: the order labels are listed in the schema and the order the
#: demonstrations appear in.  Negative first, then the four positive classes.
#: This is a prompt-policy constant and is deliberately NOT the export id order.
DDI_LABELS: tuple[str, ...] = (
    "DDI-false",
    "DDI-mechanism",
    "DDI-effect",
    "DDI-advise",
    "DDI-int",
)
DDI_NEGATIVE = "DDI-false"

#: Export id order: ieforge's own ``_RE_TASK_REGISTRY["ddi"].label_list``, which
#: is what every recorded ``predictions.jsonl`` encodes (``DDI-false`` is **4**,
#: not 0).  The LLM arms must write the same ids or two export files for the
#: same instance would disagree on what ``gold_label_id: 4`` means.  Prefer
#: :func:`label_id_order` to read it from the live dataset; this literal is the
#: offline fallback and is pinned by a test against ieforge.
DDI_LABEL_ID_ORDER: tuple[str, ...] = (
    "DDI-mechanism",
    "DDI-effect",
    "DDI-advise",
    "DDI-int",
    "DDI-false",
)


def label_id_order(dataset: Any = None) -> tuple[str, ...]:
    """Label order used for ``*_label_id`` in the export, from the dataset itself.

    ``dataset`` is any object exposing ``label_list`` (ieforge's RE dataset
    does).  Falling back to :data:`DDI_LABEL_ID_ORDER` keeps the helper usable
    in unit tests, but a live run passes the dataset so the ids follow whatever
    ieforge says today rather than a literal that can drift.
    """
    labels = getattr(dataset, "label_list", None)
    if labels:
        return tuple(str(label) for label in labels)
    return DDI_LABEL_ID_ORDER


#: One-line definitions paraphrased from the DDIExtraction 2013 guidelines.
DDI_LABEL_DEFINITIONS: Mapping[str, str] = {
    "DDI-false": "the sentence does not state an interaction between the two marked drugs",
    "DDI-mechanism": "a pharmacokinetic mechanism is described (absorption, metabolism, plasma levels, clearance, ...)",
    "DDI-effect": "an effect or a pharmacodynamic consequence of the interaction is described",
    "DDI-advise": "a recommendation or advice about using the two drugs together is given",
    "DDI-int": "an interaction is stated without giving any mechanism, effect or advice",
}

#: Demonstrations longer than this are skipped in favour of the next example of
#: the same label, so the few-shot block stays short.  Characters of the masked
#: sentence; fixed before any split other than train was read.
DEMO_MAX_CHARS = 320

SYSTEM_PROMPT = (
    "You are an expert annotator for drug-drug interaction extraction. In each "
    "sentence, the two drugs of interest are both masked as @DRUG$. Decide which "
    "single label describes the relation between these two masked drugs. Answer "
    'with a JSON object of the form {"label": "<LABEL>"} and nothing else.'
)

EVIDENCE_INSTRUCTION = (
    "A block of knowledge-graph evidence retrieved for the two drugs may follow the "
    "sentence, as [HG] ... [/HG] items listing typed hyperedges (semantics, member "
    "entities with their roles, polarity). [HEAD_ENTITY] and [TAIL_ENTITY] stand for "
    "the first and second masked drug. The evidence may be incomplete or irrelevant; "
    "the sentence remains the primary source."
)

_JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)


@dataclass(frozen=True)
class Demonstration:
    """One fixed few-shot example (text and label only; never test data)."""

    instance_id: str
    text: str
    label: str


@dataclass(frozen=True)
class PromptSpec:
    """Everything that defines the prompt; hashed into ``fingerprint``."""

    labels: tuple[str, ...] = DDI_LABELS
    negative_label: str = DDI_NEGATIVE
    definitions: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(DDI_LABEL_DEFINITIONS)
    )
    system_prompt: str = SYSTEM_PROMPT
    evidence_instruction: str = EVIDENCE_INSTRUCTION
    demonstrations: tuple[Demonstration, ...] = ()
    with_evidence: bool = False
    version: str = PROMPT_VERSION

    def fingerprint(self) -> str:
        """Stable digest of the whole prompt policy, demonstrations included."""
        payload = {
            "labels": list(self.labels),
            "negative_label": self.negative_label,
            "definitions": dict(self.definitions),
            "system_prompt": self.system_prompt,
            "evidence_instruction": self.evidence_instruction
            if self.with_evidence
            else "",
            "demonstrations": [dataclasses.asdict(d) for d in self.demonstrations],
            "with_evidence": self.with_evidence,
            "version": self.version,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def select_demonstrations(
    train_examples: Iterable[Any],
    *,
    labels: Sequence[str] = DDI_LABELS,
    max_chars: int = DEMO_MAX_CHARS,
) -> tuple[Demonstration, ...]:
    """First train example per label, in dataset order, under the length cap.

    Deterministic and test-blind by construction.  Raises if a label has no
    qualifying example rather than silently dropping a class from the prompt.
    """
    chosen: dict[str, Demonstration] = {}
    for example in train_examples:
        label = str(example.label)
        if label in chosen or label not in labels:
            continue
        text = str(example.text)
        if len(text) > max_chars:
            continue
        chosen[label] = Demonstration(
            instance_id=str(example.id), text=text, label=label
        )
        if len(chosen) == len(labels):
            break
    missing = [label for label in labels if label not in chosen]
    if missing:
        msg = f"no train demonstration under {max_chars} chars for labels {missing}"
        raise ValueError(msg)
    return tuple(chosen[label] for label in labels)


def _label_schema(spec: PromptSpec) -> str:
    lines = ["Labels:"]
    lines.extend(f"- {label}: {spec.definitions[label]}" for label in spec.labels)
    return "\n".join(lines)


def _user_turn(text: str, evidence: str | None) -> str:
    turn = f"Sentence: {text}"
    if evidence:
        turn += f"\nEvidence: {evidence}"
    return turn + "\nAnswer:"


def render_messages(
    spec: PromptSpec, text: str, evidence: str | None = None
) -> list[dict[str, str]]:
    """Chat messages: system + schema, fixed demonstrations, then the query.

    Demonstrations never carry evidence, in either arm, so the two arms differ
    in exactly one place: the evidence line of the final user turn.
    """
    if evidence and not spec.with_evidence:
        msg = "evidence given to a prompt spec that does not declare with_evidence"
        raise ValueError(msg)
    system = spec.system_prompt + "\n\n" + _label_schema(spec)
    if spec.with_evidence:
        system += "\n\n" + spec.evidence_instruction
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for demo in spec.demonstrations:
        messages.append({"role": "user", "content": _user_turn(demo.text, None)})
        messages.append(
            {"role": "assistant", "content": json.dumps({"label": demo.label})}
        )
    messages.append({"role": "user", "content": _user_turn(text, evidence)})
    return messages


def parse_label(output: str, labels: Sequence[str]) -> tuple[str | None, str]:
    r"""Return ``(label, status)``; status is ``json``, ``substring`` or ``invalid``.

    ``json``: the first ``{...}`` block parses and its ``label`` is one of
    ``labels`` (case-insensitive).  ``substring``: no valid JSON but exactly one
    label occurs in the output as a whole token -- the match is bounded by
    ``[\w-]`` on both sides so ``DDI-int`` does not fire inside
    ``DDI-interaction``.  Anything else is ``invalid`` and the caller predicts
    the negative label and counts it.
    """
    lowered = {label.lower(): label for label in labels}
    for block in _JSON_BLOCK.finditer(output):
        try:
            payload = json.loads(block.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            value = str(payload.get("label", "")).strip().lower()
            if value in lowered:
                return lowered[value], "json"
    lowered_output = output.lower()
    found = {
        label
        for label in labels
        if re.search(rf"(?<![\w-]){re.escape(label.lower())}(?![\w-])", lowered_output)
    }
    if len(found) == 1:
        return found.pop(), "substring"
    return None, "invalid"


def annotate_pair_roles(
    sub: SubHypergraph,
    head_mentions: Sequence[str],
    tail_mentions: Sequence[str],
) -> SubHypergraph:
    """Assign head/tail pair roles to linked mentions by surface form.

    The fusion arms' cache carries ``linked_mentions_v1`` provenance with
    ``pair_role: null``; the canonical serializer needs
    ``linked_mentions_role_v1`` to write ``[HEAD_ENTITY]``/``[TAIL_ENTITY]``
    instead of raw node ids.

    **What is identical and what is not.** The hyperedges, node ids and seed
    nodes are untouched, so the EVIDENCE is the same graph the fusion arm
    consumed. Only the pair-role LABELS are added here, and they are matched on
    surface text, case-insensitively -- not on node id.

    That has a known consequence, documented rather than silently fixed
    (#314 review F4): DDI contains pairs whose head and tail share a surface
    form, and head is tested first, so **both** linked mentions of such a pair
    are labelled ``head`` and the prompt renders no ``[TAIL_ENTITY]`` at all.
    The behaviour is frozen deliberately: the recorded LLM run used exactly this
    rendering, and changing it afterwards would make the record describe a run
    that never happened. Matching on node id is the right fix for any FUTURE
    run. The caveat is repeated in the LLM record's README.

    Note the both-endpoints diagnostic reads the hyperedges rather than the
    rendered text, so the render cap does not affect it -- but it does derive
    head/tail from these same `pair_role` labels, so a same-surface pair yields
    an empty tail set and the diagnostic returns False for that instance. On the
    recorded DDI cache that costs nothing (every both-endpoint edge there is
    binary and both roles resolve), but it is an undercount in principle, not a
    guarantee.
    """
    # `str(m)` rather than `m.lower()`: this is fed straight from
    # `example.entities[...].mention`, and the project's canonical reader for the
    # same field (`knowhyper/ieforge/request.py:_surface_forms`) coerces defensively. A non-string
    # mention would otherwise raise AttributeError mid-run (#314 review F8).
    heads = {str(m).lower() for m in head_mentions if m is not None}
    tails = {str(m).lower() for m in tail_mentions if m is not None}
    provenance = dict(sub.provenance)
    annotated = []
    for record in provenance.get("linked_mentions", []):
        surface = str(record.get("text", "")).lower()
        role = "head" if surface in heads else "tail" if surface in tails else None
        annotated.append({**record, "pair_role": role})
    provenance["linked_mentions"] = annotated
    provenance["linked_mentions_schema"] = "linked_mentions_role_v1"
    return SubHypergraph(
        seed_node_ids=sub.seed_node_ids,
        node_ids=sub.node_ids,
        hyperedges=sub.hyperedges,
        provenance=provenance,
    )


def recorded_instance_id(example_id: str) -> str:
    """Map a pair-row id to the id the recorded RE exports used.

    The 2026-07/08 exports predate the ``#i-j`` pair suffix ieforge now appends,
    so it is stripped; DDI has one pair per sentence row, so the base is unique.
    """
    return example_id.split("#", 1)[0]


def cache_entry_path(cache_dir: Path, example_id: str) -> Path | None:
    """Locate the SubHG cache entry for ``example_id``, or ``None`` if absent.

    Two key spellings are in circulation for the same instance: caches written
    before ieforge appended the ``#i-j`` pair suffix are keyed by the bare row id
    (``...p0.json``), later ones by the full example id (``...p0#0-1.json``).
    The full id is tried first so a cache that distinguishes pairs is never read
    through the ambiguous stripped key; the stripped key is the fallback.  A
    miss returns ``None`` -- the caller counts it, so a wrong cache directory
    shows up as a ``missing`` count rather than as silently empty evidence.
    """
    full = cache_dir / f"{example_id}.json"
    if full.is_file():
        return full
    stripped = cache_dir / f"{recorded_instance_id(example_id)}.json"
    if stripped.is_file():
        return stripped
    return None


def prediction_record(
    *,
    instance_id: str,
    gold_label: str,
    predicted_label: str,
    labels: Sequence[str],
    valid: bool,
    split: str = "test",
    dataset: str = "ddi",
) -> PredictionRecord:
    """Payload-free record in the shared export schema (score = parse validity).

    ``labels`` is the **export id order** -- ieforge's ``label_list``, via
    :func:`label_id_order` -- not the prompt order, so an LLM record and a
    recorded PubMedBERT record for the same instance encode the same label with
    the same id.
    """
    order = list(labels)
    return PredictionRecord(
        dataset=dataset,
        split=split,
        instance_id=instance_id,
        gold_label_id=order.index(gold_label),
        gold_label=gold_label,
        predicted_label_id=order.index(predicted_label),
        predicted_label=predicted_label,
        predicted_score=1.0 if valid else 0.0,
    )
