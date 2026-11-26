from __future__ import annotations

MetricValue = float | None


def retrieval_recall_at_k(
    retrieved_ids: list[str],
    expected_ids: list[str],
    k: int,
) -> MetricValue:
    """Return recall over unique retrieved ids in the top-k results."""
    expected_id_set = set(expected_ids)
    if not expected_id_set:
        return None

    retrieved_id_set = set(_unique_top_k_ids(retrieved_ids, k))
    return len(retrieved_id_set & expected_id_set) / len(expected_id_set)


def retrieval_precision_at_k(
    retrieved_ids: list[str],
    expected_ids: list[str],
    k: int,
) -> MetricValue:
    """Return precision over unique retrieved ids in the top-k results."""
    expected_id_set = set(expected_ids)
    if not expected_id_set:
        return None

    retrieved_id_set = set(_unique_top_k_ids(retrieved_ids, k))
    if not retrieved_id_set:
        return 0.0

    return len(retrieved_id_set & expected_id_set) / len(retrieved_id_set)


def retrieval_mrr(retrieved_ids: list[str], expected_ids: list[str]) -> MetricValue:
    """Return reciprocal rank for the first retrieved expected id."""
    expected_id_set = set(expected_ids)
    if not expected_id_set:
        return None

    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in expected_id_set:
            return 1.0 / rank
    return 0.0


def text_context_contains(
    text_context: str | None,
    expected_text: list[str],
) -> MetricValue:
    """Return whether all expected text snippets appear in the text context."""
    if not expected_text:
        return None
    if text_context is None:
        return 0.0
    return float(all(snippet in text_context for snippet in expected_text))


def compute_example_metrics(
    *,
    retrieved_ids: list[str],
    expected_ids: list[str],
    text_context: str | None,
    expected_text: list[str],
    k: int,
) -> dict[str, MetricValue]:
    """Compute all pure knowledge eval metrics for one example."""
    return {
        f"retrieval_recall_at_{k}": retrieval_recall_at_k(
            retrieved_ids,
            expected_ids,
            k,
        ),
        f"retrieval_precision_at_{k}": retrieval_precision_at_k(
            retrieved_ids,
            expected_ids,
            k,
        ),
        "retrieval_mrr": retrieval_mrr(retrieved_ids, expected_ids),
        "text_context_contains": text_context_contains(text_context, expected_text),
    }


def summarize_metrics(
    example_metrics: list[dict[str, MetricValue]],
) -> tuple[dict[str, MetricValue], dict[str, int]]:
    """Average metric values, ignoring unscored None values."""
    metric_names = _metric_names_in_order(example_metrics)
    sums = dict.fromkeys(metric_names, 0.0)
    num_scored_examples = dict.fromkeys(metric_names, 0)

    for metrics in example_metrics:
        for metric_name, metric_value in metrics.items():
            if metric_value is None:
                continue
            sums[metric_name] += metric_value
            num_scored_examples[metric_name] += 1

    summary = {
        metric_name: (
            sums[metric_name] / num_scored_examples[metric_name]
            if num_scored_examples[metric_name] > 0
            else None
        )
        for metric_name in metric_names
    }
    return summary, num_scored_examples


def _unique_top_k_ids(retrieved_ids: list[str], k: int) -> list[str]:
    unique_ids: list[str] = []
    seen_ids: set[str] = set()
    for retrieved_id in retrieved_ids[: max(k, 0)]:
        if retrieved_id in seen_ids:
            continue
        unique_ids.append(retrieved_id)
        seen_ids.add(retrieved_id)
    return unique_ids


def _metric_names_in_order(
    example_metrics: list[dict[str, MetricValue]],
) -> list[str]:
    metric_names: list[str] = []
    seen_names: set[str] = set()
    for metrics in example_metrics:
        for metric_name in metrics:
            if metric_name in seen_names:
                continue
            metric_names.append(metric_name)
            seen_names.add(metric_name)
    return metric_names
