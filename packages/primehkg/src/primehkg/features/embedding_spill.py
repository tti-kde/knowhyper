from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from primehkg.features.embedding_schema import (
    HE_EMBEDDING_PK_COLUMNS,
    NODE_EMBEDDING_PK_COLUMNS,
    EmbeddingTarget,
    embedding_row_sort_key,
)


class SortedSpillWriter:
    """Write each encoded chunk as one in-chunk-sorted Parquet shard for a single target.

    Shards land under ``spill_dir/<target>/shard_NNNNNN.parquet`` in generation order, each
    internally sorted by ``embedding_row_sort_key``. The Arrow schema is inferred once from the
    first non-empty chunk and reused for every shard so the shards are mergeable and the round
    trip type is stable.
    """

    def __init__(self, spill_dir: Path, target: EmbeddingTarget) -> None:
        """Initialise the writer and create the per-target spill directory."""
        self._dir = Path(spill_dir) / target
        self._dir.mkdir(parents=True, exist_ok=True)
        self._target = target
        self._n = 0
        self._rows = 0
        self._schema: pa.Schema | None = None

    def write_chunk(self, rows: list[dict[str, Any]]) -> None:
        """Sort *rows* by sort-key and write them as a single Parquet shard.

        Empty chunks are silently skipped and do not produce a shard file.
        """
        if not rows:
            return
        rows.sort(key=lambda row: embedding_row_sort_key(row, target=self._target))
        table = pa.Table.from_pylist(rows)
        if self._schema is None:
            self._schema = table.schema
        else:
            table = table.select(self._schema.names).cast(self._schema)
        pq.write_table(table, self._dir / f"shard_{self._n:06d}.parquet")
        self._n += 1
        self._rows += len(rows)

    @property
    def schema(self) -> pa.Schema | None:
        """Arrow schema inferred from the first non-empty chunk, or ``None`` if no chunks written."""
        return self._schema

    @property
    def row_count(self) -> int:
        """Total number of rows written across all shards."""
        return self._rows

    def shard_paths(self) -> list[Path]:
        """Return the ordered list of shard paths produced so far."""
        return [self._dir / f"shard_{i:06d}.parquet" for i in range(self._n)]


def merge_sorted_shards(
    shard_paths: list[Path],
    *,
    target: EmbeddingTarget,
    out_parquet: Path,
    schema: pa.Schema,
    batch_rows: int = 10_000,
) -> int:
    """k-way merge pre-sorted shards into one Parquet sorted by ``embedding_row_sort_key``.

    Sorts only the string PK columns (plus a synthetic shard-then-row global index as the final
    tiebreaker) at the Arrow/C++ layer, then gathers full rows via ``take`` so the embedding
    vector never round-trips through Python. The global-index tiebreaker makes the stable
    shard-then-row order explicit, reproducing a single stable ``sorted()`` over all rows.
    Returns the number of rows written.

    Memory note: all shard tables are held resident (as Arrow tables, vectors included) for the
    ``take`` phase -- roughly the full output size (~40 GB for 13.8M x 768 float32). This is far
    lighter than the prior per-row Python-dict merge but is not strictly streaming; it is bounded
    by total output, not chunk size. Acceptable on a large-RAM host. If shard volume outgrows RAM,
    load shards lazily (open ``ParquetFile`` and read only the needed row groups) during ``take``.
    """
    key_cols = (
        list(NODE_EMBEDDING_PK_COLUMNS)
        if target == "node"
        else list(HE_EMBEDDING_PK_COLUMNS)
    )
    shard_tables = [pq.read_table(str(path)) for path in shard_paths]
    if not shard_tables:
        # nothing to merge; write an empty table with the schema
        with pq.ParquetWriter(str(out_parquet), schema) as writer:
            writer.write_table(schema.empty_table())
        return 0

    # Build a small keys-only table (no vector) with shard/row provenance, in shard-then-row order.
    key_part_tables: list[pa.Table] = []
    shard_id_arrays: list[pa.Array] = []
    row_in_shard_arrays: list[pa.Array] = []
    for shard_id, shard_table in enumerate(shard_tables):
        n = shard_table.num_rows
        # cast keys to string to honor the str() contract exactly
        cols = {c: pc.cast(shard_table[c], pa.string()) for c in key_cols}
        key_part_tables.append(pa.table(cols))
        shard_id_arrays.append(pa.array([shard_id] * n, pa.int32()))
        row_in_shard_arrays.append(pa.array(range(n), pa.int64()))

    keys = pa.concat_tables(key_part_tables)
    total_rows = keys.num_rows
    keys = keys.append_column(
        "__global_index__", pa.array(range(total_rows), pa.int64())
    )
    shard_id_col = pa.chunked_array(shard_id_arrays)
    row_in_shard_col = pa.chunked_array(row_in_shard_arrays)

    sort_keys = [(c, "ascending") for c in key_cols]
    sort_keys.append(("__global_index__", "ascending"))
    order = pc.sort_indices(keys, sort_keys=sort_keys)  # ty: ignore[unresolved-attribute]  # dynamic pyarrow.compute function

    sorted_shard_id = pc.take(shard_id_col, order)
    sorted_row_in_shard = pc.take(row_in_shard_col, order)

    total = 0
    with pq.ParquetWriter(str(out_parquet), schema) as writer:
        for start in range(0, total_rows, batch_rows):
            end = min(start + batch_rows, total_rows)
            sids = sorted_shard_id[start:end].to_pylist()
            rids = sorted_row_in_shard[start:end].to_pylist()
            pieces: list[pa.Table] = []
            i = 0
            while i < len(sids):
                j = i
                while j < len(sids) and sids[j] == sids[i]:
                    j += 1
                idx = pa.array(rids[i:j], pa.int64())
                pieces.append(shard_tables[sids[i]].take(idx))
                i = j
            batch_table = pa.concat_tables(pieces).select(schema.names).cast(schema)
            writer.write_table(batch_table)
            total += batch_table.num_rows
    return total
