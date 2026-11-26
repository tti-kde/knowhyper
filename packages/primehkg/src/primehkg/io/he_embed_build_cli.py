from __future__ import annotations

from typing import Any

import typer
from typer.main import get_command

from primehkg.features.embedding_hash import source_text_hash
from primehkg.features.embedding_schema import EMBEDDING_DTYPE_V1
from primehkg.io.cli_console import cli_error

app = typer.Typer(
    rich_markup_mode="rich",
    help="Build he_features_embedding from he_features_text.",
)


@app.command()
def main(
    text_dir: str = typer.Option(
        ..., "--text-dir", help="Path to he_features_text DatasetDict"
    ),
    model_id: str = typer.Option(..., "--model-id", help="Encoder model identifier"),
    out_dir: str = typer.Option(..., "--out-dir", help="Output directory"),
    batch_size: int = typer.Option(512, "--batch-size", min=1),
    encode_batch_size: int = typer.Option(
        32,
        "--encode-batch-size",
        min=1,
        help="GPU batch size for sentence-transformers encode (ignored for stub).",
    ),
    encoder: str = typer.Option(
        "stub", "--encoder", help="stub or sentence-transformers"
    ),
    model_name: str | None = typer.Option(
        None, "--model-name", help="HF model name for sentence-transformers"
    ),
) -> None:
    """Encode HE text rows into he_features_embedding."""
    from pathlib import Path

    from datasets import DatasetDict, load_from_disk

    from primehkg.features.encoders.stub import StubTextEncoder

    try:
        text_dd = load_from_disk(text_dir)
    except Exception as exc:
        cli_error(f"failed to load text dataset at {text_dir!r}: {exc}")
        raise typer.Exit(1) from exc

    if not isinstance(text_dd, DatasetDict) or "he_features_text" not in text_dd:
        cli_error(f"expected he_features_text config at {text_dir!r}")
        raise typer.Exit(1)

    text_ds = text_dd["he_features_text"]
    required_columns = {"he_id", "template_id", "text"}
    missing_columns = sorted(required_columns - set(text_ds.column_names))
    if missing_columns:
        cli_error(f"missing columns in he_features_text: {missing_columns}")
        raise typer.Exit(1)

    if encoder == "sentence-transformers":
        if not model_name:
            cli_error("--model-name is required for sentence-transformers encoder")
            raise typer.Exit(1)
        from primehkg.features.encoders.sentence_transformers_encoder import (
            SentenceTransformersEncoder,
        )

        text_encoder = SentenceTransformersEncoder(
            model_name=model_name,
            model_id=model_id,
            encode_batch_size=encode_batch_size,
        )
    else:
        text_encoder = StubTextEncoder(model_id=model_id, embedding_dim=8)

    typer.echo(f"Encoding {text_ds.num_rows} texts with {model_id}")

    def encode_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        chunk_texts = [str(text) for text in batch["text"]]
        chunk_hashes = [source_text_hash(text) for text in chunk_texts]
        vectors = text_encoder.encode_batch(
            chunk_texts, source_text_hashes=chunk_hashes
        )
        return {
            "he_id": [str(he_id) for he_id in batch["he_id"]],
            "source_template_id": [
                str(template_id) for template_id in batch["template_id"]
            ],
            "input_kind": ["hyperedge_text"] * len(chunk_texts),
            "model_id": [model_id] * len(chunk_texts),
            "encoder_family": ["biomedical_text"] * len(chunk_texts),
            "encoder_revision": [""] * len(chunk_texts),
            "preprocess_version": ["text_nfkc_strip_v1"] * len(chunk_texts),
            "embedding_dim": [len(vec) for vec in vectors],
            "vector": vectors,
            "dtype": [EMBEDDING_DTYPE_V1] * len(chunk_texts),
            "source_text_hash": chunk_hashes,
            "input_hash": chunk_hashes,
            "source": ["primehkg-he-text-build"] * len(chunk_texts),
            "source_version": ["unknown"] * len(chunk_texts),
            "license_tag": [""] * len(chunk_texts),
        }

    ds = text_ds.map(
        encode_batch,
        batched=True,
        batch_size=batch_size,
        writer_batch_size=batch_size,
        remove_columns=text_ds.column_names,
        desc="Encoding HE text",
    )
    dd = DatasetDict({"he_features_embedding": ds})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(out))
    typer.echo(f"Saved {ds.num_rows} embedding rows to {out}")


def entrypoint() -> None:
    get_command(app)()


if __name__ == "__main__":
    entrypoint()
