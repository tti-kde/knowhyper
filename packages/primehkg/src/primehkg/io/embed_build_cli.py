from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import typer
from typer.main import get_command

from primehkg.features.embedding_build import (
    build_and_write_embedding_dataset,
    build_and_write_multimodal_embeddings,
)
from primehkg.features.encoders.parallel_encode import EncoderSpec, ParallelEncoderPool
from primehkg.features.encoders.stub import StubTextEncoder
from primehkg.io.cli_console import cli_error

if TYPE_CHECKING:
    from primehkg.features.encoders.base import TextEncoder

app = typer.Typer(
    rich_markup_mode="rich",
    help="Build node_features_embedding from core nodes + optional text features.",
)

CORE_DIR_OPT = typer.Option(
    ...,
    "--core-dir",
    exists=True,
    file_okay=False,
    dir_okay=True,
    help="Existing primehkg-core build (nodes split + build_manifest.json).",
)
TEXT_DATASET_DIR_OPT = typer.Option(
    None,
    "--text-dataset-dir",
    exists=True,
    file_okay=False,
    dir_okay=True,
    help="Optional features DatasetDict with node_features_text split.",
)
FEATURES_DIR_OPT = typer.Option(
    None,
    "--features-dir",
    exists=True,
    file_okay=False,
    dir_okay=True,
    help="Optional features DatasetDict with multimodal node feature splits.",
)
HE_TEXT_DIR_OPT = typer.Option(
    None,
    "--he-text-dir",
    exists=True,
    file_okay=False,
    dir_okay=True,
    help="Optional he_features_text DatasetDict for hyperedge embeddings.",
)
CACHE_DIR_OPT = typer.Option(
    None, "--cache-dir", help="Local embedding cache directory."
)
TARGETS_OPT = typer.Option(
    None,
    "--targets",
    help="Comma-separated targets for multimodal build: node,hyperedge.",
)
NODE_INPUT_KINDS_OPT = typer.Option(
    None,
    "--node-input-kinds",
    help="Comma-separated node input kinds for multimodal build.",
)
PREPROCESS_VERSION_OPT = typer.Option("text_nfkc_strip_v1", "--preprocess-version")
ENCODER_REVISION_OPT = typer.Option("", "--encoder-revision")
TEXT_MODEL_ID_OPT = typer.Option(None, "--text-model-id")
PROTEIN_MODEL_ID_OPT = typer.Option("facebook/esm2_t6_8M_UR50D", "--protein-model-id")
NUCLEOTIDE_MODEL_ID_OPT = typer.Option(
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    "--nucleotide-model-id",
)
SMILES_MODEL_ID_OPT = typer.Option("DeepChem/ChemBERTa-77M-MLM", "--smiles-model-id")
PROPERTY_MODEL_ID_OPT = typer.Option(
    "primehkg-property-normalizer:v1", "--property-model-id"
)
TRUST_REMOTE_CODE_OPT = typer.Option(
    False,
    "--trust-remote-code/--no-trust-remote-code",
    help="Allow custom model code for --encoder transformers.",
)
OUT_DIR_OPT = typer.Option(..., "--out-dir", help="Output DatasetDict directory.")
MODEL_ID_OPT = typer.Option(
    "stub:v1",
    "--model-id",
    help="Embedding model identifier stored on each row.",
)
MODALITIES_OPT = typer.Option(
    None,
    "--modalities",
    help="Comma-separated source modalities to embed (default: all collected).",
)
ENCODER_OPT = typer.Option(
    "stub",
    "--encoder",
    help="Text encoder backend (stub or sentence-transformers).",
)
MODEL_NAME_OPT = typer.Option(
    None,
    "--model-name",
    help="Hugging Face model id (required for sentence-transformers encoder).",
)
EMBEDDING_DIM_OPT = typer.Option(
    8,
    "--embedding-dim",
    help="Output dimension for the stub encoder.",
)
BATCH_SIZE_OPT = typer.Option(
    512,
    "--batch-size",
    min=1,
    help="Rows per encode_batch chunk when building embeddings.",
)
ENCODE_BATCH_SIZE_OPT = typer.Option(
    32,
    "--encode-batch-size",
    min=1,
    help="GPU batch size for sentence-transformers encode (ignored for stub).",
)
DEVICES_OPT = typer.Option(
    "auto", "--devices", help="auto | cpu | comma list e.g. cuda:0,cuda:1"
)
CHUNK_SIZE_OPT = typer.Option(
    100_000,
    "--chunk-size",
    min=1,
    help="EmbeddingInput chunk size for streaming encode",
)
CACHE_MAP_SIZE_OPT = typer.Option(
    1 << 40, "--cache-map-size", min=1 << 20, help="LMDB cache map_size in bytes"
)


_INPUT_KIND_ENCODER_FAMILY = {
    "text": "biomedical_text",
    "protein_sequence": "protein_sequence",
    "nucleotide_sequence": "nucleotide_sequence",
    "smiles": "molecular_sequence",
    "property_vector": "property_numeric",
}


def _required_encoder_families(
    *,
    targets: frozenset[str],
    node_input_kinds: frozenset[str],
) -> frozenset[str]:
    unsupported_targets = targets - {"node", "hyperedge"}
    if unsupported_targets:
        msg = f"unsupported embedding targets: {sorted(unsupported_targets)}"
        raise ValueError(msg)

    families: set[str] = set()
    if "node" in targets:
        for input_kind in node_input_kinds:
            family = _INPUT_KIND_ENCODER_FAMILY.get(input_kind)
            if family is None:
                msg = f"unsupported node input kinds: {[input_kind]}"
                raise ValueError(msg)
            families.add(family)
    if "hyperedge" in targets:
        families.add("biomedical_text")
    return frozenset(families)


def _parse_modalities(value: str | None) -> frozenset[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return None
    return frozenset(parts)


def _parse_csv_set(value: str | None) -> frozenset[str] | None:
    return _parse_modalities(value)


def _resolve_encoder(
    encoder: str,
    *,
    model_id: str,
    model_name: str | None,
    embedding_dim: int,
    encode_batch_size: int,
) -> TextEncoder:
    if encoder == "stub":
        return StubTextEncoder(embedding_dim=embedding_dim, model_id=model_id)
    if encoder == "sentence-transformers":
        if not model_name:
            cli_error("--model-name is required when --encoder sentence-transformers")
            raise typer.Exit(2)
        try:
            from primehkg.features.encoders.sentence_transformers_encoder import (
                SentenceTransformersEncoder,
            )
        except ImportError as exc:
            cli_error(str(exc))
            raise typer.Exit(2) from exc
        return SentenceTransformersEncoder(
            model_name=model_name,
            model_id=model_id,
            encode_batch_size=encode_batch_size,
        )
    cli_error(f"unsupported --encoder {encoder!r}")
    raise typer.Exit(2)


def _resolve_devices(spec: str) -> list[str]:
    """Resolve --devices spec to a device list. 'auto' = all CUDA, else cpu fallback."""
    spec = spec.strip()
    if spec and spec != "auto":
        return [d.strip() for d in spec.split(",") if d.strip()]
    try:
        import torch

        count = torch.cuda.device_count()
    except Exception:
        count = 0
    return [f"cuda:{i}" for i in range(count)] if count > 0 else ["cpu"]


def _run_build(
    *,
    core_dir: Path,
    text_dataset_dir: Path | None,
    features_dir: Path | None,
    he_text_dir: Path | None,
    cache_dir: Path | None,
    targets: str | None,
    node_input_kinds: str | None,
    preprocess_version: str,
    encoder_revision: str,
    text_model_id: str | None,
    protein_model_id: str,
    nucleotide_model_id: str,
    smiles_model_id: str,
    property_model_id: str,
    trust_remote_code: bool,
    out_dir: Path,
    model_id: str,
    modalities: str | None,
    encoder: str,
    model_name: str | None,
    embedding_dim: int,
    batch_size: int,
    encode_batch_size: int,
    devices: str,
    chunk_size: int,
    cache_map_size: int,
) -> None:
    core_path = Path(core_dir)
    text_path = Path(text_dataset_dir) if text_dataset_dir is not None else None
    out_path = Path(out_dir)
    source_modalities = _parse_modalities(modalities)
    resolved_encoder = None
    if encoder != "transformers":
        resolved_encoder = _resolve_encoder(
            encoder,
            model_id=model_id,
            model_name=model_name,
            embedding_dim=embedding_dim,
            encode_batch_size=encode_batch_size,
        )

    try:
        if any(
            value is not None
            for value in (
                features_dir,
                he_text_dir,
                cache_dir,
                targets,
                node_input_kinds,
            )
        ):
            feature_path = Path(features_dir) if features_dir is not None else text_path
            requested_kinds = _parse_csv_set(node_input_kinds) or frozenset({"text"})
            requested_targets = _parse_csv_set(targets) or frozenset({"node"})
            required_families = _required_encoder_families(
                targets=requested_targets,
                node_input_kinds=requested_kinds,
            )
            pool: ParallelEncoderPool | None = None
            if encoder == "stub":
                text_id = text_model_id or model_id
                encoder_map = {}
                if "biomedical_text" in required_families:
                    encoder_map["biomedical_text"] = StubTextEncoder(
                        embedding_dim=embedding_dim,
                        model_id=text_id,
                    )
                if "protein_sequence" in required_families:
                    encoder_map["protein_sequence"] = StubTextEncoder(
                        embedding_dim=embedding_dim,
                        model_id=protein_model_id,
                    )
                if "nucleotide_sequence" in required_families:
                    encoder_map["nucleotide_sequence"] = StubTextEncoder(
                        embedding_dim=embedding_dim,
                        model_id=nucleotide_model_id,
                    )
                if "molecular_sequence" in required_families:
                    encoder_map["molecular_sequence"] = StubTextEncoder(
                        embedding_dim=embedding_dim,
                        model_id=smiles_model_id,
                    )
                if "property_numeric" in required_families:
                    encoder_map["property_numeric"] = StubTextEncoder(
                        embedding_dim=embedding_dim,
                        model_id=property_model_id,
                    )
            elif encoder == "sentence-transformers":
                if required_families - {"biomedical_text"}:
                    cli_error(
                        "--encoder sentence-transformers supports text/hyperedge_text only; "
                        "use --encoder transformers for sequence inputs"
                    )
                    raise typer.Exit(2)
                if resolved_encoder is None:
                    cli_error("--encoder sentence-transformers failed to resolve")
                    raise typer.Exit(2)
                encoder_map = {"biomedical_text": resolved_encoder}
            elif encoder == "transformers":
                from primehkg.features.encoders.transformers_pooling_encoder import (
                    TransformersMeanPoolingEncoder,
                )

                text_name = text_model_id or "NeuML/pubmedbert-base-embeddings"
                encoder_map = {}
                family_model_name: dict[str, str] = {}
                if "biomedical_text" in required_families:
                    encoder_map["biomedical_text"] = TransformersMeanPoolingEncoder(
                        model_name=text_name,
                        model_id=text_name,
                        encode_batch_size=encode_batch_size,
                        trust_remote_code=trust_remote_code,
                    )
                    family_model_name["biomedical_text"] = text_name
                if "protein_sequence" in required_families:
                    encoder_map["protein_sequence"] = TransformersMeanPoolingEncoder(
                        model_name=protein_model_id,
                        model_id=protein_model_id,
                        encode_batch_size=encode_batch_size,
                        trust_remote_code=trust_remote_code,
                    )
                    family_model_name["protein_sequence"] = protein_model_id
                if "nucleotide_sequence" in required_families:
                    encoder_map["nucleotide_sequence"] = TransformersMeanPoolingEncoder(
                        model_name=nucleotide_model_id,
                        model_id=nucleotide_model_id,
                        encode_batch_size=encode_batch_size,
                        trust_remote_code=trust_remote_code,
                    )
                    family_model_name["nucleotide_sequence"] = nucleotide_model_id
                if "molecular_sequence" in required_families:
                    encoder_map["molecular_sequence"] = TransformersMeanPoolingEncoder(
                        model_name=smiles_model_id,
                        model_id=smiles_model_id,
                        encode_batch_size=encode_batch_size,
                        trust_remote_code=trust_remote_code,
                    )
                    family_model_name["molecular_sequence"] = smiles_model_id
                if "property_numeric" in required_families:
                    property_name = property_model_id
                    if property_name == "primehkg-property-normalizer:v1":
                        property_name = text_name
                    if property_name == text_name and "biomedical_text" in encoder_map:
                        encoder_map["property_numeric"] = encoder_map["biomedical_text"]
                    else:
                        encoder_map["property_numeric"] = (
                            TransformersMeanPoolingEncoder(
                                model_name=property_name,
                                model_id=property_name,
                                encode_batch_size=encode_batch_size,
                                trust_remote_code=trust_remote_code,
                            )
                        )
                    family_model_name["property_numeric"] = property_name
                resolved_devices = _resolve_devices(devices)
                # Spawn the GPU worker pool only when real CUDA devices are
                # present; the CPU fallback (["cpu"]) keeps pool=None and encodes
                # in-process via encoder_map (identical results, no model reload
                # in spawned workers).
                if any(d != "cpu" for d in resolved_devices):
                    specs = [
                        EncoderSpec(
                            family=fam,
                            model_name=family_model_name[fam],
                            model_id=enc.model_id,
                            encode_batch_size=encode_batch_size,
                            trust_remote_code=trust_remote_code,
                        )
                        for fam, enc in encoder_map.items()
                    ]
                    pool = ParallelEncoderPool(
                        devices=resolved_devices,
                        specs=specs,
                        encoder_kind="transformers",
                    )
            else:
                cli_error(f"unsupported --encoder {encoder!r}")
                raise typer.Exit(2)
            try:
                build_and_write_multimodal_embeddings(
                    core_dir=core_path,
                    features_dir=feature_path,
                    he_text_dir=Path(he_text_dir) if he_text_dir is not None else None,
                    out_dir=out_path,
                    cache_dir=Path(cache_dir) if cache_dir is not None else None,
                    encoders=cast("dict[str, TextEncoder]", encoder_map),
                    node_input_kinds=requested_kinds,
                    targets=requested_targets,
                    preprocess_version=preprocess_version,
                    encoder_revision=encoder_revision,
                    chunk_size=chunk_size,
                    cache_map_size=cache_map_size,
                    pool=pool,
                )
            finally:
                if pool is not None:
                    pool.close()
        else:
            if resolved_encoder is None:
                cli_error(
                    "--encoder transformers requires multimodal flags such as --targets"
                )
                raise typer.Exit(2)
            build_and_write_embedding_dataset(
                core_dir=core_path,
                text_dataset_dir=text_path,
                encoder=resolved_encoder,
                out_dir=out_path,
                source_modalities=source_modalities,
                batch_size=batch_size,
            )
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        cli_error(str(exc))
        raise typer.Exit(2) from exc

    raise typer.Exit(0)


@app.command()
def build(
    core_dir: Path = CORE_DIR_OPT,
    text_dataset_dir: Path | None = TEXT_DATASET_DIR_OPT,
    features_dir: Path | None = FEATURES_DIR_OPT,
    he_text_dir: Path | None = HE_TEXT_DIR_OPT,
    cache_dir: Path | None = CACHE_DIR_OPT,
    targets: str | None = TARGETS_OPT,
    node_input_kinds: str | None = NODE_INPUT_KINDS_OPT,
    preprocess_version: str = PREPROCESS_VERSION_OPT,
    encoder_revision: str = ENCODER_REVISION_OPT,
    text_model_id: str | None = TEXT_MODEL_ID_OPT,
    protein_model_id: str = PROTEIN_MODEL_ID_OPT,
    nucleotide_model_id: str = NUCLEOTIDE_MODEL_ID_OPT,
    smiles_model_id: str = SMILES_MODEL_ID_OPT,
    property_model_id: str = PROPERTY_MODEL_ID_OPT,
    trust_remote_code: bool = TRUST_REMOTE_CODE_OPT,
    out_dir: Path = OUT_DIR_OPT,
    model_id: str = MODEL_ID_OPT,
    modalities: str | None = MODALITIES_OPT,
    encoder: str = ENCODER_OPT,
    model_name: str | None = MODEL_NAME_OPT,
    embedding_dim: int = EMBEDDING_DIM_OPT,
    batch_size: int = BATCH_SIZE_OPT,
    encode_batch_size: int = ENCODE_BATCH_SIZE_OPT,
    devices: str = DEVICES_OPT,
    chunk_size: int = CHUNK_SIZE_OPT,
    cache_map_size: int = CACHE_MAP_SIZE_OPT,
) -> None:
    _run_build(
        core_dir=core_dir,
        text_dataset_dir=text_dataset_dir,
        features_dir=features_dir,
        he_text_dir=he_text_dir,
        cache_dir=cache_dir,
        targets=targets,
        node_input_kinds=node_input_kinds,
        preprocess_version=preprocess_version,
        encoder_revision=encoder_revision,
        text_model_id=text_model_id,
        protein_model_id=protein_model_id,
        nucleotide_model_id=nucleotide_model_id,
        smiles_model_id=smiles_model_id,
        property_model_id=property_model_id,
        trust_remote_code=trust_remote_code,
        out_dir=out_dir,
        model_id=model_id,
        modalities=modalities,
        encoder=encoder,
        model_name=model_name,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        encode_batch_size=encode_batch_size,
        devices=devices,
        chunk_size=chunk_size,
        cache_map_size=cache_map_size,
    )


def _argv_for_embed_build_cli(argv: list[str] | None) -> list[str]:
    a = list(argv) if argv is not None else list(sys.argv[1:])
    if a and a[0] == "build":
        return a[1:]
    return a


def main(argv: list[str] | None = None) -> int:
    cmd = get_command(app)
    args_list = _argv_for_embed_build_cli(argv)
    try:
        rc = cmd.main(
            args=args_list,
            prog_name="primehkg-embed-build",
            standalone_mode=False,
        )
    except SystemExit as e:
        if isinstance(e.code, int):
            return e.code
        return 1
    return int(rc) if rc is not None else 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
