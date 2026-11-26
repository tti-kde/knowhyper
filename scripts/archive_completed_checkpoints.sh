#!/usr/bin/env bash
set -euo pipefail

# Copy completed checkpoint files to durable storage, verify every sha256,
# then optionally remove only the exact source files in the verified inventory.
# Logs, Hydra configs, metrics, and run directories remain in place.

usage() {
  echo "usage: $0 --source ABS_RUN_ROOT --destination ABS_GROUP_ROOT [--move]" >&2
  exit 2
}

SOURCE=""
DESTINATION=""
MOVE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --destination) DESTINATION="$2"; shift 2 ;;
    --move) MOVE=1; shift ;;
    *) usage ;;
  esac
done
: "${SOURCE:?--source is required}"
: "${DESTINATION:?--destination is required}"

RUNS="${KNOWHYPER_HOME:-$HOME/.knowhyper}/runs"
case "$SOURCE" in
  "$RUNS/"*) ;;
  *) echo "FATAL source is outside the exact KnowHyper run tree: $SOURCE" >&2; exit 2 ;;
esac
case "$DESTINATION" in
  /*) ;;
  *) echo "FATAL destination must be an absolute path: $DESTINATION" >&2; exit 2 ;;
esac
[ -d "$SOURCE" ] || { echo "FATAL source does not exist: $SOURCE" >&2; exit 1; }
[ "$SOURCE" != "$RUNS" ] || { echo "FATAL refusing the broad runs root" >&2; exit 2; }

WORK=$(mktemp -d)
trap 'rm -rf -- "$WORK"' EXIT
LIST="$WORK/relative_paths.txt"
SOURCE_SHA="$WORK/source.sha256"
DEST_SHA="$WORK/destination.sha256"

(
  cd "$SOURCE"
  find . -type f -name '*.ckpt' -print | LC_ALL=C sort >"$LIST"
)
COUNT=$(wc -l <"$LIST")
[ "$COUNT" -gt 0 ] || { echo "FATAL no checkpoints under $SOURCE" >&2; exit 1; }

(
  cd "$SOURCE"
  while IFS= read -r relative; do sha256sum -- "$relative"; done <"$LIST"
) >"$SOURCE_SHA"

mkdir -p "$DESTINATION"
rsync -a --files-from="$LIST" "$SOURCE/" "$DESTINATION/"

(
  cd "$DESTINATION"
  while IFS= read -r relative; do sha256sum -- "$relative"; done <"$LIST"
) >"$DEST_SHA"
cmp "$SOURCE_SHA" "$DEST_SHA"

BYTES=$(
  cd "$SOURCE"
  while IFS= read -r relative; do stat -c '%s' -- "$relative"; done <"$LIST" |
    awk '{total += $1} END {print total + 0}'
)
cp "$LIST" "$DESTINATION/checkpoint_inventory.txt"
cp "$DEST_SHA" "$DESTINATION/checkpoint_sha256.txt"
{
  echo "source=$SOURCE"
  echo "destination=$DESTINATION"
  echo "checkpoint_count=$COUNT"
  echo "checkpoint_bytes=$BYTES"
  echo "sha256_verified=true"
  echo "source_removed=$MOVE"
} >"$DESTINATION/archive_manifest.txt"

if [ "$MOVE" -eq 1 ]; then
  while IFS= read -r relative; do rm -- "$SOURCE/$relative"; done <"$LIST"
  REMAINING=$(find "$SOURCE" -type f -name '*.ckpt' | wc -l)
  [ "$REMAINING" -eq 0 ] || {
    echo "FATAL $REMAINING source checkpoints remain after exact-file removal" >&2
    exit 1
  }
fi

echo "OK checkpoints=$COUNT bytes=$BYTES verified=true moved=$MOVE"
echo "archive=$DESTINATION"
