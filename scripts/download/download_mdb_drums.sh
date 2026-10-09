#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
DATA_DIR="$PROJECT_DIR/data"
TARGET="$DATA_DIR/MDBDrums"

complete() {
    [ -d "$TARGET/MDB Drums/audio/drum_only" ] &&
    [ -d "$TARGET/MDB Drums/annotations/class" ] &&
    [ "$(find "$TARGET/MDB Drums/audio/drum_only" -name '*.wav' -type f | wc -l)" -eq 23 ] &&
    [ "$(find "$TARGET/MDB Drums/annotations/class" -name '*_class.txt' -type f | wc -l)" -eq 23 ]
}

if complete; then
    echo "Found MDB Drums at $TARGET; skipping download."
    exit 0
fi
if [ -e "$TARGET" ]; then
    echo "Incomplete MDB Drums directory: $TARGET; refusing to overwrite it." >&2
    exit 1
fi

mkdir -p "$DATA_DIR"
TEMP=$(mktemp -d "$DATA_DIR/.mdb-download.XXXXXX")
trap 'rm -rf "$TEMP"' EXIT
echo "Downloading MDB Drums (isolated drums and original annotations)..."
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/CarlSouthall/MDBDrums.git "$TEMP/MDBDrums"
git -C "$TEMP/MDBDrums" sparse-checkout set 'MDB Drums/audio/drum_only' 'MDB Drums/annotations'
mv "$TEMP/MDBDrums" "$TARGET"
complete
echo "MDB Drums ready: $TARGET"
