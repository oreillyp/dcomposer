#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
DATA_DIR="$PROJECT_DIR/data"
TARGET="$DATA_DIR/e-gmd-v1.0.0"
ARCHIVE="$DATA_DIR/e-gmd-v1.0.0.zip"

# Evaluation needs the official test split's paired audio and MIDI, not training audio.
complete() {
    python - "$TARGET" <<'PY'
import csv
import sys
from pathlib import Path
root = Path(sys.argv[1])
metadata = root / 'e-gmd-v1.0.0.csv'
if not metadata.is_file():
    sys.exit(1)
with metadata.open() as stream:
    rows = [row for row in csv.DictReader(stream) if row['split'] == 'test']
sys.exit(0 if rows and all((root / row[key]).is_file() for row in rows
                         for key in ('audio_filename', 'midi_filename')) else 1)
PY
}

if complete; then
    echo "Found complete E-GMD test split at $TARGET; skipping download."
    exit 0
fi

mkdir -p "$DATA_DIR"
echo "Downloading E-GMD: 90 GB archive, 132 GB unpacked; allow space for both."
wget -c --progress=dot:giga -O "$ARCHIVE" \
    https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip
echo "7d9a264fb4c9eabd9fec09d5f8e333192f529b1a1b845d170279a977ac436053  $ARCHIVE" | sha256sum -c -
unzip -q -o "$ARCHIVE" -d "$DATA_DIR"
complete
rm "$ARCHIVE"
echo "E-GMD ready: $TARGET"
