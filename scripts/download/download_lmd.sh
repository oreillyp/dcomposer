#!/bin/bash
set -euo pipefail

################################################################################
# Environment
################################################################################

DL_SCRIPTS_DIR=$(eval dirname "$(readlink -f "$0")")
SCRIPTS_DIR="$(dirname "$DL_SCRIPTS_DIR")"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"
LOCAL_DATA_DIR="${PROJECT_DIR}/data"

mkdir -p "$LOCAL_DATA_DIR"

if [ -L "$LOCAL_DATA_DIR" ]; then
  DATA_DIR=$(readlink "$LOCAL_DATA_DIR")
else
  DATA_DIR="$LOCAL_DATA_DIR"
fi

echo "Using data directory: $DATA_DIR"

################################################################################
# Helpers
################################################################################

dir_nonempty () { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null || true)" ]; }

cleanup_partial () {
  rm -f "$DATA_DIR/lmd_full.tar.gz" || true
}
trap cleanup_partial EXIT

mkdir -p "$DATA_DIR"

################################################################################
# Download / Extract
################################################################################

TARGET_DIR="$DATA_DIR/lmd_full"
ARCHIVE="$DATA_DIR/lmd_full.tar.gz"
URL="http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"

if dir_nonempty "$TARGET_DIR"; then
  echo "Found Lakh MIDI at $TARGET_DIR — skipping."
  exit 0
fi

mkdir -p "$TARGET_DIR"

echo "Downloading Lakh MIDI..."
wget -c --progress=dot:giga -O "$ARCHIVE" "$URL"

echo "Extracting Lakh MIDI..."
tar -xzf "$ARCHIVE" -C "$TARGET_DIR" --strip-components=1
rm -f "$ARCHIVE"

echo "Done."
