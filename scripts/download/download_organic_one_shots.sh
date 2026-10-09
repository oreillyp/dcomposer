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
  rm -f "$DATA_DIR/organic_oneshots.tar" || true
}
trap cleanup_partial EXIT

mkdir -p "$DATA_DIR"

################################################################################
# Download / Extract
################################################################################

TARGET_DIR="$DATA_DIR/organic_oneshots"
ARCHIVE="$DATA_DIR/organic_oneshots.tar"
URL="https://zenodo.org/records/3994999/files/organic_oneshots.tar"

if dir_nonempty "$TARGET_DIR"; then
  echo "Found Organic One-Shots at $TARGET_DIR — skipping."
  exit 0
fi

mkdir -p "$TARGET_DIR"

echo "Downloading Organic One-Shots..."
wget -c --progress=dot:giga -O "$ARCHIVE" "$URL"

echo "Extracting Organic One-Shots..."
if ! tar -xf "$ARCHIVE" -C "$TARGET_DIR" --strip-components=1; then
  rm -rf "$TARGET_DIR"
  mkdir -p "$TARGET_DIR"
  tar -xf "$ARCHIVE" -C "$TARGET_DIR"
fi

rm -f "$ARCHIVE"
echo "Done."
