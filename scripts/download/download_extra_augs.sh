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
  rm -f "$DATA_DIR/Audio.zip" || true
}
trap cleanup_partial EXIT

mkdir -p "$DATA_DIR"


################################################################################
# RIR / Noise Database (3G)
################################################################################

RIR_REAL="$DATA_DIR/rir-database/real"
RIR_SYN="$DATA_DIR/rir-database/synthetic"
NOISE_ROOM="$DATA_DIR/noise-database/room"
NOISE_POINT="$DATA_DIR/noise-database/pointsource"

if dir_nonempty "$RIR_REAL" && dir_nonempty "$RIR_SYN" && \
   dir_nonempty "$NOISE_ROOM" && dir_nonempty "$NOISE_POINT"; then
  echo "Found RIR/Noise database — skipping."
else
  echo "Downloading RIR/Noise database..."
  mkdir -p "$RIR_REAL" "$RIR_SYN" "$NOISE_ROOM" "$NOISE_POINT"

  wget -O "$DATA_DIR/rirs-noises.zip" \
    https://www.openslr.org/resources/28/rirs_noises.zip
  echo "Unzipping RIR/Noise..."
  unzip -q "$DATA_DIR/rirs-noises.zip" -d "$DATA_DIR/"
  rm -f "$DATA_DIR/rirs-noises.zip"

  # Copy pointsource noises
  if [ -d "$DATA_DIR/RIRS_NOISES/pointsource_noises" ]; then
    cp -a "$DATA_DIR"/RIRS_NOISES/pointsource_noises/. "$NOISE_POINT"
  fi

  # Copy simulated RIRs
  if [ -d "$DATA_DIR/RIRS_NOISES/simulated_rirs" ]; then
    cp -a "$DATA_DIR"/RIRS_NOISES/simulated_rirs/. "$RIR_SYN"
  fi

  # Split real RIRs vs real room noises
  if [ -d "$DATA_DIR/RIRS_NOISES/real_rirs_isotropic_noises" ]; then
    mapfile -t room_noises < <(find "$DATA_DIR/RIRS_NOISES/real_rirs_isotropic_noises" -maxdepth 1 -type f -name '*noise*' 2>/dev/null || true)
    if [ "${#room_noises[@]}" -gt 0 ]; then
      cp -- "${room_noises[@]}" "$NOISE_ROOM"
    fi

    mapfile -t rirs < <(find "$DATA_DIR/RIRS_NOISES/real_rirs_isotropic_noises" -maxdepth 1 -type f ! -name '*noise*' 2>/dev/null || true)
    if [ "${#rirs[@]}" -gt 0 ]; then
      cp -- "${rirs[@]}" "$RIR_REAL"
    fi
  fi

  rm -rf "$DATA_DIR/RIRS_NOISES/"
fi


################################################################################
# MIT IR (~18MB)
################################################################################

MIT_DIR="$DATA_DIR/MIT-IR"
if dir_nonempty "$MIT_DIR"; then
  echo "Found MIT-IR at $MIT_DIR — skipping."
else
  echo "Downloading MIT-IR..."
  wget -c --progress=dot:giga -O "$DATA_DIR/Audio.zip" \
    https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip

  echo "Extracting MIT-IR..."
  mkdir -p "$MIT_DIR"
  unzip -q "$DATA_DIR/Audio.zip" -d "$MIT_DIR"
  rm -f "$DATA_DIR/Audio.zip"
  rm -rf "$MIT_DIR/__MACOSX" || true
fi

echo "Done."
