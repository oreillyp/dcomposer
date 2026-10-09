#!/usr/bin/env bash
set -euo pipefail

inputs= output= model=
options=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_dir|--output_dir|--dcomposer_run_dir)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                exit 1
            fi
            case "$1" in
                --input_dir) inputs=$2 ;;
                --output_dir) output=$2 ;;
                --dcomposer_run_dir) model=$2 ;;
            esac
            shift 2
            ;;
        *) options+=("$1"); shift ;;
    esac
done
if [[ -z "$inputs" || -z "$output" || -z "$model" ]]; then
    echo "Usage: bash scripts/run_decomposition_pipeline.sh --input_dir INPUTS --output_dir OUTPUTS --dcomposer_run_dir MODEL [inference options...]" >&2
    exit 1
fi
inputs=$(realpath "$inputs")
output=$(realpath -m "$output")
model=$(realpath "$model")
cd "$(dirname "$0")/.."

render_options=()
for ((i=0; i<${#options[@]}; i++)); do
    case "${options[i]}" in
        --codec_run_dir|--codec_checkpoint|--seed|--device)
            render_options+=("${options[i]}" "${options[i+1]}")
            i=$((i+1))
            ;;
        --codec_run_dir=*|--codec_checkpoint=*|--seed=*|--device=*|--codec_use_ema|--codec_no_ema)
            render_options+=("${options[i]}")
            ;;
        --codec_encoder|--codec_encoder=*)
            echo "This pipeline requires a local DOC decoder; use sample_retrieval.py for retrieval." >&2
            exit 1
            ;;
    esac
done

if [[ -e "$output" ]]; then
    echo "Refusing to overwrite $output" >&2
    exit 1
fi

echo "Generating tokens..." >&2
python scripts/run_dcomposer_directory_inference.py \
    --input_dir "$inputs" --output_dir "$output/tokens" \
    --dcomposer_run_dir "$model" --device cuda --batch_size 64 \
    --amp_encode --amp_generate --normalize_ensure_max "${options[@]}"

echo "Rendering one-shots and round trips..." >&2
python scripts/render_dcomposer_token_directory.py \
    --input_dir "$output/tokens" --oneshots_output_dir "$output/oneshots" \
    --roundtrip_output_dir "$output/roundtrip" --dcomposer_run_dir "$model" \
    --device cuda --batch_size 32 --codec_render_batch_size 16 \
    --autoencoder_n_steps 5 --amp_render "${render_options[@]}"

echo "Exporting transcripts..." >&2
python scripts/export_dcomposer_token_transcripts.py \
    --input_dir "$output/tokens" --output_dir "$output/transcripts" \
    --dcomposer_run_dir "$model"
