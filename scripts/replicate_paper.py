"""Run the paper decoding recipe and corrected evaluation on supplied inputs."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/replication/inference.yml")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--encoder", default="")
    parser.add_argument(
        "--decoder", default="", help="Local DOC run; omit for transcription only."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Scoring smoke-test limit per dataset; 0 is full evaluation.",
    )
    args = parser.parse_args()
    if not args.decoder and not args.encoder:
        parser.error(
            "Pass --encoder for decoder-free transcription, or --decoder for all tasks"
        )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    scripts = Path(__file__).resolve().parent
    predictions = output / "predictions"
    run = predictions / "final_normm16_p095_fullamp_ae5"

    def call(name, *options):
        print(f"Running {name}", flush=True)
        subprocess.run(
            [sys.executable, str(scripts / name), *map(str, options)], check=True
        )

    codec = (
        [
            "--codec_run_dir",
            args.decoder,
            "--codec_checkpoint",
            "best",
            "--codec_use_ema",
        ]
        if args.decoder
        else ["--codec_encoder", args.encoder]
    )
    call(
        "run_dcomposer_directory_inference.py",
        "--config",
        args.config,
        "--input_dir",
        args.inputs,
        "--output_dir",
        run / "tokens",
        "--dcomposer_run_dir",
        args.model,
        *codec,
    )
    if args.decoder:
        call(
            "render_dcomposer_token_directory.py",
            "--input_dir",
            run / "tokens",
            "--oneshots_output_dir",
            run / "oneshots",
            "--roundtrip_output_dir",
            run / "roundtrip",
            "--dcomposer_run_dir",
            args.model,
            "--codec_run_dir",
            args.decoder,
            "--codec_checkpoint",
            "best",
            "--codec_use_ema",
            "--device",
            "cuda",
            "--batch_size",
            32,
            "--codec_render_batch_size",
            16,
            "--autoencoder_n_steps",
            5,
            "--amp_render",
            "--oneshot_datasets",
            "synthetic",
            "--seed",
            0,
        )
    call(
        "export_dcomposer_token_transcripts.py",
        "--input_dir",
        run / "tokens",
        "--output_dir",
        run / "transcripts",
        "--dcomposer_run_dir",
        args.model,
    )
    call(
        "evaluate_decompositions.py",
        "--inputs",
        args.inputs,
        "--predictions",
        predictions,
        "--output",
        output / "scores",
        "--methods",
        "coarse",
        "--tasks",
        "transcription,oneshot,roundtrip" if args.decoder else "transcription",
        "--device",
        "cuda" if args.decoder else "cpu",
        "--batch_size",
        32,
        "--limit",
        args.limit,
    )


if __name__ == "__main__":
    main()
