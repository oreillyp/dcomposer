#!/usr/bin/env python3
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

import argbind
import torch
import yaml
from audiotools import AudioSignal

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts import train_dcomposer
from dcomposer.constants import ACOUSTIC_TOK_OFFSET, MAX_DURATION
from dcomposer.retrieval import digest, load_tokenizer, parse_events
from dcomposer.inference import load_model as instantiate_trained_dcomposer
from dcomposer.inference import preprocess_signal_for_batch
from dcomposer.windows import windows


def log(message: str):
    print(message, file=sys.stderr, flush=True)


def synchronize_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def set_use_sdpa(module, enabled: bool):
    for child in module.modules():
        if hasattr(child, "use_sdpa"):
            child.use_sdpa = bool(enabled)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="")
    p.add_argument(
        "--legacy_key_positions",
        action="store_true",
        help="Reproduce historical cached-key positional encoding.",
    )
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument(
        "--input_layout",
        choices=("directory", "evaluation"),
        default="directory",
        help="Read all audio, or only dataset/audio/ files in an evaluation bank.",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--dcomposer_run_dir",
        type=str,
        required=True,
        help="Training run directory or portable dcomposer.pt release file.",
    )
    p.add_argument("--dcomposer_checkpoint", type=str, default="latest")
    p.add_argument("--codec_run_dir", type=str, default=None)
    p.add_argument(
        "--codec_encoder",
        type=str,
        default=None,
        help="Encoder-only DOC export for decoder-free transcription.",
    )
    p.add_argument(
        "--tokenizer_weights",
        type=str,
        default="",
        help="Public base VAE weights when using --codec_encoder.",
    )
    p.add_argument("--codec_checkpoint", type=str, default=None)
    codec_ema_group = p.add_mutually_exclusive_group()
    codec_ema_group.add_argument(
        "--codec_use_ema",
        dest="codec_use_ema",
        action="store_true",
    )
    codec_ema_group.add_argument(
        "--codec_no_ema",
        dest="codec_use_ema",
        action="store_false",
    )
    p.set_defaults(codec_use_ema=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--extensions",
        type=str,
        default=".wav,.flac,.ogg,.aif,.aiff",
        help="Comma-separated audio suffixes to process.",
    )
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--duration_s",
        type=float,
        default=None,
        help="Pad/trim every input to this duration before encoding. "
        "Defaults to the run config duration, then 4.0s.",
    )
    p.add_argument(
        "--normalize_db",
        type=float,
        default=None,
        help="If set, normalize each input clip to this RMS/dBFS-style target before encoding.",
    )
    p.add_argument(
        "--normalize_lufs",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--normalize_ensure_max",
        action="store_true",
        help="After loudness normalization, scale down any clip whose absolute peak exceeds 1.0.",
    )
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p_cls", type=float, default=None)
    p.add_argument("--top_k_cls", type=int, default=None)
    p.add_argument("--top_p_ons", type=float, default=None)
    p.add_argument("--top_k_ons", type=int, default=None)
    p.add_argument("--top_p_acoustic", type=float, default=None)
    p.add_argument("--top_k_acoustic", type=int, default=None)
    p.add_argument("--temp", type=float, default=None)
    argmax_group = p.add_mutually_exclusive_group()
    argmax_group.add_argument("--argmax", dest="argmax", action="store_true")
    argmax_group.add_argument("--no_argmax", dest="argmax", action="store_false")
    p.set_defaults(argmax=None)
    p.add_argument("--eos_threshold", type=float, default=None)
    p.add_argument(
        "--enable_sdpa",
        action="store_true",
        help="Enable SDPA/fused attention. Off by default because the trusted "
        "equivalence path currently uses SDPA disabled.",
    )
    p.add_argument("--amp_encode", action="store_true")
    p.add_argument("--amp_generate", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window_overlap", type=float, default=1.0)
    for name in ("monotonic_onsets", "complete_events", "sliding_window"):
        group = p.add_mutually_exclusive_group()
        group.add_argument(f"--{name}", dest=name, action="store_true")
        group.add_argument(f"--no_{name}", dest=name, action="store_false")
        p.set_defaults(**{name: True})
    p.add_argument("--max_events", type=int, default=0)
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default="")
    config_path = preliminary.parse_known_args()[0].config
    if config_path:
        config = yaml.safe_load(Path(config_path).read_text())
        if not isinstance(config, dict):
            p.error("Inference config must be a mapping")
        allowed = {a.dest for a in p._actions} - {"help", "config"}
        if set(config) - allowed:
            p.error(f"Unknown inference settings: {sorted(set(config) - allowed)}")
        p.set_defaults(**config)
        for action in p._actions:
            if action.dest in config:
                action.required = False
    args = p.parse_args()
    if args.max_events < 0:
        p.error("--max_events must be nonnegative (0 uses the token budget)")
    return args


def discover_audio_files(
    input_dir: Path,
    extensions: Iterable[str],
    max_files: Optional[int] = None,
    input_layout: str = "directory",
) -> List[Path]:
    if input_layout not in ("directory", "evaluation"):
        raise ValueError(f"Unknown input layout: {input_layout}")
    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    files = [
        p
        for p in sorted(input_dir.rglob("*"))
        if (
            p.is_file()
            and p.suffix.lower() in ext_set
            and (
                input_layout == "directory"
                or (
                    len(p.relative_to(input_dir).parts) >= 3
                    and p.relative_to(input_dir).parts[1] == "audio"
                )
            )
        )
    ]
    if max_files is not None:
        files = files[: int(max_files)]
    return files


def infer_default_duration_s(args: Dict) -> float:
    for key in [
        "train/build_builder_config.max_duration",
        "val/build_builder_config.max_duration",
    ]:
        value = args.get(key, None)
        if value not in [None, ""]:
            return float(value)
    return float(MAX_DURATION)


def resolve_codec_cfg(
    run_args: Dict,
    cli,
) -> Dict:
    with argbind.scope(run_args):
        codec_cfg = train_dcomposer.build_codec_config()
    if cli.codec_run_dir is not None:
        codec_cfg["run_dir"] = str(Path(cli.codec_run_dir).expanduser())
    if cli.codec_checkpoint is not None:
        codec_cfg["checkpoint"] = str(cli.codec_checkpoint)
    if cli.codec_use_ema is not None:
        codec_cfg["use_ema"] = bool(cli.codec_use_ema)
    return codec_cfg


def resolve_sampling_cfg(run_args: Dict, cli, model) -> Dict:
    with argbind.scope(run_args):
        sample_cfg = train_dcomposer.build_sample_config()
    max_new_tokens = (
        int(cli.max_new_tokens)
        if cli.max_new_tokens is not None
        else (
            int(sample_cfg["max_new_tokens"])
            if sample_cfg["max_new_tokens"] is not None
            else int(model.decoder.max_len) - 1
        )
    )
    if max_new_tokens <= 0:
        raise ValueError(f"`max_new_tokens` must be positive; got {max_new_tokens}")
    max_supported = int(model.decoder.max_len) - 1
    if max_new_tokens > max_supported:
        raise ValueError(
            f"`max_new_tokens={max_new_tokens}` exceeds decoder capacity "
            f"of {max_supported} new tokens for this run."
        )
    return {
        "max_new_tokens": max_new_tokens,
        "top_p": sample_cfg["top_p"] if cli.top_p is None else float(cli.top_p),
        "top_k": sample_cfg["top_k"] if cli.top_k is None else int(cli.top_k),
        "top_p_cls": None if cli.top_p_cls is None else float(cli.top_p_cls),
        "top_k_cls": None if cli.top_k_cls is None else int(cli.top_k_cls),
        "top_p_ons": None if cli.top_p_ons is None else float(cli.top_p_ons),
        "top_k_ons": None if cli.top_k_ons is None else int(cli.top_k_ons),
        "top_p_acoustic": (
            None if cli.top_p_acoustic is None else float(cli.top_p_acoustic)
        ),
        "top_k_acoustic": (
            None if cli.top_k_acoustic is None else int(cli.top_k_acoustic)
        ),
        "temp": sample_cfg["temp"] if cli.temp is None else float(cli.temp),
        "argmax": bool(sample_cfg["argmax"])
        if cli.argmax is None
        else bool(cli.argmax),
        "eos_threshold": (
            sample_cfg["eos_threshold"]
            if cli.eos_threshold is None
            else float(cli.eos_threshold)
        ),
        "monotonic_onsets": bool(cli.monotonic_onsets),
        "complete_events": bool(cli.complete_events),
        "max_events": int(cli.max_events),
    }


def trim_prediction_row(pred: Dict, row_idx: int) -> torch.Tensor:
    length = int(pred["lengths"][row_idx].item())
    return pred["tokens"][row_idx, :length].detach().to(dtype=torch.int32, device="cpu")


def save_prediction(
    out_path: Path,
    *,
    source_path: Path,
    relative_path: Path,
    pred: Dict,
    row_idx: int,
    sampling_cfg: Dict,
    duration_s: float,
):
    tokens = trim_prediction_row(pred, row_idx=row_idx).contiguous()
    payload = {
        "tokens": tokens,
        "length": int(tokens.shape[0]),
        "source_path": str(source_path),
        "relative_path": str(relative_path),
        "duration_s": float(duration_s),
        "sampling": dict(sampling_cfg),
        "completion_added": bool(pred["completion_added"][row_idx]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def batched(items: List[Path], batch_size: int):
    bs = max(int(batch_size), 1)
    for i in range(0, len(items), bs):
        yield items[i : i + bs]


@torch.no_grad()
def predict_windows(signal, model, tokenizer, sampling_cfg, cli, device, duration_s):
    rate = int(tokenizer.sample_rate)
    plan = windows(signal.signal_length, rate, cli.window_overlap, duration_s)
    size = round(duration_s * rate)
    predictions = []
    for offset in range(0, len(plan), cli.batch_size):
        audio = []
        for start, _, _ in plan[offset : offset + cli.batch_size]:
            chunk = signal.audio_data[..., start : start + size]
            audio.append(torch.nn.functional.pad(chunk, (0, size - chunk.shape[-1])))
        batch = AudioSignal(torch.cat(audio).to(device), rate)
        with autocast_context(device, enabled=cli.amp_encode):
            inputs = train_dcomposer._encode_input_latents(batch, tokenizer)
        with autocast_context(device, enabled=cli.amp_generate):
            pred = model.inference_cached(
                input_latents=inputs["input_latents"],
                input_latent_lengths=inputs["input_latent_lengths"],
                **sampling_cfg,
            )
        for row in range(len(audio)):
            tokens = trim_prediction_row(pred, row)
            parse_events(
                tokens, model.n_acoustic_tokens, model.n_vocab - ACOUSTIC_TOK_OFFSET
            )
            predictions.append(
                {
                    "tokens": tokens,
                    "length": len(tokens),
                    "completion_added": bool(pred["completion_added"][row]),
                }
            )
        log(f"Windows {offset + len(audio)}/{len(plan)} complete")
    return {
        "windows": predictions,
        "window_plan": plan,
        "sample_rate": rate,
        "duration_s": signal.signal_length / rate,
        "acoustic_vocab": model.n_vocab - ACOUSTIC_TOK_OFFSET,
        "sampling": dict(sampling_cfg),
        "window_overlap": cli.window_overlap,
    }


def main():
    cli = parse_args()
    # The imported trainer enables autotuning; choose reproducible inference kernels.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    input_dir = Path(cli.input_dir).expanduser().resolve()
    output_dir = Path(cli.output_dir).expanduser().resolve()
    dcomposer_run_dir = Path(cli.dcomposer_run_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not cli.overwrite:
        raise FileExistsError(
            f"Refusing to mix predictions into existing outputs: {output_dir}"
        )
    if not dcomposer_run_dir.exists():
        raise FileNotFoundError(f"DComposer model not found: {dcomposer_run_dir}")
    if int(cli.batch_size) <= 0:
        raise ValueError(f"`batch_size` must be positive; got {cli.batch_size}")

    device = torch.device(
        cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, run_args, cfg_path = instantiate_trained_dcomposer(
        dcomposer_run_dir,
        checkpoint=str(cli.dcomposer_checkpoint),
        device=device,
    )
    set_use_sdpa(model, enabled=bool(cli.enable_sdpa))
    for child in model.modules():
        if hasattr(child, "legacy_key_positions"):
            child.legacy_key_positions = bool(cli.legacy_key_positions)

    codec_cfg = resolve_codec_cfg(run_args, cli)
    if cli.codec_encoder:
        encoder_bundle = torch.load(
            cli.codec_encoder, map_location="cpu", weights_only=True
        )
        config = encoder_bundle["config"]
        if model.n_acoustic_tokens != config["n_summary"] * (
            config["n_channels_summary"] // config["fsq_group_size"]
        ):
            raise ValueError("D-Composer and DOC encoder codebook counts differ")
        if (
            model.n_vocab
            != train_dcomposer.ACOUSTIC_TOK_OFFSET
            + config["n_fsq"] ** config["fsq_group_size"]
        ):
            raise ValueError("D-Composer and DOC encoder vocabularies differ")
        encoder_hash = digest(cli.codec_encoder)
        if run_args.get("codec_encoder_sha256", encoder_hash) != encoder_hash:
            raise ValueError("DOC encoder does not match the portable LM release")
        tokenizer = load_tokenizer(encoder_bundle, cli.tokenizer_weights, device)
        codec_bundle = {"run_dir": "", "checkpoint": "encoder-only", "use_ema": False}
    else:
        codec_bundle = train_dcomposer.load_codec_bundle(codec_cfg, device=device)
        if run_args.get("codec_weights_sha256"):

            weights = "ema_model.pt" if codec_cfg["use_ema"] else "model.pt"
            source = Path(codec_cfg["run_dir"]) / codec_cfg["checkpoint"] / weights
            if digest(source) != run_args["codec_weights_sha256"]:
                raise ValueError("DOC decoder does not match the portable LM release")
        tokenizer = codec_bundle["tokenizer"]
        encoder_hash = None

    duration_s = (
        infer_default_duration_s(run_args)
        if cli.duration_s is None
        else float(cli.duration_s)
    )
    normalize_db = (
        float(cli.normalize_db)
        if cli.normalize_db is not None
        else (float(cli.normalize_lufs) if cli.normalize_lufs is not None else None)
    )
    if duration_s <= 0:
        raise ValueError(f"`duration_s` must be positive; got {duration_s}")
    if cli.sliding_window and not 0 <= cli.window_overlap < duration_s:
        raise ValueError("Window overlap must be nonnegative and smaller than context")
    sampling_cfg = resolve_sampling_cfg(run_args, cli, model=model)

    files = discover_audio_files(
        input_dir,
        extensions=[x.strip() for x in cli.extensions.split(",") if x.strip()],
        max_files=cli.max_files,
        input_layout=cli.input_layout,
    )
    if not files:
        raise FileNotFoundError(f"No matching audio files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input_dir": str(input_dir),
        "input_layout": cli.input_layout,
        "output_dir": str(output_dir),
        "dcomposer_run_dir": str(dcomposer_run_dir),
        "dcomposer_checkpoint": str(cli.dcomposer_checkpoint),
        "dcomposer_config_path": str(cfg_path),
        "dcomposer_weights_sha256": digest(
            dcomposer_run_dir
            if dcomposer_run_dir.is_file()
            else dcomposer_run_dir / str(cli.dcomposer_checkpoint) / "model.pt"
        ),
        "codec_run_dir": str(codec_bundle["run_dir"]),
        "codec_checkpoint": str(codec_bundle["checkpoint"]),
        "codec_use_ema": bool(codec_bundle["use_ema"]),
        "codec_encoder_sha256": encoder_hash,
        "tokenizer_sample_rate": int(tokenizer.sample_rate),
        "tokenizer_n_channels": int(tokenizer.n_channels),
        "device": str(device),
        "batch_size": int(cli.batch_size),
        "duration_s": float(duration_s),
        "sliding_window": cli.sliding_window,
        "window_overlap": cli.window_overlap,
        "normalize_db": normalize_db,
        "normalize_ensure_max": bool(cli.normalize_ensure_max),
        "enable_sdpa": bool(cli.enable_sdpa),
        "amp_encode": bool(cli.amp_encode),
        "amp_generate": bool(cli.amp_generate),
        "sampling": sampling_cfg,
        "seed": int(cli.seed),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_deterministic": True,
        "n_files": int(len(files)),
    }
    (output_dir / "_run_metadata.json").write_text(json.dumps(metadata, indent=2))

    log(
        f"Processing {len(files)} files from {input_dir} -> {output_dir} "
        f"(batch_size={cli.batch_size}, duration_s={duration_s:.3f}, "
        f"sdpa={'on' if cli.enable_sdpa else 'off'})"
    )

    torch.manual_seed(int(cli.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cli.seed))

    if cli.sliding_window:
        for index, path in enumerate(files, 1):
            log(f"File {index}/{len(files)}: {path}")
            signal = preprocess_signal_for_batch(
                path,
                sample_rate=int(tokenizer.sample_rate),
                n_channels=int(tokenizer.n_channels),
                duration_s=None,
                normalize_db=normalize_db,
                ensure_max=cli.normalize_ensure_max,
            )
            if not torch.isfinite(signal.audio_data).all():
                raise ValueError(f"Non-finite input audio: {path}")
            payload = predict_windows(
                signal,
                model,
                tokenizer,
                sampling_cfg,
                cli,
                device,
                duration_s,
            )
            rel = path.relative_to(input_dir)
            payload.update(source_path=str(path), relative_path=str(rel))
            target = output_dir / rel.with_suffix(".pt")
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, target)
        log(f"Done: {len(files)} recordings")
        return

    t_start = time.perf_counter()
    preprocess_seconds = 0.0
    encode_seconds = 0.0
    generate_seconds = 0.0
    save_seconds = 0.0
    processed = 0
    total_saved_tokens = 0
    total_generated_new_tokens = 0
    for batch_idx, batch_paths in enumerate(batched(files, cli.batch_size), start=1):
        synchronize_if_needed(device)
        t0 = time.perf_counter()
        signals = [
            preprocess_signal_for_batch(
                path,
                sample_rate=int(tokenizer.sample_rate),
                n_channels=int(tokenizer.n_channels),
                duration_s=float(duration_s),
                normalize_db=normalize_db,
                ensure_max=bool(cli.normalize_ensure_max),
            )
            for path in batch_paths
        ]
        signal_batch = AudioSignal.batch(signals, pad_signals=True).to(device)
        synchronize_if_needed(device)
        preprocess_seconds += time.perf_counter() - t0

        synchronize_if_needed(device)
        t1 = time.perf_counter()
        with autocast_context(device, enabled=bool(cli.amp_encode)):
            encoder_inputs = train_dcomposer._encode_input_latents(
                signal_batch, tokenizer
            )
        synchronize_if_needed(device)
        encode_seconds += time.perf_counter() - t1

        synchronize_if_needed(device)
        t2 = time.perf_counter()
        with autocast_context(device, enabled=bool(cli.amp_generate)):
            pred = model.inference_cached(
                input_latents=encoder_inputs["input_latents"],
                input_latent_lengths=encoder_inputs["input_latent_lengths"],
                max_new_tokens=int(sampling_cfg["max_new_tokens"]),
                top_p=sampling_cfg["top_p"],
                top_k=sampling_cfg["top_k"],
                top_p_cls=sampling_cfg["top_p_cls"],
                top_k_cls=sampling_cfg["top_k_cls"],
                top_p_ons=sampling_cfg["top_p_ons"],
                top_k_ons=sampling_cfg["top_k_ons"],
                top_p_acoustic=sampling_cfg["top_p_acoustic"],
                top_k_acoustic=sampling_cfg["top_k_acoustic"],
                temp=float(sampling_cfg["temp"]),
                argmax=bool(sampling_cfg["argmax"]),
                eos_threshold=sampling_cfg["eos_threshold"],
                monotonic_onsets=sampling_cfg["monotonic_onsets"],
                complete_events=sampling_cfg["complete_events"],
                max_events=sampling_cfg["max_events"],
            )
        synchronize_if_needed(device)
        generate_seconds += time.perf_counter() - t2

        synchronize_if_needed(device)
        t3 = time.perf_counter()
        for row_idx, path in enumerate(batch_paths):
            try:
                parse_events(
                    trim_prediction_row(pred, row_idx),
                    model.n_acoustic_tokens,
                    model.n_vocab - ACOUSTIC_TOK_OFFSET,
                )
            except ValueError as error:
                raise ValueError(
                    f"Incomplete or invalid prediction for {path}: {error}. "
                    "No tokens from this batch were saved; check the sampling "
                    "settings and token budget."
                ) from error
        for row_idx, path in enumerate(batch_paths):
            rel = path.relative_to(input_dir)
            out_pth = output_dir / rel.with_suffix(".pt")
            length = int(pred["lengths"][row_idx].item())
            total_saved_tokens += length
            total_generated_new_tokens += max(0, length - 1)
            save_prediction(
                out_pth,
                source_path=path,
                relative_path=rel,
                pred=pred,
                row_idx=row_idx,
                sampling_cfg=sampling_cfg,
                duration_s=float(duration_s),
            )
        synchronize_if_needed(device)
        save_seconds += time.perf_counter() - t3
        processed += len(batch_paths)
        elapsed = time.perf_counter() - t_start
        files_per_second = processed / max(elapsed, 1e-8)
        remaining = len(files) - processed
        eta_seconds = remaining / max(files_per_second, 1e-8)
        log(
            f"Batch {batch_idx}: wrote {len(batch_paths)} files "
            f"({processed}/{len(files)} complete, "
            f"eta={eta_seconds:.1f}s)"
        )

    total_seconds = time.perf_counter() - t_start
    per_file = total_seconds / max(processed, 1)
    files_per_second = processed / max(total_seconds, 1e-8)
    avg_length = total_saved_tokens / max(processed, 1)
    avg_new_tokens = total_generated_new_tokens / max(processed, 1)
    log("Done.")
    log(
        "Timing summary: "
        f"total={total_seconds:.2f}s, "
        f"preprocess={preprocess_seconds:.2f}s, "
        f"encode={encode_seconds:.2f}s, "
        f"generate={generate_seconds:.2f}s, "
        f"save={save_seconds:.2f}s"
    )
    log(
        "Throughput summary: "
        f"files={processed}, "
        f"files_per_second={files_per_second:.3f}, "
        f"seconds_per_file={per_file:.3f}, "
        f"avg_saved_tokens={avg_length:.1f}, "
        f"avg_generated_new_tokens={avg_new_tokens:.1f}"
    )


if __name__ == "__main__":
    main()
