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
from audiotools import AudioSignal

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts import train_dcomposer
from dcomposer.constants import COARSE_MIDI_NOTE_TO_COARSE_LABEL
from dcomposer.constants import FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE
from dcomposer.constants import FINE_MIDI_NOTE_TO_FINE_LABEL
from dcomposer.util import load_config
from dcomposer.retrieval import digest
from dcomposer.windows import window_events, event_row


def log(message: str):
    print(message, file=sys.stderr, flush=True)


def synchronize_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--oneshots_output_dir", type=str, required=True)
    p.add_argument("--roundtrip_output_dir", type=str, required=True)
    p.add_argument("--dcomposer_run_dir", type=str, required=True)
    p.add_argument("--codec_run_dir", type=str, default=None)
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
    p.add_argument("--codec_render_batch_size", type=int, default=32)
    p.add_argument("--oneshot_duration_s", type=float, default=3.0)
    p.add_argument("--roundtrip_duration_s", type=float, default=None)
    p.add_argument("--autoencoder_n_steps", type=int, default=5)
    p.add_argument("--autoencoder_cfg_weight", type=float, default=None)
    p.add_argument("--amp_render", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--oneshot_datasets",
        type=str,
        default=None,
        help="Comma-separated top-level dataset names for which to save one-shot WAVs. Defaults to all datasets.",
    )
    return p.parse_args()


def discover_token_files(root: Path, max_files: Optional[int] = None) -> List[Path]:
    files = [
        p
        for p in sorted(root.rglob("*.pt"))
        if p.is_file() and p.name != "_run_metadata.pt"
    ]
    files = [p for p in files if p.name != "_run_metadata.json"]
    if max_files is not None:
        files = files[: int(max_files)]
    return files


def resolve_codec_cfg(run_args: Dict, cli) -> Dict:
    with argbind.scope(run_args):
        codec_cfg = train_dcomposer.build_codec_config()
    if cli.codec_run_dir is not None:
        codec_cfg["run_dir"] = str(Path(cli.codec_run_dir).expanduser())
    if cli.codec_checkpoint is not None:
        codec_cfg["checkpoint"] = str(cli.codec_checkpoint)
    if cli.codec_use_ema is not None:
        codec_cfg["use_ema"] = bool(cli.codec_use_ema)
    return codec_cfg


def fine_to_coarse(fine_note: int) -> int:
    return int(FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE.get(int(fine_note), int(fine_note)))


def oneshot_filename(event_idx: int, fine_note: int) -> str:
    fine_note = int(fine_note)
    coarse_note = fine_to_coarse(fine_note)
    fine_label = FINE_MIDI_NOTE_TO_FINE_LABEL.get(fine_note, "unknown")
    coarse_label = COARSE_MIDI_NOTE_TO_COARSE_LABEL.get(coarse_note, "unknown")
    return (
        f"{event_idx:04d}_fine{fine_note}_{fine_label}"
        f"__coarse{coarse_note}_{coarse_label}.wav"
    )


def load_token_payload(path: Path) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not (
        "tokens" in payload or "windows" in payload
    ):
        raise ValueError(f"Invalid token payload in {path}")
    return payload


def get_relative_output_key(token_path: Path, input_dir: Path, payload: Dict) -> Path:
    rel_str = payload.get("relative_path", None)
    if rel_str not in [None, ""]:
        return Path(rel_str)
    return token_path.relative_to(input_dir)


def roundtrip_output_path(
    roundtrip_output_dir: Path,
    rel_key: Path,
) -> Path:
    return roundtrip_output_dir / rel_key.with_suffix(".wav")


def oneshot_output_dir(
    oneshots_output_dir: Path,
    rel_key: Path,
) -> Path:
    return oneshots_output_dir / rel_key.with_suffix("")


def parse_oneshot_dataset_filter(raw: Optional[str]) -> Optional[set]:
    if raw in [None, ""]:
        return None
    vals = {part.strip() for part in str(raw).split(",") if part.strip()}
    return vals or None


def should_export_oneshots(rel_key: Path, dataset_filter: Optional[set]) -> bool:
    if dataset_filter is None:
        return True
    parts = rel_key.parts
    if not parts:
        return False
    return parts[0] in dataset_filter


def trim_tokens(payload: Dict) -> torch.Tensor:
    tokens = payload["tokens"]
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens)
    length = int(payload.get("length", tokens.shape[0]))
    return tokens[:length].to(dtype=torch.long, device="cpu")


def write_signal(sig: AudioSignal, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sig.cpu()
    if out.audio_data.dtype not in (
        torch.float32,
        torch.float64,
        torch.int16,
        torch.int32,
    ):
        out.audio_data = out.audio_data.to(torch.float32)
    out.write(path)


def render_batch(
    batch_paths: List[Path],
    *,
    input_dir: Path,
    oneshots_output_dir_root: Path,
    roundtrip_output_dir_root: Path,
    n_acoustic_tokens: int,
    codec_bundle: Dict,
    device: torch.device,
    oneshot_duration_s: float,
    roundtrip_duration_s: float,
    autoencoder_n_steps: int,
    autoencoder_cfg_weight: Optional[float],
    codec_render_batch_size: int,
    amp_render: bool,
    oneshot_dataset_filter: Optional[set],
    preserve_duration: bool = False,
):
    payloads = [load_token_payload(p) for p in batch_paths]
    parsed = []
    for payload in payloads:
        if "windows" in payload:
            row = event_row(
                window_events(
                    payload,
                    n_acoustic_tokens,
                    payload["acoustic_vocab"],
                )
            )
        else:
            row = train_dcomposer.parse_dcomposer_tokens(
                trim_tokens(payload).unsqueeze(0),
                n_acoustic_tokens=n_acoustic_tokens,
            )[0]
        parsed.append(row)

    sample_rate = int(codec_bundle["tokenizer"].sample_rate)
    n_channels = int(codec_bundle["tokenizer"].n_channels)
    oneshot_num_samples = int(round(float(oneshot_duration_s) * sample_rate))
    lengths = [
        round(
            float(
                payload["duration_s"]
                if preserve_duration and "windows" in payload
                else roundtrip_duration_s
            )
            * sample_rate
        )
        for payload in payloads
    ]
    roundtrip_num_samples = max(lengths)

    mix = torch.zeros(
        len(batch_paths),
        n_channels,
        roundtrip_num_samples,
        device=device,
        dtype=torch.float32,
    )

    flat_codes = []
    owners = []
    onsets = []
    notes = []
    rel_keys = []
    for row_idx, row in enumerate(parsed):
        rel_key = get_relative_output_key(
            batch_paths[row_idx], input_dir, payloads[row_idx]
        )
        rel_keys.append(rel_key)
        if should_export_oneshots(rel_key, oneshot_dataset_filter):
            oneshot_output_dir(oneshots_output_dir_root, rel_key).mkdir(
                parents=True, exist_ok=True
            )
        for event_idx, (fine_note, onset_sec, code) in enumerate(
            zip(row["notes"], row["onsets_sec"], row["codes"])
        ):
            flat_codes.append(code.to(device=device, dtype=torch.long))
            owners.append((row_idx, event_idx))
            onsets.append(float(onset_sec))
            notes.append(int(fine_note))

    render_batch_size = max(int(codec_render_batch_size), 1)
    total_decoded = 0
    for start_idx in range(0, len(flat_codes), render_batch_size):
        end_idx = min(start_idx + render_batch_size, len(flat_codes))
        code_batch = torch.stack(flat_codes[start_idx:end_idx], dim=0)
        synchronize_if_needed(device)
        with autocast_context(device, enabled=bool(amp_render)):
            decoded = train_dcomposer.decode_dcomposer_oneshots(
                code_batch,
                codec_bundle=codec_bundle,
                n_samples=oneshot_num_samples,
                n_channels=n_channels,
                sample_rate=sample_rate,
                device=device,
                n_steps=int(autoencoder_n_steps),
                cfg_weight=autoencoder_cfg_weight,
            )
        decoded = decoded.to(device)
        synchronize_if_needed(device)
        for local_i, ((row_idx, event_idx), onset_sec, fine_note) in enumerate(
            zip(
                owners[start_idx:end_idx],
                onsets[start_idx:end_idx],
                notes[start_idx:end_idx],
            )
        ):
            clip = decoded[local_i : local_i + 1]
            if should_export_oneshots(rel_keys[row_idx], oneshot_dataset_filter):
                shot_dir = oneshot_output_dir(
                    oneshots_output_dir_root, rel_keys[row_idx]
                )
                shot_name = oneshot_filename(event_idx=event_idx, fine_note=fine_note)
                write_signal(clip, shot_dir / shot_name)
            total_decoded += 1

            write_start = int(round(float(onset_sec) * sample_rate))
            if write_start >= lengths[row_idx]:
                continue
            clip_audio = clip.audio_data[
                ..., : max(0, lengths[row_idx] - write_start)
            ].to(device=device, dtype=mix.dtype)
            if clip_audio.shape[-1] == 0:
                continue
            mix[
                row_idx : row_idx + 1,
                :,
                write_start : write_start + clip_audio.shape[-1],
            ] += clip_audio

    roundtrip = AudioSignal(mix, sample_rate=sample_rate).ensure_max_of_audio()
    for row_idx, rel_key in enumerate(rel_keys):
        write_signal(
            AudioSignal(
                roundtrip.audio_data[row_idx : row_idx + 1, :, : lengths[row_idx]],
                sample_rate,
            ),
            roundtrip_output_path(roundtrip_output_dir_root, rel_key),
        )

    return {
        "n_inputs": len(batch_paths),
        "n_events_rendered": total_decoded,
    }


def main():
    cli = parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    input_dir = Path(cli.input_dir).expanduser().resolve()
    oneshots_output_dir_root = Path(cli.oneshots_output_dir).expanduser().resolve()
    roundtrip_output_dir_root = Path(cli.roundtrip_output_dir).expanduser().resolve()
    dcomposer_run_dir = Path(cli.dcomposer_run_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    for output in (oneshots_output_dir_root, roundtrip_output_dir_root):
        if output.exists() and any(output.iterdir()) and not cli.overwrite:
            raise FileExistsError(
                f"Refusing to mix renders into existing outputs: {output}"
            )
    if not dcomposer_run_dir.exists():
        raise FileNotFoundError(f"DComposer model not found: {dcomposer_run_dir}")
    if int(cli.batch_size) <= 0:
        raise ValueError(f"`batch_size` must be positive; got {cli.batch_size}")
    if int(cli.codec_render_batch_size) <= 0:
        raise ValueError(
            f"`codec_render_batch_size` must be positive; got {cli.codec_render_batch_size}"
        )
    if float(cli.oneshot_duration_s) <= 0:
        raise ValueError(
            f"`oneshot_duration_s` must be positive; got {cli.oneshot_duration_s}"
        )

    device = torch.device(
        cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_args = load_config(train_dcomposer._config_path(dcomposer_run_dir))
    n_acoustic_tokens = int(run_args["DComposer.n_acoustic_tokens"])
    codec_cfg = resolve_codec_cfg(run_args, cli)
    codec_bundle = train_dcomposer.load_codec_bundle(codec_cfg, device=device)
    weights = "ema_model.pt" if codec_cfg["use_ema"] else "model.pt"
    source = Path(codec_cfg["run_dir"]) / codec_cfg["checkpoint"] / weights
    codec_hash = digest(source)
    if run_args.get("codec_weights_sha256"):
        if codec_hash != run_args["codec_weights_sha256"]:
            raise ValueError("DOC decoder does not match the portable LM release")
    oneshot_dataset_filter = parse_oneshot_dataset_filter(cli.oneshot_datasets)
    roundtrip_duration_s = (
        float(cli.roundtrip_duration_s)
        if cli.roundtrip_duration_s is not None
        else float(run_args.get("train/build_builder_config.max_duration", 4.0))
    )
    if roundtrip_duration_s <= 0:
        raise ValueError(
            f"`roundtrip_duration_s` must be positive; got {roundtrip_duration_s}"
        )

    files = discover_token_files(input_dir, max_files=cli.max_files)
    if not files:
        raise FileNotFoundError(f"No token files found in {input_dir}")

    oneshots_output_dir_root.mkdir(parents=True, exist_ok=True)
    roundtrip_output_dir_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input_dir": str(input_dir),
        "oneshots_output_dir": str(oneshots_output_dir_root),
        "roundtrip_output_dir": str(roundtrip_output_dir_root),
        "dcomposer_run_dir": str(dcomposer_run_dir),
        "codec_run_dir": str(codec_bundle["run_dir"]),
        "codec_checkpoint": str(codec_bundle["checkpoint"]),
        "codec_use_ema": bool(codec_bundle["use_ema"]),
        "codec_weights_sha256": codec_hash,
        "device": str(device),
        "batch_size": int(cli.batch_size),
        "codec_render_batch_size": int(cli.codec_render_batch_size),
        "oneshot_duration_s": float(cli.oneshot_duration_s),
        "roundtrip_duration_s": float(roundtrip_duration_s),
        "preserve_input_duration": cli.roundtrip_duration_s is None,
        "autoencoder_n_steps": int(cli.autoencoder_n_steps),
        "autoencoder_cfg_weight": (
            None
            if cli.autoencoder_cfg_weight is None
            else float(cli.autoencoder_cfg_weight)
        ),
        "amp_render": bool(cli.amp_render),
        "seed": int(cli.seed),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_deterministic": True,
        "oneshot_datasets": (
            None if oneshot_dataset_filter is None else sorted(oneshot_dataset_filter)
        ),
        "n_files": int(len(files)),
    }
    (oneshots_output_dir_root / "_render_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    (roundtrip_output_dir_root / "_render_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    torch.manual_seed(int(cli.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cli.seed))

    log(
        f"Rendering {len(files)} token files "
        f"(batch_size={cli.batch_size}, codec_render_batch_size={cli.codec_render_batch_size})"
    )

    t_start = time.perf_counter()
    render_seconds = 0.0
    processed = 0
    total_events = 0
    bs = max(int(cli.batch_size), 1)
    for batch_idx in range(0, len(files), bs):
        batch_paths = files[batch_idx : batch_idx + bs]
        synchronize_if_needed(device)
        t0 = time.perf_counter()
        stats = render_batch(
            batch_paths,
            input_dir=input_dir,
            oneshots_output_dir_root=oneshots_output_dir_root,
            roundtrip_output_dir_root=roundtrip_output_dir_root,
            n_acoustic_tokens=n_acoustic_tokens,
            codec_bundle=codec_bundle,
            device=device,
            oneshot_duration_s=float(cli.oneshot_duration_s),
            roundtrip_duration_s=float(roundtrip_duration_s),
            preserve_duration=cli.roundtrip_duration_s is None,
            autoencoder_n_steps=int(cli.autoencoder_n_steps),
            autoencoder_cfg_weight=cli.autoencoder_cfg_weight,
            codec_render_batch_size=int(cli.codec_render_batch_size),
            amp_render=bool(cli.amp_render),
            oneshot_dataset_filter=oneshot_dataset_filter,
        )
        synchronize_if_needed(device)
        render_seconds += time.perf_counter() - t0
        processed += stats["n_inputs"]
        total_events += stats["n_events_rendered"]
        elapsed = time.perf_counter() - t_start
        files_per_second = processed / max(elapsed, 1e-8)
        remaining = len(files) - processed
        eta_seconds = remaining / max(files_per_second, 1e-8)
        log(
            f"Batch {batch_idx // bs + 1}: rendered {stats['n_inputs']} files "
            f"({processed}/{len(files)} complete, "
            f"events={stats['n_events_rendered']}, eta={eta_seconds:.1f}s)"
        )

    total_seconds = time.perf_counter() - t_start
    log("Done.")
    log(
        "Timing summary: "
        f"total={total_seconds:.2f}s, "
        f"render={render_seconds:.2f}s"
    )
    log(
        "Throughput summary: "
        f"files={processed}, "
        f"events_rendered={total_events}, "
        f"files_per_second={processed / max(total_seconds, 1e-8):.3f}, "
        f"seconds_per_file={total_seconds / max(processed, 1):.3f}"
    )


if __name__ == "__main__":
    main()
