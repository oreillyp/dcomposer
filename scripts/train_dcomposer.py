"""
Training scaffold for DComposer using a frozen codec bundle (pretrained tokenizer +
latent autoencoder) loaded from a run directory.
"""
import os
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Optional

import argbind
import rich
import torch
import torch.distributed as dist
import torch.nn.functional as F
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
from rich import pretty
from rich.traceback import install
from torch.utils.tensorboard import SummaryWriter

pretty.install()
install()


@contextmanager
def chdir(path: str):
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def dump_resolved_config(args, save_path: str, model_name: str):
    cfg = {
        k: v
        for k, v in args.items()
        if k not in ["args.unknown", "args.load", "args.save", "args.debug"]
    }
    cfg["model_name"] = model_name
    argbind.dump_args(cfg, Path(save_path) / "conf.yml")


_path = Path(__file__).parent.parent
sys.path.insert(0, str(_path))

with chdir(_path):
    from dcomposer.constants import DEFAULT_COARSE_LOUDNESS_RANGES
    from dcomposer.constants import ACOUSTIC_TOK_OFFSET
    from dcomposer.constants import EOS_TOK
    from dcomposer.constants import MAX_DURATION
    from dcomposer.constants import MAX_ONSET_TOK
    from dcomposer.constants import MIDI_TIME_RES
    from dcomposer.constants import MIDI_TO_TOK
    from dcomposer.constants import ONSET_TOK_OFFSET
    from dcomposer.data.one_shot_midi_dataset import OneShotMidiDataset
    from dcomposer.data.one_shot_midi_dataset import build_one_shot_midi_batch
    from dcomposer.data.one_shot_midi_dataset import format_dcomposer_targets
    from dcomposer.model import LatentFlowAutoencoder
    from dcomposer.model import DComposer
    from dcomposer.inference import load_codec_bundle, decode_dcomposer_oneshots
    from dcomposer.pipelines.tokenizer import Tokenizer
    from dcomposer.pipelines.tokenizer.tokenizer import TokenSequence
    from dcomposer.nn.lr_schedulers import LinearWarmupCosineDecayLR
    from dcomposer import transforms
    from dcomposer.util import count_parameters, ensure_dir, load_config, print

warnings.filterwarnings("ignore", category=UserWarning)

torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))


def AdamW(
    params,
    lr: float = 0.001,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    amsgrad: bool = False,
    maximize: bool = False,
    use_zero: bool = False,
):
    # Keep a small, argbind-friendly wrapper around torch.optim.AdamW. The
    # `use_zero` arg is retained for callsite/config compatibility and is a no-op
    # with the plain PyTorch optimizer.
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=tuple(betas),
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        maximize=maximize,
    )


AdamW = argbind.bind(AdamW)
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)
DComposer = argbind.bind(DComposer)
OneShotMidiDataset = argbind.bind(OneShotMidiDataset, "train", "val")

filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "NormalizedBaseTransform",
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(
    transforms,
    "train_oneshot",
    "train_mix",
    "val_oneshot",
    "val_mix",
    filter_fn=filter_fn,
)


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train_oneshot", "train_mix", "val_oneshot", "val_mix")
def build_transform(
    prob: float = 1.0,
    names: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    return transforms.Compose(*to_tfm(names), prob=prob)


@argbind.bind("train", "val")
def build_builder_config(
    max_duration: Optional[float] = None,
    p_speed: float = 0.0,
    speed_ratio: tuple = ("uniform", 0.9, 1.1),
    p_timing_jitter: float = 0.0,
    timing_jitter_std: tuple = ("uniform", 0.005, 0.02),
    p_velocity_jitter: float = 0.0,
    velocity_jitter_std: tuple = ("uniform", 1.0, 8.0),
    p_time_shift: float = 0.0,
    time_shift: tuple = ("uniform", 0.0, 0.25),
    p_normalize_loudness: float = 1.0,
):
    return {
        "max_duration": max_duration,
        "p_speed": float(p_speed),
        "speed_ratio": tuple(speed_ratio),
        "p_timing_jitter": float(p_timing_jitter),
        "timing_jitter_std": tuple(timing_jitter_std),
        "p_velocity_jitter": float(p_velocity_jitter),
        "velocity_jitter_std": tuple(velocity_jitter_std),
        "p_time_shift": float(p_time_shift),
        "time_shift": tuple(time_shift),
        "p_normalize_loudness": float(p_normalize_loudness),
        "coarse_loudness_ranges": DEFAULT_COARSE_LOUDNESS_RANGES,
        "velocity_to_gain": lambda v: (v.float() / 127.0).pow(0.75),
    }


@argbind.bind("train", "val")
def build_target_config(
    tokenizer_batch_size: int = 128,
    scale: bool = True,
):
    return {
        "tokenizer_batch_size": int(tokenizer_batch_size),
        "scale": bool(scale),
    }


@argbind.bind(without_prefix=True)
def build_compile_config(
    compile_model: bool = False,
    compile_mode: str = "default",
    compile_backend: str = "inductor",
    compile_dynamic: bool = False,
    compile_fullgraph: bool = False,
):
    return {
        "compile_model": bool(compile_model),
        "compile_mode": compile_mode,
        "compile_backend": compile_backend,
        "compile_dynamic": bool(compile_dynamic),
        "compile_fullgraph": bool(compile_fullgraph),
    }


@argbind.bind(without_prefix=True)
def build_scheduler_config(
    scheduler_name: str = "exponential",
    gamma: float = 1.0,
    warmup_steps: int = 0,
    decay_steps: int = 100_000,
    min_scale: float = 0.0,
):
    assert scheduler_name in ["exponential", "linear_warmup_cosine_decay"]
    return {
        "scheduler_name": scheduler_name,
        "gamma": float(gamma),
        "warmup_steps": int(warmup_steps),
        "decay_steps": int(decay_steps),
        "min_scale": float(min_scale),
    }


@argbind.bind(without_prefix=True)
def build_memory_config(
    log_gpu_memory: bool = True,
):
    return {
        "log_gpu_memory": bool(log_gpu_memory),
    }


@argbind.bind(without_prefix=True)
def build_codec_config(
    codec_run_dir: str = None,
    codec_checkpoint: str = "latest",
    codec_use_ema: bool = True,
):
    return {
        "run_dir": codec_run_dir,
        "checkpoint": codec_checkpoint,
        "use_ema": bool(codec_use_ema),
    }


@argbind.bind(without_prefix=True)
def build_sample_config(
    top_p: float = None,
    top_k: int = None,
    temp: float = 1.0,
    argmax: bool = False,
    eos_threshold: float = None,
    max_new_tokens: int = None,
    render_max_duration: float = MAX_DURATION,
    autoencoder_n_steps: int = 1,
    autoencoder_cfg_weight: float = None,
):
    return {
        "top_p": None if top_p is None else float(top_p),
        "top_k": None if top_k is None else int(top_k),
        "temp": float(temp),
        "argmax": bool(argmax),
        "eos_threshold": None if eos_threshold is None else float(eos_threshold),
        "max_new_tokens": None if max_new_tokens is None else int(max_new_tokens),
        "render_max_duration": float(render_max_duration),
        "autoencoder_n_steps": int(autoencoder_n_steps),
        "autoencoder_cfg_weight": (
            None if autoencoder_cfg_weight is None else float(autoencoder_cfg_weight)
        ),
    }


def _prefix_kwargs(args: Dict, prefix: str) -> Dict:
    out = {}
    needle = f"{prefix}."
    for key, value in args.items():
        if key.startswith(needle):
            out[key[len(needle) :]] = value
    return out


def _config_path(run_dir: Path) -> Path:
    if run_dir.is_file() and run_dir.suffix == ".pt":
        return run_dir
    for name in ["conf.yml", "flow.yml"]:
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No copied config found in {run_dir}; expected conf.yml or flow.yml"
    )


@torch.no_grad()
@torch.no_grad()
def assert_codec_matches_dataset_duration(dataset, codec_bundle, device):
    tokenizer = codec_bundle["tokenizer"]
    autoencoder = codec_bundle["autoencoder"]
    n_samples = int(round(float(dataset.duration) * int(dataset.sample_rate)))
    signal = AudioSignal(
        torch.zeros(1, int(dataset.num_channels), n_samples, device=device),
        sample_rate=int(dataset.sample_rate),
    )
    seq = tokenizer.encode(signal, scale=False, no_grad=True)
    seq_len = int(seq.tokens.shape[-1])
    max_len = int(autoencoder.max_len)
    if seq_len > max_len:
        raise ValueError(
            "Codec/dataset length mismatch: "
            f"dataset.duration={float(dataset.duration):.3f}s produced latent length {seq_len}, "
            f"but codec autoencoder max_len={max_len}."
        )
    if seq_len < max_len:
        print(
            "Warning: dataset duration does not use full codec context: "
            f"latent length {seq_len} < max_len {max_len}."
        )


def build_scheduler(optimizer, cfg: Dict):
    if cfg["scheduler_name"] == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg["gamma"])
    return LinearWarmupCosineDecayLR(
        optimizer,
        warmup_steps=cfg["warmup_steps"],
        decay_steps=cfg["decay_steps"],
        min_scale=cfg["min_scale"],
    )


def _assert_valid_audio(signal, name: str):
    x = signal.audio_data
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name}: non-finite audio detected")
    if x.abs().amax().item() > 1.0 + 1e-6:
        raise RuntimeError(f"{name}: audio exceeded valid range [-1, 1]")


def _assert_valid_tokens(tokens: torch.Tensor, n_vocab: int):
    if tokens.dtype != torch.long:
        raise RuntimeError("target tokens must be torch.long")
    if tokens.numel() == 0:
        raise RuntimeError("empty target token tensor")
    if int(tokens.amin().item()) < 0 or int(tokens.amax().item()) >= int(n_vocab):
        raise RuntimeError(f"target tokens out of range for vocabulary size {n_vocab}")


def _effective_max_signal_duration(dataset, builder_cfg: Dict) -> float:
    if builder_cfg.get("max_duration", None) is not None:
        return float(builder_cfg["max_duration"])
    return float(dataset.max_midi_duration) + float(dataset.duration)


@torch.no_grad()
def assert_dcomposer_compatibility(
    dataset, builder_cfg: Dict, codec_bundle: Dict, model
):
    tokenizer = codec_bundle["tokenizer"]
    autoencoder = codec_bundle["autoencoder"]
    device = next(tokenizer.parameters()).device

    if int(model.n_acoustic_tokens) != int(autoencoder.n_codebooks):
        raise ValueError(
            "DComposer/codec mismatch: "
            f"model.n_acoustic_tokens={int(model.n_acoustic_tokens)} "
            f"!= autoencoder.n_codebooks={int(autoencoder.n_codebooks)}"
        )

    if int(model.n_vocab) != int(ACOUSTIC_TOK_OFFSET + autoencoder.n_vocab):
        raise ValueError(
            "DComposer/vocab mismatch: "
            f"model.n_vocab={int(model.n_vocab)} "
            f"!= expected {int(ACOUSTIC_TOK_OFFSET + autoencoder.n_vocab)}"
        )

    if int(model.n_latent_channels) != int(tokenizer.latent_dim):
        raise ValueError(
            "DComposer/tokenizer mismatch: "
            f"model.n_latent_channels={int(model.n_latent_channels)} "
            f"!= tokenizer.latent_dim={int(tokenizer.latent_dim)}"
        )

    max_signal_duration = _effective_max_signal_duration(dataset, builder_cfg)
    if max_signal_duration > float(MAX_DURATION):
        raise ValueError(
            "DComposer target/onset configuration mismatch: "
            f"effective max signal duration {max_signal_duration:.3f}s exceeds "
            f"MAX_DURATION={float(MAX_DURATION):.3f}s"
        )

    if model.use_encoder:
        n_samples = int(round(max_signal_duration * int(dataset.sample_rate)))
        signal = AudioSignal(
            torch.zeros(1, int(dataset.num_channels), n_samples, device=device),
            sample_rate=int(dataset.sample_rate),
        )
        seq = tokenizer.encode(signal, scale=True, no_grad=True)
        seq_len = int(seq.tokens.shape[-1])
        if seq_len > int(model.encoder.max_len):
            raise ValueError(
                "DComposer encoder/context mismatch: "
                f"signal duration {max_signal_duration:.3f}s produces latent length {seq_len}, "
                f"but model max_len_encoder={int(model.encoder.max_len)}"
            )

    max_target_len = (
        1 + int(dataset.max_midi_events) * (2 + int(autoencoder.n_codebooks)) + 1
    )
    if max_target_len > int(model.decoder.max_len):
        raise ValueError(
            "DComposer decoder/context mismatch: "
            f"max target length upper bound {max_target_len} exceeds "
            f"model max_len_decoder={int(model.decoder.max_len)}"
        )


def _tok_to_midi() -> Dict[int, int]:
    return {tok: midi for midi, tok in MIDI_TO_TOK.items()}


@torch.no_grad()
def parse_dcomposer_tokens(tokens: torch.Tensor, n_acoustic_tokens: int):
    tok_to_midi = _tok_to_midi()
    parsed = []
    for row in tokens:
        row = row.to(torch.long)
        i = 1  # skip BOS
        notes = []
        onsets_sec = []
        codes = []
        while i < row.numel():
            tok = int(row[i].item())
            if tok == EOS_TOK:
                break
            if tok not in tok_to_midi:
                break
            if i + 1 + n_acoustic_tokens >= row.numel():
                break
            onset_tok = int(row[i + 1].item())
            if not (ONSET_TOK_OFFSET <= onset_tok < ACOUSTIC_TOK_OFFSET):
                break
            acoustic = row[i + 2 : i + 2 + n_acoustic_tokens]
            if acoustic.numel() != n_acoustic_tokens:
                break
            if torch.any(acoustic < ACOUSTIC_TOK_OFFSET):
                break
            notes.append(tok_to_midi[tok])
            onset_step = onset_tok - ONSET_TOK_OFFSET
            if onset_step < 0 or onset_step > MAX_ONSET_TOK:
                break
            onsets_sec.append(float(onset_step) / float(MIDI_TIME_RES))
            codes.append((acoustic - ACOUSTIC_TOK_OFFSET).to(torch.long))
            i += 2 + n_acoustic_tokens
        parsed.append(
            {
                "notes": notes,
                "onsets_sec": onsets_sec,
                "codes": codes,
            }
        )
    return parsed


@torch.no_grad()
def render_dcomposer_audio(
    parsed_row: Dict,
    codec_bundle: Dict,
    oneshot_num_samples: int,
    sample_rate: int,
    n_channels: int,
    max_duration: float,
    device,
    autoencoder_n_steps: int = 1,
    autoencoder_cfg_weight: Optional[float] = None,
) -> AudioSignal:
    max_len = max(1, int(round(float(max_duration) * float(sample_rate))))
    if len(parsed_row["codes"]) == 0:
        return AudioSignal(
            torch.zeros(1, int(n_channels), max_len, device=device),
            sample_rate=int(sample_rate),
        )

    code_batch = torch.stack(parsed_row["codes"], dim=0).to(
        device=device, dtype=torch.long
    )
    decoded = decode_dcomposer_oneshots(
        code_batch,
        codec_bundle=codec_bundle,
        n_samples=int(oneshot_num_samples),
        n_channels=int(n_channels),
        sample_rate=int(sample_rate),
        device=device,
        n_steps=int(autoencoder_n_steps),
        cfg_weight=autoencoder_cfg_weight,
    )

    mix = torch.zeros(
        1,
        int(n_channels),
        max_len,
        device=device,
        dtype=decoded.audio_data.dtype,
    )
    for i, onset_sec in enumerate(parsed_row["onsets_sec"]):
        start = int(round(float(onset_sec) * float(sample_rate)))
        if start >= max_len:
            continue
        clip = decoded.audio_data[i : i + 1]
        clip = clip[..., : max(0, max_len - start)]
        if clip.shape[-1] == 0:
            continue
        mix[..., start : start + clip.shape[-1]] += clip

    sig = AudioSignal(mix, sample_rate=int(sample_rate))
    sig = sig.ensure_max_of_audio()
    return sig


def _log_transcript_figure(
    writer,
    step: int,
    tag: str,
    gt_row: Dict,
    pred_row: Dict,
    max_duration: float,
):
    if writer is None:
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    for ax, row, title in [
        (axes[0], gt_row, "Ground Truth"),
        (axes[1], pred_row, "Prediction"),
    ]:
        if len(row["notes"]) > 0:
            ax.scatter(row["onsets_sec"], row["notes"], s=18)
        ax.set_ylabel("MIDI")
        ax.set_title(title)
        ax.set_xlim(0.0, float(max_duration))
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    writer.add_figure(tag, fig, step)
    plt.close(fig)


@torch.no_grad()
def _log_audio_to_tb(writer, tag: str, signal: AudioSignal, step: int):
    if writer is None:
        return
    audio = signal.audio_data.detach().cpu()
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 2:
        audio = audio.mean(dim=0)
    writer.add_audio(tag, audio, step, sample_rate=int(signal.sample_rate))


@torch.no_grad()
def render_target_linear_audio(
    batch: Dict,
    row_idx: int,
    max_duration: float,
    device,
) -> AudioSignal:
    target_oneshots = batch["targets"]["oneshots"]
    oneshot_num_samples = int(batch["targets"]["oneshot_num_samples"][row_idx].item())
    n_events = int(batch["arrangement"]["notes_lengths"][row_idx].item())
    sample_rate = int(batch["signal"].sample_rate)
    n_channels = int(batch["signal"].num_channels)
    max_len = max(1, int(round(float(max_duration) * float(sample_rate))))

    if n_events <= 0 or oneshot_num_samples <= 0:
        return AudioSignal(
            torch.zeros(1, n_channels, max_len, device=device),
            sample_rate=sample_rate,
        )

    audio_i = target_oneshots.audio_data[
        row_idx : row_idx + 1, :, : n_events * oneshot_num_samples
    ].to(device=device)
    onsets_sec = batch["arrangement"]["onsets_sec"][row_idx, 0, :n_events].to(
        device=device
    )

    mix = torch.zeros(
        1,
        n_channels,
        max_len,
        dtype=audio_i.dtype,
        device=device,
    )
    for j in range(n_events):
        start = j * oneshot_num_samples
        stop = start + oneshot_num_samples
        clip = audio_i[:, :, start:stop]
        onset = int(round(float(onsets_sec[j].item()) * float(sample_rate)))
        if onset >= max_len:
            continue
        clip = clip[..., : max(0, max_len - onset)]
        if clip.shape[-1] == 0:
            continue
        mix[..., onset : onset + clip.shape[-1]] += clip

    sig = AudioSignal(mix, sample_rate=sample_rate)
    sig = sig.ensure_max_of_audio()
    return sig


@torch.no_grad()
def _encode_input_latents(signal: AudioSignal, tokenizer) -> Dict:
    seq = tokenizer.encode(signal, scale=True, no_grad=True)
    # Tokenizer.encode(no_grad=True) may return inference-mode tensors; clone so
    # the trainable DComposer encoder can safely use them in autograd.
    latents = seq.tokens.detach().clone()
    lengths = torch.full(
        (latents.shape[0],),
        latents.shape[-1],
        dtype=torch.long,
        device=latents.device,
    )
    return {
        "input_latents": latents,
        "input_latent_lengths": lengths,
    }


@torch.no_grad()
def prepare_dcomposer_batch(
    batch,
    accel,
    oneshot_transform: Optional[Callable],
    mix_transform: Optional[Callable],
    builder_cfg: Dict,
    target_cfg: Dict,
    codec_bundle: Dict,
    n_vocab: int,
    need_input_latents: bool = True,
):
    batch = util.prepare_batch(batch, accel.device)
    batch = build_one_shot_midi_batch(
        batch,
        oneshot_transform=oneshot_transform,
        mix_transform=mix_transform,
        state=batch["idx"],
        **builder_cfg,
    )
    _assert_valid_audio(batch["signal"], "post-builder")

    targets = format_dcomposer_targets(
        batch,
        tokenizer=codec_bundle["tokenizer"],
        autoencoder=codec_bundle["autoencoder"],
        tokenizer_batch_size=target_cfg["tokenizer_batch_size"],
        scale=target_cfg["scale"],
    )
    target_tokens = targets["tokens"].to(accel.device)
    target_lengths = targets["lengths"].to(accel.device)
    _assert_valid_tokens(target_tokens, n_vocab=n_vocab)
    encoder_inputs = (
        _encode_input_latents(batch["signal"], codec_bundle["tokenizer"])
        if need_input_latents
        else {"input_latents": None, "input_latent_lengths": None}
    )

    return {
        "idx": batch["idx"],
        "signal": batch["signal"],
        "signal_lengths": batch["signal_lengths"],
        "arrangement": batch["arrangement"],
        "targets": batch["targets"],
        "input_latents": encoder_inputs["input_latents"],
        "input_latent_lengths": encoder_inputs["input_latent_lengths"],
        "target_tokens": target_tokens,
        "target_lengths": target_lengths,
    }


@torch.no_grad()
def summarize_token_lengths(lengths: torch.Tensor, device: torch.device):
    return {
        "target/mean_len": lengths.float().mean().to(device),
        "target/max_len": lengths.float().amax().to(device),
    }


@torch.no_grad()
def summarize_token_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
):
    pred = logits.argmax(dim=-1)
    correct = (pred == labels) & mask
    denom = mask.sum().clamp_min(1)
    return correct.float().sum() / denom.float()


@torch.no_grad()
def summarize_token_type_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
):
    pred = logits.argmax(dim=-1)

    def _acc(type_mask: torch.Tensor):
        use = mask & type_mask
        denom = use.sum().clamp_min(1)
        return ((pred == labels) & use).float().sum() / denom.float()

    midi_hi = max(MIDI_TO_TOK.values())
    note_mask = (labels >= (EOS_TOK + 1)) & (labels <= midi_hi)
    onset_mask = (labels >= ONSET_TOK_OFFSET) & (labels < ACOUSTIC_TOK_OFFSET)
    acoustic_mask = labels >= ACOUSTIC_TOK_OFFSET
    eos_mask = labels == EOS_TOK

    return {
        "target/acc_note": _acc(note_mask),
        "target/acc_onset": _acc(onset_mask),
        "target/acc_acoustic": _acc(acoustic_mask),
        "target/acc_eos": _acc(eos_mask),
    }


@dataclass
class State:
    model: DComposer
    optimizer: AdamW
    scheduler: torch.optim.lr_scheduler.LRScheduler
    codec_bundle: Dict
    train_oneshot_tfm: transforms.Compose
    train_mix_tfm: transforms.Compose
    val_oneshot_tfm: transforms.Compose
    val_mix_tfm: transforms.Compose
    train_data: OneShotMidiDataset
    val_data: OneShotMidiDataset
    train_builder_cfg: Dict
    val_builder_cfg: Dict
    train_target_cfg: Dict
    val_target_cfg: Dict
    compile_cfg: Dict
    scheduler_cfg: Dict
    memory_cfg: Dict
    sample_cfg: Dict
    tracker: Tracker


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
):
    with argbind.scope(args):
        codec_cfg = build_codec_config()
        compile_cfg = build_compile_config()
        scheduler_cfg = build_scheduler_config()
        memory_cfg = build_memory_config()
        sample_cfg = build_sample_config()

    codec_bundle = load_codec_bundle(codec_cfg, torch.device(accel.device))
    with argbind.scope(args):
        model = DComposer()
    extras = {}

    tracker.print(model)
    print(f"Trainable parameters: {count_parameters(model)}")

    if resume:
        load_dir = f"{save_path}/{tag}"
        model_pth = Path(load_dir) / "model.pt"
        extras_pth = Path(load_dir) / "extras.pt"
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{load_dir}")
        if model_pth.exists():
            model.load_state_dict(torch.load(model_pth, map_location="cpu"))
        if extras_pth.exists():
            extras = torch.load(extras_pth, map_location="cpu", weights_only=False)

    if compile_cfg["compile_model"]:
        tracker.print(
            "Compiling model.forward "
            f"(backend={compile_cfg['compile_backend']}, "
            f"mode={compile_cfg['compile_mode']}, "
            f"dynamic={compile_cfg['compile_dynamic']}, "
            f"fullgraph={compile_cfg['compile_fullgraph']})"
        )
        model.forward = torch.compile(
            model.forward,
            mode=compile_cfg["compile_mode"],
            backend=compile_cfg["compile_backend"],
            dynamic=compile_cfg["compile_dynamic"],
            fullgraph=compile_cfg["compile_fullgraph"],
        )

    model = accel.prepare_model(model)

    with argbind.scope(args):
        optimizer = AdamW(model.parameters(), use_zero=accel.use_ddp)
        scheduler = build_scheduler(optimizer, scheduler_cfg)

    if "optimizer" in extras:
        optimizer.load_state_dict(extras["optimizer"])
    if "scheduler" in extras:
        scheduler.load_state_dict(extras["scheduler"])
    if "tracker" in extras:
        tracker.load_state_dict(extras["tracker"])

    with argbind.scope(args, "train_oneshot"):
        train_oneshot_tfm = build_transform()
    with argbind.scope(args, "train_mix"):
        train_mix_tfm = build_transform()
    with argbind.scope(args, "val_oneshot"):
        val_oneshot_tfm = build_transform()
    with argbind.scope(args, "val_mix"):
        val_mix_tfm = build_transform()

    with argbind.scope(args, "train"):
        train_data = OneShotMidiDataset()
        train_builder_cfg = build_builder_config()
        train_target_cfg = build_target_config()

    with argbind.scope(args, "val"):
        val_data = OneShotMidiDataset()
        val_builder_cfg = build_builder_config()
        val_target_cfg = build_target_config()

    assert_codec_matches_dataset_duration(
        train_data, codec_bundle, torch.device(accel.device)
    )
    assert_codec_matches_dataset_duration(
        val_data, codec_bundle, torch.device(accel.device)
    )
    assert_dcomposer_compatibility(
        train_data, train_builder_cfg, codec_bundle, accel.unwrap(model)
    )
    assert_dcomposer_compatibility(
        val_data, val_builder_cfg, codec_bundle, accel.unwrap(model)
    )

    return State(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        codec_bundle=codec_bundle,
        train_oneshot_tfm=train_oneshot_tfm,
        train_mix_tfm=train_mix_tfm,
        val_oneshot_tfm=val_oneshot_tfm,
        val_mix_tfm=val_mix_tfm,
        train_data=train_data,
        val_data=val_data,
        train_builder_cfg=train_builder_cfg,
        val_builder_cfg=val_builder_cfg,
        train_target_cfg=train_target_cfg,
        val_target_cfg=val_target_cfg,
        compile_cfg=compile_cfg,
        scheduler_cfg=scheduler_cfg,
        memory_cfg=memory_cfg,
        sample_cfg=sample_cfg,
        tracker=tracker,
    )


@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    output = {}
    state.model.eval()
    n_vocab = int(ACOUSTIC_TOK_OFFSET + state.codec_bundle["autoencoder"].n_vocab)

    batch = prepare_dcomposer_batch(
        batch=batch,
        accel=accel,
        oneshot_transform=state.val_oneshot_tfm,
        mix_transform=state.val_mix_tfm,
        builder_cfg=state.val_builder_cfg,
        target_cfg=state.val_target_cfg,
        codec_bundle=state.codec_bundle,
        n_vocab=n_vocab,
        need_input_latents=bool(accel.unwrap(state.model).use_encoder),
    )

    inp = batch["target_tokens"][:, :-1]
    labels = batch["target_tokens"][:, 1:]
    valid_lengths = (batch["target_lengths"] - 1).clamp_min(0)
    idx = torch.arange(inp.shape[1], device=inp.device)[None, :]
    mask = idx < valid_lengths[:, None]
    ignore_index = -100
    labels_masked = labels.masked_fill(~mask, ignore_index)

    with accel.autocast():
        logits = accel.unwrap(state.model)(
            inp,
            input_latents=batch["input_latents"],
            input_latent_lengths=batch["input_latent_lengths"],
        )
        logits_constrained = accel.unwrap(state.model).constrain_teacher_forcing_logits(
            logits
        )
        logits = torch.where(mask[:, :, None], logits_constrained, logits)
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels_masked,
            reduction="sum",
            ignore_index=ignore_index,
        )
        loss = loss / mask.sum().clamp_min(1).float()

    output["loss/ce"] = loss.detach()
    output["target/acc"] = summarize_token_accuracy(
        logits.detach(), labels, mask
    ).detach()
    output.update(
        {
            k: v.detach()
            for k, v in summarize_token_type_accuracy(
                logits.detach(), labels, mask
            ).items()
        }
    )
    output.update(summarize_token_lengths(batch["target_lengths"], logits.device))
    return output


@timer()
def train_loop(state, batch, accel):
    output = {}
    state.model.train()
    device = torch.device(accel.device)
    n_vocab = int(ACOUSTIC_TOK_OFFSET + state.codec_bundle["autoencoder"].n_vocab)
    if state.memory_cfg["log_gpu_memory"] and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    batch = prepare_dcomposer_batch(
        batch=batch,
        accel=accel,
        oneshot_transform=state.train_oneshot_tfm,
        mix_transform=state.train_mix_tfm,
        builder_cfg=state.train_builder_cfg,
        target_cfg=state.train_target_cfg,
        codec_bundle=state.codec_bundle,
        n_vocab=n_vocab,
        need_input_latents=bool(accel.unwrap(state.model).use_encoder),
    )

    inp = batch["target_tokens"][:, :-1]
    labels = batch["target_tokens"][:, 1:]
    valid_lengths = (batch["target_lengths"] - 1).clamp_min(0)
    idx = torch.arange(inp.shape[1], device=inp.device)[None, :]
    mask = idx < valid_lengths[:, None]
    ignore_index = -100

    with accel.autocast():
        logits = state.model(
            inp,
            input_latents=batch["input_latents"],
            input_latent_lengths=batch["input_latent_lengths"],
        )
        logits_constrained = accel.unwrap(state.model).constrain_teacher_forcing_logits(
            logits
        )
        logits = torch.where(mask[:, :, None], logits_constrained, logits)
        labels_masked = labels.masked_fill(~mask, ignore_index)
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels_masked,
            reduction="sum",
            ignore_index=ignore_index,
        )
        loss = loss / mask.sum().clamp_min(1).float()

    with torch.no_grad():
        output["loss/ce"] = loss.detach()
        output["target/acc"] = summarize_token_accuracy(
            logits.detach(), labels, mask
        ).detach()
        output.update(
            {
                k: v.detach()
                for k, v in summarize_token_type_accuracy(
                    logits.detach(), labels, mask
                ).items()
            }
        )
        output.update(summarize_token_lengths(batch["target_lengths"], logits.device))

    state.optimizer.zero_grad()
    accel.backward(loss)
    accel.scaler.unscale_(state.optimizer)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.model.parameters(), 1.0
    ).detach()
    accel.step(state.optimizer)
    state.scheduler.step()
    accel.update()

    output["other/learning_rate"] = torch.as_tensor(
        state.optimizer.param_groups[0]["lr"], device=logits.device
    )
    output["other/batch_size"] = torch.as_tensor(
        batch["signal"].batch_size * accel.world_size, device=logits.device
    )
    if state.memory_cfg["log_gpu_memory"] and device.type == "cuda":
        denom = float(1024**3)
        output["memory/gpu_alloc_gb"] = torch.as_tensor(
            torch.cuda.memory_allocated(device) / denom, device=logits.device
        )
        output["memory/gpu_reserved_gb"] = torch.as_tensor(
            torch.cuda.memory_reserved(device) / denom, device=logits.device
        )
        output["memory/gpu_peak_alloc_gb"] = torch.as_tensor(
            torch.cuda.max_memory_allocated(device) / denom, device=logits.device
        )

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, accel, save_iters, save_path):
    metadata = {"logs": state.tracker.history}
    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "loss/ce"):
        state.tracker.print("Best model so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step}")

    for tag in tags:
        save_dir = f"{save_path}/{tag}"
        ensure_dir(save_dir)
        model_pth = Path(save_dir) / "model.pt"
        extras_pth = Path(save_dir) / "extras.pt"
        extras = {
            "optimizer": state.optimizer.state_dict(),
            "scheduler": state.scheduler.state_dict(),
            "tracker": state.tracker.state_dict(),
            "metadata": metadata,
            "codec": {
                "run_dir": state.codec_bundle["run_dir"],
                "checkpoint": state.codec_bundle["checkpoint"],
                "use_ema": state.codec_bundle["use_ema"],
                "tokenizer_name": state.codec_bundle["tokenizer"].name,
                "autoencoder_name": type(state.codec_bundle["autoencoder"]).__name__,
                "autoencoder_n_vocab": int(state.codec_bundle["autoencoder"].n_vocab),
                "autoencoder_n_codebooks": int(
                    state.codec_bundle["autoencoder"].n_codebooks
                ),
                "autoencoder_max_len": int(state.codec_bundle["autoencoder"].max_len),
            },
            "compile": state.compile_cfg,
            "scheduler_cfg": state.scheduler_cfg,
        }
        torch.save(accel.unwrap(state.model).state_dict(), model_pth)
        torch.save(extras, extras_pth)


@torch.no_grad()
@argbind.bind(without_prefix=True)
def save_samples(state, accel, sample_idx, writer):
    if writer is None:
        return
    state.tracker.print("Saving DComposer samples to TensorBoard")
    samples = [state.val_data[idx] for idx in sample_idx]
    batch = state.val_data.collate(samples)
    n_vocab = int(ACOUSTIC_TOK_OFFSET + state.codec_bundle["autoencoder"].n_vocab)
    batch = prepare_dcomposer_batch(
        batch=batch,
        accel=accel,
        oneshot_transform=state.val_oneshot_tfm,
        mix_transform=state.val_mix_tfm,
        builder_cfg=state.val_builder_cfg,
        target_cfg=state.val_target_cfg,
        codec_bundle=state.codec_bundle,
        n_vocab=n_vocab,
        need_input_latents=bool(accel.unwrap(state.model).use_encoder),
    )
    model = accel.unwrap(state.model)
    max_new_tokens = state.sample_cfg["max_new_tokens"]
    if max_new_tokens is None:
        max_new_tokens = int(batch["target_lengths"].amax().item()) - 1
    max_new_tokens = min(max(1, int(max_new_tokens)), int(model.decoder.max_len) - 1)

    was_training = model.training
    model.eval()
    pred = model.inference(
        input_latents=batch["input_latents"],
        input_latent_lengths=batch["input_latent_lengths"],
        max_new_tokens=max_new_tokens,
        top_p=state.sample_cfg["top_p"],
        top_k=state.sample_cfg["top_k"],
        temp=state.sample_cfg["temp"],
        argmax=state.sample_cfg["argmax"],
        eos_threshold=state.sample_cfg["eos_threshold"],
    )
    if was_training:
        model.train()

    gt_rows = parse_dcomposer_tokens(
        batch["target_tokens"], n_acoustic_tokens=model.n_acoustic_tokens
    )
    pred_rows = parse_dcomposer_tokens(
        pred["tokens"], n_acoustic_tokens=model.n_acoustic_tokens
    )

    for i in range(batch["signal"].batch_size):
        gt_audio = batch["signal"][i].cpu()
        _log_audio_to_tb(
            writer, f"samples/gt_audio/sample_{i}", gt_audio, state.tracker.step
        )

        oracle_audio = render_target_linear_audio(
            batch,
            row_idx=i,
            max_duration=min(
                float(state.sample_cfg["render_max_duration"]),
                float(MAX_DURATION),
            ),
            device=torch.device(accel.device),
        ).cpu()
        _log_audio_to_tb(
            writer,
            f"samples/oracle_linear_audio/sample_{i}",
            oracle_audio,
            state.tracker.step,
        )

        pred_audio = render_dcomposer_audio(
            pred_rows[i],
            codec_bundle=state.codec_bundle,
            oneshot_num_samples=int(batch["targets"]["oneshot_num_samples"][i].item()),
            sample_rate=int(batch["signal"].sample_rate),
            n_channels=int(batch["signal"].num_channels),
            max_duration=min(
                float(state.sample_cfg["render_max_duration"]),
                float(MAX_DURATION),
            ),
            device=torch.device(accel.device),
            autoencoder_n_steps=state.sample_cfg["autoencoder_n_steps"],
            autoencoder_cfg_weight=state.sample_cfg["autoencoder_cfg_weight"],
        ).cpu()
        _log_audio_to_tb(
            writer, f"samples/pred_audio/sample_{i}", pred_audio, state.tracker.step
        )

        _log_transcript_figure(
            writer,
            state.tracker.step,
            f"samples/transcript/sample_{i}",
            gt_rows[i],
            pred_rows[i],
            max_duration=min(
                float(state.sample_cfg["render_max_duration"]),
                float(MAX_DURATION),
            ),
        )

        writer.add_text(
            f"samples/text/sample_{i}",
            (
                f"gt_events={len(gt_rows[i]['notes'])}, "
                f"pred_events={len(pred_rows[i]['notes'])}"
            ),
            state.tracker.step,
        )

    writer.add_scalar(
        "samples/target_mean_len",
        float(batch["target_lengths"].float().mean().detach().cpu()),
        state.tracker.step,
    )
    writer.add_scalar(
        "samples/audio_mean_len_sec",
        float(
            batch["signal_lengths"].float().mean().detach().cpu()
            / batch["signal"].sample_rate
        ),
        state.tracker.step,
    )


def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        val_loop(batch, state, accel)
    if hasattr(state.optimizer, "consolidate_state_dict"):
        state.optimizer.consolidate_state_dict()
    return {}


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "runs/dcomposer",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    val_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    sample_idx: list = [0, 1, 2, 3],
):
    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    if accel.local_rank == 0:
        dump_resolved_config(args, save_path, model_name="dcomposer")
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )

    state = load(args, accel, tracker, save_path)

    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=False,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    train_dataloader = get_infinite_loader(train_dataloader)

    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=False,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
    )

    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    with tracker.live:
        for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
            train_loop(state, batch, accel)

            last_iter = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )
            if tracker.step % sample_freq == 0 or last_iter:
                save_samples(state, accel, sample_idx, writer)

            if tracker.step % val_freq == 0 or last_iter:
                validate(state, val_dataloader, accel)
                checkpoint(state, accel, save_iters, save_path)
                tracker.done("val", f"Iteration {tracker.step}")

            if last_iter:
                break


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
