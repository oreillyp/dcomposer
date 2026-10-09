"""Model loading and audio preparation shared by command-line and app inference."""
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from audiotools import AudioSignal

from .dsp import resample
from .model import DComposer, LatentFlowAutoencoder
from .pipelines.tokenizer import Tokenizer, TokenSequence
from .util import load_config


def config_path(run_dir):
    run_dir = Path(run_dir)
    if run_dir.is_file():
        return run_dir
    for name in ("conf.yml", "flow.yml"):
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No saved configuration in {run_dir}")


def load_model(run_dir, checkpoint, device):
    run_dir = Path(run_dir)
    path = config_path(run_dir)
    if run_dir.is_file():
        bundle = torch.load(run_dir, map_location="cpu", weights_only=True)
        args, state = bundle["args"], bundle["state_dict"]
    else:
        args = load_config(path)
        state = torch.load(
            run_dir / checkpoint / "model.pt", map_location="cpu", weights_only=True
        )
    model = DComposer(
        **{k.split(".", 1)[1]: v for k, v in args.items() if k.startswith("DComposer.")}
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, args, path


def load_codec_bundle(cfg, device):
    if not cfg["run_dir"]:
        raise ValueError("codec_run_dir must be provided")
    run_path = Path(cfg["run_dir"]).expanduser()
    checkpoint = run_path / str(cfg["checkpoint"])
    extras_path = checkpoint / "extras.pt"
    weights = checkpoint / ("ema_model.pt" if cfg["use_ema"] else "model.pt")
    if not extras_path.is_file():
        raise FileNotFoundError(f"Missing extras.pt in {checkpoint}")
    if not weights.is_file():
        raise FileNotFoundError(f"Missing requested model weights: {weights}")
    path = config_path(run_path)
    args = load_config(path)
    extras = torch.load(extras_path, map_location="cpu", weights_only=False)
    config = {
        k.split(".", 1)[1]: v
        for k, v in args.items()
        if k.startswith("LatentFlowAutoencoder.")
    }
    if args.get("model_name", "flow") != "flow" or not config:
        raise ValueError(f"Expected a paper DOC flow model in {path}")
    model = LatentFlowAutoencoder(**config)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.to(device).eval().requires_grad_(False)
    kwargs = {
        k.split(".", 1)[1]: v for k, v in args.items() if k.startswith("Tokenizer.")
    }
    for name in ("name", "normalize_db", "loudness_exclude_silence"):
        if name in extras.get("tokenizer", {}):
            kwargs[name] = extras["tokenizer"][name]
    tokenizer = Tokenizer(**kwargs).to(device)
    if "tokenizer_scale" in extras:
        scale = extras["tokenizer_scale"]
        tokenizer.set_scale(
            mean=scale.get("mean"),
            std=scale.get("std"),
            per_channel=bool(scale.get("per_channel", True)),
        )
    return dict(
        run_dir=str(run_path),
        checkpoint=str(cfg["checkpoint"]),
        use_ema=bool(cfg["use_ema"]),
        config_path=str(path),
        tokenizer=tokenizer,
        autoencoder=model,
        extras=extras,
    )


def preprocess_signal_for_batch(
    path,
    *,
    sample_rate,
    n_channels,
    duration_s,
    normalize_db: Optional[float] = None,
    ensure_max=False,
):
    audio, rate = sf.read(str(path), always_2d=True, dtype="float32")
    signal = AudioSignal(torch.from_numpy(audio.T).unsqueeze(0), sample_rate=int(rate))
    if signal.num_channels != n_channels:
        signal.audio_data = signal.audio_data.mean(1, keepdim=True).repeat(
            1, n_channels, 1
        )
    if int(signal.sample_rate) != int(sample_rate):
        signal = resample(signal, int(sample_rate))
    if normalize_db is not None:
        x = signal.audio_data
        rms = torch.sqrt((x**2).mean(dim=(1, 2)).clamp_min(1e-12))
        gain_db = torch.as_tensor(
            [float(normalize_db)], device=x.device, dtype=x.dtype
        ) - 20 * torch.log10(rms)
        ln10 = torch.log(torch.tensor([10.0], device=x.device, dtype=x.dtype))
        signal.audio_data = x * torch.exp(gain_db / 20 * ln10).view(x.shape[0], 1, 1)
        if ensure_max:
            signal = signal.ensure_max_of_audio()
    if duration_s is not None:
        samples = int(round(float(duration_s) * int(sample_rate)))
        if signal.signal_length > samples:
            signal.truncate_samples(samples)
        elif signal.signal_length < samples:
            signal = signal.zero_pad_to(samples)
    return signal


@torch.no_grad()
def decode_dcomposer_oneshots(
    codes,
    codec_bundle,
    n_samples,
    n_channels,
    sample_rate,
    device,
    n_steps=1,
    cfg_weight=None,
):
    latents = codec_bundle["autoencoder"].inference(
        codes=codes.to(device=device, dtype=torch.long),
        n_steps=int(n_steps),
        cfg_weight=cfg_weight,
    )
    sequence = TokenSequence(
        tokens=latents,
        extras=dict(
            sample_rate=int(sample_rate),
            n_channels=int(n_channels),
            signal_length=int(n_samples),
        ),
        scaled=True,
    )
    return codec_bundle["tokenizer"].decode(sequence, no_grad=True)
