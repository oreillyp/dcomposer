"""Shared paper evaluation audio preparation and magnitude distances."""
import soundfile as sf
import torch
from audiotools import AudioSignal

from dcomposer.dsp import resample

WINDOWS = (2048, 1024, 512, 256, 128, 64)


def load_audio(path, samples, normalize=False):
    if path is None:
        return AudioSignal(torch.zeros(1, 1, samples), sample_rate=44100)
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    signal = AudioSignal(torch.from_numpy(audio.T.copy()).unsqueeze(0), sample_rate=sr)
    if not torch.isfinite(signal.audio_data).all():
        raise ValueError(f"Nonfinite audio: {path}")
    if normalize:
        x = signal.audio_data
        rms = x.square().mean().clamp_min(1e-12).sqrt()
        signal.audio_data = x * (10 ** (-16 / 20) / rms)
        signal.ensure_max_of_audio()
    signal.to_mono()
    if sr != 44100:
        signal = resample(signal, 44100)
    if signal.signal_length > samples:
        signal.truncate_samples(samples)
    elif signal.signal_length < samples:
        signal = signal.zero_pad_to(samples)
    return signal


@torch.no_grad()
def spectral_scores(ref, pred):
    x, y = ref.audio_data, pred.audio_data
    gain = (
        x.square().mean((1, 2)).clamp_min(1e-12)
        / y.square().mean((1, 2)).clamp_min(1e-12)
    ).sqrt()
    scores = {
        f"{loss}_{mode}": torch.zeros(x.shape[0], device=x.device)
        for loss in ("mse", "l1")
        for mode in ("raw", "rms_matched")
    }
    for window in WINDOWS:
        ref.stft(window_length=window, hop_length=window // 4)
        pred.stft(window_length=window, hop_length=window // 4)
        a, b = ref.magnitude, pred.magnitude
        for mode, estimate in (
            ("raw", b),
            ("rms_matched", b * gain[:, None, None, None]),
        ):
            delta = a - estimate
            scores[f"mse_{mode}"] += delta.square().flatten(1).mean(1)
            scores[f"l1_{mode}"] += delta.abs().flatten(1).mean(1)
    if not all(torch.isfinite(v).all() for v in scores.values()):
        raise FloatingPointError("Nonfinite spectral score")
    return {k: v.cpu().tolist() for k, v in scores.items()}
