from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from audiotools import AudioSignal


def _resolve_device(device: Optional[str] = None) -> torch.device:
    if device in [None, ""]:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _match_num_channels(sig: AudioSignal, num_channels: int) -> AudioSignal:
    nc = int(num_channels)
    if sig.num_channels == nc:
        return sig
    if nc == 1:
        return sig.to_mono()
    if sig.num_channels == 1:
        sig.audio_data = sig.audio_data.repeat(1, nc, 1)
        return sig
    sig.audio_data = sig.audio_data.mean(dim=1, keepdim=True).repeat(1, nc, 1)
    return sig


def _canonicalize_audio(
    path: Path,
    sample_rate: int,
    num_channels: int,
    device: Optional[str] = None,
) -> np.ndarray:
    from dcomposer import dsp

    audio, src_sr = sf.read(str(path), always_2d=True, dtype="float32")
    x = torch.from_numpy(audio.T)[None, :, :]
    sig = AudioSignal(x, sample_rate=int(src_sr)).to(_resolve_device(device))
    sig = _match_num_channels(sig, int(num_channels))
    sig = dsp.resample(sig, int(sample_rate), inplace=False)
    y = sig.audio_data.detach().cpu().squeeze(0).numpy()
    return np.ascontiguousarray(y, dtype=np.float32)


def pack_audio_rows(
    rows: Iterable[Dict],
    *,
    resolve_path: Callable[[Dict], Path],
    packed_audio_path: Path,
    sample_rate: int,
    num_channels: int,
    device: Optional[str] = None,
) -> Dict[str, Dict]:
    packed_audio_path = Path(packed_audio_path).expanduser().resolve()
    packed_audio_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_by_path: Dict[str, Dict] = {}
    frame_offset = 0

    with open(packed_audio_path, "wb") as f:
        for row in rows:
            source_path = resolve_path(row).expanduser().resolve()
            key = str(source_path)
            if key in metadata_by_path:
                continue

            audio = _canonicalize_audio(
                source_path,
                sample_rate=int(sample_rate),
                num_channels=int(num_channels),
                device=device,
            )
            channels, num_frames = audio.shape
            f.write(audio.T.tobytes(order="C"))
            metadata_by_path[key] = {
                "packed_audio_path": str(packed_audio_path),
                "packed_audio_offset": int(frame_offset),
                "packed_audio_num_frames": int(num_frames),
                "packed_audio_num_channels": int(channels),
                "packed_audio_sample_rate": int(sample_rate),
                "packed_audio_dtype": "float32",
            }
            frame_offset += int(num_frames)

    return metadata_by_path
