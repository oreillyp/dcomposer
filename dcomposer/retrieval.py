"""Decoder-free DOC encoding and nearest-neighbor sample selection."""
import hashlib
import csv
from pathlib import Path

import torch
from audiotools import AudioSignal

from .constants import (
    ACOUSTIC_TOK_OFFSET,
    BOS_TOK,
    EOS_TOK,
    COARSE_MIDI_NOTE_TO_COARSE_LABEL,
    FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE,
    MIDI_TIME_RES,
    MIDI_TO_TOK,
    ONSET_TOK_OFFSET,
)
from .model.flow_autoencoder import LatentFlowAutoencoder


def parse_events(tokens, count, n_vocab):
    """Parse complete event sequences; never accept a truncated prediction."""
    tokens = torch.as_tensor(tokens)
    if tokens.ndim != 1 or tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected a one-dimensional integer token sequence")
    if not len(tokens) or int(tokens[0]) != BOS_TOK:
        raise ValueError("Missing BOS token")
    notes = {tok: midi for midi, tok in MIDI_TO_TOK.items()}
    events = []
    i = 1
    while i < len(tokens):
        if int(tokens[i]) == EOS_TOK:
            if i != len(tokens) - 1:
                raise ValueError("Unexpected tokens after EOS")
            return events
        if int(tokens[i]) not in notes or i + 2 + count > len(tokens):
            raise ValueError(f"Malformed event at token {i}")
        onset = int(tokens[i + 1])
        if not ONSET_TOK_OFFSET <= onset < ACOUSTIC_TOK_OFFSET:
            raise ValueError(f"Invalid onset token at {i + 1}")
        codes = tokens[i + 2 : i + 2 + count] - ACOUSTIC_TOK_OFFSET
        if torch.any((codes < 0) | (codes >= n_vocab)):
            raise ValueError(f"Invalid acoustic codes at {i + 2}")
        note = notes[int(tokens[i])]
        coarse = FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE.get(note, note)
        events.append(
            (
                note,
                COARSE_MIDI_NOTE_TO_COARSE_LABEL[coarse],
                (onset - ONSET_TOK_OFFSET) / MIDI_TIME_RES,
                codes,
            )
        )
        i += 2 + count
    raise ValueError("Missing EOS token (prediction may have been truncated)")


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_tokenizer(bundle, weights="", device="cpu"):
    """Restore the public base VAE and saved scaling, not the DOC decoder."""
    from .pipelines.tokenizer.tokenizer import Tokenizer

    kwargs = dict(bundle["tokenizer"], load_saved_scale=False)
    if weights:
        kwargs["ckpt_pth"] = weights
    tokenizer = Tokenizer(**kwargs).to(device).eval()
    scale = bundle["scale"]
    tokenizer.set_scale(
        mean=scale.get("mean"),
        std=scale.get("std"),
        per_channel=scale.get("per_channel", True),
    )
    return tokenizer


def read_samples(manifest):
    manifest = Path(manifest).resolve()
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty sample manifest")
    paths, labels = [], []
    for row in rows:
        path = Path(row["oneshot"]).expanduser()
        if not path.is_absolute():
            path = manifest.parent / path
        if not path.is_file() or path.suffix.lower() != ".wav":
            raise ValueError(f"Expected an existing user sample WAV: {path}")
        label = row["coarse_label"].strip()
        if label not in COARSE_MIDI_NOTE_TO_COARSE_LABEL.values():
            raise ValueError(f"Unknown coarse_label: {label}")
        paths.append(str(path.resolve()))
        labels.append(label)
    return paths, labels


def load_audio(path, sample_rate, channels):
    signal = AudioSignal(path).resample(sample_rate)
    if signal.num_channels != channels:
        signal.audio_data = signal.audio_data.mean(1, keepdim=True).repeat(
            1, channels, 1
        )
    if not torch.isfinite(signal.audio_data).all():
        raise ValueError(f"Non-finite audio: {path}")
    return signal


@torch.inference_mode()
def index_samples(
    encoder_path,
    manifest,
    output,
    tokenizer_weights,
    device,
    batch_size,
    seed,
    progress=None,
):
    paths, labels = read_samples(manifest)
    bundle = torch.load(encoder_path, map_location="cpu", weights_only=True)
    encoder = DOCEncoder(**bundle["config"])
    encoder.load_state_dict(bundle["state_dict"], strict=True, assign=True)
    encoder.to(device).eval()
    tokenizer = load_tokenizer(bundle, tokenizer_weights, device)
    torch.manual_seed(seed)
    n_samples = int(bundle["duration"] * bundle["sample_rate"])
    codes = []
    for start in range(0, len(paths), batch_size):
        audio = []
        for path in paths[start : start + batch_size]:
            signal = load_audio(path, bundle["sample_rate"], tokenizer.n_channels)
            x = signal.audio_data[..., :n_samples]
            audio.append(torch.nn.functional.pad(x, (0, n_samples - x.shape[-1])))
        signal = AudioSignal(torch.cat(audio).to(device), bundle["sample_rate"])
        latents = tokenizer.encode(signal, scale=True).tokens
        codes.append(encoder.quantize(encoder.encode(latents))[2].cpu())
        print(f"Indexed {min(start+batch_size, len(paths))}/{len(paths)}", flush=True)
        if progress is not None:
            progress(
                f"Indexed {min(start+batch_size, len(paths))}/{len(paths)} samples"
            )
    torch.save(
        dict(
            codes=torch.cat(codes),
            paths=paths,
            labels=labels,
            n_fsq=encoder.n_fsq,
            group_size=encoder.fsq_group_size,
            encoder_sha256=digest(encoder_path),
            seed=seed,
            sample_rate=bundle["sample_rate"],
            n_channels=tokenizer.n_channels,
        ),
        output,
    )


class DOCEncoder(torch.nn.Module):
    """The existing DOC encoder/FSQ, without decoder or upsampler weights."""

    def __init__(self, **config):
        super().__init__()
        if config.get("bottleneck_type", "fsq") != "fsq":
            raise ValueError("Sample retrieval currently supports FSQ DOC models only")
        # Reuse the exact architecture without allocating the discarded decoder.
        with torch.device("meta"):
            model = LatentFlowAutoencoder(**config)
        for name in (
            "in_proj_enc",
            "out_proj_enc",
            "register_norm_pre",
            "register_norm_post",
            "encoder",
            "bottleneck",
            "registers",
        ):
            setattr(self, name, getattr(model, name))
        for name in (
            "max_len",
            "n_channels_in",
            "n_channels_summary",
            "n_summary",
            "n_fsq",
            "fsq_group_size",
            "bottleneck_type",
        ):
            setattr(self, name, getattr(model, name))

    encode = LatentFlowAutoencoder.encode
    quantize = LatentFlowAutoencoder.quantize
    _quantize_with_losses = LatentFlowAutoencoder._quantize_with_losses
    n_vocab = LatentFlowAutoencoder.n_vocab
    n_codebooks = LatentFlowAutoencoder.n_codebooks


class SampleBank:
    """Search unpacked FSQ coordinates, or Hamming distance on packed tokens.

    Labels are the existing coarse drum labels. Ties select the first CSV row.
    No learned decoder is constructed or called by this class.
    """

    def __init__(self, codes, paths, labels, n_fsq, group_size, device="cpu"):
        if n_fsq < 2 or group_size < 1:
            raise ValueError("Invalid FSQ geometry")
        self.n_fsq = n_fsq
        self.group_size = group_size
        self.paths = list(paths)
        self.labels = list(labels)
        self.codes = torch.as_tensor(codes, device=device)
        self.coordinates = self.unpack(self.codes)
        if (
            not len(self.codes)
            or len(self.codes) != len(paths)
            or len(paths) != len(labels)
        ):
            raise ValueError("Expected nonempty, equally sized codes, paths and labels")
        if any(not str(label).strip() for label in self.labels):
            raise ValueError("Empty corpus class label")

    def unpack(self, codes):
        if (
            codes.ndim != 2
            or codes.shape[1] == 0
            or codes.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Expected a [samples, codebooks] integer tensor")
        if torch.any((codes < 0) | (codes >= self.n_fsq**self.group_size)):
            raise ValueError("Acoustic code outside the FSQ vocabulary")
        powers = self.n_fsq ** torch.arange(self.group_size, device=codes.device)
        digits = (codes[..., None] // powers) % self.n_fsq
        return ((digits.float() - self.n_fsq // 2) / (self.n_fsq // 2)).flatten(1)

    @torch.no_grad()
    def search(self, codes, labels=None, metric="l2", chunk_size=4096):
        codes = torch.as_tensor(codes, device=self.codes.device)
        coordinates = self.unpack(codes)
        if codes.shape[1] != self.codes.shape[1]:
            raise ValueError("Query and corpus codebook counts differ")
        if metric not in ("l2", "hamming") or chunk_size < 1:
            raise ValueError("Use l2 or hamming and a positive chunk_size")
        if labels is not None:
            if len(labels) != len(codes):
                raise ValueError("Expected one class label per query")
            missing = set(labels) - set(self.labels)
            if missing:
                raise ValueError(
                    f"No corpus samples for requested classes: {sorted(missing)}"
                )
        best = torch.full((len(codes),), float("inf"), device=codes.device)
        indices = torch.zeros(len(codes), dtype=torch.long, device=codes.device)
        for start in range(0, len(self.codes), chunk_size):
            end = start + chunk_size
            if metric == "l2":
                distances = torch.cdist(
                    coordinates,
                    self.coordinates[start:end],
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
            else:
                distances = (
                    (codes[:, None] != self.codes[None, start:end]).float().mean(-1)
                )
            if labels is not None:
                allowed = torch.tensor(
                    [[q == k for k in self.labels[start:end]] for q in labels],
                    device=codes.device,
                )
                distances.masked_fill_(~allowed, float("inf"))
            values, local = distances.min(1)
            better = values < best
            indices = torch.where(better, local + start, indices)
            best = torch.minimum(best, values)
        return indices, best
