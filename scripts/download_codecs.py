"""Prepare the submission-state codecs from their official checkpoints."""
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import argbind
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dcomposer.pipelines.tokenizer.codicodec import CodiCodecVAE
from dcomposer.pipelines.tokenizer.melodyflow import MelodyFlowVAE
from dcomposer.pipelines.tokenizer.stable_audio import StableAudioVAE

# Official repo, revision, filename, SHA-256, local filename.
FILES = {
    "melodyflow": (
        "facebook/melodyflow-t24-30secs",
        "77bcfce24371bf29a06152c72169162c6f2791de",
        "compression_state_dict.bin",
        "c075ee7c5b13d50937d1e4f197f3e940c3f3b74207857cb0e1e17891010fdc6d",
        "melodyflow_encodec_vae.pt",
    ),
    "codicodec": (
        "SonyCSLParis/CoDiCodec",
        "f6aa343c06104f743563e86485e69e55dfd30d58",
        "codicodec.pt",
        "27827ac41100386193fa4fd04d783c62d526a7b06ffa9e9252992eb3eb4bdb6c",
        "codicodec_continuous.pt",
    ),
    "stable_audio": (
        "stabilityai/stable-audio-open-1.0",
        "f21265c1e2710b3bd2386596943f0007f55f802e",
        "vae_model.ckpt",
        "771265f2e9a7fa9c3b7899be2d6a5a93954032e5e696edd62239ba7f1cd67116",
        "stable_audio_vae.pt",
    ),
}


def convert(source, output, codec):
    with open(source, "rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != FILES[codec][3]:
        raise ValueError(f"Official checkpoint checksum mismatch: {actual}")
    # MelodyFlow/CoDiCodec metadata includes Python objects: check the pinned hash first.
    checkpoint = torch.load(
        source, map_location="cpu", weights_only=(codec == "stable_audio")
    )
    if codec == "melodyflow":
        state = {
            k: v
            for k, v in checkpoint["best_state"].items()
            if k.startswith(("encoder.", "decoder."))
        }
    elif codec == "codicodec":
        state = checkpoint["gen_state_dict"]
    else:
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not state or any(
        not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
        for v in state.values()
    ):
        raise ValueError("Missing or non-finite codec weights")
    torch.save(state, output)
    # Strict architecture checks before publishing; no model code is changed.
    {
        "melodyflow": MelodyFlowVAE,
        "codicodec": CodiCodecVAE,
        "stable_audio": StableAudioVAE,
    }[codec](output)


@argbind.bind()
def run(codec: str = "melodyflow", output: str = "", source: str = ""):
    if codec not in FILES:
        raise ValueError(f"Choose a codec from {list(FILES)}")
    repo, revision, filename, _, local_name = FILES[codec]
    output = Path(output or f"pretrained/tokenizer/{codec}/{local_name}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    print(f"Official {codec} weights: review https://huggingface.co/{repo}", flush=True)
    print(f"Preparing {output}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        tmp = Path(tmp)
        if not source and codec == "stable_audio":
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as error:
                raise ImportError(
                    "Stable Audio download needs: pip install huggingface_hub"
                ) from error
            print(
                "Stable Audio requires accepting the model terms and running hf auth login.",
                flush=True,
            )
            source = hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=revision,
                token=True,
                local_dir=tmp,
            )
        elif not source:
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
            print(f"Downloading {url}", flush=True)
            source = tmp / filename
            with urllib.request.urlopen(url, timeout=60) as response, source.open(
                "wb"
            ) as stream:
                downloaded = 0
                while chunk := response.read(8 * 1024 * 1024):
                    stream.write(chunk)
                    downloaded += len(chunk)
                    print(f"Downloaded {downloaded / 1024**2:.0f} MiB", flush=True)
        converted = tmp / "codec.pt"
        convert(source, converted, codec)
        # Publish only a verified, loadable file, without replacing existing files.
        os.link(converted, output)
    print(f"Verified and saved {output}", flush=True)


if __name__ == "__main__":
    with argbind.scope(argbind.parse_args()):
        run()
