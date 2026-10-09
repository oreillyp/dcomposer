"""Portable inference weights, without training state or a trained DOC decoder."""
from pathlib import Path

import torch

from .constants import ACOUSTIC_TOK_OFFSET
from .model import DComposer
from .retrieval import DOCEncoder, digest
from .util import load_config


def export_encoder(run_dir, checkpoint, weights, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    run_dir = Path(run_dir)
    config = load_config(run_dir / "conf.yml")
    prefix = "LatentFlowAutoencoder."
    model_config = {
        k[len(prefix) :]: v for k, v in config.items() if k.startswith(prefix)
    }
    if not model_config or config.get("model_name", "flow") != "flow":
        raise ValueError("Expected a saved LatentFlowAutoencoder run configuration")
    weights_path = run_dir / checkpoint / weights
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    encoder = DOCEncoder(**model_config)
    selected = {k: state[k].clone() for k in encoder.state_dict()}
    if any(not torch.isfinite(v).all() for v in selected.values()):
        raise ValueError("Non-finite DOC encoder weights")
    encoder.load_state_dict(selected, strict=True, assign=True)
    extras = torch.load(
        run_dir / checkpoint / "extras.pt", map_location="cpu", weights_only=False
    )
    tokenizer = {
        key: extras.get("tokenizer", {}).get(key, config.get("Tokenizer." + key))
        for key in (
            "name",
            "normalize_db",
            "loudness_exclude_silence",
            "scale_mean",
            "scale_std",
            "scale_per_channel",
        )
        if key in extras.get("tokenizer", {}) or "Tokenizer." + key in config
    }
    scale = {
        k: v
        for k, v in extras.get("tokenizer_scale", {}).items()
        if k in ("mean", "std", "per_channel")
    }
    for key in ("mean", "std"):
        value = scale.get(key)
        if tokenizer.get("scale_" + key, True) and value is None:
            raise ValueError(
                f"Missing saved tokenizer {key}; refusing a different encoding"
            )
        if value is not None and (
            not torch.isfinite(value).all() or (key == "std" and torch.any(value <= 0))
        ):
            raise ValueError(f"Invalid tokenizer {key}")
    duration = config.get("train/build_batch_config.max_audio_len")
    if duration is None or duration <= 0:
        raise ValueError("Missing saved training audio duration")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model_config,
            "state_dict": encoder.state_dict(),
            "tokenizer": tokenizer,
            "scale": scale,
            "duration": duration,
            "sample_rate": config["sample_rate"],
            "weights_sha256": digest(weights_path),
        },
        output,
    )
    print(f"Exported encoder only: {output}", flush=True)


def export_lm(
    run_dir,
    checkpoint,
    codec_run,
    codec_checkpoint,
    codec_weights,
    encoder_path,
    output,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    run_dir = Path(run_dir).resolve()
    args = load_config(run_dir / "conf.yml")
    saved_codec = Path(args["codec_run_dir"])
    if not saved_codec.is_absolute():
        saved_codec = run_dir.parent.parent / saved_codec
    if (
        saved_codec.resolve() != Path(codec_run).resolve()
        or args["codec_checkpoint"] != codec_checkpoint
        or ("ema_model.pt" if args["codec_use_ema"] else "model.pt") != codec_weights
    ):
        raise ValueError(
            "Selected DOC checkpoint does not match the saved LM configuration"
        )
    encoder = torch.load(encoder_path, map_location="cpu", weights_only=True)
    config = {
        k[len("DComposer.") :]: v for k, v in args.items() if k.startswith("DComposer.")
    }
    if args.get("OneShotMidiDataset.hierarchy_level") != "coarse":
        raise ValueError("The public release exports the coarse D-Composer only")
    doc = DOCEncoder(**encoder["config"])
    if (
        config["n_acoustic_tokens"] != doc.n_codebooks
        or config["n_vocab"] != ACOUSTIC_TOK_OFFSET + doc.n_vocab
    ):
        raise ValueError("LM and DOC vocabularies differ")
    if config["n_latent_channels"] != doc.n_channels_in:
        raise ValueError("LM and DOC base-VAE latent channels differ")
    source = run_dir / checkpoint / "model.pt"
    state = torch.load(source, map_location="cpu", weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Non-finite LM weights")
    with torch.device("meta"):
        model = DComposer(**config)
    model.load_state_dict(state, strict=True, assign=True)
    keys = (
        "top_p",
        "top_k",
        "temp",
        "argmax",
        "eos_threshold",
        "max_new_tokens",
        "render_max_duration",
        "autoencoder_n_steps",
        "autoencoder_cfg_weight",
        "OneShotMidiDataset.hierarchy_level",
        "train/build_builder_config.max_duration",
        "val/build_builder_config.max_duration",
        "codec_checkpoint",
        "codec_use_ema",
    )
    public_args = {
        k: v for k, v in args.items() if k.startswith("DComposer.") or k in keys
    }
    public_args["codec_encoder_sha256"] = digest(encoder_path)
    public_args["codec_weights_sha256"] = encoder["weights_sha256"]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "args": public_args,
            "state_dict": model.state_dict(),
            "weights_sha256": digest(source),
        },
        output,
    )
    print(f"Exported coarse LM: {output}", flush=True)
