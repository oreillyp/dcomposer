"""
Adapted from https://github.com/descriptinc/descript-audio-codec/blob/main/scripts/train.py
"""
import copy
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
from audiotools.metrics.distance import L1Loss
from audiotools.metrics.spectral import MelSpectrogramLoss
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


@torch.no_grad()
def assert_tokenizer_length_matches_model_max_len(
    args,
    tokenizer,
    model,
    device,
):
    sample_rate = int(args["OneShotDataset.sample_rate"])
    num_channels = int(args["OneShotDataset.num_channels"])
    max_audio_len = float(
        args.get(
            "sample/build_batch_config.max_audio_len",
            args["train/build_batch_config.max_audio_len"],
        )
    )
    n_samples = int(max_audio_len * sample_rate)
    signal = AudioSignal(
        torch.zeros(1, num_channels, n_samples, device=torch.device(device)),
        sample_rate=sample_rate,
    )
    seq = tokenizer.encode(signal, scale=False, no_grad=True)
    seq_len = int(seq.tokens.shape[-1])
    max_len = int(model.max_len)
    if seq_len != max_len:
        raise ValueError(
            "Tokenizer/model length mismatch: "
            f"max_audio_len={max_audio_len:.3f}s at sample_rate={sample_rate} "
            f"produced latent length {seq_len}, but {type(model).__name__}.max_len={max_len}. "
            "Update the config so these values agree."
        )


_path = Path(__file__).parent.parent
sys.path.insert(0, str(_path))

with chdir(_path):
    from dcomposer.data.oneshot_dataset import OneShotDataset
    from dcomposer.data.oneshot_dataset import build_oneshot_batch
    from dcomposer.model.flow_autoencoder import LatentFlowAutoencoder
    from dcomposer.model.flow_autoencoder import flow
    from dcomposer.nn.lr_schedulers import LinearWarmupCosineDecayLR
    from dcomposer.pipelines.tokenizer import TokenSequence
    from dcomposer.pipelines.tokenizer import Tokenizer
    from dcomposer import transforms
    from dcomposer.util import count_parameters, print, ensure_dir

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
LatentFlowAutoencoder = argbind.bind(LatentFlowAutoencoder)
Tokenizer = argbind.bind(Tokenizer)
OneShotDataset = argbind.bind(OneShotDataset, "train", "val")


filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "NormalizedBaseTransform",
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(
    transforms,
    "train",
    "val",
    filter_fn=filter_fn,
)


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train", "val")
def build_transform(
    prob: float = 1.0,
    names: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    transform = transforms.Compose(*to_tfm(names), prob=prob)
    return transform


@argbind.bind("train", "val", "sample")
def build_batch_config(
    max_audio_len: float = 3.0,
    out_batch_size: int = None,
    p_mixup_intra_class: float = 0.0,
    p_mixup_inter_class: float = 0.0,
    alpha: float = 1.0,
):
    return {
        "max_audio_len": float(max_audio_len),
        "out_batch_size": out_batch_size,
        "p_mixup_intra_class": float(p_mixup_intra_class),
        "p_mixup_inter_class": float(p_mixup_inter_class),
        "alpha": float(alpha),
    }


@argbind.bind(without_prefix=True)
def build_scale_config(
    init_from_first_batch: bool = False,
):
    return {
        "init_from_first_batch": bool(init_from_first_batch),
    }


@argbind.bind(without_prefix=True)
def build_ema_config(
    use_ema: bool = False,
    ema_decay: float = 0.9999,
    ema_device: str = "gpu",
):
    assert ema_device in ["gpu", "cpu"]
    return {
        "use_ema": bool(use_ema),
        "ema_decay": float(ema_decay),
        "ema_device": ema_device,
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
def build_audio_metric_config(
    audio_metric_freq: int = 0,
    audio_metric_num_batches: int = 0,
    audio_metric_max_examples: int = None,
):
    return {
        "freq": int(audio_metric_freq),
        "num_batches": int(audio_metric_num_batches),
        "max_examples": audio_metric_max_examples,
    }


@argbind.bind(without_prefix=True)
def build_loss_config(
    commitment_loss_weight: float = 0.25,
    codebook_loss_weight: float = 1.0,
):
    return {
        "commitment_loss_weight": float(commitment_loss_weight),
        "codebook_loss_weight": float(codebook_loss_weight),
    }


@argbind.bind(without_prefix=True)
def build_sample_config(
    n_steps: int = 100,
    t_init: list = [0.0, 0.25],
    solver: str = "euler",
    t_schedule: str = "linear",
    t_schedule_scale: float = 1.0,
    cfg_weight: float = None,
):
    return {
        "n_steps": int(n_steps),
        "t_init": [float(x) for x in t_init],
        "solver": solver,
        "t_schedule": t_schedule,
        "t_schedule_scale": float(t_schedule_scale),
        "cfg_weight": cfg_weight,
    }


@argbind.bind("train", "val")
def build_timestep_config(
    mode: str = "power",
    power: float = 2.0,
    t_min: float = 0.0,
    t_max: float = 0.999,
):
    return {
        "mode": mode,
        "power": float(power),
        "t_min": float(t_min),
        "t_max": float(t_max),
    }


def sample_t(
    n_batch: int,
    device: torch.device,
    cfg: Dict,
):
    t_min = float(cfg["t_min"])
    t_max = float(cfg["t_max"])
    assert 0.0 <= t_min <= t_max <= 1.0

    t = torch.rand(n_batch, dtype=torch.float32, device=device)

    if cfg["mode"] == "uniform":
        pass
    elif cfg["mode"] == "power":
        t = t.pow(float(cfg["power"]))
    else:
        raise ValueError(f"Unknown timestep mode: {cfg['mode']}")

    t = t_min + (t_max - t_min) * t
    return t.clamp(max=t_max)


def _assert_valid_audio(signal, name: str):
    x = signal.audio_data
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name}: non-finite audio detected")
    if x.abs().amax().item() > 1.0 + 1e-6:
        raise RuntimeError(f"{name}: audio exceeded valid range [-1, 1]")


def _pad_or_trim_audio(signal, signal_lengths, max_audio_len: float):
    n_samples = int(max_audio_len * signal.sample_rate)
    signal_lengths = signal_lengths.clamp(max=n_samples)
    signal.audio_data = signal.audio_data[..., :n_samples]
    signal.audio_data = F.pad(
        signal.audio_data,
        (0, max(0, n_samples - signal.signal_length)),
    )
    return signal, signal_lengths


@torch.no_grad()
def prepare_signal_batch(
    batch,
    accel,
    transform: Callable,
    batch_cfg: Dict,
    state: int,
):
    batch = util.prepare_batch(batch, accel.device)
    batch = build_oneshot_batch(
        batch,
        out_batch_size=batch_cfg["out_batch_size"],
        class_weights=None,
        p_mixup_intra_class=batch_cfg["p_mixup_intra_class"],
        p_mixup_inter_class=batch_cfg["p_mixup_inter_class"],
        alpha=batch_cfg["alpha"],
        state=state,
    )
    _assert_valid_audio(batch["signal"], "post-mixing")

    idx = batch["idx"]
    signal = batch["signal"]
    signal_lengths = batch["signal_lengths"]

    pad_amt = 1.0 - (
        signal_lengths.float().sum()
        / float(
            signal.batch_size * int(batch_cfg["max_audio_len"] * signal.sample_rate)
        )
    )
    pad_amt = float(pad_amt.clamp(0.0, 1.0).item())

    signal, signal_lengths = _pad_or_trim_audio(
        signal, signal_lengths, batch_cfg["max_audio_len"]
    )

    if transform is not None:
        kwargs = transform.batch_instantiate(idx.tolist(), signal.clone())
        signal = transform.transform(signal.clone(), **kwargs)

    _assert_valid_audio(signal, "post-transform")

    return {
        "idx": idx,
        "signal": signal,
        "signal_lengths": signal_lengths,
        "pad_amt": pad_amt,
    }


@torch.no_grad()
def encode_latents(
    tokenizer: Tokenizer,
    signal,
    signal_lengths: torch.Tensor,
    max_len: int,
    scale: bool = True,
):
    seq = tokenizer.encode(signal.clone(), scale=scale, no_grad=True)
    latents = seq.tokens.detach()
    n_batch, n_channels, n_frames = latents.shape

    latents_lengths = (
        n_frames * signal_lengths.clone().float() / float(signal.signal_length)
    ).long()
    latents_lengths = latents_lengths.clamp(min=0, max=max_len)

    latents = latents[..., :max_len]
    latents = F.pad(latents, (0, max(0, max_len - latents.shape[-1])))

    assert latents.shape == (n_batch, n_channels, max_len)
    return seq, latents, latents_lengths


@torch.no_grad()
def maybe_init_tokenizer_scale(
    state,
    signal,
    scale_cfg: Dict,
):
    if not scale_cfg["init_from_first_batch"]:
        return

    if state.scale_initialized:
        return

    seq = state.tokenizer.encode(signal.clone(), scale=False, no_grad=True)
    latents = seq.tokens.detach()

    if state.tokenizer.scale_mean_enabled:
        mean = latents.mean(dim=(0, 2))
        if not state.tokenizer.scale_per_channel:
            mean = mean.mean()
    else:
        mean = (
            torch.zeros(latents.shape[1], device=latents.device)
            if state.tokenizer.scale_per_channel
            else 0.0
        )

    if state.tokenizer.scale_std_enabled:
        std = latents.std(dim=(0, 2), unbiased=False)
        if not state.tokenizer.scale_per_channel:
            std = std.mean()
    else:
        std = (
            torch.ones(latents.shape[1], device=latents.device)
            if state.tokenizer.scale_per_channel
            else 1.0
        )

    state.tokenizer.set_scale(
        mean=mean,
        std=std,
        per_channel=state.tokenizer.scale_per_channel,
    )
    state.scale_initialized = True
    state.tracker.print("Initialized tokenizer scale statistics from first batch")


def build_scheduler(optimizer, cfg: Dict):
    if cfg["scheduler_name"] == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg["gamma"])
    return LinearWarmupCosineDecayLR(
        optimizer,
        warmup_steps=cfg["warmup_steps"],
        decay_steps=cfg["decay_steps"],
        min_scale=cfg["min_scale"],
    )


@torch.no_grad()
def summarize_codes(codes: torch.Tensor, n_vocab: int, device: torch.device):
    p = torch.bincount(codes.reshape(-1), minlength=n_vocab).float()
    p = p / p.sum().clamp_min(1.0)
    entropy = -(p * p.clamp_min(1e-12).log()).sum()
    usage_frac = (p > 0).float().mean()
    return {
        "codes/entropy": entropy.to(device),
        "codes/usage_frac": usage_frac.to(device),
    }


@torch.no_grad()
def summarize_code_hist(code_hist: torch.Tensor):
    p = code_hist.to(torch.float32)
    p = p / p.sum().clamp_min(1.0)
    entropy = -(p * p.clamp_min(1e-12).log()).sum()
    usage_frac = (p > 0).float().mean()
    return {
        "codes/entropy": entropy,
        "codes/usage_frac": usage_frac,
    }


@torch.no_grad()
def update_ema(ema_model, model, decay: float):
    one_minus_decay = 1.0 - float(decay)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.lerp_(
            param.detach().to(device=ema_param.device, dtype=ema_param.dtype),
            one_minus_decay,
        )
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(
            buffer.detach().to(device=ema_buffer.device, dtype=ema_buffer.dtype)
        )


@contextmanager
def use_eval_model(state, accel):
    if state.ema_model is None:
        model = accel.unwrap(state.model)
        model.eval()
        yield model
        return

    moved = False
    if state.ema_cfg["ema_device"] == "cpu":
        state.ema_model = state.ema_model.to(str(accel.device))
        moved = True

    state.ema_model.eval()
    try:
        yield state.ema_model
    finally:
        if moved:
            state.ema_model = state.ema_model.to("cpu")


def _iter_params(*items):
    for item in items:
        if item is None:
            continue
        if isinstance(item, torch.nn.Parameter):
            yield item
        elif isinstance(item, torch.nn.Module):
            yield from item.parameters()
        elif isinstance(item, (list, tuple)):
            yield from _iter_params(*item)


def _grad_norm(params, device):
    sq_sum = None
    for param in params:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        val = grad.detach().float().pow(2).sum()
        sq_sum = val if sq_sum is None else (sq_sum + val)
    if sq_sum is None:
        return torch.zeros((), device=device)
    return torch.sqrt(sq_sum).to(device=device)


def _log_module_grad_norms(output, model, device):
    output["other/grad_norm_encoder"] = _grad_norm(
        _iter_params(
            model.in_proj_enc,
            model.out_proj_enc,
            model.register_norm_pre,
            model.register_norm_post,
            model.encoder,
            model.registers,
        ),
        device=device,
    )
    output["other/grad_norm_bottleneck"] = _grad_norm(
        _iter_params(model.bottleneck),
        device=device,
    )
    if model.upsample:
        output["other/grad_norm_upsampler"] = _grad_norm(
            _iter_params(
                model.in_proj_up_summary,
                model.upsampler,
                model.up_registers,
                getattr(model, "summary_drop_emb", None),
            ),
            device=device,
        )
    output["other/grad_norm_decoder"] = _grad_norm(
        _iter_params(
            model.in_proj_dec,
            model.out_proj_dec,
            model.out_norm,
            model.decoder,
            model.adaln,
            model.t_emb,
            getattr(model, "uncond_emb", None),
        ),
        device=device,
    )


@dataclass
class State:
    model: LatentFlowAutoencoder
    ema_model: Optional[LatentFlowAutoencoder]
    optimizer: AdamW
    scheduler: torch.optim.lr_scheduler.LRScheduler
    tokenizer: Tokenizer
    train_tfm: transforms.Compose
    val_tfm: transforms.Compose
    train_data: OneShotDataset
    val_data: OneShotDataset
    train_batch_cfg: Dict
    val_batch_cfg: Dict
    sample_batch_cfg: Dict
    sample_cfg: Dict
    train_t_cfg: Dict
    val_t_cfg: Dict
    scale_cfg: Dict
    ema_cfg: Dict
    compile_cfg: Dict
    scheduler_cfg: Dict
    memory_cfg: Dict
    audio_metric_cfg: Dict
    loss_cfg: Dict
    tracker: Tracker
    scale_initialized: bool = False
    val_code_hist: Optional[torch.Tensor] = None
    last_audio_metric_step: Optional[int] = None


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
):
    model, extras = LatentFlowAutoencoder(), {}
    tracker.print(model)
    print(f"Trainable parameters: {count_parameters(model)}")

    tokenizer = Tokenizer().to(accel.device)
    with argbind.scope(args):
        ema_cfg = build_ema_config()
        compile_cfg = build_compile_config()
        scheduler_cfg = build_scheduler_config()
        scale_cfg = build_scale_config()
        memory_cfg = build_memory_config()
        audio_metric_cfg = build_audio_metric_config()
        loss_cfg = build_loss_config()
        sample_cfg = build_sample_config()

    assert_tokenizer_length_matches_model_max_len(args, tokenizer, model, accel.device)

    if resume:
        load_dir = f"{save_path}/{tag}"
        model_pth = Path(load_dir) / "model.pt"
        ema_model_pth = Path(load_dir) / "ema_model.pt"
        extras_pth = Path(load_dir) / "extras.pt"

        tracker.print(f"Resuming from {str(Path('.').absolute())}/{load_dir}")
        if model_pth.exists():
            sd = torch.load(model_pth, map_location="cpu")
            model.load_state_dict(sd)
        if extras_pth.exists():
            extras = torch.load(extras_pth, map_location="cpu", weights_only=False)

    ema_model = None
    if ema_cfg["use_ema"]:
        ema_model = copy.deepcopy(model)
        saved_ema = extras.get("ema", {})
        if resume and bool(saved_ema.get("enabled", False)) and ema_model_pth.exists():
            ema_sd = torch.load(ema_model_pth, map_location="cpu")
            ema_model.load_state_dict(ema_sd)
        ema_device = str(accel.device) if ema_cfg["ema_device"] == "gpu" else "cpu"
        ema_model = ema_model.to(ema_device)
        ema_model.requires_grad_(False)
        ema_model.eval()

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

    n_vocab = model.n_vocab
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

    scale_initialized = not scale_cfg["init_from_first_batch"]
    if "tokenizer_scale" in extras:
        scale = extras["tokenizer_scale"]
        tokenizer.set_scale(
            mean=scale.get("mean", None),
            std=scale.get("std", None),
            per_channel=bool(scale.get("per_channel", True)),
        )
        scale_initialized = bool(scale.get("initialized", True))

    with argbind.scope(args, "train"):
        train_data = OneShotDataset()
        train_tfm = build_transform()
        train_batch_cfg = build_batch_config()
        train_t_cfg = build_timestep_config()

    with argbind.scope(args, "val"):
        val_data = OneShotDataset()
        val_tfm = build_transform()
        val_batch_cfg = build_batch_config()
        val_t_cfg = build_timestep_config()

    with argbind.scope(args, "sample"):
        sample_batch_cfg = build_batch_config()

    return State(
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        train_tfm=train_tfm,
        val_tfm=val_tfm,
        train_data=train_data,
        val_data=val_data,
        train_batch_cfg=train_batch_cfg,
        val_batch_cfg=val_batch_cfg,
        sample_batch_cfg=sample_batch_cfg,
        sample_cfg=sample_cfg,
        train_t_cfg=train_t_cfg,
        val_t_cfg=val_t_cfg,
        scale_cfg=scale_cfg,
        ema_cfg=ema_cfg,
        compile_cfg=compile_cfg,
        scheduler_cfg=scheduler_cfg,
        memory_cfg=memory_cfg,
        audio_metric_cfg=audio_metric_cfg,
        loss_cfg=loss_cfg,
        tracker=tracker,
        scale_initialized=scale_initialized,
        val_code_hist=torch.zeros(
            n_vocab,
            device=torch.device(accel.device),
            dtype=torch.long,
        ),
        last_audio_metric_step=None,
    )


@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    output = {}

    state.model.eval()

    batch = prepare_signal_batch(
        batch=batch,
        accel=accel,
        transform=state.val_tfm,
        batch_cfg=state.val_batch_cfg,
        state=int(state.tracker.step),
    )
    maybe_init_tokenizer_scale(state, batch["signal"], state.scale_cfg)

    _seq, latents, _latents_lengths = encode_latents(
        tokenizer=state.tokenizer,
        signal=batch["signal"],
        signal_lengths=batch["signal_lengths"],
        max_len=accel.unwrap(state.model).max_len,
        scale=True,
    )

    n_batch = latents.shape[0]
    t = sample_t(n_batch, accel.device, state.val_t_cfg)

    model = (
        state.ema_model if state.ema_model is not None else accel.unwrap(state.model)
    )
    with accel.autocast():
        pred, target, q, _drop_quant, _uncond = model(latents, t=t)
        recon_loss = F.mse_loss(pred, target, reduction="none").mean()
        commitment_loss = q["commitment_loss"].mean()
        codebook_loss = q["codebook_loss"].mean()
        loss = (
            recon_loss
            + state.loss_cfg["commitment_loss_weight"] * commitment_loss
            + state.loss_cfg["codebook_loss_weight"] * codebook_loss
        )

    code_hist = torch.bincount(
        q["codes"].reshape(-1),
        minlength=accel.unwrap(state.model).n_vocab,
    )
    state.val_code_hist += code_hist.to(state.val_code_hist.device)

    output["loss/mse"] = recon_loss.detach()
    output["loss/total"] = loss.detach()
    output["loss/commitment"] = commitment_loss.detach()
    output["loss/codebook"] = codebook_loss.detach()
    output["latent/mean"] = latents.mean().detach()
    output["latent/std"] = latents.std(unbiased=False).detach()
    output["memory/pad_amt"] = torch.as_tensor(batch["pad_amt"], device=latents.device)

    return output


@timer()
def train_loop(state, batch, accel):
    state.model.train()
    output = {}
    device = torch.device(accel.device)
    if state.memory_cfg["log_gpu_memory"] and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    batch = prepare_signal_batch(
        batch=batch,
        accel=accel,
        transform=state.train_tfm,
        batch_cfg=state.train_batch_cfg,
        state=int(state.tracker.step),
    )
    maybe_init_tokenizer_scale(state, batch["signal"], state.scale_cfg)

    with torch.no_grad():
        _seq, latents, _latents_lengths = encode_latents(
            tokenizer=state.tokenizer,
            signal=batch["signal"],
            signal_lengths=batch["signal_lengths"],
            max_len=accel.unwrap(state.model).max_len,
            scale=True,
        )

    n_batch = latents.shape[0]
    t = sample_t(n_batch, accel.device, state.train_t_cfg)

    with accel.autocast():
        pred, target, q, drop_quant, uncond = state.model(latents, t=t)
        recon_loss = F.mse_loss(pred, target, reduction="none").mean()
        commitment_loss = q["commitment_loss"].mean()
        codebook_loss = q["codebook_loss"].mean()
        loss = (
            recon_loss
            + state.loss_cfg["commitment_loss_weight"] * commitment_loss
            + state.loss_cfg["codebook_loss_weight"] * codebook_loss
        )

    with torch.no_grad():
        output["loss/mse"] = recon_loss.detach()
        output["loss/total"] = loss.detach()
        output["loss/commitment"] = commitment_loss.detach()
        output["loss/codebook"] = codebook_loss.detach()
        output["latent/mean"] = latents.mean().detach()
        output["latent/std"] = latents.std(unbiased=False).detach()
        output["memory/pad_amt"] = torch.as_tensor(
            batch["pad_amt"], device=latents.device
        )
        output["other/drop_quant_frac"] = drop_quant.float().mean().detach()
        output["other/uncond_frac"] = uncond.float().mean().detach()

    state.optimizer.zero_grad()
    accel.backward(loss)

    accel.scaler.unscale_(state.optimizer)
    _log_module_grad_norms(
        output=output,
        model=accel.unwrap(state.model),
        device=latents.device,
    )
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.model.parameters(), 1.0
    ).detach()
    accel.step(state.optimizer)
    state.scheduler.step()
    accel.update()

    output["other/learning_rate"] = torch.as_tensor(
        state.optimizer.param_groups[0]["lr"], device=latents.device
    )
    output["other/batch_size"] = torch.as_tensor(
        batch["signal"].batch_size * accel.world_size, device=latents.device
    )
    if state.memory_cfg["log_gpu_memory"] and device.type == "cuda":
        denom = float(1024**3)
        output["memory/gpu_alloc_gb"] = torch.as_tensor(
            torch.cuda.memory_allocated(device) / denom, device=latents.device
        )
        output["memory/gpu_reserved_gb"] = torch.as_tensor(
            torch.cuda.memory_reserved(device) / denom, device=latents.device
        )
        output["memory/gpu_peak_alloc_gb"] = torch.as_tensor(
            torch.cuda.max_memory_allocated(device) / denom, device=latents.device
        )
    if state.ema_model is not None:
        update_ema(
            state.ema_model,
            accel.unwrap(state.model),
            decay=state.ema_cfg["ema_decay"],
        )

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "loss/mse"):
        state.tracker.print("Best model so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step}")

    for tag in tags:
        save_dir = f"{save_path}/{tag}"
        ensure_dir(save_dir)
        model_pth = Path(save_dir) / "model.pt"
        ema_model_pth = Path(save_dir) / "ema_model.pt"
        extras_pth = Path(save_dir) / "extras.pt"

        scale = state.tokenizer.get_scale()
        extras = {
            "optimizer": state.optimizer.state_dict(),
            "scheduler": state.scheduler.state_dict(),
            "tracker": state.tracker.state_dict(),
            "metadata": metadata,
            "tokenizer_scale": {
                "mean": scale["mean"],
                "std": scale["std"],
                "per_channel": state.tokenizer.scale_per_channel,
                "initialized": state.scale_initialized,
            },
            "tokenizer": {
                "name": state.tokenizer.name,
                "normalize_db": state.tokenizer.normalize_db,
                "loudness_exclude_silence": state.tokenizer.loudness_exclude_silence,
            },
            "ema": {
                "enabled": state.ema_model is not None,
                "decay": state.ema_cfg["ema_decay"],
                "device": state.ema_cfg["ema_device"],
            },
            "compile": state.compile_cfg,
            "scheduler_cfg": state.scheduler_cfg,
        }
        torch.save(accel.unwrap(state.model).state_dict(), model_pth)
        if state.ema_model is not None:
            torch.save(state.ema_model.state_dict(), ema_model_pth)
        elif ema_model_pth.exists():
            ema_model_pth.unlink()
        torch.save(extras, extras_pth)


@torch.no_grad()
@argbind.bind(without_prefix=True)
def save_samples(
    state,
    accel,
    sample_idx,
    writer,
):
    state.tracker.print("Saving audio samples to TensorBoard")
    n_steps = state.sample_cfg["n_steps"]
    t_init = state.sample_cfg["t_init"]
    solver = state.sample_cfg["solver"]
    t_schedule = state.sample_cfg["t_schedule"]
    t_schedule_scale = state.sample_cfg["t_schedule_scale"]
    cfg_weight = state.sample_cfg["cfg_weight"]

    samples = [state.val_data[idx] for idx in sample_idx]
    batch = state.val_data.collate(samples)
    batch = prepare_signal_batch(
        batch=batch,
        accel=accel,
        transform=None,
        batch_cfg=state.sample_batch_cfg,
        state=0,
    )
    maybe_init_tokenizer_scale(state, batch["signal"], state.scale_cfg)

    seq, latents, _latents_lengths = encode_latents(
        tokenizer=state.tokenizer,
        signal=batch["signal"],
        signal_lengths=batch["signal_lengths"],
        max_len=accel.unwrap(state.model).max_len,
        scale=True,
    )
    sample_device = latents.device
    n_batch, _n_channels, n_frames = seq.tokens.shape

    audio_dict = {}
    if state.tracker.step == 0:
        audio_dict["signal"] = batch["signal"].clone()
        audio_dict["signal"].audio_data = audio_dict["signal"].audio_data.detach()

    with use_eval_model(state, accel) as model:
        summary = model.encode(latents)
        _latents_prequant, _latents_quant, codes = model.quantize(summary)

        for value in t_init:
            t0 = torch.full((n_batch,), float(value), device=accel.device)
            x_t = None
            if value > 0.0:
                x_t, _target = flow(x_0=latents, t=t0)

            pred = model.inference(
                codes=codes,
                n_steps=n_steps,
                t_init=t0,
                x_t=x_t,
                cfg_weight=cfg_weight,
                solver=solver,
                t_schedule=t_schedule,
                t_schedule_scale=t_schedule_scale,
            )

            seq_out = seq.clone()
            seq_out.tokens = pred[..., :n_frames].detach()
            audio = state.tokenizer.decode(seq_out)
            audio.audio_data = audio.audio_data.detach()
            audio_dict[f"recon_t_{value:0.2f}"] = audio
            del pred, seq_out, audio

    for key, value in audio_dict.items():
        for n_batch_idx in range(value.batch_size):
            value[n_batch_idx].cpu().write_audio_to_tb(
                f"{key}/sample_{n_batch_idx}.wav", writer, state.tracker.step
            )

    del audio_dict, summary, codes, latents, seq, batch, samples
    if sample_device.type == "cuda":
        torch.cuda.empty_cache()


def log_audio_metric_step_sweep(
    writer,
    step: int,
    prefix: str,
    x_values,
    metric_curves: Dict[float, Dict[str, list]],
):
    if writer is None:
        return

    import matplotlib.pyplot as plt

    for metric_key, metric_label in [
        ("l1", "L1"),
        ("mel", "MelSpectrogramLoss"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 3))
        for value in sorted(metric_curves):
            ax.plot(
                x_values,
                metric_curves[float(value)][metric_key],
                marker="o",
                label=f"t_init={float(value):0.2f}",
            )
        ax.set_xlabel("Inference Steps")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} vs Inference Steps")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        writer.add_figure(f"{prefix}/{metric_key}_vs_n_steps", fig, step)
        plt.close(fig)


@torch.no_grad()
def evaluate_audio_metrics(state, val_dataloader, accel, writer):
    cfg = state.audio_metric_cfg
    if writer is None or cfg["freq"] <= 0 or cfg["num_batches"] <= 0:
        return
    state.tracker.print("Computing audio reconstruction metrics")
    n_seen_examples = 0

    with use_eval_model(state, accel) as model:
        l1_loss = L1Loss().to(str(accel.device))
        mel_loss = MelSpectrogramLoss().to(str(accel.device))
        n_steps = state.sample_cfg["n_steps"]
        t_init = state.sample_cfg["t_init"]
        solver = state.sample_cfg["solver"]
        t_schedule = state.sample_cfg["t_schedule"]
        t_schedule_scale = state.sample_cfg["t_schedule_scale"]
        cfg_weight = state.sample_cfg["cfg_weight"]
        plot_steps = sorted(
            {
                max(1, int(v))
                for v in [1, n_steps // 4, n_steps // 2, n_steps, n_steps * 2]
            }
        )
        totals = {
            float(value): {"l1": 0.0, "mel": 0.0, "n_examples": 0} for value in t_init
        }

        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= cfg["num_batches"]:
                break

            batch = prepare_signal_batch(
                batch=batch,
                accel=accel,
                transform=state.val_tfm,
                batch_cfg=state.val_batch_cfg,
                state=int(state.tracker.step),
            )
            maybe_init_tokenizer_scale(state, batch["signal"], state.scale_cfg)

            if cfg["max_examples"] is not None:
                remaining = int(cfg["max_examples"]) - n_seen_examples
                if remaining <= 0:
                    break
                if batch["signal"].batch_size > remaining:
                    batch["idx"] = batch["idx"][:remaining]
                    batch["signal"] = batch["signal"][:remaining]
                    batch["signal_lengths"] = batch["signal_lengths"][:remaining]

            should_log_plot = batch_idx == 0
            if should_log_plot:
                plot_curves = {float(value): {"l1": [], "mel": []} for value in t_init}

            seq, latents, _latents_lengths = encode_latents(
                tokenizer=state.tokenizer,
                signal=batch["signal"],
                signal_lengths=batch["signal_lengths"],
                max_len=accel.unwrap(state.model).max_len,
                scale=True,
            )

            n_batch, _n_channels, n_frames = seq.tokens.shape
            summary = model.encode(latents)
            _latents_prequant, _latents_quant, codes = model.quantize(summary)

            for value in t_init:
                t0 = torch.full(
                    (n_batch,),
                    float(value),
                    device=accel.device,
                    dtype=torch.float32,
                )
                x_t = None
                if value > 0.0:
                    x_t, _target = flow(x_0=latents, t=t0)

                pred = model.inference(
                    codes=codes,
                    n_steps=n_steps,
                    t_init=t0,
                    x_t=x_t,
                    cfg_weight=cfg_weight,
                    solver=solver,
                    t_schedule=t_schedule,
                    t_schedule_scale=t_schedule_scale,
                )

                seq_out = seq.clone()
                seq_out.tokens = pred[..., :n_frames].detach()
                audio = state.tokenizer.decode(seq_out)

                ref_signal = batch["signal"].clone()
                pred_signal = audio.clone().to(str(accel.device))
                n_samples = min(ref_signal.signal_length, pred_signal.signal_length)
                signal_lengths = (
                    batch["signal_lengths"].to(accel.device).clamp(max=n_samples).long()
                )
                ref_signal.audio_data = ref_signal.audio_data[..., :n_samples]
                pred_signal.audio_data = pred_signal.audio_data[..., :n_samples]
                idx = torch.arange(n_samples, device=accel.device)
                mask = idx.view(1, 1, -1) < signal_lengths.view(-1, 1, 1)
                ref_signal.audio_data = ref_signal.audio_data * mask.to(
                    ref_signal.audio_data.dtype
                )
                pred_signal.audio_data = pred_signal.audio_data * mask.to(
                    pred_signal.audio_data.dtype
                )

                batch_l1 = float(l1_loss(pred_signal, ref_signal).detach().cpu())
                batch_mel = float(mel_loss(pred_signal, ref_signal).detach().cpu())
                totals[float(value)]["l1"] += batch_l1 * n_batch
                totals[float(value)]["mel"] += batch_mel * n_batch
                totals[float(value)]["n_examples"] += n_batch

                if should_log_plot:
                    for plot_n_steps in plot_steps:
                        plot_pred = model.inference(
                            codes=codes,
                            n_steps=plot_n_steps,
                            t_init=t0,
                            x_t=x_t,
                            cfg_weight=cfg_weight,
                            solver=solver,
                            t_schedule=t_schedule,
                            t_schedule_scale=t_schedule_scale,
                        )
                        plot_seq = seq.clone()
                        plot_seq.tokens = plot_pred[..., :n_frames].detach()
                        plot_audio = state.tokenizer.decode(plot_seq)
                        plot_signal = plot_audio.clone().to(str(accel.device))
                        plot_signal.audio_data = plot_signal.audio_data[..., :n_samples]
                        plot_signal.audio_data = plot_signal.audio_data * mask.to(
                            plot_signal.audio_data.dtype
                        )
                        plot_curves[float(value)]["l1"].append(
                            float(l1_loss(plot_signal, ref_signal).detach().cpu())
                        )
                        plot_curves[float(value)]["mel"].append(
                            float(mel_loss(plot_signal, ref_signal).detach().cpu())
                        )
            n_seen_examples += n_batch

            if should_log_plot:
                log_audio_metric_step_sweep(
                    writer=writer,
                    step=state.tracker.step,
                    prefix="audio_metrics/plots",
                    x_values=plot_steps,
                    metric_curves=plot_curves,
                )

    for value, stats in totals.items():
        if stats["n_examples"] == 0:
            continue
        writer.add_scalar(
            f"audio_metrics/val_t_{value:0.2f}/l1",
            stats["l1"] / float(stats["n_examples"]),
            state.tracker.step,
        )
        writer.add_scalar(
            f"audio_metrics/val_t_{value:0.2f}/melspec",
            stats["mel"] / float(stats["n_examples"]),
            state.tracker.step,
        )


def validate(state, val_dataloader, accel):
    state.val_code_hist.zero_()
    with use_eval_model(state, accel):
        for batch in val_dataloader:
            val_loop(batch, state, accel)

    if (
        getattr(accel, "use_ddp", False)
        and dist.is_available()
        and dist.is_initialized()
    ):
        dist.all_reduce(state.val_code_hist, op=dist.ReduceOp.SUM)

    code_metrics = summarize_code_hist(state.val_code_hist)
    for key, value in code_metrics.items():
        scalar = float(value.detach().cpu())
        state.tracker.metrics["val"]["value"][key] = scalar
        state.tracker.metrics["val"]["mean"][key].update(scalar)
    if hasattr(state.optimizer, "consolidate_state_dict"):
        state.optimizer.consolidate_state_dict()
    return {}


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "runs/autoencoder",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    val_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    sample_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
):
    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    if accel.local_rank == 0:
        dump_resolved_config(args, save_path, model_name="flow")
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

    global train_loop, val_loop, validate, save_samples, evaluate_audio_metrics, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    evaluate_audio_metrics = when(lambda: accel.local_rank == 0)(evaluate_audio_metrics)
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
                if state.audio_metric_cfg["freq"] > 0 and (
                    last_iter
                    or state.last_audio_metric_step is None
                    or (tracker.step - state.last_audio_metric_step)
                    >= state.audio_metric_cfg["freq"]
                ):
                    evaluate_audio_metrics(state, val_dataloader, accel, writer)
                    state.last_audio_metric_step = int(tracker.step)
                checkpoint(state, save_iters, save_path)
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
