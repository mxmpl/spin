import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import yaml
from torch import nn
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchaudio.pipelines import HUBERT_BASE
from tqdm import tqdm

from spin import Spin, build_spin_dataloader


@dataclass(frozen=True)
class Config:
    train_manifest: str
    val_manifests: dict[str, str]
    speaker_info: str
    save_path: str
    seed: int = 0

    log_interval: int = 100
    val_interval: int = 1000

    min_audio_len: int = 40_000
    max_audio_len: int = 1_000_000
    random_crop_len: int = 272_000
    num_workers: int = 8

    max_steps: int = 5000
    lr: float = 1e-4
    weight_decay: float = 1e-6
    max_norm: int = 10
    warmup_steps: int = 2500
    lr_start_factor: float = 0
    lr_end_factor: float = 0.01
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"

    codebook_dim: int = 256
    codebook_size: int = 2048
    epsilon: float = 0.02
    temperature: float = 0.1
    sinkhorn_iters: int = 3
    prob_ratio: float = 0.1

    exclude_from_backbone_freeze: list[str] = field(
        default_factory=lambda: [
            "encoder.transformer.layers.10",
            "encoder.transformer.layers.11",
        ],
    )


def dataloader(manifest: str, cfg: Config, *, is_training: bool) -> DataLoader:
    return build_spin_dataloader(
        manifest,
        cfg.speaker_info,
        min_audio_len=cfg.min_audio_len,
        max_audio_len=cfg.max_audio_len,
        random_crop_len=cfg.random_crop_len,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        is_training=is_training,
    )


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float | torch.Tensor, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def flush(self) -> float:
        avg = self.avg
        if isinstance(avg, torch.Tensor):
            avg = avg.item()
        self.reset()
        return avg


class ModelWrapper(nn.Module):
    def __init__(self, cfg: Config) -> None:
        self.backbone = HUBERT_BASE.get_model()
        inp_dim = self.backbone.encoder.feature_projection.out_features
        self.spin = Spin(
            inp_dim=inp_dim,
            codebook_dim=cfg.spin_codebook_dim,
            codebook_size=cfg.spin_codebook_size,
            epsilon=cfg.spin_epsilon,
            temperature=cfg.spin_temperature,
            sinkhorn_iters=cfg.spin_sinkhorn_iters,
            prob_ratio=cfg.spin_prob_ratio,
        )
        self.freeze_backbone(cfg.exclude_from_backbone_freeze)

    def freeze_backbone(self, exclude: list[str]) -> None:
        self.backbone.eval()
        for name, param in self.backbone.named_parameters():
            if not any(name.startswith(n) for n in exclude):
                param.requires_grad_(requires_grad=False)

    def forward(self, wav: torch.Tensor, wav_len: torch.Tensor) -> torch.Tensor:
        feats, _ = self.backbone(wav, wav_len)
        return self.spin(feats)


def validate(model: ModelWrapper, loaders: dict[str, DataLoader], device: torch.device) -> dict[str, float]:
    model.eval()
    losses = {}
    for name, loader in loaders.items():
        loss, n = torch.zeros(1, device=device, dtype=torch.float32), 0
        with torch.no_grad():
            for wav, wav_len in loader:
                loss += model(wav, wav_len).mean()
                n += 1
        losses[name] = (loss / n).item()
    model.train()
    return losses


def main(cfg: Config) -> None:
    device = torch.device("cuda")
    dtype = getattr(torch, cfg.dtype)
    mixed_precision = cfg.dtype != "float32"

    model = ModelWrapper(cfg).to(device)
    opt = Adam(
        tuple(p for p in model.parameters() if p.requires_grad),
        lr=cfg.opt_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = SequentialLR(
        opt,
        [
            LinearLR(opt, start_factor=cfg.lr_start_factor, end_factor=1),
            LinearLR(opt, start_factor=1, end_factor=cfg.lr_end_factor),
        ],
        [cfg.warmup_steps],
    )
    scaler = GradScaler("cuda", enabled=mixed_precision)
    train_loader = dataloader(cfg.train_manifest, cfg, is_training=True)
    val_loaders = {k: dataloader(v, cfg, is_training=False) for k, v in cfg.val_manifests.items()}

    pbar = tqdm(total=cfg.max_steps, desc="Training")
    step, epoch, avg_loss = 0, 0, AverageMeter()
    while step < cfg.max_steps:
        train_loader.batch_sampler.set_epoch(epoch)
        for wav, wav_len in train_loader:
            with torch.autocast("cuda", dtype, mixed_precision):
                loss = model(wav.to(device), wav_len.to(device))
            loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            clip_grad_norm_(model.parameters(), cfg.max_norm)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            scheduler.step()

            avg_loss.update(loss.detach())
            pbar.update()
            step += 1
            if step % cfg.log_interval == 0:
                pbar.set_postfix(loss=avg_loss.flush())
            if step % cfg.val_interval == 0:
                val_losses = validate(model, val_loaders, device)
                val_losses_str = "\t".join(f"{k}: {v:.3f}" for k, v in val_losses.items())
                print(f"Validation loss at step {step}\t" + val_losses_str)
            if step >= cfg.max_steps:
                break
        epoch += 1

    torch.save(model.state_dict(), cfg.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Spin on top of HuBERT base")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    with args.open() as f:
        cfg = yaml.safe_load(f)
    main(Config(**cfg))
