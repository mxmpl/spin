import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import wandb
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

    wandb_project: str
    wandb_name: str
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"

    seed: int = 0
    log_interval: int = 100
    val_interval: int = 1000

    batch_len: int = 4_096_000
    min_audio_len: int = 40_000
    max_audio_len: int = 1_000_000
    random_crop_len: int = 272_000
    num_workers: int = 10

    max_steps: int = 5000
    lr: float = 1e-4
    weight_decay: float = 1e-6
    max_norm: int = 10
    warmup_steps: int = 2500
    lr_start_factor: float = 1e-12
    lr_end_factor: float = 0.01
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"

    codebook_dim: int = 256
    codebook_size: int = 2048
    epsilon: float = 0.02
    temperature: float = 0.1
    sinkhorn_iters: int = 3

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
        batch_len=cfg.batch_len,
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
        super().__init__()
        self.backbone = HUBERT_BASE.get_model()
        self.spin = Spin(
            inp_dim=self.backbone.encoder.feature_projection.projection.out_features,
            codebook_dim=cfg.codebook_dim,
            codebook_size=cfg.codebook_size,
            epsilon=cfg.epsilon,
            temperature=cfg.temperature,
            sinkhorn_iters=cfg.sinkhorn_iters,
        )
        self.freeze_backbone(cfg.exclude_from_backbone_freeze)

    def freeze_backbone(self, exclude: list[str]) -> None:
        for name, param in self.backbone.named_parameters():
            if not any(name.startswith(n) for n in exclude):
                param.requires_grad_(requires_grad=False)

    def forward(self, wav: torch.Tensor, wav_len: torch.Tensor) -> torch.Tensor:
        x, _ = self.backbone(wav, wav_len)
        return self.spin(x)


def validate(model: ModelWrapper, loaders: dict[str, DataLoader], device: torch.device) -> dict[str, float]:
    model.eval()
    losses = {}
    for name, loader in loaders.items():
        loss = torch.zeros(1, device=device, dtype=torch.float32)
        with torch.no_grad():
            for wav, wav_len in loader:
                loss += model(wav.to(device), wav_len.to(device)).mean()
        losses[name] = (loss / len(loader)).item()
    model.train()
    return losses


def main(cfg: Config) -> None:
    save_path = Path(cfg.save_path)
    save_path.mkdir(exist_ok=True)
    wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, mode=cfg.wandb_mode, dir=save_path, config=cfg)

    device = torch.device("cuda")
    dtype = getattr(torch, cfg.dtype)
    mixed_precision = cfg.dtype != "float32"

    model = ModelWrapper(cfg).train().to(device)
    opt = Adam(tuple(p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = SequentialLR(
        opt,
        [
            LinearLR(opt, start_factor=cfg.lr_start_factor, end_factor=1, total_iters=cfg.warmup_steps),
            LinearLR(opt, start_factor=1, end_factor=cfg.lr_end_factor, total_iters=cfg.max_steps - cfg.warmup_steps),
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
                lr = scheduler.get_last_lr()[0]  # Maybe shifted by 1 step but not important
                infos = {"epoch": epoch, "batch_size": wav.size(0), "loss": avg_loss.flush(), "lr": lr}
                wandb.log({f"train/{key}": val for key, val in infos.items()}, step=step)
                pbar.set_postfix(loss=infos["loss"])
            if step % cfg.val_interval == 0:
                pbar.set_description("Validation ongoing...")
                val_losses = validate(model, val_loaders, device)
                wandb.log({f"{key}/loss: {val}": val for key, val in val_losses.items()}, step=step)
                pbar.set_description("Training")
            if step >= cfg.max_steps:
                break
        epoch += 1

    torch.save(model.state_dict(), save_path / f"{cfg.wandb_name}.pt")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Spin on top of HuBERT base")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    main(Config(**cfg))
