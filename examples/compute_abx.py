import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from fastabx import zerospeech_abx
from torch import nn
from torchaudio.pipelines import HUBERT_BASE
from tqdm import tqdm

from spin import Spin


class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = HUBERT_BASE.get_model()
        self.spin = Spin(inp_dim=self.backbone.encoder.feature_projection.projection.out_features)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim != 2:
            raise ValueError("Input should be 2D")
        features = []
        y, _ = self.backbone.extract_features(x)
        features += y[-2:]  # Last two layers, to update if needed
        x = self.spin.head(y[-1])
        features.append(x)
        x = F.normalize(x, p=2, dim=1)
        x = self.spin.codebook(x)
        features.append(x)
        x = F.log_softmax(x / self.spin.temperature, dim=1)
        features.append(x)
        return features


def extract_features(checkpoint: str, wav: Path, dest: Path, extension: str) -> None:
    offset = 11
    dest.mkdir(exist_ok=True)
    device = torch.device("cuda")
    model = FeatureExtractor().eval().to(device)
    model.load_state_dict(torch.load(checkpoint))
    targets = []
    for path in tqdm(list(wav.rglob(f"*{extension}"))):
        x, sr = torchaudio.load(str(path))
        assert sr == 16_000
        features = model(x.to(device))
        for i, feats in enumerate(features):
            target = dest / str(offset + i)
            if not target.is_dir():
                targets.append(target)
                target.mkdir()
            torch.save(feats.squeeze().cpu(), target / f"{path.stem}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("wav", type=str)
    parser.add_argument("item", type=str)
    parser.add_argument("--dest", default=os.getenv("JOBSCRATCH"))
    parser.add_argument("--extension", type=str, default=".flac")
    parser.add_argument("--speaker", choices=["within", "across"], default="within")
    args = parser.parse_args()

    extract_features(args.checkpoint, Path(args.wav), Path(args.dest), args.extension)

    layers = [11, 12, 13, 14, 15]  # To update
    for layer in layers:
        distance = "kl_symmetric" if layer == layers[-1] else "cosine"
        score = zerospeech_abx(
            args.item,
            Path(args.dest) / str(layer),
            distance=distance,
            speaker=args.speaker,
            max_size_group=30,
        )
        print(f"{layer}\t{score:.4%}")
