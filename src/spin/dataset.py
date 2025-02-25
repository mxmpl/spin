from collections.abc import Iterator

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from spin.augment import BySpeakerDataAugmentation

SAMPLE_RATE = 16_000


class MaxLengthBatchSampler:
    def __init__(
        self, *, lengths: list[int], max_length: int, cropped_length: int, shuffle: bool, drop_last: bool, seed: int
    ) -> None:
        self.lengths = lengths
        self.max_length = max_length
        self.cropped_length = cropped_length
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator:
        batch_list = []
        batch = []
        cur_length = 0
        for i in range(len(self.lengths)):
            new_batch = [*batch, i]
            cur_length += min(self.lengths[i], self.cropped_length)
            if cur_length <= self.max_length:
                batch = new_batch
            elif len(batch) == 0:
                raise ValueError(
                    f"There is a single length {self.lengths[i]} larger than "
                    f"max_length {self.max_length}. Please increase "
                    "the max_length."
                )
            else:
                batch_list.append(batch)
                batch = [i]
                cur_length = min(self.lengths[i], self.cropped_length)

        if len(batch) > 0 and not self.drop_last:
            batch_list.append(batch)

        if self.shuffle:
            generator = torch.Generator().manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(batch_list), generator=generator).tolist()
        else:
            indices = list(range(len(batch_list)))
        for i in indices:
            yield batch_list[i]

    def __len__(self) -> int:
        return len(list(iter(self)))


class AudioPretrainDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        speaker_info: str,
        min_audio_len: int,
        max_audio_len: int,
        random_crop_len: int,
        seed: int,
        *,
        is_training: bool,
    ) -> None:
        super().__init__()
        self.augment = BySpeakerDataAugmentation(speaker_info, seed=seed)
        self.rng = np.random.default_rng(seed=seed)
        self.is_training = is_training
        self.random_crop_len = random_crop_len

        manifest = pd.read_csv(manifest_path)
        assert set(manifest.columns) == {"fileid", "path", "num_frames", "speaker"}
        manifest = manifest[(manifest["num_frames"] >= min_audio_len) & (manifest["num_frames"] <= max_audio_len)]
        manifest = manifest.sort_values(by="num_frames", ascending=False).reset_index()
        self.data = manifest[["path", "speaker"]].to_dict(orient="index")
        self.lengths = manifest["num_frames"].to_list()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> list[torch.FloatTensor]:
        path, spk = self.data[index]["path"], self.data[index]["speaker"]
        wav, sr = sf.read(path)
        assert sr == SAMPLE_RATE
        if wav.ndim == 2:  # (time, channel)
            wav = wav.mean(1)
        wav = wav.astype(np.float32)
        if self.random_crop_len > 0 and len(wav) > self.random_crop_len:
            idx = self.rng.randint(0, len(wav) - self.random_crop_len)
            wav = wav[idx : idx + self.random_crop_len]
        wavs = [wav, self.augment(wav, sr, str(spk), random=self.is_training)]
        wavs = [torch.FloatTensor(w) for w in wavs]
        return [F.layer_norm(w, w.shape) for w in wavs]


def collate_fn(batch: list[list[tuple[torch.FloatTensor, list[int]]]]) -> tuple[torch.Tensor, torch.Tensor]:
    wav_list, wav_len = [], []
    for wavs in batch:
        for _, w in enumerate(wavs):
            wav_list.append(w)
            wav_len.append(len(w))
    return pad_sequence(wav_list, batch_first=True), torch.LongTensor(wav_len)


def build_spin_dataloader(
    manifest: str,
    speaker_info: str,
    *,
    min_audio_len: int,
    max_audio_len: int,
    random_crop_len: int,
    num_workers: int,
    seed: int,
    is_training: bool,
) -> DataLoader:
    dataset = AudioPretrainDataset(
        manifest,
        speaker_info,
        min_audio_len,
        max_audio_len,
        random_crop_len,
        seed,
        is_training=is_training,
    )
    batch_sampler = MaxLengthBatchSampler(
        lengths=dataset.lengths,
        max_length=max_audio_len,
        cropped_length=random_crop_len,
        shuffle=is_training,
        drop_last=is_training,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        # persistent_workers=is_training,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
