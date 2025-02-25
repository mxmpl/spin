import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.signal import sosfilt

from spin.external import change_gender, change_gender_f0, params2sos

type Audio = npt.NDArray[np.float32]


class BySpeakerDataAugmentation:
    def __init__(self, speaker_info: str | Path, seed: int = 0) -> None:
        with Path(speaker_info).open("r") as f:
            self.spk2info = json.load(f)
        self.rng = np.random.default_rng(seed=seed)
        self.Fc = np.exp(np.linspace(np.log(60), np.log(7600), 10))
        self.Qmin, self.Qmax = 2, 5

    def get_spk_info(self, spk: str) -> tuple[int, int]:
        lo, hi = self.spk2info[spk]
        if lo == 50:
            lo = 75
        if spk == "1447":
            lo, hi = 60, 400
        return lo, hi

    def random_eq(self, wav: Audio, sr: int) -> Audio:
        z = self.rng.uniform(0, 1, size=(10,))
        Q = self.Qmin * (self.Qmax / self.Qmin) ** z
        G = self.rng.uniform(-12, 12, size=(10,))
        sos = params2sos(G, self.Fc, Q, sr)
        return sosfilt(sos, wav)

    def random_formant_f0(self, wav: Audio, sr: int, spk: str) -> Audio:
        lo, hi = self.get_spk_info(spk)

        ratio_fs = self.rng.uniform(1, 1.4)
        coin = self.rng.random() > 0.5
        ratio_fs = coin * ratio_fs + (1 - coin) * (1 / ratio_fs)

        ratio_ps = self.rng.uniform(1, 2)
        coin = self.rng.random() > 0.5
        ratio_ps = coin * ratio_ps + (1 - coin) * (1 / ratio_ps)

        ratio_pr = self.rng.uniform(1, 1.5)
        coin = self.rng.random() > 0.5
        ratio_pr = coin * ratio_pr + (1 - coin) * (1 / ratio_pr)

        return change_gender(wav, sr, lo, hi, ratio_fs, ratio_ps, ratio_pr)

    def fixed_formant_f0(self, wav: Audio, sr: int, spk: str) -> Audio:
        lo, hi = self.spk2info[spk]
        if lo == 50:
            lo = 75
            ratio_fs, f0_med, ratio_pr = 1.2, 300, 1.2
        else:
            ratio_fs, f0_med, ratio_pr = 0.8, 100, 0.8
        return change_gender_f0(wav, sr, lo, hi, ratio_fs, f0_med, ratio_pr)

    def __call__(self, wav: Audio, sr: int, spk: str, *, random: bool) -> Audio:
        if random:
            wav = self.random_formant_f0(wav, sr, spk)
            wav = self.random_eq(wav, sr)
        else:
            wav = self.fixed_formant_f0(wav, sr, spk)
        return np.clip(wav, -1.0, 1.0)
