"""Dataset wrappers for Slice 5 (baseline reproduction).

T5.1 ships `StubCSI` for end-to-end pipeline plumbing on random tensors —
same shape convention as Slice 1 so encoder code transfers directly. T5.2
will add the real Widar3.0 cross-subject loader.

Convention: each sample is a real-valued CSI tensor shaped `(T, S, A)` where
    T = time samples
    S = subcarriers (30 for Intel 5300)
    A = antenna pairs (3 for the typical Widar3.0 1-TX × 3-RX setup)
plus a single integer gesture label in `[0, NUM_CLASSES)`.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

CSI_T = 100
CSI_S = 30
CSI_A = 3
NUM_CLASSES = 6


class StubCSI(Dataset):
    """Random CSI samples — pipeline plumbing only, no real signal.

    Identical interface to Slice 1's `StubCSI` so the scaffold ports cleanly.
    """

    def __init__(
        self,
        num_samples: int = 10,
        time_steps: int = CSI_T,
        subcarriers: int = CSI_S,
        antenna_pairs: int = CSI_A,
        num_classes: int = NUM_CLASSES,
        seed: int = 0,
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        self._x = torch.randn(
            num_samples, time_steps, subcarriers, antenna_pairs, generator=g
        )
        self._y = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self._x[i], int(self._y[i].item())
