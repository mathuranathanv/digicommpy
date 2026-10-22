"""One tap per frequency bin, once a cyclic prefix has made the channel circular."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def add_cyclic_prefix(block: NDArray, cp: int) -> NDArray:
    """Copy the last `cp` samples to the front of the block."""
    if cp < 0 or cp > block.size:
        raise ValueError("cyclic prefix length must lie between 0 and the block length")
    if cp == 0:
        return np.asarray(block, dtype=complex).copy()
    return np.concatenate([block[-cp:], block])


def through_channel(waveform: NDArray, h: NDArray, n_keep: int, cp: int) -> NDArray:
    """Linear convolution, then the samples that a long enough prefix makes circular."""
    y = np.convolve(np.asarray(waveform, dtype=complex), np.asarray(h, dtype=complex))
    return y[cp : cp + n_keep]


def mmse_fde(block: NDArray, h: NDArray, n0: float) -> NDArray:
    """Equalize one cyclic block. n0 = 0 is zero-forcing, one complex tap per bin."""
    H = np.fft.fft(np.asarray(h, dtype=complex), n=block.size)
    taps = np.conj(H) / (np.abs(H) ** 2 + n0)
    return np.fft.ifft(taps * np.fft.fft(block))
