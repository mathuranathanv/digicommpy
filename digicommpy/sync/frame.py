"""Packet start from a known preamble."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalized_correlation(received: NDArray, preamble: NDArray) -> NDArray[np.floating]:
    """Sliding correlation in [0, 1]. A peak of 1 means the window is a scaled copy of the preamble.

    Normalization by the window energy is what makes one threshold work at every received power.
    """
    preamble = np.asarray(preamble, dtype=complex)
    received = np.asarray(received, dtype=complex)
    lp = preamble.size
    num = np.correlate(received, preamble, mode="valid")
    energy = np.convolve(np.abs(received) ** 2, np.ones(lp), mode="valid")
    denom = np.sqrt(energy * float(np.sum(np.abs(preamble) ** 2)))
    return np.abs(num) / np.maximum(denom, 1e-12)
