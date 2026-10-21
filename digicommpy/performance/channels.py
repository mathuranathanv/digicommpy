"""AWGN and flat fading at one sample per symbol.

The number you pass is Es/N0 in dB, measured on the vector you pass in.
Eb/N0 is Es/N0 minus 10 log10(k). Do not hide that shift inside this function.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def add_awgn(symbols: NDArray, esn0_db: float, rng: np.random.Generator) -> NDArray:
    """Add white Gaussian noise at a commanded symbol SNR.

    Complex symbols get independent I and Q noise, each of variance N0/2,
    so E[|n|^2] = N0 = Es / (Es/N0). Real symbols get variance N0/2.
    """
    es = float(np.mean(np.abs(symbols) ** 2))
    esn0 = 10 ** (esn0_db / 10)
    n0 = es / esn0
    sigma = np.sqrt(n0 / 2)
    noise = sigma * (rng.normal(size=symbols.shape) + 1j * rng.normal(size=symbols.shape))
    if np.isrealobj(symbols):
        noise = noise.real
    return symbols + noise


def rayleigh(n: int, rng: np.random.Generator) -> NDArray[np.complexfloating]:
    """Complex Rayleigh fade with E[|h|^2] = 1. The phase is kept."""
    return (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)


def rician(n: int, K: float, rng: np.random.Generator) -> NDArray[np.complexfloating]:
    """Complex Rician fade, E[|h|^2] = 1. K is the linear LOS-to-scatter ratio.

    K = 0 is Rayleigh. The line-of-sight term sits on the real axis; a
    common phase rotation does not change the error rate after derotation.
    """
    if K < 0:
        raise ValueError("K must be nonnegative")
    sigma = np.sqrt(1 / (2 * (K + 1)))
    los = np.sqrt(K / (K + 1))
    return los + sigma * (rng.normal(size=n) + 1j * rng.normal(size=n))


def derotate(received: NDArray, h: NDArray) -> NDArray:
    """Coherent matched filter for a known flat fade: y = h* r / |h|^2."""
    return received * np.conj(h) / (np.abs(h) ** 2)
