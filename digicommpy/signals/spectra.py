"""FFT helpers for Chapter 1 §1.2.

Computational convention locked for this book (SciPy / NumPy / Matlab):

    X[k] = sum_{n=0}^{N-1} x[n] exp(-j*2*pi*k*n/N)     # fft
    x[n] = (1/N) sum_{k=0}^{N-1} X[k] exp(+j*2*pi*k*n/N)  # ifft

A real cosine of amplitude A, exactly on bin k (0 < k < N/2), produces
|X[k]| = |X[N-k]| = A*N/2. Recover A as 2*|X[k]| / sum(w), where w is
the analysis window (w = 1 for a rectangular window, sum(w) = N).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fftfreq, fftshift
from scipy.signal.windows import hann


def bin_frequencies(n: int, fs: float, *, shifted: bool = False) -> NDArray[np.floating]:
    """Bin centers in hertz.

    Even N, shifted: -fs/2, ..., fs/2 - fs/N.
    Odd N, shifted: -(N-1)/2 * fs/N, ..., +(N-1)/2 * fs/N.
    There is a single Nyquist bin only when N is even.
    """
    f = fftfreq(n, d=1.0 / fs)
    return fftshift(f) if shifted else f


def periodic_hann(n: int) -> NDArray[np.floating]:
    """Periodic Hann window for spectral analysis (not the symmetric FIR window).

    w[n] = 0.5 - 0.5*cos(2*pi*n/N). Coherent gain is 1/2; ENBW is 1.5 bins.
    """
    return hann(n, sym=False)


def window_metrics(w: NDArray[np.floating]) -> tuple[float, float]:
    """Return (coherent gain, ENBW in bins)."""
    coherent_gain = float(np.sum(w) / w.size)
    enbw_bins = float(w.size * np.sum(w**2) / (np.sum(w) ** 2))
    return coherent_gain, enbw_bins


def cosine_amplitude(
    spectrum: NDArray[np.complexfloating],
    k: int,
    window_sum: float,
) -> float:
    """Amplitude of a real cosine from one positive-frequency bin.

    Uses 2*|X[k]|/sum(w). Do not use this at k = 0 or at the Nyquist bin.
    """
    if k <= 0 or k >= spectrum.size // 2:
        raise ValueError("k must be a positive-frequency bin strictly below Nyquist")
    return float(2 * np.abs(spectrum[k]) / window_sum)
