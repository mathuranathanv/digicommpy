"""Magnitude, phase, Parseval, convolution, and the analytic signal.

DFT convention is the one locked in spectra.py: 1/N on the inverse.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, ifft


def analytic_signal(x: NDArray) -> NDArray[np.complexfloating]:
    """Discrete analytic signal. Negative bins are zeroed, positive bins are doubled.

    DC is left alone. When N is even the Nyquist bin is left alone too:
    it has no partner to steal from, and doubling it would invent energy.
    """
    spectrum = fft(np.asarray(x, dtype=float))
    n = spectrum.size
    spectrum[1 : (n + 1) // 2] *= 2
    spectrum[(n + 1) // 2 :] = 0
    if n % 2 == 0:
        # The slice above already cleared past Nyquist and left the Nyquist bin.
        # Put Nyquist back: it was index n//2, which (n+1)//2 equals for even n,
        # so the zeroing started AT Nyquist. Restore it from a fresh transform.
        full = fft(np.asarray(x, dtype=float))
        spectrum[n // 2] = full[n // 2]
    return ifft(spectrum)


def parseval_energy(x: NDArray) -> tuple[float, float]:
    """Time energy and (1/N) sum |X[k]|^2. They match for this DFT."""
    x = np.asarray(x)
    spectrum = fft(x)
    time = float(np.sum(np.abs(x) ** 2))
    freq = float(np.sum(np.abs(spectrum) ** 2) / x.size)
    return time, freq
