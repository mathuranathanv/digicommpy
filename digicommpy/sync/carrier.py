"""Carrier frequency and phase, after timing has produced one sample per symbol."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def decide_qpsk(sample: complex) -> complex:
    """Nearest axis-aligned unit-energy QPSK point: ±1 or ±j."""
    if abs(sample.real) >= abs(sample.imag):
        return complex(np.sign(sample.real) or 1.0, 0.0)
    return complex(0.0, np.sign(sample.imag) or 1.0)


def phase_errors(received: NDArray, mu: float, theta0: float = 0.0) -> NDArray[np.floating]:
    """Decision-directed phase loop.

    theta is the receiver's estimate of the channel rotation.
    The detector is Im(a_hat* y), with y the derotated sample.
    """
    theta = theta0
    path = np.empty(received.size)
    for i, r in enumerate(received):
        y = r * np.exp(-1j * theta)
        a_hat = decide_qpsk(y)
        err = float(np.imag(np.conj(a_hat) * y))
        theta = theta + mu * err
        path[i] = theta
    return path


def fourth_power_cfo(symbols: NDArray, nfft: int | None = None) -> float:
    """QPSK frequency offset in cycles per symbol.

    A fourth power strips ±1 and ±j, leaving a tone at four times the offset.
    The estimate aliases once |offset| exceeds 1/8 cycle per symbol.
    """
    z4 = np.asarray(symbols, dtype=complex) ** 4
    n = z4.size if nfft is None else nfft
    spec = np.fft.fftshift(np.fft.fft(z4, n=n))
    freqs = np.fft.fftshift(np.fft.fftfreq(spec.size, d=1.0))
    return float(freqs[int(np.argmax(np.abs(spec)))] / 4.0)
