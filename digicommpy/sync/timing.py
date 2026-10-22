"""Symbol timing on a root-raised-cosine matched filter."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from digicommpy.passband.pulses import root_raised_cosine


def rrc_matched(symbols: NDArray, sps: int, beta: float, span: int) -> tuple[NDArray, int]:
    """Transmit and receive RRC. Returns the matched waveform and the sample delay of symbol 0.

    The delay is the index of the peak that belongs to a symbol placed at sample 0.
    Symbol k, sampled on time, sits at index k*sps + delay.
    """
    h = root_raised_cosine(span, sps, beta)
    up = np.zeros(symbols.size * sps, dtype=complex)
    up[::sps] = symbols
    wave = np.convolve(np.convolve(up, h), h)
    delay = len(h) - 1
    return wave, delay


def fractional_sample(wave: NDArray, index: float) -> complex:
    """Linear interpolation. index is in samples and must lie inside the record."""
    i0 = int(np.floor(index))
    if i0 < 0 or i0 + 1 >= wave.size:
        raise IndexError("fractional sample fell off the record")
    mu = index - i0
    return (1.0 - mu) * wave[i0] + mu * wave[i0 + 1]


def gardner_error(wave: NDArray, delay: int, sps: int, tau: float, k: int) -> float:
    """One Gardner error at symbol k, for a sampling phase tau in symbol periods.

    e = Re{ y*(k - 1/2) * (y(k) - y(k-1)) }.
    A positive tau samples later than the matched-filter peak.
    """
    yk = fractional_sample(wave, (k + tau) * sps + delay)
    y_prev = fractional_sample(wave, (k - 1 + tau) * sps + delay)
    y_mid = fractional_sample(wave, (k - 0.5 + tau) * sps + delay)
    return float(np.real(np.conj(y_mid) * (yk - y_prev)))


def gardner_scurve(
    wave: NDArray,
    delay: int,
    sps: int,
    tau: float,
    k0: int,
    k1: int,
) -> float:
    """Mean Gardner timing error from symbol `k0` up to, but not including, `k1`."""
    errors = [gardner_error(wave, delay, sps, tau, k) for k in range(k0, k1)]
    return float(np.mean(errors))


def timing_loop(
    wave: NDArray,
    delay: int,
    sps: int,
    tau0: float,
    mu: float,
    k0: int,
    k1: int,
) -> NDArray[np.floating]:
    """First-order loop. The S-curve has positive slope, so the update subtracts."""
    tau = tau0
    path = np.empty(k1 - k0)
    for i, k in enumerate(range(k0, k1)):
        err = gardner_error(wave, delay, sps, tau, k)
        tau = tau - mu * err
        path[i] = tau
    return path
