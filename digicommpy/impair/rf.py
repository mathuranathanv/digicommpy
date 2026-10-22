"""DC offset, single-branch IQ imbalance, and a Wiener phase walk."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_iq(symbols: NDArray, g: float, phi: float) -> NDArray:
    """Single-branch imbalance. I is untouched. Q is scaled by g and skewed by phi radians.

    z_i = r_i
    z_q = -g sin(phi) r_i + g cos(phi) r_q
    """
    i = np.real(symbols)
    q = np.imag(symbols)
    zq = -g * np.sin(phi) * i + g * np.cos(phi) * q
    return i + 1j * zq


def apply_dc(symbols: NDArray, dc_i: float, dc_q: float) -> NDArray:
    """Add a constant offset on the I rail and on the Q rail."""
    return np.asarray(symbols, dtype=complex) + (dc_i + 1j * dc_q)


def remove_dc(symbols: NDArray) -> NDArray:
    """A zero-mean constellation has no DC. Subtract the sample mean."""
    return np.asarray(symbols, dtype=complex) - np.mean(symbols)


def compensate_iq_blind(symbols: NDArray) -> NDArray:
    """Moseley-style blind correction from the signs and the mean absolute values."""
    zi = np.real(symbols)
    zq = np.imag(symbols)
    theta1 = -np.mean(np.sign(zi) * zq)
    theta2 = np.mean(np.abs(zi))
    theta3 = np.mean(np.abs(zq))
    c1 = theta1 / theta2
    inside = theta3**2 - theta1**2
    if inside <= 0 or theta2 == 0:
        raise ValueError("blind IQ estimate failed; the record is too short or too skewed")
    c2 = np.sqrt(inside / theta2**2)
    return zi + 1j * (c1 * zi + zq) / c2


def compensate_iq_known(symbols: NDArray, g: float, phi: float) -> NDArray:
    """Invert the single-branch model when g and phi have already been measured."""
    zi = np.real(symbols)
    zq = np.imag(symbols)
    # zq = -g sin(phi) zi_true + g cos(phi) zq_true, and zi = zi_true
    q = (zq + g * np.sin(phi) * zi) / (g * np.cos(phi))
    return zi + 1j * q


def phase_walk(symbols: NDArray, sigma: float, rng: np.random.Generator) -> NDArray:
    """Multiply by a random-walk phase. sigma is radians per symbol."""
    steps = rng.normal(0.0, sigma, size=np.asarray(symbols).size)
    return np.asarray(symbols, dtype=complex) * np.exp(1j * np.cumsum(steps))
