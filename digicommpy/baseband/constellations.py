"""Complex-baseband constellations. Average symbol energy is 1.

Square QAM is defined when M is an even power of two (4, 16, 64, 256).
32-QAM and 128-QAM are cross constellations; they are not built here.
16-APSK uses the DVB-S2 4+12 ring arrangement.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _unit_energy(s: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    power = np.mean(np.abs(s) ** 2)
    return s / np.sqrt(power)


def pam(M: int) -> NDArray[np.complexfloating]:
    """M-PAM on the real axis, points ±1, ±3, ..., ±(M-1), unit average energy."""
    if M < 2 or (M & (M - 1)):
        raise ValueError("M must be a power of two and at least 2")
    m = np.arange(M)
    a = 2 * m + 1 - M
    return _unit_energy(a.astype(complex))


def psk(M: int) -> NDArray[np.complexfloating]:
    """Unit-energy M-PSK. Index m is Gray-labeled: adjacent points differ by one bit."""
    if M < 2 or (M & (M - 1)):
        raise ValueError("M must be a power of two and at least 2")
    m = np.arange(M)
    gray = m ^ (m >> 1)
    # Place Gray index k at angle 2*pi*k/M, then reorder so output[m] is symbol m.
    angles = np.empty(M)
    angles[gray] = 2 * np.pi * np.arange(M) / M
    return np.exp(1j * angles)


def qam(M: int) -> NDArray[np.complexfloating]:
    """Square Gray M-QAM. Each axis is a Gray PAM; adjacent neighbors differ by one bit."""
    side = int(np.sqrt(M))
    if side * side != M or side < 2:
        raise ValueError("square QAM needs M in {4, 16, 64, 256, ...}")
    m = np.arange(side)
    gray = m ^ (m >> 1)
    levels = 2 * np.arange(side) + 1 - side
    # levels[gray[i]] is the amplitude whose Gray label is i
    amp = np.empty(side)
    amp[gray] = levels
    bits = int(np.log2(side))
    s = np.empty(M, dtype=complex)
    for sym in range(M):
        i_idx = sym & (side - 1)
        q_idx = sym >> bits
        s[sym] = amp[i_idx] + 1j * amp[q_idx]
    return _unit_energy(s)


def apsk_16(gamma: float = 2.7) -> NDArray[np.complexfloating]:
    """DVB-S2-style 16-APSK: 4 points on the inner ring, 12 on the outer.

    gamma = R_outer / R_inner. Outer ring is rotated by pi/12 relative to
    the inner ring. Average energy is 1.
    """
    inner = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
    outer = gamma * np.exp(1j * (np.pi / 12 + np.arange(12) * 2 * np.pi / 12))
    return _unit_energy(np.concatenate([inner, outer]))


def fsk_basis(M: int) -> NDArray[np.complexfloating]:
    """Coherent orthogonal M-FSK as M unit vectors in C^M. Row m is symbol m."""
    return np.eye(M, dtype=complex)


def ml_detect(received: NDArray[np.complexfloating], constellation: NDArray[np.complexfloating]) -> NDArray[np.integer]:
    """Minimum Euclidean distance. received is shape (N,), constellation shape (M,)."""
    # |r - s|^2 = |r|^2 + |s|^2 - 2 Re(r conj(s))
    r = received.reshape(-1, 1)
    s = constellation.reshape(1, -1)
    dist = np.abs(r - s) ** 2
    return np.argmin(dist, axis=1)


def fsk_detect_coherent(received: NDArray[np.complexfloating]) -> NDArray[np.integer]:
    """received shape (N, M). Pick the largest real part (phase is known)."""
    return np.argmax(np.real(received), axis=1)


def fsk_detect_noncoherent(received: NDArray[np.complexfloating]) -> NDArray[np.integer]:
    """received shape (N, M). Pick the largest envelope. Phase is unknown."""
    return np.argmax(np.abs(received), axis=1)
