"""Passband waveform fragments used by Chapter 2 figures and demos.

Carrier and symbol rate are kept small on purpose (fc = 50 Hz, Rb = 10 bit/s)
so a printed figure can show the carrier inside the envelope.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from digicommpy.passband.pulses import root_raised_cosine


def differential_encode(bits: NDArray) -> NDArray[np.integer]:
    """c[k] = c[k-1] XOR b[k], with c[-1] = 0. A later sign flip cancels in the decode."""
    c = np.empty(bits.size, dtype=int)
    prev = 0
    for i, b in enumerate(bits.astype(int)):
        prev = prev ^ int(b)
        c[i] = prev
    return c


def differential_decode(coded: NDArray, reference: int = 0) -> NDArray[np.integer]:
    """Undo differential_encode. A phase inversion flips every bit and the reference."""
    prev = np.concatenate([[reference], coded[:-1]])
    return np.bitwise_xor(coded.astype(int), prev.astype(int))


def dbpsk_product_bits(symbols: NDArray[np.complexfloating]) -> NDArray[np.integer]:
    """Optimum binary DPSK decision from the complex product of successive symbols.

    symbols[k] is the matched-filter sample. Bit 0 means ``no phase change.''
    """
    prod = symbols[1:] * np.conj(symbols[:-1])
    # Positive real part: phase held. Negative: phase flipped.
    return (np.real(prod) < 0).astype(int)


def upsample_hold(symbols: NDArray, sps: int) -> NDArray:
    return np.repeat(symbols, sps)


def bpsk_baseband(bits: NDArray, sps: int) -> NDArray[np.floating]:
    """Map bit 1 -> +1, bit 0 -> -1, and hold for `sps` samples."""
    symbols = 2.0 * bits.astype(float) - 1.0
    return upsample_hold(symbols, sps)


def bpsk_passband(bits: NDArray, sps: int, fc: float, fs: float) -> tuple[NDArray, NDArray]:
    bb = bpsk_baseband(bits, sps)
    t = np.arange(bb.size) / fs
    return t, bb * np.cos(2 * np.pi * fc * t)


def qpsk_symbols(bits: NDArray) -> NDArray[np.complexfloating]:
    """Gray QPSK on the unit circle: 00, 01, 11, 10 around the plane."""
    if bits.size % 2:
        raise ValueError("QPSK needs an even number of bits")
    b = bits.reshape(-1, 2)
    idx = b[:, 0] * 2 + b[:, 1]
    # Gray: 00 -> 0, 01 -> 1, 11 -> 3, 10 -> 2, then place at pi/4 + m*pi/2
    gray = np.array([0, 1, 3, 2])
    m = gray[idx]
    return np.exp(1j * (np.pi / 4 + m * np.pi / 2))


def shaped_qpsk(bits: NDArray, sps: int, beta: float = 0.35, span: int = 8) -> NDArray:
    symbols = qpsk_symbols(bits)
    h = root_raised_cosine(span, sps, beta)
    stuffed = np.zeros(symbols.size * sps, dtype=complex)
    stuffed[::sps] = symbols
    return np.convolve(stuffed, h)


def shaped_oqpsk(bits: NDArray, sps: int, beta: float = 0.35, span: int = 8) -> NDArray:
    """Same symbols as QPSK, quadrature arm delayed by half a symbol."""
    s = qpsk_symbols(bits)
    h = root_raised_cosine(span, sps, beta)
    i = np.zeros(s.size * sps)
    q = np.zeros(s.size * sps + sps // 2)
    i[::sps] = s.real
    q[sps // 2 :: sps] = s.imag
    i = np.convolve(i, h)
    q = np.convolve(q, h)
    n = min(i.size, q.size)
    return i[:n] + 1j * q[:n]


def pi4_dqpsk_symbols(bits: NDArray) -> NDArray[np.complexfloating]:
    """Differential QPSK whose phase step is ±pi/4 or ±3pi/4, never 0 or pi."""
    if bits.size % 2:
        raise ValueError("pi/4-DQPSK needs an even number of bits")
    # dibit 00, 01, 11, 10 -> +pi/4, +3pi/4, -3pi/4, -pi/4
    delta_table = {
        (0, 0): np.pi / 4,
        (0, 1): 3 * np.pi / 4,
        (1, 1): -3 * np.pi / 4,
        (1, 0): -np.pi / 4,
    }
    pairs = bits.reshape(-1, 2)
    phi = 0.0
    out = np.empty(pairs.shape[0], dtype=complex)
    for k, pair in enumerate(pairs):
        phi += delta_table[(int(pair[0]), int(pair[1]))]
        out[k] = np.exp(1j * phi)
    return out


def shaped_pi4(bits: NDArray, sps: int, beta: float = 0.35, span: int = 8) -> NDArray:
    symbols = pi4_dqpsk_symbols(bits)
    h = root_raised_cosine(span, sps, beta)
    stuffed = np.zeros(symbols.size * sps, dtype=complex)
    stuffed[::sps] = symbols
    return np.convolve(stuffed, h)


def cpm_phase(bits: NDArray, sps: int, h: float) -> NDArray[np.floating]:
    """Continuous phase for binary CPFSK. bits in {0, 1}, frequency pulse is a rectangle.

    Phase advance per symbol is ±pi*h. MSK is h = 1/2. A jump of pi (h = 1)
    is ambiguous on a circle: +pi and -pi are the same point.
    """
    a = 2.0 * bits.astype(float) - 1.0
    step = a * np.pi * h / sps
    return np.cumsum(np.repeat(step, sps))


def cpm_complex(bits: NDArray, sps: int, h: float) -> NDArray[np.complexfloating]:
    return np.exp(1j * cpm_phase(bits, sps, h))


def gaussian_freq_pulse(bt: float, sps: int, span: int = 4) -> NDArray[np.floating]:
    """Gaussian filter impulse, sampled, unit sum. bt is the BT product."""
    if bt <= 0:
        raise ValueError("BT must be positive")
    t = (np.arange(-span * sps // 2, span * sps // 2 + 1)) / sps
    # h(t) proportional to exp(-2 (pi BT t)^2 / ln 2), t in symbol periods.
    h = np.exp(-2.0 * (np.pi * bt * t) ** 2 / np.log(2.0))
    return h / np.sum(h)


def gmsk_complex(bits: NDArray, sps: int, bt: float) -> NDArray[np.complexfloating]:
    """GMSK complex envelope, h = 1/2, Gaussian-filtered rectangular frequency pulse."""
    a = 2.0 * bits.astype(float) - 1.0
    rect = np.repeat(a, sps)
    g = gaussian_freq_pulse(bt, sps)
    freq = np.convolve(rect, g, mode="same")
    phase = np.cumsum(freq) * (np.pi / 2.0) / sps
    return np.exp(1j * phase)
