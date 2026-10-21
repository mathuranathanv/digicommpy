"""Small codes used to show a coding gain and an interleaver."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from digicommpy.performance.error_rates import qfunc


def repetition_encode(bits: NDArray, repeats: int = 3) -> NDArray[np.integer]:
    return np.repeat(np.asarray(bits, dtype=int), repeats)


def repetition_decode(channel_bits: NDArray, repeats: int = 3) -> NDArray[np.integer]:
    """Majority vote. `repeats` must be odd."""
    bits = np.asarray(channel_bits, dtype=int).reshape(-1, repeats)
    return (np.sum(bits, axis=1) > repeats / 2).astype(int)


def hard_repetition_ber(ebn0_db: NDArray | float, repeats: int = 3) -> NDArray:
    """Exact hard-decision BER of odd repetition, plotted against Eb/N0.

    Each channel use gets Eb/repeats, because one information bit is sent
    `repeats` times. Forgetting that shift draws the coded curve in the wrong place.
    """
    if repeats % 2 == 0 or repeats < 1:
        raise ValueError("majority decoding wants an odd number of repeats")
    ec = 10 ** ((np.asarray(ebn0_db, dtype=float) - 10 * np.log10(repeats)) / 10)
    p = qfunc(np.sqrt(2 * ec))
    # P(at least (repeats+1)/2 channel errors)
    need = repeats // 2 + 1
    ber = np.zeros(np.shape(p), dtype=float)
    for k in range(need, repeats + 1):
        ber = ber + _binom(repeats, k) * p**k * (1 - p) ** (repeats - k)
    return ber


def hard_repetition_ber_rayleigh(ebn0_db: NDArray | float, repeats: int = 3) -> NDArray:
    """Hard repetition on independent Rayleigh fades, one fade per channel bit.

    The repeats are interleaved. Three copies that see the same fade do not
    earn this curve.
    """
    from digicommpy.performance.error_rates import ber_rayleigh_bpsk

    if repeats % 2 == 0 or repeats < 1:
        raise ValueError("majority decoding wants an odd number of repeats")
    ec_db = np.asarray(ebn0_db, dtype=float) - 10 * np.log10(repeats)
    p = np.asarray(ber_rayleigh_bpsk(ec_db), dtype=float)
    need = repeats // 2 + 1
    ber = np.zeros(np.shape(p), dtype=float)
    for k in range(need, repeats + 1):
        ber = ber + _binom(repeats, k) * p**k * (1 - p) ** (repeats - k)
    return ber


def _binom(n: int, k: int) -> int:
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c


def hamming74_encode(data: NDArray) -> NDArray[np.integer]:
    """(7, 4) Hamming, parity bits in positions 0, 1 and 3."""
    d = np.asarray(data, dtype=int).reshape(-1, 4)
    d1, d2, d3, d4 = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return np.column_stack([p1, p2, d1, p3, d2, d3, d4]).astype(int).ravel()


def hamming74_decode(codeword: NDArray) -> NDArray[np.integer]:
    """Correct one bit, then return the four data bits. Two errors are a wrong codeword."""
    r = np.asarray(codeword, dtype=int).reshape(-1, 7).copy()
    p1, p2, d1, p3, d2, d3, d4 = (r[:, i] for i in range(7))
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    syndrome = s1 + 2 * s2 + 4 * s3
    for row, syn in enumerate(syndrome):
        if syn:
            r[row, syn - 1] ^= 1
    return r[:, [2, 4, 5, 6]].ravel()


def block_interleave(bits: NDArray, rows: int, cols: int) -> NDArray:
    """Write by rows, read by columns."""
    return np.asarray(bits, dtype=int).reshape(rows, cols).ravel(order="F")


def block_deinterleave(bits: NDArray, rows: int, cols: int) -> NDArray:
    """Undo block_interleave. The array was read by columns."""
    return np.asarray(bits, dtype=int).reshape((rows, cols), order="F").ravel(order="C")
