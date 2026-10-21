"""Pulse-shaping filters for the passband chapter.

Time is in symbol periods: t = n / L, where L is samples per symbol.
The Edition-1 formulas wrote sinc(n) with n the raw sample index. That
puts a zero on every sample and the pulse collapses. Use t = n/L.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _time_base(span_symbols: int, sps: int) -> NDArray[np.floating]:
    if span_symbols < 2 or span_symbols % 2 != 0:
        raise ValueError("span_symbols must be an even integer >= 2")
    if sps < 2:
        raise ValueError("sps (samples per symbol) must be >= 2")
    n = np.arange(-span_symbols * sps // 2, span_symbols * sps // 2 + 1)
    return n / sps


def rect_pulse(span_symbols: int, sps: int) -> NDArray[np.floating]:
    """Length-L rectangular pulse, repeated across `span_symbols` only as a hold of one symbol.

    Returned vector is one symbol long (L ones), normalized to unit energy.
    `span_symbols` is accepted so the call shape matches the other pulses; a
    rectangle longer than one symbol is intentional ISI and is not this pulse.
    """
    del span_symbols
    p = np.ones(sps)
    return p / np.linalg.norm(p)


def raised_cosine(span_symbols: int, sps: int, alpha: float) -> NDArray[np.floating]:
    """Nyquist raised-cosine pulse. alpha is the excess-bandwidth factor in [0, 1]."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    t = _time_base(span_symbols, sps)
    p = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            p[i] = 1.0
        elif alpha > 0 and abs(abs(ti) - 1.0 / (2.0 * alpha)) < 1e-8:
            p[i] = (alpha / 2.0) * np.sin(np.pi / (2.0 * alpha))
        else:
            p[i] = np.sinc(ti) * np.cos(np.pi * alpha * ti) / (1.0 - (2.0 * alpha * ti) ** 2)
    return p


def root_raised_cosine(span_symbols: int, sps: int, beta: float) -> NDArray[np.floating]:
    """Unit-energy root-raised-cosine pulse. beta is the roll-off in [0, 1].

    Transmit and receive each use this pulse. Their cascade (sampled at
    symbol instants) is the raised-cosine, which is zero at nonzero
    multiples of the symbol period.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    t = _time_base(span_symbols, sps)
    p = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            p[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-8:
            p[i] = (
                beta
                / np.sqrt(2.0)
                * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
                )
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(
                np.pi * ti * (1.0 + beta)
            )
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            p[i] = num / den
    energy = np.sum(p**2)
    return p / np.sqrt(energy)
