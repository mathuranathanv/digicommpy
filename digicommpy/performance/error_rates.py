"""Symbol-error formulas used as the ruler in Chapter 4.

Es/N0 and Eb/N0 enter in dB except where the name ends in `_lin`.
MPSK for M > 4 is the usual nearest-neighbor approximation, not an identity.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc


def qfunc(x: np.ndarray | float) -> np.ndarray:
    """Gaussian right-tail probability Q(x)."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2))


def _lin(db: np.ndarray | float) -> np.ndarray:
    return 10 ** (np.asarray(db, dtype=float) / 10)


def ser_bpsk(ebn0_db: np.ndarray | float) -> np.ndarray:
    """BPSK bit error rate. For BPSK this is also the symbol error rate."""
    return qfunc(np.sqrt(2 * _lin(ebn0_db)))


def ser_qpsk(esn0_db: np.ndarray | float) -> np.ndarray:
    """Exact QPSK symbol error rate. Gray bit error rate equals BPSK at the same Eb/N0."""
    q = qfunc(np.sqrt(_lin(esn0_db)))
    return 2 * q - q**2


def ser_mpsk_approx(M: int, esn0_db: np.ndarray | float) -> np.ndarray:
    """Nearest-neighbor approximation. Do not use it for M = 2: it is twice too large."""
    if M < 4:
        raise ValueError("use ser_bpsk for M = 2; this approximation is for M >= 4")
    return 2 * qfunc(np.sin(np.pi / M) * np.sqrt(2 * _lin(esn0_db)))


def ser_mqam(M: int, esn0_db: np.ndarray | float) -> np.ndarray:
    """Exact symbol error rate of square M-QAM."""
    side = int(np.sqrt(M))
    if side * side != M:
        raise ValueError("square QAM only")
    a = 1 - 1 / side
    q = qfunc(np.sqrt(3 * _lin(esn0_db) / (M - 1)))
    return 4 * a * q - 4 * a * a * q * q


def ser_awgn(kind: str, esn0_db: np.ndarray | float, M: int = 2) -> np.ndarray:
    """Dispatch. `esn0_db` is Eb/N0 when kind is 'bpsk', and Es/N0 otherwise."""
    if kind == "bpsk":
        return ser_bpsk(esn0_db)
    if kind == "qpsk":
        return ser_qpsk(esn0_db)
    if kind == "psk":
        return ser_mpsk_approx(M, esn0_db)
    if kind == "qam":
        return ser_mqam(M, esn0_db)
    raise ValueError(kind)


def ber_rayleigh_bpsk(ebn0_db: np.ndarray | float) -> np.ndarray:
    """Average BPSK bit error rate on a unit-power Rayleigh fade, perfect CSI."""
    g = _lin(ebn0_db)
    return 0.5 * (1 - np.sqrt(g / (1 + g)))


def ber_rician_bpsk(
    ebn0_db: np.ndarray | float,
    K: float,
    rng: np.random.Generator,
    n_fades: int = 40_000,
) -> np.ndarray:
    """Average BPSK error rate by integrating the Q-function over the fade.

    K = 0 reproduces the Rayleigh closed form, up to the Monte Carlo tolerance.
    """
    from digicommpy.performance.channels import rician

    db = np.atleast_1d(np.asarray(ebn0_db, dtype=float))
    h = rician(n_fades, K, rng)
    power = np.abs(h) ** 2
    out = np.empty(db.size)
    for i, db_i in enumerate(db):
        gamma = power * _lin(db_i)
        out[i] = float(np.mean(qfunc(np.sqrt(2 * gamma))))
    return out[0] if np.ndim(ebn0_db) == 0 else out
