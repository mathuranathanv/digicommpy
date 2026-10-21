"""Symbol-spaced linear equalizers and a one-tap decision-feedback canceller."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def channel_matrix(h: NDArray, n_taps: int) -> NDArray[np.complexfloating]:
    """Toeplitz matrix so that H @ w is the convolution of h with the equalizer."""
    h = np.asarray(h, dtype=complex).ravel()
    rows = h.size + n_taps - 1
    H = np.zeros((rows, n_taps), dtype=complex)
    for col in range(n_taps):
        H[col : col + h.size, col] = h
    return H


def best_delay(h: NDArray, n_taps: int) -> int:
    """Delay whose cursor the zero-forcing solution can hit most closely."""
    H = channel_matrix(h, n_taps)
    P = H @ np.linalg.pinv(H)
    return int(np.argmax(np.real(np.diag(P))))


def zf_taps(h: NDArray, n_taps: int, delay: int | None = None) -> tuple[NDArray, int]:
    """Least-squares zero-forcing taps. Forces the cascade toward a delayed impulse."""
    if delay is None:
        delay = best_delay(h, n_taps)
    H = channel_matrix(h, n_taps)
    target = np.zeros(H.shape[0], dtype=complex)
    target[delay] = 1.0
    w, *_ = np.linalg.lstsq(H, target, rcond=None)
    return w, delay


def mmse_taps(
    h: NDArray, n_taps: int, n0: float, delay: int, es: float = 1.0
) -> NDArray:
    """FIR MMSE taps for white symbols of energy es and white noise of variance n0.

    The noise term is what zero-forcing leaves out. It stops the equalizer
    inverting a deep fade.
    """
    H = channel_matrix(h, n_taps)
    target = np.zeros(H.shape[0], dtype=complex)
    target[delay] = 1.0
    gram = H.conj().T @ H + (n0 / es) * np.eye(n_taps)
    return np.linalg.solve(gram, H.conj().T @ target)


def apply_equalizer(
    symbols: NDArray,
    h: NDArray,
    w: NDArray,
    delay: int,
    n0: float,
    rng: np.random.Generator,
) -> NDArray:
    """Convolve with h, add noise of variance n0, convolve with w, drop the delay."""
    received = np.convolve(symbols, np.asarray(h, dtype=complex))
    if n0 > 0:
        sigma = np.sqrt(n0 / 2)
        received = received + sigma * (
            rng.normal(size=received.shape) + 1j * rng.normal(size=received.shape)
        )
    y = np.convolve(received, w)
    return y[delay : delay + symbols.size]


def dfe_bpsk(received: NDArray, h: NDArray) -> NDArray[np.floating]:
    """Cancel one post-cursor with the previous decision, then slice BPSK.

    h = [cursor, postcursor]. The precursor is absent on purpose: feedback
    cannot cancel a symbol that has not been decided yet.
    """
    h = np.asarray(h, dtype=complex).ravel()
    if h.size != 2:
        raise ValueError("this teaching DFE expects h = [cursor, postcursor]")
    decisions = np.empty(received.size)
    prev = 0.0
    for k, r in enumerate(received):
        cleaned = (r - h[1] * prev) / h[0]
        prev = 1.0 if cleaned.real >= 0 else -1.0
        decisions[k] = prev
    return decisions


def lms(
    received: NDArray,
    desired: NDArray,
    n_taps: int,
    mu: float,
    delay: int,
) -> tuple[NDArray, NDArray]:
    """Train an FIR equalizer. Returns taps and the squared error at each update.

    The window of `received` that multiplies the taps ends at index k+delay,
    matching np.convolve.
    """
    w = np.zeros(n_taps, dtype=complex)
    power = []
    for k in range(desired.size):
        end = k + delay
        start = end - n_taps + 1
        if start < 0 or end >= received.size:
            continue
        x = received[start : end + 1][::-1]
        y = np.dot(w, x)
        err = desired[k] - y
        w = w + mu * err * np.conj(x)
        power.append(float(np.abs(err) ** 2))
    return w, np.asarray(power)
