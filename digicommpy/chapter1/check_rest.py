"""Checks for the rest of Chapter 1."""
from __future__ import annotations

import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import welch

from digicommpy.signals.analytic import analytic_signal, parseval_energy


def main() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=256)
    t, f = parseval_energy(x)
    assert abs(t - f) / t < 1e-10

    # A one-sample delay multiplies bin k by exp(-j 2 pi k / N).
    n = 32
    tone = np.cos(2 * np.pi * 3 * np.arange(n) / n)
    delayed = np.roll(tone, 1)
    phase = np.angle(fft(delayed)[3] / fft(tone)[3])
    assert abs(phase - (-2 * np.pi * 3 / n)) < 1e-8

    h = np.array([1.0, 0.5, 0.25])
    s = rng.normal(size=8)
    linear = np.convolve(s, h)
    circular = ifft(fft(s) * fft(h, n=s.size)).real
    assert linear.size == s.size + h.size - 1
    wrapped = linear[: s.size].copy()
    wrapped[: h.size - 1] += linear[s.size :]
    assert np.max(np.abs(circular - wrapped)) < 1e-8

    t = np.arange(256) / 64
    cos = np.cos(2 * np.pi * 4 * t)  # 4 Hz, fs=64, exactly 16 cycles in 256? 4*256/64=16. Yes.
    z = analytic_signal(cos)
    assert np.max(np.abs(np.abs(z) - 1)) < 1e-8
    assert np.max(np.abs(z.real - cos)) < 1e-8

    # Welch's noise floor varies less than one raw periodogram segment.
    noise = rng.normal(size=8192)
    _, pxx = welch(noise, nperseg=256)
    periodogram = np.abs(fft(noise[:256])) ** 2
    assert np.std(pxx) < np.std(periodogram)
    print("chapter 1 rest self-check: ok")


if __name__ == "__main__":
    main()
