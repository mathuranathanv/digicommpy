"""Chapter 1 §1.2 — read an FFT without fooling yourself.

Reproduces the published 10 Hz / fs=320 / N=256 example, then the
off-bin case the first edition never showed.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import fft, fftshift, ifft, ifftshift, rfft

from digicommpy.signals.spectra import (
    bin_frequencies,
    cosine_amplitude,
    periodic_hann,
    window_metrics,
)


def on_bin_cosine(n: int = 256, fc: float = 10.0, fs: float = 320.0):
    """First N samples of cos(2*pi*fc*t). For these numbers, exactly 8 cycles."""
    t = np.arange(n) / fs
    x = np.cos(2 * np.pi * fc * t)
    return t, x


def demonstrate() -> None:
    fc, fs, n = 10.0, 320.0, 256
    # Published record length: arange(0, 2, 1/fs) is 640 samples at this fs.
    t_full = np.arange(0.0, 2.0, 1.0 / fs)
    assert t_full.size == 640

    t, x = on_bin_cosine(n, fc, fs)
    assert np.isclose(fc * n / fs, 8.0)  # integer cycles: why the peak is clean

    spectrum = fft(x)
    assert np.isclose(spectrum[0], 0.0, atol=1e-12)
    assert np.argmax(np.abs(spectrum[: n // 2])) == 8
    assert np.isclose(np.abs(spectrum[8]), n / 2)  # 128, not 0.5
    assert np.isclose(cosine_amplitude(spectrum, 8, window_sum=n), 1.0)
    assert np.allclose(np.real(ifft(spectrum)), x, atol=1e-12)

    # Negative-frequency partner is raw index N-8 = 248 (0-based).
    assert np.isclose(np.abs(spectrum[n - 8]), n / 2)
    hand_hz = np.arange(n) * (fs / n)          # Edition-1 axis: 0 .. fs-df
    assert np.isclose(hand_hz[n - 8], fs - fc)  # 310 Hz — a label, not a second tone
    f_raw = bin_frequencies(n, fs, shifted=False)
    assert np.isclose(f_raw[n - 8], -fc)        # fftfreq already stores -10 Hz there
    f_shift = bin_frequencies(n, fs, shifted=True)
    shifted = fftshift(spectrum)
    peak_hz = f_shift[np.isclose(np.abs(shifted), n / 2)]
    assert np.allclose(np.sort(peak_hz), [-fc, fc])

    # NumPy negative-index "manual fftshift" works only by wraparound, even N.
    sample_index = np.arange(-n // 2, n // 2)
    assert np.allclose(spectrum[sample_index], fftshift(spectrum))

    # Real-input path matches the first half of the complex FFT.
    assert np.allclose(rfft(x), spectrum[: n // 2 + 1])

    # Off-bin tone: 10.3 Hz is 8.24 bins. Rectangular window scallops.
    x_off = np.cos(2 * np.pi * 10.3 * t)
    spec_off = fft(x_off)
    amp_rect = cosine_amplitude(spec_off, 8, window_sum=n)
    w = periodic_hann(n)
    cg, enbw = window_metrics(w)
    assert np.isclose(cg, 0.5)
    assert np.isclose(enbw, 1.5)
    spec_w = fft(x_off * w)
    amp_hann = cosine_amplitude(spec_w, 8, window_sum=float(np.sum(w)))
    # Measured on this grid (truth is 1). Hann reduces the bias; it does not remove it.
    assert 0.90 < amp_rect < 0.92
    assert 0.96 < amp_hann < 0.97

    # Odd-N axis. Edition 1's formula [-(N+1)/2, (N+1)/2 - 1] has N+1 points.
    n_odd = 9
    k = np.round(bin_frequencies(n_odd, n_odd, shifted=True)).astype(int)
    assert k[0] == -(n_odd - 1) // 2 and k[-1] == (n_odd - 1) // 2
    assert k.size == n_odd

    # fftshift / ifftshift textbook sequences
    even = np.arange(8)
    odd = np.arange(9)
    assert np.array_equal(fftshift(even), [4, 5, 6, 7, 0, 1, 2, 3])
    assert np.array_equal(fftshift(odd), [5, 6, 7, 8, 0, 1, 2, 3, 4])
    assert np.array_equal(ifftshift(fftshift(odd)), odd)
    assert not np.array_equal(fftshift(odd), ifftshift(odd))

    print(
        "§1.2 FFT self-check: ok  "
        f"(rect amp @ 10.3 Hz = {amp_rect:.3f}, Hann = {amp_hann:.3f})"
    )


if __name__ == "__main__":
    demonstrate()
