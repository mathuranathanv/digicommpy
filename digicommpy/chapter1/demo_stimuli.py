"""Chapter 1 §1.1 demos. Run as a script; each function is self-contained."""
from __future__ import annotations

import numpy as np
from scipy import signal

from digicommpy.signals import (
    chirp_signal,
    complex_tone,
    gaussian_pulse,
    rect_pulse,
    sine_wave,
    square_wave,
)


def sine_wave_demo() -> None:
    import matplotlib.pyplot as plt

    t, g = sine_wave(f=10, over_samp_rate=30, phase=np.pi / 3, n_cyl=5)
    plt.plot(t, g)
    plt.title(r"Sine wave $f=10\,\mathrm{Hz}$, $\phi=\pi/3$")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def complex_tone_demo() -> None:
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq

    t, z = complex_tone(f=10, fs=50, n_samples=256, phase=0.0)
    fig, ax = plt.subplots(2, 1, figsize=(6, 5))
    ax[0].plot(t, z.real, label=r"$\Re z$")
    ax[0].plot(t, z.imag, label=r"$\Im z$")
    ax[0].set_xlabel("Time (s)")
    ax[0].legend()
    ax[0].set_title(r"Analytic tone $e^{j2\pi 10 t}$, $f_s=50\,\mathrm{Hz}$")

    Z = fft(z)
    f = fftfreq(z.size, d=t[1] - t[0])
    ax[1].stem(f, np.abs(Z) / z.size, basefmt=" ")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel(r"$|Z[k]|/N$")
    ax[1].set_title(r"One line at $+10\,\mathrm{Hz}$ (no negative image)")
    fig.tight_layout()
    plt.show()


def square_wave_demo() -> None:
    import matplotlib.pyplot as plt

    t, g = square_wave(f=10, over_samp_rate=30, n_cyl=5)
    plt.plot(t, g, drawstyle="steps-post")
    plt.title(r"Square wave $f=10\,\mathrm{Hz}$, $50\%$ duty (signum of sine)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def scipy_square_wave_20pct() -> None:
    """Separate figure: 20% duty, so even harmonics exist."""
    import matplotlib.pyplot as plt

    f, L, n_cyl = 10, 30, 5
    fs = L * f
    t = np.arange(int(n_cyl * L)) / fs
    g = signal.square(2 * np.pi * f * t, duty=0.2)
    plt.plot(t, g, drawstyle="steps-post")
    plt.title(r"Square wave $f=10\,\mathrm{Hz}$, $20\%$ duty (SciPy)")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def rect_pulse_demo() -> None:
    import matplotlib.pyplot as plt

    t, g = rect_pulse(A=1, fs=500, T=0.2)
    plt.plot(t, g, drawstyle="steps-post")
    plt.title(r"Rectangular pulse $T=0.2\,\mathrm{s}$, midpoint edges")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def gaussian_pulse_demo() -> None:
    import matplotlib.pyplot as plt

    t, g = gaussian_pulse(fs=80, sigma=0.1)
    plt.plot(t, g)
    plt.title(r"Gaussian pulse $\sigma=0.1\,\mathrm{s}$")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def chirp_demo() -> None:
    """DIY phase-integral chirp, asserted against SciPy.

    Record ends at t1 so the last sample is at instantaneous frequency f1.
    """
    import matplotlib.pyplot as plt

    fs = 500
    t1 = 0.5
    t = np.linspace(0.0, t1, int(round(t1 * fs)) + 1)
    f0, f1, phase = 1.0, 20.0, 0.0
    g = chirp_signal(t, f0=f0, t1=t1, f1=f1, phase=phase)
    # SciPy's phi is in degrees; ours is in radians.
    g_ref = signal.chirp(
        t, f0=f0, t1=t1, f1=f1, phi=np.degrees(phase), method="linear"
    )
    if not np.allclose(g, g_ref, atol=1e-12):
        raise AssertionError("chirp_signal does not match scipy.signal.chirp")

    t0 = float(t[0])
    k = (f1 - f0) / (t1 - t0)
    tau = t - t0
    theta = 2 * np.pi * ((k / 2) * tau + f0) * tau + phase
    fi = np.gradient(theta, t) / (2 * np.pi)
    fi_theory = k * tau + f0

    fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=False)
    ax[0].plot(t, g)
    ax[0].set_title(rf"Linear FM chirp $f_i: {f0}\to{f1}\,\mathrm{{Hz}}$ in ${t1}\,\mathrm{{s}}$")
    ax[0].set_xlabel("Time (s)")
    ax[1].plot(t, fi_theory, label=r"$f_i(t)=kt+f_0$")
    ax[1].plot(t, fi, "--", label=r"$(1/2\pi)\,d\theta/dt$")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Hz")
    ax[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Headless check: generators and the SciPy chirp contract.
    t, g = sine_wave(10, 30, np.pi / 3, 5)
    assert t.size == 150 and g.size == 150
    t, z = complex_tone(10, 50, 256)
    assert z.dtype.kind == "c"
    t, g = square_wave(10, 30, 5)
    assert set(np.unique(np.round(g, 12))).issubset({-1.0, 0.0, 1.0})
    t, g = rect_pulse(1, 500, 0.2)
    assert g[g.size // 2] == 1.0
    t, g = gaussian_pulse(80, 0.1)
    assert np.isclose(g.max(), 1 / (np.sqrt(2 * np.pi) * 0.1))
    t1 = 0.5
    t = np.linspace(0.0, t1, int(round(t1 * 500)) + 1)
    g = chirp_signal(t, 1.0, t1, 20.0)
    g_ref = signal.chirp(t, f0=1.0, t1=t1, f1=20.0, phi=0.0, method="linear")  # phi in deg
    assert np.allclose(g, g_ref, atol=1e-12)
    print("§1.1 stimulus self-check: ok")
