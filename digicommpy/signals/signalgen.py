"""
Stimulus generators for Chapter 1.

DIY implementations. SciPy equivalents are a check, not the definition.

@author: Mathuranathan Viswanathan
Original: Jul 15, 2019
Professor-revision: 2026
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sine_wave(
    f: float,
    over_samp_rate: int,
    phase: float,
    n_cyl: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Real sine of frequency *f* hertz.

    Parameters
    ----------
    f : float
        Tone frequency, Hz. Must be nonzero.
    over_samp_rate : int
        Display/implementation oversampling *L* (samples per cycle).
        Sampling rate is ``fs = L * f``. For a real tone, ``L > 2``
        is the Nyquist requirement; typical plots use ``L = 30``.
    phase : float
        Phase offset, radians.
    n_cyl : int
        Number of cycles to generate.

    Returns
    -------
    t, g : ndarray
        Time base in seconds and ``sin(2*pi*f*t + phase)``.
    """
    if f == 0:
        raise ValueError("f must be nonzero")
    if over_samp_rate < 2:
        raise ValueError("over_samp_rate must be >= 2 for a real sine")
    fs = over_samp_rate * f
    n_samples = int(n_cyl * over_samp_rate)
    t = np.arange(n_samples) / fs
    g = np.sin(2 * np.pi * f * t + phase)
    return t, g


def complex_tone(
    f: float,
    fs: float,
    n_samples: int,
    phase: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Analytic tone ``exp(1j*(2*pi*f*t + phase))``.

    For this one-sided signal the sampling requirement is ``fs > abs(f)``,
    not ``fs > 2*f``.
    """
    if fs <= abs(f):
        raise ValueError("fs must exceed |f| for a complex tone")
    t = np.arange(n_samples) / fs
    z = np.exp(1j * (2 * np.pi * f * t + phase))
    return t, z


def square_wave(
    f: float,
    over_samp_rate: int,
    n_cyl: int,
    *,
    binary: bool = False,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """50% square wave via ``sign(sin(2*pi*f*t))``.

    Parameters
    ----------
    binary : bool
        If True, replace the exact zeros that ``sign`` returns at the
        sine zero-crossings with +1, giving a strict ±1 waveform.
    """
    if f == 0:
        raise ValueError("f must be nonzero")
    fs = over_samp_rate * f
    n_samples = int(n_cyl * over_samp_rate)
    t = np.arange(n_samples) / fs
    g = np.sign(np.sin(2 * np.pi * f * t))
    if binary:
        g = np.where(g == 0.0, 1.0, g)
    return t, g


def rect_pulse(
    A: float,
    fs: float,
    T: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Isolated rectangular pulse, midpoint convention at the jumps.

    Observation window is 1 s, centered, with an integer number of
    samples so that t = 0 is a sample. Edges at ±T/2 take the value A/2.
    Requires ``fs`` integer and ``T*fs/2`` integer.
    """
    fs_i = int(round(fs))
    if abs(fs_i - fs) > 1e-9:
        raise ValueError("fs must be an integer number of samples per second")
    half_samp = T * fs_i / 2.0
    n_edge = int(round(half_samp))
    if abs(half_samp - n_edge) > 1e-9:
        raise ValueError("T*fs/2 must be an integer so edges land on samples")

    n0 = fs_i // 2
    t = (np.arange(fs_i) - n0) / fs_i
    g = np.zeros(fs_i)
    g[n0 - n_edge + 1 : n0 + n_edge] = A
    g[n0 - n_edge] = A / 2
    g[n0 + n_edge] = A / 2
    return t, g


def gaussian_pulse(
    fs: float,
    sigma: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Centered Gaussian pulse, same 1 s window as ``rect_pulse``.

    ``g(t) = 1/(sqrt(2*pi)*sigma) * exp(-t^2 / (2*sigma^2))``.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    fs_i = int(round(fs))
    if abs(fs_i - fs) > 1e-9:
        raise ValueError("fs must be an integer number of samples per second")
    n0 = fs_i // 2
    t = (np.arange(fs_i) - n0) / fs_i
    g = np.exp(-t**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    return t, g


def chirp_signal(
    t: NDArray[np.floating],
    f0: float,
    t1: float,
    f1: float,
    phase: float = 0.0,
) -> NDArray[np.floating]:
    """Linear FM chirp from the phase integral.

    Instantaneous frequency is ``f0`` hertz at ``t[0]`` (taken as the
    sweep origin) and ``f1`` hertz at time ``t1``. ``phase`` is radians.

    The compact form ``cos(2*pi*f(t)*t + phase)`` with
    ``f(t) = (k/2)*t + f0`` is used. Instantaneous frequency is
    ``fi(t) = k*t + f0 = 2*f(t) - f0``, not ``f(t)``.

    Matches ``scipy.signal.chirp(..., method='linear')`` when the
    SciPy sweep origin is ``t=0``. SciPy's ``phi`` argument is in
    *degrees*; this function takes ``phase`` in radians.
    """
    t0 = float(t[0])
    duration = t1 - t0
    if duration == 0:
        raise ValueError("t1 must differ from t[0]")
    k = (f1 - f0) / duration
    # Phase integral about the sweep origin t0, then evaluated on t.
    tau = t - t0
    return np.cos(2 * np.pi * ((k / 2) * tau + f0) * tau + phase)
