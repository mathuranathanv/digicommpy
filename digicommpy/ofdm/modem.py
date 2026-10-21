"""One OFDM symbol: IFFT, cyclic prefix, one tap per subcarrier."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from digicommpy.equalize.fde import add_cyclic_prefix, through_channel


def modulate(symbols: NDArray, cp: int) -> NDArray:
    """IFFT with NumPy's default scaling, then a cyclic prefix.

    fft(ifft(X)) returns X. The channel frequency response that matches this
    pair is fft(h), not the unitary transform.
    """
    time = np.fft.ifft(np.asarray(symbols, dtype=complex))
    return add_cyclic_prefix(time, cp)


def demodulate(waveform: NDArray, nfft: int, cp: int, H: NDArray | None = None) -> NDArray:
    """Drop the prefix, FFT, and optionally apply one MMSE/ZF tap per bin.

    Pass H = fft(h, nfft). A zero-forcing tap is 1/H[k].
    """
    body = np.asarray(waveform, dtype=complex)[cp : cp + nfft]
    spectrum = np.fft.fft(body)
    if H is None:
        return spectrum
    return spectrum / H


def receive(symbols: NDArray, h: NDArray, cp: int) -> NDArray:
    """Modulate, pass through h, demodulate with no equalizer."""
    tx = modulate(symbols, cp)
    y = through_channel(tx, h, symbols.size, cp)
    return demodulate(y, symbols.size, 0)


def channel_bins(h: NDArray, nfft: int) -> NDArray:
    return np.fft.fft(np.asarray(h, dtype=complex), n=nfft)
