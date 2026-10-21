"""Numerical checks for Chapters 2 and 3. No plots."""
from __future__ import annotations

import numpy as np

from digicommpy.baseband import apsk_16, ml_detect, pam, psk, qam
from digicommpy.baseband.constellations import fsk_detect_coherent, fsk_detect_noncoherent
from digicommpy.passband.pulses import raised_cosine, root_raised_cosine
from digicommpy.passband.waveforms import cpm_phase, pi4_dqpsk_symbols, qpsk_symbols


def _hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


def main() -> None:
    rc = raised_cosine(8, 8, 0.35)
    # Sample the pulse at symbol centers. Index of t=0 is the middle.
    mid = rc.size // 2
    sps = 8
    taps = rc[mid :: sps][:4]
    assert abs(taps[0] - 1) < 1e-9
    assert np.max(np.abs(taps[1:])) < 1e-9

    h = root_raised_cosine(8, 8, 0.35)
    nyq = np.convolve(h, h)
    mid = nyq.size // 2
    off = nyq[mid + sps :: sps][:3]
    assert np.max(np.abs(off)) < 0.02

    bits = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    s = qpsk_symbols(bits)
    assert np.allclose(np.abs(s), 1)

    dphi = np.diff(np.angle(pi4_dqpsk_symbols(bits)))
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    assert np.all(np.isin(np.round(dphi / (np.pi / 4)), [-3, -1, 1, 3]))

    phase = cpm_phase(np.array([1, 0, 1, 1]), sps=8, h=0.5)
    # bits 1,0,1,1 -> amplitudes +1,-1,+1,+1. Net +2 steps of pi/2, so +pi.
    assert np.isclose(phase[-1], np.pi)

    from digicommpy.passband.waveforms import dbpsk_product_bits, differential_decode, differential_encode
    raw = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = differential_encode(raw)
    assert np.array_equal(differential_decode(1 - coded, reference=1), raw)
    sym = np.concatenate([[1.0], 1 - 2 * coded]) * np.exp(1j * 0.7)
    assert np.array_equal(dbpsk_product_bits(sym), raw)

    for M in (4, 16, 64):
        c = qam(M)
        assert np.isclose(np.mean(np.abs(c) ** 2), 1)
    c8 = psk(8)
    # Gray neighbors on the circle differ by one bit.
    order = np.argsort(np.angle(c8))
    for a, b in zip(order, np.roll(order, -1)):
        assert _hamming(int(a), int(b)) == 1

    c = pam(4)
    r = c[2] + 0.01
    assert ml_detect(np.array([r]), c)[0] == 2

    a16 = apsk_16()
    assert a16.size == 16 and np.isclose(np.mean(np.abs(a16) ** 2), 1)

    rx = np.array([[0.2, 1.5, 0.1]])
    assert fsk_detect_coherent(rx)[0] == 1
    rx_nc = np.array([[0.2, 0.3 + 2j, 0.1]])
    assert fsk_detect_noncoherent(rx_nc)[0] == 1
    print("chapters 2-3 self-check: ok")


if __name__ == "__main__":
    main()
