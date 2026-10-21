"""Numerical checks for the Chapter 4 ruler. No plots."""
from __future__ import annotations

import numpy as np

from digicommpy.baseband import ml_detect, psk
from digicommpy.performance import add_awgn, derotate, rayleigh, rician, ser_awgn
from digicommpy.performance.error_rates import ber_rayleigh_bpsk, ber_rician_bpsk, ser_bpsk, ser_qpsk


def main() -> None:
    rng = np.random.default_rng(4)
    # Unit-energy QPSK at 10 dB Es/N0 must come back with noise power 0.1.
    s = psk(4)[rng.integers(0, 4, 200_000)]
    r = add_awgn(s, esn0_db=10, rng=rng)
    n0 = float(np.mean(np.abs(r - s) ** 2))
    assert abs(n0 - 0.1) < 0.002

    # 0 dB BPSK: Q(sqrt(2)) is about 0.0786.
    assert abs(float(ser_bpsk(0)) - 0.0786496) < 1e-5

    # Gray QPSK at the same Eb/N0 has Es/N0 = Eb/N0 + 3.01 dB.
    # SER = 2 Pb - Pb^2, with Pb equal to the BPSK error rate.
    pb = float(ser_bpsk(6))
    ps = float(ser_qpsk(6 + 10 * np.log10(2)))
    assert abs(ps - (2 * pb - pb**2)) < 1e-12

    # The M=4 nearest-neighbor formula drops the square term, so it sits above the exact SER.
    approx = float(ser_awgn("psk", 6 + 10 * np.log10(2), M=4))
    assert approx > ps

    # Rayleigh closed form at 20 dB is near 1/(4 gamma).
    ray = float(ber_rayleigh_bpsk(20))
    assert abs(ray - 0.5 * (1 - np.sqrt(100 / 101))) < 1e-12
    assert abs(ray - 1 / 400) / ray < 0.02

    h = rayleigh(100_000, rng)
    assert abs(float(np.mean(np.abs(h) ** 2)) - 1) < 0.01
    hk = rician(100_000, K=4, rng=rng)
    assert abs(float(np.mean(np.abs(hk) ** 2)) - 1) < 0.01

    # K = 0 matches the Rayleigh formula.
    rice0 = float(ber_rician_bpsk(10, K=0, rng=rng))
    assert abs(rice0 - float(ber_rayleigh_bpsk(10))) / rice0 < 0.03

    # A short Monte Carlo at 6 dB Eb/N0. Theory is about 2.4e-3.
    bits = rng.integers(0, 2, 80_000)
    bpsk = 1 - 2 * bits
    rx = add_awgn(bpsk.astype(float), esn0_db=6, rng=rng)
    ber = float(np.mean((rx < 0).astype(int) != bits))
    theory = float(ser_bpsk(6))
    assert abs(ber - theory) / theory < 0.25

    # Flat fade: add the noise before derotation. Derotating first and then
    # adding the unfaded noise variance would invent the wrong SNR.
    sent = rng.integers(0, 4, 40_000)
    sym = psk(4)[sent]
    fade = rayleigh(sent.size, rng)
    y = derotate(add_awgn(fade * sym, esn0_db=12, rng=rng), fade)
    faded = float(np.mean(ml_detect(y, psk(4)) != sent))
    clean = float(np.mean(ml_detect(add_awgn(sym, 12, rng), psk(4)) != sent))
    assert faded > clean
    print("chapter 4 self-check: ok")
    print(f"BPSK @ 6 dB: simulated {ber:.4e}, theory {theory:.4e}")
    print(f"QPSK @ 12 dB Es/N0: faded SER {faded:.4e}, AWGN SER {clean:.4e}")


if __name__ == "__main__":
    main()
