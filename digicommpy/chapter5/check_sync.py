"""Numerical checks for Chapter 5. No plots."""
from __future__ import annotations

import numpy as np

from digicommpy.baseband import psk
from digicommpy.performance.error_rates import qfunc
from digicommpy.sync import (
    fourth_power_cfo,
    gardner_scurve,
    normalized_correlation,
    phase_errors,
    rrc_matched,
    timing_loop,
)


def main() -> None:
    rng = np.random.default_rng(5)
    sps, span, beta = 8, 8, 0.5
    nsym = 800
    symbols = psk(4)[rng.integers(0, 4, nsym)]
    wave, delay = rrc_matched(symbols, sps, beta, span)

    # A single impulse peaks at `delay`.
    impulse, delay_i = rrc_matched(np.array([1.0 + 0j]), sps, beta, span)
    assert int(np.argmax(np.abs(impulse))) == delay_i

    k0, k1 = span + 2, nsym - span - 2
    taus = np.linspace(-0.4, 0.4, 9)
    curve = np.array([gardner_scurve(wave, delay, sps, t, k0, k1) for t in taus])
    # Odd, and a positive late sample produces a positive error.
    assert curve[0] < 0 < curve[-1]
    assert abs(curve[len(curve) // 2]) < 0.02
    slope = (curve[len(curve) // 2 + 1] - curve[len(curve) // 2 - 1]) / (
        taus[len(taus) // 2 + 1] - taus[len(taus) // 2 - 1]
    )
    assert slope > 0

    # A fixed carrier phase does not move the S-curve.
    rotated, delay_r = rrc_matched(symbols * np.exp(1j * 0.7), sps, beta, span)
    plain = gardner_scurve(wave, delay, sps, 0.2, k0, k1)
    turned = gardner_scurve(rotated, delay_r, sps, 0.2, k0, k1)
    assert abs(plain - turned) / abs(plain) < 0.05

    path = timing_loop(wave, delay, sps, tau0=0.2, mu=0.3, k0=k0, k1=k0 + 400)
    assert abs(path[-1]) < 0.03

    # Wider excess bandwidth, steeper detector.
    wide, delay_w = rrc_matched(symbols, sps, 0.5, span)
    narrow, delay_n = rrc_matched(symbols, sps, 0.2, span)
    g_wide = gardner_scurve(wide, delay_w, sps, 0.15, k0, k1)
    g_narrow = gardner_scurve(narrow, delay_n, sps, 0.15, k0, k1)
    assert g_wide > g_narrow > 0

    sent = psk(4)[rng.integers(0, 4, 2000)]
    truth = 0.4
    phases = phase_errors(sent * np.exp(1j * truth), mu=0.2)
    assert abs(phases[-1] - truth) < 0.05

    n = np.arange(4000)
    df = 0.01
    spun = psk(4)[rng.integers(0, 4, n.size)] * np.exp(1j * 2 * np.pi * df * n)
    est = fourth_power_cfo(spun, nfft=8 * spun.size)
    assert abs(est - df) < 1e-3

    preamble = psk(4)[rng.integers(0, 4, 32)]
    start = 100
    payload = psk(4)[rng.integers(0, 4, 200)]
    record = np.concatenate([psk(4)[rng.integers(0, 4, start)], preamble, payload])
    corr = normalized_correlation(record, preamble)
    assert int(np.argmax(corr)) == start
    assert corr[start] > 0.99

    word = psk(4)[rng.integers(0, 4, 16)]
    theta_hat = truth + np.pi / 2
    y = word * np.exp(1j * truth) * np.exp(-1j * theta_hat)
    # Magnitude cannot see a quadrant slip: every rotation has the same |correlation|.
    # The angle of the correlation is the slip.
    angle = float(np.angle(np.sum(y * np.conj(word))))
    assert abs(angle - (-np.pi / 2)) < 1e-8
    gamma = 10 ** (8 / 10)
    theta = np.deg2rad(20)
    predicted = float(qfunc(np.sqrt(2 * gamma) * np.cos(theta)))
    assert predicted > float(qfunc(np.sqrt(2 * gamma)))

    print("chapter 5 self-check: ok")
    print(f"Gardner slope at 0 (beta=0.5) = {slope:.3f}")
    print(f"error at +0.15 T: beta 0.5 = {g_wide:.4f}, beta 0.2 = {g_narrow:.4f}")
    print(f"timing loop ended at {path[-1]:+.4f} symbols")
    print(f"phase loop ended at {phases[-1]:.4f} rad (truth {truth})")
    print(f"fourth-power CFO {est:.5f} cycles/symbol (truth {df})")
    print(f"BPSK at 8 dB, 20 deg: BER {predicted:.4e}")


if __name__ == "__main__":
    main()
