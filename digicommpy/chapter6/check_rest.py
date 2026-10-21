"""Checks for chapters 6 through 10."""
from __future__ import annotations

import numpy as np

from digicommpy.baseband import psk, qam
from digicommpy.coding import (
    block_deinterleave,
    block_interleave,
    hamming74_decode,
    hamming74_encode,
    hard_repetition_ber,
    repetition_decode,
    repetition_encode,
)
from digicommpy.equalize import (
    add_cyclic_prefix,
    apply_equalizer,
    dfe_bpsk,
    lms,
    mmse_fde,
    mmse_taps,
    through_channel,
    zf_taps,
)
from digicommpy.impair import apply_dc, apply_iq, compensate_iq_known, phase_walk, remove_dc
from digicommpy.mimo import mmse, zf
from digicommpy.ofdm import channel_bins, demodulate, modulate
from digicommpy.performance.error_rates import ser_bpsk


def check_equalize() -> None:
    rng = np.random.default_rng(6)
    h = np.array([0.2, 1.0, 0.3])
    h = h / np.linalg.norm(h)
    w, delay = zf_taps(h, n_taps=11)
    cascade = np.convolve(h, w)
    assert abs(cascade[delay] - 1) < 1e-4
    assert np.max(np.abs(np.delete(cascade, delay))) < 1e-3

    symbols = psk(4)[rng.integers(0, 4, 2000)]
    n0 = 10 ** (-12 / 10)
    y = apply_equalizer(symbols, h, w, delay, n0, rng)
    raw = np.convolve(symbols, h)[: symbols.size]
    err_zf = float(np.mean(np.abs(y - symbols) ** 2))
    err_raw = float(np.mean(np.abs(raw - symbols) ** 2))
    assert err_zf < err_raw

    notch = np.array([1.0, -0.95])
    notch = notch / np.linalg.norm(notch)
    wz, dz = zf_taps(notch, n_taps=15)
    wm = mmse_taps(notch, 15, n0=10 ** (-8 / 10), delay=dz)
    bits = rng.integers(0, 2, 8000)
    s = (1 - 2 * bits).astype(complex)
    yz = apply_equalizer(s, notch, wz, dz, 10 ** (-8 / 10), rng)
    ym = apply_equalizer(s, notch, wm, dz, 10 ** (-8 / 10), rng)
    ez = float(np.mean(np.sign(yz.real) != s.real))
    em = float(np.mean(np.sign(ym.real) != s.real))
    assert em < ez

    h2 = np.array([1.0, 0.8])
    h2 = h2 / np.linalg.norm(h2)
    r = np.convolve(s[:4000], h2)[:4000]
    r = r + np.sqrt(10 ** (-10 / 10) / 2) * rng.normal(size=r.size)
    dfe = dfe_bpsk(r, h2)
    # A slicer that ignores the post-cursor.
    naive = np.sign(r.real)
    assert np.mean(dfe != s[:4000].real) < np.mean(naive != s[:4000].real)

    train = psk(4)[rng.integers(0, 4, 12000)]
    received = np.convolve(train, h)
    _, power = lms(received, train, n_taps=11, mu=0.05, delay=delay)
    assert np.mean(power[:40]) > 0.2
    assert np.mean(power[-300:]) < 1e-3

    block = psk(4)[rng.integers(0, 4, 64)]
    cp = 4
    tx = add_cyclic_prefix(block, cp)
    body = through_channel(tx, h, block.size, cp)
    back = mmse_fde(body, h, n0=0.0)
    assert np.max(np.abs(back - block)) < 1e-8
    print("chapter 6 self-check: ok")
    print(f"ZF delay {delay}, notch BER ZF {ez:.3f} MMSE {em:.3f}")


def check_impair() -> None:
    rng = np.random.default_rng(7)
    s = qam(16)[rng.integers(0, 16, 4000)]
    z = apply_dc(apply_iq(s, g=0.8, phi=np.deg2rad(12)), 0.2, -0.15)
    y = compensate_iq_known(remove_dc(z), g=0.8, phi=np.deg2rad(12))
    assert np.mean(np.abs(y - s) ** 2) < 1e-4
    walked = phase_walk(psk(4)[rng.integers(0, 4, 500)], sigma=0.02, rng=rng)
    assert np.mean(np.abs(walked) ** 2) > 0.9
    print("chapter 7 self-check: ok")


def check_ofdm() -> None:
    rng = np.random.default_rng(8)
    nfft, cp = 64, 8
    h = np.array([1.0, 0.4, 0.2, -0.1])
    h = h / np.linalg.norm(h)
    X = psk(4)[rng.integers(0, 4, nfft)]
    Y = demodulate(modulate(X, cp), nfft, cp)  # no channel: ifft/fft round trip, prefix dropped
    # modulate puts prefix on ifft(X); demodulate drops cp samples of the time signal.
    # Without a channel the body is exactly ifft(X). 
    assert np.max(np.abs(Y - X)) < 1e-8

    from digicommpy.equalize.fde import through_channel

    tx = modulate(X, cp)
    y = through_channel(tx, h, nfft, cp)
    H = channel_bins(h, nfft)
    hat = demodulate(y, nfft, 0, H)
    assert np.max(np.abs(hat - X)) < 1e-8

    # A missing prefix leaves the tail of the previous symbol in this one.
    y_short = through_channel(modulate(X, cp=0), h, nfft, cp=0)
    hat_short = demodulate(y_short, nfft, 0, H)
    assert np.mean(np.abs(hat_short - X) ** 2) > 1e-3
    print("chapter 8 self-check: ok")


def check_mimo() -> None:
    rng = np.random.default_rng(9)
    H = np.array([[1.0, 0.3], [0.2, 1.0]], dtype=complex)
    X = psk(4)[rng.integers(0, 4, (2, 1000))]
    Y = H @ X
    assert np.max(np.abs(zf(H, Y) - X)) < 1e-8
    n0 = 10 ** (-20 / 10)
    noise = np.sqrt(n0 / 2) * (rng.normal(size=Y.shape) + 1j * rng.normal(size=Y.shape))
    bad = np.array([[1.0, 1.0], [1.0, 1.01]], dtype=complex)
    good_err = float(np.mean(np.abs(mmse(H, Y + noise, n0) - X) ** 2))
    bad_err = float(np.mean(np.abs(mmse(bad, bad @ X + noise, n0) - X) ** 2))
    assert bad_err > 10 * good_err
    print("chapter 9 self-check: ok")
    print(f"MMSE error good H {good_err:.4e}, ill-conditioned {bad_err:.4e}")


def check_coding() -> None:
    rng = np.random.default_rng(10)
    bits = rng.integers(0, 2, 40)
    assert np.array_equal(repetition_decode(repetition_encode(bits, 3), 3), bits)
    data = rng.integers(0, 2, 16)
    code = hamming74_encode(data)
    flipped = code.copy()
    flipped[3] ^= 1
    assert np.array_equal(hamming74_decode(flipped), data)
    grid = rng.integers(0, 2, 12)
    assert np.array_equal(block_deinterleave(block_interleave(grid, 3, 4), 3, 4), grid)
    # At 12 dB the rate-1/3 code is still paying for the repeats.
    assert float(hard_repetition_ber(8)) > float(ser_bpsk(8))
    from digicommpy.coding import hard_repetition_ber_rayleigh
    from digicommpy.performance.error_rates import ber_rayleigh_bpsk
    assert float(hard_repetition_ber_rayleigh(20)) < float(ber_rayleigh_bpsk(20))
    print("chapter 10 self-check: ok")
    print(f"uncoded @ 8 dB {float(ser_bpsk(8)):.3e}, repetition-3 {float(hard_repetition_ber(8)):.3e}")
    print(
        f"Rayleigh @ 20 dB uncoded {float(ber_rayleigh_bpsk(20)):.3e}, "
        f"interleaved repetition-3 {float(hard_repetition_ber_rayleigh(20)):.3e}"
    )


if __name__ == "__main__":
    check_equalize()
    check_impair()
    check_ofdm()
    check_mimo()
    check_coding()
