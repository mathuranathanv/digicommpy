# digicommpy

Companion code for *Digital Modulations using Python* by Mathuranathan Viswanathan.

The package holds the signal, modem, synchronization, equalizer, impairment, OFDM, MIMO, and coding routines used in the book, plus a short check script for each chapter that has one.

## Install

Python 3.10 or newer.

```bash
pip install -e .
```

After that, `import digicommpy` works from any directory. Do not add the folder with `sys.path`.

## Layout

| Module | Contents |
|---|---|
| `signals` | Test signals, spectra, and the analytic signal |
| `passband` | Pulse shaping and passband waveforms |
| `baseband` | Constellations |
| `performance` | Channels and error rates |
| `sync` | Timing, carrier, and frame synchronization |
| `equalize` | Linear equalizers and frequency-domain equalization |
| `impair` | DC, gain, and phase impairments |
| `ofdm` | OFDM modem |
| `mimo` | Spatial multiplexing |
| `coding` | Repetition and block coding |
| `chapter1` … | Checks that follow the printed listings |

Tested with NumPy 1.24 or newer, SciPy 1.11 or newer, and Matplotlib 3.8 or newer.
