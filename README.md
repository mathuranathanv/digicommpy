# digicommpy

Companion code for *Digital Modulations using Python* by Mathuranathan Viswanathan.

The package holds the routines the chapters run: test signals, passband and baseband modems, error rates, synchronization, equalizers, RF impairments, OFDM, MIMO, and coding. A few chapters also include a short check script that reruns the printed listings.

Released under the MIT License. See [LICENSE](LICENSE).

## Install

Python 3.10 or newer. NumPy, SciPy, and Matplotlib are installed with the package.

```bash
pip install git+https://github.com/mathuranathanv/digicommpy.git
```

After that, `import digicommpy` works from any directory. Do not add the folder with `sys.path`.

From a checkout of this repository, the same install is:

```bash
pip install -e .
```

`pip install digicommpy` asks PyPI for a release. Nothing is uploaded there. The version in this repository is `2.0.0a1`, and pip does not select a pre-release unless the version is requested, so that short command will not install this package.

## Chapters

Each folder under `digicommpy/` is one part of the book.

| Folder | Chapter |
|---|---|
| `signals` | 1. Essentials of Signal Processing |
| `passband` | 2. Passband Modulators and Demodulators |
| `baseband` | 3. Complex Baseband Modems |
| `performance` | 4. Uncoded Performance |
| `sync` | 5. Synchronization Essentials |
| `equalize` | 6. Equalization |
| `impair` | 7. RF Impairments |
| `ofdm` | 8. OFDM |
| `mimo` | 9. MIMO, an Introduction |
| `coding` | 10. From Uncoded Error Rates to Coded Ones |

`chapter1`, `chapter2`, `chapter4`, `chapter5`, and `chapter6` are check scripts for those chapters. Chapters 3, 7, 8, 9, and 10 have no separate check script. The functions those chapters call live in the folders in the table.

Tested with NumPy 1.24 or newer, SciPy 1.11 or newer, and Matplotlib 3.8 or newer.
