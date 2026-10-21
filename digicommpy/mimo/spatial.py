"""A few spatial streams on one time-frequency slot."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def zf(H: NDArray, Y: NDArray) -> NDArray:
    """Separate streams by the pseudoinverse. Y is Nr x N, return is Nt x N."""
    W = np.linalg.pinv(H)
    return W @ Y


def mmse(H: NDArray, Y: NDArray, n0: float) -> NDArray:
    """MMSE combiner. n0 is the noise variance on each receive antenna."""
    Ht = H.conj().T
    gram = Ht @ H + n0 * np.eye(H.shape[1])
    W = np.linalg.solve(gram, Ht)
    return W @ Y
