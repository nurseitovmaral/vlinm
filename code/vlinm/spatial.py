from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def knn_indices(coordinates: NDArray[np.floating], k: int) -> NDArray[np.int64]:
    xy = np.asarray(coordinates, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("coordinates must be finite and have shape (n, 2)")
    if not 1 <= k < len(xy):
        raise ValueError("k must be positive and smaller than the number of points")
    distance2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(distance2, np.inf)
    return np.argsort(distance2, axis=1, kind="stable")[:, :k].astype(np.int64)


def weighted_graph(coordinates: NDArray[np.floating], k: int = 15) -> tuple[NDArray[np.int64], NDArray[np.float64], float]:
    xy = np.asarray(coordinates, dtype=np.float64)
    neighbours = knn_indices(xy, k)
    distance2 = ((xy[:, None, :] - xy[neighbours]) ** 2).sum(axis=-1)
    sigma = float(np.median(np.sqrt(distance2[:, 0])))
    if sigma <= 0:
        raise ValueError("median nearest-neighbour distance must be positive")
    weights = np.exp(-distance2 / (2 * sigma**2))
    return neighbours, weights, sigma
