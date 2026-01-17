from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve
from skimage import morphology

from ..config import M5Config


def skeletonize_and_measure(mask: np.ndarray, cfg: M5Config) -> tuple[np.ndarray, float]:
    skel = morphology.skeletonize(mask > 0)
    skel = skel.astype(bool)

    if cfg.prune_branches_enabled and int(cfg.prune_length_px) > 0:
        skel = _prune_endpoints(skel, int(cfg.prune_length_px))

    length_px = _skeleton_length_px(skel)
    return skel, float(length_px)


def _prune_endpoints(skel: np.ndarray, iterations: int) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    out = skel.copy()
    for _ in range(int(iterations)):
        n = convolve(out.astype(np.uint8), kernel, mode="constant", cval=0)
        neighbor_count = n - out.astype(np.uint8)
        endpoints = out & (neighbor_count == 1)
        if not endpoints.any():
            break
        out = out & (~endpoints)
    return out


def _skeleton_length_px(skel: np.ndarray) -> float:
    sk = skel.astype(bool)

    right = sk[:, :-1] & sk[:, 1:]
    down = sk[:-1, :] & sk[1:, :]
    down_right = sk[:-1, :-1] & sk[1:, 1:]
    down_left = sk[:-1, 1:] & sk[1:, :-1]

    length = (
        right.sum(dtype=np.float64)
        + down.sum(dtype=np.float64)
        + np.sqrt(2.0) * (down_right.sum(dtype=np.float64) + down_left.sum(dtype=np.float64))
    )
    return float(length)
