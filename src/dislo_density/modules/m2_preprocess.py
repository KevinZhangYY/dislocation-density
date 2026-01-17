from __future__ import annotations

import numpy as np
from skimage import filters, morphology, restoration

from ..config import M2Config


def preprocess(roi: np.ndarray, cfg: M2Config) -> np.ndarray:
    img = roi.astype(np.float32)

    if cfg.background_suppression_enabled:
        if cfg.background_method == "rolling_ball":
            bg = restoration.rolling_ball(img, radius=int(cfg.rolling_ball_radius_px))
            img = img - bg
        elif cfg.background_method == "top_hat":
            selem = morphology.disk(int(cfg.tophat_radius_px))
            img = morphology.black_tophat(img, selem)
        else:
            raise ValueError(f"Unknown background_method: {cfg.background_method}")

    img = filters.gaussian(img, sigma=float(cfg.gaussian_sigma), preserve_range=True)

    p1, p99 = np.percentile(img, [1, 99])
    if p99 > p1:
        img = (img - p1) / (p99 - p1)
    img = np.clip(img, 0.0, 1.0)
    return img
