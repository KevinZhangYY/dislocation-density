from __future__ import annotations

import numpy as np
from skimage import filters

from ..config import M3Config


def enhance_lines(img: np.ndarray, cfg: M3Config) -> np.ndarray:
    sigmas = np.linspace(float(cfg.sigmas_min), float(cfg.sigmas_max), int(cfg.sigmas_num))
    sigmas = [float(s) for s in sigmas]

    if cfg.method == "frangi":
        out = filters.frangi(img, sigmas=sigmas, black_ridges=bool(cfg.black_ridges))
    elif cfg.method == "sato":
        out = filters.sato(img, sigmas=sigmas, black_ridges=bool(cfg.black_ridges))
    else:
        raise ValueError(f"Unknown line enhancement method: {cfg.method}")

    out = np.asarray(out, dtype=np.float32)
    p1, p99 = np.percentile(out, [1, 99])
    if p99 > p1:
        out = (out - p1) / (p99 - p1)
    out = np.clip(out, 0.0, 1.0)
    return out
