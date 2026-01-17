from __future__ import annotations

import numpy as np
from skimage import measure, morphology

from ..config import M4Config


def segment_and_clean(enhanced: np.ndarray, cfg: M4Config) -> np.ndarray:
    thr = float(np.quantile(enhanced, float(cfg.threshold_quantile)))
    mask = enhanced > thr

    if int(cfg.morph_open_radius_px) > 0:
        mask = morphology.opening(mask, morphology.disk(int(cfg.morph_open_radius_px)))
    if int(cfg.morph_close_radius_px) > 0:
        mask = morphology.closing(mask, morphology.disk(int(cfg.morph_close_radius_px)))

    mask = morphology.remove_small_objects(mask, min_size=int(cfg.min_size_px), connectivity=2)

    if cfg.long_line_filter_enabled:
        mask = _filter_long_lines(mask, cfg)

    return mask.astype(bool)


def _filter_long_lines(mask: np.ndarray, cfg: M4Config) -> np.ndarray:
    lab = measure.label(mask, connectivity=2)
    if lab.max() == 0:
        return mask

    props = measure.regionprops(lab)
    keep = np.ones(lab.max() + 1, dtype=bool)
    for p in props:
        if p.major_axis_length > float(cfg.long_line_max_length_px) and p.eccentricity >= float(
            cfg.long_line_min_eccentricity
        ):
            keep[p.label] = False

    filtered = keep[lab]
    return filtered
