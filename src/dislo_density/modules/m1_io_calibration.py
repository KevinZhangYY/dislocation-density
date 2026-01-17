from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from skimage import measure

from ..config import M1Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class M1Result:
    raw: np.ndarray
    rois: list[np.ndarray]
    roi_boxes: list[tuple[int, int, int, int]]
    nm_per_px: float
    nm_per_px_source: str
    metadata: dict[str, Any]
    scale_bar_bbox: tuple[int, int, int, int] | None = None


def load_image_and_calibrate(path: Path, cfg: M1Config) -> M1Result:
    raw, meta = _read_grayscale_preserve(path)

    rois, roi_boxes = _resolve_rois(raw, cfg)

    nm_per_px, source, sb_bbox = _calibrate_nm_per_px(raw=raw, roi=rois[0], meta=meta, cfg=cfg)
    return M1Result(
        raw=raw,
        rois=rois,
        roi_boxes=roi_boxes,
        nm_per_px=nm_per_px,
        nm_per_px_source=source,
        metadata=meta,
        scale_bar_bbox=sb_bbox,
    )


def _resolve_rois(raw: np.ndarray, cfg: M1Config) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    h, w = raw.shape
    if cfg.rois:
        rois: list[np.ndarray] = []
        boxes: list[tuple[int, int, int, int]] = []
        for r in cfg.rois:
            x0 = int(np.clip(int(r.get("x0", 0)), 0, w - 1))
            y0 = int(np.clip(int(r.get("y0", 0)), 0, h - 1))
            x1 = int(np.clip(int(r.get("x1", w)), x0 + 1, w))
            y1 = int(np.clip(int(r.get("y1", h)), y0 + 1, h))
            rois.append(raw[y0:y1, x0:x1])
            boxes.append((x0, y0, x1, y1))
        if not rois:
            raise ValueError("m1.rois provided but empty")
        return rois, boxes

    if cfg.auto_roi:
        roi, roi_box = _auto_crop_roi(raw, cfg)
        return [roi], [roi_box]

    return [raw], [(0, 0, w, h)]


def _read_grayscale_preserve(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        arr = page.asarray()
        meta: dict[str, Any] = {}
        try:
            meta["tiff_tags"] = {t.name: t.value for t in page.tags.values()}
        except Exception:
            meta["tiff_tags"] = {}
        if tf.imagej_metadata is not None:
            meta["imagej_metadata"] = dict(tf.imagej_metadata)
        if tf.ome_metadata is not None:
            meta["ome_metadata"] = tf.ome_metadata
        try:
            desc = page.description
        except Exception:
            desc = None
        if desc:
            meta["image_description"] = desc

    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        else:
            arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape={arr.shape}")

    return arr, meta


def _auto_crop_roi(raw: np.ndarray, cfg: M1Config) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = raw.shape
    border = max(int(min(h, w) * float(cfg.roi_border_fraction)), int(cfg.roi_border_px))
    bottom_drop = max(int(h * float(cfg.roi_drop_bottom_fraction)), int(cfg.roi_drop_bottom_px))

    x0 = int(np.clip(border, 0, w - 1))
    y0 = int(np.clip(border, 0, h - 1))
    x1 = int(np.clip(w - border, x0 + 1, w))
    y1 = int(np.clip(h - border - bottom_drop, y0 + 1, h))

    roi = raw[y0:y1, x0:x1]
    return roi, (x0, y0, x1, y1)


def _calibrate_nm_per_px(
    raw: np.ndarray,
    roi: np.ndarray,
    meta: dict[str, Any],
    cfg: M1Config,
) -> tuple[float, str]:
    if cfg.manual_nm_per_px is not None:
        if cfg.manual_nm_per_px <= 0:
            raise ValueError("m1.manual_nm_per_px must be > 0")
        return float(cfg.manual_nm_per_px), "manual"

    if cfg.prefer_metadata_nm_per_px:
        nm = _extract_nm_per_px_from_metadata(meta)
        if nm is not None:
            return float(nm), "metadata"

    if cfg.scale_bar_enabled:
        res = _extract_nm_per_px_from_scale_bar(raw, cfg)
        if res is not None:
            nm, bbox = res
            return float(nm), "scale_bar", bbox

    raise RuntimeError(
        "nm/px calibration failed: no usable metadata. "
        "Provide 'Manual nm/px' directly, OR provide 'Scale Bar Value (nm)' so we can measure the bar pixels."
    )


def _extract_nm_per_px_from_metadata(meta: dict[str, Any]) -> float | None:
    candidates: list[str] = []
    tags = meta.get("tiff_tags") or {}
    for k in ("XResolution", "YResolution", "ResolutionUnit"):
        if k in tags:
            candidates.append(f"{k}={tags[k]}")
    if "image_description" in meta:
        candidates.append(str(meta["image_description"]))
    if "imagej_metadata" in meta:
        candidates.extend([f"{k}={v}" for k, v in (meta["imagej_metadata"] or {}).items()])

    text_blob = "\n".join(map(str, candidates))
    nm = _parse_length_to_nm(_find_first_length_token(text_blob))
    if nm is not None:
        return nm

    return None


def _find_first_length_token(text: str) -> tuple[float, str] | None:
    patterns = [
        r"(?:nm[/ ]?px|nm per pixel)\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"(?:pixel\s*size|pixelsize|pix\s*size)\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(nm|um|µm|μm|pm|a|å|angstrom)",
        r"([0-9]*\.?[0-9]+)\s*(nm|um|µm|μm|pm|a|å|angstrom)\s*(?:/px|per\s*pixel)",
    ]
    lower = text.lower()
    for pat in patterns:
        m = re.search(pat, lower, flags=re.IGNORECASE)
        if not m:
            continue
        if m.lastindex == 1:
            return float(m.group(1)), "nm"
        if m.lastindex and m.lastindex >= 2:
            return float(m.group(1)), str(m.group(2))
    return None


def _parse_length_to_nm(token: tuple[float, str] | None) -> float | None:
    if token is None:
        return None
    value, unit = token
    u = unit.lower()
    if u in {"nm"}:
        return float(value)
    if u in {"um", "µm", "μm"}:
        return float(value) * 1_000.0
    if u in {"pm"}:
        return float(value) * 0.001
    if u in {"a", "å", "angstrom"}:
        return float(value) * 0.1
    return None


def _extract_nm_per_px_from_scale_bar(raw: np.ndarray, cfg: M1Config) -> tuple[float, tuple[int, int, int, int] | None] | None:
    h, w = raw.shape
    band_h = max(32, int(h * float(cfg.scale_bar_search_height_fraction)))
    y0 = max(0, h - band_h)
    band = raw[y0:h, :]
    
    bar_len_px = None
    band_bbox = None
    
    # 1. Try traditional region-props based detection (for white/black bars)
    ret = _detect_bar_regionprops(band, cfg)
    if ret:
        bar_len_px, band_bbox = ret
    
    # 2. If that fails, try morphological horizontal line detection (specifically for black lines)
    if bar_len_px is None:
        ret = _detect_bar_morphology(band, cfg)
        if ret:
            bar_len_px, band_bbox = ret
        
    if bar_len_px is None:
        return None

    # Adjust bbox to global coordinates
    # band_bbox is (minr, minc, maxr, maxc) relative to band
    # global bbox needs y+y0
    minr, minc, maxr, maxc = band_bbox if band_bbox else (0,0,0,0)
    global_bbox = (minr + y0, minc, maxr + y0, maxc)

    # If user provided manual length (e.g. from UI input), use it
    if cfg.manual_scale_bar_nm is not None and cfg.manual_scale_bar_nm > 0:
        length_nm = float(cfg.manual_scale_bar_nm)
        nm_per_px = length_nm / float(bar_len_px)
        logger.info("Scale bar: manual %.1f nm over auto-detected %d px => %.6f nm/px", length_nm, bar_len_px, nm_per_px)
        return nm_per_px, global_bbox

    # If no manual length, we cannot proceed as OCR is removed
    return None


def _detect_bar_regionprops(band: np.ndarray, cfg: M1Config) -> tuple[int, tuple[int, int, int, int]] | None:
    band_f = band.astype(np.float32)
    p1, p99 = np.percentile(band_f, [1, 99])
    if p99 <= p1:
        return None
    norm = (band_f - p1) / (p99 - p1)
    norm = np.clip(norm, 0.0, 1.0)

    q_hi = float(cfg.scale_bar_threshold_quantile)
    q_hi = float(np.clip(q_hi, 0.5, 0.9999))
    thr_hi = float(np.quantile(norm, q_hi))
    thr_lo = float(np.quantile(norm, 1.0 - q_hi))

    bw_hi = norm >= thr_hi
    bw_lo = norm <= thr_lo

    cand = _pick_scale_bar_component(bw_hi, cfg.scale_bar_min_aspect) or _pick_scale_bar_component(
        bw_lo, cfg.scale_bar_min_aspect
    )
    if cand is None:
        return None

    (minr, minc, maxr, maxc), _ = cand
    width = maxc - minc
    return (width, (minr, minc, maxr, maxc)) if width > 0 else None


def _detect_bar_morphology(band: np.ndarray, cfg: M1Config) -> tuple[int, tuple[int, int, int, int]] | None:
    """
    Detects long horizontal black lines using morphological opening.
    Steps:
    1. Binarize (black is foreground)
    2. Open with long horizontal kernel
    3. Find longest component
    """
    from skimage import morphology
    
    # 1. Binarize: assume bar is dark. 
    # Use simple percentile threshold or Otsu. Let's use percentile for robustness against shading.
    # Dark pixels are < 20th percentile (adjustable)
    thresh = np.percentile(band, 20)
    binary = band < thresh # True for dark pixels
    
    # 2. Morphological opening with horizontal line
    # Length of kernel: min_size for a bar. say 50px or 1/10 of width
    min_len = max(30, band.shape[1] // 20)
    selem = morphology.rectangle(1, min_len) 
    
    # Since 'binary' has True for objects, we use binary_opening
    opened = morphology.binary_opening(binary, selem)
    
    # 3. Label and find components
    lab = measure.label(opened)
    if lab.max() == 0:
        return None
        
    props = measure.regionprops(lab)
    best_len = 0
    best_bbox = None
    
    for p in props:
        minr, minc, maxr, maxc = p.bbox
        width = maxc - minc
        height = maxr - minr
        
        # Must be very flat
        if height > 15: # Arbitrary thickness limit
            continue
            
        aspect = width / max(1, height)
        if aspect < cfg.scale_bar_min_aspect:
            continue
            
        if width > best_len:
            best_len = width
            best_bbox = (minr, minc, maxr, maxc)
            
    return (best_len, best_bbox) if best_len > 0 and best_bbox else None


def _pick_scale_bar_component(bw: np.ndarray, min_aspect: float) -> tuple[tuple[int, int, int, int], float] | None:
    lab = measure.label(bw, connectivity=2)
    if lab.max() == 0:
        return None
    props = measure.regionprops(lab)
    best = None
    best_score = -1.0
    for p in props:
        minr, minc, maxr, maxc = p.bbox
        hh = maxr - minr
        ww = maxc - minc
        if hh <= 0 or ww <= 0:
            continue
        aspect = ww / float(hh)
        if aspect < float(min_aspect):
            continue
        area = float(p.area)
        score = area * aspect
        if score > best_score:
            best_score = score
            best = ((minr, minc, maxr, maxc), score)
    return best
