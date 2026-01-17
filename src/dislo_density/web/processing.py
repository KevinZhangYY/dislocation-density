from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import AppConfig
from ..modules.m1_io_calibration import load_image_and_calibrate
from ..modules.m2_preprocess import preprocess
from ..modules.m3_line_enhance import enhance_lines
from ..modules.m4_segment import segment_and_clean
from ..modules.m5_skeleton_length import skeletonize_and_measure
from ..modules.m6_metrics import compute_metrics
from ..modules.m7_visualize import render_overlay_rgb


@dataclass(frozen=True)
class StepArtifacts:
    raw: np.ndarray
    roi: np.ndarray
    pre: np.ndarray
    enhanced: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    overlay_rgb: np.ndarray


def run_steps_from_path(image_path: str, cfg: AppConfig, roi_id: int = 0) -> dict[str, Any]:
    m1 = load_image_and_calibrate(path=_as_path(image_path), cfg=cfg.m1)
    if roi_id < 0 or roi_id >= len(m1.rois):
        raise ValueError("roi_id out of range")

    raw = m1.raw
    roi = m1.rois[roi_id]
    roi_box = m1.roi_boxes[roi_id]

    # Visualize scale bar if detected
    raw_vis = raw
    if m1.scale_bar_bbox:
        try:
            from skimage import color, draw
            
            # Convert to RGB for visualization
            if raw.ndim == 2:
                # Check dtype to ensure correct conversion/range
                if raw.dtype == np.uint8:
                    raw_vis = color.gray2rgb(raw)
                    green = (0, 255, 0)
                elif raw.dtype == np.uint16:
                    # gray2rgb preserves uint16
                    raw_vis = color.gray2rgb(raw)
                    green = (0, 65535, 0)
                elif np.issubdtype(raw.dtype, np.floating):
                    raw_vis = color.gray2rgb(raw)
                    mx = raw_vis.max()
                    green = (0.0, 1.0 if mx <= 1.0 else mx, 0.0)
                else:
                    # Fallback: normalize to uint8
                    p1, p99 = np.percentile(raw, [1, 99])
                    norm = np.clip((raw - p1) / (p99 - p1), 0, 1)
                    raw_vis = color.gray2rgb((norm * 255).astype(np.uint8))
                    green = (0, 255, 0)
            else:
                raw_vis = raw.copy()
                green = (0, 255, 0) # Assumption

            minr, minc, maxr, maxc = m1.scale_bar_bbox
            h, w = raw_vis.shape[:2]
            
            # Draw a thick rectangle
            rr, cc = draw.rectangle_perimeter(
                start=(minr, minc), 
                end=(maxr, maxc), 
                shape=(h, w), 
                clip=True
            )
            
            # Thicken the line
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r_off = np.clip(rr + dr, 0, h - 1)
                    c_off = np.clip(cc + dc, 0, w - 1)
                    raw_vis[r_off, c_off] = green

        except Exception:
            # Fallback to original if drawing fails
            raw_vis = raw

    pre = preprocess(roi, cfg.m2)
    enhanced = enhance_lines(pre, cfg.m3)
    mask = segment_and_clean(enhanced, cfg.m4)
    skeleton, length_px = skeletonize_and_measure(mask, cfg.m5)
    overlay_rgb = render_overlay_rgb(roi, skeleton)

    metrics = compute_metrics(
        length_px=float(length_px),
        roi_shape=roi.shape,
        nm_per_px=float(m1.nm_per_px),
        thickness_nm=cfg.m6.thickness_nm,
    )

    return {
        "nm_per_px": float(m1.nm_per_px),
        "nm_per_px_source": m1.nm_per_px_source,
        "roi_box_xyxy": [int(v) for v in roi_box],
        "roi_id": int(roi_id),
        "roi_count": int(len(m1.rois)),
        "length_px": float(length_px),
        "metrics": {k: (None if v is None else float(v)) for k, v in metrics.items()},
        "steps": StepArtifacts(
            raw=raw_vis,
            roi=roi,
            pre=pre,
            enhanced=enhanced,
            mask=mask.astype(np.uint8) * 255,
            skeleton=skeleton.astype(np.uint8) * 255,
            overlay_rgb=overlay_rgb,
        ),
    }


def encode_png_base64(arr: np.ndarray, cmap: str | None = None, max_dim: int = 1200) -> str:
    from PIL import Image
    
    img = _maybe_downscale(arr, max_dim=max_dim)
    
    # Handle colormaps simply by normalizing to 0-255 grayscale or simple RGB mapping
    # Since we removed matplotlib, we can't use 'magma' easily.
    # For now, we'll stick to grayscale for 'gray' cmap and a simple reddish map for 'magma' if needed,
    # or just grayscale for robustness.
    
    # Normalize to 0-255 uint8
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
             img = np.clip(img, 0.0, 1.0) * 255.0
        else:
             # Normalize min-max
             mn, mx = img.min(), img.max()
             if mx > mn:
                 img = (img - mn) / (mx - mn) * 255.0
             else:
                 img = img * 0
        img = img.astype(np.uint8)

    if cmap == "magma":
        # Simple pseudo-magma (black -> purple -> orange -> yellow)
        # We can just return grayscale for now to save size/complexity, 
        # or map it using a small LUT. Let's use grayscale for Vercel efficiency.
        pass

    pil_img = Image.fromarray(img)
    
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _maybe_downscale(arr: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = arr.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return arr
    scale = max_dim / float(m)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    try:
        from skimage.transform import resize
    except Exception:
        step = int(np.ceil(m / max_dim))
        return arr[::step, ::step]

    if arr.ndim == 2:
        out = resize(arr, (new_h, new_w), preserve_range=True, anti_aliasing=True)
        return out.astype(arr.dtype, copy=False)
    out = resize(arr, (new_h, new_w, arr.shape[2]), preserve_range=True, anti_aliasing=True)
    return out.astype(arr.dtype, copy=False)


def _as_path(p: str):
    from pathlib import Path

    return Path(p)
