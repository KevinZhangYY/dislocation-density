from __future__ import annotations

from pathlib import Path

import numpy as np


def render_overlay_rgb(roi: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    from skimage.morphology import dilation, square

    base = _to_u8(roi)
    rgb = np.stack([base, base, base], axis=-1)
    
    # Dilate the skeleton to make it thicker (3x3 kernel)
    thick_skel = dilation(skeleton.astype(bool), square(3))
    
    # Use bright red for high contrast
    mask = thick_skel > 0
    rgb[mask, 0] = 255
    rgb[mask, 1] = 0
    rgb[mask, 2] = 0
    
    return rgb


def render_qc_panel(
    raw: np.ndarray,
    roi: np.ndarray,
    enhanced: np.ndarray,
    mask: np.ndarray,
    skeleton: np.ndarray,
    out_path: Path,
    dpi: int,
) -> None:
    # Use PIL instead of matplotlib to save size
    from PIL import Image, ImageDraw, ImageFont

    # Create a 2x2 grid
    # Define cell size (e.g. 300x300)
    cell_w, cell_h = 400, 400
    grid_w, grid_h = cell_w * 2, cell_h * 2
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    
    def _prep(arr, label):
        # Resize to cell size
        # Normalize first
        u8 = _to_u8(arr)
        img = Image.fromarray(u8)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        
        # Add label
        draw = ImageDraw.Draw(img)
        # Draw text at top-left
        draw.text((10, 10), label, fill=(255, 0, 0)) # Red text
        return img

    # 1. Raw
    img_raw = _prep(raw, "Raw")
    canvas.paste(img_raw, (0, 0))
    
    # 2. ROI
    img_roi = _prep(roi, "ROI")
    canvas.paste(img_roi, (cell_w, 0))
    
    # 3. Enhanced
    img_enh = _prep(enhanced, "Enhanced")
    canvas.paste(img_enh, (0, cell_h))
    
    # 4. Mask+Skel
    overlay = render_overlay_rgb(roi, skeleton)
    img_ov = Image.fromarray(overlay).resize((cell_w, cell_h), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img_ov)
    draw.text((10, 10), "Mask+Skeleton", fill=(255, 255, 0))
    canvas.paste(img_ov, (cell_w, cell_h))
    
    canvas.save(out_path, format="PNG")


def _to_u8(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    if p99 > p1:
        arr = (arr - p1) / (p99 - p1)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)
