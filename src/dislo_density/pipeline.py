from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import __version__
from .config import AppConfig
from .logging_utils import setup_logging
from .modules.m1_io_calibration import load_image_and_calibrate
from .modules.m2_preprocess import preprocess
from .modules.m3_line_enhance import enhance_lines
from .modules.m4_segment import segment_and_clean
from .modules.m5_skeleton_length import skeletonize_and_measure
from .modules.m6_metrics import compute_metrics
from .modules.m7_visualize import render_overlay_rgb, render_qc_panel

logger = logging.getLogger(__name__)


def _iter_images(input_path: Path, glob_pat: str, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    if recursive:
        return sorted(input_path.rglob(glob_pat))
    return sorted(input_path.glob(glob_pat))


import csv

def run_batch(input_path: Path, out_dir: Path, cfg: AppConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, cfg.m8.log_level)

    images = _iter_images(input_path, cfg.m8.glob, cfg.m8.recursive)
    if not images:
        raise FileNotFoundError(f"No images matched under: {input_path}")

    per_roi_rows: list[dict] = []

    for img_path in images:
        logger.info("Processing %s", img_path.name)
        record = run_single(img_path, out_dir, cfg)
        per_roi_rows.extend(record["per_roi_rows"])

    csv_path = out_dir / "summary.csv"
    if per_roi_rows:
        keys = per_roi_rows[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_roi_rows)
        logger.info("Wrote %s", csv_path.name)
        
        # Calculate stats (mean, std) for numeric columns
        # We'll do a simple pass
        # Identify numeric columns first
        numeric_keys = [
            k for k, v in per_roi_rows[0].items() 
            if isinstance(v, (int, float)) and k not in ("roi_id", "roi_x0", "roi_y0", "roi_x1", "roi_y1")
        ]
        
        if numeric_keys:
            import statistics
            stats_rows = []
            for k in numeric_keys:
                vals = [r[k] for r in per_roi_rows if r[k] is not None]
                if vals:
                    mean_val = statistics.mean(vals)
                    std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0
                    stats_rows.append({"metric": k, "mean": mean_val, "std": std_val})
            
            if stats_rows:
                stats_path = out_dir / "summary_stats.csv"
                with open(stats_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std"])
                    writer.writeheader()
                    writer.writerows(stats_rows)
                logger.info("Wrote %s", stats_path.name)


def run_single(image_path: Path, out_dir: Path, cfg: AppConfig) -> dict:
    image_name = image_path.stem
    per_image_dir = out_dir / image_name
    per_image_dir.mkdir(parents=True, exist_ok=True)

    m1 = load_image_and_calibrate(image_path, cfg.m1)
    raw = m1.raw

    per_roi_rows: list[dict] = []
    roi_payloads: list[dict] = []
    for roi_id, (roi, roi_box) in enumerate(zip(m1.rois, m1.roi_boxes)):
        pre = preprocess(roi, cfg.m2)
        enhanced = enhance_lines(pre, cfg.m3)
        mask = segment_and_clean(enhanced, cfg.m4)
        skel, length_px = skeletonize_and_measure(mask, cfg.m5)

        metrics = compute_metrics(
            length_px=length_px,
            roi_shape=roi.shape,
            nm_per_px=m1.nm_per_px,
            thickness_nm=cfg.m6.thickness_nm,
        )

        per_roi_dir = per_image_dir / f"roi_{roi_id:02d}"
        per_roi_dir.mkdir(parents=True, exist_ok=True)

        overlay_path = None
        qc_path = None
        if cfg.m7.overlay_enabled:
            overlay_path = per_roi_dir / "overlay.png"
            from PIL import Image
            
            ov = render_overlay_rgb(roi, skel)
            Image.fromarray(ov).save(overlay_path)
        if cfg.m7.qc_panel_enabled:
            qc_path = per_roi_dir / "qc.png"
            render_qc_panel(
                raw=raw,
                roi=roi,
                enhanced=enhanced,
                mask=mask,
                skeleton=skel,
                out_path=qc_path,
                dpi=cfg.m7.qc_dpi,
            )

        per_roi_rows.append(
            {
                "image": image_path.name,
                "roi_id": int(roi_id),
                "nm_per_px": float(m1.nm_per_px),
                "nm_per_px_source": m1.nm_per_px_source,
                "roi_x0": int(roi_box[0]),
                "roi_y0": int(roi_box[1]),
                "roi_x1": int(roi_box[2]),
                "roi_y1": int(roi_box[3]),
                "length_px": float(length_px),
                **{k: (None if v is None else float(v)) for k, v in metrics.items()},
                "overlay_path": str(overlay_path) if overlay_path else None,
                "qc_path": str(qc_path) if qc_path else None,
                "version": __version__,
            }
        )

        roi_payloads.append(
            {
                "roi_id": int(roi_id),
                "roi_box_xyxy": [int(v) for v in roi_box],
                "length_px": float(length_px),
                "metrics": {k: (None if v is None else float(v)) for k, v in metrics.items()},
                "artifacts": {"overlay": str(overlay_path) if overlay_path else None, "qc": str(qc_path) if qc_path else None},
            }
        )

    payload = {
        "image": str(image_path),
        "version": __version__,
        "nm_per_px": float(m1.nm_per_px),
        "nm_per_px_source": m1.nm_per_px_source,
        "config": asdict(cfg),
        "rois": roi_payloads,
    }
    (per_image_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"per_roi_rows": per_roi_rows}
