from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class M1Config:
    manual_nm_per_px: float | None = None
    manual_scale_bar_nm: float | None = None  # User can provide "500 nm" directly
    rois: list[dict[str, int]] | None = None
    auto_roi: bool = True
    roi_border_fraction: float = 0.02
    roi_border_px: int = 0
    roi_drop_bottom_fraction: float = 0.12
    roi_drop_bottom_px: int = 0
    prefer_metadata_nm_per_px: bool = True
    scale_bar_enabled: bool = True
    scale_bar_search_height_fraction: float = 0.22
    scale_bar_min_aspect: float = 6.0
    scale_bar_threshold_quantile: float = 0.995


@dataclass(frozen=True)
class M2Config:
    gaussian_sigma: float = 1.0
    background_suppression_enabled: bool = False
    background_method: str = "rolling_ball"
    rolling_ball_radius_px: int = 50
    tophat_radius_px: int = 25


@dataclass(frozen=True)
class M3Config:
    method: str = "frangi"
    black_ridges: bool = True
    sigmas_min: float = 1.5
    sigmas_max: float = 3.0
    sigmas_num: int = 4


@dataclass(frozen=True)
class M4Config:
    threshold_quantile: float = 0.92
    morph_open_radius_px: int = 1
    morph_close_radius_px: int = 2
    min_size_px: int = 64
    long_line_filter_enabled: bool = False
    long_line_max_length_px: int = 1200
    long_line_min_eccentricity: float = 0.995


@dataclass(frozen=True)
class M5Config:
    prune_branches_enabled: bool = True
    prune_length_px: int = 8


@dataclass(frozen=True)
class M6Config:
    thickness_nm: float | None = None


@dataclass(frozen=True)
class M7Config:
    overlay_enabled: bool = True
    qc_panel_enabled: bool = True
    qc_dpi: int = 150


@dataclass(frozen=True)
class M8Config:
    recursive: bool = True
    glob: str = "*.tif*"
    workers: int = 1
    log_level: str = "INFO"


@dataclass(frozen=True)
class AppConfig:
    m1: M1Config = dataclasses.field(default_factory=M1Config)
    m2: M2Config = dataclasses.field(default_factory=M2Config)
    m3: M3Config = dataclasses.field(default_factory=M3Config)
    m4: M4Config = dataclasses.field(default_factory=M4Config)
    m5: M5Config = dataclasses.field(default_factory=M5Config)
    m6: M6Config = dataclasses.field(default_factory=M6Config)
    m7: M7Config = dataclasses.field(default_factory=M7Config)
    m8: M8Config = dataclasses.field(default_factory=M8Config)


def _coerce_dataclass(dc_type: type[Any], raw: dict[str, Any] | None) -> Any:
    if raw is None:
        return dc_type()
    allowed = {f.name for f in dataclasses.fields(dc_type)}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    return dc_type(**filtered)


def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must be a YAML mapping")

    return AppConfig(
        m1=_coerce_dataclass(M1Config, raw.get("m1")),
        m2=_coerce_dataclass(M2Config, raw.get("m2")),
        m3=_coerce_dataclass(M3Config, raw.get("m3")),
        m4=_coerce_dataclass(M4Config, raw.get("m4")),
        m5=_coerce_dataclass(M5Config, raw.get("m5")),
        m6=_coerce_dataclass(M6Config, raw.get("m6")),
        m7=_coerce_dataclass(M7Config, raw.get("m7")),
        m8=_coerce_dataclass(M8Config, raw.get("m8")),
    )
