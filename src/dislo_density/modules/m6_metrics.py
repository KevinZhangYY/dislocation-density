from __future__ import annotations

import numpy as np


def compute_metrics(
    length_px: float,
    roi_shape: tuple[int, int],
    nm_per_px: float,
    thickness_nm: float | None,
) -> dict[str, float | None]:
    h, w = roi_shape
    area_px2 = float(h) * float(w)
    
    # Convert nm inputs to SI (meters) for calculation
    # nm_per_px is in nm/px, so we multiply by 1e-9 to get m/px
    m_per_px = float(nm_per_px) * 1e-9
    
    length_m = float(length_px) * m_per_px
    area_m2 = area_px2 * m_per_px * m_per_px

    rho_a_m_inv = None
    if area_m2 > 0:
        rho_a_m_inv = length_m / area_m2

    rho_m2 = None
    if thickness_nm is not None and thickness_nm > 0 and area_m2 > 0:
        thickness_m = float(thickness_nm) * 1e-9
        volume_m3 = area_m2 * thickness_m
        rho_m2 = length_m / volume_m3

    return {
        "LLL_m": length_m,
        "AAA_m2": area_m2,
        "rhoA_m-1": rho_a_m_inv,
        "rho_m-2": rho_m2,
    }
