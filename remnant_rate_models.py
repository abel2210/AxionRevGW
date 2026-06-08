from __future__ import annotations

import numpy as np


MPC = 3.085677581491367e22
GPC = 1.0e3 * MPC
YEAR = 365.25 * 24.0 * 3600.0


def kerr_horizon_frequency_dimensionless(spin: float) -> float:
    """Return M Omega_H for a Kerr black hole with dimensionless spin a_*."""
    spin = float(spin)
    if not 0.0 <= spin < 1.0:
        raise ValueError("spin must satisfy 0 <= a_* < 1.")
    return spin / (2.0 * (1.0 + np.sqrt(1.0 - spin**2)))


def superradiant_spin_threshold(alpha: float, azimuthal_m: int) -> float:
    """Return the minimum a_* satisfying m Omega_H > alpha."""
    if azimuthal_m <= 0:
        raise ValueError("azimuthal_m must be positive.")
    x = float(alpha) / float(azimuthal_m)
    if x <= 0.0:
        return 0.0
    if x >= 0.5:
        return np.inf
    return 4.0 * x / (1.0 + 4.0 * x**2)


def p_superradiant_remnant(
    alpha: float,
    azimuthal_m: int,
    remnant_spin: float = 0.7,
    model: str = "gerosa_berti_2017_step",
) -> float:
    """First-pass remnant-cloud occupation gate for the chosen spin benchmark."""
    del model
    threshold = superradiant_spin_threshold(alpha, azimuthal_m)
    return float(remnant_spin > threshold)


def rate_evolution_weight(z, evolution: str = "constant"):
    """Dimensionless redshift evolution normalized to unity at z=0."""
    z = np.asarray(z, dtype=float)
    key = str(evolution or "constant").strip().lower()
    if key in {"constant", "flat", "none"}:
        return np.ones_like(z, dtype=float)
    if key in {"sfr", "madau", "madau_dickinson", "madau-dickinson"}:
        numerator = (1.0 + z) ** 2.7 / (1.0 + ((1.0 + z) / 2.9) ** 5.6)
        normalization = 1.0 / (1.0 + (1.0 / 2.9) ** 5.6)
        return numerator / normalization
    raise ValueError(f"Unsupported SGWB rate evolution model: {evolution!r}")


def effective_local_rate_gpc3_yr(
    r0_gpc3_yr: float = 1.0,
    f_ret: float = 1.0,
    f_2g: float = 1.0,
    f_cloud: float = 1.0,
    f_duty: float = 1.0,
) -> float:
    return float(r0_gpc3_yr) * float(f_ret) * float(f_2g) * float(f_cloud) * float(f_duty)


def remnant_cloud_rate_density_si(
    z,
    r0_gpc3_yr: float = 1.0,
    evolution: str = "constant",
    f_ret: float = 1.0,
    f_2g: float = 1.0,
    f_cloud: float = 1.0,
    f_duty: float = 1.0,
):
    rate_gpc3_yr = effective_local_rate_gpc3_yr(
        r0_gpc3_yr=r0_gpc3_yr,
        f_ret=f_ret,
        f_2g=f_2g,
        f_cloud=f_cloud,
        f_duty=f_duty,
    )
    return rate_gpc3_yr * rate_evolution_weight(z, evolution=evolution) / (YEAR * GPC**3)
