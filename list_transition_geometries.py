from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.special import eval_genlaguerre

from transition_geometry import compute_transition_geometry, state_label


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = BASE_DIR / "transition_geometry_factors.csv"
OVERLAP_MAX_X = 5.0e3
OVERLAP_GRID_POINTS = 4096

TRANSITIONS = (
    ("highfre322 upward", (3, 0, 0), (3, 2, 2)),
    ("highfre322 downward", (3, 2, 2), (3, 0, 0)),
    ("highfre211 upward", (2, 1, -1), (2, 1, 1)),
    ("highfre211 downward", (2, 1, 1), (2, 1, -1)),
    ("highfre644 upward", (5, 4, 4), (6, 4, 4)),
    ("highfre644 downward", (6, 4, 4), (5, 4, 4)),
)


def radial_wavefunction_dimensionless(state, x):
    n, l, _ = state
    rho = 2.0 * x / n
    normalization = (2.0 / n) ** 1.5 * math.sqrt(
        math.factorial(n - l - 1) / (2.0 * n * math.factorial(n + l))
    )
    laguerre = eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    return normalization * np.exp(-x / n) * rho**l * laguerre


def main():
    rows = []
    for name, initial_state, final_state in TRANSITIONS:
        geometry = compute_transition_geometry(
            initial_state,
            final_state,
            radial_wavefunction_dimensionless,
            overlap_max_x=OVERLAP_MAX_X,
            overlap_grid_points=OVERLAP_GRID_POINTS,
        )
        rows.append(
            {
                "name": name,
                "initial_state": state_label(initial_state),
                "final_state": state_label(final_state),
                "delta_m": geometry.delta_m,
                "pattern": geometry.pattern,
                "radial_overlap": geometry.radial_overlap,
                "waveform_geom_factor": geometry.waveform_geom_factor,
                "source_angle_average_factor": geometry.source_angle_average_factor,
                "snr_angle_prefactor": math.sqrt(geometry.source_angle_average_factor),
                "observer_projected_rss": geometry.observer_projected_rss,
            }
        )

    fieldnames = list(rows[0])
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "transition             states              dm  pattern                 "
        "F_geom        <P^2>     sqrt(<P^2>)"
    )
    for row in rows:
        states = f"{row['initial_state']}->{row['final_state']}"
        print(
            f"{row['name']:<22} {states:<18} {row['delta_m']:>3}  "
            f"{row['pattern']:<22} {row['waveform_geom_factor']:>11.6g}  "
            f"{row['source_angle_average_factor']:>8.6g}  "
            f"{row['snr_angle_prefactor']:>11.6g}"
        )
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
