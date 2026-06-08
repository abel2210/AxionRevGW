from pathlib import Path
import math

import numpy as np
from scipy.special import eval_genlaguerre


OUT_DIR = Path("diagnostics")
STATES = [(5, 4, 4), (6, 4, 4)]
M1_MSUN = 1.0
Q = 0.01
ALPHA = 0.30
E_RES = 0.64
TRANSITION_HZ = 5.38123349


def radial_wavefunction_dimensionless(state, x):
    """Hydrogenic radial function in x = r / r_c with r_c = r_g / alpha^2."""
    n, ell, _m = state
    rho = 2.0 * x / n
    prefactor = math.sqrt(
        (2.0 / n) ** 3
        * math.factorial(n - ell - 1)
        / (2.0 * n * math.factorial(n + ell))
    )
    return prefactor * np.exp(-rho / 2.0) * rho**ell * eval_genlaguerre(n - ell - 1, 2 * ell + 1, rho)


def cumulative_probability(state, x_grid):
    radial = radial_wavefunction_dimensionless(state, x_grid)
    density = x_grid**2 * np.abs(radial) ** 2
    cumulative = np.zeros_like(x_grid)
    cumulative[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(x_grid))
    cumulative /= cumulative[-1]
    return density, cumulative


def interp_x_for_probability(x_grid, cumulative, probability):
    return float(np.interp(probability, cumulative, x_grid))


def interp_probability(x_grid, cumulative, x_value):
    return float(np.interp(x_value, x_grid, cumulative))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    x_grid = np.linspace(1.0e-6, 180.0, 400_000)

    omega_orb_hz = TRANSITION_HZ
    omega_orb_rad_s = 2.0 * math.pi * omega_orb_hz
    # In geometric units, Omega * GM/c^3 = 2 pi f t_sun for M1 = 1 Msun.
    t_sun = 4.925490947e-6
    omega_geom = omega_orb_rad_s * t_sun * M1_MSUN
    total_mass_factor = 1.0 + Q
    a_over_rg = (total_mass_factor / omega_geom**2) ** (1.0 / 3.0)
    a_over_rc = a_over_rg * ALPHA**2
    r_peri_over_rc = a_over_rc * (1.0 - E_RES)
    r_apo_over_rc = a_over_rc * (1.0 + E_RES)

    rows = []
    for state in STATES:
        density, cumulative = cumulative_probability(state, x_grid)
        x_peak = float(x_grid[np.argmax(density)])
        x50 = interp_x_for_probability(x_grid, cumulative, 0.50)
        x90 = interp_x_for_probability(x_grid, cumulative, 0.90)
        rows.append(
            {
                "state": f"|{state[0]}{state[1]}{state[2]}>",
                "x_peak": x_peak,
                "x50": x50,
                "x90": x90,
                "a_res_over_x_peak": a_over_rc / x_peak,
                "r_peri_over_x90": r_peri_over_rc / x90,
                "p_in_r_peri": interp_probability(x_grid, cumulative, r_peri_over_rc),
                "p_in_a_res": interp_probability(x_grid, cumulative, a_over_rc),
                "p_in_r_apo": interp_probability(x_grid, cumulative, r_apo_over_rc),
            }
        )

    csv_path = OUT_DIR / "compact_bohr_domain.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "state,x_peak,x50,x90,a_res_over_x_peak,r_peri_over_x90,"
            "p_in_r_peri,p_in_a_res,p_in_r_apo\n"
        )
        for row in rows:
            handle.write(
                "{state},{x_peak:.6g},{x50:.6g},{x90:.6g},{a_res_over_x_peak:.6g},"
                "{r_peri_over_x90:.6g},{p_in_r_peri:.6g},{p_in_a_res:.6g},{p_in_r_apo:.6g}\n".format(
                    **row
                )
            )

    md_path = OUT_DIR / "compact_bohr_domain.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Compact Bohr crossing domain check\n\n")
        handle.write(f"- M1 = {M1_MSUN:g} Msun, q = {Q:g}, alpha = {ALPHA:g}, e_res = {E_RES:g}\n")
        handle.write(f"- transition frequency = {TRANSITION_HZ:.8g} Hz\n")
        handle.write(f"- a_res/r_c = {a_over_rc:.6g}\n")
        handle.write(f"- r_peri/r_c = {r_peri_over_rc:.6g}\n")
        handle.write(f"- r_apo/r_c = {r_apo_over_rc:.6g}\n\n")
        handle.write("| state | x_peak | x50 | x90 | a_res/x_peak | r_peri/x90 | P_in(a_res) |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            md_state = row["state"].replace("|", r"\|")
            handle.write(
                f"| {md_state} | {row['x_peak']:.4g} | {row['x50']:.4g} | "
                f"{row['x90']:.4g} | {row['a_res_over_x_peak']:.4g} | "
                f"{row['r_peri_over_x90']:.4g} | {row['p_in_a_res']:.4g} |\n"
            )

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
