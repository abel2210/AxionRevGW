import csv
import math
from pathlib import Path

import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.special import eval_genlaguerre


FIGURE_DIR = Path("figures")
DIAG_DIR = Path("diagnostics")

ALPHA_REF = 0.30
Q_REF = 1.0e-3
E_RES = 0.6373
SPIN = 0.70
Z_LZ_REF = 0.5740777240495206
C_OUT_REF = math.sqrt((1.0 - math.exp(-2.0 * math.pi * Z_LZ_REF)) * math.exp(-2.0 * math.pi * Z_LZ_REF))

STATE_LOWER = (5, 4, 4)
STATE_UPPER = (6, 4, 4)
RESONANCE_HARMONIC = 1

REFERENCE_CLOUD_FRACTION = 1.0e-4
REFERENCE_PRIMARY_MASS_MSUN = 1.0e-2
REFERENCE_PEAK_STRAIN = 9.531702219146433e-27
TARGET_PEAK_STRAIN = 1.0e-23
REFERENCE_DISTANCE_KPC = 1.0


def omega_real_geom(state, alpha, spin=SPIN):
    n, ell, m = state
    term1 = 1.0
    term2 = -alpha**2 / (2.0 * n**2)
    term3 = -alpha**4 / (8.0 * n**4)
    term4 = ((4.0 * ell - 6.0 * n + 2.0) / (2.0 * (ell + 0.5) * n**4)) * alpha**4
    term5 = 0.0
    if ell > 0:
        term5 = (
            2.0
            * m
            * spin
            * alpha**5
            / (n**3 * ell * (ell + 0.5) * (ell + 1.0))
        )
    return alpha * (term1 + term2 + term3 + term4 + term5)


def transition_delta_omega_geom(alpha):
    return abs(omega_real_geom(STATE_UPPER, alpha) - omega_real_geom(STATE_LOWER, alpha))


def peters_eccentricity_factor(eccentricity):
    e2 = eccentricity * eccentricity
    return (1.0 + 73.0 * e2 / 24.0 + 37.0 * e2 * e2 / 96.0) / (1.0 - e2) ** 3.5


def c_out_from_z(z_lz):
    p_tr = 1.0 - np.exp(-2.0 * np.pi * z_lz)
    return np.sqrt(np.clip(p_tr * (1.0 - p_tr), 0.0, None))


def superradiance_allowed(alpha, spin=SPIN, state=STATE_UPPER):
    m = abs(state[2])
    r_plus = 1.0 + math.sqrt(max(1.0 - spin * spin, 0.0))
    omega_h = spin / (2.0 * r_plus)
    return omega_real_geom(state, alpha) < m * omega_h


def radial_wavefunction_dimensionless(state, x):
    n, ell, _m = state
    rho = 2.0 * x / n
    prefactor = math.sqrt(
        (2.0 / n) ** 3
        * math.factorial(n - ell - 1)
        / (2.0 * n * math.factorial(n + ell))
    )
    return prefactor * np.exp(-rho / 2.0) * rho**ell * eval_genlaguerre(n - ell - 1, 2 * ell + 1, rho)


def radial_cdf(state, x_grid):
    radial = radial_wavefunction_dimensionless(state, x_grid)
    density = x_grid**2 * np.abs(radial) ** 2
    cumulative = np.zeros_like(x_grid)
    cumulative[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(x_grid))
    cumulative /= cumulative[-1]
    return density, cumulative


def compact_support_numbers():
    x_grid = np.linspace(1.0e-6, 180.0, 300_000)
    density_low, cdf_low = radial_cdf(STATE_LOWER, x_grid)
    density_up, cdf_up = radial_cdf(STATE_UPPER, x_grid)
    return {
        "x_peak_lower": float(x_grid[np.argmax(density_low)]),
        "x90_lower": float(np.interp(0.90, cdf_low, x_grid)),
        "x_peak_upper": float(x_grid[np.argmax(density_up)]),
        "x90_upper": float(np.interp(0.90, cdf_up, x_grid)),
        "x_grid": x_grid,
        "cdf_lower": cdf_low,
        "cdf_upper": cdf_up,
    }


def resonant_orbit_over_rc(alpha_grid, q_grid):
    delta = transition_delta_omega_geom(alpha_grid)
    omega_res = delta / RESONANCE_HARMONIC
    a_over_rg = ((1.0 + q_grid) / np.maximum(omega_res, 1.0e-300) ** 2) ** (1.0 / 3.0)
    return a_over_rg * alpha_grid**2


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    alpha_values = np.linspace(0.10, 0.35, 220)
    q_values = np.geomspace(1.0e-4, 1.0e-2, 220)
    alpha_grid, q_grid = np.meshgrid(alpha_values, q_values)

    z_lz = (
        Z_LZ_REF
        * (q_grid / Q_REF)
        * (alpha_grid / ALPHA_REF) ** (-5.0)
        * peters_eccentricity_factor(E_RES)
        / peters_eccentricity_factor(E_RES)
    )
    c_out = c_out_from_z(z_lz)
    sr_allowed = np.vectorize(superradiance_allowed)(alpha_grid)

    support = compact_support_numbers()
    a_over_rc = resonant_orbit_over_rc(alpha_grid, q_grid)
    r_peri_over_rc = a_over_rc * (1.0 - E_RES)
    r_peri_over_x90_upper = r_peri_over_rc / support["x90_upper"]

    peak_strain = (
        REFERENCE_PEAK_STRAIN
        * (c_out / max(C_OUT_REF, 1.0e-300))
        * (alpha_grid / ALPHA_REF) ** 2
    )
    required_cloud_fraction = (
        REFERENCE_CLOUD_FRACTION
        * TARGET_PEAK_STRAIN
        / np.maximum(peak_strain, 1.0e-300)
    )
    required_cloud_fraction_raw = required_cloud_fraction.copy()
    required_cloud_fraction = np.clip(required_cloud_fraction, 1.0e-4, 1.0e-1)

    csv_path = DIAG_DIR / "bohr_domain_visibility_map.csv"
    sample_alphas = [0.18, 0.24, 0.30, 0.34]
    sample_qs = [1.0e-4, 1.0e-3, 1.0e-2]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "alpha",
                "q",
                "z_lz_scaling",
                "c_out",
                "superradiant_allowed",
                "a_res_over_rc",
                "r_peri_over_x90_upper",
                "required_cloud_fraction_for_hpeak_1e-23_at_1kpc",
            ]
        )
        for alpha in sample_alphas:
            for q in sample_qs:
                z_val = Z_LZ_REF * (q / Q_REF) * (alpha / ALPHA_REF) ** (-5.0)
                c_val = float(c_out_from_z(np.asarray(z_val)))
                a_val = float(resonant_orbit_over_rc(np.asarray(alpha), np.asarray(q)))
                peak_val = REFERENCE_PEAK_STRAIN * (c_val / C_OUT_REF) * (alpha / ALPHA_REF) ** 2
                req_val = REFERENCE_CLOUD_FRACTION * TARGET_PEAK_STRAIN / max(peak_val, 1.0e-300)
                writer.writerow(
                    [
                        f"{alpha:.6g}",
                        f"{q:.6g}",
                        f"{z_val:.6g}",
                        f"{c_val:.6g}",
                        int(superradiance_allowed(alpha)),
                        f"{a_val:.6g}",
                        f"{a_val * (1.0 - E_RES) / support['x90_upper']:.6g}",
                        f"{req_val:.6g}",
                    ]
                )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.65,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.05), constrained_layout=True)
    ax0, ax1 = axes

    x_edges = alpha_values
    y_edges = q_values
    c0 = ax0.pcolormesh(
        x_edges,
        y_edges,
        c_out,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=0.5,
        rasterized=True,
    )
    ax0.contour(alpha_grid, q_grid, c_out, levels=[0.1, 0.3], colors=["white", "white"], linewidths=[0.75, 1.05])
    ax0.contour(alpha_grid, q_grid, z_lz, levels=[1.0], colors=["#D95F02"], linewidths=0.95, linestyles="--")
    compact_boundary_present = (
        np.nanmin(r_peri_over_x90_upper) <= 1.0
        <= np.nanmax(r_peri_over_x90_upper)
    )
    if compact_boundary_present:
        ax0.contour(alpha_grid, q_grid, r_peri_over_x90_upper, levels=[1.0], colors=["#2B2B2B"], linewidths=0.8, linestyles=":")
    ax0.contourf(alpha_grid, q_grid, sr_allowed.astype(float), levels=[-0.1, 0.5], colors=["#D0D0D0"], alpha=0.55)
    ax0.plot(ALPHA_REF, Q_REF, marker="*", ms=7.5, color="#E31A1C", mec="white", mew=0.45, zorder=5)
    ax0.set_yscale("log")
    ax0.set_xlabel(r"$\alpha$")
    ax0.set_ylabel(r"$q=M_2/M_1$")
    ax0.set_title(r"outgoing coherence")
    cb0 = fig.colorbar(c0, ax=ax0, pad=0.01)
    cb0.set_label(r"$C_{\rm out}$")
    ax0.text(0.108, 1.35e-4, r"$z_{\rm LZ}=1$", color="#FFB000", fontsize=7.2)
    compact_label = (
        r"$r_p=x_{90}^{|644\rangle}$"
        if compact_boundary_present
        else r"$r_p<x_{90}^{|644\rangle}$ throughout"
    )
    ax0.text(0.108, 2.45e-4, compact_label, color="white", fontsize=7.2)
    if bool(np.all(sr_allowed)):
        ax0.text(0.337, 1.35e-4, r"SR allowed", color="white", fontsize=7.0, ha="right")

    c1 = ax1.pcolormesh(
        x_edges,
        y_edges,
        required_cloud_fraction,
        shading="auto",
        cmap="magma_r",
        norm=LogNorm(vmin=1.0e-4, vmax=1.0e-1),
        rasterized=True,
    )
    ax1.contour(alpha_grid, q_grid, c_out, levels=[0.1, 0.3], colors=["#355C7D", "#355C7D"], linewidths=[0.75, 1.05])
    ax1.contourf(alpha_grid, q_grid, sr_allowed.astype(float), levels=[-0.1, 0.5], colors=["#D0D0D0"], alpha=0.50)
    ax1.plot(ALPHA_REF, Q_REF, marker="*", ms=7.5, color="#E31A1C", mec="white", mew=0.45, zorder=5)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"$q=M_2/M_1$")
    ax1.set_title(r"cloud normalization for reference strain")
    cb1 = fig.colorbar(c1, ax=ax1, pad=0.01)
    cb1.set_label(r"required $M_c/M_1$")
    ax1.text(
        0.106,
        1.35e-4,
        rf"$h_{{\rm pk}}={TARGET_PEAK_STRAIN:.0e}$ at {REFERENCE_DISTANCE_KPC:g} kpc",
        color="white",
        fontsize=7,
    )
    ax1.text(0.337, 7.2e-2, r"$>10^{-1}$ saturated", color="white", fontsize=7.0, ha="right")

    for ax in axes:
        ax.set_xlim(alpha_values.min(), alpha_values.max())
        ax.set_ylim(q_values.min(), q_values.max())
        ax.tick_params(which="both", top=True, right=True)
        ax.grid(False)

    fig_path = FIGURE_DIR / "bohr_domain_visibility_map.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    png_path = FIGURE_DIR / "bohr_domain_visibility_map.png"
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    plt.close(fig)

    fig_single, axes_single = plt.subplots(2, 1, figsize=(3.38, 5.55), constrained_layout=True)
    ax0, ax1 = axes_single

    c0 = ax0.pcolormesh(
        x_edges,
        y_edges,
        c_out,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=0.5,
        rasterized=True,
    )
    ax0.contour(alpha_grid, q_grid, c_out, levels=[0.1, 0.3], colors=["white", "white"], linewidths=[0.75, 1.05])
    ax0.contour(alpha_grid, q_grid, z_lz, levels=[1.0], colors=["#D95F02"], linewidths=0.95, linestyles="--")
    if compact_boundary_present:
        ax0.contour(alpha_grid, q_grid, r_peri_over_x90_upper, levels=[1.0], colors=["#2B2B2B"], linewidths=0.8, linestyles=":")
    ax0.contourf(alpha_grid, q_grid, sr_allowed.astype(float), levels=[-0.1, 0.5], colors=["#D0D0D0"], alpha=0.55)
    ax0.plot(ALPHA_REF, Q_REF, marker="*", ms=7.0, color="#E31A1C", mec="white", mew=0.45, zorder=5)
    ax0.set_yscale("log")
    ax0.set_xlabel(r"$\alpha$")
    ax0.set_ylabel(r"$q=M_2/M_1$")
    ax0.set_title(r"(a) outgoing coherence")
    cb0 = fig_single.colorbar(c0, ax=ax0, pad=0.01)
    cb0.set_label(r"$C_{\rm out}$")
    ax0.text(0.108, 1.35e-4, r"$z_{\rm LZ}=1$", color="#FFB000", fontsize=7.0)
    ax0.text(0.108, 2.45e-4, compact_label, color="white", fontsize=7.0)
    if bool(np.all(sr_allowed)):
        ax0.text(0.337, 1.35e-4, r"SR allowed", color="white", fontsize=6.8, ha="right")

    c1 = ax1.pcolormesh(
        x_edges,
        y_edges,
        required_cloud_fraction,
        shading="auto",
        cmap="magma_r",
        norm=LogNorm(vmin=1.0e-4, vmax=1.0e-1),
        rasterized=True,
    )
    ax1.contour(alpha_grid, q_grid, c_out, levels=[0.1, 0.3], colors=["#355C7D", "#355C7D"], linewidths=[0.75, 1.05])
    ax1.contourf(alpha_grid, q_grid, sr_allowed.astype(float), levels=[-0.1, 0.5], colors=["#D0D0D0"], alpha=0.50)
    ax1.plot(ALPHA_REF, Q_REF, marker="*", ms=7.0, color="#E31A1C", mec="white", mew=0.45, zorder=5)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"$q=M_2/M_1$")
    ax1.set_title(r"(b) cloud normalization")
    cb1 = fig_single.colorbar(c1, ax=ax1, pad=0.01)
    cb1.set_label(r"required $M_c/M_1$")
    ax1.text(
        0.106,
        1.35e-4,
        rf"$h_{{\rm pk}}={TARGET_PEAK_STRAIN:.0e}$ at {REFERENCE_DISTANCE_KPC:g} kpc",
        color="white",
        fontsize=6.8,
    )
    ax1.text(0.337, 7.2e-2, r"$>10^{-1}$ saturated", color="white", fontsize=6.8, ha="right")

    for ax in axes_single:
        ax.set_xlim(alpha_values.min(), alpha_values.max())
        ax.set_ylim(q_values.min(), q_values.max())
        ax.tick_params(which="both", top=True, right=True)
        ax.grid(False)

    single_path = FIGURE_DIR / "bohr_domain_visibility_map_single.pdf"
    fig_single.savefig(single_path, bbox_inches="tight")
    single_png_path = FIGURE_DIR / "bohr_domain_visibility_map_single.png"
    fig_single.savefig(single_png_path, dpi=260, bbox_inches="tight")
    plt.close(fig_single)
    print(f"wrote {fig_path}")
    print(f"wrote {png_path}")
    print(f"wrote {single_path}")
    print(f"wrote {single_png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
