from __future__ import annotations

from pathlib import Path

import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from adiabaticlimit import AdiabaticPhaseDiagram


BASE_DIR = Path(__file__).resolve().parent
FIGURE_PATH = BASE_DIR / "figures" / "bohr_quench_revival_evidence.pdf"

Q_REF = 1.0e-2
E_REF = 0.64
ALPHA_QUENCH = 0.18
ALPHA_REFERENCE = 0.30
M1_MSUN = 1.0


def lz_probability_and_coherence(z_lz: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z_lz, dtype=float)
    probability = 1.0 - np.exp(-2.0 * np.pi * z)
    coherence = np.sqrt(np.clip(probability * (1.0 - probability), 0.0, None))
    return probability, coherence


def solve_standard_lz(
    z_lz: float,
    width_max: float = 42.0,
    samples: int = 2600,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the canonical LZ equation and plot it in local coupling-width units."""
    coupling = np.sqrt(float(z_lz))
    tau_max = width_max * coupling

    def rhs(tau: float, y: np.ndarray) -> list[float]:
        cg = y[0] + 1j * y[1]
        ce = y[2] + 1j * y[3]
        d_cg = -1j * (0.5 * tau * cg + coupling * ce)
        d_ce = -1j * (coupling * cg - 0.5 * tau * ce)
        return [d_cg.real, d_cg.imag, d_ce.real, d_ce.imag]

    tau_values = np.linspace(-tau_max, tau_max, int(samples))
    solution = solve_ivp(
        rhs,
        (float(tau_values[0]), float(tau_values[-1])),
        [1.0, 0.0, 0.0, 0.0],
        t_eval=tau_values,
        method="DOP853",
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    cg = solution.y[0] + 1j * solution.y[1]
    ce = solution.y[2] + 1j * solution.y[3]
    coherence = np.abs(np.conj(cg) * ce)
    width_values = tau_values / max(coupling, 1.0e-300)
    return width_values, coherence, float(abs(np.conj(cg[-1]) * ce[-1]))


def transition_frequency_ratio(diagram: AdiabaticPhaseDiagram, alpha_grid: np.ndarray) -> np.ndarray:
    omega_ref = abs(
        diagram._omega_real_geom(diagram.initial_state, ALPHA_REFERENCE)
        - diagram._omega_real_geom(diagram.final_state, ALPHA_REFERENCE)
    )
    omega = np.array(
        [
            abs(
                diagram._omega_real_geom(diagram.initial_state, float(alpha))
                - diagram._omega_real_geom(diagram.final_state, float(alpha))
            )
            for alpha in alpha_grid
        ],
        dtype=float,
    )
    return omega / omega_ref


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 7.4,
            "axes.labelsize": 7.8,
            "axes.titlesize": 7.8,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.75,
        }
    )

    diagram = AdiabaticPhaseDiagram(
        initial_state=(6, 4, 4),
        final_state=(5, 4, 4),
        resonance_harmonic=1,
        eccentricity=E_REF,
    )

    alpha_grid = np.linspace(0.16, 0.40, 260)
    z_values = np.array(
        [
            diagram.compute_z_parameter(float(alpha), Q_REF, M_bh_solar=M1_MSUN, eccentricity=E_REF)
            for alpha in alpha_grid
        ],
        dtype=float,
    )
    _, coherence_values = lz_probability_and_coherence(z_values)
    frequency_ratio = transition_frequency_ratio(diagram, alpha_grid)

    marker_data = {}
    for alpha in (ALPHA_QUENCH, ALPHA_REFERENCE):
        z_value = float(diagram.compute_z_parameter(alpha, Q_REF, M_bh_solar=M1_MSUN, eccentricity=E_REF))
        p_lz, c_lz = lz_probability_and_coherence(z_value)
        marker_data[alpha] = {
            "z": z_value,
            "p": float(p_lz),
            "c": float(c_lz),
            "freq_ratio": float(transition_frequency_ratio(diagram, np.array([alpha]))[0]),
        }

    width_quench, coh_quench, c_num_quench = solve_standard_lz(marker_data[ALPHA_QUENCH]["z"])
    width_ref, coh_ref, c_num_ref = solve_standard_lz(marker_data[ALPHA_REFERENCE]["z"])

    cm = 1.0 / 2.54
    fig = plt.figure(figsize=(8.6 * cm, 8.7 * cm))
    grid = fig.add_gridspec(2, 1, height_ratios=(0.94, 1.06), hspace=0.30)
    ax0 = fig.add_subplot(grid[0])
    ax1 = fig.add_subplot(grid[1])

    ax0.plot(alpha_grid, z_values, color="#1F2937", lw=1.2)
    ax0.axhline(1.0, color="0.42", lw=0.72, ls="--")
    ax0.fill_between(alpha_grid, 1.0, np.maximum(z_values, 1.0), color="#B45309", alpha=0.10, linewidth=0.0)
    ax0.set_xlim(alpha_grid[0], alpha_grid[-1])
    ax0.set_yscale("log")
    ax0.set_ylim(5.0e-2, 1.5e1)
    ax0.set_xlabel(r"$\alpha$")
    ax0.set_ylabel(r"$z_{\rm LZ}$", color="#1F2937")
    ax0.tick_params(axis="y", labelcolor="#1F2937")
    ax0.tick_params(direction="in", top=True, right=True, length=3.0, width=0.65)
    ax0.text(
        0.18,
        1.35,
        "adiabatic depletion",
        color="#9A3412",
        fontsize=6.2,
        ha="left",
        va="bottom",
    )

    ax0b = ax0.twinx()
    ax0b.plot(alpha_grid, coherence_values, color="#2563EB", lw=1.35)
    ax0b.fill_between(alpha_grid, 0.0, coherence_values, color="#6BAED6", alpha=0.13, linewidth=0.0)
    ax0b.set_ylim(0.0, 0.53)
    ax0b.set_ylabel(r"$C_{\rm out}$", color="#2563EB")
    ax0b.tick_params(axis="y", labelcolor="#2563EB", direction="in", length=3.0, width=0.65, pad=1.0)

    top_ax = ax0.twiny()
    top_ax.set_xlim(ax0.get_xlim())
    tick_alphas = np.array([0.18, 0.24, 0.30, 0.36])
    tick_ratios = transition_frequency_ratio(diagram, tick_alphas)
    top_ax.set_xticks(tick_alphas)
    top_ax.set_xticklabels([f"{value:.2f}" for value in tick_ratios])
    top_ax.set_xlabel(r"$\omega_{\rm tr}/\omega_{\rm tr}(\alpha=0.30)$", labelpad=2.0)
    top_ax.tick_params(direction="in", length=3.0, width=0.65, pad=1.0)

    for alpha, color, label, y_text in (
        (ALPHA_QUENCH, "#9A3412", "slow passage", 5.6),
        (ALPHA_REFERENCE, "#047857", "rapid reference", 0.23),
    ):
        data = marker_data[alpha]
        ax0.plot(alpha, data["z"], marker="o", ms=4.4, mfc="white", mec=color, mew=1.2, zorder=4)
        ax0b.plot(alpha, data["c"], marker="s", ms=3.6, mfc="white", mec=color, mew=1.0, zorder=4)
        ax0.axvline(alpha, color=color, lw=0.78, ls="--", alpha=0.58)
        c_label = r"$C\simeq0$" if data["c"] < 1.0e-3 else rf"$C={data['c']:.2g}$"
        ax0.text(
            alpha + 0.006,
            y_text,
            rf"$\alpha={alpha:.2f}$" + "\n" + rf"$z={data['z']:.2g}$" + "\n" + c_label,
            color=color,
            ha="left",
            va="center",
            fontsize=6.4,
        )

    ax1.plot(width_quench, coh_quench, color="#9A3412", lw=1.1, label=rf"slow: $\alpha={ALPHA_QUENCH:.2f}$")
    ax1.plot(width_ref, coh_ref, color="#047857", lw=1.1, label=rf"rapid: $\alpha={ALPHA_REFERENCE:.2f}$")
    ax1.axvline(0.0, color="0.20", lw=0.65, alpha=0.65)
    ax1.axhline(marker_data[ALPHA_QUENCH]["c"], color="#9A3412", lw=0.72, ls=":", alpha=0.88)
    ax1.axhline(marker_data[ALPHA_REFERENCE]["c"], color="#047857", lw=0.72, ls=":", alpha=0.88)
    ax1.set_xlim(-42.0, 42.0)
    ax1.set_ylim(0.0, 0.53)
    ax1.set_xlabel(r"local passage time $(t-t_{\rm res})|\dot\delta|/|\eta|$")
    ax1.set_ylabel(r"$|c_i^\ast c_f|$")
    ax1.legend(loc="upper right", frameon=False, handlelength=1.5, borderpad=0.2)
    ax1.tick_params(direction="in", top=True, right=True, length=3.0, width=0.65)

    for panel, ax in (("(a)", ax0), ("(b)", ax1)):
        ax.text(-0.12, 1.02, panel, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.6)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {FIGURE_PATH}")
    for alpha, data in marker_data.items():
        print(
            f"alpha={alpha:.3f}: z={data['z']:.6g}, P_LZ={data['p']:.6g}, "
            f"C_out={data['c']:.6g}, f_ratio={data['freq_ratio']:.6g}"
        )


if __name__ == "__main__":
    main()
