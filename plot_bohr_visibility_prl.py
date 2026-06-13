from __future__ import annotations

from pathlib import Path

import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from bohr_lz_tools import lz_probability_and_coherence, solve_standard_lz


BASE_DIR = Path(__file__).resolve().parent
DIAGNOSTICS_DIR = BASE_DIR / "diagnostics"
PUBLIC_FIGURES_DIR = BASE_DIR / "figures"

ALPHA_CSV = DIAGNOSTICS_DIR / "bohr_alpha_family.csv"
SWEEP_CSV = DIAGNOSTICS_DIR / "bohr_visibility_sweep.csv"
PUBLIC_FIGURE = PUBLIC_FIGURES_DIR / "bohr_visibility_prl_four_panel.pdf"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 7.1,
            "axes.labelsize": 7.3,
            "axes.titlesize": 7.4,
            "legend.fontsize": 6.3,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.linewidth": 0.72,
        }
    )


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.shape == ():
        data = data.reshape(1)
    return {name: np.asarray(data[name]) for name in data.dtype.names or ()}


def finite_alpha_rows(alpha_data: dict[str, np.ndarray]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for idx, flag in enumerate(alpha_data["full_time_domain"]):
        if int(flag) != 1:
            continue
        rows.append(
            {
                "alpha": float(alpha_data["alpha"][idx]),
                "frequency_ratio": float(alpha_data["transition_frequency_ratio"][idx]),
                "z_lz": float(alpha_data["z_lz"][idx]),
                "c_out_lz": float(alpha_data["c_out_lz"][idx]),
                "c_post": float(alpha_data["c_post_num"][idx]),
                "h_peak_post": float(alpha_data["h_peak_post"][idx]),
            }
        )
    rows.sort(key=lambda item: item["alpha"])
    return rows


def plot_probability_vs_coherence(ax: plt.Axes) -> None:
    z_grid = np.logspace(-4.2, 1.1, 700)
    p_tr, c_out = lz_probability_and_coherence(z_grid)
    ax.plot(z_grid, p_tr, color="#111827", lw=1.25, label=r"$P_{\rm tr}$")
    ax.plot(z_grid, c_out, color="#2563EB", lw=1.35, label=r"$C_{\rm out}$")
    ax.axvline(np.log(2.0) / (2.0 * np.pi), color="#2563EB", lw=0.72, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1.0e-4, 12.0)
    ax.set_ylim(-0.02, 1.04)
    ax.set_ylabel("LZ outcome")
    ax.set_title("transition probability is not visibility")
    ax.legend(frameon=False, loc="center left", handlelength=1.6)
    ax.tick_params(direction="in", top=True, right=True, length=3.0, width=0.65)
    ax.text(1.7e-4, 0.12, "weak\npassage", color="0.32", fontsize=6.2, ha="left")
    ax.text(2.4, 0.13, "adiabatic\ntransfer", color="0.32", fontsize=6.2, ha="left")


def plot_normalized_waveform(ax: plt.Axes, sweep_data: dict[str, np.ndarray]) -> None:
    z_grid = np.logspace(-4.2, 1.1, 700)
    _, c_out = lz_probability_and_coherence(z_grid)
    ax.plot(z_grid, c_out, color="#2563EB", lw=1.25, label=r"$C_{\rm out}$")

    z = np.asarray(sweep_data["z_lz"], dtype=float)
    h = np.asarray(sweep_data["h_peak_post_over_A0"], dtype=float)
    labels = np.asarray(sweep_data["representative_label"]).astype(str)
    ax.plot(z, h, marker="o", ms=3.0, mfc="white", mec="#111827", mew=0.65, ls="None", label=r"$h_{\rm pk,post}/\mathcal{A}_0$")
    for tag, color in (("slow", "#9A3412"), ("reference", "#047857"), ("very_fast", "#4B5563")):
        mask = labels == tag
        if not np.any(mask):
            continue
        ax.plot(z[mask], h[mask], marker="o", ms=4.6, mfc="white", mec=color, mew=1.15, ls="None")
        ax.axvline(float(z[mask][0]), color=color, lw=0.65, ls=":", alpha=0.65)

    ax.set_xscale("log")
    ax.set_xlim(1.0e-4, 12.0)
    ax.set_ylim(-0.01, 0.53)
    ax.set_ylabel("post-crossing signal")
    ax.set_title("fixed-normalization sweep")
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)
    ax.tick_params(direction="in", top=True, right=True, length=3.0, width=0.65)


def plot_alpha_family(ax: plt.Axes, alpha_data: dict[str, np.ndarray], rows: list[dict[str, float]]) -> None:
    alpha = np.asarray(alpha_data["alpha"], dtype=float)
    z_lz = np.asarray(alpha_data["z_lz"], dtype=float)
    c_out = np.asarray(alpha_data["c_out_lz"], dtype=float)

    ax.plot(alpha, z_lz, color="#111827", lw=1.12)
    ax.axhline(1.0, color="0.35", lw=0.68, ls="--")
    ax.set_yscale("log")
    ax.set_xlim(0.17, 0.325)
    ax.set_ylim(8.0e-2, 5.5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$z_{\rm LZ}$")
    ax.set_title("physical Bohr family")
    ax.tick_params(direction="in", top=True, right=False, length=3.0, width=0.65)

    ax2 = ax.twinx()
    ax2.plot(alpha, c_out, color="#2563EB", lw=1.05)
    ax2.set_ylim(-0.01, 0.43)
    ax2.set_ylabel(r"$C_{\rm out}$", color="#2563EB")
    ax2.tick_params(axis="y", labelcolor="#2563EB", direction="in", length=3.0, width=0.65)

    for row, color in zip(rows, ("#9A3412", "#047857")):
        ax.plot(row["alpha"], row["z_lz"], marker="o", ms=4.6, mfc="white", mec=color, mew=1.1, zorder=4)
        ax2.plot(row["alpha"], row["c_post"], marker="s", ms=4.0, mfc="white", mec=color, mew=1.0, zorder=4)
        ax.axvline(row["alpha"], color=color, lw=0.68, ls=":", alpha=0.7)


def plot_local_envelopes(ax: plt.Axes) -> None:
    cases = [
        ("slow", 3.744165383985, "#9A3412", r"slow, $z=3.74$"),
        ("finite", 0.3432074641569, "#047857", r"finite, $z=0.34$"),
        ("weak", 1.0e-4, "#4B5563", r"weak, $z=10^{-4}$"),
    ]
    for _, z_lz, color, label in cases:
        x, coherence, c_post = solve_standard_lz(z_lz, width_max=95.0, samples=3600)
        ax.plot(x, coherence, color=color, lw=0.86, alpha=0.48)
        ax.hlines(c_post, 22.0, 95.0, color=color, lw=1.18, label=label)
    ax.axvline(0.0, color="0.20", lw=0.65, alpha=0.7)
    ax.axvspan(22.0, 95.0, color="0.86", lw=0.0, alpha=0.34)
    ax.set_xlim(-45.0, 95.0)
    ax.set_ylim(-0.01, 0.53)
    ax.set_xlabel(r"local passage variable")
    ax.set_ylabel(r"$|c_i^\ast \tilde c_f|$")
    ax.set_title("post-crossing envelope")
    ax.legend(frameon=False, loc="upper right", handlelength=1.4)
    ax.tick_params(direction="in", top=True, right=True, length=3.0, width=0.65)


def main() -> None:
    configure_matplotlib()
    alpha_data = read_numeric_csv(ALPHA_CSV)
    sweep_data = read_numeric_csv(SWEEP_CSV)
    rows = finite_alpha_rows(alpha_data)

    cm = 1.0 / 2.54
    fig, axes = plt.subplots(2, 2, figsize=(17.4 * cm, 12.0 * cm))
    plot_probability_vs_coherence(axes[0, 0])
    plot_normalized_waveform(axes[0, 1], sweep_data)
    plot_alpha_family(axes[1, 0], alpha_data, rows)
    plot_local_envelopes(axes[1, 1])
    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.ravel()):
        ax.text(-0.14, 1.03, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8)
    for ax in axes[0, :]:
        ax.set_xlabel(r"$z_{\rm LZ}=|\eta|^2/|\dot\delta|$")
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.93, wspace=0.36, hspace=0.42)

    PUBLIC_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PUBLIC_FIGURE, dpi=300, bbox_inches="tight")
    print(f"Wrote {PUBLIC_FIGURE}")
    plt.close(fig)

    if len(rows) >= 2:
        ratio = rows[-1]["c_post"] / max(rows[0]["c_post"], 1.0e-300)
        print(f"Physical C_post ratio {rows[-1]['alpha']:.2f}/{rows[0]['alpha']:.2f}: {ratio:.6e}")


if __name__ == "__main__":
    main()
