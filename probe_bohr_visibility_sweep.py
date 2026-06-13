from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from adiabaticlimit import AdiabaticPhaseDiagram
from bohr_lz_tools import solve_standard_lz
from probe_bohr_alpha_family import (
    BH_SPIN,
    E_REF,
    Q_REF,
    TRANSITION_FINAL,
    TRANSITION_INITIAL,
    lz_probability_and_coherence,
    run_full_alpha,
    transition_frequency_ratio,
)


BASE_DIR = Path(__file__).resolve().parent
DIAGNOSTICS_DIR = BASE_DIR / "diagnostics"
FIGURES_DIR = BASE_DIR / "figures"

CSV_PATH = DIAGNOSTICS_DIR / "bohr_visibility_sweep.csv"
SUMMARY_PATH = DIAGNOSTICS_DIR / "bohr_visibility_sweep.md"

FINITE_Z_FIGURE = FIGURES_DIR / "bohr_finite_crossing_zlz.pdf"
FINITE_ENV_FIGURE = FIGURES_DIR / "bohr_finite_crossing_envelope.pdf"
SWEEP_Z_FIGURE = FIGURES_DIR / "bohr_sweep_rate_zlz.pdf"
SWEEP_ENV_FIGURE = FIGURES_DIR / "bohr_sweep_rate_envelope.pdf"
COMBINED_FIGURE = FIGURES_DIR / "bohr_visibility_two_group_four_panel.pdf"


Z_REF = 0.3432074641569
Z_SLOW = 3.744165383985
Z_FAST_WEAK = 1.0e-4
DISPLAY_WIDTH = 180.0


@dataclass(frozen=True)
class SweepPoint:
    z_lz: float
    lambda_sweep: float
    p_tr_lz: float
    c_out_lz: float
    p_g_post: float
    p_e_post: float
    h_peak_post_over_a0: float
    label: str


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.labelsize": 7.4,
            "axes.titlesize": 7.5,
            "legend.fontsize": 6.4,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "axes.linewidth": 0.72,
        }
    )


def sweep_label(z_lz: float) -> str:
    if np.isclose(z_lz, Z_SLOW, rtol=5.0e-3):
        return "slow"
    if np.isclose(z_lz, Z_REF, rtol=5.0e-3):
        return "reference"
    if np.isclose(z_lz, Z_FAST_WEAK, rtol=5.0e-3):
        return "very_fast"
    return ""


def build_sweep_points() -> list[SweepPoint]:
    z_values = np.array([10.0, 5.0, Z_SLOW, 2.0, 1.0, 0.5, Z_REF, 0.2, 0.1, 0.03, 0.01, 0.003, Z_FAST_WEAK])
    points: list[SweepPoint] = []
    for z_lz in z_values:
        p_tr, c_out = lz_probability_and_coherence(float(z_lz))
        p_tr = float(p_tr)
        c_out = float(c_out)
        points.append(
            SweepPoint(
                z_lz=float(z_lz),
                lambda_sweep=float(Z_REF / float(z_lz)),
                p_tr_lz=p_tr,
                c_out_lz=c_out,
                p_g_post=float(1.0 - p_tr),
                p_e_post=p_tr,
                h_peak_post_over_a0=c_out,
                label=sweep_label(float(z_lz)),
            )
        )
    points.sort(key=lambda item: item.z_lz, reverse=True)
    return points


def representative_sweep_envelopes() -> dict[str, dict[str, np.ndarray | float]]:
    cases = {
        "slow": {"z": Z_SLOW, "color": "#9A3412", "label": r"slow: $z_{\rm LZ}=3.74$"},
        "reference": {"z": Z_REF, "color": "#047857", "label": r"finite: $z_{\rm LZ}=0.34$"},
        "very fast": {"z": Z_FAST_WEAK, "color": "#4B5563", "label": r"too fast: $z_{\rm LZ}=10^{-4}$"},
    }
    out: dict[str, dict[str, np.ndarray | float]] = {}
    for key, meta in cases.items():
        width, coherence, c_window = solve_standard_lz(float(meta["z"]), width_max=DISPLAY_WIDTH, samples=4200)
        _, c_exact = lz_probability_and_coherence(float(meta["z"]))
        out[key] = {
            "x": width,
            "coherence": coherence,
            "c_window": float(c_window),
            "c_exact": float(c_exact),
            "z": float(meta["z"]),
            "color": str(meta["color"]),
            "label": str(meta["label"]),
        }
    return out


def alpha_family_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diagram = AdiabaticPhaseDiagram(
        initial_state=TRANSITION_INITIAL,
        final_state=TRANSITION_FINAL,
        resonance_harmonic=1,
        eccentricity=E_REF,
    )
    alpha_grid = np.linspace(0.16, 0.40, 320)
    z_grid = np.array(
        [
            diagram.compute_z_parameter(float(alpha), Q_REF, M_bh_solar=1.0, eccentricity=E_REF)
            for alpha in alpha_grid
        ],
        dtype=float,
    )
    _, c_grid = lz_probability_and_coherence(z_grid)
    freq_ratio = transition_frequency_ratio(diagram, alpha_grid)
    return alpha_grid, z_grid, c_grid, freq_ratio


def write_sweep_outputs(points: list[SweepPoint], finite_results) -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "z_lz",
                "lambda_sweep",
                "p_tr_lz",
                "c_out_lz",
                "p_g_post",
                "p_e_post",
                "c_post",
                "h_peak_post_over_A0",
                "representative_label",
            ],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "z_lz": f"{point.z_lz:.12e}",
                    "lambda_sweep": f"{point.lambda_sweep:.12e}",
                    "p_tr_lz": f"{point.p_tr_lz:.12e}",
                    "c_out_lz": f"{point.c_out_lz:.12e}",
                    "p_g_post": f"{point.p_g_post:.12e}",
                    "p_e_post": f"{point.p_e_post:.12e}",
                    "c_post": f"{point.c_out_lz:.12e}",
                    "h_peak_post_over_A0": f"{point.h_peak_post_over_a0:.12e}",
                    "representative_label": point.label,
                }
            )

    finite_by_alpha = {round(item.alpha, 12): item for item in finite_results}
    alpha018 = finite_by_alpha.get(round(0.18, 12))
    alpha030 = finite_by_alpha.get(round(0.30, 12))
    ref_point = min(points, key=lambda item: abs(item.z_lz - Z_REF))
    slow_point = min(points, key=lambda item: abs(item.z_lz - Z_SLOW))
    very_fast_point = min(points, key=lambda item: abs(item.z_lz - Z_FAST_WEAK))

    lines = [
        "# Bohr visibility sweep validation",
        "",
        "Generated by `probe_bohr_visibility_sweep.py`.",
        "",
        "The final evidence is organized as two groups of four panels:",
        "",
        r"- finite-crossing group: physical alpha-family \(z_{\rm LZ}\) curve and full time-domain envelope;",
        r"- visibility group: fixed-normalization sweep-rate curve and controlled local envelope.",
        "",
        "## Controlled sweep-rate result",
        "",
        "| case | z_LZ | lambda | P_e | P_g | C_post=h/A0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| slow | {slow_point.z_lz:.6g} | {slow_point.lambda_sweep:.3e} | {slow_point.p_e_post:.6g} | {slow_point.p_g_post:.6g} | {slow_point.c_out_lz:.6g} |",
        f"| reference | {ref_point.z_lz:.6g} | {ref_point.lambda_sweep:.3e} | {ref_point.p_e_post:.6g} | {ref_point.p_g_post:.6g} | {ref_point.c_out_lz:.6g} |",
        f"| very fast | {very_fast_point.z_lz:.6g} | {very_fast_point.lambda_sweep:.3e} | {very_fast_point.p_e_post:.6g} | {very_fast_point.p_g_post:.6g} | {very_fast_point.c_out_lz:.6g} |",
        "",
        r"Only the sweep rate changes in this controlled test.  The waveform normalization is fixed, so \(h_{\rm pk,post}/\mathcal{A}_0=C_{\rm post}\).",
        "",
        "## Physical finite-crossing bridge",
        "",
    ]
    if alpha018 is not None and alpha030 is not None:
        lines.extend(
            [
                "| alpha | z_LZ | C_post | h_pk,post |",
                "| ---: | ---: | ---: | ---: |",
                f"| 0.18 | {alpha018.z_lz:.6g} | {alpha018.c_post_num:.6g} | {alpha018.h_peak_post:.6e} |",
                f"| 0.30 | {alpha030.z_lz:.6g} | {alpha030.c_post_num:.6g} | {alpha030.h_peak_post:.6e} |",
                "",
                f"The physical alpha-family coherence ratio is {alpha030.c_post_num / max(alpha018.c_post_num, 1.0e-300):.3e}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            rf"- finite \(z_{{\rm LZ}}\): `{FINITE_Z_FIGURE.relative_to(BASE_DIR).as_posix()}`",
            f"- finite envelope: `{FINITE_ENV_FIGURE.relative_to(BASE_DIR).as_posix()}`",
            rf"- sweep \(z_{{\rm LZ}}\): `{SWEEP_Z_FIGURE.relative_to(BASE_DIR).as_posix()}`",
            f"- sweep envelope: `{SWEEP_ENV_FIGURE.relative_to(BASE_DIR).as_posix()}`",
            f"- combined four-panel figure: `{COMBINED_FIGURE.relative_to(BASE_DIR).as_posix()}`",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_finite_z(ax, alpha_grid, z_grid, c_grid, finite_results, show_title=True):
    ax.plot(alpha_grid, z_grid, color="#111827", lw=1.1)
    ax.axhline(1.0, color="0.35", lw=0.7, ls="--")
    ax.set_yscale("log")
    ax.set_xlim(alpha_grid[0], alpha_grid[-1])
    ax.set_ylim(5.0e-2, 1.6e1)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$z_{\rm LZ}$")
    if show_title:
        ax.set_title("finite-crossing family")
    ax2 = ax.twinx()
    ax2.plot(alpha_grid, c_grid, color="#2563EB", lw=1.0, alpha=0.75)
    ax2.set_ylim(0.0, 0.53)
    ax2.set_ylabel(r"$C_{\rm out}^{\rm LZ}$", color="#2563EB")
    ax2.tick_params(axis="y", labelcolor="#2563EB", direction="in", length=3.0, width=0.65)
    for item, color in zip(finite_results, ["#9A3412", "#047857"]):
        ax.plot(item.alpha, item.z_lz, marker="o", ms=4.2, mfc="white", mec=color, mew=1.1, zorder=4)
        ax2.plot(item.alpha, item.c_post_num, marker="s", ms=3.8, mfc="white", mec=color, mew=1.0, zorder=4)
        ax.axvline(item.alpha, color=color, lw=0.7, ls="--", alpha=0.55)
    ax.tick_params(direction="in", top=True, right=False, length=3.0, width=0.65)
    return ax2


def plot_finite_envelope(ax, finite_results, show_title=True):
    for item, color, label in zip(
        finite_results,
        ["#9A3412", "#047857"],
        [r"$\alpha=0.18$", r"$\alpha=0.30$"],
    ):
        ax.plot(item.local_x, item.coherence, color=color, lw=1.0, label=label)
        ax.axhline(item.c_post_num, color=color, lw=0.65, ls=":", alpha=0.9)
    ax.axvline(0.0, color="0.2", lw=0.65)
    ax.axvline(42.0, color="0.5", lw=0.6, ls="--")
    ax.set_xlim(-42.0, 50.0)
    ax.set_ylim(0.0, 0.53)
    ax.set_xlabel(r"local passage variable")
    ax.set_ylabel(r"$|c_i^\ast\tilde c_f|$")
    if show_title:
        ax.set_title("finite-crossing envelope")
    ax.legend(frameon=False, loc="upper right", handlelength=1.5)
    ax.tick_params(direction="in", top=True, right=True, length=3.0, width=0.65)


def plot_sweep_z(ax, points: list[SweepPoint], show_title=True):
    z_curve = np.logspace(-4.2, 1.05, 600)
    _, c_curve = lz_probability_and_coherence(z_curve)
    ax.plot(z_curve, c_curve, color="#2563EB", lw=1.25, label=r"$C_{\rm post}=C_{\rm out}^{\rm LZ}$")
    ax.plot(z_curve, c_curve, color="#111827", lw=0.75, ls="--", label=r"$h_{\rm pk,post}/\mathcal{A}_0$")
    colors = {"slow": "#9A3412", "reference": "#047857", "very_fast": "#4B5563", "": "0.55"}
    for point in points:
        if point.label:
            ax.plot(point.z_lz, point.c_out_lz, marker="o", ms=4.5, mfc="white", mec=colors[point.label], mew=1.1)
            ax.axvline(point.z_lz, color=colors[point.label], lw=0.65, ls=":", alpha=0.65)
    ax.set_xscale("log")
    ax.set_xlim(1.0e-4, 12.0)
    ax.set_ylim(0.0, 0.53)
    ax.set_xlabel(r"$z_{\rm LZ}=|\eta|^2/|\dot\delta|$")
    ax.set_ylabel(r"normalized post-crossing signal")
    if show_title:
        ax.set_title("same crossing: sweep-rate control")
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)
    ax.tick_params(direction="in", top=True, right=True, length=3.0, width=0.65)


def plot_sweep_envelope(ax, envelopes, show_title=True):
    for key in ("slow", "reference", "very fast"):
        item = envelopes[key]
        x = np.asarray(item["x"], dtype=float)
        coherence = np.asarray(item["coherence"], dtype=float)
        color = str(item["color"])
        label = str(item["label"])
        ax.plot(x, coherence, color=color, lw=0.9, alpha=0.55)
        ax.hlines(float(item["c_exact"]), 25.0, DISPLAY_WIDTH, color=color, lw=1.15, label=label)
    ax.axvline(0.0, color="0.2", lw=0.65)
    ax.axvspan(25.0, DISPLAY_WIDTH, color="0.86", alpha=0.35, lw=0.0)
    ax.set_xlim(-55.0, DISPLAY_WIDTH)
    ax.set_ylim(0.0, 0.53)
    ax.set_xlabel(r"local passage variable")
    ax.set_ylabel(r"$|c_i^\ast\tilde c_f|$")
    if show_title:
        ax.set_title("controlled-sweep envelope")
    ax.legend(frameon=False, loc="upper right", handlelength=1.35, borderaxespad=0.25)
    ax.tick_params(direction="in", top=True, right=True, length=3.0, width=0.65)


def save_single_panel(path: Path, plotter, *args):
    cm = 1.0 / 2.54
    fig, ax = plt.subplots(figsize=(8.5 * cm, 6.2 * cm))
    plotter(ax, *args)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_combined(alpha_grid, z_grid, c_grid, finite_results, points, envelopes):
    cm = 1.0 / 2.54
    fig, axes = plt.subplots(2, 2, figsize=(17.4 * cm, 12.2 * cm))
    plot_finite_z(axes[0, 0], alpha_grid, z_grid, c_grid, finite_results, show_title=True)
    plot_finite_envelope(axes[1, 0], finite_results, show_title=True)
    plot_sweep_z(axes[0, 1], points, show_title=True)
    plot_sweep_envelope(axes[1, 1], envelopes, show_title=True)
    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.ravel()):
        ax.text(-0.13, 1.03, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8)
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.08, top=0.94, wspace=0.34, hspace=0.46)
    COMBINED_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMBINED_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    finite_results = [run_full_alpha(0.18), run_full_alpha(0.30)]
    finite_results.sort(key=lambda item: item.alpha)
    alpha_grid, z_grid, c_grid, _ = alpha_family_curves()
    points = build_sweep_points()
    envelopes = representative_sweep_envelopes()

    write_sweep_outputs(points, finite_results)

    save_single_panel(FINITE_Z_FIGURE, plot_finite_z, alpha_grid, z_grid, c_grid, finite_results)
    save_single_panel(FINITE_ENV_FIGURE, plot_finite_envelope, finite_results)
    save_single_panel(SWEEP_Z_FIGURE, plot_sweep_z, points)
    save_single_panel(SWEEP_ENV_FIGURE, plot_sweep_envelope, envelopes)
    save_combined(alpha_grid, z_grid, c_grid, finite_results, points, envelopes)

    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    for path in (FINITE_Z_FIGURE, FINITE_ENV_FIGURE, SWEEP_Z_FIGURE, SWEEP_ENV_FIGURE, COMBINED_FIGURE):
        print(f"Wrote {path}")
    for point in points:
        if point.label:
            print(
                f"{point.label}: z={point.z_lz:.6g}, lambda={point.lambda_sweep:.3e}, "
                f"P_e={point.p_e_post:.6g}, C_post={point.c_out_lz:.6g}"
            )


if __name__ == "__main__":
    main()
