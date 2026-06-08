from __future__ import annotations

import csv
from pathlib import Path

import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.ticker import MaxNLocator


BASE_DIR = Path(__file__).resolve().parent
SNR_DATA_DIR = BASE_DIR / "snr_scan_data"
PROBE_DIR = BASE_DIR / "diagnostics" / "lowfre_mismatch_threshold_probe"
PROBE_REPORT_CSV = PROBE_DIR / "lowfre_mismatch_threshold_probe.csv"
PROBE_MISMATCH_DATA_DIR = PROBE_DIR / "mismatch_data"
FIGURE_DIR = BASE_DIR / "figures"

SNR_COLORBAR_MAX = 200.0
SNR_OVER_COLOR = "#1f2a7a"
REFERENCE_DISTANCE_KPC = 100.0

SNR_CASES = (
    ("211", r"$|21{-}1\rangle\!\to\!|211\rangle$"),
    ("211v", r"$|211\rangle\!\to\!|21{-}1\rangle$"),
    ("322", r"$|300\rangle\!\to\!|322\rangle$"),
)

MISMATCH_CASES = (
    (
        "211",
        "lowfre211_snrprobe_decigo_mismatch_upward.txt",
        r"$211\uparrow$",
    ),
    (
        "211v",
        "lowfre211v_snrprobe_decigo_mismatch_downward.txt",
        r"$211\downarrow$",
    ),
    (
        "322",
        "lowfre322_snrprobe_decigo_mismatch_upward.txt",
        r"$322\uparrow$",
    ),
    (
        "322v",
        "lowfre322v_snrprobe_decigo_mismatch_downward.txt",
        r"$322\downarrow$",
    ),
)

MISMATCH_STYLES = (
    {"color": "#0072B2", "linestyle": "-", "linewidth": 1.35},
    {"color": "#D55E00", "linestyle": "--", "linewidth": 1.35},
    {"color": "#009E73", "linestyle": "-.", "linewidth": 1.35},
    {"color": "#CC79A7", "linestyle": ":", "linewidth": 1.7},
)


def load_snr_grid(case_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = SNR_DATA_DIR / f"{case_key}_decigo_axion_snr_alpha_dl_grid.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing SNR grid: {path}")
    data = np.load(path)
    alpha = np.asarray(data["alpha"], dtype=float)
    distance_kpc = np.asarray(data["distance_kpc"], dtype=float)
    snr = np.asarray(data["snr"], dtype=float)
    return alpha, distance_kpc, snr


def load_probe_points() -> dict[str, dict[str, float]]:
    if not PROBE_REPORT_CSV.exists():
        return {}
    points = {}
    with PROBE_REPORT_CSV.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                points[row["preset"]] = {
                    "alpha": float(row["alpha"]),
                    "distance_kpc": float(row["distance_kpc"]),
                    "scan_snr": float(row["scan_snr"]),
                    "line_mhz": float(row["transition_freq_obs_mhz"]),
                    "mismatch": float(row["final_mismatch"]),
                    "threshold": float(row["final_threshold_N_over_2rho2"]),
                }
            except (KeyError, ValueError):
                continue
    return points


def load_mismatch_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 5:
        raise ValueError(f"{path} has {data.shape[1]} columns; expected at least 5.")

    observation_years = np.asarray(data[:, 0], dtype=float)
    effective_years = np.asarray(data[:, 1], dtype=float)
    mismatch = np.asarray(data[:, 2], dtype=float)
    threshold = np.asarray(data[:, 4], dtype=float)
    x_values = np.where(np.isfinite(effective_years), effective_years, observation_years)
    finite = np.isfinite(x_values) & np.isfinite(mismatch) & np.isfinite(threshold)
    if not np.any(finite):
        raise ValueError(f"{path} does not contain finite mismatch samples.")

    x_values = x_values[finite]
    mismatch = mismatch[finite]
    threshold = threshold[finite]
    order = np.argsort(x_values)
    x_values = x_values[order]
    mismatch = mismatch[order]
    threshold = threshold[order]
    if x_values[0] > 0.0:
        x_values = np.concatenate(([0.0], x_values))
        mismatch = np.concatenate(([0.0], mismatch))
        threshold = np.concatenate(([np.nan], threshold))
    return x_values, mismatch, threshold


def plot_snr_panel(
    ax,
    case_key: str,
    title: str,
    cmap,
    color_norm,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    probe_points: dict[str, dict[str, float]] | None = None,
) -> object:
    alpha, distance_kpc, snr = load_snr_grid(case_key)
    grid = snr.T
    masked_grid = np.ma.masked_invalid(grid)
    levels = np.linspace(0.0, SNR_COLORBAR_MAX, 81)
    image = ax.contourf(
        alpha,
        distance_kpc,
        masked_grid,
        levels=levels,
        cmap=cmap,
        norm=color_norm,
        extend="max",
    )

    finite_grid = grid[np.isfinite(grid)]
    if finite_grid.size:
        grid_min = float(np.min(finite_grid))
        grid_max = float(np.max(finite_grid))
        contour_levels = [level for level in (10.0, 30.0, 50.0, 100.0, 200.0) if grid_min < level < grid_max]
    else:
        contour_levels = []
    if contour_levels:
        low_contours = [level for level in contour_levels if level < 100.0]
        high_contours = [level for level in contour_levels if level >= 100.0]
        if low_contours:
            contours = ax.contour(alpha, distance_kpc, grid, levels=low_contours, colors="#111111", linewidths=0.55)
            ax.clabel(contours, fmt="%g", fontsize=4.5, colors="#111111")
        if high_contours:
            contours = ax.contour(alpha, distance_kpc, grid, levels=high_contours, colors="#B00020", linewidths=0.65)
            ax.clabel(contours, fmt="%g", fontsize=4.5, colors="#B00020")

    ax.axhline(REFERENCE_DISTANCE_KPC, color="#62d26f", lw=0.65, alpha=0.9)
    if probe_points and case_key in probe_points:
        point = probe_points[case_key]
        ax.scatter(
            [point["alpha"]],
            [point["distance_kpc"]],
            marker="*",
            s=22,
            facecolor="#FFD54F",
            edgecolor="#101010",
            linewidth=0.35,
            zorder=7,
        )
    ax.set_yscale("log")
    if show_xlabel:
        ax.set_xlabel(r"$\alpha$", labelpad=0.8)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(r"$d_L$ [kpc]", labelpad=0.8)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_title(title, pad=1.0)
    ax.set_xlim(float(np.min(alpha)), float(np.max(alpha)))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.grid(True, which="both", alpha=0.13, linewidth=0.35)
    return image


def plot_mismatch_panel(ax, show_ylabel: bool = False) -> None:
    max_x = 0.0
    max_y = 0.0
    for (_, filename, label), style in zip(MISMATCH_CASES, MISMATCH_STYLES):
        path = PROBE_MISMATCH_DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing SNR-selected mismatch curve: {path}")
        x_values, mismatch, threshold = load_mismatch_curve(path)
        ax.plot(x_values, mismatch, **style)
        threshold_plot = np.where((threshold > 0.0) & (threshold <= 1.05), threshold, np.nan)
        ax.plot(
            x_values,
            threshold_plot,
            color=style["color"],
            linestyle=(0, (2.2, 1.5)),
            linewidth=0.65,
            alpha=0.7,
        )
        ax.text(
            x_values[-1] - 0.006 * max(float(np.nanmax(x_values)), 1.0),
            min(max(float(mismatch[-1]), 0.035), 0.99),
            label,
            color=style["color"],
            fontsize=4.3,
            ha="right",
            va="center",
        )
        max_x = max(max_x, float(np.nanmax(x_values)))
        max_y = max(max_y, float(np.nanmax(mismatch)), float(np.nanmax(threshold_plot)))

    ax.set_xlabel(r"$T_{\rm obs}$ [yr]", labelpad=0.8)
    if show_ylabel:
        ax.set_ylabel("Mismatch", labelpad=0.8)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlim(0.0, max_x if max_x > 0.0 else 1.0)
    ax.set_ylim(0.0, min(1.05, 1.10 * max(max_y, 1.0e-12)))
    ax.set_title("SNR-selected mismatch", pad=1.0)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=3.2)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=1.8)
    ax.minorticks_on()
    ax.grid(False)
    ax.text(
        0.05,
        0.08,
        r"dashed: $N/(2\rho^2)$",
        transform=ax.transAxes,
        fontsize=4.3,
        ha="left",
        va="bottom",
    )


def main() -> None:
    probe_points = load_probe_points()
    cmap = LinearSegmentedColormap.from_list(
        "cool_decigo_snr",
        ["#f7fbff", "#dbeaf4", "#bdd8e9", "#8fbfd8", "#5a9ec6", "#2f79b7", "#2350a1"],
        N=256,
    )
    cmap.set_over(SNR_OVER_COLOR)
    color_norm = PowerNorm(gamma=0.58, vmin=0.0, vmax=SNR_COLORBAR_MAX)

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 5.8,
            "axes.labelsize": 5.7,
            "axes.titlesize": 5.9,
            "xtick.labelsize": 5.1,
            "ytick.labelsize": 5.1,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
        }
    ):
        cm_to_inch = 1.0 / 2.54
        fig, axes = plt.subplots(2, 2, figsize=(8.6 * cm_to_inch, 7.2 * cm_to_inch), constrained_layout=False)
        image = None
        panel_options = (
            {"show_xlabel": False, "show_ylabel": True},
            {"show_xlabel": False, "show_ylabel": False},
            {"show_xlabel": True, "show_ylabel": True},
        )
        for ax, (case_key, title), options in zip(axes.ravel()[:3], SNR_CASES, panel_options):
            image = plot_snr_panel(ax, case_key, title, cmap, color_norm, probe_points=probe_points, **options)
        plot_mismatch_panel(axes[1, 1], show_ylabel=False)

        for ax in axes.ravel():
            ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=2.4, pad=1.0)
            ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=1.3, pad=1.0)

        fig.subplots_adjust(left=0.13, right=0.845, bottom=0.105, top=0.925, wspace=0.065, hspace=0.11)

        if image is not None:
            cax = fig.add_axes([0.872, 0.30, 0.018, 0.52])
            colorbar = fig.colorbar(
                image,
                cax=cax,
                ticks=[0, 25, 50, 100, 150, 200],
            )
            colorbar.set_label("DECIGO axion-line SNR", fontsize=5.4, labelpad=1.2)
            colorbar.ax.tick_params(labelsize=4.8, length=2.0, pad=0.8)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "lowfre_decigo_resolved_diagnostics.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
