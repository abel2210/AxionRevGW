from __future__ import annotations

import csv
from pathlib import Path

import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import MaxNLocator


BASE_DIR = Path(__file__).resolve().parent
SNR_DATA_DIR = BASE_DIR / "snr_scan_data"
PROBE_DIR = BASE_DIR / "diagnostics" / "lowfre_mismatch_threshold_probe"
PROBE_REPORT_CSV = PROBE_DIR / "lowfre_mismatch_threshold_probe.csv"
PROBE_MISMATCH_DATA_DIR = PROBE_DIR / "mismatch_data"
PROBE_WAVEFORM_DATA_DIR = PROBE_DIR / "waveform_data"
FIGURE_DIR = BASE_DIR / "figures"

SNR_COLORBAR_MIN = 1.0e-4
SNR_COLORBAR_MAX = 1.0e2
SNR_OVER_COLOR = "#1f2a7a"
REFERENCE_DISTANCE_KPC = 100.0

SNR_CASES = (
    ("211v", r"$|211\rangle\!\to\!|21{-}1\rangle$", r"$q=1/150$"),
    ("322v", r"$|322\rangle\!\to\!|300\rangle$", r"$q=10^{-4}$"),
)

MISMATCH_CASES = (
    (
        "211v",
        "lowfre211v_snrprobe_decigo_mismatch_downward.txt",
        r"$211\downarrow$",
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


def load_mismatch_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 5:
        raise ValueError(f"{path} has {data.shape[1]} columns; expected at least 5.")

    observation_years = np.asarray(data[:, 0], dtype=float)
    effective_years = np.asarray(data[:, 1], dtype=float)
    mismatch = np.asarray(data[:, 2], dtype=float)
    x_values = np.where(np.isfinite(effective_years), effective_years, observation_years)
    finite = np.isfinite(x_values) & np.isfinite(mismatch)
    if not np.any(finite):
        raise ValueError(f"{path} does not contain finite mismatch samples.")

    x_values = x_values[finite]
    mismatch = mismatch[finite]
    order = np.argsort(x_values)
    x_values = x_values[order]
    mismatch = mismatch[order]
    if x_values[0] > 0.0:
        x_values = np.concatenate(([0.0], x_values))
        mismatch = np.concatenate(([0.0], mismatch))
    return x_values, mismatch


def load_resonance_time_years(case_key: str) -> float:
    if PROBE_REPORT_CSV.exists():
        with PROBE_REPORT_CSV.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("preset") == case_key and row.get("resonance_time_obs_s"):
                    return float(row["resonance_time_obs_s"]) / (365.25 * 86400.0)

    path = PROBE_WAVEFORM_DATA_DIR / f"lowfre{case_key}_snrprobe_axion_strain_time_window_downward.txt"
    if not path.exists():
        return np.nan
    prefix = "# selected_resonance_time_obs_s="
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(prefix):
                return float(line[len(prefix):].strip()) / (365.25 * 86400.0)
            if not line.startswith("#"):
                break
    return np.nan


def plot_snr_panel(
    ax,
    case_key: str,
    title: str,
    cmap,
    color_norm,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> object:
    alpha, distance_kpc, snr = load_snr_grid(case_key)
    grid = snr.T
    masked_grid = np.ma.masked_where(~np.isfinite(grid) | (grid <= 0.0), grid)
    levels = np.geomspace(SNR_COLORBAR_MIN, SNR_COLORBAR_MAX, 81)
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
        contour_levels = [level for level in (1.0, 8.0, 30.0, 100.0) if grid_min < level < grid_max]
    else:
        contour_levels = []
    if contour_levels:
        low_contours = [level for level in contour_levels if level < 30.0]
        high_contours = [level for level in contour_levels if level >= 30.0]
        if low_contours:
            contours = ax.contour(alpha, distance_kpc, grid, levels=low_contours, colors="#111111", linewidths=0.55)
            ax.clabel(contours, fmt="%g", fontsize=4.5, colors="#111111")
        if high_contours:
            contours = ax.contour(alpha, distance_kpc, grid, levels=high_contours, colors="#B00020", linewidths=0.65)
            ax.clabel(contours, fmt="%g", fontsize=4.5, colors="#B00020")

    ax.axhline(REFERENCE_DISTANCE_KPC, color="#62d26f", lw=0.65, alpha=0.9)
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
        x_values, mismatch = load_mismatch_curve(path)
        ax.plot(x_values, mismatch, label=label, **style)
        max_x = max(max_x, float(np.nanmax(x_values)))
        max_y = max(max_y, float(np.nanmax(mismatch)))

    resonance_times = np.asarray([load_resonance_time_years(case_key) for case_key, *_ in MISMATCH_CASES])
    resonance_times = resonance_times[np.isfinite(resonance_times)]
    if resonance_times.size:
        left = float(np.min(resonance_times))
        right = float(np.max(resonance_times))
        ax.axvspan(left, right, color="0.55", alpha=0.18, lw=0.0)
        ax.text(
            0.5 * (left + right),
            0.96,
            r"$t_{\rm res}$",
            transform=ax.get_xaxis_transform(),
            color="0.32",
            fontsize=4.5,
            ha="center",
            va="top",
        )

    ax.set_xlabel(r"$T_{\rm obs}$ [yr]", labelpad=0.8)
    if show_ylabel:
        ax.set_ylabel("Mismatch", labelpad=0.8)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlim(0.0, max_x if max_x > 0.0 else 1.0)
    ax.set_ylim(0.0, min(1.05, 1.10 * max(max_y, 1.0e-12)))
    ax.set_title("Fixed-parameter waveform mismatch", pad=1.0)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=3.2)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=1.8)
    ax.minorticks_on()
    ax.grid(False)
    ax.legend(loc="upper left", frameon=False, fontsize=4.5, handlelength=2.2, borderaxespad=0.35)


def main() -> None:
    cmap = LinearSegmentedColormap.from_list(
        "cool_decigo_snr",
        ["#f7fbff", "#dbeaf4", "#bdd8e9", "#8fbfd8", "#5a9ec6", "#2f79b7", "#2350a1"],
        N=256,
    )
    cmap.set_over(SNR_OVER_COLOR)
    color_norm = LogNorm(vmin=SNR_COLORBAR_MIN, vmax=SNR_COLORBAR_MAX)

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
        fig = plt.figure(figsize=(8.6 * cm_to_inch, 7.2 * cm_to_inch))
        grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.82))
        snr_axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
        mismatch_ax = fig.add_subplot(grid[1, :])
        image = None
        panel_options = (
            {"show_xlabel": True, "show_ylabel": True},
            {"show_xlabel": True, "show_ylabel": False},
        )
        for ax, (case_key, title, source_label), options in zip(snr_axes, SNR_CASES, panel_options):
            image = plot_snr_panel(ax, case_key, f"{title}\n{source_label}", cmap, color_norm, **options)
        plot_mismatch_panel(mismatch_ax, show_ylabel=True)

        for ax in (*snr_axes, mismatch_ax):
            ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=2.4, pad=1.0)
            ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=1.3, pad=1.0)

        fig.subplots_adjust(left=0.14, right=0.845, bottom=0.12, top=0.93, wspace=0.07, hspace=0.34)

        if image is not None:
            cax = fig.add_axes([0.872, 0.30, 0.018, 0.52])
            colorbar = fig.colorbar(
                image,
                cax=cax,
                ticks=[1.0e-4, 1.0e-2, 1.0, 1.0e2],
            )
            colorbar.set_label("DECIGO transition SNR", fontsize=5.4, labelpad=1.2)
            colorbar.ax.tick_params(labelsize=4.8, length=2.0, pad=0.8)

    def save_with_fallback(path: Path) -> Path:
        candidates = [path]
        candidates.extend(path.with_name(f"{path.stem}_updated{idx}{path.suffix}") for idx in range(1, 20))
        last_error = None
        for candidate in candidates:
            try:
                fig.savefig(candidate, dpi=300, bbox_inches="tight")
                return candidate
            except PermissionError as exc:
                last_error = exc
                continue
        raise PermissionError(f"Could not write {path} or any fallback path.") from last_error

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "lowfre_decigo_resolved_diagnostics.pdf"
    saved_out = save_with_fallback(out)
    print(f"Saved {saved_out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
