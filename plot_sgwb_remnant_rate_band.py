from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


BASE_DIR = Path(__file__).resolve().parent
FREQUENCY_DATA_DIR = BASE_DIR / "frequency_data"
FIGURE_DIR = BASE_DIR / "figures"


CURVE_SPECS = (
    (
        "highfre_axionplusbackreaction",
        {
            "upward": r"$\left|300\right\rangle\rightarrow\left|322\right\rangle$",
            "downward": r"$\left|322\right\rangle\rightarrow\left|300\right\rangle$",
        },
        "#0072B2",
    ),
    (
        "highfre644_axionplusbackreaction",
        {
            "upward": r"$\left|544\right\rangle\rightarrow\left|644\right\rangle$",
            "downward": r"$\left|644\right\rangle\rightarrow\left|544\right\rangle$",
        },
        "#009E73",
    ),
    (
        "highfre_pure_binary_template",
        {"upward": "pure template", "downward": "pure template"},
        "#C44E52",
    ),
)

PLOT_MIN_FREQUENCY_HZ = {
    "highfre_pure_binary_template": 5.0e-1,
}

SENSITIVITY_FILES = {
    "CE": BASE_DIR / "CE.csv",
    "DECIGO": BASE_DIR / "DECIGO.csv",
    "ET": BASE_DIR / "ET.csv",
    "LISA": BASE_DIR / "lisa.csv",
}

SENSITIVITY_STYLES = {
    "CE": {"color": "#4C566A", "linestyle": "--", "linewidth": 1.6, "alpha": 0.95},
    "DECIGO": {"color": "#7E57C2", "linestyle": "--", "linewidth": 1.6, "alpha": 0.95},
    "ET": {"color": "#8D6E63", "linestyle": "--", "linewidth": 1.6, "alpha": 0.95},
    "LISA": {"color": "#D16BA5", "linestyle": "--", "linewidth": 1.6, "alpha": 0.95},
}


@dataclass(frozen=True)
class OmegaCurve:
    key: str
    label: str
    color: str
    frequency_hz: np.ndarray
    omega_gw: np.ndarray
    r_eff_fiducial: float
    rate_model: str
    rate_evolution: str
    z_model: float


def parse_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.startswith("#"):
                break
            line = raw_line[1:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def normalize_suffix(suffix: str) -> str:
    suffix = str(suffix or "").strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    return suffix


def load_omega_curve(direction: str, key: str, label: str, color: str, curve_suffix: str = "") -> OmegaCurve:
    suffix = normalize_suffix(curve_suffix)
    path = FREQUENCY_DATA_DIR / f"{key}{suffix}_{direction}_omega_gw.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run powerspectrum.py and powerspectrumv.py before this script."
        )

    metadata = parse_metadata(path)
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]

    frequency_hz = np.asarray(data[:, 0], dtype=float)
    omega_gw = np.asarray(data[:, 1], dtype=float)
    valid = (
        np.isfinite(frequency_hz)
        & np.isfinite(omega_gw)
        & (frequency_hz > 0.0)
        & (omega_gw > 0.0)
    )

    return OmegaCurve(
        key=key,
        label=label,
        color=color,
        frequency_hz=frequency_hz[valid],
        omega_gw=omega_gw[valid],
        r_eff_fiducial=float(metadata.get("r_eff0_gpc3_yr", metadata.get("r0_gpc3_yr", "1.0"))),
        rate_model=metadata.get("rate_model", "unknown"),
        rate_evolution=metadata.get("rate_evolution", "constant"),
        z_model=float(metadata.get("z_model", "nan")),
    )


def load_sensitivity_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]

    frequency_hz = np.asarray(data[:, 0], dtype=float)
    sensitivity = np.asarray(data[:, 1], dtype=float)
    valid = (
        np.isfinite(frequency_hz)
        & np.isfinite(sensitivity)
        & (frequency_hz > 0.0)
        & (sensitivity > 0.0)
    )
    frequency_hz = frequency_hz[valid]
    sensitivity = sensitivity[valid]

    order = np.argsort(frequency_hz)
    frequency_hz = frequency_hz[order]
    sensitivity = sensitivity[order]

    unique_freq = np.unique(frequency_hz)
    envelope_sensitivity = []
    for freq in unique_freq:
        mask = frequency_hz == freq
        envelope_sensitivity.append(np.min(sensitivity[mask]))

    return np.asarray(unique_freq, dtype=float), np.asarray(envelope_sensitivity, dtype=float)


def smooth_sensitivity_curve(
    frequency_hz: np.ndarray,
    sensitivity: np.ndarray,
    sample_points: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    if frequency_hz.size < 3:
        return frequency_hz, sensitivity

    log_frequency = np.log10(frequency_hz)
    log_sensitivity = np.log10(sensitivity)
    interpolator = PchipInterpolator(log_frequency, log_sensitivity, extrapolate=False)
    dense_log_frequency = np.linspace(log_frequency.min(), log_frequency.max(), sample_points)
    dense_log_sensitivity = interpolator(dense_log_frequency)
    return 10.0**dense_log_frequency, 10.0**dense_log_sensitivity


def load_all_sensitivity_curves() -> list[dict[str, object]]:
    curves = []
    for detector_name, path in SENSITIVITY_FILES.items():
        frequency_hz, sensitivity = load_sensitivity_curve(path)
        smooth_frequency_hz, smooth_sensitivity = smooth_sensitivity_curve(frequency_hz, sensitivity)
        curves.append(
            {
                "label": detector_name,
                "frequency_hz": smooth_frequency_hz,
                "sensitivity": smooth_sensitivity,
                "style": SENSITIVITY_STYLES[detector_name],
            }
        )
    return curves


def plot_direction(
    direction: str,
    r_eff_min: float,
    r_eff_max: float,
    curve_suffix: str,
    output_suffix: str,
) -> Path:
    curves = [
        load_omega_curve(direction=direction, key=key, label=labels[direction], color=color, curve_suffix=curve_suffix)
        for key, labels, color in CURVE_SPECS
    ]
    sensitivity_curves = load_all_sensitivity_curves()
    z_values = np.array([curve.z_model for curve in curves], dtype=float)
    z_display = float(z_values[0]) if np.allclose(z_values, z_values[0], equal_nan=True) else np.nan
    evolutions = {curve.rate_evolution for curve in curves}
    evolution_display = next(iter(evolutions)) if len(evolutions) == 1 else "mixed"

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cm_to_inch = 1.0 / 2.54
    fig, ax = plt.subplots(figsize=(8.0 * cm_to_inch, 9.0 * cm_to_inch))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.35)

    power_handles = []
    sensitivity_handles = []

    for curve in curves:
        plot_min = float(PLOT_MIN_FREQUENCY_HZ.get(curve.key, 0.0))
        plot_mask = curve.frequency_hz >= plot_min
        if not np.any(plot_mask):
            continue
        frequency_hz = curve.frequency_hz[plot_mask]
        omega_gw = curve.omega_gw[plot_mask]
        scale_low = r_eff_min / curve.r_eff_fiducial
        scale_high = r_eff_max / curve.r_eff_fiducial
        band_low = omega_gw * min(scale_low, scale_high)
        band_high = omega_gw * max(scale_low, scale_high)

        ax.fill_between(
            frequency_hz,
            band_low,
            band_high,
            color=curve.color,
            alpha=0.12,
            linewidth=0.0,
        )
        line, = ax.plot(
            frequency_hz,
            omega_gw,
            color=curve.color,
            linewidth=1.9,
            label=curve.label,
        )
        power_handles.append(line)

    for sensitivity_curve in sensitivity_curves:
        line, = ax.plot(
            sensitivity_curve["frequency_hz"],
            sensitivity_curve["sensitivity"],
            label=sensitivity_curve["label"],
            **sensitivity_curve["style"],
        )
        sensitivity_handles.append(line)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observed frequency [Hz]", fontsize=8.5, labelpad=0.8)
    ax.set_ylabel(r"$\Omega_{\rm GW}(f)$", fontsize=8.5, labelpad=2.0)
    ax.set_xlim(left=1.0e-4)
    ax.set_ylim(bottom=1.0e-39)
    ax.grid(False)
    ax.margins(x=0.01, y=0.02)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=4.0,
        width=0.8,
        labelsize=7.4,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=2.0,
        width=0.6,
    )

    power_legend = fig.legend(
        handles=power_handles,
        title=(
            rf"$z_{{\rm model}}={z_display:g}$, {evolution_display}; "
            rf"solid: $R_{{\rm eff}}={curves[0].r_eff_fiducial:g}$; "
            rf"shaded: $[{r_eff_min:g},{r_eff_max:g}]\,{{\rm Gpc}}^{{-3}}{{\rm yr}}^{{-1}}$"
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        ncol=2,
        frameon=False,
        fontsize=7.3,
        title_fontsize=7.0,
        handlelength=2.6,
        columnspacing=1.0,
        labelspacing=0.4,
    )
    fig.add_artist(power_legend)
    fig.legend(
        handles=sensitivity_handles,
        title="Detector",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        fontsize=7.3,
        title_fontsize=7.6,
        handlelength=2.6,
        columnspacing=0.9,
        labelspacing=0.4,
    )

    suffix = normalize_suffix(output_suffix)
    output_path = FIGURE_DIR / f"sgwb_power_spectra_with_remnant_rate_band{suffix}_{direction}.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SGWB spectra with an effective remnant-cloud event-rate benchmark band."
    )
    parser.add_argument(
        "--r-eff-min",
        type=float,
        default=1.0,
        help="Lower edge of the effective remnant-cloud rate band in Gpc^-3 yr^-1. Default: 1.",
    )
    parser.add_argument(
        "--r-eff-max",
        type=float,
        default=1.0e4,
        help="Upper edge of the effective remnant-cloud rate band in Gpc^-3 yr^-1. Default: 1e4.",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=("upward", "downward"),
        default=("upward", "downward"),
        help="Which SGWB directions to plot.",
    )
    parser.add_argument(
        "--curve-suffix",
        default="",
        help="Suffix inserted before _upward/_downward in omega-curve filenames. Example: sfr.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix inserted into the generated figure names. Example: sfr.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.r_eff_min <= 0.0 or args.r_eff_max <= 0.0:
        raise ValueError("R_eff band edges must be positive.")

    for direction in args.directions:
        output_path = plot_direction(
            direction=direction,
            r_eff_min=args.r_eff_min,
            r_eff_max=args.r_eff_max,
            curve_suffix=args.curve_suffix,
            output_suffix=args.output_suffix,
        )
        first_curve = load_omega_curve(
            direction=direction,
            key=CURVE_SPECS[0][0],
            label=CURVE_SPECS[0][1][direction],
            color=CURVE_SPECS[0][2],
            curve_suffix=args.curve_suffix,
        )
        print(
            f"Saved {output_path} with R_eff=[{args.r_eff_min:g}, {args.r_eff_max:g}] "
            f"Gpc^-3 yr^-1, fiducial R_eff={first_curve.r_eff_fiducial:g}, "
            f"evolution={first_curve.rate_evolution}, z_model={first_curve.z_model:g}."
        )


if __name__ == "__main__":
    main()
