from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import time

import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

import highfre644
import highfre644v
from remnant_rate_models import (
    effective_local_rate_gpc3_yr,
    kerr_horizon_frequency_dimensionless,
    remnant_cloud_rate_density_si,
    superradiant_spin_threshold,
)
from transition_geometry import angle_average_factor_from_metadata


BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = BASE_DIR / "benchmark_highfreq_q001"

ALPHA = 0.30
BH_SPIN = 0.70
ECCENTRICITY_INIT = 0.65
CLOUD_MASS_FRACTION = 0.005
DISTANCE_MPC = 0.001
SPECTRUM_WINDOW_ORBITS = 160.0
SPECTRUM_WINDOW_MODE = "first_selected_orbits"
SUMMARY_HALF_WINDOW_ORBITS = 40.0
SUMMARY_LZ_WINDOW_FACTOR = 0.55

C = 2.99792458e8
G = 6.6743e-11
MPC = 3.085677581491367e22
GPC = 1.0e3 * MPC
YEAR = 365.25 * 24.0 * 3600.0
H0_KM_S_MPC = 67.74
H0 = H0_KM_S_MPC * 1.0e3 / MPC
OMEGA_M = 0.3089
OMEGA_L = 1.0 - OMEGA_M
RHO_C = 3.0 * H0**2 * C**2 / (8.0 * np.pi * G)

SENSITIVITY_FILES = {
    "CE": BASE_DIR / "CE.csv",
    "DECIGO": BASE_DIR / "DECIGO.csv",
    "ET": BASE_DIR / "ET.csv",
    "LISA": BASE_DIR / "lisa.csv",
}

SENSITIVITY_STYLES = {
    "CE": {"color": "#4C566A", "linestyle": "--", "linewidth": 1.0, "alpha": 0.75},
    "DECIGO": {"color": "#7E57C2", "linestyle": "--", "linewidth": 1.0, "alpha": 0.75},
    "ET": {"color": "#8D6E63", "linestyle": "--", "linewidth": 1.0, "alpha": 0.75},
    "LISA": {"color": "#D16BA5", "linestyle": "--", "linewidth": 1.0, "alpha": 0.75},
}


@dataclass(frozen=True)
class Benchmark:
    tag: str
    primary_mass_msun: float
    mass_ratio: float = 0.01

    @property
    def secondary_mass_msun(self) -> float:
        return self.primary_mass_msun * self.mass_ratio


@dataclass(frozen=True)
class SGWBRateConfig:
    r0_gpc3_yr: float = 1.0
    z_model: float = 10.0
    rate_evolution: str = "constant"
    f_ret: float = 1.0
    f_2g: float = 1.0
    f_cloud: float = 1.0
    f_duty: float = 1.0
    remnant_spin: float = BH_SPIN
    spin_model: str = "fixed_reference"


BENCHMARKS = (
    Benchmark(tag="m1_1_q001", primary_mass_msun=1.0),
    Benchmark(tag="m1_0p1_q001", primary_mass_msun=0.1),
    Benchmark(tag="m1_0p01_q001", primary_mass_msun=0.01),
)

RATE_CONFIG = SGWBRateConfig()


def relative_to_base(path: Path) -> str:
    return str(path.resolve().relative_to(BASE_DIR.resolve()))


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


def load_frequency_amplitude_export(path: Path) -> tuple[dict[str, str], np.ndarray]:
    metadata = parse_metadata(path)
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return metadata, data


def hubble_e(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)


def remnant_cloud_rate_si(z: np.ndarray, config: SGWBRateConfig) -> np.ndarray:
    return remnant_cloud_rate_density_si(
        z,
        r0_gpc3_yr=config.r0_gpc3_yr,
        evolution=config.rate_evolution,
        f_ret=config.f_ret,
        f_2g=config.f_2g,
        f_cloud=config.f_cloud,
        f_duty=config.f_duty,
    )


def effective_rate(config: SGWBRateConfig) -> float:
    return effective_local_rate_gpc3_yr(
        r0_gpc3_yr=config.r0_gpc3_yr,
        f_ret=config.f_ret,
        f_2g=config.f_2g,
        f_cloud=config.f_cloud,
        f_duty=config.f_duty,
    )


def transition_cloud_m(label: str) -> int:
    if "644" in label:
        return 4
    if "322" in label:
        return 2
    return 0


def sgwb_header_lines(
    label: str,
    metadata: dict[str, str],
    config: SGWBRateConfig,
    nu_cut_hz: float,
    empty: bool = False,
) -> list[str]:
    alpha = float(metadata.get("alpha", ALPHA))
    cloud_m = transition_cloud_m(label)
    threshold = superradiant_spin_threshold(alpha, cloud_m) if cloud_m > 0 else np.nan
    lines = [
        f"label={label}",
        "rate_model=effective_small_bh_event_rate",
        f"r0_gpc3_yr={config.r0_gpc3_yr:.16e}",
        f"r_eff0_gpc3_yr={effective_rate(config):.16e}",
        f"rate_evolution={config.rate_evolution}",
        f"z_model={config.z_model:.16e}",
        f"f_ret={config.f_ret:.16e}",
        f"f_2g={config.f_2g:.16e}",
        f"f_cloud={config.f_cloud:.16e}",
        f"f_duty={config.f_duty:.16e}",
        f"remnant_spin={config.remnant_spin:.16e}",
        f"spin_model={config.spin_model}",
        f"kerr_m_omega_h={kerr_horizon_frequency_dimensionless(config.remnant_spin):.16e}",
        f"cloud_azimuthal_m={cloud_m:d}",
        f"superradiant_spin_threshold={threshold:.16e}",
        f"primary_mass_msun={float(metadata.get('primary_mass_msun', 'nan')):.16e}",
        f"secondary_mass_msun={float(metadata.get('secondary_mass_msun', 'nan')):.16e}",
        f"nu_cut_hz={nu_cut_hz:.16e}",
    ]
    if empty:
        lines.append("empty_source_spectrum=true")
    lines.append("columns=frequency_hz omega_gw")
    return lines


def build_energy_spectrum(metadata: dict[str, str], data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freq_obs_hz = np.asarray(data[:, 0], dtype=float)
    amplitude_obs = np.asarray(data[:, 1], dtype=float)
    redshift = float(metadata.get("redshift", "0.0"))
    distance_m = float(metadata["luminosity_distance_m"])

    source_freq_hz = freq_obs_hz * (1.0 + redshift)
    source_amplitude = amplitude_obs / (1.0 + redshift)
    angle_average_factor = angle_average_factor_from_metadata(metadata)
    dE_dnu = (
        # Total emitted energy spectrum after integrating over source angles.
        (2.0 * np.pi**2 * C**3 / G)
        * distance_m**2
        * source_freq_hz**2
        * angle_average_factor
        * source_amplitude**2
    )

    valid = (
        np.isfinite(source_freq_hz)
        & np.isfinite(dE_dnu)
        & (source_freq_hz > 0.0)
        & (dE_dnu > 0.0)
    )
    source_freq_hz = source_freq_hz[valid]
    dE_dnu = dE_dnu[valid]
    order = np.argsort(source_freq_hz)
    return source_freq_hz[order], dE_dnu[order]


def build_log_interpolator(x_values: np.ndarray, y_values: np.ndarray):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    log_x = np.log(x_values)
    log_y = np.log(y_values)
    x_min = float(x_values[0])
    x_max = float(x_values[-1])

    def interpolate(x_new):
        x_new = np.asarray(x_new, dtype=float)
        result = np.zeros_like(x_new)
        mask = (x_new >= x_min) & (x_new <= x_max)
        if np.any(mask):
            result[mask] = np.exp(np.interp(np.log(x_new[mask]), log_x, log_y))
        return result

    return interpolate, x_min, x_max


def make_redshift_grid(z_upper: float, points: int = 512) -> np.ndarray:
    if z_upper <= 0.0:
        return np.array([0.0], dtype=float)
    if z_upper < 1.0e-6:
        return np.linspace(0.0, z_upper, 32)
    return np.expm1(np.linspace(0.0, np.log1p(z_upper), points))


def compute_omega_gw_curve(
    source_freq_hz: np.ndarray,
    dE_dnu: np.ndarray,
    metadata: dict[str, str],
    label: str,
    output_path: Path,
    config: SGWBRateConfig = RATE_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(source_freq_hz) == 0 or len(dE_dnu) == 0:
        freq_obs_hz = np.logspace(-4.0, 5.0, 500)
        omega_gw = np.zeros_like(freq_obs_hz)
        header = "\n".join(sgwb_header_lines(label, metadata, config, 0.0, empty=True))
        np.savetxt(output_path, np.column_stack((freq_obs_hz, omega_gw)), header=header, comments="# ")
        return freq_obs_hz, omega_gw

    interpolate_dE, nu_min, nu_cut = build_log_interpolator(source_freq_hz, dE_dnu)
    freq_min_obs = max(nu_min / (1.0 + config.z_model), 1.0e-8)
    freq_max_obs = nu_cut
    freq_obs_hz = np.logspace(np.log10(freq_min_obs), np.log10(freq_max_obs), 500)
    omega_gw = np.zeros_like(freq_obs_hz)

    for idx, freq_obs in enumerate(freq_obs_hz):
        z_sup = min(config.z_model, max(nu_cut / freq_obs - 1.0, 0.0))
        if z_sup <= 0.0:
            continue
        z_grid = make_redshift_grid(z_sup)
        nu_source = (1.0 + z_grid) * freq_obs
        integrand = (
            remnant_cloud_rate_si(z_grid, config)
            * interpolate_dE(nu_source)
            / ((1.0 + z_grid) * hubble_e(z_grid))
        )
        omega_gw[idx] = freq_obs * np.trapezoid(integrand, z_grid) / (RHO_C * H0)

    header = "\n".join(sgwb_header_lines(label, metadata, config, nu_cut))
    np.savetxt(output_path, np.column_stack((freq_obs_hz, omega_gw)), header=header, comments="# ")
    return freq_obs_hz, omega_gw


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

    unique_frequency = np.unique(frequency_hz)
    envelope = []
    for frequency in unique_frequency:
        mask = frequency_hz == frequency
        envelope.append(np.min(sensitivity[mask]))
    return unique_frequency, np.asarray(envelope, dtype=float)


def smooth_sensitivity_curve(frequency_hz: np.ndarray, sensitivity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if frequency_hz.size < 3:
        return frequency_hz, sensitivity
    log_frequency = np.log10(frequency_hz)
    log_sensitivity = np.log10(sensitivity)
    interpolator = PchipInterpolator(log_frequency, log_sensitivity, extrapolate=False)
    dense_log_frequency = np.linspace(log_frequency.min(), log_frequency.max(), 1200)
    dense_log_sensitivity = interpolator(dense_log_frequency)
    return 10.0**dense_log_frequency, 10.0**dense_log_sensitivity


def load_all_sensitivity_curves() -> list[dict[str, object]]:
    curves = []
    for detector_name, path in SENSITIVITY_FILES.items():
        if not path.exists():
            continue
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


def first_selected_event(results: dict) -> dict:
    for key in ("local_lz_events", "resonance_events"):
        events = list(results["cloud"].get(key, []))
        selected = [event for event in events if bool(event.get("selected", False))]
        crossed = [event for event in selected if bool(event.get("crossed", False))]
        if crossed:
            crossed.sort(key=lambda event: float(event.get("t_source", 0.0)))
            return crossed[0]
        if selected:
            selected.sort(key=lambda event: float(event.get("t_source", 0.0)))
            return selected[0]
    events = list(results["cloud"].get("local_lz_events", [])) or list(results["cloud"].get("resonance_events", []))
    if events:
        events.sort(key=lambda event: float(event.get("t_source", 0.0)))
        return events[0]
    raise RuntimeError("No selected Bohr resonance event found.")


def relative_delta_a(results: dict, t_values: np.ndarray) -> np.ndarray:
    orbit = results["orbit"]
    template = results["template_orbit"]
    a = np.interp(t_values, orbit["t"], orbit["a"])
    a_template = np.interp(t_values, template["t"], template["a"])
    a_res = np.interp(float(results["cloud"]["resonance_time"]), orbit["t"], orbit["a"])
    return (a - a_template) / max(abs(a_res), 1.0e-300)


def binary_phase_residual_cycles(results: dict, t_values: np.ndarray, t_res: float) -> np.ndarray:
    orbit = results["orbit"]
    template = results["template_orbit"]
    phi = np.asarray(orbit["solution"].sol(t_values)[2], dtype=float)
    phi_template = np.asarray(template["solution"].sol(t_values)[2], dtype=float)
    phi_res = float(np.asarray(orbit["solution"].sol(t_res)[2], dtype=float))
    phi_template_res = float(np.asarray(template["solution"].sol(t_res)[2], dtype=float))
    return (phi - phi_template - (phi_res - phi_template_res)) / (2.0 * np.pi)


def direction_signed_axion_strain(sim, cloud: dict, t_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cg_r, cg_i, ce_r, ce_i = cloud["solution"].sol(t_values)
    overlap = np.conj(cg_r + 1j * cg_i) * (ce_r + 1j * ce_i)
    signed_omega = getattr(sim, "transition_energy_change_omega", sim.transition_omega)
    phase = signed_omega * t_values
    h_axion = -sim._cloud_amplitude() * (
        overlap.real * np.cos(phase) - overlap.imag * np.sin(phase)
    )
    return h_axion, np.abs(overlap)


def axis_scale(span_seconds: float) -> tuple[float, str]:
    if span_seconds < 0.3:
        return 1.0e3, "ms"
    if span_seconds < 300.0:
        return 1.0, "s"
    if span_seconds < 3.0 * 3600.0:
        return 1.0 / 60.0, "min"
    if span_seconds < 7.0 * 86400.0:
        return 1.0 / 3600.0, "hr"
    return 1.0 / 86400.0, "days"


def strain_scale_exponent(*arrays: np.ndarray) -> int:
    peak = 0.0
    for values in arrays:
        if np.size(values):
            peak = max(peak, float(np.nanmax(np.abs(values))))
    if not np.isfinite(peak) or peak <= 0.0:
        return 0
    return int(np.floor(np.log10(peak)))


def add_phase_residual_inset(ax, x_values: np.ndarray, phase_residual: np.ndarray, color: str) -> None:
    inset = ax.inset_axes([0.56, 0.13, 0.38, 0.32])
    inset.set_facecolor((1.0, 1.0, 1.0, 0.88))
    inset.axhline(0.0, color="0.45", lw=0.8, alpha=0.25)
    inset.axvline(0.0, color="0.45", lw=0.8, alpha=0.25)
    inset.plot(x_values, phase_residual, color=color, lw=0.85)
    ylim = max(float(np.nanmax(np.abs(phase_residual))) * 1.15, 1.0e-5)
    inset.set_ylim(-ylim, ylim)
    inset.set_xticks([np.nanmin(x_values), 0.0, np.nanmax(x_values)])
    inset.set_title(r"$\Delta\Phi_{\rm bin}/2\pi$", fontsize=4.6, pad=0.6)
    inset.tick_params(axis="both", which="major", labelsize=4.0, direction="in", top=True, right=True, pad=0.4)
    for spine in inset.spines.values():
        spine.set_linewidth(0.45)
        spine.set_alpha(0.75)


def plot_case(
    ax_orbit,
    ax_wave,
    sim,
    results: dict,
    title: str,
    orbit_color: str,
    time_scale: float,
    time_unit: str,
    half_window_s: float,
    strain_exponent: int,
    show_left_labels: bool,
    show_coherence_axis: bool,
) -> dict[str, float]:
    event = first_selected_event(results)
    t_res = float(event.get("t_source", results["cloud"]["resonance_time"]))
    orbit = results["orbit"]
    t_start = max(float(orbit["t"][0]), t_res - half_window_s)
    t_stop = min(float(orbit["t"][-1]), t_res + half_window_s)
    t_orbit = np.linspace(t_start, t_stop, 1600)
    delta_a = relative_delta_a(results, t_orbit)

    window = sim.build_waveform_window_between(
        orbit,
        results["cloud"],
        t_start,
        t_stop,
        sample_points=12000,
    )
    t_wave = np.asarray(window["t_source"], dtype=float)
    x_wave = (t_wave - t_res) * time_scale
    x_orbit = (t_orbit - t_res) * time_scale
    h_axion, coherence = direction_signed_axion_strain(sim, results["cloud"], t_wave)
    phase_residual = binary_phase_residual_cycles(results, t_wave, t_res)

    for ax in (ax_orbit, ax_wave):
        ax.axvline(0.0, color="0.15", lw=0.85, alpha=0.78)

    ax_orbit.plot(x_orbit, delta_a, color=orbit_color, lw=1.15)
    ax_orbit.set_title(title, fontsize=6.1, pad=1.2)
    if show_left_labels:
        ax_orbit.set_ylabel(r"$(a-a_{\rm P})/a_{\rm res}$", fontsize=5.8, labelpad=1.0)
    else:
        ax_orbit.tick_params(axis="y", labelleft=False)
    delta_limit = max(1.2 * float(np.nanmax(np.abs(delta_a))), 1.0e-4)
    ax_orbit.set_ylim(-delta_limit, delta_limit)

    stride = max(1, int(np.ceil(len(x_wave) / 1400)))
    scaled_h = h_axion / (10.0**strain_exponent)
    ax_wave.plot(
        x_wave[::stride],
        scaled_h[::stride],
        color="#1F78B4",
        lw=0.42,
        alpha=0.82,
    )
    if show_left_labels:
        ax_wave.set_ylabel(rf"$h_a/10^{{{strain_exponent}}}$", fontsize=5.8, color="#1F78B4", labelpad=1.0)
    else:
        ax_wave.tick_params(axis="y", labelleft=False)
    ax_wave.tick_params(axis="y", labelcolor="#1F78B4")
    ax_wave.set_xlabel(rf"$t-t_{{\rm res}}$ [{time_unit}]", fontsize=5.8, labelpad=1.0)

    ax_coh = ax_wave.twinx()
    ax_wave.set_zorder(ax_coh.get_zorder() + 1)
    ax_wave.patch.set_visible(False)
    ax_coh.patch.set_visible(False)
    ax_coh.plot(x_wave, coherence, color="#D55E00", lw=1.0, alpha=0.96)
    if show_coherence_axis:
        ax_coh.set_ylabel(r"$|c_i^\ast\tilde c_f|$", fontsize=5.8, color="#D55E00", labelpad=1.0)
        ax_coh.tick_params(axis="y", labelcolor="#D55E00", pad=1.0)
    else:
        ax_coh.tick_params(axis="y", labelright=False, right=False)

    add_phase_residual_inset(ax_wave, x_wave, phase_residual, orbit_color)

    for ax in (ax_orbit, ax_wave, ax_coh):
        ax.tick_params(axis="both", which="major", labelsize=5.1, direction="in", top=True, right=True, pad=1.0)
    for ax in (ax_orbit, ax_wave):
        ax.set_xlim(-half_window_s * time_scale, half_window_s * time_scale)
        ax.grid(False)

    return {
        "peak_axion_strain": float(np.nanmax(np.abs(h_axion))),
        "peak_coherence": float(np.nanmax(coherence)),
        "max_abs_delta_a": float(np.nanmax(np.abs(delta_a))),
        "max_abs_phase_cycles": float(np.nanmax(np.abs(phase_residual))),
        "t_res_source_s": t_res,
    }


def plot_orbit_time_summary(benchmark: Benchmark, up_sim, up_results: dict, down_sim, down_results: dict) -> tuple[Path, dict[str, float], dict[str, float]]:
    bench_dir = BENCHMARK_ROOT / benchmark.tag
    figure_dir = bench_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    local_period_s = 1.0 / max(up_sim.transition_frequency_hz / max(up_sim.resonance_harmonic, 1), 1.0e-300)
    half_window_s = SUMMARY_HALF_WINDOW_ORBITS * local_period_s
    for results in (up_results, down_results):
        for event in results["cloud"].get("local_lz_events", []):
            if not event.get("selected", False):
                continue
            start = float(event.get("lz_window_start_source", event.get("t_source", 0.0)))
            stop = float(event.get("lz_window_stop_source", event.get("t_source", 0.0)))
            half_window_s = max(half_window_s, SUMMARY_LZ_WINDOW_FACTOR * 0.5 * max(stop - start, 0.0))
    scale, unit = axis_scale(2.0 * half_window_s)

    up_event = first_selected_event(up_results)
    down_event = first_selected_event(down_results)
    up_t = float(up_event.get("t_source", up_results["cloud"]["resonance_time"]))
    down_t = float(down_event.get("t_source", down_results["cloud"]["resonance_time"]))
    up_window = up_sim.build_waveform_window_between(
        up_results["orbit"],
        up_results["cloud"],
        max(float(up_results["orbit"]["t"][0]), up_t - half_window_s),
        min(float(up_results["orbit"]["t"][-1]), up_t + half_window_s),
        sample_points=3000,
    )
    down_window = down_sim.build_waveform_window_between(
        down_results["orbit"],
        down_results["cloud"],
        max(float(down_results["orbit"]["t"][0]), down_t - half_window_s),
        min(float(down_results["orbit"]["t"][-1]), down_t + half_window_s),
        sample_points=3000,
    )
    up_h, _ = direction_signed_axion_strain(up_sim, up_results["cloud"], up_window["t_source"])
    down_h, _ = direction_signed_axion_strain(down_sim, down_results["cloud"], down_window["t_source"])
    common_strain_exponent = strain_scale_exponent(up_h, down_h)

    cm_to_inch = 1.0 / 2.54
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.6 * cm_to_inch, 7.1 * cm_to_inch),
        sharex=True,
        constrained_layout=False,
    )
    up_metrics = plot_case(
        axes[0, 0],
        axes[1, 0],
        up_sim,
        up_results,
        r"$|544\rangle\rightarrow|644\rangle$",
        "#B23A48",
        scale,
        unit,
        half_window_s,
        common_strain_exponent,
        show_left_labels=True,
        show_coherence_axis=False,
    )
    down_metrics = plot_case(
        axes[0, 1],
        axes[1, 1],
        down_sim,
        down_results,
        r"$|644\rangle\rightarrow|544\rangle$",
        "#009E73",
        scale,
        unit,
        half_window_s,
        common_strain_exponent,
        show_left_labels=False,
        show_coherence_axis=True,
    )
    axes[0, 0].text(0.03, 0.86, "upward", transform=axes[0, 0].transAxes, fontsize=5.8)
    axes[0, 1].text(0.03, 0.86, "downward", transform=axes[0, 1].transAxes, fontsize=5.8)
    fig.subplots_adjust(left=0.14, right=0.88, bottom=0.115, top=0.915, wspace=0.055, hspace=0.055)

    output_path = figure_dir / f"bohr_orbit_time_summary_644_pair_{benchmark.tag}.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path, up_metrics, down_metrics


def run_transition(module, benchmark: Benchmark, direction: str):
    bench_dir = BENCHMARK_ROOT / benchmark.tag
    module_suffix = "" if direction == "upward" else "v"
    module_stem = f"{benchmark.tag}_highfre644{module_suffix}"
    sim = module.EccentricResonantTidalGA(
        M_bh=benchmark.primary_mass_msun,
        M_star=benchmark.secondary_mass_msun,
        alpha=ALPHA,
        bh_spin=BH_SPIN,
        distance_Mpc=DISTANCE_MPC,
        z=0.0,
        e_init=ECCENTRICITY_INIT,
        f_orb_init=None,
        cloud_mass_fraction=CLOUD_MASS_FRACTION,
        save_figure_dir=relative_to_base(bench_dir / "figures"),
        save_frequency_data_dir=relative_to_base(bench_dir / "frequency_data"),
        save_time_series_data_dir=relative_to_base(bench_dir / "waveform_data"),
        module_stem=module_stem,
        direction_tag=direction,
    )
    scaled_initial_duration_yr = 2.0e-8 * (benchmark.primary_mass_msun / 1.0e-3) * (0.1 / benchmark.mass_ratio)
    max_duration_yr = max(5.0e-6, 80.0 * scaled_initial_duration_yr)
    duration_yr = sim.recommended_duration_to_cover_selected_resonance(
        initial_duration_yr=scaled_initial_duration_yr,
        max_duration_yr=max_duration_yr,
        post_event_padding_orbits=220.0,
    )
    start = time.time()
    results = sim.run(
        duration_yr=duration_yr,
        secular_samples=900,
        zoom_orbits=20,
        zoom_points=8192,
        spectrum_orbits=SPECTRUM_WINDOW_ORBITS,
        spectrum_points=8192,
        spectrum_pad_factor=4,
        spectrum_window_mode=SPECTRUM_WINDOW_MODE,
        save_exports=True,
    )
    elapsed_s = time.time() - start
    sim.print_summary(results, elapsed_s)
    return sim, results, duration_yr, elapsed_s


def omega_from_export(export_path: Path, label: str, output_path: Path) -> tuple[np.ndarray, np.ndarray]:
    metadata, data = load_frequency_amplitude_export(export_path)
    source_freq_hz, dE_dnu = build_energy_spectrum(metadata, data)
    return compute_omega_gw_curve(source_freq_hz, dE_dnu, metadata, label, output_path)


def plot_sgwb_benchmark(benchmark: Benchmark, curves: list[dict[str, object]]) -> Path:
    bench_dir = BENCHMARK_ROOT / benchmark.tag
    figure_dir = bench_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_curves = load_all_sensitivity_curves()

    cm_to_inch = 1.0 / 2.54
    fig, ax = plt.subplots(figsize=(8.0 * cm_to_inch, 9.4 * cm_to_inch))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.90, bottom=0.43)

    handles = []
    for curve in curves:
        frequency = np.asarray(curve["frequency_hz"], dtype=float)
        omega = np.asarray(curve["omega_gw"], dtype=float)
        valid = np.isfinite(frequency) & np.isfinite(omega) & (frequency > 0.0) & (omega > 0.0)
        if not np.any(valid):
            continue
        frequency = frequency[valid]
        omega = omega[valid]
        band_low = omega
        band_high = 1.0e4 * omega
        ax.fill_between(frequency, band_low, band_high, color=curve["color"], alpha=0.11, linewidth=0.0)
        line, = ax.plot(
            frequency,
            omega,
            color=curve["color"],
            linewidth=1.65,
            linestyle=curve.get("linestyle", "-"),
            label=curve["label"],
        )
        handles.append(line)

    sensitivity_handles = []
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
    ax.set_title(
        rf"$M_1={benchmark.primary_mass_msun:g}M_\odot$, $q={benchmark.mass_ratio:g}$",
        fontsize=8.0,
        pad=2.0,
    )
    ax.set_xlim(left=1.0e-4)
    ax.set_ylim(bottom=1.0e-39)
    ax.grid(False)
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=4.0, width=0.8, labelsize=7.3)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=2.0, width=0.6)

    power_legend = fig.legend(
        handles=handles,
        title=rf"solid: $R_{{\rm eff}}=1$; shaded: $1$--$10^4$",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.14),
        ncol=1,
        frameon=False,
        fontsize=7.1,
        title_fontsize=6.9,
        handlelength=2.4,
        labelspacing=0.35,
    )
    fig.add_artist(power_legend)
    fig.legend(
        handles=sensitivity_handles,
        title="Detector",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
        fontsize=7.1,
        title_fontsize=7.2,
        handlelength=2.2,
        columnspacing=0.8,
        labelspacing=0.35,
    )

    output_path = figure_dir / f"sgwb_power_spectra_with_rate_band_{benchmark.tag}.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_sgwb_products(benchmark: Benchmark) -> tuple[Path, dict[str, float]]:
    bench_dir = BENCHMARK_ROOT / benchmark.tag
    frequency_dir = bench_dir / "frequency_data"
    up_total = frequency_dir / f"{benchmark.tag}_highfre644_axion_backreaction_total_frequency_amplitude.txt"
    down_total = frequency_dir / f"{benchmark.tag}_highfre644v_axion_backreaction_total_frequency_amplitude.txt"
    pure_template = frequency_dir / f"{benchmark.tag}_highfre644_pure_binary_template_frequency_amplitude.txt"

    curves = []
    peak_values: dict[str, float] = {}
    for key, export_path, label, color, linestyle in (
        ("upward", up_total, r"$|544\rangle\rightarrow|644\rangle$", "#B23A48", "-"),
        ("downward", down_total, r"$|644\rangle\rightarrow|544\rangle$", "#009E73", "-"),
        ("pure", pure_template, "pure PN template", "#C44E52", "--"),
    ):
        omega_path = frequency_dir / f"{benchmark.tag}_highfre644_{key}_omega_gw.txt"
        frequency_hz, omega_gw = omega_from_export(export_path, f"highfre644 {key}", omega_path)
        curves.append(
            {
                "key": key,
                "label": label,
                "frequency_hz": frequency_hz,
                "omega_gw": omega_gw,
                "color": color,
                "linestyle": linestyle,
            }
        )
        positive = omega_gw[np.isfinite(omega_gw) & (omega_gw > 0.0)]
        peak_values[f"omega_peak_{key}"] = float(np.max(positive)) if positive.size else 0.0

    sgwb_path = plot_sgwb_benchmark(benchmark, curves)
    return sgwb_path, peak_values


def write_summary(rows: list[dict[str, object]], best_tag: str) -> None:
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = BENCHMARK_ROOT / "summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = BENCHMARK_ROOT / "summary.md"
    lines = [
        "# q=0.01 high-frequency Bohr benchmarks",
        "",
        f"Shared parameters: alpha={ALPHA}, a_star={BH_SPIN}, e0={ECCENTRICITY_INIT}, Mc/M1={CLOUD_MASS_FRACTION}, spectrum window={SPECTRUM_WINDOW_ORBITS:g} orbits.",
        f"Selection criterion used by this script: largest peak Omega_GW among the upward and downward |544><->|644| curves. Current best: `{best_tag}`.",
        "",
        "| tag | M1 [Msun] | q | f_trans [Hz] | max Omega | max |h_a| up | max |h_a| down | orbit figure | SGWB figure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        max_omega = max(float(row["omega_peak_upward"]), float(row["omega_peak_downward"]))
        lines.append(
            "| {tag} | {primary_mass_msun:.6g} | {mass_ratio:.3g} | {transition_frequency_hz:.6g} | "
            "{max_omega:.6e} | {up_peak_axion_strain:.6e} | {down_peak_axion_strain:.6e} | "
            "{orbit_figure} | {sgwb_figure} |".format(
                tag=row["tag"],
                primary_mass_msun=float(row["primary_mass_msun"]),
                mass_ratio=float(row["mass_ratio"]),
                transition_frequency_hz=float(row["transition_frequency_hz"]),
                max_omega=max_omega,
                up_peak_axion_strain=float(row["up_peak_axion_strain"]),
                down_peak_axion_strain=float(row["down_peak_axion_strain"]),
                orbit_figure=Path(str(row["orbit_figure"])).name,
                sgwb_figure=Path(str(row["sgwb_figure"])).name,
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def run_benchmark(benchmark: Benchmark) -> dict[str, object]:
    print(f"\n=== Running {benchmark.tag}: M1={benchmark.primary_mass_msun:g} Msun, q={benchmark.mass_ratio:g} ===")
    up_sim, up_results, up_duration_yr, up_elapsed_s = run_transition(highfre644, benchmark, "upward")
    down_sim, down_results, down_duration_yr, down_elapsed_s = run_transition(highfre644v, benchmark, "downward")
    orbit_path, up_metrics, down_metrics = plot_orbit_time_summary(benchmark, up_sim, up_results, down_sim, down_results)
    sgwb_path, sgwb_metrics = build_sgwb_products(benchmark)
    up_event = first_selected_event(up_results)
    down_event = first_selected_event(down_results)

    return {
        "tag": benchmark.tag,
        "primary_mass_msun": benchmark.primary_mass_msun,
        "secondary_mass_msun": benchmark.secondary_mass_msun,
        "mass_ratio": benchmark.mass_ratio,
        "alpha": ALPHA,
        "bh_spin": BH_SPIN,
        "eccentricity_init": ECCENTRICITY_INIT,
        "cloud_mass_fraction": CLOUD_MASS_FRACTION,
        "transition_frequency_hz": up_sim.transition_frequency_hz,
        "orbital_frequency_res_hz": up_sim.transition_frequency_hz / max(up_sim.resonance_harmonic, 1),
        "up_duration_yr": up_duration_yr,
        "down_duration_yr": down_duration_yr,
        "up_elapsed_s": up_elapsed_s,
        "down_elapsed_s": down_elapsed_s,
        "up_resonance_time_s": float(up_event.get("t_source", up_results["cloud"]["resonance_time"])),
        "down_resonance_time_s": float(down_event.get("t_source", down_results["cloud"]["resonance_time"])),
        "up_crossed": bool(up_event.get("crossed", False)),
        "down_crossed": bool(down_event.get("crossed", False)),
        "up_peak_axion_strain": up_metrics["peak_axion_strain"],
        "down_peak_axion_strain": down_metrics["peak_axion_strain"],
        "up_peak_coherence": up_metrics["peak_coherence"],
        "down_peak_coherence": down_metrics["peak_coherence"],
        "up_max_abs_delta_a": up_metrics["max_abs_delta_a"],
        "down_max_abs_delta_a": down_metrics["max_abs_delta_a"],
        "up_max_abs_phase_cycles": up_metrics["max_abs_phase_cycles"],
        "down_max_abs_phase_cycles": down_metrics["max_abs_phase_cycles"],
        "omega_peak_upward": sgwb_metrics["omega_peak_upward"],
        "omega_peak_downward": sgwb_metrics["omega_peak_downward"],
        "omega_peak_pure": sgwb_metrics["omega_peak_pure"],
        "orbit_figure": relative_to_base(orbit_path),
        "sgwb_figure": relative_to_base(sgwb_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run q=0.01 high-frequency Bohr benchmark models.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional benchmark tags to run. Defaults to all three q=0.01 benchmarks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_tags = set(args.only or [])
    selected = [benchmark for benchmark in BENCHMARKS if not selected_tags or benchmark.tag in selected_tags]
    if not selected:
        raise SystemExit(f"No benchmark selected from tags: {', '.join(benchmark.tag for benchmark in BENCHMARKS)}")

    rows = [run_benchmark(benchmark) for benchmark in selected]
    best = max(
        rows,
        key=lambda row: max(float(row["omega_peak_upward"]), float(row["omega_peak_downward"])),
    )
    write_summary(rows, str(best["tag"]))
    print(f"Best by peak Omega_GW: {best['tag']}")


if __name__ == "__main__":
    main()
