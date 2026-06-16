from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time

import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

import highfre322v
import highfre644v
from pure_peters_template import save_pure_peters_frequency_amplitude
from remnant_rate_models import (
    effective_local_rate_gpc3_yr,
    kerr_horizon_frequency_dimensionless,
    remnant_cloud_rate_density_si,
    superradiant_spin_threshold,
)
from transition_geometry import angle_average_factor_from_metadata


BASE_DIR = Path(__file__).resolve().parent
FREQUENCY_DATA_DIR = BASE_DIR / "frequency_data"
FIGURE_DIR = BASE_DIR / "figures"
EXPECTED_HIGHFREQ_ALPHA = 0.30
EXPECTED_HIGHFREQ_BH_SPIN = 0.7
EXPECTED_HIGHFREQ_CLOUD_MASS_FRACTION = 0.005
EXPECTED_ETA_MODEL = "finite_separation_fourier"
PURE_PETERS_F_ORB_INIT_HZ = 5.112
EXPECTED_SPECTRUM_WINDOW_MODE = "first_selected_orbits"
EXPECTED_SPECTRUM_WINDOW_ORBITS = 160.0
PURE_PETERS_WINDOW_ORBITS = EXPECTED_SPECTRUM_WINDOW_ORBITS

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


@dataclass(frozen=True)
class SGWBRateConfig:
    r0_gpc3_yr: float = 1.0
    z_model: float = 10.0
    rate_evolution: str = "constant"
    f_ret: float = 1.0
    f_2g: float = 1.0
    f_cloud: float = 1.0
    f_duty: float = 1.0
    remnant_spin: float = EXPECTED_HIGHFREQ_BH_SPIN
    spin_model: str = "GerosaBerti2017"


def _env_float(name, default):
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return float(default)
    return float(raw_value)


def _env_str(name, default):
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return str(default)
    return raw_value.strip()


def _env_suffix(name="SGWB_OUTPUT_SUFFIX"):
    suffix = os.environ.get(name, "").strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    return suffix


RATE_CONFIG = SGWBRateConfig(
    r0_gpc3_yr=_env_float("SGWB_R0_GPC3_YR", 1.0),
    z_model=_env_float("SGWB_Z_MODEL", 10.0),
    rate_evolution=_env_str("SGWB_RATE_EVOLUTION", "constant"),
    f_ret=_env_float("SGWB_F_RET", 1.0),
    f_2g=_env_float("SGWB_F_2G", 1.0),
    f_cloud=_env_float("SGWB_F_CLOUD", 1.0),
    f_duty=_env_float("SGWB_F_DUTY", 1.0),
    remnant_spin=_env_float("SGWB_REMNANT_SPIN", EXPECTED_HIGHFREQ_BH_SPIN),
    spin_model=_env_str("SGWB_SPIN_MODEL", "GerosaBerti2017"),
)
SGWB_OUTPUT_SUFFIX = _env_suffix()
POWER_SPECTRUM_DIRECTION = "downward"


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

POWER_SPECTRUM_STYLES = {
    "highfre axion+backreaction": {"color": "#0072B2", "linewidth": 1.9},
    "highfre644 axion+backreaction": {"color": "#009E73", "linewidth": 1.9},
    "highfre pure binary template": {"color": "#C44E52", "linewidth": 1.9},
}

POWER_SPECTRUM_LABELS = {
    "highfre axion+backreaction": r"$\left|322\right\rangle\rightarrow\left|300\right\rangle$",
    "highfre644 axion+backreaction": r"$\left|644\right\rangle\rightarrow\left|544\right\rangle$",
    "highfre pure binary template": "pure template",
}

DETECTOR_LABELS = {
    "CE sensitivity": "CE",
    "DECIGO sensitivity": "DECIGO",
    "ET sensitivity": "ET",
    "LISA sensitivity": "LISA",
}


@dataclass(frozen=True)
class ExportTarget:
    label: str
    path: Path
    builder_name: str


EXPORT_TARGETS = (
    ExportTarget(
        label="highfre axion+backreaction",
        path=FREQUENCY_DATA_DIR / "highfre322v_axion_backreaction_total_frequency_amplitude.txt",
        builder_name="highfre",
    ),
    ExportTarget(
        label="highfre644 axion+backreaction",
        path=FREQUENCY_DATA_DIR / "highfre644v_axion_backreaction_total_frequency_amplitude.txt",
        builder_name="highfre644",
    ),
    ExportTarget(
        label="highfre pure binary template",
        path=FREQUENCY_DATA_DIR / "highfre_pure_peters_frequency_amplitude.txt",
        builder_name="pure_peters",
    ),
)


def build_highfre_exports():
    simulator = highfre322v.EccentricResonantTidalGA(
        M_bh=1.0,
        M_star=0.01,
        alpha=EXPECTED_HIGHFREQ_ALPHA,
        bh_spin=EXPECTED_HIGHFREQ_BH_SPIN,
        distance_Mpc=0.001,
        z=0.0,
        e_init=0.65,
        f_orb_init=None,
        resonance_harmonic=4,
        max_harmonic=8,
        multi_harmonic_drive=True,
        harmonics_to_keep=8,
        binary_harmonics=12,
        eta_ref_hz=None,
        Gamma_decay_hz=None,
        transition_family="fine",
        initial_state=(3, 2, 2),
        final_state=(3, 0, 0),
        cloud_mass_fraction=EXPECTED_HIGHFREQ_CLOUD_MASS_FRACTION,
        geom_factor=None,
    )
    duration_yr = simulator.recommended_duration_to_cover_selected_resonance(
        initial_duration_yr=2.0e-4,
        max_duration_yr=5.0e-2,
        post_event_padding_orbits=220.0,
    )
    return simulator.run(
        duration_yr=duration_yr,
        secular_samples=900,
        zoom_orbits=20,
        zoom_points=8192,
        spectrum_orbits=EXPECTED_SPECTRUM_WINDOW_ORBITS,
        spectrum_points=8192,
        spectrum_pad_factor=4,
        spectrum_window_mode=EXPECTED_SPECTRUM_WINDOW_MODE,
    )


def build_highfre644_exports():
    simulator = highfre644v.EccentricResonantTidalGA(
        M_bh=1.0,
        M_star=0.01,
        alpha=EXPECTED_HIGHFREQ_ALPHA,
        bh_spin=EXPECTED_HIGHFREQ_BH_SPIN,
        distance_Mpc=0.001,
        z=0.0,
        e_init=0.65,
        f_orb_init=None,
        resonance_harmonic=1,
        max_harmonic=8,
        multi_harmonic_drive=True,
        harmonics_to_keep=8,
        binary_harmonics=12,
        eta_ref_hz=None,
        Gamma_decay_hz=None,
        transition_family="bohr",
        initial_state=(6, 4, 4),
        final_state=(5, 4, 4),
        tidal_m=0,
        cloud_mass_fraction=EXPECTED_HIGHFREQ_CLOUD_MASS_FRACTION,
        geom_factor=None,
    )
    duration_yr = simulator.recommended_duration_to_cover_selected_resonance(
        initial_duration_yr=2.0e-4,
        max_duration_yr=5.0e-2,
        post_event_padding_orbits=220.0,
    )
    return simulator.run(
        duration_yr=duration_yr,
        secular_samples=900,
        zoom_orbits=20,
        zoom_points=8192,
        spectrum_orbits=EXPECTED_SPECTRUM_WINDOW_ORBITS,
        spectrum_points=8192,
        spectrum_pad_factor=4,
        spectrum_window_mode=EXPECTED_SPECTRUM_WINDOW_MODE,
    )


def build_pure_peters_exports():
    return save_pure_peters_frequency_amplitude(
        FREQUENCY_DATA_DIR / "highfre_pure_peters_frequency_amplitude.txt",
        primary_mass_msun=1.0,
        secondary_mass_msun=0.01,
        eccentricity_init=0.65,
        orbital_frequency_init_hz=PURE_PETERS_F_ORB_INIT_HZ,
        distance_mpc=0.001,
        redshift=0.0,
        window_orbits=PURE_PETERS_WINDOW_ORBITS,
        sample_points=8192,
        pad_factor=4,
    )


BUILDERS = {
    "highfre": build_highfre_exports,
    "highfre644": build_highfre644_exports,
    "pure_peters": build_pure_peters_exports,
}


def peters_export_needs_rebuild(path: Path) -> bool:
    if not path.exists():
        return True
    metadata = parse_metadata(path)
    required_metadata = (
        "sample_points",
        "pad_factor",
        "fft_nyquist_hz",
        "fft_dt_obs_s",
        "fft_df_hz",
        "fft_n_samples",
        "fft_n_fft",
        "fft_window",
        "fft_window_beta",
        "fft_mean_removed",
        "fourier_convention",
        "fft_amplitude_units",
        "frequency_frame",
    )
    if any(key not in metadata for key in required_metadata):
        return True
    try:
        f_orb_init = float(metadata["orbital_frequency_init_hz"])
        window_orbits = float(metadata["window_orbits"])
        primary_mass_msun = float(metadata["primary_mass_msun"])
        secondary_mass_msun = float(metadata["secondary_mass_msun"])
        eccentricity_init = float(metadata["eccentricity_init"])
        redshift = float(metadata["redshift"])
        sample_points = int(metadata["sample_points"])
        pad_factor = int(metadata["pad_factor"])
        window_beta = float(metadata["fft_window_beta"])
    except (KeyError, ValueError):
        return True
    return not (
        metadata.get("template_model") == "peters"
        and metadata.get("frequency_frame") == "observer"
        and metadata.get("fft_window") == "kaiser"
        and metadata.get("fft_mean_removed") == "1"
        and metadata.get("fourier_convention") == "h_tilde_integral_h_exp_minus_2pi_i_f_t_dt"
        and metadata.get("fft_amplitude_units") == "strain_seconds"
        and np.isclose(f_orb_init, PURE_PETERS_F_ORB_INIT_HZ)
        and np.isclose(window_orbits, PURE_PETERS_WINDOW_ORBITS)
        and np.isclose(primary_mass_msun, 1.0)
        and np.isclose(secondary_mass_msun, 0.01)
        and np.isclose(eccentricity_init, 0.65)
        and np.isclose(redshift, 0.0)
        and sample_points == 8192
        and pad_factor == 4
        and np.isclose(window_beta, 14.0)
    )


def export_needs_rebuild(target: ExportTarget) -> bool:
    path = target.path
    if target.builder_name == "pure_peters":
        return peters_export_needs_rebuild(path)
    if not path.exists():
        return True
    metadata = parse_metadata(path)
    try:
        alpha = float(metadata["alpha"])
        bh_spin = float(metadata["bh_spin"])
    except (KeyError, ValueError):
        return True
    if target.builder_name in {"highfre", "highfre644"}:
        if metadata.get("eta_model") != EXPECTED_ETA_MODEL:
            return True
        required_metadata = (
            "cloud_evolution_mode",
            "resonance_harmonic",
            "max_harmonic",
            "multi_harmonic_drive",
            "harmonics_to_keep",
            "active_harmonics",
            "lz_window_widths",
            "backreaction_gate_mode",
            "backreaction_gate_width_factor",
            "fft_df_hz",
            "fft_window",
            "fft_window_beta",
            "fft_mean_removed",
            "fourier_convention",
            "fft_amplitude_units",
        )
        if any(key not in metadata for key in required_metadata):
            return True
        if "fft_nyquist_hz" not in metadata:
            return True
        if metadata.get("spectrum_window_mode") != EXPECTED_SPECTRUM_WINDOW_MODE:
            return True
        try:
            window_orbits = float(metadata["spectrum_window_orbits"])
        except (KeyError, ValueError):
            return True
        if not np.isclose(window_orbits, EXPECTED_SPECTRUM_WINDOW_ORBITS):
            return True
        try:
            transition_frequency_obs_hz = float(metadata["transition_frequency_obs_hz"])
            full_data = np.loadtxt(path, comments="#", usecols=(0,))
        except (KeyError, OSError, ValueError):
            return True
        if full_data.size == 0 or float(np.max(full_data)) < 1.2 * transition_frequency_obs_hz:
            return True
        try:
            primary_mass_msun = float(metadata["primary_mass_msun"])
            secondary_mass_msun = float(metadata["secondary_mass_msun"])
            eccentricity_init = float(metadata["eccentricity_init"])
            cloud_mass_fraction = float(metadata["cloud_mass_fraction"])
        except (KeyError, ValueError):
            return True
        if not (
            np.isclose(primary_mass_msun, 1.0)
            and np.isclose(secondary_mass_msun, 0.01)
            and np.isclose(eccentricity_init, 0.65)
            and np.isclose(cloud_mass_fraction, EXPECTED_HIGHFREQ_CLOUD_MASS_FRACTION)
        ):
            return True
    return not (
        np.isclose(alpha, EXPECTED_HIGHFREQ_ALPHA)
        and np.isclose(bh_spin, EXPECTED_HIGHFREQ_BH_SPIN)
    )


def ensure_frequency_exports():
    missing_by_builder = {}
    for target in EXPORT_TARGETS:
        if export_needs_rebuild(target):
            missing_by_builder.setdefault(target.builder_name, []).append(target.path)

    if not missing_by_builder:
        return

    for builder_name, missing_paths in missing_by_builder.items():
        print(f"Generating frequency-domain exports via {builder_name} ...")
        BUILDERS[builder_name]()
        unresolved = [path for path in missing_paths if not path.exists()]
        if unresolved:
            missing_str = ", ".join(str(path) for path in unresolved)
            raise FileNotFoundError(f"Expected export files were not created: {missing_str}")


def parse_metadata(path: Path):
    metadata = {}
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


def load_frequency_amplitude_export(path: Path):
    metadata = parse_metadata(path)
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return metadata, data


def hubble_e(z):
    z = np.asarray(z, dtype=float)
    return np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)


def characteristic_mass_msun(metadata):
    m1 = float(metadata["primary_mass_msun"])
    m2 = float(metadata["secondary_mass_msun"])
    return 0.5 * (m1 + m2)


def transition_cloud_m(label: str) -> int:
    if "644" in label:
        return 4
    if "322" in label or "highfre axion" in label:
        return 2
    return 0


def remnant_cloud_rate_si(z, config: SGWBRateConfig):
    return remnant_cloud_rate_density_si(
        z,
        r0_gpc3_yr=config.r0_gpc3_yr,
        evolution=config.rate_evolution,
        f_ret=config.f_ret,
        f_2g=config.f_2g,
        f_cloud=config.f_cloud,
        f_duty=config.f_duty,
    )


def sgwb_header_lines(label, metadata, config: SGWBRateConfig, nu_cut, mass_msun, empty=False):
    alpha = float(metadata.get("alpha", EXPECTED_HIGHFREQ_ALPHA))
    cloud_m = transition_cloud_m(label)
    threshold = superradiant_spin_threshold(alpha, cloud_m) if cloud_m > 0 else np.nan
    r_eff0 = effective_local_rate_gpc3_yr(
        r0_gpc3_yr=config.r0_gpc3_yr,
        f_ret=config.f_ret,
        f_2g=config.f_2g,
        f_cloud=config.f_cloud,
        f_duty=config.f_duty,
    )
    lines = [
        f"label={label}",
        f"source_module={metadata.get('module', 'unknown')}",
        f"source_component={metadata.get('component', 'unknown')}",
        f"source_transition_family={metadata.get('transition_family', 'none')}",
        f"source_frequency_frame={metadata.get('frequency_frame', 'unknown')}",
        f"source_redshift={float(metadata.get('redshift', '0.0')):.16e}",
        f"source_luminosity_distance_m={float(metadata.get('luminosity_distance_m', 'nan')):.16e}",
        "rate_model=remnant_cloud_effective",
        f"r0_gpc3_yr={config.r0_gpc3_yr:.16e}",
        f"r_eff0_gpc3_yr={r_eff0:.16e}",
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
        f"nu_cut_hz={nu_cut:.16e}",
        f"characteristic_mass_msun={mass_msun:.16e}",
        f"hubble_h0_si={H0:.16e}",
        f"rho_c_energy_density_si={RHO_C:.16e}",
        "omega_gw_formula=f_over_rho_c_H0_integral_R_eff_dE_dnu_over_1plusz_Ez",
        "single_event_spectrum_frame=local_source_z0",
        "energy_spectrum_prefactor=two_pi_squared_c_cubed_over_G",
        f"source_angle_average_factor={angle_average_factor_from_metadata(metadata):.16e}",
        "single_event_fourier_convention=h_tilde_integral_h_exp_minus_2pi_i_f_t_dt",
    ]
    if empty:
        lines.append("empty_source_spectrum=true")
    lines.append("columns=frequency_hz omega_gw")
    return lines


def build_energy_spectrum(metadata, data):
    freq_obs_hz = np.asarray(data[:, 0], dtype=float)
    amplitude_obs = np.asarray(data[:, 1], dtype=float)
    redshift = float(metadata.get("redshift", "0.0"))
    if metadata.get("frequency_frame") != "observer":
        raise ValueError("Frequency export must use observer-frame frequencies.")
    if not np.isclose(redshift, 0.0, atol=1.0e-14):
        raise ValueError(
            "SGWB energy-spectrum builder expects a local z=0 waveform export; "
            "cosmological redshift is applied only in the population integral."
        )
    distance_m = float(metadata["luminosity_distance_m"])

    source_freq_hz = freq_obs_hz
    source_amplitude = amplitude_obs
    angle_average_factor = angle_average_factor_from_metadata(metadata)
    dE_dnu = (
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


def build_log_interpolator(x_values, y_values):
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


def make_redshift_grid(z_upper, points=512):
    if z_upper <= 0.0:
        return np.array([0.0], dtype=float)
    if z_upper < 1.0e-6:
        return np.linspace(0.0, z_upper, 32)
    return np.expm1(np.linspace(0.0, np.log1p(z_upper), points))


def compute_omega_gw_curve(source_freq_hz, dE_dnu, metadata, label, config: SGWBRateConfig):
    if len(source_freq_hz) == 0 or len(dE_dnu) == 0:
        freq_obs_hz = np.logspace(-4.0, 5.0, 500)
        omega_gw = np.zeros_like(freq_obs_hz)
        mass_msun = characteristic_mass_msun(metadata)
        FREQUENCY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        curve_path = FREQUENCY_DATA_DIR / (
            f"{label.replace(' ', '_').replace('+', 'plus')}{SGWB_OUTPUT_SUFFIX}_{POWER_SPECTRUM_DIRECTION}_omega_gw.txt"
        )
        header = "\n".join(sgwb_header_lines(label, metadata, config, 0.0, mass_msun, empty=True))
        np.savetxt(curve_path, np.column_stack((freq_obs_hz, omega_gw)), header=header, comments="# ")
        print(f"Saved zero SGWB curve data: {curve_path}")
        return freq_obs_hz, omega_gw

    interpolate_dE, nu_min, nu_cut = build_log_interpolator(source_freq_hz, dE_dnu)
    freq_min_obs = max(nu_min / (1.0 + config.z_model), 1.0e-8)
    freq_max_obs = nu_cut
    freq_obs_hz = np.logspace(np.log10(freq_min_obs), np.log10(freq_max_obs), 500)
    omega_gw = np.zeros_like(freq_obs_hz)
    mass_msun = characteristic_mass_msun(metadata)

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

    FREQUENCY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    curve_path = FREQUENCY_DATA_DIR / (
        f"{label.replace(' ', '_').replace('+', 'plus')}{SGWB_OUTPUT_SUFFIX}_{POWER_SPECTRUM_DIRECTION}_omega_gw.txt"
    )
    header = "\n".join(sgwb_header_lines(label, metadata, config, nu_cut, mass_msun))
    np.savetxt(curve_path, np.column_stack((freq_obs_hz, omega_gw)), header=header, comments="# ")
    print(f"Saved SGWB curve data: {curve_path}")
    return freq_obs_hz, omega_gw


def load_sensitivity_curve(path: Path):
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]

    freq_hz = np.asarray(data[:, 0], dtype=float)
    sensitivity = np.asarray(data[:, 1], dtype=float)
    valid = (
        np.isfinite(freq_hz)
        & np.isfinite(sensitivity)
        & (freq_hz > 0.0)
        & (sensitivity > 0.0)
    )
    freq_hz = freq_hz[valid]
    sensitivity = sensitivity[valid]

    order = np.argsort(freq_hz)
    freq_hz = freq_hz[order]
    sensitivity = sensitivity[order]

    unique_freq = []
    envelope_sensitivity = []
    for freq_value in np.unique(freq_hz):
        mask = np.isclose(freq_hz, freq_value, rtol=0.0, atol=1.0e-15 * max(freq_value, 1.0))
        unique_freq.append(freq_value)
        envelope_sensitivity.append(np.min(sensitivity[mask]))

    return np.asarray(unique_freq, dtype=float), np.asarray(envelope_sensitivity, dtype=float)


def smooth_sensitivity_curve(freq_hz, sensitivity, sample_points=1200):
    if freq_hz.size < 2:
        return freq_hz, sensitivity

    log_freq = np.log10(freq_hz)
    log_sensitivity = np.log10(sensitivity)
    interpolator = PchipInterpolator(log_freq, log_sensitivity, extrapolate=False)
    dense_log_freq = np.linspace(log_freq.min(), log_freq.max(), int(sample_points))
    dense_log_sensitivity = interpolator(dense_log_freq)
    return 10.0 ** dense_log_freq, 10.0 ** dense_log_sensitivity


def load_all_sensitivity_curves():
    curves = []
    for detector_name, path in SENSITIVITY_FILES.items():
        freq_hz, sensitivity = load_sensitivity_curve(path)
        smooth_freq_hz, smooth_sensitivity = smooth_sensitivity_curve(freq_hz, sensitivity)
        output_path = FREQUENCY_DATA_DIR / f"{detector_name.lower()}_sensitivity_smoothed.txt"
        FREQUENCY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            output_path,
            np.column_stack((smooth_freq_hz, smooth_sensitivity)),
            header="columns=frequency_hz sensitivity\nsource_file=" + str(path.name),
            comments="# ",
        )
        print(f"Saved smoothed sensitivity curve: {output_path}")
        curves.append(
            {
                "label": f"{detector_name} sensitivity",
                "frequency_hz": smooth_freq_hz,
                "sensitivity": smooth_sensitivity,
                "style": SENSITIVITY_STYLES.get(detector_name, {}),
            }
        )
    return curves


def plot_omega_curves(curves, sensitivity_curves):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cm_to_inch = 1.0 / 2.54
    fig, ax = plt.subplots(figsize=(8.0 * cm_to_inch, 9.0 * cm_to_inch))
    # Reserve more room for the stacked figure-level legends in a narrow two-column layout.
    fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.35)
    power_handles = []
    sensitivity_handles = []

    for curve in curves:
        positive = curve["omega_gw"] > 0.0
        line, = ax.plot(
            curve["frequency_hz"][positive],
            curve["omega_gw"][positive],
            **POWER_SPECTRUM_STYLES.get(curve["label"], {"linewidth": 2.8}),
            label=POWER_SPECTRUM_LABELS.get(curve["label"], curve["label"]),
        )
        power_handles.append(line)

    for sensitivity_curve in sensitivity_curves:
        line, = ax.plot(
            sensitivity_curve["frequency_hz"],
            sensitivity_curve["sensitivity"],
            label=DETECTOR_LABELS.get(sensitivity_curve["label"], sensitivity_curve["label"]),
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
        title="Power Spectra",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        ncol=2,
        frameon=False,
        fontsize=7.3,
        title_fontsize=7.6,
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

    pdf_path = FIGURE_DIR / f"sgwb_power_spectra_with_sensitivity{SGWB_OUTPUT_SUFFIX}_downward.pdf"
    try:
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot: {pdf_path}")
    except PermissionError:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fallback_path = FREQUENCY_DATA_DIR / f"{pdf_path.stem}_{timestamp}{pdf_path.suffix}"
        fig.savefig(fallback_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot: {fallback_path}")
    plt.close(fig)


def main():
    ensure_frequency_exports()

    curves = []
    for target in EXPORT_TARGETS:
        metadata, data = load_frequency_amplitude_export(target.path)
        source_freq_hz, dE_dnu = build_energy_spectrum(metadata, data)
        freq_obs_hz, omega_gw = compute_omega_gw_curve(
            source_freq_hz=source_freq_hz,
            dE_dnu=dE_dnu,
            metadata=metadata,
            label=target.label,
            config=RATE_CONFIG,
        )
        curves.append(
            {
                "label": target.label,
                "frequency_hz": freq_obs_hz,
                "omega_gw": omega_gw,
            }
        )

    sensitivity_curves = load_all_sensitivity_curves()
    plot_omega_curves(curves, sensitivity_curves)


if __name__ == "__main__":
    main()
