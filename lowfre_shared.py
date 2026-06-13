import time
import os
from pathlib import Path
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.signal.windows import kaiser, tukey
from scipy.special import eval_genlaguerre, jv, sph_harm_y
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from transition_geometry import compute_transition_geometry

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

def spherical_harmonic(m, l, phi, theta):
    return sph_harm_y(l, m, theta, phi)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DISTANCE_KPC = 100.0
DEFAULT_DISTANCE_MPC = DEFAULT_DISTANCE_KPC / 1.0e3
DEFAULT_LOWFREQ_CLOUD_MASS_PROFILE = "saturated"
LOWFREQ_CLOUD_MASS_PROFILE_FRACTIONS = {
    "saturated": 5.0e-2,
    "paper": 5.0e-2,
    "upper": 5.0e-2,
    "effective": 3.0e-4,
    "conservative": 3.0e-4,
}
DEFAULT_LOWFREQ_CLOUD_MASS_FRACTION = LOWFREQ_CLOUD_MASS_PROFILE_FRACTIONS[
    DEFAULT_LOWFREQ_CLOUD_MASS_PROFILE
]
DECIGO_UNIT_COUNT = 4.0


def resolve_default_lowfreq_cloud_mass_fraction():
    override = os.environ.get("LOWFREQ_CLOUD_MASS_FRACTION")
    if override is not None:
        return float(override)
    profile = os.environ.get("LOWFREQ_CLOUD_MASS_PROFILE", DEFAULT_LOWFREQ_CLOUD_MASS_PROFILE)
    profile = str(profile).strip().lower()
    if profile not in LOWFREQ_CLOUD_MASS_PROFILE_FRACTIONS:
        valid = ", ".join(sorted(LOWFREQ_CLOUD_MASS_PROFILE_FRACTIONS))
        raise ValueError(f"Unknown LOWFREQ_CLOUD_MASS_PROFILE={profile!r}; choose one of {valid}.")
    return float(LOWFREQ_CLOUD_MASS_PROFILE_FRACTIONS[profile])

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "figure.figsize": (12, 11),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
    }
)


def solve_kepler(mean_anomaly, eccentricity, max_iter=15, tol=1e-12):
    """Solve M = E - e sin(E) for scalar or vector mean anomaly."""
    mean_anomaly = np.asarray(mean_anomaly, dtype=float)
    e = np.asarray(eccentricity, dtype=float)
    if e.ndim == 0:
        e = np.full_like(mean_anomaly, float(np.clip(e, 0.0, 0.999)))
    else:
        e = np.broadcast_to(np.clip(e, 0.0, 0.999), mean_anomaly.shape).astype(float)

    flat_m = np.ascontiguousarray(mean_anomaly.reshape(-1))
    flat_e = np.ascontiguousarray(e.reshape(-1))
    guess = _solve_kepler_numba(flat_m, flat_e, max_iter, tol)
    return guess.reshape(mean_anomaly.shape)


@njit(cache=True)
def _solve_kepler_numba(mean_anomaly, eccentricity, max_iter, tol):
    guess = np.empty_like(mean_anomaly)
    for i in range(mean_anomaly.size):
        guess[i] = mean_anomaly[i] if eccentricity[i] < 0.8 else np.pi

    for _ in range(max_iter):
        max_step = 0.0
        for i in range(mean_anomaly.size):
            residual = guess[i] - eccentricity[i] * math.sin(guess[i]) - mean_anomaly[i]
            jacobian = 1.0 - eccentricity[i] * math.cos(guess[i])
            step = residual / jacobian
            guess[i] -= step
            abs_step = abs(step)
            if abs_step > max_step:
                max_step = abs_step
        if max_step < tol:
            break
    return guess


class _SolutionSlice:
    """Expose selected components of a dense_output solution with the same `.sol()` API."""

    def __init__(self, full_solution, component_slice):
        self._full_solution = full_solution
        self._component_slice = component_slice

    def sol(self, t_eval):
        return self._full_solution.sol(t_eval)[self._component_slice]


@dataclass(frozen=True)
class LowFrequencyRunProfile:
    source_kwargs: dict
    run_kwargs: dict
    detector_names: tuple[str, ...]


def build_paper_inspired_lowfreq_profile():
    """
    Full low-frequency defaults adapted to the current DECIGO-focused pipeline.

    We keep alpha and M_bh to be supplied by the caller, while aligning the remaining
    physical knobs with the intended low-frequency IMRI setup:
    M_star = 0.5 Msun by default, e_in ~= 0.3.

    By default this profile keeps a saturated-cloud benchmark.  Saturated
    superradiant clouds can reach percent-to-ten-percent fractions of the
    black-hole mass, and the resulting strong orbital feedback is part of the
    low-frequency floating/early-resonance physics rather than automatically a
    numerical pathology.  Use LOWFREQ_CLOUD_MASS_PROFILE=effective or an explicit
    LOWFREQ_CLOUD_MASS_FRACTION override for conservative control runs.
    """
    default_cloud_mass_fraction = resolve_default_lowfreq_cloud_mass_fraction()

    source_kwargs = {
        "distance_Mpc": DEFAULT_DISTANCE_MPC,
        "z": 0.022,
        "e_init": 0.3,
        "f_orb_init": None,
        "resonance_harmonic": 4,
        "max_harmonic": 8,
        "multi_harmonic_drive": False,
        "harmonics_to_keep": 8,
        "binary_harmonics": 12,
        "transition_frequency_hz": None,
        "eta_ref_hz": None,
        "Gamma_decay_hz": None,
        "orbital_start_ratio": 0.999998,
        "target_resonance_delay_obs_days": 96.0,
        "cloud_initial_state": "bare",
        "cloud_mass_fraction": default_cloud_mass_fraction,
        "geom_factor": None,
        "orbital_backreaction_mode": "selected_rwa",
        "cloud_evolution_mode": "band_gated",
        "solver_profile": "accurate",
        "detector_names": ("DECIGO",),
        "detector_curve_kinds": {"DECIGO": "characteristic_strain"},
        "hansen_e_samples": 48,
        "hansen_M_samples": 1024,
        "overlap_grid_points": 4096,
    }
    run_kwargs = {
        "duration_yr": 0.4,
        "secular_samples": 1440,
        "zoom_orbits": 20,
        "zoom_points": 4096,
        "spectrum_points": 4096,
        "spectrum_pad_factor": 4,
        "mismatch_max_years": 0.4,
        "mismatch_time_samples": 12,
    }
    return LowFrequencyRunProfile(
        source_kwargs=source_kwargs,
        run_kwargs=run_kwargs,
        detector_names=tuple(source_kwargs["detector_names"]),
    )


def build_default_lowfreq_simulator(
    simulator_cls,
    *,
    M_bh=1500.0,
    M_star=0.5,
    alpha=0.25,
    mass_ratio=None,
):
    profile = build_paper_inspired_lowfreq_profile()
    source_kwargs = dict(profile.source_kwargs)
    companion_mass = float(M_star) if mass_ratio is None else float(mass_ratio) * float(M_bh)
    source_kwargs.update(
        {
            "M_bh": float(M_bh),
            "M_star": companion_mass,
            "alpha": float(alpha),
        }
    )
    return simulator_cls(**source_kwargs), profile


def apply_default_lowfreq_source_kwargs(kwargs):
    profile = build_paper_inspired_lowfreq_profile()
    for key, value in profile.source_kwargs.items():
        kwargs.setdefault(key, value)
    kwargs.setdefault("M_bh", 1500.0)
    kwargs.setdefault("M_star", 0.5)
    kwargs.setdefault("alpha", 0.25)
    return kwargs


def run_default_lowfreq_entry(
    simulator_cls,
    *,
    transition_description,
    M_bh=1500.0,
    M_star=0.5,
    alpha=0.25,
    mass_ratio=None,
):
    start = time.time()
    simulator, profile = build_default_lowfreq_simulator(
        simulator_cls,
        M_bh=M_bh,
        M_star=M_star,
        alpha=alpha,
        mass_ratio=mass_ratio,
    )
    print(f"Building {Path(simulator._module_stem).stem}: low-frequency eccentric inspiral + harmonic comb model...")
    print(f"Using {transition_description}.")
    print(
        "Run mode: full numerical; "
        f"detectors={','.join(profile.detector_names)}, solver={simulator.solver_profile}, "
        f"q={simulator.M_star / simulator.M:.3g}, e_init={simulator.e_init:.3f}, "
        f"Mc/M={simulator.cloud_mass_fraction:.3g}, "
        f"d_L={simulator.d_L / simulator.Mpc * 1.0e3:.1f} kpc, "
        f"target_resonance_delay={simulator.initial_frequency_setup.get('target_delay_obs_days', np.nan):.1f} obs days, "
        f"cloud_initial_state={simulator.cloud_initial_state}, "
        f"duration_yr={profile.run_kwargs['duration_yr']:.2f}"
    )
    results = simulator.run(**profile.run_kwargs)
    elapsed = time.time() - start
    simulator.print_summary(results, elapsed)
    simulator.plot_summary(results)
    simulator.plot_detector_comparison(results)


class EccentricResonantTidalGA:
    """
    Eccentric inspiral + selective tidal harmonic excitation + dissipative two-level dynamics.

    The orbital sector follows the Peters equations for a(t), e(t), and the mean anomaly Phi(t).
    The cloud sector follows the n-th harmonic RWA for a user-selected resonance harmonic.
    At the same time, the full harmonic comb is computed and exposed for diagnostics.
    """

    DETECTOR_BANDS_HZ = {
        "DECIGO": (2.0e-3, 1.0e1),
        "ET": (1.0e-3, 5.0e3),
        "CE": (1.0e-3, 5.0e3),
    }
    _MIXING_OVERLAP_CACHE = {}
    _HANSEN_TABLE_CACHE = {}
    _DETECTOR_NOISE_CACHE = {}
    _CACHE_LOCK = threading.RLock()

    def __init__(
        self,
        # --- 源参数：黑洞质量、伴星质量、云参数 ---
        M_bh=150.0,
        M_star=1.0,
        alpha=0.3,
        distance_Mpc=DEFAULT_DISTANCE_MPC,
        z=0.022,
        e_init=0.65,
        # 若设为 None，则用能级差 DeltaE / n_res 自动反推共振附近的初始轨道频率
        f_orb_init=None,
        resonance_harmonic=4,
        max_harmonic=30,
        multi_harmonic_drive=True,
        harmonics_to_keep=None,
        binary_harmonics=12,
        # 若设为 None，则由参数求解器公式驱动得到 DeltaE
        transition_frequency_hz=None,
        # 若设为 None，则 eta/Gamma 会分别使用潮汐矩阵元公式和末态衰减公式
        eta_ref_hz=None,
        Gamma_decay_hz=None,
        hubble_constant_km_s_mpc=67.66,
        detector_curve_kinds=None,
        # 跃迁类型：默认先用 fine transition，更适合当前参数求解器公式
        transition_family="fine",
        # 如果想手动指定两能级，可直接传 (n,l,m)
        initial_state=None,
        final_state=None,
        bh_spin=0.99,
        target_resonance_delay_obs_days=None,
        # 用于把初始轨道放在共振点下方一点，保证会扫过共振
        orbital_start_ratio=0.999998,
        # True: eta 用论文 A.6 数值版；False: 回退到旧的经验缩放
        use_formula_eta=True,
        overlap_grid_points=4096,
        overlap_max_x=5.0e3,
        # 潮汐多极矩默认取 l*=2, m*=2
        tidal_l=2,
        tidal_m=None,
        cloud_mass_fraction=DEFAULT_LOWFREQ_CLOUD_MASS_FRACTION,
        geom_factor=None,
        include_orbital_backreaction=True,
        orbital_backreaction_mode="selected_rwa",
        cloud_evolution_mode="band_gated",
        resonance_band_width_factor=8.0,
        floating_band_fraction_threshold=0.05,
        floating_detuning_drift_threshold=2.0,
        cloud_initial_state="bare",
        orbit_orientation_sign=1,
        solver_profile="accurate",
        solver_method=None,
        solver_rtol=None,
        hansen_e_samples=48,
        hansen_M_samples=1024,
        parallel_workers=None,
        mismatch_threshold_d=13.0,
        save_figure_dir="figures",
        save_figure_formats=("pdf",),
        save_time_series_data_dir="waveform_data",
        save_mismatch_data_dir="mismatch_data",
        detector_names=("DECIGO",),
        match_band_orbital_half_width=6.0,
        match_band_strategy="detector",
        module_stem=None,
        direction_tag="upward",
        degenerate_bundle_merge_factor=1.25,
    ):
        self.G = 6.6743e-11
        self.c = 2.99792458e8
        self.M_sun = 1.98847e30
        self.Mpc = 3.085677581491367e22
        self.AU = 1.495978707e11
        self.yr = 365.25 * 24.0 * 3600.0
        self.hbar = 1.054571817e-34
        self.eV = 1.602176634e-19
        self.H0 = float(hubble_constant_km_s_mpc) * 1000.0 / self.Mpc

        self.M = M_bh * self.M_sun
        self.M_star = M_star * self.M_sun
        self.M_tot = self.M + self.M_star
        self.M_chirp = (self.M * self.M_star) ** (3.0 / 5.0) / (self.M_tot ** (1.0 / 5.0))

        self.alpha = float(alpha)
        self.d_L = distance_Mpc * self.Mpc
        self.z = float(z)

        self.transition_family = str(transition_family).lower()
        self.initial_state = initial_state
        self.final_state = final_state
        self._manual_transition_frequency_hz = (
            None if transition_frequency_hz is None else float(transition_frequency_hz)
        )
        self.resonance_harmonic = int(resonance_harmonic)
        self.bh_spin = float(bh_spin)
        self.orbital_start_ratio = float(orbital_start_ratio)
        self.target_resonance_delay_obs_days = (
            None if target_resonance_delay_obs_days is None else float(target_resonance_delay_obs_days)
        )
        self.use_formula_eta = bool(use_formula_eta)
        self.overlap_grid_points = int(overlap_grid_points)
        self.overlap_max_x = float(overlap_max_x)
        self.transition_solver_data = self._compute_transition_quantities()
        requested_transition_frequency_hz = float(
            transition_frequency_hz
            if transition_frequency_hz is not None
            else self.transition_solver_data["transition_frequency_hz"]
        )

        self.e_init = float(e_init)
        if f_orb_init is None:
            # Start below the selected resonance unless a manual frequency is supplied.
            self.f_orb_init = self._initial_orbital_frequency_for_transition(
                requested_transition_frequency_hz,
                self.resonance_harmonic,
                self.M,
                self.M_star,
                self.e_init,
            )
        else:
            self.f_orb_init = float(f_orb_init)
            self.initial_frequency_setup = {
                "mode": "manual_f_orb_init",
                "target_delay_obs_days": np.nan,
                "actual_delay_obs_days": np.nan,
                "frequency_ratio_to_selected_resonance": float(
                    self.f_orb_init
                    / max(requested_transition_frequency_hz / self.resonance_harmonic, 1.0e-300)
                ),
                "status": "manual",
            }
        self.Omega_init = 2.0 * np.pi * self.f_orb_init
        self.a_init = (self.G * self.M_tot / self.Omega_init**2) ** (1.0 / 3.0)
        self.period_init = 1.0 / self.f_orb_init

        self.harmonics = np.arange(1, int(max_harmonic) + 1, dtype=int)
        if self.resonance_harmonic not in self.harmonics:
            raise ValueError("resonance_harmonic must lie inside [1, max_harmonic].")
        self.harmonic_to_index = {n: idx for idx, n in enumerate(self.harmonics)}
        self.multi_harmonic_drive = bool(multi_harmonic_drive)
        self.harmonics_to_keep = len(self.harmonics) if harmonics_to_keep is None else int(harmonics_to_keep)
        self.binary_harmonics = int(binary_harmonics)

        self.transition_frequency_hz = requested_transition_frequency_hz
        self.transition_omega = 2.0 * np.pi * self.transition_frequency_hz
        self.transition_energy_sign = float(self.transition_solver_data["transition_energy_sign"])
        self.transition_energy_change_omega = self.transition_energy_sign * self.transition_omega
        auto_gamma_hz = self.transition_solver_data["gamma_decay_hz"]
        self.eta_ref_hz = None if eta_ref_hz is None else float(eta_ref_hz)
        self.Gamma_decay_hz = float(Gamma_decay_hz if Gamma_decay_hz is not None else auto_gamma_hz)
        self.eta_ref = None
        self.Gamma_decay = 2.0 * np.pi * self.Gamma_decay_hz

        self.tidal_l = int(tidal_l)
        self.radial_power = self.tidal_l + 1

        self.r_g = self.G * self.M / self.c**2
        self.r_c = self.r_g / (self.alpha**2)
        self.cloud_mass_fraction = cloud_mass_fraction
        self.Mc_max = cloud_mass_fraction * self.M
        self.geom_factor = None if geom_factor is None else float(geom_factor)
        self.include_orbital_backreaction = bool(include_orbital_backreaction)
        self.cloud_initial_state = str(cloud_initial_state).lower()
        if self.cloud_initial_state not in {"bare", "adiabatic"}:
            raise ValueError("cloud_initial_state must be 'bare' or 'adiabatic'.")
        self.orbit_orientation_sign = 1.0 if float(orbit_orientation_sign) >= 0.0 else -1.0
        self.solver_profile = str(solver_profile).lower()
        if self.solver_profile not in {"accurate", "fast"}:
            raise ValueError("solver_profile must be 'accurate' or 'fast'.")
        self.solver_method = solver_method or ("RK45" if self.solver_profile == "fast" else "DOP853")
        self.solver_rtol = float(solver_rtol if solver_rtol is not None else (1.0e-9 if self.solver_profile == "fast" else 1.0e-12))
        self.orbit_atol = [1.0e-6, 1.0e-11, 1.0e-8] if self.solver_profile == "fast" else [1.0e-6, 1.0e-12, 1.0e-9]
        self.coupled_atol = (
            [1.0e-6, 1.0e-11, 1.0e-8, 1.0e-10, 1.0e-10, 1.0e-12, 1.0e-12]
            if self.solver_profile == "fast"
            else [1.0e-6, 1.0e-12, 1.0e-9, 1.0e-12, 1.0e-12, 1.0e-14, 1.0e-14]
        )
        self.delta_m_transition = float(
            self.transition_solver_data["final_state"][2] - self.transition_solver_data["initial_state"][2]
        )
        self.tidal_m = int(self.delta_m_transition if tidal_m is None else tidal_m)
        # The harmonic comb below is stored for positive orbital overtones n=1..N.
        # For downward transitions the resonant tidal component has negative m, whose
        # positive-frequency coefficient is the conjugate of the +|m| component.  Using
        # |m| here keeps upward/downward matrix-element magnitudes symmetric while the
        # signed delta_m_transition still controls the angular-momentum backreaction.
        self.hansen_tidal_m = abs(self.tidal_m)
        self.orbital_backreaction_mode = str(orbital_backreaction_mode).lower()
        if self.orbital_backreaction_mode not in {"selected_rwa", "coherent_drive"}:
            raise ValueError("orbital_backreaction_mode must be 'selected_rwa' or 'coherent_drive'.")
        self.cloud_evolution_mode = str(cloud_evolution_mode or "band_gated").lower()
        if self.cloud_evolution_mode not in {"auto", "full", "floating_full", "band_gated"}:
            raise ValueError("cloud_evolution_mode must be 'auto', 'full', 'floating_full', or 'band_gated'.")
        self.resonance_band_width_factor = float(resonance_band_width_factor)
        self.floating_band_fraction_threshold = float(floating_band_fraction_threshold)
        self.floating_detuning_drift_threshold = float(floating_detuning_drift_threshold)
        self._runtime_cloud_evolution_mode = self.cloud_evolution_mode
        self._runtime_resonance_gate_times = None
        self._runtime_resonance_gate_values = None
        self._runtime_reference_band_diagnostics = {}
        self.backreaction_macro_scale = (self.G * self.M * self.Mc_max) / (self.alpha * self.c)
        self.delta_E_orbit_backreaction = (
            self.backreaction_macro_scale * self.transition_energy_change_omega
        )
        self.delta_L_orbit_backreaction = self.backreaction_macro_scale * self.delta_m_transition
        self.delta_E_high_low_backreaction = self.backreaction_macro_scale * self.transition_omega
        self.delta_m_high_low = self.transition_energy_sign * self.delta_m_transition

        self.transition_geometry = self._compute_transition_geometry()
        if self.geom_factor is None:
            self.geom_factor = self.transition_geometry.waveform_geom_factor

        self.hansen_e_samples = int(hansen_e_samples)
        self.hansen_M_samples = int(hansen_M_samples)
        self._hansen_e_grid = None
        self._hansen_real = None
        self._hansen_imag = None

        self._binary_coeff_norm = float(self._binary_strain_coeff(2, np.array([0.0]))[0])
        requested_detector_names = tuple(str(name).upper() for name in detector_names)
        if not requested_detector_names:
            raise ValueError("detector_names must contain at least one detector.")
        invalid_detector_names = tuple(
            name for name in requested_detector_names if name not in self.DETECTOR_BANDS_HZ
        )
        if invalid_detector_names:
            raise ValueError("Unknown detector names: " + ", ".join(invalid_detector_names))
        self.detector_names = requested_detector_names
        cpu_count = os.cpu_count() or 1
        self.parallel_workers = max(1, int(parallel_workers if parallel_workers is not None else min(8, cpu_count)))
        self.mismatch_threshold_d = float(mismatch_threshold_d)
        default_curve_kinds = {"DECIGO": "characteristic_strain"}
        if detector_curve_kinds is not None:
            default_curve_kinds.update(
                {str(key).upper(): str(value).lower() for key, value in detector_curve_kinds.items()}
            )
        self.detector_curve_kinds = {
            str(key).upper(): str(value).lower() for key, value in default_curve_kinds.items()
        }
        self.figure_dir = SCRIPT_DIR / save_figure_dir
        self.figure_formats = tuple(fmt for fmt in save_figure_formats if str(fmt).lower() == "pdf") or ("pdf",)
        self.time_series_data_dir = (
            None if save_time_series_data_dir is None else SCRIPT_DIR / save_time_series_data_dir
        )
        self.mismatch_data_dir = (
            None if save_mismatch_data_dir is None else SCRIPT_DIR / save_mismatch_data_dir
        )
        self._module_stem = Path(__file__).stem if module_stem is None else str(module_stem)
        self._direction_tag = str(direction_tag)
        self.adiabatic_z_threshold = 1.0
        self.match_band_orbital_half_width = float(match_band_orbital_half_width)
        self.match_band_strategy = str(match_band_strategy).lower()
        if self.match_band_strategy not in {"detector", "feature"}:
            raise ValueError("match_band_strategy must be 'detector' or 'feature'.")
        self.degenerate_bundle_merge_factor = float(max(degenerate_bundle_merge_factor, 0.0))
        # 预先算好态间的角向/径向重叠，后面每次算 eta(a,e) 时直接插值
        self.mixing_overlap_data = self._get_cached_mixing_overlaps()
        if self.eta_ref_hz is None:
            self.eta_ref_hz = self._formula_eta_hz(self.a_init)
        self.eta_ref = 2.0 * np.pi * self.eta_ref_hz

    def _peters_delay_to_selected_resonance_source_s(
        self,
        f_orb_init,
        transition_frequency_hz,
        resonance_harmonic,
        primary_mass_kg,
        companion_mass_kg,
        eccentricity_initial,
    ):
        f_res = float(transition_frequency_hz) / max(float(resonance_harmonic), 1.0e-300)
        f_init = float(f_orb_init)
        if not np.isfinite(f_init) or not np.isfinite(f_res) or f_init <= 0.0 or f_res <= 0.0:
            return np.nan
        if f_init >= f_res:
            return 0.0

        total_mass = float(primary_mass_kg) + float(companion_mass_kg)
        omega_init = 2.0 * np.pi * f_init
        omega_res = 2.0 * np.pi * f_res
        a_init = (self.G * total_mass / omega_init**2) ** (1.0 / 3.0)
        a_res = (self.G * total_mass / omega_res**2) ** (1.0 / 3.0)
        c0_value = self._peters_c0(a_init, eccentricity_initial)
        e_res = self._solve_eccentricity_at_resonance(c0_value, a_res, eccentricity_initial)
        if not np.isfinite(e_res):
            return np.nan
        return self._integrate_peters_time_to_e(
            eccentricity_initial,
            e_res,
            c0_value,
            primary_mass_kg,
            companion_mass_kg,
        )

    def _solve_initial_frequency_for_resonance_delay(
        self,
        transition_frequency_hz,
        resonance_harmonic,
        primary_mass_kg,
        companion_mass_kg,
        eccentricity_initial,
        target_delay_source_s,
    ):
        f_res = float(transition_frequency_hz) / max(float(resonance_harmonic), 1.0e-300)
        if not np.isfinite(f_res) or f_res <= 0.0:
            return np.nan, {"status": "invalid_resonance_frequency", "actual_delay_source_s": np.nan}

        target_delay = max(float(target_delay_source_s), 0.0)
        if target_delay <= 0.0:
            f_init = self.orbital_start_ratio * f_res
            return f_init, {"status": "zero_target_fallback_ratio", "actual_delay_source_s": 0.0}

        def delay_for_ratio(ratio):
            return self._peters_delay_to_selected_resonance_source_s(
                ratio * f_res,
                transition_frequency_hz,
                resonance_harmonic,
                primary_mass_kg,
                companion_mass_kg,
                eccentricity_initial,
            )

        hi_ratio = min(max(self.orbital_start_ratio, 1.0e-8), 1.0 - 1.0e-10)
        hi_delay = delay_for_ratio(hi_ratio)
        if not np.isfinite(hi_delay) or hi_delay > target_delay:
            hi_ratio = 1.0 - 1.0e-10
            hi_delay = delay_for_ratio(hi_ratio)

        lo_ratio = 0.9
        lo_delay = delay_for_ratio(lo_ratio)
        while np.isfinite(lo_delay) and lo_delay < target_delay and lo_ratio > 1.0e-5:
            lo_ratio *= 0.5
            lo_delay = delay_for_ratio(lo_ratio)

        if not (np.isfinite(lo_delay) and np.isfinite(hi_delay) and lo_delay >= target_delay >= hi_delay):
            f_init = self.orbital_start_ratio * f_res
            actual_delay = delay_for_ratio(self.orbital_start_ratio)
            return f_init, {
                "status": "target_delay_bracket_failed_fallback_ratio",
                "actual_delay_source_s": actual_delay,
            }

        def residual(ratio):
            return delay_for_ratio(ratio) - target_delay

        ratio_solution = float(brentq(residual, lo_ratio, hi_ratio, xtol=1.0e-12, rtol=1.0e-12, maxiter=100))
        actual_delay = delay_for_ratio(ratio_solution)
        return ratio_solution * f_res, {
            "status": "target_delay_solved",
            "actual_delay_source_s": actual_delay,
        }

    def _initial_orbital_frequency_for_transition(
        self,
        transition_frequency_hz,
        resonance_harmonic,
        primary_mass_kg,
        companion_mass_kg,
        eccentricity_initial,
    ):
        f_res = float(transition_frequency_hz) / max(float(resonance_harmonic), 1.0e-300)
        if self.target_resonance_delay_obs_days is None:
            f_init = self.orbital_start_ratio * f_res
            delay_source = self._peters_delay_to_selected_resonance_source_s(
                f_init,
                transition_frequency_hz,
                resonance_harmonic,
                primary_mass_kg,
                companion_mass_kg,
                eccentricity_initial,
            )
            self.initial_frequency_setup = {
                "mode": "orbital_start_ratio",
                "target_delay_obs_days": np.nan,
                "actual_delay_obs_days": float(delay_source * (1.0 + self.z) / 86400.0)
                if np.isfinite(delay_source)
                else np.nan,
                "frequency_ratio_to_selected_resonance": float(f_init / max(f_res, 1.0e-300)),
                "status": "ratio",
            }
            return f_init

        target_source_s = float(self.target_resonance_delay_obs_days) * 86400.0 / (1.0 + self.z)
        f_init, solve_info = self._solve_initial_frequency_for_resonance_delay(
            transition_frequency_hz,
            resonance_harmonic,
            primary_mass_kg,
            companion_mass_kg,
            eccentricity_initial,
            target_source_s,
        )
        if not np.isfinite(f_init) or f_init <= 0.0:
            f_init = self.orbital_start_ratio * f_res
            solve_info = {"status": "invalid_solution_fallback_ratio", "actual_delay_source_s": np.nan}
        actual_delay_source = float(solve_info.get("actual_delay_source_s", np.nan))
        self.initial_frequency_setup = {
            "mode": "target_resonance_delay",
            "target_delay_obs_days": float(self.target_resonance_delay_obs_days),
            "actual_delay_obs_days": float(actual_delay_source * (1.0 + self.z) / 86400.0)
            if np.isfinite(actual_delay_source)
            else np.nan,
            "frequency_ratio_to_selected_resonance": float(f_init / max(f_res, 1.0e-300)),
            "status": str(solve_info.get("status", "unknown")),
        }
        return f_init

    def _peters_c0(self, semi_major_axis, eccentricity):
        e_safe = float(np.clip(eccentricity, 1.0e-8, 0.999))
        return (
            float(semi_major_axis)
            * (1.0 - e_safe**2)
            / (e_safe ** (12.0 / 19.0))
            / (1.0 + 121.0 * e_safe**2 / 304.0) ** (870.0 / 2299.0)
        )

    def _peters_a_of_e(self, c0_value, eccentricity):
        e_safe = np.clip(np.asarray(eccentricity, dtype=float), 1.0e-8, 0.999)
        return (
            float(c0_value)
            * (e_safe ** (12.0 / 19.0))
            / np.maximum(1.0 - e_safe**2, 1.0e-12)
            * (1.0 + 121.0 * e_safe**2 / 304.0) ** (870.0 / 2299.0)
        )

    def _peters_rhs_for_params(self, semi_major_axis, eccentricity, primary_mass_kg, companion_mass_kg):
        e = float(np.clip(eccentricity, 0.0, 0.999))
        one_minus_e2 = max(1.0e-12, 1.0 - e * e)
        total_mass = primary_mass_kg + companion_mass_kg
        prefactor = self.G**3 * total_mass * primary_mass_kg * companion_mass_kg / self.c**5

        dadt = (
            -(64.0 / 5.0)
            * prefactor
            / (semi_major_axis**3 * one_minus_e2 ** 3.5)
            * (1.0 + (73.0 / 24.0) * e**2 + (37.0 / 96.0) * e**4)
        )
        dedt = (
            -(304.0 / 15.0)
            * e
            * prefactor
            / (semi_major_axis**4 * one_minus_e2 ** 2.5)
            * (1.0 + (121.0 / 304.0) * e**2)
        )
        orbital_omega = np.sqrt(self.G * total_mass / semi_major_axis**3)
        return dadt, dedt, orbital_omega

    def _mixing_overlap_cache_key(self):
        return (
            tuple(self.transition_solver_data["initial_state"]),
            tuple(self.transition_solver_data["final_state"]),
            int(self.overlap_grid_points),
            round(self.overlap_max_x, 12),
        )

    def _get_cached_mixing_overlaps(self):
        cache_key = self._mixing_overlap_cache_key()
        with self.__class__._CACHE_LOCK:
            cached = self.__class__._MIXING_OVERLAP_CACHE.get(cache_key)
            if cached is None:
                cached = self._precompute_mixing_overlaps()
                self.__class__._MIXING_OVERLAP_CACHE[cache_key] = cached
        return cached

    def _hansen_cache_key(self):
        e_max = min(0.95, max(0.05, self.e_init))
        return (
            tuple(self.harmonics.tolist()),
            int(self.radial_power),
            int(self.hansen_tidal_m),
            int(self.hansen_e_samples),
            int(self.hansen_M_samples),
            round(float(e_max), 12),
        )

    def _peters_rhs(self, a, e):
        e = np.clip(e, 0.0, 0.999)
        one_minus_e2 = max(1.0e-12, 1.0 - e * e)
        prefactor = self.G**3 * self.M_tot * self.M * self.M_star / self.c**5

        dadt = (
            -(64.0 / 5.0)
            * prefactor
            / (a**3 * one_minus_e2 ** 3.5)
            * (1.0 + (73.0 / 24.0) * e**2 + (37.0 / 96.0) * e**4)
        )
        dedt = (
            -(304.0 / 15.0)
            * e
            * prefactor
            / (a**4 * one_minus_e2 ** 2.5)
            * (1.0 + (121.0 / 304.0) * e**2)
        )
        orbital_omega = np.sqrt(self.G * self.M_tot / a**3)
        return dadt, dedt, orbital_omega

    def _coherence_diagnostics(self, cg, ce_tilde):
        overlap = np.conj(cg) * ce_tilde
        overlap_abs = np.abs(overlap)
        population_norm = np.abs(cg) ** 2 + np.abs(ce_tilde) ** 2
        safe_norm = np.maximum(population_norm, 1.0e-300)
        return {
            "overlap": overlap,
            "overlap_abs": overlap_abs,
            "population_norm": population_norm,
            "overlap_abs_normalized": overlap_abs / safe_norm,
        }

    def _orbit_rhs(self, _, y):
        a, e, phi = y
        dadt, dedt, dphidt = self._peters_rhs(a, e)
        return [dadt, dedt, dphidt]

    def _select_active_harmonic_indices(self):
        resonance_idx = self.harmonic_to_index[self.resonance_harmonic]
        if self.multi_harmonic_drive:
            candidate_order = np.argsort(np.abs(self.harmonics - self.resonance_harmonic))
            active_indices = np.sort(candidate_order[: self.harmonics_to_keep])
        else:
            active_indices = np.array([resonance_idx], dtype=int)
        active_harmonics = self.harmonics[active_indices]
        return resonance_idx, active_indices, active_harmonics

    def _initial_cloud_amplitudes(self, active_indices, active_harmonics):
        if self.cloud_initial_state == "bare":
            return 1.0 + 0.0j, 0.0 + 0.0j

        eta_vec = self._eta_vector(self.a_init, self.e_init)
        phase_factors = np.exp(-1j * (np.asarray(active_harmonics, dtype=float) - self.resonance_harmonic) * 0.0)
        eta_drive = np.sum(eta_vec[np.asarray(active_indices, dtype=int)] * phase_factors)
        detuning = self.resonance_harmonic * self.Omega_init - self.transition_omega
        hamiltonian = np.array(
            [
                [0.5 * detuning, eta_drive],
                [np.conj(eta_drive), -0.5 * detuning],
            ],
            dtype=np.complex128,
        )
        _, eigenvectors = np.linalg.eigh(hamiltonian)
        chosen = eigenvectors[:, int(np.argmax(np.abs(eigenvectors[0, :])))]
        if abs(chosen[0]) > 0.0:
            chosen = chosen * np.exp(-1j * np.angle(chosen[0]))
        norm = np.sqrt(np.sum(np.abs(chosen) ** 2))
        if not np.isfinite(norm) or norm <= 0.0:
            return 1.0 + 0.0j, 0.0 + 0.0j
        chosen = chosen / norm
        return complex(chosen[0]), complex(chosen[1])

    def _coupled_rhs(self, t_val, y_val, active_indices, active_harmonics):
        a, e, phi, cg_r, cg_i, ce_r, ce_i = y_val
        a = max(a, 1.0e-12)
        e = float(np.clip(e, 0.0, 0.999))
        cg = cg_r + 1j * cg_i
        ce_tilde = ce_r + 1j * ce_i

        dadt_peters, dedt_peters, omega_val = self._peters_rhs(a, e)
        eta_vec = self._eta_vector(a, e)
        phase_factors = np.exp(-1j * (active_harmonics - self.resonance_harmonic) * phi)
        eta_drive = np.sum(eta_vec[active_indices] * phase_factors)
        detuning = self.resonance_harmonic * omega_val - self.transition_omega
        resonance_gate = self._resonance_band_gate(
            t_val,
            omega_val,
            eta_vec,
            active_indices,
            active_harmonics,
        )
        eta_drive_gated = eta_drive * resonance_gate

        d_cg = -1j * (0.5 * detuning * cg + eta_drive_gated * ce_tilde)
        d_ce = -1j * (np.conj(eta_drive_gated) * cg - 0.5 * detuning * ce_tilde) - self.Gamma_decay * ce_tilde

        dadt = dadt_peters
        dedt = dedt_peters
        if self.include_orbital_backreaction and resonance_gate > 0.0:
            overlap = np.conj(cg) * ce_tilde
            if self.orbital_backreaction_mode == "coherent_drive":
                eta_backreaction = eta_drive_gated
            else:
                eta_backreaction = eta_vec[self.harmonic_to_index[self.resonance_harmonic]] * resonance_gate
            final_state_rate = -2.0 * np.imag(eta_backreaction * overlap)
            high_state_rate = self.transition_energy_sign * final_state_rate
            reduced_mass = self.M * self.M_star / self.M_tot
            one_minus_e2 = max(1.0e-12, 1.0 - e * e)
            l_orb = reduced_mass * np.sqrt(self.G * self.M_tot * a * one_minus_e2)
            energy_coeff = (a * self.delta_E_high_low_backreaction) / max(self.G * self.M * self.M_star, 1.0e-60)
            angular_coeff = self.backreaction_macro_scale * self.delta_m_high_low / max(l_orb, 1.0e-60)
            dadt -= (2.0 * a * energy_coeff) * high_state_rate
            e_safe = max(e, 1.0e-12)
            dedt -= ((1.0 - e**2) / e_safe) * (energy_coeff - angular_coeff) * high_state_rate

        return [dadt, dedt, omega_val, d_cg.real, d_cg.imag, d_ce.real, d_ce.imag]

    def _resolved_cloud_evolution_mode(self):
        return str(getattr(self, "_runtime_cloud_evolution_mode", self.cloud_evolution_mode)).lower()

    def _reference_gate_at_time(self, t_val):
        gate_times = getattr(self, "_runtime_resonance_gate_times", None)
        gate_values = getattr(self, "_runtime_resonance_gate_values", None)
        if gate_times is None or gate_values is None:
            return None
        gate_times = np.asarray(gate_times, dtype=float)
        gate_values = np.asarray(gate_values, dtype=float)
        if gate_times.size == 0 or gate_values.size != gate_times.size:
            return None
        return float(np.interp(float(t_val), gate_times, gate_values, left=gate_values[0], right=gate_values[-1]))

    def _resonance_band_gate(self, t_val, omega_val, eta_vec, active_indices, active_harmonics):
        mode = self._resolved_cloud_evolution_mode()
        if mode in {"full", "floating_full"}:
            return 1.0
        reference_gate = self._reference_gate_at_time(t_val)
        if reference_gate is not None:
            return 1.0 if reference_gate >= 0.5 else 0.0
        detunings = np.asarray(active_harmonics, dtype=float) * float(omega_val) - self.transition_omega
        closest_local = int(np.argmin(np.abs(detunings)))
        eta_width = abs(eta_vec[int(active_indices[closest_local])])
        width = abs(self.resonance_band_width_factor) * max(float(eta_width), 1.0e-300)
        if width <= 0.0:
            return 0.0
        return 1.0 if abs(float(detunings[closest_local])) <= width else 0.0

    def _resolve_transition_states(self):
        # 默认态选择和参数求解器/论文讨论保持一致：
        # fine: |322> -> |300>, hyperfine: |211> -> |21-1>
        if self.initial_state is not None and self.final_state is not None:
            return tuple(self.initial_state), tuple(self.final_state)

        if self.transition_family == "fine":
            return (3, 0, 0), (3, 2, 2)
        if self.transition_family == "hyperfine":
            return (2, 1, -1), (2, 1, 1)

        raise ValueError(f"Unknown transition_family '{self.transition_family}'.")

    def _infer_transition_energy_sign(self, initial_state, final_state):
        n_i, l_i, m_i = initial_state
        n_f, l_f, m_f = final_state
        if n_f != n_i:
            return float(np.sign(n_f - n_i))
        if l_f != l_i:
            return float(np.sign(l_f - l_i))
        if m_f != m_i:
            return float(np.sign(m_f - m_i))
        return 0.0

    def _paper_hyperfine_transition_frequency_hz(self, initial_state, final_state, alpha_value=None, mass_kg=None):
        """
        Hyperfine resonance frequency based on eqs. (2.17) and (2.19) of arXiv:2503.18121.

        The paper quotes the orbital resonance frequency f0. In this code the quantity
        `transition_frequency_hz` is matched by `resonance_harmonic * f_orb`, so we return
        n_res * f0 to keep the existing resonance condition unchanged.
        """
        n_i, l_i, m_i = initial_state
        n_f, l_f, m_f = final_state
        if n_i != n_f or l_i != l_f or m_i == m_f:
            return None
        if l_i <= 0:
            return None

        alpha_eval = float(self.alpha if alpha_value is None else alpha_value)
        mass_eval = float(self.M if mass_kg is None else mass_kg)
        m_abs = abs(int(m_i))
        denom = (
            (n_i**3)
            * (2 * l_i)
            * (2 * l_i + 1)
            * (2 * l_i + 2)
            * (m_abs**2 + 4.0 * alpha_eval**2)
        )
        if denom <= 0.0:
            return None

        orbital_omega_geom = 64.0 * m_abs * alpha_eval**7 / denom
        geometric_to_si_angular = self.c**3 / (self.G * mass_eval)
        orbital_frequency_hz = orbital_omega_geom * geometric_to_si_angular / (2.0 * np.pi)
        return float(self.resonance_harmonic) * orbital_frequency_hz

    def _omega_real_geom(self, state):
        n, l, _ = state
        alpha = self.alpha
        term1 = 1.0
        term2 = -alpha**2 / (2.0 * n**2)
        term3 = -alpha**4 / (8.0 * n**4)
        term4 = ((4.0 * l - 6.0 * n + 2.0) / (2.0 * n * (l + 1.0))) * (alpha**4 / n**3)
        return alpha * (term1 + term2 + term3 + term4)

    def _gamma_geom(self, state):
        n, l, m = state
        n_r = n - l - 1
        chi = self.bh_spin
        r_plus = 1.0 + math.sqrt(max(0.0, 1.0 - chi**2))
        omega_h = chi / (2.0 * r_plus)

        coeff_num = 2 ** (4 * l + 2) * math.factorial(2 * l + n_r + 1)
        coeff_den = ((l + n_r + 1) ** (2 * l + 4)) * math.factorial(n_r) * math.factorial(l)
        coeff1 = coeff_num / coeff_den
        coeff2 = (math.factorial(l) / (math.factorial(2 * l) * math.factorial(2 * l + 1))) ** 2

        prod_term = 1.0
        for j in range(1, l + 1):
            prod_term *= j**2 * (1.0 - chi**2) + 4.0 * (m * omega_h - self.alpha) ** 2

        c_nlm = coeff1 * coeff2 * prod_term
        return c_nlm * (m * omega_h - self.alpha) * self.alpha ** (4 * l + 5)

    def _compute_transition_quantities(self):
        # 1. 用两态的实部频率差给出 DeltaE
        # 2. 用末态的 Gamma 给出黑洞吸收耗散率
        initial_state, final_state = self._resolve_transition_states()
        omega_i_geom = self._omega_real_geom(initial_state)
        omega_f_geom = self._omega_real_geom(final_state)
        signed_delta_omega_geom = omega_f_geom - omega_i_geom
        delta_omega_geom = abs(signed_delta_omega_geom)
        transition_energy_sign = float(np.sign(signed_delta_omega_geom))
        manual_transition_frequency_hz = self._manual_transition_frequency_hz
        geometric_to_si_angular = self.c**3 / (self.G * self.M)
        if delta_omega_geom <= 0.0:
            paper_transition_frequency_hz = self._paper_hyperfine_transition_frequency_hz(initial_state, final_state)
            if manual_transition_frequency_hz is not None:
                transition_frequency_hz = float(manual_transition_frequency_hz)
            elif paper_transition_frequency_hz is not None:
                transition_frequency_hz = float(paper_transition_frequency_hz)
            else:
                raise ValueError(
                    "The selected transition has zero energy splitting under the current parameter-solver formula. "
                    "Use a fine transition or provide transition_frequency_hz explicitly."
                )
        else:
            transition_frequency_hz = delta_omega_geom * geometric_to_si_angular / (2.0 * np.pi)
        if transition_energy_sign == 0.0:
            transition_energy_sign = self._infer_transition_energy_sign(initial_state, final_state)
        if transition_energy_sign == 0.0:
            raise ValueError("Cannot infer the sign of E_final - E_initial for the selected transition.")
        gamma_decay_hz = abs(self._gamma_geom(final_state)) * geometric_to_si_angular / (2.0 * np.pi)
        boson_angular_frequency = self.alpha * geometric_to_si_angular
        boson_mass_eV = self.hbar * boson_angular_frequency / self.eV

        return {
            "initial_state": initial_state,
            "final_state": final_state,
            "signed_delta_omega_geom": signed_delta_omega_geom,
            "delta_omega_geom": delta_omega_geom,
            "transition_energy_sign": transition_energy_sign,
            "transition_frequency_hz": transition_frequency_hz,
            "gamma_decay_hz": gamma_decay_hz,
            "boson_mass_eV": boson_mass_eV,
        }

    def _radial_wavefunction_dimensionless(self, state, x):
        n, l, _ = state
        rho = 2.0 * x / n
        normalization = (2.0 / n) ** 1.5 * math.sqrt(
            math.factorial(n - l - 1) / (2.0 * n * math.factorial(n + l))
        )
        laguerre = eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
        return normalization * np.exp(-x / n) * rho**l * laguerre

    def _compute_transition_geometry(self):
        return compute_transition_geometry(
            self.transition_solver_data["initial_state"],
            self.transition_solver_data["final_state"],
            self._radial_wavefunction_dimensionless,
            overlap_max_x=self.overlap_max_x,
            overlap_grid_points=self.overlap_grid_points,
        )

    def _compute_angular_overlap(self, initial_state, final_state):
        _, l_i, m_i = initial_state
        _, l_f, m_f = final_state
        m_star = m_f - m_i

        theta = np.linspace(0.0, np.pi, 320)
        phi = np.linspace(0.0, 2.0 * np.pi, 640, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

        y_tidal = spherical_harmonic(m_star, 2, phi_grid, theta_grid)
        y_i = spherical_harmonic(m_i, l_i, phi_grid, theta_grid)
        y_f = spherical_harmonic(m_f, l_f, phi_grid, theta_grid)

        integrand = y_tidal * y_i * np.conj(y_f) * np.sin(theta_grid)
        phi_integral = np.trapezoid(integrand, phi, axis=1)
        full_integral = np.trapezoid(phi_integral, theta)
        return abs(full_integral)

    def _precompute_mixing_overlaps(self):
        # 预计算论文 A.3-A.5 中的径向/角向重叠，
        # 后续 eta(A.6) 只需要随轨道半径做插值，避免重复积分
        initial_state = self.transition_solver_data["initial_state"]
        final_state = self.transition_solver_data["final_state"]

        x_grid = np.logspace(-6, np.log10(self.overlap_max_x), self.overlap_grid_points)
        radial_i = self._radial_wavefunction_dimensionless(initial_state, x_grid)
        radial_f = self._radial_wavefunction_dimensionless(final_state, x_grid)

        inner_integrand = x_grid**4 * radial_i * radial_f
        outer_integrand = (radial_i * radial_f) / x_grid

        inner_cumulative = cumulative_trapezoid(inner_integrand, x_grid, initial=0.0)
        outer_total = np.trapezoid(outer_integrand, x_grid)
        outer_cumulative = outer_total - cumulative_trapezoid(outer_integrand, x_grid, initial=0.0)

        angular_overlap = self._compute_angular_overlap(initial_state, final_state)

        return {
            "x_grid": x_grid,
            "inner_cumulative": inner_cumulative,
            "outer_cumulative": outer_cumulative,
            "angular_overlap": angular_overlap,
        }

    def _formula_eta_hz(self, semi_major_axis):
        # 论文 2503.18121 Appendix A 的 A.6 式数值版。
        # 这里返回的是 SI 单位下的混合频率 eta/(2pi)，单位 Hz。
        q_mass = self.M_star / self.M
        orbital_omega = np.sqrt(self.G * self.M_tot / semi_major_axis**3)
        m_omega = self.G * self.M * orbital_omega / self.c**3
        x_star = np.clip(
            semi_major_axis / self.r_c,
            self.mixing_overlap_data["x_grid"][0],
            self.mixing_overlap_data["x_grid"][-1],
        )

        i_in = np.interp(x_star, self.mixing_overlap_data["x_grid"], self.mixing_overlap_data["inner_cumulative"])
        i_out = np.interp(x_star, self.mixing_overlap_data["x_grid"], self.mixing_overlap_data["outer_cumulative"])
        i_a = self.mixing_overlap_data["angular_overlap"]

        term_inner = q_mass * m_omega * i_in / (self.alpha**3 * (1.0 + q_mass))
        term_outer = (
            self.alpha**7
            * q_mass
            * (1.0 + q_mass) ** (2.0 / 3.0)
            * i_out
            / max(m_omega, 1.0e-30) ** (7.0 / 3.0)
        )
        eta_over_omega = (3.0 * np.pi / 10.0) * i_a * abs(term_inner + term_outer)
        return eta_over_omega * orbital_omega / (2.0 * np.pi)

    def _ensure_hansen_table(self):
        if self._hansen_e_grid is not None:
            return

        cache_key = self._hansen_cache_key()
        with self.__class__._CACHE_LOCK:
            cached = self.__class__._HANSEN_TABLE_CACHE.get(cache_key)
            if cached is None:
                e_max = min(0.95, max(0.05, self.e_init))
                e_grid = np.linspace(0.0, e_max, self.hansen_e_samples)
                real_table = np.zeros((len(self.harmonics), self.hansen_e_samples))
                imag_table = np.zeros((len(self.harmonics), self.hansen_e_samples))

                mean_anomaly = np.linspace(0.0, 2.0 * np.pi, self.hansen_M_samples, endpoint=False)
                phase_matrix = np.exp(1j * np.outer(self.harmonics, mean_anomaly))

                for idx, ecc in enumerate(e_grid):
                    eccentric_anomaly = solve_kepler(mean_anomaly, ecc)
                    radial_ratio = 1.0 - ecc * np.cos(eccentric_anomaly)
                    cos_true = (np.cos(eccentric_anomaly) - ecc) / radial_ratio
                    sin_true = np.sqrt(max(1.0e-14, 1.0 - ecc**2)) * np.sin(eccentric_anomaly) / radial_ratio
                    true_anomaly = np.arctan2(sin_true, cos_true)

                    base = radial_ratio ** (-self.radial_power) * np.exp(
                        -1j * self.hansen_tidal_m * true_anomaly
                    )
                    coeffs = phase_matrix @ base / mean_anomaly.size

                    real_table[:, idx] = coeffs.real
                    imag_table[:, idx] = coeffs.imag

                cached = (e_grid, real_table, imag_table)
                self.__class__._HANSEN_TABLE_CACHE[cache_key] = cached

        self._hansen_e_grid, self._hansen_real, self._hansen_imag = cached

    def _solve_eccentricity_at_resonance(self, c0_value, semi_major_axis_target, eccentricity_initial):
        e_initial = float(np.clip(eccentricity_initial, 1.0e-8, 0.999))

        def residual(e_val):
            return float(self._peters_a_of_e(c0_value, e_val) - semi_major_axis_target)

        residual_initial = residual(e_initial)
        if abs(residual_initial) <= max(1.0e-12 * semi_major_axis_target, 1.0e-18):
            return e_initial

        e_floor = 1.0e-8
        residual_floor = residual(e_floor)
        if residual_initial * residual_floor > 0.0:
            return np.nan
        return float(brentq(residual, e_floor, e_initial, maxiter=200))

    def _integrate_peters_time_to_e(
        self,
        eccentricity_initial,
        eccentricity_target,
        c0_value,
        primary_mass_kg,
        companion_mass_kg,
        num_points=256,
    ):
        e_start = float(np.clip(eccentricity_initial, 1.0e-8, 0.999))
        e_stop = float(np.clip(eccentricity_target, 1.0e-8, 0.999))
        if e_stop >= e_start:
            return 0.0

        e_grid = np.linspace(e_stop, e_start, int(max(16, num_points)))
        a_grid = self._peters_a_of_e(c0_value, e_grid)
        dtde = np.empty_like(e_grid)
        for idx, (a_val, e_val) in enumerate(zip(a_grid, e_grid)):
            _, dedt, _ = self._peters_rhs_for_params(a_val, e_val, primary_mass_kg, companion_mass_kg)
            dtde[idx] = 1.0 / max(abs(dedt), 1.0e-60)
        return float(np.trapezoid(dtde, e_grid))

    def _interp_hansen(self, eccentricity):
        self._ensure_hansen_table()
        ecc = np.asarray(eccentricity, dtype=float)
        ecc = np.clip(ecc, self._hansen_e_grid[0], self._hansen_e_grid[-1])

        if ecc.ndim == 0:
            idx_hi = int(np.clip(np.searchsorted(self._hansen_e_grid, float(ecc), side="right"), 1, self._hansen_e_grid.size - 1))
            idx_lo = idx_hi - 1
            e_lo = self._hansen_e_grid[idx_lo]
            e_hi = self._hansen_e_grid[idx_hi]
            weight_hi = 0.0 if np.isclose(e_hi, e_lo) else float((float(ecc) - e_lo) / (e_hi - e_lo))
            weight_lo = 1.0 - weight_hi
            real = weight_lo * self._hansen_real[:, idx_lo] + weight_hi * self._hansen_real[:, idx_hi]
            imag = weight_lo * self._hansen_imag[:, idx_lo] + weight_hi * self._hansen_imag[:, idx_hi]
            return real + 1j * imag

        flat_ecc = ecc.reshape(-1)
        idx_hi = np.clip(np.searchsorted(self._hansen_e_grid, flat_ecc, side="right"), 1, self._hansen_e_grid.size - 1)
        idx_lo = idx_hi - 1
        e_lo = self._hansen_e_grid[idx_lo]
        e_hi = self._hansen_e_grid[idx_hi]
        denom = np.where(np.isclose(e_hi, e_lo), 1.0, e_hi - e_lo)
        weight_hi = (flat_ecc - e_lo) / denom
        weight_lo = 1.0 - weight_hi

        real = weight_lo[None, :] * self._hansen_real[:, idx_lo] + weight_hi[None, :] * self._hansen_real[:, idx_hi]
        imag = weight_lo[None, :] * self._hansen_imag[:, idx_lo] + weight_hi[None, :] * self._hansen_imag[:, idx_hi]
        return (real + 1j * imag).reshape((len(self.harmonics),) + ecc.shape)

    def _eta_vector(self, semi_major_axis, eccentricity):
        semi_major_axis, eccentricity = np.broadcast_arrays(
            np.asarray(semi_major_axis, dtype=float),
            np.asarray(eccentricity, dtype=float),
        )
        hansen = self._interp_hansen(eccentricity)
        scale = (self.a_init / semi_major_axis) ** self.radial_power
        if scale.ndim == 0:
            return self.eta_ref * float(scale) * hansen
        return self.eta_ref * hansen * scale.reshape((1,) + scale.shape)

    def solve_orbit(self, duration_yr=0.4, secular_samples=3000):
        total_time = duration_yr * self.yr
        max_step = min(total_time / (1800.0 if self.solver_profile == "fast" else 2500.0), 20.0 * self.period_init if self.solver_profile == "fast" else 10.0 * self.period_init)

        sol = solve_ivp(
            self._orbit_rhs,
            (0.0, total_time),
            [self.a_init, self.e_init, 0.0],
            dense_output=True,
            max_step=max_step,
            method=self.solver_method,
            rtol=self.solver_rtol,
            atol=self.orbit_atol,
        )

        t_grid = np.linspace(0.0, sol.t[-1], int(secular_samples))
        a, e, phi = sol.sol(t_grid)
        e = np.clip(e, 0.0, 0.999)
        omega = np.sqrt(self.G * self.M_tot / a**3)

        return {
            "solution": sol,
            "t": t_grid,
            "a": a,
            "e": e,
            "phi": phi,
            "omega": omega,
            "f_orb": omega / (2.0 * np.pi),
        }

    def _build_reference_cloud_on_orbit(self, orbit, active_harmonics):
        omega_grid = np.asarray(orbit["omega"], dtype=float)
        active_harmonics = np.asarray(active_harmonics, dtype=int)
        eta_series = self._eta_vector(orbit["a"], orbit["e"])
        detuning_series = self.harmonics[:, None] * omega_grid[None, :] - self.transition_omega
        active_indices = np.asarray([self.harmonic_to_index[int(n)] for n in active_harmonics], dtype=int)
        reference_gate = self._instantaneous_resonance_band_gate_series(
            omega_grid,
            eta_series,
            active_indices,
            active_harmonics,
        )
        return {
            "t": np.asarray(orbit["t"], dtype=float),
            "eta_series": eta_series,
            "detuning_series": detuning_series,
            "active_harmonics": active_harmonics,
            "resonance_band_gate": reference_gate,
        }

    def _instantaneous_resonance_band_gate_series(
        self,
        omega_grid,
        eta_series,
        active_indices,
        active_harmonics,
    ):
        omega_grid = np.asarray(omega_grid, dtype=float)
        eta_series = np.asarray(eta_series, dtype=np.complex128)
        active_indices = np.asarray(active_indices, dtype=int)
        active_harmonics = np.asarray(active_harmonics, dtype=float)
        detunings = active_harmonics[:, None] * omega_grid[None, :] - self.transition_omega
        eta_abs = np.abs(eta_series[active_indices, :])
        widths = abs(self.resonance_band_width_factor) * np.maximum(eta_abs, 1.0e-300)
        in_band = np.any(np.abs(detunings) <= widths, axis=0)
        return in_band.astype(float)

    def _choose_runtime_cloud_mode(self, reference_band_diagnostics):
        requested_mode = self.cloud_evolution_mode
        if requested_mode != "auto":
            return requested_mode
        selected_band_fraction = float(reference_band_diagnostics.get("selected_band_fraction", 0.0))
        active_band_fraction = float(reference_band_diagnostics.get("active_band_fraction", 0.0))
        wide_near_resonance = max(selected_band_fraction, active_band_fraction) >= self.floating_band_fraction_threshold
        return "floating_full" if wide_near_resonance else "band_gated"

    def _configure_runtime_resonance_gate(
        self,
        duration_yr,
        secular_samples,
        active_harmonics,
    ):
        reference_orbit = self.solve_orbit(duration_yr=duration_yr, secular_samples=secular_samples)
        reference_cloud = self._build_reference_cloud_on_orbit(reference_orbit, active_harmonics)
        reference_cloud["resonance_events"] = self._build_resonance_events(reference_orbit, reference_cloud)
        reference_cloud["degenerate_resonance_groups"] = self._collect_degenerate_resonance_groups(
            reference_cloud["resonance_events"]
        )
        reference_cloud["resonance_band_diagnostics"] = self._resonance_band_diagnostics(
            reference_orbit,
            reference_cloud,
        )

        runtime_mode = self._choose_runtime_cloud_mode(reference_cloud["resonance_band_diagnostics"])
        reference_cloud["resonance_band_diagnostics"]["auto_runtime_decision"] = runtime_mode
        self._runtime_cloud_evolution_mode = runtime_mode
        self._runtime_reference_band_diagnostics = dict(reference_cloud["resonance_band_diagnostics"])
        self._runtime_resonance_gate_times = np.asarray(reference_orbit["t"], dtype=float)
        if runtime_mode in {"full", "floating_full"}:
            self._runtime_resonance_gate_values = np.ones_like(self._runtime_resonance_gate_times, dtype=float)
        else:
            self._runtime_resonance_gate_values = np.asarray(
                reference_cloud["resonance_band_gate"],
                dtype=float,
            )
        return reference_orbit, reference_cloud

    def solve_coupled_system(self, duration_yr=1.0, secular_samples=3000):
        total_time = duration_yr * self.yr
        max_step = min(total_time / (1800.0 if self.solver_profile == "fast" else 2500.0), 20.0 * self.period_init if self.solver_profile == "fast" else 10.0 * self.period_init)
        resonance_idx, active_indices, active_harmonics = self._select_active_harmonic_indices()
        reference_orbit, reference_cloud = self._configure_runtime_resonance_gate(
            duration_yr,
            secular_samples,
            active_harmonics,
        )
        cg0, ce0 = self._initial_cloud_amplitudes(active_indices, active_harmonics)

        sol = solve_ivp(
            lambda t_val, y_val: self._coupled_rhs(
                t_val,
                y_val,
                active_indices,
                active_harmonics,
            ),
            (0.0, total_time),
            [self.a_init, self.e_init, 0.0, cg0.real, cg0.imag, ce0.real, ce0.imag],
            dense_output=True,
            max_step=max_step,
            method=self.solver_method,
            rtol=self.solver_rtol,
            atol=self.coupled_atol,
        )

        t_grid = np.linspace(0.0, sol.t[-1], int(secular_samples))
        a, e, phi, cg_real, cg_imag, ce_real, ce_imag = sol.sol(t_grid)
        e = np.clip(e, 0.0, 0.999)
        omega = np.sqrt(self.G * self.M_tot / np.maximum(a, 1.0e-30) ** 3)
        cg = cg_real + 1j * cg_imag
        ce_tilde = ce_real + 1j * ce_imag

        eta_series = self._eta_vector(a, e)

        detuning_series = self.harmonics[:, None] * omega[None, :] - self.transition_omega
        closest_harmonic = self.harmonics[np.argmin(np.abs(detuning_series), axis=0)]
        resonance_track = detuning_series[resonance_idx]
        resonance_time = t_grid[np.argmin(np.abs(resonance_track))]
        coherence = self._coherence_diagnostics(cg, ce_tilde)
        overlap = coherence["overlap"]
        selected_final_state_rate = -2.0 * np.imag(eta_series[resonance_idx] * overlap)
        selected_high_state_rate = self.transition_energy_sign * selected_final_state_rate
        resonance_band_gate = self._resonance_band_gate_series(
            omega,
            eta_series,
            active_indices,
            active_harmonics,
            t_grid=t_grid,
        )

        orbit = {
            "solution": _SolutionSlice(sol, slice(0, 3)),
            "t": t_grid,
            "a": a,
            "e": e,
            "phi": phi,
            "omega": omega,
            "f_orb": omega / (2.0 * np.pi),
        }
        cloud = {
            "solution": _SolutionSlice(sol, slice(3, 7)),
            "t": t_grid,
            "cg": cg,
            "ce_tilde": ce_tilde,
            "eta_series": eta_series,
            "detuning_series": detuning_series,
            "closest_harmonic": closest_harmonic,
            "active_harmonics": active_harmonics,
            "resonance_time": resonance_time,
            "pop_ground": np.abs(cg) ** 2,
            "pop_excited": np.abs(ce_tilde) ** 2,
            "overlap": overlap,
            "overlap_abs": coherence["overlap_abs"],
            "population_norm": coherence["population_norm"],
            "overlap_abs_normalized": coherence["overlap_abs_normalized"],
            "selected_final_state_rate": selected_final_state_rate,
            "selected_high_state_rate": selected_high_state_rate,
            "resonance_band_gate": resonance_band_gate,
            "reference_orbit": reference_orbit,
            "reference_resonance_events": reference_cloud.get("resonance_events", []),
            "reference_degenerate_resonance_groups": reference_cloud.get("degenerate_resonance_groups", []),
            "reference_resonance_band_diagnostics": reference_cloud.get("resonance_band_diagnostics", {}),
            "production_cloud_evolution_mode": self._resolved_cloud_evolution_mode(),
        }
        cloud["resonance_events"] = self._build_resonance_events(orbit, cloud)
        cloud["degenerate_resonance_groups"] = self._collect_degenerate_resonance_groups(
            cloud["resonance_events"]
        )
        cloud["resonance_band_diagnostics"] = self._resonance_band_diagnostics(orbit, cloud)
        return orbit, cloud

    def _resonance_band_gate_series(
        self,
        omega_grid,
        eta_series,
        active_indices,
        active_harmonics,
        t_grid=None,
    ):
        omega_grid = np.asarray(omega_grid, dtype=float)
        eta_series = np.asarray(eta_series, dtype=np.complex128)
        mode = self._resolved_cloud_evolution_mode()
        if mode in {"full", "floating_full"}:
            return np.ones_like(omega_grid, dtype=float)
        if t_grid is not None:
            gate_times = getattr(self, "_runtime_resonance_gate_times", None)
            gate_values = getattr(self, "_runtime_resonance_gate_values", None)
            if gate_times is not None and gate_values is not None:
                gate_times = np.asarray(gate_times, dtype=float)
                gate_values = np.asarray(gate_values, dtype=float)
                if gate_times.size and gate_values.size == gate_times.size:
                    return np.interp(
                        np.asarray(t_grid, dtype=float),
                        gate_times,
                        gate_values,
                        left=gate_values[0],
                        right=gate_values[-1],
                    )
        active_indices = np.asarray(active_indices, dtype=int)
        active_harmonics = np.asarray(active_harmonics, dtype=float)
        return self._instantaneous_resonance_band_gate_series(
            omega_grid,
            eta_series,
            active_indices,
            active_harmonics,
        )

    def _resonance_band_diagnostics(self, orbit, cloud):
        t_grid = np.asarray(orbit["t"], dtype=float)
        if t_grid.size < 2:
            return {}
        omega_grid = np.asarray(orbit["omega"], dtype=float)
        eta_series = np.asarray(cloud["eta_series"], dtype=np.complex128)
        active_harmonics = np.asarray(cloud["active_harmonics"], dtype=int)
        active_indices = np.asarray([self.harmonic_to_index[int(n)] for n in active_harmonics], dtype=int)
        detunings = active_harmonics[:, None] * omega_grid[None, :] - self.transition_omega
        eta_abs = np.abs(eta_series[active_indices, :])
        eta_widths = np.maximum(eta_abs, 1.0e-300)
        ratios = np.abs(detunings) / eta_widths
        min_ratio = np.min(ratios, axis=0)
        gate = np.asarray(cloud.get("resonance_band_gate", np.zeros_like(t_grid)), dtype=float)
        total_span = max(float(t_grid[-1] - t_grid[0]), 1.0e-300)
        in_band = gate > 0.0
        band_fraction = float(np.mean(in_band))
        band_time_source = float(np.trapezoid(in_band.astype(float), t_grid))
        selected_idx = self.harmonic_to_index[int(self.resonance_harmonic)]
        selected_det = np.asarray(cloud["detuning_series"][selected_idx], dtype=float)
        selected_eta = np.abs(eta_series[selected_idx])
        selected_ratio = np.abs(selected_det) / np.maximum(selected_eta, 1.0e-300)
        best_idx = int(np.argmin(np.abs(selected_det)))
        slope = float(np.gradient(selected_det, t_grid, edge_order=1)[best_idx])
        eta_best = float(selected_eta[best_idx])
        tau_lz = eta_best / max(abs(slope), 1.0e-300)
        selected_width_time = abs(self.resonance_band_width_factor) * tau_lz
        selected_band_mask = np.abs(selected_det) <= abs(self.resonance_band_width_factor) * np.maximum(selected_eta, 1.0e-300)
        selected_band_fraction = float(np.mean(selected_band_mask))
        det_in_band = selected_det[selected_band_mask]
        if det_in_band.size:
            detuning_drift_in_band = float((np.max(det_in_band) - np.min(det_in_band)) / max(eta_best, 1.0e-300))
        else:
            detuning_drift_in_band = np.inf
        floating_candidate = bool(
            selected_band_fraction >= self.floating_band_fraction_threshold
            and detuning_drift_in_band <= self.floating_detuning_drift_threshold
        )
        sustained_near_resonance_candidate = bool(
            selected_band_fraction >= self.floating_band_fraction_threshold
            or band_fraction >= self.floating_band_fraction_threshold
        )
        return {
            "requested_mode": self.cloud_evolution_mode,
            "mode": self._resolved_cloud_evolution_mode(),
            "band_width_factor": float(self.resonance_band_width_factor),
            "active_band_fraction": band_fraction,
            "active_band_time_source_s": band_time_source,
            "active_band_time_obs_days": band_time_source * (1.0 + self.z) / 86400.0,
            "selected_tau_lz_source_s": float(tau_lz),
            "selected_tau_lz_obs_days": float(tau_lz * (1.0 + self.z) / 86400.0),
            "selected_band_half_width_source_s": float(selected_width_time),
            "selected_band_fraction": selected_band_fraction,
            "selected_start_in_band": bool(selected_band_mask[0]) if selected_band_mask.size else False,
            "selected_end_in_band": bool(selected_band_mask[-1]) if selected_band_mask.size else False,
            "selected_start_abs_detuning_over_eta": float(selected_ratio[0]) if selected_ratio.size else np.nan,
            "selected_end_abs_detuning_over_eta": float(selected_ratio[-1]) if selected_ratio.size else np.nan,
            "selected_min_abs_detuning_over_eta": float(np.nanmin(selected_ratio)) if selected_ratio.size else np.nan,
            "selected_median_abs_detuning_over_eta": float(np.nanmedian(selected_ratio)) if selected_ratio.size else np.nan,
            "selected_detuning_drift_in_band_over_eta": detuning_drift_in_band,
            "floating_candidate": floating_candidate,
            "sustained_near_resonance_candidate": sustained_near_resonance_candidate,
            "min_abs_detuning_over_eta": float(np.nanmin(min_ratio)),
            "median_abs_detuning_over_eta": float(np.nanmedian(min_ratio)),
            "max_gate": float(np.max(gate)) if gate.size else 0.0,
        }

    def _build_resonance_events(self, orbit, cloud):
        t_grid = np.asarray(orbit["t"], dtype=float)
        a_grid = np.asarray(orbit["a"], dtype=float)
        e_grid = np.asarray(orbit["e"], dtype=float)
        omega_grid = np.asarray(orbit["omega"], dtype=float)
        detuning_series = np.asarray(cloud["detuning_series"], dtype=float)
        active_harmonics = np.asarray(cloud["active_harmonics"], dtype=int)
        event_harmonics = self.harmonics[: max(1, min(int(self.harmonics_to_keep), len(self.harmonics)))]
        events = []

        eta_series = np.asarray(cloud.get("eta_series", np.zeros_like(detuning_series)), dtype=np.complex128)
        for harmonic in event_harmonics:
            idx = self.harmonic_to_index[int(harmonic)]
            detuning_track = detuning_series[idx]
            crossing_indices = np.where(detuning_track[:-1] * detuning_track[1:] <= 0.0)[0]
            crossed = crossing_indices.size > 0
            closest_at_boundary = False
            closest_boundary = ""
            if crossed:
                left_idx = int(crossing_indices[0])
                right_idx = left_idx + 1
                det_left = float(detuning_track[left_idx])
                det_right = float(detuning_track[right_idx])
                t_left = float(t_grid[left_idx])
                t_right = float(t_grid[right_idx])
                if np.isclose(det_right, det_left):
                    t_res = 0.5 * (t_left + t_right)
                else:
                    t_res = t_left + (0.0 - det_left) * (t_right - t_left) / (det_right - det_left)
                detuning_abs_min = 0.0
            else:
                best_idx = int(np.argmin(np.abs(detuning_track)))
                t_res = float(t_grid[best_idx])
                detuning_abs_min = float(abs(detuning_track[best_idx]))
                closest_at_boundary = best_idx in (0, t_grid.size - 1)
                if best_idx == 0:
                    closest_boundary = "start"
                elif best_idx == t_grid.size - 1:
                    closest_boundary = "end"

            omega_res = float(np.interp(t_res, t_grid, omega_grid))
            a_res = float(np.interp(t_res, t_grid, a_grid))
            e_res = float(np.interp(t_res, t_grid, e_grid))
            eta_track = eta_series[idx]
            eta_real = float(np.interp(t_res, t_grid, np.real(eta_track)))
            eta_imag = float(np.interp(t_res, t_grid, np.imag(eta_track)))
            eta_complex = eta_real + 1j * eta_imag
            eta_n = abs(eta_complex)
            if t_grid.size >= 2:
                detuning_sweep = abs(float(np.gradient(detuning_track, t_grid)[np.argmin(np.abs(t_grid - t_res))]))
            else:
                detuning_sweep = np.nan
            if not np.isfinite(detuning_sweep) or detuning_sweep <= 0.0:
                adiabatic_z = np.inf
            else:
                adiabatic_z = float(eta_n**2 / max(detuning_sweep, 1.0e-60))
            resonance_width_source_s = self._estimate_resonance_width_source_s(eta_n, detuning_sweep)
            event_record = {
                "harmonic": int(harmonic),
                "channel_name": "primary",
                "time_source": t_res,
                "time_obs": t_res * (1.0 + self.z),
                "orbital_frequency_obs": omega_res / (2.0 * np.pi * (1.0 + self.z)),
                "orbital_period_source": 2.0 * np.pi / max(omega_res, 1.0e-30),
                "semi_major_axis_m": a_res,
                "eccentricity": e_res,
                "detuning_abs_min": detuning_abs_min,
                "eta_n_hz": float(eta_n / (2.0 * np.pi)),
                "eta_phase_rad": float(np.angle(eta_complex)) if eta_n > 0.0 else 0.0,
                "detuning_sweep_angular_per_s": float(detuning_sweep),
                "resonance_width_source_s": float(resonance_width_source_s)
                if np.isfinite(resonance_width_source_s)
                else np.nan,
                "adiabatic_z": adiabatic_z,
                "z_value": adiabatic_z,
                "adiabatic_survives": bool(adiabatic_z <= self.adiabatic_z_threshold),
                "crossed_zero": bool(crossed),
                "closest_at_boundary": bool(closest_at_boundary),
                "closest_boundary": closest_boundary,
                "is_selected_resonance": int(harmonic) == int(self.resonance_harmonic),
            }
            events.append(event_record)

        events.sort(key=lambda item: item["time_source"])
        return self._annotate_degenerate_resonance_groups(events)

    def _filter_effective_resonance_events(self, resonance_events, require_crossing=True):
        effective = []
        for event in resonance_events:
            if require_crossing and not event.get("crossed_zero", False):
                continue
            adiabatic_z = event.get("adiabatic_z", event.get("z_value", np.inf))
            if not np.isfinite(adiabatic_z):
                continue
            if adiabatic_z <= self.adiabatic_z_threshold:
                effective.append(dict(event))
        return effective

    def _estimate_resonance_width_source_s(self, eta_angular, detuning_sweep_angular_per_s):
        eta_scale = float(abs(eta_angular))
        sweep_scale = float(abs(detuning_sweep_angular_per_s))
        if not np.isfinite(eta_scale) or not np.isfinite(sweep_scale) or eta_scale <= 0.0 or sweep_scale <= 0.0:
            return np.nan
        return float(4.0 * eta_scale / sweep_scale)

    def _resonance_bundle_half_width_source_s(self, event):
        width_source_s = float(event.get("resonance_width_source_s", np.nan))
        if not np.isfinite(width_source_s) or width_source_s <= 0.0:
            width_source_s = float(event.get("orbital_period_source", 0.0))
        return 0.5 * self.degenerate_bundle_merge_factor * max(width_source_s, 0.0)

    def _annotate_degenerate_resonance_groups(self, resonance_events):
        annotated = [dict(event) for event in resonance_events]
        if not annotated:
            return annotated

        valid_indices = [
            idx
            for idx, event in enumerate(annotated)
            if np.isfinite(event.get("time_source", np.nan))
        ]
        valid_indices.sort(key=lambda idx: float(annotated[idx]["time_source"]))

        group_id = 0
        current_group = []
        current_stop = -np.inf

        def flush_current_group():
            nonlocal group_id, current_group, current_stop
            if not current_group:
                return
            group_events = [annotated[idx] for idx in current_group]
            harmonics = tuple(sorted({int(event["harmonic"]) for event in group_events}))
            channels = tuple(sorted({str(event.get("channel_name", "primary")) for event in group_events}))
            group_size = len(current_group)
            for idx in current_group:
                annotated[idx]["degenerate_group_id"] = int(group_id)
                annotated[idx]["degenerate_group_size"] = int(group_size)
                annotated[idx]["degenerate_group_harmonics"] = harmonics
                annotated[idx]["degenerate_group_channels"] = channels
            group_id += 1
            current_group = []
            current_stop = -np.inf

        for idx in valid_indices:
            event = annotated[idx]
            half_width = self._resonance_bundle_half_width_source_s(event)
            start = float(event["time_source"]) - half_width
            stop = float(event["time_source"]) + half_width
            if current_group and start > current_stop:
                flush_current_group()
            current_group.append(idx)
            current_stop = max(current_stop, stop)
        flush_current_group()

        for idx, event in enumerate(annotated):
            if "degenerate_group_id" in event:
                continue
            event["degenerate_group_id"] = int(group_id)
            event["degenerate_group_size"] = 1
            event["degenerate_group_harmonics"] = (int(event["harmonic"]),)
            event["degenerate_group_channels"] = (str(event.get("channel_name", "primary")),)
            group_id += 1
        return annotated

    def _collect_degenerate_resonance_groups(self, resonance_events):
        grouped = {}
        for event in resonance_events:
            group_id = int(event.get("degenerate_group_id", -1))
            grouped.setdefault(group_id, []).append(event)

        summaries = []
        for group_id in sorted(grouped):
            group_events = grouped[group_id]
            finite_times = [float(event["time_source"]) for event in group_events if np.isfinite(event.get("time_source", np.nan))]
            finite_freqs = [
                float(event["orbital_frequency_obs"])
                for event in group_events
                if np.isfinite(event.get("orbital_frequency_obs", np.nan))
            ]
            summaries.append(
                {
                    "group_id": int(group_id),
                    "size": len(group_events),
                    "harmonics": tuple(sorted({int(event["harmonic"]) for event in group_events})),
                    "channels": tuple(sorted({str(event.get("channel_name", "primary")) for event in group_events})),
                    "time_source_min": float(min(finite_times)) if finite_times else np.nan,
                    "time_source_max": float(max(finite_times)) if finite_times else np.nan,
                    "orbital_frequency_obs_min": float(min(finite_freqs)) if finite_freqs else np.nan,
                    "orbital_frequency_obs_max": float(max(finite_freqs)) if finite_freqs else np.nan,
                    "events": [dict(event) for event in group_events],
                }
            )
        return summaries

    def _binary_strain_coeff(self, harmonic, eccentricity):
        eccentricity = np.asarray(eccentricity, dtype=float)
        x_arg = harmonic * eccentricity
        j_minus_2 = jv(harmonic - 2, x_arg)
        j_minus_1 = jv(harmonic - 1, x_arg)
        j_0 = jv(harmonic, x_arg)
        j_plus_1 = jv(harmonic + 1, x_arg)
        j_plus_2 = jv(harmonic + 2, x_arg)

        term1 = j_minus_2 - 2.0 * eccentricity * j_minus_1 + (2.0 / harmonic) * j_0
        term1 += 2.0 * eccentricity * j_plus_1 - j_plus_2
        term2 = j_minus_2 - 2.0 * j_0 + j_plus_2
        term3 = j_0

        power_coeff = (harmonic**4 / 32.0) * (
            term1**2
            + (1.0 - eccentricity**2) * term2**2
            + (4.0 / (3.0 * harmonic**2)) * term3**2
        )
        return np.sqrt(np.maximum(power_coeff, 0.0)) / harmonic

    def _cloud_amplitude(self):
        prefactor = (4.0 * self.G) / (self.c**4 * self.d_L)
        quadrupole_scale = self.Mc_max * self.r_c**2
        return prefactor * self.transition_omega**2 * quadrupole_scale * self.geom_factor

    def _binary_strain_time_domain(self, semi_major_axis, eccentricity, mean_anomaly):
        """
        Reconstruct the eccentric binary waveform directly from the Kepler orbit.

        This avoids the Gibbs-like sawtooth artifact that appears when only a finite
        number of harmonic comb teeth are kept in the time-domain synthesis.
        """
        mean_anomaly = np.mod(mean_anomaly, 2.0 * np.pi)
        eccentricity = np.clip(np.asarray(eccentricity, dtype=float), 0.0, 0.999)
        semi_major_axis = np.asarray(semi_major_axis, dtype=float)

        eccentric_anomaly = solve_kepler(mean_anomaly, eccentricity)
        cos_e = np.cos(eccentric_anomaly)
        sin_e = np.sin(eccentric_anomaly)
        one_minus_e2 = np.maximum(1.0e-14, 1.0 - eccentricity**2)
        radial_factor = np.maximum(1.0e-14, 1.0 - eccentricity * cos_e)

        x = semi_major_axis * (cos_e - eccentricity)
        y = semi_major_axis * np.sqrt(one_minus_e2) * sin_e
        radius = semi_major_axis * radial_factor

        mean_motion = np.sqrt(self.G * self.M_tot / semi_major_axis**3)
        e_dot = mean_motion / radial_factor
        x_dot = -semi_major_axis * sin_e * e_dot
        y_dot = semi_major_axis * np.sqrt(one_minus_e2) * cos_e * e_dot

        acc_factor = -self.G * self.M_tot / radius**3
        x_ddot = acc_factor * x
        y_ddot = acc_factor * y

        reduced_mass = self.M * self.M_star / self.M_tot
        i_ddot_xx = 2.0 * reduced_mass * (x_dot**2 + x * x_ddot)
        i_ddot_yy = 2.0 * reduced_mass * (y_dot**2 + y * y_ddot)

        prefactor = self.G / (self.c**4 * self.d_L)
        return prefactor * (i_ddot_yy - i_ddot_xx)

    def _evaluate_waveform_window_on_grid(self, orbit, cloud, t_zoom):
        a_zoom, e_zoom, phi_zoom = orbit["solution"].sol(t_zoom)
        cg_r, cg_i, ce_r, ce_i = cloud["solution"].sol(t_zoom)
        cg_zoom = cg_r + 1j * cg_i
        ce_zoom = ce_r + 1j * ce_i
        coherence = self._coherence_diagnostics(cg_zoom, ce_zoom)
        overlap_zoom = coherence["overlap"]

        h_back = self._binary_strain_time_domain(a_zoom, e_zoom, phi_zoom)
        cloud_phase = self.transition_omega * t_zoom
        h_axion = -self._cloud_amplitude() * (
            overlap_zoom.real * np.cos(cloud_phase) - overlap_zoom.imag * np.sin(cloud_phase)
        )

        orbital_omega_res = np.interp(cloud["resonance_time"], orbit["t"], orbit["omega"])
        orbital_period_res = 2.0 * np.pi / orbital_omega_res
        resonance_time_obs = cloud["resonance_time"] * (1.0 + self.z)
        t_obs = t_zoom * (1.0 + self.z)

        return {
            "t_source": t_zoom,
            "t_obs": t_obs,
            "t_rel_obs": t_obs - resonance_time_obs,
            "h_back": h_back,
            "h_binary": h_back,
            "h_axion": h_axion,
            "h_total": h_back + h_axion,
            "overlap_abs": coherence["overlap_abs"],
            "population_norm": coherence["population_norm"],
            "overlap_abs_normalized": coherence["overlap_abs_normalized"],
            "phi": phi_zoom,
            "e": e_zoom,
            "orbital_period_res": orbital_period_res,
            "orbital_frequency_res_obs": orbital_omega_res / (2.0 * np.pi * (1.0 + self.z)),
            "resonance_time_obs": resonance_time_obs,
        }

    def _evaluate_binary_window_on_grid(self, orbit, t_zoom, resonance_time_source):
        a_zoom, e_zoom, phi_zoom = orbit["solution"].sol(t_zoom)
        h_pure = self._binary_strain_time_domain(a_zoom, e_zoom, phi_zoom)
        orbital_omega_res = np.interp(resonance_time_source, orbit["t"], orbit["omega"])
        orbital_period_res = 2.0 * np.pi / orbital_omega_res
        resonance_time_obs = resonance_time_source * (1.0 + self.z)
        t_obs = np.asarray(t_zoom, dtype=float) * (1.0 + self.z)
        zeros = np.zeros_like(h_pure)
        return {
            "t_source": np.asarray(t_zoom, dtype=float),
            "t_obs": t_obs,
            "t_rel_obs": t_obs - resonance_time_obs,
            "h_pure": h_pure,
            "h_binary": h_pure,
            "h_axion": zeros,
            "h_total": h_pure,
            "overlap_abs": zeros,
            "population_norm": zeros,
            "overlap_abs_normalized": zeros,
            "phi": phi_zoom,
            "e": e_zoom,
            "orbital_period_res": orbital_period_res,
            "orbital_frequency_res_obs": orbital_omega_res / (2.0 * np.pi * (1.0 + self.z)),
            "resonance_time_obs": resonance_time_obs,
        }

    def build_waveform_window(self, orbit, cloud, window_orbits=8, sample_points=2400):
        t_res = cloud["resonance_time"]
        orbital_omega_res = np.interp(t_res, orbit["t"], orbit["omega"])
        orbital_period_res = 2.0 * np.pi / orbital_omega_res
        half_window = 0.5 * window_orbits * orbital_period_res

        t_start = max(0.0, t_res - half_window)
        t_stop = min(orbit["t"][-1], t_res + half_window)
        phase_start = np.interp(t_start, orbit["t"], orbit["phi"])
        phase_stop = np.interp(t_stop, orbit["t"], orbit["phi"])
        sample_points = int(sample_points)
        phi_zoom_grid = np.linspace(phase_start, phase_stop, sample_points)
        inv_spline_t = CubicSpline(orbit["phi"], orbit["t"])
        t_zoom = inv_spline_t(phi_zoom_grid)
        return self._evaluate_waveform_window_on_grid(orbit, cloud, t_zoom)

    def build_waveform_window_for_source_interval(
        self,
        orbit,
        cloud,
        t_start_source,
        t_stop_source,
        sample_points=2400,
    ):
        t_start = max(float(orbit["t"][0]), float(t_start_source))
        t_stop = min(float(orbit["t"][-1]), float(t_stop_source))
        if t_stop <= t_start:
            t_zoom = np.array([t_start], dtype=float)
        else:
            t_zoom = np.linspace(t_start, t_stop, int(max(16, sample_points)))
        window = self._evaluate_waveform_window_on_grid(orbit, cloud, t_zoom)
        window["window_start_source"] = t_start
        window["window_stop_source"] = t_stop
        return window

    def _summary_window_events(self, cloud, max_events=2):
        target_count = int(max(1, max_events))
        event_pool = cloud.get("reference_resonance_events", None)
        if not event_pool:
            event_pool = cloud.get("resonance_events", [])
        crossed_events = [
            dict(event)
            for event in event_pool
            if event.get("crossed_zero", False) and np.isfinite(event.get("time_source", np.nan))
        ]
        crossed_events.sort(key=lambda item: float(item["time_source"]))
        selected_events = crossed_events[:target_count]

        if selected_events:
            selected_events.sort(key=lambda item: float(item["time_source"]))
            return selected_events

        fallback_events = [
            dict(event)
            for event in event_pool
            if np.isfinite(event.get("time_source", np.nan))
        ]
        if fallback_events:
            anchor_time = float(cloud["resonance_time"])
            fallback_events.sort(
                key=lambda item: (
                    abs(float(item["time_source"]) - anchor_time),
                    not bool(item.get("is_selected_resonance", False)),
                    float(item["time_source"]),
                )
            )
            for event in fallback_events:
                event_time = float(event["time_source"])
                already_used = any(
                    abs(event_time - float(existing["time_source"]))
                    <= max(1.0e-6, 1.0e-9 * max(abs(event_time), 1.0))
                    for existing in selected_events
                )
                if already_used:
                    continue
                selected_events.append(event)
                if len(selected_events) >= target_count:
                    break

        selected_events.sort(key=lambda item: float(item["time_source"]))
        return selected_events

    def build_summary_time_window(
        self,
        orbit,
        cloud,
        window_orbits=20,
        sample_points=4096,
        min_resonance_points=2,
        summary_window_hours=0.4 * 365.25 * 24.0,
    ):
        summary_events = self._summary_window_events(cloud, max_events=min_resonance_points)
        if not summary_events:
            if summary_window_hours is None:
                window = self.build_waveform_window(
                    orbit,
                    cloud,
                    window_orbits=window_orbits,
                    sample_points=sample_points,
                )
            else:
                window = self.build_waveform_window_for_source_interval(
                    orbit,
                    cloud,
                    0.0,
                    float(summary_window_hours) * 3600.0 / (1.0 + self.z),
                    sample_points=sample_points,
                )
            window["summary_resonance_events"] = []
            return window

        first_event = summary_events[0]
        last_event = summary_events[-1]
        periods = [
            float(event.get("orbital_period_source", np.nan))
            for event in summary_events
            if np.isfinite(event.get("orbital_period_source", np.nan))
        ]
        local_period = max(periods) if periods else 2.0 * np.pi / max(np.interp(first_event["time_source"], orbit["t"], orbit["omega"]), 1.0e-30)
        padding_source_s = 0.5 * float(window_orbits) * local_period
        if len(summary_events) == 1:
            padding_source_s = max(padding_source_s, 0.10 * float(orbit["t"][-1] - orbit["t"][0]))

        if summary_window_hours is None:
            t_start = float(first_event["time_source"]) - padding_source_s
            t_stop = float(last_event["time_source"]) + padding_source_s
        else:
            t_start = 0.0
            t_stop = float(summary_window_hours) * 3600.0 / (1.0 + self.z)
        duration_obs_s = max((t_stop - t_start) * (1.0 + self.z), 0.0)
        transition_frequency_obs_hz = self.transition_omega / (2.0 * np.pi * (1.0 + self.z))
        adaptive_points = int(
            min(
                40000,
                max(
                    int(sample_points),
                    np.ceil(8.0 * duration_obs_s * max(transition_frequency_obs_hz, 1.0e-6)) + 1,
                ),
            )
        )
        window = self.build_waveform_window_for_source_interval(
            orbit,
            cloud,
            t_start,
            t_stop,
            sample_points=adaptive_points,
        )
        window["summary_resonance_events"] = summary_events
        window["summary_sample_points"] = adaptive_points
        window["reference_resonance_time_source"] = float(first_event["time_source"])
        window["reference_resonance_time_obs"] = float(first_event["time_obs"])
        window["reference_resonance_harmonic"] = int(first_event["harmonic"])
        return window

    def _estimate_highest_analysis_frequency_obs(self, analysis_events):
        event_frequency_obs = [
            float(event["orbital_frequency_obs"])
            for event in analysis_events
            if np.isfinite(event.get("orbital_frequency_obs", np.nan))
        ]
        orbital_max = max(event_frequency_obs) if event_frequency_obs else 0.0
        return max(
            self.transition_omega / (2.0 * np.pi * (1.0 + self.z)),
            self.binary_harmonics * orbital_max,
        )

    def _target_fft_sample_points_for_interval(
        self,
        duration_source_s,
        analysis_events,
        minimum_points,
        oversample_per_cycle=6.0,
        point_cap=500000,
    ):
        duration_obs_s = float(duration_source_s) * (1.0 + self.z)
        highest_frequency_obs = self._estimate_highest_analysis_frequency_obs(analysis_events)
        target_points = int(np.ceil(max(1.0, oversample_per_cycle * duration_obs_s * highest_frequency_obs))) + 1
        return int(max(minimum_points, min(target_points, point_cap)))

    def build_binary_window_for_source_interval(
        self,
        orbit,
        t_start_source,
        t_stop_source,
        resonance_time_source,
        sample_points=2400,
    ):
        t_start = max(float(orbit["t"][0]), float(t_start_source))
        t_stop = min(float(orbit["t"][-1]), float(t_stop_source))
        if t_stop <= t_start:
            t_zoom = np.array([t_start], dtype=float)
        else:
            t_zoom = np.linspace(t_start, t_stop, int(max(16, sample_points)))
        window = self._evaluate_binary_window_on_grid(orbit, t_zoom, resonance_time_source)
        window["window_start_source"] = t_start
        window["window_stop_source"] = t_stop
        return window

    def build_multi_resonance_analysis_windows(
        self,
        orbit_signal,
        cloud_signal,
        orbit_template,
        sample_points=4096,
        padding_orbits=1.0,
    ):
        resonance_events = list(cloud_signal.get("resonance_events", ()))
        if not resonance_events:
            signal_window = self.build_waveform_window(orbit_signal, cloud_signal, window_orbits=8, sample_points=sample_points)
            template_window = self._evaluate_binary_window_on_grid(
                orbit_template,
                signal_window["t_source"],
                cloud_signal["resonance_time"],
            )
            fallback_event = {
                "harmonic": int(self.resonance_harmonic),
                "time_source": float(cloud_signal["resonance_time"]),
                "time_obs": float(cloud_signal["resonance_time"] * (1.0 + self.z)),
                "orbital_frequency_obs": float(signal_window["orbital_frequency_res_obs"]),
                "orbital_period_source": float(signal_window["orbital_period_res"] / (1.0 + self.z)),
                "crossed_zero": False,
            }
            signal_window["analysis_event_harmonics"] = np.array([self.resonance_harmonic], dtype=int)
            template_window["analysis_event_harmonics"] = np.array([self.resonance_harmonic], dtype=int)
            signal_window["analysis_events"] = [fallback_event]
            template_window["analysis_events"] = [dict(fallback_event)]
            return signal_window, template_window

        effective_events = self._filter_effective_resonance_events(resonance_events, require_crossing=True)
        crossed_events = [event for event in resonance_events if event["crossed_zero"]]
        analysis_events = effective_events if effective_events else []
        if not analysis_events:
            signal_window = self.build_waveform_window(
                orbit_signal,
                cloud_signal,
                window_orbits=8,
                sample_points=sample_points,
            )
            template_window = self._evaluate_binary_window_on_grid(
                orbit_template,
                signal_window["t_source"],
                cloud_signal["resonance_time"],
            )
            fallback_event = min(
                crossed_events if crossed_events else resonance_events,
                key=lambda item: abs(float(item["time_source"]) - float(cloud_signal["resonance_time"])),
            )
            signal_window["analysis_event_harmonics"] = np.array([fallback_event["harmonic"]], dtype=int)
            template_window["analysis_event_harmonics"] = np.array([fallback_event["harmonic"]], dtype=int)
            signal_window["analysis_events"] = [dict(fallback_event)]
            template_window["analysis_events"] = [dict(fallback_event)]
            signal_window["analysis_sample_points"] = int(max(16, sample_points))
            template_window["analysis_sample_points"] = int(max(16, sample_points))
            return signal_window, template_window
        first_event = analysis_events[0]
        last_event = analysis_events[-1]
        t_start = first_event["time_source"] - float(padding_orbits) * first_event["orbital_period_source"]
        t_stop = last_event["time_source"] + float(padding_orbits) * last_event["orbital_period_source"]
        adaptive_points = self._target_fft_sample_points_for_interval(
            t_stop - t_start,
            analysis_events,
            minimum_points=int(max(16, sample_points)),
        )
        signal_window = self.build_waveform_window_for_source_interval(
            orbit_signal,
            cloud_signal,
            t_start,
            t_stop,
            sample_points=adaptive_points,
        )
        template_window = self.build_binary_window_for_source_interval(
            orbit_template,
            signal_window["window_start_source"],
            signal_window["window_stop_source"],
            cloud_signal["resonance_time"],
            sample_points=adaptive_points,
        )
        event_harmonics = np.asarray([event["harmonic"] for event in analysis_events], dtype=int)
        signal_window["analysis_event_harmonics"] = event_harmonics
        template_window["analysis_event_harmonics"] = event_harmonics
        signal_window["analysis_events"] = [dict(event) for event in analysis_events]
        template_window["analysis_events"] = [dict(event) for event in analysis_events]
        signal_window["analysis_sample_points"] = adaptive_points
        template_window["analysis_sample_points"] = adaptive_points
        return signal_window, template_window

    def _select_analysis_events_for_interval(self, cloud_signal, t_start_source, t_stop_source):
        resonance_events = list(cloud_signal.get("resonance_events", ()))
        selected = [
            dict(event)
            for event in resonance_events
            if float(t_start_source) <= float(event["time_source"]) <= float(t_stop_source)
        ]
        selected = self._filter_effective_resonance_events(selected, require_crossing=True)
        if selected:
            return selected

        effective_events = self._filter_effective_resonance_events(resonance_events, require_crossing=True)
        if effective_events:
            midpoint = 0.5 * (float(t_start_source) + float(t_stop_source))
            closest_event = min(effective_events, key=lambda item: abs(float(item["time_source"]) - midpoint))
            return [dict(closest_event)]
        if resonance_events:
            midpoint = 0.5 * (float(t_start_source) + float(t_stop_source))
            closest_event = min(resonance_events, key=lambda item: abs(float(item["time_source"]) - midpoint))
            return [dict(closest_event)]
        return []

    def solve_local_harmonic_windows(
        self,
        orbit,
        cloud,
        padding_orbits=1.5,
        sample_points=1024,
    ):
        resonance_events = self._filter_effective_resonance_events(cloud.get("resonance_events", []), require_crossing=True)
        if not resonance_events:
            return []

        orbit_t = np.asarray(orbit["t"], dtype=float)
        spline_a = CubicSpline(orbit_t, orbit["a"])
        spline_e = CubicSpline(orbit_t, orbit["e"])
        spline_omega = CubicSpline(orbit_t, orbit["omega"])
        orbit_phi = np.asarray(orbit["phi"], dtype=float)

        outputs = []
        for event in resonance_events:
            harmonic = int(event["harmonic"])
            half_window = 0.5 * float(padding_orbits) * event["orbital_period_source"]
            t_start = max(float(orbit_t[0]), event["time_source"] - half_window)
            t_stop = min(float(orbit_t[-1]), event["time_source"] + half_window)
            if t_stop <= t_start:
                continue

            def rhs(t_val, y_val):
                a_val = float(spline_a(t_val))
                e_val = float(spline_e(t_val))
                omega_val = float(spline_omega(t_val))
                phi_val = float(np.interp(t_val, orbit_t, orbit_phi))
                cg = y_val[0] + 1j * y_val[1]
                ce_tilde = y_val[2] + 1j * y_val[3]

                eta_vec = self._eta_vector(a_val, e_val)
                eta_scalar = eta_vec[self.harmonic_to_index[harmonic]] * np.exp(
                    -1j * (harmonic - self.resonance_harmonic) * phi_val
                )
                detuning = harmonic * omega_val - self.transition_omega
                d_cg = -1j * (0.5 * detuning * cg + eta_scalar * ce_tilde)
                d_ce = -1j * (np.conj(eta_scalar) * cg - 0.5 * detuning * ce_tilde) - self.Gamma_decay * ce_tilde
                return [d_cg.real, d_cg.imag, d_ce.real, d_ce.imag]

            local_sol = solve_ivp(
                rhs,
                (t_start, t_stop),
                [1.0, 0.0, 0.0, 0.0],
                dense_output=True,
                max_step=min((t_stop - t_start) / 250.0, event["orbital_period_source"] / 32.0),
                method=self.solver_method,
                rtol=self.solver_rtol,
                atol=1.0e-12 if self.solver_profile == "fast" else 1.0e-14,
            )

            t_eval = np.linspace(t_start, t_stop, int(max(64, sample_points)))
            cg_r, cg_i, ce_r, ce_i = local_sol.sol(t_eval)
            cg = cg_r + 1j * cg_i
            ce_tilde = ce_r + 1j * ce_i
            coherence = self._coherence_diagnostics(cg, ce_tilde)
            overlap_abs = coherence["overlap_abs"]

            outputs.append(
                {
                    "harmonic": harmonic,
                    "time_source": event["time_source"],
                    "window_start_source": t_start,
                    "window_stop_source": t_stop,
                    "peak_excited_population": float(np.max(np.abs(ce_tilde) ** 2)),
                    "peak_overlap_abs": float(np.max(overlap_abs)),
                    "integrated_overlap_abs_s": float(np.trapezoid(overlap_abs, t_eval)),
                    "solution": local_sol,
                }
            )

        return outputs

    def build_common_start_windows(
        self,
        orbit_signal,
        cloud_signal,
        orbit_template,
        duration_obs_s,
        sample_points=2400,
        start_time_source=0.0,
    ):
        """Build start-aligned windows for the physical signal and the pure-binary template."""
        start_time_source = max(0.0, float(start_time_source))
        duration_source_s = float(duration_obs_s) / (1.0 + self.z)
        t_stop = min(orbit_signal["t"][-1], orbit_template["t"][-1], start_time_source + duration_source_s)
        if t_stop <= start_time_source:
            t_zoom = np.array([start_time_source], dtype=float)
        else:
            t_zoom = np.linspace(start_time_source, t_stop, int(max(16, sample_points)))
        signal_window = self._evaluate_waveform_window_on_grid(orbit_signal, cloud_signal, t_zoom)
        template_window = self._evaluate_binary_window_on_grid(
            orbit_template,
            t_zoom,
            cloud_signal["resonance_time"],
        )
        signal_window["window_start_source"] = float(t_zoom[0])
        signal_window["window_stop_source"] = float(t_zoom[-1])
        template_window["window_start_source"] = float(t_zoom[0])
        template_window["window_stop_source"] = float(t_zoom[-1])
        analysis_events = self._select_analysis_events_for_interval(cloud_signal, t_zoom[0], t_zoom[-1])
        event_harmonics = np.asarray([event["harmonic"] for event in analysis_events], dtype=int)
        signal_window["analysis_events"] = analysis_events
        template_window["analysis_events"] = [dict(event) for event in analysis_events]
        signal_window["analysis_event_harmonics"] = event_harmonics
        template_window["analysis_event_harmonics"] = event_harmonics
        return signal_window, template_window

    def _suggest_max_display_hz(self, f_transition_obs_hz, f_orb_res_obs_hz, freq_hz_max):
        # Keep the displayed spectrum wide enough to contain both the resonance peak
        # and the retained binary harmonic comb, instead of hard-clipping at 35 mHz.
        highest_feature_hz = max(
            float(f_transition_obs_hz),
            float(self.binary_harmonics) * float(f_orb_res_obs_hz),
        )
        return min(float(freq_hz_max), 1.25 * highest_feature_hz)

    def build_windowed_fft(self, waveform_window, pad_factor=8, tukey_alpha=0.03):
        # 频域分析统一在这里做：
        # 1. 先去均值，抑制低频泄漏
        # 2. mismatch 和展示图都用 Tukey 窗
        # 3. rFFT 乘 dt，近似连续傅里叶变换
        t_obs = waveform_window["t_obs"]
        dt_obs = t_obs[1] - t_obs[0]
        n_samples = len(t_obs)
        n_fft = max(n_samples, int(pad_factor * n_samples))

        # Remove DC offsets before FFT to suppress low-frequency leakage into the ~10 mHz band.
        h_back = np.asarray(waveform_window.get("h_back", waveform_window["h_binary"]), dtype=float)
        h_pure = np.asarray(waveform_window.get("h_pure", h_back), dtype=float)
        h_back_centered = h_back - np.mean(h_back)
        h_pure_centered = h_pure - np.mean(h_pure)
        h_axion_centered = waveform_window["h_axion"] - np.mean(waveform_window["h_axion"])
        h_total_centered = waveform_window["h_total"] - np.mean(waveform_window["h_total"])

        mismatch_window = tukey(n_samples, alpha=tukey_alpha)
        spectrum_window = tukey(n_samples, alpha=tukey_alpha)

        h_back_tapered = h_back_centered * mismatch_window
        h_pure_tapered = h_pure_centered * mismatch_window
        h_axion_tapered = h_axion_centered * mismatch_window
        h_total_tapered = h_total_centered * mismatch_window

        h_back_spectrum = h_back_centered * spectrum_window
        h_pure_spectrum = h_pure_centered * spectrum_window
        h_axion_spectrum = h_axion_centered * spectrum_window
        h_total_spectrum = h_total_centered * spectrum_window

        freq_hz = np.fft.rfftfreq(n_fft, dt_obs)
        df_hz = freq_hz[1] - freq_hz[0]

        h_tilde_back = np.fft.rfft(h_back_tapered, n=n_fft) * dt_obs
        h_tilde_pure = np.fft.rfft(h_pure_tapered, n=n_fft) * dt_obs
        h_tilde_axion = np.fft.rfft(h_axion_tapered, n=n_fft) * dt_obs
        h_tilde_total = np.fft.rfft(h_total_tapered, n=n_fft) * dt_obs

        h_tilde_back_plot = np.fft.rfft(h_back_spectrum, n=n_fft) * dt_obs
        h_tilde_pure_plot = np.fft.rfft(h_pure_spectrum, n=n_fft) * dt_obs
        h_tilde_axion_plot = np.fft.rfft(h_axion_spectrum, n=n_fft) * dt_obs
        h_tilde_total_plot = np.fft.rfft(h_total_spectrum, n=n_fft) * dt_obs

        h_c_back = 2.0 * freq_hz * np.abs(h_tilde_back_plot)
        h_c_pure = 2.0 * freq_hz * np.abs(h_tilde_pure_plot)
        h_c_axion = 2.0 * freq_hz * np.abs(h_tilde_axion_plot)
        h_c_total = 2.0 * freq_hz * np.abs(h_tilde_total_plot)

        f_transition_obs_hz = self.transition_omega / (2.0 * np.pi * (1.0 + self.z))
        f_orb_res_obs_hz = waveform_window["orbital_frequency_res_obs"]
        max_display_hz = self._suggest_max_display_hz(
            f_transition_obs_hz,
            f_orb_res_obs_hz,
            freq_hz[-1],
        )

        return {
            "t_obs": t_obs,
            "dt_obs": dt_obs,
            "df_hz": df_hz,
            "n_fft": n_fft,
            "match_window": mismatch_window,
            "spectrum_window": spectrum_window,
            "match_tukey_alpha": tukey_alpha,
            "freq_hz": freq_hz,
            "h_tilde_back": h_tilde_back,
            "h_tilde_binary": h_tilde_back,
            "h_tilde_pure": h_tilde_pure,
            "h_tilde_axion": h_tilde_axion,
            "h_tilde_total": h_tilde_total,
            "h_tilde_back_plot": h_tilde_back_plot,
            "h_tilde_binary_plot": h_tilde_back_plot,
            "h_tilde_pure_plot": h_tilde_pure_plot,
            "h_tilde_axion_plot": h_tilde_axion_plot,
            "h_tilde_total_plot": h_tilde_total_plot,
            "h_c_back": h_c_back,
            "h_c_binary": h_c_back,
            "h_c_pure": h_c_pure,
            "h_c_axion": h_c_axion,
            "h_c_total": h_c_total,
            "f_transition_obs_hz": f_transition_obs_hz,
            "f_orb_res_obs_hz": f_orb_res_obs_hz,
            "max_display_hz": max_display_hz,
            "analysis_events": [dict(event) for event in waveform_window.get("analysis_events", [])],
            "analysis_event_harmonics": np.asarray(waveform_window.get("analysis_event_harmonics", np.array([], dtype=int)), dtype=int),
        }

    def build_frequency_spectrum(self, waveform_window, pad_factor=64): # <--- pad_factor 默认值拉高到 64
        t_obs = waveform_window["t_obs"]
        dt_obs = t_obs[1] - t_obs[0]
        n_samples = len(t_obs)
        
        n_fft = int(pad_factor * n_samples)

        # 引入 Kaiser 窗
        taper = kaiser(n_samples, beta=14.0)

        # 加窗：只对真实数据段加窗
        h_back = waveform_window.get("h_back", waveform_window["h_binary"]) * taper
        h_axion = waveform_window["h_axion"] * taper
        h_total = waveform_window["h_total"] * taper

        # 执行 FFT：numpy 的 rfft 发现 n_fft > len(h_binary) 时，会自动在尾部补零
        freq_hz = np.fft.rfftfreq(n_fft, dt_obs)
        spec_back = np.abs(np.fft.rfft(h_back, n=n_fft))
        spec_axion = np.abs(np.fft.rfft(h_axion, n=n_fft))
        spec_total = np.abs(np.fft.rfft(h_total, n=n_fft))

        # 归一化处理
        normalization = max(np.max(spec_back), np.max(spec_total), 1.0e-30)
        
        # ... 后续的频率范围计算和返回字典保持不变 ...
        f_transition_obs_hz = self.transition_omega / (2.0 * np.pi * (1.0 + self.z))
        f_orb_res_obs_hz = waveform_window["orbital_frequency_res_obs"]
        max_display_hz = self._suggest_max_display_hz(
            f_transition_obs_hz,
            f_orb_res_obs_hz,
            freq_hz[-1],
        )

        return {
            "freq_hz": freq_hz,
            "spec_back": spec_back / normalization,
            "spec_binary": spec_back / normalization,
            "spec_axion": spec_axion / normalization,
            "spec_total": spec_total / normalization,
            "f_transition_obs_hz": f_transition_obs_hz,
            "f_orb_res_obs_hz": f_orb_res_obs_hz,
            "max_display_hz": max_display_hz,
        }

    def _detector_curve_to_strain_psd(self, detector_name, freq_hz, raw_value, curve_kind):
        curve_kind = str(curve_kind).lower()
        freq_hz = np.asarray(freq_hz, dtype=float)
        raw_value = np.asarray(raw_value, dtype=float)

        if curve_kind in {"omega_gw", "omega", "omega_density"}:
            # Convert Omega_GW sensitivity to the one-sided strain PSD used in <h|h>.
            # h_c^2 = 3 H0^2 Omega_GW / (pi^2 f^2), and S_n(f) = h_c^2 / f.
            h_char_sq = 3.0 * self.H0**2 * raw_value / (np.pi**2 * np.maximum(freq_hz, 1.0e-30) ** 2)
            return np.maximum(h_char_sq / np.maximum(freq_hz, 1.0e-30), 1.0e-60)
        if curve_kind in {"characteristic_strain", "hc", "h_char"}:
            return np.maximum(raw_value**2 / np.maximum(freq_hz, 1.0e-30), 1.0e-60)
        if curve_kind in {"asd", "amplitude_spectral_density"}:
            return np.maximum(raw_value**2, 1.0e-60)
        if curve_kind in {"psd", "strain_psd"}:
            return np.maximum(raw_value, 1.0e-60)
        raise ValueError(f"Unknown detector curve kind '{curve_kind}' for {detector_name}.")

    def _load_detector_noise_curve(self, detector):
        detector_name = str(detector).upper()
        curve_kind = self.detector_curve_kinds.get(detector_name)
        if curve_kind is None:
            return None

        cache_key = (detector_name, curve_kind, round(float(self.H0), 30))
        with self.__class__._CACHE_LOCK:
            cached = self.__class__._DETECTOR_NOISE_CACHE.get(cache_key)
            if cached is not None:
                return cached

            curve_path = SCRIPT_DIR / f"{detector_name}.csv"
            if not curve_path.exists():
                return None

            data = np.loadtxt(curve_path, delimiter=",")
            data = np.asarray(data, dtype=float)
            if data.ndim != 2 or data.shape[1] < 2:
                return None

            freq = np.asarray(data[:, 0], dtype=float)
            raw_value = np.asarray(data[:, 1], dtype=float)
            mask = np.isfinite(freq) & np.isfinite(raw_value) & (freq > 0.0) & (raw_value > 0.0)
            if np.count_nonzero(mask) < 2:
                return None

            order = np.argsort(freq[mask])
            freq_sorted = freq[mask][order]
            raw_sorted = raw_value[mask][order]
            psd_sorted = self._detector_curve_to_strain_psd(detector_name, freq_sorted, raw_sorted, curve_kind)
            curve = {
                "freq_hz": freq_sorted,
                "raw_value": raw_sorted,
                "psd": psd_sorted,
                "kind": curve_kind,
                "path": curve_path,
            }
            self.__class__._DETECTOR_NOISE_CACHE[cache_key] = curve
            return curve

    def _detector_band_hz(self, detector):
        detector_name = str(detector).upper()
        curve = self._load_detector_noise_curve(detector_name)
        if curve is not None:
            return float(curve["freq_hz"][0]), float(curve["freq_hz"][-1])
        return self.DETECTOR_BANDS_HZ[detector_name]

    def build_detector_psd(self, detector, f_hz):
        detector_name = detector.lower()
        safe_f = np.maximum(np.asarray(f_hz, dtype=float), 1.0e-8)

        curve = self._load_detector_noise_curve(detector)
        if curve is not None:
            freq_curve = curve["freq_hz"]
            psd_curve = curve["psd"]
            log_psd = np.interp(
                np.log10(safe_f),
                np.log10(freq_curve),
                np.log10(psd_curve),
                left=np.log10(psd_curve[0]),
                right=np.log10(psd_curve[-1]),
            )
            return np.maximum(10.0**log_psd, 1.0e-60)

        if detector_name == "decigo":
            f_p = 7.36
            psd = (
                7.05e-48 * (1.0 + (safe_f / f_p) ** 2)
                + 4.8e-51 * safe_f ** (-4) / (1.0 + (safe_f / f_p) ** 2)
                + 5.33e-52 * safe_f ** (-4)
            )
            return np.maximum(psd, 1.0e-60)
        elif detector_name == "et":
            x = safe_f / 100.0
            asd = 1.0e-25 * (
                2.39e-27 * x ** (-15.64)
                + 0.349 * x ** (-2.145)
                + 1.76 * x ** (-0.12)
                + 0.409 * x ** 1.10
            )
            return np.maximum(asd**2, 1.0e-60)
        elif detector_name == "ce":
            freq_anchor = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000], dtype=float)
            asd_anchor = np.array([4e-23, 1e-24, 1.5e-25, 2.5e-26, 8e-27, 6e-27, 8e-27, 1.6e-26, 4.5e-26, 1.5e-25])
            asd = 10.0 ** np.interp(np.log10(safe_f), np.log10(freq_anchor), np.log10(asd_anchor))
            return np.maximum(asd**2, 1.0e-60)
        else:
            raise ValueError(f"Unknown detector '{detector}'.")

    def _save_figure(self, fig, stem):
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for fmt in self.figure_formats:
            out_path = self.figure_dir / f"{stem}.{fmt}"
            try:
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                saved_paths.append(out_path)
            except PermissionError:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                fallback_path = self.figure_dir / f"{stem}_{timestamp}.{fmt}"
                fig.savefig(fallback_path, dpi=300, bbox_inches="tight")
                saved_paths.append(fallback_path)
        print("Saved figure:", ", ".join(str(path) for path in saved_paths))

    def _finalize_figure(self, fig, stem):
        self._save_figure(fig, stem)
        if "agg" in plt.get_backend().lower():
            plt.close(fig)
        else:
            plt.show()

    def _time_series_stem(self, kind):
        return f"{self._module_stem}_axion_strain_{kind}_{self._direction_tag}"

    def _mismatch_figure_stem(self):
        return f"{self._module_stem}_decigo_mismatch_{self._direction_tag}"

    def _mismatch_data_stem(self, detector):
        return f"{self._module_stem}_{str(detector).lower()}_mismatch_{self._direction_tag}"

    def _summary_figure_stem(self):
        return f"{self._module_stem}_orbit_time_summary_{self._direction_tag}"

    def _state_text(self, state):
        return "".join(map(str, state))

    def _save_time_series_data(self, waveform_window, cloud, stem, series_label):
        if self.time_series_data_dir is None:
            return None

        self.time_series_data_dir.mkdir(parents=True, exist_ok=True)
        t_source = np.asarray(waveform_window["t_source"], dtype=float)
        t_obs = np.asarray(waveform_window["t_obs"], dtype=float)
        t_rel_obs = np.asarray(waveform_window["t_rel_obs"], dtype=float)
        h_back = np.asarray(
            waveform_window.get("h_back", waveform_window.get("h_binary", np.zeros_like(t_source))),
            dtype=float,
        )
        h_axion = np.asarray(waveform_window["h_axion"], dtype=float)
        h_total = np.asarray(waveform_window["h_total"], dtype=float)
        overlap_abs = np.asarray(
            waveform_window.get("overlap_abs", np.full_like(h_axion, np.nan)),
            dtype=float,
        )
        population_norm = np.asarray(
            waveform_window.get("population_norm", np.full_like(h_axion, np.nan)),
            dtype=float,
        )

        valid = (
            np.isfinite(t_source)
            & np.isfinite(t_obs)
            & np.isfinite(t_rel_obs)
            & np.isfinite(h_back)
            & np.isfinite(h_axion)
            & np.isfinite(h_total)
            & np.isfinite(overlap_abs)
            & np.isfinite(population_norm)
        )
        table = np.column_stack(
            (
                t_source[valid],
                t_obs[valid],
                t_rel_obs[valid],
                t_rel_obs[valid] / 3600.0,
                h_back[valid],
                h_axion[valid],
                h_total[valid],
                overlap_abs[valid],
                population_norm[valid],
            )
        )

        event_harmonics = np.asarray(
            waveform_window.get("analysis_event_harmonics", np.array([], dtype=int)),
            dtype=int,
        )
        header_lines = [
            f"module={self._module_stem}",
            f"series={series_label}",
            f"transition_family={self.transition_family}",
            f"transition_states=|{''.join(map(str, self.transition_solver_data['initial_state']))}> -> |{''.join(map(str, self.transition_solver_data['final_state']))}>",
            f"redshift={self.z:.16e}",
            f"luminosity_distance_m={self.d_L:.16e}",
            f"primary_mass_msun={self.M / self.M_sun:.16e}",
            f"secondary_mass_msun={self.M_star / self.M_sun:.16e}",
            f"transition_frequency_obs_hz={self.transition_omega / (2.0 * np.pi * (1.0 + self.z)):.16e}",
            f"selected_resonance_time_obs_s={cloud['resonance_time'] * (1.0 + self.z):.16e}",
            "analysis_event_harmonics=" + ",".join(str(n) for n in event_harmonics),
            "columns=t_source_s t_obs_s t_rel_obs_s t_rel_obs_hr h_back h_axion h_total overlap_abs population_norm",
        ]
        out_path = self.time_series_data_dir / f"{stem}.txt"
        np.savetxt(out_path, table, header="\n".join(header_lines), comments="# ")
        print(f"Saved time-series data: {out_path}")
        return out_path

    def _save_mismatch_time_series_data(self, mismatch_time_series):
        if self.mismatch_data_dir is None:
            return {}

        self.mismatch_data_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}
        for detector, series in mismatch_time_series.items():
            observation_years = np.asarray(series.get("observation_years", []), dtype=float)
            if observation_years.size == 0:
                continue

            def column(key, default=np.nan):
                values = np.asarray(
                    series.get(key, np.full(observation_years.shape, default, dtype=float)),
                    dtype=float,
                )
                if values.shape == observation_years.shape:
                    return values
                if values.size == 1:
                    return np.full(observation_years.shape, float(values[0]), dtype=float)
                resized = np.full(observation_years.shape, default, dtype=float)
                n_copy = min(values.size, observation_years.size)
                resized[:n_copy] = values[:n_copy]
                return resized

            effective_observation_years = column("effective_observation_years")
            mismatch = column("mismatch")
            raw_mismatch = column("raw_mismatch")
            distinguishability_threshold = column("distinguishability_threshold")
            snr = column("snr")
            table = np.column_stack(
                (
                    observation_years,
                    effective_observation_years,
                    mismatch,
                    raw_mismatch,
                    distinguishability_threshold,
                    snr,
                )
            )

            initial_state = self.transition_solver_data["initial_state"]
            final_state = self.transition_solver_data["final_state"]
            analysis_band_hz = series.get("analysis_band_hz", (np.nan, np.nan))
            header_lines = [
                f"module={self._module_stem}",
                f"detector={detector}",
                f"transition_family={self.transition_family}",
                f"transition_states=|{self._state_text(initial_state)}> -> |{self._state_text(final_state)}>",
                f"direction={self._direction_tag}",
                f"method={series.get('method', 'unknown')}",
                f"redshift={self.z:.16e}",
                f"luminosity_distance_m={self.d_L:.16e}",
                f"primary_mass_msun={self.M / self.M_sun:.16e}",
                f"secondary_mass_msun={self.M_star / self.M_sun:.16e}",
                f"alpha={self.alpha:.16e}",
                f"transition_frequency_obs_hz={self.transition_omega / (2.0 * np.pi * (1.0 + self.z)):.16e}",
                f"selected_resonance_harmonic={self.resonance_harmonic}",
                f"available_observation_years={float(series.get('available_observation_years', np.nan)):.16e}",
                f"alignment_duration_s={float(series.get('alignment_duration_s', np.nan)):.16e}",
                f"alignment_time_shift_s={float(series.get('alignment_time_shift_s', np.nan)):.16e}",
                f"alignment_phase_shift_rad={float(series.get('alignment_phase_shift_rad', np.nan)):.16e}",
                f"analysis_band_low_hz={float(analysis_band_hz[0]):.16e}",
                f"analysis_band_high_hz={float(analysis_band_hz[1]):.16e}",
                "columns=observation_years effective_observation_years mismatch raw_mismatch distinguishability_threshold snr",
            ]
            out_path = self.mismatch_data_dir / f"{self._mismatch_data_stem(detector)}.txt"
            np.savetxt(out_path, table, header="\n".join(header_lines), comments="# ")
            print(f"Saved mismatch data: {out_path}")
            saved_paths[detector] = out_path
        return saved_paths

    def _observer_days(self, t_source):
        return np.asarray(t_source, dtype=float) * (1.0 + self.z) / 86400.0

    def _build_mismatch_observation_years(self, duration_yr, mismatch_max_years, mismatch_time_samples):
        year_stop = max(min(mismatch_max_years, duration_yr), 1.0e-8)
        year_start = max(year_stop / max(int(mismatch_time_samples), 2), 1.0e-8)
        if mismatch_time_samples <= 1 or np.isclose(year_start, year_stop, rtol=1.0e-12, atol=0.0):
            return np.array([year_stop], dtype=float)
        return np.linspace(year_start, year_stop, int(mismatch_time_samples))

    def _compute_analysis_band_hz(self, detector, frequency_domain_signal, frequency_domain_template):
        freq_hz = frequency_domain_signal["freq_hz"]
        df_hz = frequency_domain_signal["df_hz"]
        f_res_low = min(
            frequency_domain_signal["f_transition_obs_hz"],
            frequency_domain_template["f_transition_obs_hz"],
        )
        f_res_high = max(
            frequency_domain_signal["f_transition_obs_hz"],
            frequency_domain_template["f_transition_obs_hz"],
        )
        f_low, f_high = self._detector_band_hz(detector)
        detector_band_low = max(f_low, df_hz)
        detector_band_high = min(f_high, freq_hz[-1])
        if self.match_band_strategy == "detector":
            return float(detector_band_low), float(detector_band_high), float(f_res_high)

        analysis_events = list(frequency_domain_signal.get("analysis_events", [])) + list(
            frequency_domain_template.get("analysis_events", [])
        )
        orbital_frequency_events = np.asarray(
            [
                float(event["orbital_frequency_obs"])
                for event in analysis_events
                if np.isfinite(event.get("orbital_frequency_obs", np.nan))
            ],
            dtype=float,
        )

        if orbital_frequency_events.size == 0:
            f_orb_res = max(
                frequency_domain_signal["f_orb_res_obs_hz"],
                frequency_domain_template["f_orb_res_obs_hz"],
            )
            band_half_width = self.match_band_orbital_half_width * f_orb_res
            band_low = max(detector_band_low, f_res_low - band_half_width)
            band_high = min(detector_band_high, f_res_high + band_half_width)
            if band_high <= band_low:
                return float(detector_band_low), float(detector_band_high), float(f_orb_res)
            return float(band_low), float(band_high), float(f_orb_res)

        orbital_frequency_min = float(np.min(orbital_frequency_events))
        orbital_frequency_max = float(np.max(orbital_frequency_events))
        band_half_width = self.match_band_orbital_half_width * orbital_frequency_max
        band_low = max(detector_band_low, orbital_frequency_min - 0.5 * band_half_width)
        highest_feature_hz = max(
            f_res_high,
            self.binary_harmonics * orbital_frequency_max,
        )
        band_high = min(detector_band_high, highest_feature_hz + band_half_width)
        if band_high <= band_low:
            return float(detector_band_low), float(detector_band_high), orbital_frequency_max
        return float(band_low), float(band_high), orbital_frequency_max

    def _analysis_mask_or_raise(self, detector, freq_hz, band_low, band_high):
        analysis_mask = (freq_hz >= band_low) & (freq_hz <= band_high)
        if not np.any(analysis_mask):
            raise ValueError(
                f"No analysis frequencies available for detector {detector}; "
                f"analysis_band=[{band_low:.6e}, {band_high:.6e}] Hz, "
                f"fft_band=[{freq_hz[0]:.6e}, {freq_hz[-1]:.6e}] Hz, "
                f"df={freq_hz[1] - freq_hz[0] if freq_hz.size > 1 else np.nan:.6e} Hz, "
                f"strategy={self.match_band_strategy}."
            )
        return analysis_mask

    def _match_band_diagnostics(
        self,
        detector,
        frequency_domain_signal,
        frequency_domain_template,
        band_low,
        band_high,
        axion_norm,
        residual_norm,
    ):
        transition_frequency_obs_hz = float(frequency_domain_signal.get("f_transition_obs_hz", np.nan))
        template_transition_frequency_obs_hz = float(
            frequency_domain_template.get("f_transition_obs_hz", np.nan)
        )
        if not np.isfinite(transition_frequency_obs_hz):
            line_position = "unknown"
            line_offset_hz = np.nan
            line_in_band = False
        elif transition_frequency_obs_hz < band_low:
            line_position = "below"
            line_offset_hz = float(band_low - transition_frequency_obs_hz)
            line_in_band = False
        elif transition_frequency_obs_hz > band_high:
            line_position = "above"
            line_offset_hz = float(transition_frequency_obs_hz - band_high)
            line_in_band = False
        else:
            line_position = "in"
            line_offset_hz = 0.0
            line_in_band = True

        prefactor = self._detector_snr_prefactor(detector)
        return {
            "transition_frequency_obs_hz": transition_frequency_obs_hz,
            "template_transition_frequency_obs_hz": template_transition_frequency_obs_hz,
            "axion_line_in_band": bool(line_in_band),
            "line_band_position": line_position,
            "line_band_offset_hz": line_offset_hz,
            "axion_band_norm": float(axion_norm),
            "residual_norm": float(residual_norm),
            "residual_snr": prefactor * float(np.sqrt(max(residual_norm, 0.0))),
        }

    def _format_match_band_note(self, metrics):
        transition_frequency_obs_hz = float(metrics.get("transition_frequency_obs_hz", np.nan))
        if not np.isfinite(transition_frequency_obs_hz):
            return ""

        line_position = str(metrics.get("line_band_position", "unknown"))
        line_offset_hz = float(metrics.get("line_band_offset_hz", np.nan))
        if line_position == "in":
            line_text = f"line={transition_frequency_obs_hz * 1.0e3:.3f} mHz(in-band)"
        elif np.isfinite(line_offset_hz):
            line_text = (
                f"line={transition_frequency_obs_hz * 1.0e3:.3f} mHz"
                f"({line_position} band by {line_offset_hz * 1.0e3:.3f} mHz)"
            )
        else:
            line_text = f"line={transition_frequency_obs_hz * 1.0e3:.3f} mHz({line_position})"

        residual_snr = float(metrics.get("residual_snr", np.nan))
        if np.isfinite(residual_snr):
            line_text += f", residual-SNR={residual_snr:.3e}"
        return ", " + line_text

    def compute_detector_match_pair(
        self,
        detector,
        frequency_domain_signal,
        frequency_domain_template,
        signal_key="h_tilde_total",
        template_key="h_tilde_total",
        analysis_band_hz=None,
    ):
        """Detector-weighted overlap between two independently generated waveforms."""
        freq_hz = frequency_domain_signal["freq_hz"]
        if (
            freq_hz.shape != frequency_domain_template["freq_hz"].shape
            or not np.allclose(freq_hz, frequency_domain_template["freq_hz"])
        ):
            raise ValueError("Signal and template FFT grids must match for pairwise mismatch evaluation.")

        df_hz = frequency_domain_signal["df_hz"]
        n_fft = frequency_domain_signal["n_fft"]
        dt_obs = frequency_domain_signal["dt_obs"]
        h1 = frequency_domain_signal[signal_key]
        h2 = frequency_domain_template[template_key]

        if analysis_band_hz is None:
            band_low, band_high, _ = self._compute_analysis_band_hz(
                detector,
                frequency_domain_signal,
                frequency_domain_template,
            )
        else:
            band_low, band_high = map(float, analysis_band_hz)
        analysis_mask = self._analysis_mask_or_raise(detector, freq_hz, band_low, band_high)

        freq_band = freq_hz[analysis_mask]
        psd_band = self.build_detector_psd(detector, freq_band)
        h1_band = h1[analysis_mask]
        h2_band = h2[analysis_mask]
        h_axion_band = frequency_domain_signal["h_tilde_axion"][analysis_mask]

        norm1_density = 4.0 * np.abs(h1_band) ** 2 / psd_band
        norm2_density = 4.0 * np.abs(h2_band) ** 2 / psd_band
        norm1 = np.sum(norm1_density) * df_hz
        norm2 = np.sum(norm2_density) * df_hz
        axion_norm_density = 4.0 * np.abs(h_axion_band) ** 2 / psd_band
        axion_norm = np.sum(axion_norm_density) * df_hz

        q_band = 4.0 * df_hz * h1_band * np.conj(h2_band) / psd_band
        q_full = np.zeros(n_fft, dtype=np.complex128)
        band_indices = np.nonzero(analysis_mask)[0]
        q_full[band_indices] = q_band

        matched_filter = np.fft.fft(q_full)
        best_index = int(np.argmax(np.abs(matched_filter)))
        best_corr = matched_filter[best_index]
        t_shift = best_index * dt_obs if best_index <= n_fft // 2 else (best_index - n_fft) * dt_obs
        phase_shift = np.angle(best_corr)

        denom = np.sqrt(max(norm1 * norm2, 1.0e-60))
        faithfulness = float(np.clip(np.abs(best_corr) / denom, 0.0, 1.0))
        mismatch = float(np.clip(1.0 - faithfulness, 0.0, 1.0))
        snr = self._detector_snr_prefactor(detector) * float(np.sqrt(max(axion_norm, 0.0)))
        distinguishability_threshold = self.mismatch_threshold_d / max(2.0 * snr**2, 1.0e-60)

        aligned_h2 = h2_band * np.exp(1j * (2.0 * np.pi * freq_band * t_shift + phase_shift))
        residual_density = 4.0 * np.abs(h1_band - aligned_h2) ** 2 / psd_band
        residual_cumulative = np.cumsum(residual_density) * df_hz
        residual_norm = float(residual_cumulative[-1])
        residual_total = max(residual_cumulative[-1], 1.0e-60)
        cumulative_contribution = residual_cumulative / residual_total
        band_diagnostics = self._match_band_diagnostics(
            detector,
            frequency_domain_signal,
            frequency_domain_template,
            band_low,
            band_high,
            axion_norm,
            residual_norm,
        )

        return {
            "detector": detector,
            "freq_hz": freq_band,
            "psd": psd_band,
            "h_n": np.sqrt(freq_band * psd_band),
            "snr": snr,
            "faithfulness": faithfulness,
            "mismatch": mismatch,
            "distinguishability_threshold": distinguishability_threshold,
            "analysis_band_hz": (band_low, band_high),
            "time_shift_s": t_shift,
            "phase_shift_rad": phase_shift,
            "cumulative_contribution": cumulative_contribution,
            "signal_key": signal_key,
            "template_key": template_key,
            **band_diagnostics,
        }

    def compute_detector_match_pair_fixed_alignment(
        self,
        detector,
        frequency_domain_signal,
        frequency_domain_template,
        time_shift_s,
        phase_shift_rad,
        signal_key="h_tilde_total",
        template_key="h_tilde_total",
        analysis_band_hz=None,
    ):
        """Detector-weighted overlap using a fixed, pre-fitted global time/phase alignment."""
        freq_hz = frequency_domain_signal["freq_hz"]
        if (
            freq_hz.shape != frequency_domain_template["freq_hz"].shape
            or not np.allclose(freq_hz, frequency_domain_template["freq_hz"])
        ):
            raise ValueError("Signal and template FFT grids must match for pairwise mismatch evaluation.")

        df_hz = frequency_domain_signal["df_hz"]
        h1 = frequency_domain_signal[signal_key]
        h2 = frequency_domain_template[template_key]

        if analysis_band_hz is None:
            band_low, band_high, _ = self._compute_analysis_band_hz(
                detector,
                frequency_domain_signal,
                frequency_domain_template,
            )
        else:
            band_low, band_high = map(float, analysis_band_hz)
        analysis_mask = self._analysis_mask_or_raise(detector, freq_hz, band_low, band_high)

        freq_band = freq_hz[analysis_mask]
        psd_band = self.build_detector_psd(detector, freq_band)
        h1_band = h1[analysis_mask]
        h2_band = h2[analysis_mask]
        h_axion_band = frequency_domain_signal["h_tilde_axion"][analysis_mask]

        norm1_density = 4.0 * np.abs(h1_band) ** 2 / psd_band
        norm2_density = 4.0 * np.abs(h2_band) ** 2 / psd_band
        norm1 = np.sum(norm1_density) * df_hz
        norm2 = np.sum(norm2_density) * df_hz
        axion_norm_density = 4.0 * np.abs(h_axion_band) ** 2 / psd_band
        axion_norm = np.sum(axion_norm_density) * df_hz

        aligned_h2 = h2_band * np.exp(1j * (2.0 * np.pi * freq_band * time_shift_s + phase_shift_rad))
        inner = 4.0 * np.sum(h1_band * np.conj(aligned_h2) / psd_band) * df_hz
        denom = np.sqrt(max(norm1 * norm2, 1.0e-60))
        faithfulness = float(np.clip(np.abs(inner) / denom, 0.0, 1.0))
        mismatch = float(np.clip(1.0 - faithfulness, 0.0, 1.0))
        snr = self._detector_snr_prefactor(detector) * float(np.sqrt(max(axion_norm, 0.0)))
        distinguishability_threshold = self.mismatch_threshold_d / max(2.0 * snr**2, 1.0e-60)
        residual_density = 4.0 * np.abs(h1_band - aligned_h2) ** 2 / psd_band
        residual_norm = float(np.sum(residual_density) * df_hz)
        band_diagnostics = self._match_band_diagnostics(
            detector,
            frequency_domain_signal,
            frequency_domain_template,
            band_low,
            band_high,
            axion_norm,
            residual_norm,
        )

        return {
            "detector": detector,
            "freq_hz": freq_band,
            "psd": psd_band,
            "h_n": np.sqrt(freq_band * psd_band),
            "snr": snr,
            "faithfulness": faithfulness,
            "mismatch": mismatch,
            "distinguishability_threshold": distinguishability_threshold,
            "analysis_band_hz": (band_low, band_high),
            "time_shift_s": time_shift_s,
            "phase_shift_rad": phase_shift_rad,
            **band_diagnostics,
        }

    def _filter_available_detectors(self, computation_results, context):
        available = {}
        skipped = []
        for detector in self.detector_names:
            try:
                available[detector] = computation_results(detector)
            except ValueError as exc:
                skipped.append((detector, str(exc)))
        if not available:
            details = "; ".join(f"{det}: {msg}" for det, msg in skipped) or "no detector results"
            raise ValueError(f"{context}: no detectors had a usable analysis band ({details}).")
        if skipped:
            skipped_text = "; ".join(f"{det}: {msg}" for det, msg in skipped)
            print(f"[lowfre] Skipped detectors with no usable analysis band -> {skipped_text}")
        return available

    def _parallel_map_ordered(self, func, items, max_workers=None):
        items = list(items)
        if len(items) <= 1 or self.parallel_workers <= 1:
            return [func(item) for item in items]

        workers = min(max_workers or self.parallel_workers, len(items))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(func, items))

    def compute_detector_mismatch_vs_time(
        self,
        detector,
        orbit_signal,
        cloud_signal,
        orbit_template,
        observation_years,
        reference_window,
        pad_factor=8,
        tukey_alpha=0.03,
        start_time_source=0.0,
    ):
        """Cumulative mismatch(T_obs) from observation start, with a single initial fit."""
        observation_years = np.asarray(observation_years, dtype=float)
        obs_span_ref = max(reference_window["t_obs"][-1] - reference_window["t_obs"][0], 1.0e-12)
        points_per_obs_second = len(reference_window["t_obs"]) / obs_span_ref
        start_time_source = max(0.0, float(start_time_source))
        available_obs_duration = max(
            0.0,
            (min(orbit_signal["t"][-1], orbit_template["t"][-1]) - start_time_source) * (1.0 + self.z),
        )
        mismatch_fft_point_cap = 750000

        def _target_sample_points(duration_obs_s):
            requested_points = max(128, int(np.ceil(points_per_obs_second * duration_obs_s)))
            return min(requested_points, mismatch_fft_point_cap)

        alignment_duration_obs_s = min(max(observation_years[0] * self.yr, obs_span_ref / 8.0), available_obs_duration)
        projected_points = []
        effective_observation_years = []
        for obs_year in observation_years:
            duration_obs_s = min(max(obs_year * self.yr, alignment_duration_obs_s), available_obs_duration)
            effective_observation_years.append(duration_obs_s / self.yr)
            projected_points.append(max(128, int(np.ceil(points_per_obs_second * duration_obs_s))))
        max_projected_points = max(projected_points) if projected_points else 128

        if max_projected_points > mismatch_fft_point_cap:
            print(
                f"[{self._module_stem}] mismatch(T_obs) for {detector} uses exact capped FFT sampling: "
                f"requested up to {max_projected_points} samples, capped at {mismatch_fft_point_cap}."
            )

        reference_fd = self.build_windowed_fft(
            reference_window,
            pad_factor=pad_factor,
            tukey_alpha=tukey_alpha,
        )
        band_low, band_high, _ = self._compute_analysis_band_hz(detector, reference_fd, reference_fd)
        fixed_analysis_band_hz = (band_low, band_high)

        alignment_points = _target_sample_points(alignment_duration_obs_s)
        alignment_signal_window, alignment_template_window = self.build_common_start_windows(
            orbit_signal,
            cloud_signal,
            orbit_template,
            alignment_duration_obs_s,
            sample_points=alignment_points,
            start_time_source=start_time_source,
        )
        alignment_fd_signal = self.build_windowed_fft(alignment_signal_window, pad_factor=pad_factor, tukey_alpha=tukey_alpha)
        alignment_fd_template = self.build_windowed_fft(alignment_template_window, pad_factor=pad_factor, tukey_alpha=tukey_alpha)
        alignment_match = self.compute_detector_match_pair(
            detector,
            alignment_fd_signal,
            alignment_fd_template,
            signal_key="h_tilde_total",
            template_key="h_tilde_pure",
            analysis_band_hz=fixed_analysis_band_hz,
        )
        fixed_time_shift_s = alignment_match["time_shift_s"]
        fixed_phase_shift_rad = alignment_match["phase_shift_rad"]

        def worker(obs_year):
            duration_obs_s = min(max(obs_year * self.yr, alignment_duration_obs_s), available_obs_duration)
            sample_points = _target_sample_points(duration_obs_s)
            signal_window, template_window = self.build_common_start_windows(
                orbit_signal,
                cloud_signal,
                orbit_template,
                duration_obs_s,
                sample_points=sample_points,
                start_time_source=start_time_source,
            )
            signal_fd = self.build_windowed_fft(signal_window, pad_factor=pad_factor, tukey_alpha=tukey_alpha)
            template_fd = self.build_windowed_fft(template_window, pad_factor=pad_factor, tukey_alpha=tukey_alpha)
            match = self.compute_detector_match_pair_fixed_alignment(
                detector,
                signal_fd,
                template_fd,
                fixed_time_shift_s,
                fixed_phase_shift_rad,
                signal_key="h_tilde_total",
                template_key="h_tilde_pure",
                analysis_band_hz=fixed_analysis_band_hz,
            )
            return match["mismatch"], match["distinguishability_threshold"], match["snr"]

        outputs = self._parallel_map_ordered(worker, observation_years)
        raw_mismatch_values = np.asarray([item[0] for item in outputs], dtype=float)
        mismatch_values = raw_mismatch_values.copy()
        finite_mismatch = np.isfinite(raw_mismatch_values)
        if np.any(finite_mismatch):
            mismatch_values[finite_mismatch] = np.maximum.accumulate(raw_mismatch_values[finite_mismatch])
        finite_pairs = finite_mismatch[:-1] & finite_mismatch[1:]
        downward_steps = (
            raw_mismatch_values[1:][finite_pairs] - raw_mismatch_values[:-1][finite_pairs]
            if np.any(finite_pairs)
            else np.array([], dtype=float)
        )
        max_downward_correction = float(max(0.0, -np.min(downward_steps))) if downward_steps.size else 0.0
        threshold_values = np.asarray([item[1] for item in outputs], dtype=float)
        snr_values = np.asarray([item[2] for item in outputs], dtype=float)

        return {
            "detector": detector,
            "observation_years": observation_years,
            "effective_observation_years": np.asarray(effective_observation_years, dtype=float),
            "mismatch": mismatch_values,
            "raw_mismatch": raw_mismatch_values,
            "monotone_envelope_applied": bool(max_downward_correction > 0.0),
            "max_downward_correction": max_downward_correction,
            "distinguishability_threshold": threshold_values,
            "snr": snr_values,
            "available_observation_years": available_obs_duration / self.yr,
            "alignment_duration_s": alignment_duration_obs_s,
            "alignment_time_shift_s": fixed_time_shift_s,
            "alignment_phase_shift_rad": fixed_phase_shift_rad,
            "analysis_band_hz": fixed_analysis_band_hz,
            "method": "exact_fft_capped_monotone" if max_projected_points > mismatch_fft_point_cap else "exact_fft_monotone",
        }

    def compute_detector_mismatch_vs_time_or_empty_band(
        self,
        detector,
        orbit_signal,
        cloud_signal,
        orbit_template,
        observation_years,
        reference_window,
        pad_factor=8,
        tukey_alpha=0.03,
        start_time_source=0.0,
    ):
        """Return the exact detector mismatch(T_obs), or a zero observable series if no detector band exists."""
        try:
            return self.compute_detector_mismatch_vs_time(
                detector,
                orbit_signal,
                cloud_signal,
                orbit_template,
                observation_years,
                reference_window=reference_window,
                pad_factor=pad_factor,
                tukey_alpha=tukey_alpha,
                start_time_source=start_time_source,
            )
        except ValueError as exc:
            print(
                f"[{self._module_stem}] exact mismatch(T_obs) unavailable for {detector}; "
                f"marking detector mismatch as unavailable ({exc})"
            )
            observation_years = np.asarray(observation_years, dtype=float)
            nan_values = np.full_like(observation_years, np.nan, dtype=float)
            return {
                "detector": detector,
                "observation_years": observation_years,
                "effective_observation_years": observation_years,
                "mismatch": nan_values,
                "faithfulness": nan_values,
                "snr": nan_values,
                "distinguishability_threshold": nan_values,
                "available_observation_years": np.nan,
                "method": "detector_unavailable",
                "exact_error": str(exc),
            }

    def _detector_snr_prefactor(self, detector):
        detector_name = str(detector).upper()
        if detector_name == "DECIGO":
            return float(np.sqrt(1024.0 * np.pi * DECIGO_UNIT_COUNT / 15.0))
        return float(np.sqrt(max(self.transition_geometry.source_angle_average_factor, 0.0)))

    def _transition_strain_scale_diagnostics(self):
        delta_m_abs = max(abs(float(self.delta_m_transition)), 1.0)
        orbital_omega_2503 = self.transition_omega / delta_m_abs
        h0_eq317 = (
            24.0
            * self.cloud_mass_fraction
            * self.r_g
            / self.d_L
            * (self.G * self.M * orbital_omega_2503 / self.c**3) ** 2
            / self.alpha**4
        )
        return {
            "code_h0": float(self._cloud_amplitude()),
            "eq317_h0": float(h0_eq317),
            "eq317_waveform_prefactor": float(4.0 * delta_m_abs**2 * h0_eq317),
            "code_to_eq317_waveform_prefactor": float(
                self._cloud_amplitude() / max(4.0 * delta_m_abs**2 * h0_eq317, 1.0e-300)
            ),
            "delta_m_abs": delta_m_abs,
        }

    def _backreaction_scale_diagnostics(self):
        reduced_mass = self.M * self.M_star / self.M_tot
        one_minus_e2 = max(1.0e-12, 1.0 - self.e_init**2)
        orbital_energy_abs = self.G * self.M * self.M_star / (2.0 * max(self.a_init, 1.0e-30))
        orbital_angular_momentum = reduced_mass * np.sqrt(self.G * self.M_tot * self.a_init * one_minus_e2)
        return {
            "orbital_binding_energy_j": float(orbital_energy_abs),
            "orbital_angular_momentum_js": float(orbital_angular_momentum),
            "transition_energy_over_orbit": float(
                abs(self.delta_E_orbit_backreaction) / max(orbital_energy_abs, 1.0e-300)
            ),
            "transition_angular_momentum_over_orbit": float(
                abs(self.delta_L_orbit_backreaction) / max(abs(orbital_angular_momentum), 1.0e-300)
            ),
        }

    def _selected_harmonic_crossing_on_orbit(self, orbit):
        t_grid = np.asarray(orbit["t"], dtype=float)
        omega_grid = np.asarray(orbit["omega"], dtype=float)
        if t_grid.size < 2 or omega_grid.size != t_grid.size:
            return {"crossed_zero": False, "time_source": np.nan, "detuning_abs_min": np.nan}

        detuning = self.resonance_harmonic * omega_grid - self.transition_omega
        crossing_indices = np.where(detuning[:-1] * detuning[1:] <= 0.0)[0]
        if crossing_indices.size:
            left_idx = int(crossing_indices[0])
            right_idx = left_idx + 1
            det_left = float(detuning[left_idx])
            det_right = float(detuning[right_idx])
            t_left = float(t_grid[left_idx])
            t_right = float(t_grid[right_idx])
            if np.isclose(det_right, det_left):
                t_cross = 0.5 * (t_left + t_right)
            else:
                t_cross = t_left + (0.0 - det_left) * (t_right - t_left) / (det_right - det_left)
            return {"crossed_zero": True, "time_source": float(t_cross), "detuning_abs_min": 0.0}

        best_idx = int(np.argmin(np.abs(detuning)))
        return {
            "crossed_zero": False,
            "time_source": float(t_grid[best_idx]),
            "detuning_abs_min": float(abs(detuning[best_idx])),
        }

    def run(
        self,
        duration_yr=1.0,
        secular_samples=3000,
        zoom_orbits=8,
        zoom_points=4096,
        spectrum_orbits=36,
        spectrum_points=8192,
        spectrum_pad_factor=8,
        tukey_alpha=0.03,
        mismatch_max_years=4.0,
        mismatch_time_samples=12,
    ):
        """Solve the inspiral once, then build separate products for plotting and mismatch."""
        orbit, cloud = self.solve_coupled_system(duration_yr=duration_yr, secular_samples=secular_samples)
        template_orbit = self.solve_orbit(duration_yr=duration_yr, secular_samples=secular_samples)
        effective_events = self._filter_effective_resonance_events(cloud.get("resonance_events", []), require_crossing=True)
        time_window = self.build_summary_time_window(
            orbit,
            cloud,
            window_orbits=zoom_orbits,
            sample_points=zoom_points,
            min_resonance_points=2,
            summary_window_hours=float(duration_yr) * self.yr * (1.0 + self.z) / 3600.0,
        )
        if "reference_resonance_time_source" not in time_window:
            time_window["reference_resonance_time_source"] = cloud["resonance_time"]
            time_window["reference_resonance_time_obs"] = cloud["resonance_time"] * (1.0 + self.z)
            time_window["reference_resonance_harmonic"] = self.resonance_harmonic
        selected_spectrum_window = self.build_waveform_window(
            orbit,
            cloud,
            window_orbits=spectrum_orbits,
            sample_points=spectrum_points,
        )
        selected_template_spectrum_window = self._evaluate_binary_window_on_grid(
            template_orbit,
            selected_spectrum_window["t_source"],
            cloud["resonance_time"],
        )
        analysis_signal_window, analysis_template_window = self.build_multi_resonance_analysis_windows(
            orbit,
            cloud,
            template_orbit,
            sample_points=spectrum_points,
            padding_orbits=max(0.5, 0.5 * spectrum_orbits),
        )
        if "window_start_source" not in analysis_signal_window:
            analysis_signal_window["window_start_source"] = float(np.asarray(analysis_signal_window["t_source"])[0])
        if "window_stop_source" not in analysis_signal_window:
            analysis_signal_window["window_stop_source"] = float(np.asarray(analysis_signal_window["t_source"])[-1])
        if "window_start_source" not in analysis_template_window:
            analysis_template_window["window_start_source"] = float(np.asarray(analysis_template_window["t_source"])[0])
        if "window_stop_source" not in analysis_template_window:
            analysis_template_window["window_stop_source"] = float(np.asarray(analysis_template_window["t_source"])[-1])
        spectrum_window = analysis_signal_window
        pure_template_spectrum_window = analysis_template_window
        selected_frequency_domain = self.build_windowed_fft(
            selected_spectrum_window,
            pad_factor=spectrum_pad_factor,
            tukey_alpha=tukey_alpha,
        )
        selected_template_frequency_domain = self.build_windowed_fft(
            selected_template_spectrum_window,
            pad_factor=spectrum_pad_factor,
            tukey_alpha=tukey_alpha,
        )
        frequency_domain = self.build_windowed_fft(
            spectrum_window,
            pad_factor=spectrum_pad_factor,
            tukey_alpha=tukey_alpha,
        )
        pure_template_frequency_domain = self.build_windowed_fft(
            pure_template_spectrum_window,
            pad_factor=spectrum_pad_factor,
            tukey_alpha=tukey_alpha,
        )
        spectrum = self.build_frequency_spectrum(spectrum_window, pad_factor=spectrum_pad_factor)
        try:
            selected_detector_match = self._filter_available_detectors(
                lambda detector: self.compute_detector_match_pair(
                    detector,
                    selected_frequency_domain,
                    selected_template_frequency_domain,
                    signal_key="h_tilde_total",
                    template_key="h_tilde_pure",
                ),
                context="selected-resonance detector mismatch",
            )
        except ValueError as exc:
            print(f"[{self._module_stem}] selected-resonance detector mismatch unavailable ({exc})")
            selected_detector_match = {}
        try:
            detector_match = self._filter_available_detectors(
                lambda detector: self.compute_detector_match_pair(
                    detector,
                    frequency_domain,
                    pure_template_frequency_domain,
                    signal_key="h_tilde_total",
                    template_key="h_tilde_pure",
                ),
                context="detector mismatch",
            )
        except ValueError as exc:
            if selected_detector_match:
                print(f"[{self._module_stem}] multi-resonance detector mismatch unavailable; falling back to selected-resonance exact result ({exc})")
                detector_match = dict(selected_detector_match)
                frequency_domain = selected_frequency_domain
                pure_template_frequency_domain = selected_template_frequency_domain
                spectrum_window = selected_spectrum_window
                analysis_signal_window = selected_spectrum_window
                analysis_template_window = selected_template_spectrum_window
                if "window_start_source" not in analysis_signal_window:
                    analysis_signal_window["window_start_source"] = float(np.asarray(analysis_signal_window["t_source"])[0])
                if "window_stop_source" not in analysis_signal_window:
                    analysis_signal_window["window_stop_source"] = float(np.asarray(analysis_signal_window["t_source"])[-1])
            else:
                print(f"[{self._module_stem}] multi-resonance detector mismatch unavailable and no selected-resonance fallback exists ({exc})")
                detector_match = {}
        mismatch_years = self._build_mismatch_observation_years(
            duration_yr,
            mismatch_max_years,
            mismatch_time_samples,
        )
        mismatch_detector_names = tuple(detector_match.keys()) or self.detector_names
        mismatch_time_series = {}
        for detector in mismatch_detector_names:
            try:
                mismatch_time_series[detector] = self.compute_detector_mismatch_vs_time_or_empty_band(
                    detector,
                    orbit,
                    cloud,
                    template_orbit,
                    mismatch_years,
                    reference_window=analysis_signal_window,
                    pad_factor=spectrum_pad_factor,
                    tukey_alpha=tukey_alpha,
                    start_time_source=0.0,
                )
            except ValueError as exc:
                print(f"[{self._module_stem}] mismatch(T_obs) unavailable for {detector} ({exc})")
        if not mismatch_time_series:
            print(f"[{self._module_stem}] no mismatch(T_obs) series could be built for {','.join(self.detector_names)}")
        local_harmonic_windows = self.solve_local_harmonic_windows(
            orbit,
            cloud,
            padding_orbits=max(1.0, 0.5 * spectrum_orbits),
            sample_points=max(256, spectrum_points // 2),
        )
        time_window_data_path = self._save_time_series_data(
            time_window,
            cloud,
            self._time_series_stem("time_window"),
            "selected_resonance_time_window",
        )
        analysis_window_data_path = self._save_time_series_data(
            analysis_signal_window,
            cloud,
            self._time_series_stem("analysis_window"),
            "multi_resonance_analysis_window",
        )
        mismatch_data_paths = self._save_mismatch_time_series_data(mismatch_time_series)
        return {
            "orbit": orbit,
            "template_orbit": template_orbit,
            "cloud": cloud,
            "time_window": time_window,
            "selected_spectrum_window": selected_spectrum_window,
            "spectrum_window": spectrum_window,
            "analysis_signal_window": analysis_signal_window,
            "analysis_template_window": analysis_template_window,
            "selected_frequency_domain": selected_frequency_domain,
            "selected_template_frequency_domain": selected_template_frequency_domain,
            "frequency_domain": frequency_domain,
            "pure_template_frequency_domain": pure_template_frequency_domain,
            "binary_template_frequency_domain": pure_template_frequency_domain,
            "detector_match": detector_match,
            "selected_detector_match": selected_detector_match,
            "mismatch_time_series": mismatch_time_series,
            "local_harmonic_windows": local_harmonic_windows,
            "time_window_data_path": time_window_data_path,
            "analysis_window_data_path": analysis_window_data_path,
            "mismatch_data_paths": mismatch_data_paths,
            "spectrum": spectrum,
        }

    def print_summary(self, results, elapsed_s):
        orbit = results["orbit"]
        cloud = results["cloud"]

        t_days = cloud["resonance_time"] * (1.0 + self.z) / 86400.0
        f_transition_obs_mhz = self.transition_omega / (2.0 * np.pi * (1.0 + self.z)) * 1.0e3
        eta_initial = np.abs(cloud["eta_series"][:, 0]) / (2.0 * np.pi)
        eta_res = np.abs(cloud["eta_series"][:, np.argmin(np.abs(orbit["t"] - cloud["resonance_time"]))]) / (
            2.0 * np.pi
        )

        print(f"Runtime: {elapsed_s:.2f} s")
        print(
            f"Transition states: |{''.join(map(str, self.transition_solver_data['initial_state']))}> -> "
            f"|{''.join(map(str, self.transition_solver_data['final_state']))}> "
            f"({self.transition_family})"
        )
        print(
            f"Solver-driven parameters: M={self.M / self.M_sun:.2f} Msun, "
            f"M_star={self.M_star / self.M_sun:.2f} Msun, q={self.M_star / self.M:.3e}, "
            f"alpha={self.alpha:.3f}, "
            f"f_orb_init={self.f_orb_init*1.0e3:.3f} mHz, "
            f"eta(A.6 @ a_init)={self.eta_ref_hz:.3e} Hz, "
            f"Gamma_decay={self.Gamma_decay_hz:.3e} Hz"
        )
        initial_setup = getattr(self, "initial_frequency_setup", {})
        print(
            "Initial-frequency setup: "
            f"mode={initial_setup.get('mode', 'unknown')}, "
            f"f_init/f_res={initial_setup.get('frequency_ratio_to_selected_resonance', np.nan):.9f}, "
            f"target={initial_setup.get('target_delay_obs_days', np.nan):.3f} obs days, "
            f"Peters actual={initial_setup.get('actual_delay_obs_days', np.nan):.3f} obs days, "
            f"status={initial_setup.get('status', 'unknown')}"
        )
        print(
            f"Solver profile: {self.solver_profile}, method={self.solver_method}, "
            f"rtol={self.solver_rtol:.1e}, numba_kepler={'on' if _NUMBA_AVAILABLE else 'off'}"
        )
        print(
            "Pipeline scope: original coupled-ODE waveform/mismatch pipeline; "
            "external PRL/2403/2512-style proxy diagnostics are not used in the production summary."
        )
        band_diag = cloud.get("resonance_band_diagnostics", {})
        reference_band_diag = cloud.get("reference_resonance_band_diagnostics", {})
        production_cloud_mode = cloud.get("production_cloud_evolution_mode", self._resolved_cloud_evolution_mode())
        print(
            "Resonance-local cloud mode: "
            f"requested={self.cloud_evolution_mode} -> production={production_cloud_mode}, "
            f"band_width={self.resonance_band_width_factor:.3g}|eta|"
        )
        print(
            "Peters reference resonance band: "
            f"active_fraction={reference_band_diag.get('active_band_fraction', np.nan):.3e}, "
            f"active_time={reference_band_diag.get('active_band_time_obs_days', np.nan):.3e} obs days, "
            f"selected_fraction={reference_band_diag.get('selected_band_fraction', np.nan):.3e}, "
            f"selected |det|/|eta| start/end="
            f"{reference_band_diag.get('selected_start_abs_detuning_over_eta', np.nan):.3g}/"
            f"{reference_band_diag.get('selected_end_abs_detuning_over_eta', np.nan):.3g}, "
            f"sustained_near_resonance={reference_band_diag.get('sustained_near_resonance_candidate', False)}"
        )
        print(
            "Coupled-track resonance band: "
            f"active_fraction={band_diag.get('active_band_fraction', np.nan):.3e}, "
            f"active_time={band_diag.get('active_band_time_obs_days', np.nan):.3e} obs days, "
            f"selected_tau_LZ={band_diag.get('selected_tau_lz_obs_days', np.nan):.3e} obs days, "
            f"selected_fraction={band_diag.get('selected_band_fraction', np.nan):.3e}, "
            f"detuning_drift/eta={band_diag.get('selected_detuning_drift_in_band_over_eta', np.nan):.3g}, "
            f"floating_candidate={band_diag.get('floating_candidate', False)}"
        )
        strain_scale = self._transition_strain_scale_diagnostics()
        print(
            "Transition strain scale: "
            f"h0_code={strain_scale['code_h0']:.3e}, "
            f"h0_2503Eq3.17={strain_scale['eq317_h0']:.3e}, "
            f"4|dm|^2*h0_2503={strain_scale['eq317_waveform_prefactor']:.3e}, "
            f"code/(4|dm|^2*h0_2503)={strain_scale['code_to_eq317_waveform_prefactor']:.3g}, "
            f"delta_m={strain_scale['delta_m_abs']:.0f}"
        )
        backreaction_scale = self._backreaction_scale_diagnostics()
        print(
            f"Orbital backreaction: {'enabled' if self.include_orbital_backreaction else 'disabled'}, "
            f"mode={self.orbital_backreaction_mode}, tidal_m={self.tidal_m}, "
            f"cloud_initial_state={self.cloud_initial_state}, "
            f"DeltaE_tot={self.delta_E_orbit_backreaction:.3e} J, "
            f"DeltaL_tot={self.delta_L_orbit_backreaction:.3e} J*s"
        )
        print(
            "Backreaction scale check: "
            f"|DeltaE_cloud|/|E_orb,init|={backreaction_scale['transition_energy_over_orbit']:.3e}, "
            f"|DeltaL_cloud|/|L_orb,init|={backreaction_scale['transition_angular_momentum_over_orbit']:.3e}"
        )
        if (
            backreaction_scale["transition_energy_over_orbit"] > 1.0
            or backreaction_scale["transition_angular_momentum_over_orbit"] > 1.0
        ):
            print(
                "Backreaction scale warning: cloud reservoir exceeds the initial companion-orbit scale; "
                "early crossing shifts and strong-feedback behavior are parameter driven."
            )
        template_orbit = results.get("template_orbit")
        if template_orbit is not None:
            vacuum_crossing = self._selected_harmonic_crossing_on_orbit(template_orbit)
            print(
                "Pure Peters selected-resonance check: "
                f"n={self.resonance_harmonic}, "
                f"t_obs={vacuum_crossing['time_source'] * (1.0 + self.z) / 86400.0:.3f} days"
                f"{'' if vacuum_crossing.get('crossed_zero', False) else '(no-cross)'}, "
                f"|detuning|min={vacuum_crossing.get('detuning_abs_min', np.nan):.3e} rad/s"
            )
        high_state_rate = np.asarray(cloud.get("selected_high_state_rate", []), dtype=float)
        finite_high_rate = high_state_rate[np.isfinite(high_state_rate)]
        if finite_high_rate.size:
            print(
                "Eq.(31) sign convention: "
                f"Deltaomega_high-low={self.transition_omega:.3e} rad/s, "
                f"Deltam_high-low={self.delta_m_high_low:.0f}, "
                f"selected R_high range=[{np.min(finite_high_rate):.3e}, {np.max(finite_high_rate):.3e}] 1/s"
            )
        print(f"Boson mass from alpha: {self.transition_solver_data['boson_mass_eV']:.3e} eV")
        print(f"Selected RWA harmonic: n = {self.resonance_harmonic}")
        print(f"Observed transition frequency: {f_transition_obs_mhz:.4f} mHz")
        print(f"Closest approach to resonance: t_obs = {t_days:.3f} days")
        print("Active tidal harmonics in the cloud dynamics:", ", ".join(str(n) for n in cloud["active_harmonics"]))
        resonance_events = cloud.get("resonance_events", [])
        if resonance_events:
            print(
                "Harmonics scanned for resonance diagnostics: "
                + ", ".join(str(event["harmonic"]) for event in resonance_events)
            )
        if resonance_events:
            effective_events = self._filter_effective_resonance_events(resonance_events, require_crossing=True)
            degenerate_groups = [
                group
                for group in cloud.get("degenerate_resonance_groups", [])
                if group.get("size", 0) > 1
            ]
            print(
                "Effective non-quenched harmonics n="
                + (",".join(str(event["harmonic"]) for event in effective_events) if effective_events else "none")
            )
            if degenerate_groups:
                print(
                    "Degenerate harmonic bundles: "
                    + ", ".join(
                        f"G{group['group_id']}[n={','.join(str(n) for n in group['harmonics'])}]"
                        for group in degenerate_groups
                    )
                )
            crossed_events = [event for event in resonance_events if event.get("crossed_zero", False)]
            non_crossing_events = [event for event in resonance_events if not event.get("crossed_zero", False)]

            def format_crossing_event(event):
                return (
                    f"n={event['harmonic']}:{event['time_obs']/86400.0:.3f}"
                    f",z_ad={event.get('adiabatic_z', np.nan):.3g}"
                    f"{'' if event.get('adiabatic_survives', False) else '[quenched]'}"
                    f"{'*' if event['is_selected_resonance'] else ''}"
                )

            def format_non_crossing_event(event):
                if event.get("closest_at_boundary", False):
                    location = event.get("closest_boundary", "boundary") or "boundary"
                else:
                    location = "interior"
                return (
                    f"n={event['harmonic']}:{location}@{event['time_obs']/86400.0:.3f}"
                    f",|det|={event.get('detuning_abs_min', np.nan):.3e}"
                )

            reference_events = cloud.get("reference_resonance_events", [])
            if reference_events:
                reference_crossed = [event for event in reference_events if event.get("crossed_zero", False)]
                reference_non_crossing = [event for event in reference_events if not event.get("crossed_zero", False)]
                print(
                    "Peters reference resonance crossings (obs days): "
                    + (
                        ", ".join(format_crossing_event(event) for event in reference_crossed)
                        if reference_crossed
                        else "none"
                    )
                )
                if reference_non_crossing:
                    print(
                        "Peters reference no-cross boundary/closest diagnostics (not resonances): "
                        + ", ".join(format_non_crossing_event(event) for event in reference_non_crossing)
                    )
            print(
                "Coupled-track resonance crossings (obs days): "
                + (", ".join(format_crossing_event(event) for event in crossed_events) if crossed_events else "none")
            )
            if non_crossing_events:
                print(
                    "Coupled-track no-cross boundary/closest diagnostics (not resonances): "
                    + ", ".join(format_non_crossing_event(event) for event in non_crossing_events)
                )
            selected_event = next(
                (event for event in resonance_events if event.get("is_selected_resonance", False)),
                None,
            )
            if selected_event is not None:
                width_obs_days = (
                    float(selected_event.get("resonance_width_source_s", np.nan))
                    * (1.0 + self.z)
                    / 86400.0
                )
                if np.isfinite(width_obs_days):
                    run_obs_days = float(orbit["t"][-1]) * (1.0 + self.z) / 86400.0
                    width_ratio = width_obs_days / max(run_obs_days, 1.0e-30)
                    print(
                        "Selected resonance width: "
                        f"n={selected_event['harmonic']}, "
                        f"Delta t_LZ~{width_obs_days:.3f} obs days, "
                        f"Delta t_LZ/T_obs={width_ratio:.3g}"
                    )
                    if width_ratio >= 0.5:
                        print(
                            "2512.17887 benchmark note: wide/partial transition on this observation window; "
                            "balance-law/IA proxies are qualitative here."
                        )
        print(
            f"Time-domain t=0 reference: n={results['time_window'].get('reference_resonance_harmonic', self.resonance_harmonic)}, "
            f"t_obs={results['time_window'].get('reference_resonance_time_obs', cloud['resonance_time'] * (1.0 + self.z))/86400.0:.3f} days"
        )
        analysis_window = results.get("analysis_signal_window", results["spectrum_window"])
        print(
            "Exact FFT detector analysis window: "
            f"[{analysis_window['window_start_source'] / self.yr:.4f}, {analysis_window['window_stop_source'] / self.yr:.4f}] yr "
            f"in source time, harmonics="
            + ",".join(str(n) for n in analysis_window.get("analysis_event_harmonics", []))
        )
        if results.get("local_harmonic_windows"):
            print(
                "Local single-harmonic windows: "
                + ", ".join(
                    f"n={item['harmonic']}:peak|cg*ce|={item['peak_overlap_abs']:.3e}"
                    for item in results["local_harmonic_windows"]
                )
            )
        peak_ratio = np.max(np.abs(results["time_window"]["h_axion"])) / np.max(
            np.abs(results["time_window"]["h_back"])
        )
        time_window_norm = np.asarray(results["time_window"].get("population_norm", []), dtype=float)
        raw_coherence = np.asarray(results["time_window"].get("overlap_abs", []), dtype=float)
        finite_norm = time_window_norm[np.isfinite(time_window_norm)]
        finite_raw = raw_coherence[np.isfinite(raw_coherence)]
        print(
            "Initial comb strengths |eta_n|/(2pi) [Hz]: "
            + ", ".join(f"n={n}:{val:.3e}" for n, val in zip(self.harmonics, eta_initial))
        )
        print(
            "Comb strengths near resonance [Hz]: "
            + ", ".join(f"n={n}:{val:.3e}" for n, val in zip(self.harmonics, eta_res))
        )
        if finite_norm.size and finite_raw.size:
            print(
                "Coherence diagnostics: "
                f"max raw |cg*ce|={np.max(finite_raw):.3e}, "
                f"state norm range=[{np.min(finite_norm):.3e}, {np.max(finite_norm):.3e}]"
            )
        cloud_overlap = np.asarray(cloud.get("overlap_abs", []), dtype=float)
        cloud_t = np.asarray(cloud.get("t", []), dtype=float)
        pop_ground = np.asarray(cloud.get("pop_ground", []), dtype=float)
        pop_excited = np.asarray(cloud.get("pop_excited", []), dtype=float)
        finite_cloud_overlap = np.isfinite(cloud_overlap)
        if (
            cloud_overlap.size
            and cloud_t.size == cloud_overlap.size
            and pop_ground.size == cloud_overlap.size
            and pop_excited.size == cloud_overlap.size
            and np.any(finite_cloud_overlap)
        ):
            peak_idx = int(np.nanargmax(cloud_overlap))
            print(
                "Cloud-state diagnostics: "
                f"final |ci|^2={pop_ground[-1]:.3e}, final |cf|^2={pop_excited[-1]:.3e}, "
                f"peak |ci*cf|={cloud_overlap[peak_idx]:.3e} "
                f"at t_obs={cloud_t[peak_idx] * (1.0 + self.z) / 86400.0:.3f} days"
            )
        time_t_obs = np.asarray(results["time_window"].get("t_obs", []), dtype=float)
        if finite_raw.size and time_t_obs.size == raw_coherence.size:
            peak_window_idx = int(np.nanargmax(raw_coherence))
            edge_note = " (near window edge)" if peak_window_idx in {0, raw_coherence.size - 1} else ""
            print(
                "Displayed-window diagnostics: "
                f"window=[{time_t_obs[0] / 86400.0:.3f}, {time_t_obs[-1] / 86400.0:.3f}] obs days, "
                f"peak raw |cg*ce|={raw_coherence[peak_window_idx]:.3e} "
                f"at t_obs={time_t_obs[peak_window_idx] / 86400.0:.3f} days{edge_note}"
            )
        print(f"Zoom-window peak ratio max|h_axion|/max|h_back| = {peak_ratio:.3e}")
        print(
            f"Mismatch FFT window: Tukey(alpha={results['frequency_domain']['match_tukey_alpha']:.3f}); "
            "spectrum display window: Tukey; FFT inputs are demeaned."
        )
        if results.get("time_window_data_path") is not None:
            print(f"Saved selected-resonance time-series data -> {results['time_window_data_path']}")
        if results.get("analysis_window_data_path") is not None:
            print(f"Saved multi-resonance analysis time-series data -> {results['analysis_window_data_path']}")
        print("Detector mismatch now compares: h_total vs h_pure (pure Peters binary template).")
        summary_detectors = tuple(
            dict.fromkeys(
                tuple(results.get("detector_match", {}).keys())
                + tuple(results.get("mismatch_time_series", {}).keys())
            )
        )
        for detector in summary_detectors:
            metrics = results.get("detector_match", {}).get(detector)
            selected_metrics = results.get("selected_detector_match", {}).get(detector)
            final_series = results["mismatch_time_series"].get(detector)
            if selected_metrics is not None:
                print(
                    f"{detector}: selected-resonance exact axion-band SNR={selected_metrics['snr']:.3e}, "
                    f"faithfulness={selected_metrics['faithfulness']:.6f}, "
                    f"mismatch={selected_metrics['mismatch']:.3e}, "
                    f"band=[{selected_metrics['analysis_band_hz'][0]*1.0e3:.3f}, {selected_metrics['analysis_band_hz'][1]*1.0e3:.3f}] mHz"
                    f"{self._format_match_band_note(selected_metrics)}"
                )
            if metrics is not None:
                print(
                    f"{detector}: cumulative multi-resonance exact axion-band SNR={metrics['snr']:.3e}, "
                    f"faithfulness={metrics['faithfulness']:.6f}, mismatch={metrics['mismatch']:.3e}"
                    ", "
                    f"band=[{metrics['analysis_band_hz'][0]*1.0e3:.3f}, {metrics['analysis_band_hz'][1]*1.0e3:.3f}] mHz"
                    f"{self._format_match_band_note(metrics)}"
                )
            elif final_series is not None:
                print(
                    f"{detector}: exact detector mismatch unavailable "
                    f"({final_series.get('method', 'unknown')}); {final_series.get('exact_error', 'no details')}"
                )
            if final_series is not None:
                final_rho = float(np.asarray(final_series.get("snr", [np.nan]), dtype=float)[-1])
                print(
                    f"  final mismatch-series d/(2rho^2) at "
                    f"T_eff={final_series.get('effective_observation_years', final_series['observation_years'])[-1]:.4f} yr: "
                    f"{final_series['distinguishability_threshold'][-1]:.3e}"
                    f" (rho_mismatch={final_rho:.3e})"
                )
                if final_series.get("monotone_envelope_applied", False):
                    print(
                        "  mismatch(T) monotone envelope applied; "
                        f"largest removed downward step={final_series.get('max_downward_correction', 0.0):.3e}"
                    )

    def _plot_mismatch_axis(self, ax, results):
        detector_colors = {"DECIGO": "#1f77b4", "ET": "#2ca02c", "CE": "#9467bd"}
        active_detectors = tuple(results["mismatch_time_series"].keys())
        if not active_detectors:
            ax.text(0.5, 0.5, "No detector mismatch data", ha="center", va="center")
            ax.set_axis_off()
            return

        plotted_detectors = []
        unavailable_notes = []
        for detector in active_detectors:
            mismatch_series = results["mismatch_time_series"][detector]
            method = mismatch_series.get("method", "exact_fft")
            observation_years = np.asarray(
                mismatch_series.get("effective_observation_years", mismatch_series["observation_years"]),
                dtype=float,
            )
            mismatch_values = np.asarray(mismatch_series["mismatch"], dtype=float)
            finite_mask = np.isfinite(observation_years) & np.isfinite(mismatch_values)
            if not np.any(finite_mask):
                unavailable_notes.append(f"{detector}: {mismatch_series.get('exact_error', method)}")
                continue

            label = detector if method.startswith("exact") else f"{detector} ({method})"
            x_plot = np.concatenate(([0.0], observation_years[finite_mask]))
            y_plot = np.concatenate(([0.0], mismatch_values[finite_mask]))
            ax.plot(
                x_plot,
                y_plot,
                color=detector_colors[detector],
                lw=2.0,
                label=label,
            )
            plotted_detectors.append(detector)

        if not plotted_detectors:
            ax.text(0.5, 0.58, "No finite detector mismatch data", ha="center", va="center")
            if unavailable_notes:
                ax.text(
                    0.5,
                    0.42,
                    "\n".join(unavailable_notes[:2]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    wrap=True,
                )
            ax.set_axis_off()
            return

        ax.set_xlabel("Observation time (yr)")
        ax.set_ylabel("Mismatch")
        max_obs_year = max(
            float(np.nanmax(results["mismatch_time_series"][detector]["observation_years"]))
            for detector in plotted_detectors
        )
        ax.set_xlim(0.0, max_obs_year)
        y_mismatch_max = max(
            np.nanmax(np.asarray(results["mismatch_time_series"][detector]["mismatch"], dtype=float))
            for detector in plotted_detectors
        )
        ax.set_ylim(0.0, 1.15 * max(y_mismatch_max, 1.0e-12))
        #ax.set_title(title)
        methods = {results["mismatch_time_series"][detector].get("method", "exact_fft") for detector in active_detectors}
        if any(method.startswith("detector_") for method in methods):
            ax.text(
                0.98,
                0.04,
                "some detectors unavailable",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                alpha=0.75,
            )
        ax.legend(loc="best", framealpha=0.9)

    def plot_detector_comparison(self, results):
        """Mismatch diagnostics against the pure-binary template."""
        fig_b, ax_left = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
        self._plot_mismatch_axis(ax_left, results)
        self._finalize_figure(fig_b, self._mismatch_figure_stem())

    def _selected_plot_events(self, cloud):
        events = []
        for event in cloud.get("reference_resonance_events", []):
            if event.get("is_selected_resonance", False):
                item = dict(event)
                item["_plot_source"] = "pure_peters"
                events.append(item)
        for event in cloud.get("resonance_events", []):
            if event.get("is_selected_resonance", False):
                item = dict(event)
                item["_plot_source"] = "coupled"
                events.append(item)
        return events

    def plot_summary(self, results):
        """Figure 1: orbital evolution plus axion-only time-domain signal."""
        orbit = results["orbit"]
        cloud = results["cloud"]
        time_window = results["time_window"]

        t_days = self._observer_days(orbit["t"])
        selected_plot_events = self._selected_plot_events(cloud)
        if not selected_plot_events:
            selected_plot_events = list(time_window.get("summary_resonance_events", []))

        cloud_overlap = np.asarray(cloud.get("overlap_abs", []), dtype=float)
        cloud_t = np.asarray(cloud.get("t", []), dtype=float)
        center_source = float(cloud.get("resonance_time", 0.0))
        peak_source = np.nan
        if cloud_overlap.size and cloud_t.size == cloud_overlap.size and np.any(np.isfinite(cloud_overlap)):
            peak_source = float(cloud_t[int(np.nanargmax(cloud_overlap))])
        def event_source_time_for_signal(event):
            return float(event.get("time_source", event.get("t_source", np.nan)))

        def event_crossed_for_signal(event):
            return bool(event.get("crossed_zero", event.get("crossed", False)))

        coupled_crossings = [
            event
            for event in selected_plot_events
            if event.get("_plot_source") != "pure_peters"
            and event_crossed_for_signal(event)
            and np.isfinite(event_source_time_for_signal(event))
        ]
        pure_crossings = [
            event
            for event in selected_plot_events
            if event.get("_plot_source") == "pure_peters"
            and event_crossed_for_signal(event)
            and np.isfinite(event_source_time_for_signal(event))
        ]
        interior_closest = [
            event
            for event in selected_plot_events
            if str(event.get("closest_position", "")).lower() == "interior"
            and np.isfinite(event_source_time_for_signal(event))
        ]
        if coupled_crossings:
            signal_center_source = event_source_time_for_signal(coupled_crossings[0])
        elif pure_crossings:
            signal_center_source = event_source_time_for_signal(pure_crossings[0])
        elif interior_closest:
            signal_center_source = event_source_time_for_signal(interior_closest[0])
        elif np.isfinite(peak_source):
            signal_center_source = peak_source
        else:
            signal_center_source = center_source

        signal_center_source = float(np.clip(signal_center_source, orbit["t"][0], orbit["t"][-1]))
        signal_cycles = float(
            os.environ.get(
                "LOWFREQ_ORBIT_SUMMARY_SIGNAL_CYCLES",
                os.environ.get("ORBIT_SUMMARY_SIGNAL_CYCLES", "100.0"),
            )
        )
        omega_center = float(np.interp(signal_center_source, orbit["t"], orbit["omega"]))
        period_center = 2.0 * np.pi / max(omega_center, 1.0e-300)
        half_signal = 0.5 * max(signal_cycles, 1.0) * period_center
        t_start_signal = signal_center_source - half_signal
        t_stop_signal = signal_center_source + half_signal
        samples_per_cycle = float(os.environ.get("ORBIT_SUMMARY_SAMPLES_PER_CYCLE", "64.0"))
        signal_samples = int(max(2048, min(20000, np.ceil(max(signal_cycles, 1.0) * samples_per_cycle))))
        local_signal_window = self.build_waveform_window_for_source_interval(
            orbit,
            cloud,
            t_start_signal,
            t_stop_signal,
            sample_points=signal_samples,
        )
        window_start_obs = float(local_signal_window["t_obs"][0])
        fig = plt.figure(figsize=(7.1, 4.8), constrained_layout=True)
        grid = fig.add_gridspec(2, 1, hspace=0.14)

        ax0 = fig.add_subplot(grid[0, 0])
        a_orbit = np.asarray(orbit["a"], dtype=float)
        e_orbit = np.asarray(orbit["e"], dtype=float)
        a_au = a_orbit / self.AU
        ax0.plot(t_days, a_au, color="#B23A48", lw=0.9, alpha=0.88, label=r"$a$")
        ax0.set_xlabel("Observer time (days)")
        ax0.set_ylabel("Semi-major axis (AU)", color="#B23A48")
        ax0.tick_params(axis="y", labelcolor="#B23A48")

        ax0b = ax0.twinx()
        ax0.set_zorder(ax0b.get_zorder() + 1)
        ax0.patch.set_visible(False)
        ax0b.patch.set_visible(False)
        ax0b.plot(t_days, e_orbit, color="#2C7FB8", lw=0.9, alpha=0.82, label=r"$e$")
        ax0b.set_ylabel("Eccentricity", color="#2C7FB8")
        ax0b.tick_params(axis="y", labelcolor="#2C7FB8")

        def event_time_obs(event):
            return float(event.get("time_obs", event.get("t_obs", np.nan)))

        def event_is_crossed(event):
            return bool(event.get("crossed_zero", event.get("crossed", False)))

        def event_style(event):
            if event.get("_plot_source") == "pure_peters":
                return "0.35", "--", 0.60
            if event_is_crossed(event):
                return "k", "-", 0.70
            return "#D55E00", ":", 0.55

        for event in selected_plot_events:
            t_event_obs = event_time_obs(event)
            if not np.isfinite(t_event_obs):
                continue
            color, linestyle, alpha = event_style(event)
            ax0.axvline(t_event_obs / 86400.0, color=color, linestyle=linestyle, alpha=alpha)
        lines, labels = ax0.get_legend_handles_labels()
        lines_b, labels_b = ax0b.get_legend_handles_labels()
        ax0.legend(lines + lines_b, labels + labels_b, loc="best", framealpha=0.88, fontsize=7)
        #ax0.set_title("1. Peters Evolution of Semi-major Axis and Eccentricity")

        ax1 = fig.add_subplot(grid[1, 0])
        t_rel_hours = (np.asarray(local_signal_window["t_obs"], dtype=float) - window_start_obs) / 3600.0
        h_axion = np.asarray(local_signal_window["h_axion"], dtype=float)
        overlap_display = np.asarray(local_signal_window["overlap_abs"], dtype=float)
        ax1.plot(t_rel_hours, h_axion, color="#5B8DB8", lw=0.55, alpha=0.48, label=r"$h_a$")
        for event in selected_plot_events:
            event_rel_hr = (event_time_obs(event) - window_start_obs) / 3600.0
            if event_rel_hr < t_rel_hours[0] or event_rel_hr > t_rel_hours[-1]:
                continue
            color, linestyle, alpha = event_style(event)
            ax1.axvline(event_rel_hr, color=color, linestyle=linestyle, alpha=alpha)
        ax1.set_xlabel("Observer time from window start (hours)")
        ax1.set_ylabel(r"Axion strain $h_a$", color="#1F78B4")
        ax1.tick_params(axis="y", labelcolor="#1F78B4")
        ax1b = ax1.twinx()
        ax1.set_zorder(ax1b.get_zorder() + 1)
        ax1.patch.set_visible(False)
        ax1b.patch.set_visible(False)
        ax1b.plot(t_rel_hours, overlap_display, color="#D55E00", lw=1.15, alpha=0.96, label=r"$|c_i^* \tilde{c}_f|$")
        ax1b.set_ylabel(r"$|c_i^* \tilde{c}_f|$", color="#E67E22")
        ax1b.tick_params(axis="y", labelcolor="#E67E22")
        lines, labels = ax1.get_legend_handles_labels()
        lines_b, labels_b = ax1b.get_legend_handles_labels()
        ax1.legend(lines + lines_b, labels + labels_b, loc="upper right", framealpha=0.9, fontsize=7)

        for axis in fig.axes:
            axis.tick_params(axis="both", labelsize=8)
            axis.xaxis.label.set_size(9)
            axis.yaxis.label.set_size(9)

        self._finalize_figure(fig, self._summary_figure_stem())

if __name__ == "__main__":
    start = time.time()
    LOWFREQ_DETECTOR_NAMES = ("DECIGO",)
    LOWFREQ_TRANSITION_PRESET = os.environ.get("LOWFREQ_TRANSITION_PRESET", "322").lower()
    LOWFREQ_TRANSITION_PRESETS = {
        "211": {
            "family": "hyperfine",
            "initial": (2, 1, -1),
            "final": (2, 1, 1),
            "module_stem": "lowfre211",
            "direction_tag": "upward",
            "description": "hyperfine transition |21-1> -> |211|",
        },
        "211v": {
            "family": "hyperfine",
            "initial": (2, 1, 1),
            "final": (2, 1, -1),
            "module_stem": "lowfre211v",
            "direction_tag": "downward",
            "description": "hyperfine transition |211> -> |21-1|",
        },
        "322": {
            "family": "fine",
            "initial": (3, 0, 0),
            "final": (3, 2, 2),
            "module_stem": "lowfre322",
            "direction_tag": "upward",
            "description": "fine transition |300> -> |322|",
        },
        "322v": {
            "family": "fine",
            "initial": (3, 2, 2),
            "final": (3, 0, 0),
            "module_stem": "lowfre322v",
            "direction_tag": "downward",
            "description": "fine transition |322> -> |300|",
        },
    }
    if LOWFREQ_TRANSITION_PRESET not in LOWFREQ_TRANSITION_PRESETS:
        raise ValueError(
            "Unknown LOWFREQ_TRANSITION_PRESET="
            f"{LOWFREQ_TRANSITION_PRESET!r}; choose one of "
            + ", ".join(sorted(LOWFREQ_TRANSITION_PRESETS))
        )
    lowfreq_transition_preset = LOWFREQ_TRANSITION_PRESETS[LOWFREQ_TRANSITION_PRESET]
    lowfreq_orbital_backreaction_mode = os.environ.get(
        "LOWFREQ_ORBITAL_BACKREACTION_MODE",
        "selected_rwa",
    )
    lowfreq_cloud_evolution_mode = os.environ.get(
        "LOWFREQ_CLOUD_EVOLUTION_MODE",
        "band_gated",
    )
    lowfreq_duration_yr = 0.4
    lowfreq_secular_samples = 1440
    lowfreq_zoom_points = 4096
    lowfreq_spectrum_points = 4096
    lowfreq_spectrum_pad_factor = 4
    lowfreq_mismatch_max_years = 0.4
    lowfreq_mismatch_time_samples = 12
    lowfreq_orbital_start_ratio = 0.999998
    lowfreq_target_resonance_delay_obs_days = float(os.environ.get("LOWFREQ_TARGET_RESONANCE_DELAY_DAYS", "96.0"))
    lowfreq_cloud_initial_state = os.environ.get("LOWFREQ_CLOUD_INITIAL_STATE", "bare")
    lowfreq_cloud_mass_fraction = resolve_default_lowfreq_cloud_mass_fraction()
    lowfreq_hansen_e_samples = 48
    lowfreq_hansen_M_samples = 1024
    lowfreq_overlap_grid_points = 4096
    lowfreq_multi_harmonic_drive = False
    lowfreq_harmonics_to_keep = 8
    lowfreq_transition_family = lowfreq_transition_preset["family"]
    lowfreq_initial_state = lowfreq_transition_preset["initial"]
    lowfreq_final_state = lowfreq_transition_preset["final"]
    lowfreq_transition_frequency_hz = None

    # 这组默认值对应低频主脚本：
    # - 150 Msun 量级黑洞
    # lowfre211 default run:
    # - M_bh = 150 Msun, M_star = 0.1 Msun
    # - hyperfine transition |21-1> -> |211| with multi-harmonic driving
    simulator = EccentricResonantTidalGA(
        M_bh=1500.0,
        M_star=0.5,
        alpha=0.25,
        distance_Mpc=DEFAULT_DISTANCE_MPC,
        z=0.022,
        e_init=0.65,
        f_orb_init=None,
        resonance_harmonic=4,
        max_harmonic=8,
        multi_harmonic_drive=lowfreq_multi_harmonic_drive,
        harmonics_to_keep=lowfreq_harmonics_to_keep,
        binary_harmonics=12,
        transition_frequency_hz=lowfreq_transition_frequency_hz,
        eta_ref_hz=None,
        Gamma_decay_hz=None,
        transition_family=lowfreq_transition_family,
        initial_state=lowfreq_initial_state,
        final_state=lowfreq_final_state,
        orbital_backreaction_mode=lowfreq_orbital_backreaction_mode,
        cloud_evolution_mode=lowfreq_cloud_evolution_mode,
        cloud_initial_state=lowfreq_cloud_initial_state,
        orbital_start_ratio=lowfreq_orbital_start_ratio,
        target_resonance_delay_obs_days=lowfreq_target_resonance_delay_obs_days,
        cloud_mass_fraction=lowfreq_cloud_mass_fraction,
        geom_factor=None,
        solver_profile="accurate",
        hansen_e_samples=lowfreq_hansen_e_samples,
        hansen_M_samples=lowfreq_hansen_M_samples,
        overlap_grid_points=lowfreq_overlap_grid_points,
        detector_names=LOWFREQ_DETECTOR_NAMES,
        module_stem=lowfreq_transition_preset["module_stem"],
        direction_tag=lowfreq_transition_preset["direction_tag"],
    )

    print(
        f"Building {lowfreq_transition_preset['module_stem']}: "
        "low-frequency eccentric inspiral + harmonic comb model..."
    )
    print(
        f"Using preset {LOWFREQ_TRANSITION_PRESET}: "
        f"{lowfreq_transition_preset['description']} with multi-harmonic cloud driving."
    )
    print(
        "Run mode: full numerical; "
        f"detectors={','.join(LOWFREQ_DETECTOR_NAMES)}, solver=accurate, "
        f"target_resonance_delay={lowfreq_target_resonance_delay_obs_days:.1f} obs days, "
        f"duration_yr={lowfreq_duration_yr:.2f}, "
        f"transition=|{''.join(map(str, lowfreq_initial_state))}> -> |{''.join(map(str, lowfreq_final_state))}>, "
        f"harmonics_to_keep={lowfreq_harmonics_to_keep}, "
        f"orbital_backreaction_mode={lowfreq_orbital_backreaction_mode}, "
        f"cloud_evolution_mode={lowfreq_cloud_evolution_mode}, "
        f"cloud_initial_state={lowfreq_cloud_initial_state}"
    )
    results = simulator.run(
        duration_yr=lowfreq_duration_yr,
        secular_samples=lowfreq_secular_samples,
        zoom_orbits=20,
        zoom_points=lowfreq_zoom_points,
        spectrum_points=lowfreq_spectrum_points,
        spectrum_pad_factor=lowfreq_spectrum_pad_factor,
        mismatch_max_years=lowfreq_mismatch_max_years,
        mismatch_time_samples=lowfreq_mismatch_time_samples,
    )
    elapsed = time.time() - start

    simulator.print_summary(results, elapsed)
    simulator.plot_summary(results)
    simulator.plot_detector_comparison(results)
