import time
import os
from pathlib import Path
import math
import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.special import eval_genlaguerre, jv, sph_harm_y
from scipy.interpolate import CubicSpline

from transition_geometry import compute_transition_geometry

def spherical_harmonic(m, l, phi, theta):
    return sph_harm_y(l, m, theta, phi)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HIGHFREQ_ALPHA = 0.3
DEFAULT_HIGHFREQ_BH_SPIN = 0.7
DEFAULT_HIGHFREQ_CLOUD_MASS_PROFILE = "saturated"
HIGHFREQ_CLOUD_MASS_PROFILE_FRACTIONS = {
    "saturated": 5.0e-2,
    "paper": 5.0e-2,
    "upper": 1.0e-1,
    "moderate": 5.0e-2,
    "effective": 3.0e-2,
    "conservative": 3.0e-2,
}


def resolve_default_highfreq_alpha(default_alpha=DEFAULT_HIGHFREQ_ALPHA):
    override = os.environ.get("HIGHFREQ_ALPHA")
    if override is not None:
        return float(override)
    return float(default_alpha)


def resolve_default_highfreq_bh_spin(default_spin=DEFAULT_HIGHFREQ_BH_SPIN):
    override = os.environ.get("HIGHFREQ_BH_SPIN")
    if override is not None:
        return float(override)
    return float(default_spin)


def resolve_default_highfreq_cloud_mass_fraction(default_fraction=5.0e-2):
    override = os.environ.get("HIGHFREQ_CLOUD_MASS_FRACTION")
    if override is not None:
        return float(override)
    profile = os.environ.get("HIGHFREQ_CLOUD_MASS_PROFILE", DEFAULT_HIGHFREQ_CLOUD_MASS_PROFILE)
    profile = str(profile).strip().lower()
    if profile not in HIGHFREQ_CLOUD_MASS_PROFILE_FRACTIONS:
        valid = ", ".join(sorted(HIGHFREQ_CLOUD_MASS_PROFILE_FRACTIONS))
        raise ValueError(f"Unknown HIGHFREQ_CLOUD_MASS_PROFILE={profile!r}; choose one of {valid}.")
    return float(HIGHFREQ_CLOUD_MASS_PROFILE_FRACTIONS.get(profile, default_fraction))


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

    guess = np.where(e < 0.8, mean_anomaly, np.pi * np.ones_like(mean_anomaly))

    for _ in range(max_iter):
        residual = guess - e * np.sin(guess) - mean_anomaly
        jacobian = 1.0 - e * np.cos(guess)
        step = residual / jacobian
        guess -= step
        if np.max(np.abs(step)) < tol:
            break
    return guess


class _SolutionSlice:
    """Expose selected components of a dense_output solution with the same `.sol()` API."""

    def __init__(self, full_solution, component_slice):
        self._full_solution = full_solution
        self._component_slice = component_slice

    def sol(self, t_eval):
        return self._full_solution.sol(t_eval)[self._component_slice]


class _TabulatedSolution:
    """Linear `.sol()` interpolation for piecewise/event-based trajectories."""

    def __init__(self, t_grid, values):
        self.t_grid = np.asarray(t_grid, dtype=float)
        self.values = np.asarray(values, dtype=float)
        if self.values.ndim != 2:
            raise ValueError("values must have shape (n_components, n_times).")
        if self.values.shape[1] != self.t_grid.size:
            raise ValueError("values and t_grid have incompatible lengths.")

    def sol(self, t_eval):
        t_eval = np.asarray(t_eval, dtype=float)
        flat_t = t_eval.reshape(-1)
        rows = [np.interp(flat_t, self.t_grid, row) for row in self.values]
        return np.asarray(rows).reshape((self.values.shape[0],) + t_eval.shape)


class EccentricResonantTidalGA:
    """
    Eccentric inspiral + selective tidal harmonic excitation + dissipative two-level dynamics.

    The orbital sector follows the Peters equations for a(t), e(t), and the mean anomaly Phi(t).
    The cloud sector follows the n-th harmonic RWA for a user-selected resonance harmonic.
    At the same time, the full harmonic comb is computed and exposed for diagnostics.
    """


    def __init__(
        self,
        # --- Source parameters: black-hole mass, companion mass, and cloud parameters ---
        M_bh=0.001,
        M_star=0.0001,
        alpha=DEFAULT_HIGHFREQ_ALPHA,
        distance_Mpc=0.001,
        z=0.0,
        e_init=0.65,
        f_orb_init=None,
        resonance_harmonic=1,
        max_harmonic=30,
        multi_harmonic_drive=True,
        harmonics_to_keep=None,
        binary_harmonics=12,
        transition_frequency_hz=None,
        eta_ref_hz=None,
        Gamma_decay_hz=None,
        transition_family="fine",
        initial_state=None,
        final_state=None,
        bh_spin=DEFAULT_HIGHFREQ_BH_SPIN,
        orbital_start_ratio=0.95,
        use_formula_eta=True,
        eta_model="finite_separation_fourier",
        overlap_grid_points=4096,
        overlap_max_x=5.0e3,
        tidal_l=2,
        tidal_m=0,
        cloud_mass_fraction=5.0e-2,
        geom_factor=None,
        observer_theta=np.pi / 2.0,
        observer_phi=0.0,
        include_orbital_backreaction=True,
        cloud_evolution_mode="event_lz_impulse",
        lz_window_widths=8.0,
        max_resonance_events=16,
        backreaction_gate_mode="landau_zener",
        backreaction_gate_width_factor=0.25,
        hansen_e_samples=48,
        hansen_M_samples=1024,
        parallel_workers=None,
        save_figure_dir="figures",
        save_figure_formats=("pdf",),
        save_frequency_data_dir="frequency_data",
        save_time_series_data_dir="waveform_data",
        module_stem=None,
        direction_tag="waveform",
        **legacy_unused_options,
    ):
        self.G = 6.6743e-11
        self.c = 2.99792458e8
        self.M_sun = 1.98847e30
        self.Mpc = 3.085677581491367e22
        self.AU = 1.495978707e11
        self.yr = 365.25 * 24.0 * 3600.0
        self.hbar = 1.054571817e-34
        self.eV = 1.602176634e-19

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
        self.use_formula_eta = bool(use_formula_eta)
        self.eta_model = str(eta_model or "finite_separation_fourier").lower()
        allowed_eta_models = {
            "finite_separation_fourier",
            "semimajor_finite_overlap_hansen",
            "powerlaw_hansen",
        }
        if self.eta_model not in allowed_eta_models:
            raise ValueError(f"eta_model must be one of {sorted(allowed_eta_models)}.")
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
            self.f_orb_init = self.orbital_start_ratio * requested_transition_frequency_hz / self.resonance_harmonic
        else:
            self.f_orb_init = float(f_orb_init)
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
        self._manual_eta_ref_hz = eta_ref_hz is not None
        self.eta_ref_hz = None if eta_ref_hz is None else float(eta_ref_hz)
        self.Gamma_decay_hz = float(Gamma_decay_hz if Gamma_decay_hz is not None else auto_gamma_hz)
        self.eta_ref = None
        self.Gamma_decay = 2.0 * np.pi * self.Gamma_decay_hz

        self.tidal_l = int(tidal_l)
        self.tidal_m = int(tidal_m)
        # The harmonic comb is stored for positive orbital overtones n=1..N.
        # For negative-m tidal components the positive-frequency coefficient is
        # the conjugate of the +|m| component.  Use |m| for the Hansen magnitude;
        # the signed transition Delta m is kept separately for angular-momentum
        # backreaction.
        self.hansen_tidal_m = abs(self.tidal_m)
        self.radial_power = self.tidal_l + 1

        self.r_g = self.G * self.M / self.c**2
        self.r_c = self.r_g / (self.alpha**2)
        self.cloud_mass_fraction = cloud_mass_fraction
        self.Mc_max = cloud_mass_fraction * self.M
        self.observer_theta = float(observer_theta)
        self.observer_phi = float(observer_phi)
        self._auto_geom_factor = geom_factor is None
        self.geom_factor = None if geom_factor is None else float(geom_factor)
        self.include_orbital_backreaction = bool(include_orbital_backreaction)
        self.cloud_evolution_mode = str(cloud_evolution_mode or "event_lz_impulse").lower()
        self.lz_window_widths = float(lz_window_widths)
        self.max_resonance_events = int(max_resonance_events)
        self.backreaction_gate_mode = str(backreaction_gate_mode or "off").lower()
        self.backreaction_gate_width_factor = float(backreaction_gate_width_factor)
        self.delta_m_transition = float(
            self.transition_solver_data["final_state"][2] - self.transition_solver_data["initial_state"][2]
        )
        self.backreaction_macro_scale = (self.G * self.M * self.Mc_max) / (self.alpha * self.c)
        # Signed cloud energy change per transition, DeltaE_cloud = E_final - E_initial.
        # The orbital RHS already applies the opposite sign, so downward transitions
        # (negative DeltaE_cloud) feed energy back into the orbit.
        self.delta_E_orbit_backreaction = self.backreaction_macro_scale * self.transition_energy_change_omega
        # For Δm = 0 Bohr transitions this reduces to zero angular-momentum exchange.
        self.delta_L_orbit_backreaction = self.backreaction_macro_scale * self.delta_m_transition
        self.delta_E_high_low_backreaction = self.backreaction_macro_scale * self.transition_omega
        self.delta_m_high_low = self.transition_energy_sign * self.delta_m_transition

        self.hansen_e_samples = int(hansen_e_samples)
        self.hansen_M_samples = int(hansen_M_samples)
        self._hansen_e_grid = None
        self._hansen_real = None
        self._hansen_imag = None
        self._fourier_mean_anomaly = None
        self._fourier_phase_matrix = None

        self.transition_geometry = self._compute_transition_geometry()
        if self.geom_factor is None:
            self.geom_factor = self.transition_geometry.waveform_geom_factor

        self._binary_coeff_norm = float(self._binary_strain_coeff(2, np.array([0.0]))[0])
        self.parallel_workers = max(1, int(1 if parallel_workers is None else parallel_workers))
        self.module_stem = str(module_stem or Path(__file__).stem)
        self.direction_tag = str(direction_tag or "waveform")
        self.figure_dir = SCRIPT_DIR / save_figure_dir
        self.figure_formats = tuple(fmt for fmt in save_figure_formats if str(fmt).lower() == "pdf") or ("pdf",)
        self.frequency_data_dir = None if save_frequency_data_dir is None else SCRIPT_DIR / save_frequency_data_dir
        self.waveform_data_dir = None if save_time_series_data_dir is None else SCRIPT_DIR / save_time_series_data_dir
        # Precompute angular/radial overlaps once; eta(a,e) will later use interpolation.
        self.mixing_overlap_data = self._precompute_mixing_overlaps()
        if self.eta_ref_hz is None:
            self.eta_ref_hz = self._formula_eta_hz(self.a_init)
        self.eta_ref = 2.0 * np.pi * self.eta_ref_hz

    def _peters_rhs(self, a, e):
        a_floor = max(float(getattr(self, "r_g", 0.0)), 1.0e-30)
        a = float(a)
        if not np.isfinite(a) or a <= a_floor:
            a = a_floor
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

    def _backreaction_gate(self, omega_val, eta_vec, active_indices, active_harmonics):
        if self.backreaction_gate_mode in {"off", "none", "disabled", "continuous"}:
            return 1.0

        detunings = np.asarray(active_harmonics, dtype=float) * float(omega_val) - self.transition_omega
        closest_local = int(np.argmin(np.abs(detunings)))
        detuning_abs = float(abs(detunings[closest_local]))
        eta_width = float(abs(eta_vec[active_indices[closest_local]]))
        width = max(abs(self.backreaction_gate_width_factor) * eta_width, 1.0e-30)

        if self.backreaction_gate_mode in {"landau_zener", "lz", "gaussian"}:
            x = detuning_abs / width
            if x > 40.0:
                return 0.0
            return float(np.exp(-0.5 * x * x))

        if self.backreaction_gate_mode in {"lorentzian", "breit_wigner"}:
            x = detuning_abs / width
            return float(1.0 / (1.0 + x * x))

        raise ValueError(f"Unknown backreaction_gate_mode '{self.backreaction_gate_mode}'.")

    def _coupled_rhs(self, _, y_val, active_indices, active_harmonics):
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

        d_cg = -1j * (0.5 * detuning * cg + eta_drive * ce_tilde)
        d_ce = -1j * (np.conj(eta_drive) * cg - 0.5 * detuning * ce_tilde) - self.Gamma_decay * ce_tilde

        dadt = dadt_peters
        dedt = dedt_peters
        if self.include_orbital_backreaction:
            overlap = np.conj(cg) * ce_tilde
            if self.multi_harmonic_drive:
                eta_backreaction = eta_drive
            else:
                eta_backreaction = eta_vec[self.harmonic_to_index[self.resonance_harmonic]]
            final_state_rate = -2.0 * np.imag(eta_backreaction * overlap)
            high_state_rate = self.transition_energy_sign * final_state_rate
            high_state_rate *= self._backreaction_gate(omega_val, eta_vec, active_indices, active_harmonics)
            reduced_mass = self.M * self.M_star / self.M_tot
            one_minus_e2 = max(1.0e-12, 1.0 - e * e)
            l_orb = reduced_mass * np.sqrt(self.G * self.M_tot * a * one_minus_e2)
            # Orbit loses the high-minus-low cloud energy and angular momentum
            # when the high-level population grows; downward transitions have
            # high_state_rate < 0 and therefore feed energy back to the orbit.
            energy_coeff = (a * self.delta_E_high_low_backreaction) / max(self.G * self.M * self.M_star, 1.0e-60)
            angular_coeff = self.backreaction_macro_scale * self.delta_m_high_low / max(l_orb, 1.0e-60)
            dadt -= (2.0 * a * energy_coeff) * high_state_rate
            e_safe = max(e, 1.0e-12)
            dedt -= ((1.0 - e**2) / e_safe) * (energy_coeff - angular_coeff) * high_state_rate

        return [dadt, dedt, omega_val, d_cg.real, d_cg.imag, d_ce.real, d_ce.imag]

    def _resolve_transition_states(self):
        # Default states are kept consistent with the parameter solver and paper text:
        # fine: |322> -> |300>, hyperfine: |211> -> |21-1>
        if self.initial_state is not None and self.final_state is not None:
            return tuple(self.initial_state), tuple(self.final_state)

        if self.transition_family == "fine":
            return (3, 2, 2), (3, 0, 0)
        if self.transition_family == "hyperfine":
            return (2, 1, 1), (2, 1, -1)
        if self.transition_family == "bohr":
            return (6, 4, 4), (5, 4, 4)

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

        The paper gives the orbital resonance frequency. This code stores the two-level
        transition frequency matched by n_res * f_orb, so the returned value includes
        the selected resonance harmonic.
        """
        n_i, l_i, m_i = initial_state
        n_f, l_f, m_f = final_state
        if n_i != n_f or l_i != l_f or m_i == m_f or l_i <= 0:
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
        # Numerically evaluate the omega_R and Gamma formulas from the parameter solver:
        # 1. DeltaE from the real-frequency splitting of the chosen states
        # 2. Gamma from the final-state black-hole absorption
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
                    "Use a fine/Bohr transition or provide transition_frequency_hz explicitly."
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

    def _compute_quadrupole_matrix_dimensionless(self, initial_state, final_state):
        """
        Dimensionless quadrupole matrix for the transition, following the direct
        radial/angular integration strategy used in 辐射角解析6d5d3d2s.py.
        """
        x_grid = np.logspace(-6, np.log10(self.overlap_max_x), max(512, self.overlap_grid_points))
        radial_i = self._radial_wavefunction_dimensionless(initial_state, x_grid)
        radial_f = self._radial_wavefunction_dimensionless(final_state, x_grid)
        radial_integral = np.trapezoid((x_grid**4) * radial_i * radial_f, x_grid)

        n_i, l_i, m_i = initial_state
        n_f, l_f, m_f = final_state
        theta = np.linspace(0.0, np.pi, 180)
        phi = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        y_i = spherical_harmonic(m_i, l_i, phi_grid, theta_grid)
        y_f = spherical_harmonic(m_f, l_f, phi_grid, theta_grid)

        n_components = (
            np.sin(theta_grid) * np.cos(phi_grid),
            np.sin(theta_grid) * np.sin(phi_grid),
            np.cos(theta_grid),
        )
        matrix = np.zeros((3, 3), dtype=np.complex128)
        for i in range(3):
            for j in range(3):
                angular_integrand = np.conj(y_i) * y_f * n_components[i] * n_components[j] * np.sin(theta_grid)
                phi_integral = np.trapezoid(angular_integrand, phi, axis=1)
                matrix[i, j] = radial_integral * np.trapezoid(phi_integral, theta)
        return matrix

    def _compute_transition_geometry(self):
        return compute_transition_geometry(
            self.transition_solver_data["initial_state"],
            self.transition_solver_data["final_state"],
            self._radial_wavefunction_dimensionless,
            overlap_max_x=self.overlap_max_x,
            overlap_grid_points=self.overlap_grid_points,
            observer_theta=self.observer_theta,
            observer_phi=self.observer_phi,
        )

    def _compute_geom_factor(self):
        """
        Observer-direction-dependent GW amplitude factor from the transition quadrupole.

        For axisymmetric Bohr transitions with Δm = 0, use the dedicated pulsation
        kernel proportional to (3 cos^2 θ - 1) instead of the generic TT projector.
        """
        return self._compute_transition_geometry().waveform_geom_factor

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
        # Precompute the radial/angular overlaps entering the eta estimate.
        # Later calls only interpolate with the instantaneous orbital radius.
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
        # Numerical implementation of Eq. (A.6) in 2503.18121.
        # Returns eta/(2pi) in SI units, i.e. in Hz.
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

    def _finite_overlap_terms(self, semi_major_axis, x_star):
        q_mass = self.M_star / self.M
        orbital_omega = np.sqrt(self.G * self.M_tot / semi_major_axis**3)
        m_omega = self.G * self.M * orbital_omega / self.c**3
        x_star = np.clip(
            np.asarray(x_star, dtype=float),
            self.mixing_overlap_data["x_grid"][0],
            self.mixing_overlap_data["x_grid"][-1],
        )

        i_in = np.interp(
            x_star,
            self.mixing_overlap_data["x_grid"],
            self.mixing_overlap_data["inner_cumulative"],
        )
        i_out = np.interp(
            x_star,
            self.mixing_overlap_data["x_grid"],
            self.mixing_overlap_data["outer_cumulative"],
        )

        term_inner = q_mass * m_omega * i_in / (self.alpha**3 * (1.0 + q_mass))
        term_outer = (
            self.alpha**7
            * q_mass
            * (1.0 + q_mass) ** (2.0 / 3.0)
            * i_out
            / max(m_omega, 1.0e-30) ** (7.0 / 3.0)
        )
        return term_inner + term_outer

    def _ensure_orbital_fourier_grid(self):
        if self._fourier_mean_anomaly is not None and self._fourier_phase_matrix is not None:
            return
        mean_anomaly = np.linspace(0.0, 2.0 * np.pi, self.hansen_M_samples, endpoint=False)
        self._fourier_mean_anomaly = mean_anomaly
        self._fourier_phase_matrix = np.exp(1j * np.outer(self.harmonics, mean_anomaly))

    def _finite_separation_fourier_coeffs(self, semi_major_axis, eccentricity):
        self._ensure_orbital_fourier_grid()
        mean_anomaly = self._fourier_mean_anomaly
        eccentricity = float(np.clip(eccentricity, 0.0, 0.999))
        eccentric_anomaly = solve_kepler(mean_anomaly, eccentricity)
        radial_ratio = 1.0 - eccentricity * np.cos(eccentric_anomaly)
        cos_true = (np.cos(eccentric_anomaly) - eccentricity) / radial_ratio
        sin_true = np.sqrt(max(1.0e-14, 1.0 - eccentricity**2)) * np.sin(eccentric_anomaly) / radial_ratio
        true_anomaly = np.arctan2(sin_true, cos_true)

        x_star = (float(semi_major_axis) / self.r_c) * radial_ratio
        finite_overlap = self._finite_overlap_terms(float(semi_major_axis), x_star)
        base = finite_overlap * np.exp(-1j * self.hansen_tidal_m * true_anomaly)
        return self._fourier_phase_matrix @ base / mean_anomaly.size

    def _ensure_hansen_table(self):
        if self._hansen_e_grid is not None:
            return

        self._ensure_orbital_fourier_grid()
        e_max = min(0.95, max(0.05, self.e_init))
        self._hansen_e_grid = np.linspace(0.0, e_max, self.hansen_e_samples)
        self._hansen_real = np.zeros((len(self.harmonics), self.hansen_e_samples))
        self._hansen_imag = np.zeros((len(self.harmonics), self.hansen_e_samples))

        mean_anomaly = self._fourier_mean_anomaly
        phase_matrix = self._fourier_phase_matrix

        for idx, ecc in enumerate(self._hansen_e_grid):
            eccentric_anomaly = solve_kepler(mean_anomaly, ecc)
            radial_ratio = 1.0 - ecc * np.cos(eccentric_anomaly)
            cos_true = (np.cos(eccentric_anomaly) - ecc) / radial_ratio
            sin_true = np.sqrt(max(1.0e-14, 1.0 - ecc**2)) * np.sin(eccentric_anomaly) / radial_ratio
            true_anomaly = np.arctan2(sin_true, cos_true)

            # Coefficients of (a/R)^(l+1) exp(-i m f) =
            # sum_n X_n(e) exp(-i n M).  For m=0 this is the radial Hansen comb.
            base = radial_ratio ** (-self.radial_power) * np.exp(-1j * self.hansen_tidal_m * true_anomaly)
            coeffs = phase_matrix @ base / mean_anomaly.size

            self._hansen_real[:, idx] = coeffs.real
            self._hansen_imag[:, idx] = coeffs.imag

    def _interp_hansen(self, eccentricity):
        self._ensure_hansen_table()
        ecc = float(np.clip(eccentricity, self._hansen_e_grid[0], self._hansen_e_grid[-1]))
        real = np.array([np.interp(ecc, self._hansen_e_grid, row) for row in self._hansen_real])
        imag = np.array([np.interp(ecc, self._hansen_e_grid, row) for row in self._hansen_imag])
        return real + 1j * imag

    def _eta_vector(self, semi_major_axis, eccentricity):
        if self._manual_eta_ref_hz:
            hansen = self._interp_hansen(eccentricity)
            scale = (self.a_init / semi_major_axis) ** self.radial_power
            eta_base = self.eta_ref * scale
            return eta_base * hansen
        if self.eta_model == "powerlaw_hansen":
            hansen = self._interp_hansen(eccentricity)
            scale = (self.a_init / semi_major_axis) ** self.radial_power
            return self.eta_ref * scale * hansen
        if self.eta_model == "semimajor_finite_overlap_hansen":
            hansen = self._interp_hansen(eccentricity)
            # This is a semimajor-axis anchored finite-separation overlap
            # multiplied by eccentric Hansen harmonics.  It is not a full
            # Fourier decomposition of the finite-separation kernel along an
            # eccentric orbit.
            eta_base = 2.0 * np.pi * self._formula_eta_hz(semi_major_axis)
            return eta_base * hansen
        coeffs = self._finite_separation_fourier_coeffs(semi_major_axis, eccentricity)
        orbital_omega = np.sqrt(self.G * self.M_tot / float(semi_major_axis) ** 3)
        eta_over_omega = (3.0 * np.pi / 10.0) * self.mixing_overlap_data["angular_overlap"] * coeffs
        return orbital_omega * eta_over_omega

    def _choose_transition_frequency(self, orbit):
        if self.transition_frequency_hz is not None:
            return 2.0 * np.pi * float(self.transition_frequency_hz)

        omega_start = orbit["omega"][0]
        omega_end = orbit["omega"][-1]
        return self.resonance_harmonic * 0.5 * (omega_start + omega_end)

    def solve_orbit(self, duration_yr=1.0, secular_samples=3000):
        total_time = duration_yr * self.yr
        max_step = min(total_time / 2500.0, 10.0 * self.period_init)

        sol = solve_ivp(
            self._orbit_rhs,
            (0.0, total_time),
            [self.a_init, self.e_init, 0.0],
            dense_output=True,
            max_step=max_step,
            rtol=1.0e-9,
            atol=[1.0e-6, 1.0e-12, 1.0e-9],
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

    def _solve_peters_segment(self, t_start, t_stop, y_start, max_step=None):
        if t_stop <= t_start:
            return None
        if max_step is None:
            max_step = min((t_stop - t_start) / 500.0, 10.0 * self.period_init)
        return solve_ivp(
            self._orbit_rhs,
            (float(t_start), float(t_stop)),
            [float(y_start[0]), float(y_start[1]), float(y_start[2])],
            dense_output=True,
            max_step=max_step,
            rtol=1.0e-9,
            atol=[1.0e-6, 1.0e-12, 1.0e-9],
        )

    def _orbit_dict_from_solution(self, sol, t_start, t_stop, samples):
        t_grid = np.linspace(float(t_start), float(t_stop), int(max(16, samples)))
        a, e, phi = sol.sol(t_grid)
        e = np.clip(e, 0.0, 0.999)
        omega = np.sqrt(self.G * self.M_tot / np.maximum(a, 1.0e-30) ** 3)
        return {
            "solution": sol,
            "t": t_grid,
            "a": a,
            "e": e,
            "phi": phi,
            "omega": omega,
            "f_orb": omega / (2.0 * np.pi),
        }

    def _find_next_event_on_segment(self, sol, t_start, t_stop, active_harmonics, min_time):
        if t_stop <= min_time:
            return None
        samples = max(600, int(1800 * (t_stop - t_start) / max(t_stop, self.period_init)))
        samples = min(samples, 6000)
        segment_orbit = self._orbit_dict_from_solution(sol, t_start, t_stop, samples)
        events = self._build_resonance_events_from_orbit(segment_orbit)
        candidates = [
            event
            for event in events
            if event["crossed"]
            and int(event["harmonic"]) in set(int(n) for n in active_harmonics)
            and float(event["t_source"]) > float(min_time)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: item["t_source"])
        return candidates[0]

    def _selected_pure_resonance_event(self, duration_yr, secular_samples=1600):
        orbit = self.solve_orbit(duration_yr=float(duration_yr), secular_samples=int(secular_samples))
        events = self._build_resonance_events_from_orbit(orbit)
        selected = [event for event in events if event.get("selected", False)]
        if not selected:
            return None, orbit
        selected.sort(key=lambda item: (not bool(item.get("crossed", False)), float(item.get("t_source", np.inf))))
        return selected[0], orbit

    def recommended_duration_to_cover_selected_resonance(
        self,
        initial_duration_yr=2.0e-8,
        max_duration_yr=5.0e-6,
        post_event_padding_orbits=80.0,
    ):
        duration = float(initial_duration_yr)
        event = None
        for _ in range(16):
            event, _ = self._selected_pure_resonance_event(duration)
            if event is not None and event.get("crossed", False):
                omega_event = self.resonance_harmonic and (self.transition_omega / max(self.resonance_harmonic, 1))
                local_period = 2.0 * np.pi / max(float(omega_event), 1.0e-300)
                target_stop = float(event["t_source"]) + float(post_event_padding_orbits) * local_period
                return min(max(duration, 1.05 * target_stop / self.yr), float(max_duration_yr))
            if duration >= max_duration_yr:
                return duration
            duration = min(2.0 * duration, float(max_duration_yr))
        return duration

    def _decay_cloud_state(self, state, dt):
        cg, ce = complex(state[0]), complex(state[1])
        if dt > 0.0:
            ce *= np.exp(-self.Gamma_decay * float(dt))
        return cg, ce

    def _append_segment_samples(
        self,
        store,
        t_values,
        a_values,
        e_values,
        phi_values,
        cg_values,
        ce_values,
    ):
        t_values = np.asarray(t_values, dtype=float)
        if t_values.size == 0:
            return
        if store["t"] and np.isclose(float(t_values[0]), float(store["t"][-1]), rtol=0.0, atol=1.0e-14):
            start = 1
        else:
            start = 0
        if start >= t_values.size:
            return
        store["t"].extend(t_values[start:].tolist())
        store["a"].extend(np.asarray(a_values, dtype=float)[start:].tolist())
        store["e"].extend(np.asarray(e_values, dtype=float)[start:].tolist())
        store["phi"].extend(np.asarray(phi_values, dtype=float)[start:].tolist())
        store["cg"].extend(np.asarray(cg_values, dtype=np.complex128)[start:].tolist())
        store["ce"].extend(np.asarray(ce_values, dtype=np.complex128)[start:].tolist())

    def _append_peters_cloud_segment(self, store, sol, t_start, t_stop, state, samples):
        if t_stop <= t_start:
            return state
        t_values = np.linspace(float(t_start), float(t_stop), int(max(2, samples)))
        a_values, e_values, phi_values = sol.sol(t_values)
        cg0, ce0 = complex(state[0]), complex(state[1])
        cg_values = np.full(t_values.size, cg0, dtype=np.complex128)
        ce_values = ce0 * np.exp(-self.Gamma_decay * (t_values - float(t_start)))
        self._append_segment_samples(store, t_values, a_values, np.clip(e_values, 0.0, 0.999), phi_values, cg_values, ce_values)
        return complex(cg_values[-1]), complex(ce_values[-1])

    def _local_lz_half_width(self, event):
        eta_abs = max(float(event.get("eta_abs", 0.0)), 1.0e-30)
        slope_abs = max(float(event.get("detuning_slope_abs", 0.0)), 1.0e-300)
        return max(abs(self.lz_window_widths) * eta_abs / slope_abs, 4.0 * self.period_init)

    def _solve_local_lz_event(self, event, orbit_event_state, cloud_state, t_start, t_stop):
        harmonic = int(event["harmonic"])
        harmonic_idx = self.harmonic_to_index[harmonic]
        t_event = float(event["t_source"])
        slope = max(float(event.get("detuning_slope_abs", 0.0)), 1.0e-300)
        eta_vec = self._eta_vector(float(orbit_event_state[0]), float(orbit_event_state[1]))
        phase = np.exp(-1j * (harmonic - self.resonance_harmonic) * float(orbit_event_state[2]))
        eta_event = eta_vec[harmonic_idx] * phase

        cg0, ce0 = self._decay_cloud_state(cloud_state, max(0.0, t_start - float(event.get("state_time", t_start))))

        def rhs(t_val, y_val):
            cg = y_val[0] + 1j * y_val[1]
            ce = y_val[2] + 1j * y_val[3]
            detuning = slope * (float(t_val) - t_event)
            d_cg = -1j * (0.5 * detuning * cg + eta_event * ce)
            d_ce = -1j * (np.conj(eta_event) * cg - 0.5 * detuning * ce) - self.Gamma_decay * ce
            return [d_cg.real, d_cg.imag, d_ce.real, d_ce.imag]

        if t_stop <= t_start:
            return np.array([t_start]), np.array([cg0]), np.array([ce0])
        sol = solve_ivp(
            rhs,
            (float(t_start), float(t_stop)),
            [cg0.real, cg0.imag, ce0.real, ce0.imag],
            dense_output=True,
            method="DOP853",
            rtol=1.0e-11,
            atol=[1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12],
            max_step=max((t_stop - t_start) / 500.0, 1.0e-8),
        )
        n_samples = int(max(64, min(4096, np.ceil((t_stop - t_start) / max(self.period_init, 1.0e-12) * 24))))
        t_values = np.linspace(float(t_start), float(t_stop), n_samples)
        cg_r, cg_i, ce_r, ce_i = sol.sol(t_values)
        return t_values, cg_r + 1j * cg_i, ce_r + 1j * ce_i

    def _high_state_population(self, cg, ce):
        return float(abs(ce) ** 2 if self.transition_energy_sign > 0.0 else abs(cg) ** 2)

    def _apply_orbital_impulse(self, a, e, delta_high_population):
        if not self.include_orbital_backreaction or abs(delta_high_population) <= 0.0:
            return float(a), float(e)
        reduced_mass = self.M * self.M_star / self.M_tot
        one_minus_e2 = max(1.0e-12, 1.0 - float(e) * float(e))
        energy_orbit = -self.G * self.M * self.M_star / (2.0 * float(a))
        angular_orbit = reduced_mass * np.sqrt(self.G * self.M_tot * float(a) * one_minus_e2)
        # Same convention as the continuous RHS: a positive delta_high_population
        # means the cloud stores the high-minus-low energy/angular momentum.
        energy_orbit -= self.delta_E_high_low_backreaction * float(delta_high_population)
        angular_orbit -= self.backreaction_macro_scale * self.delta_m_high_low * float(delta_high_population)
        if energy_orbit >= -1.0e-300:
            return float(a), float(e)
        a_new = -self.G * self.M * self.M_star / (2.0 * energy_orbit)
        ecc_arg = 1.0 - angular_orbit**2 / max(reduced_mass**2 * self.G * self.M_tot * a_new, 1.0e-300)
        e_new = float(np.sqrt(np.clip(ecc_arg, 0.0, 0.999**2)))
        return float(a_new), e_new

    def solve_event_based_system(self, duration_yr=1.0, secular_samples=3000):
        total_time = duration_yr * self.yr
        resonance_idx, active_indices, active_harmonics = self._select_active_harmonic_indices()
        max_step = min(total_time / 2500.0, 10.0 * self.period_init)
        t_current = 0.0
        y_current = np.array([self.a_init, self.e_init, 0.0], dtype=float)
        cloud_state = (1.0 + 0.0j, 0.0 + 0.0j)
        state_time = 0.0
        processed_events = []
        min_event_time = 0.0
        store = {"t": [], "a": [], "e": [], "phi": [], "cg": [], "ce": []}

        for _ in range(max(1, self.max_resonance_events)):
            if t_current >= total_time:
                break
            trial = self._solve_peters_segment(t_current, total_time, y_current, max_step=max_step)
            if trial is None:
                break
            event = self._find_next_event_on_segment(
                trial,
                t_current,
                total_time,
                active_harmonics,
                min_event_time,
            )
            if event is None:
                remaining_samples = max(32, int(secular_samples * (total_time - t_current) / max(total_time, 1.0)))
                cloud_state = self._append_peters_cloud_segment(
                    store,
                    trial,
                    t_current,
                    total_time,
                    cloud_state,
                    remaining_samples,
                )
                t_current = total_time
                state_time = total_time
                break

            t_event = float(event["t_source"])
            half_width = self._local_lz_half_width(event)
            t_local_start = max(t_current, t_event - half_width)
            t_local_stop = min(total_time, t_event + half_width)
            pre_samples = max(16, int(secular_samples * max(t_local_start - t_current, 0.0) / max(total_time, 1.0)))
            cloud_state = self._append_peters_cloud_segment(
                store,
                trial,
                t_current,
                t_local_start,
                cloud_state,
                pre_samples,
            )
            state_time = t_local_start

            orbit_pre = self._solve_peters_segment(t_local_start, t_event, trial.sol(t_local_start), max_step=max_step)
            if orbit_pre is None:
                y_event = np.asarray(trial.sol(t_event), dtype=float)
            else:
                y_event = np.asarray(orbit_pre.sol(t_event), dtype=float)

            event_with_state = dict(event)
            event_with_state["state_time"] = state_time
            high_before = self._high_state_population(*cloud_state)
            t_lz, cg_lz, ce_lz = self._solve_local_lz_event(
                event_with_state,
                y_event,
                cloud_state,
                t_local_start,
                t_local_stop,
            )
            high_after = self._high_state_population(cg_lz[-1], ce_lz[-1])
            delta_high = high_after - high_before
            a_after, e_after = self._apply_orbital_impulse(y_event[0], y_event[1], delta_high)

            if t_event > t_local_start:
                t_pre = t_lz[t_lz <= t_event]
                if t_pre.size:
                    if orbit_pre is None:
                        a_pre, e_pre, phi_pre = trial.sol(t_pre)
                    else:
                        a_pre, e_pre, phi_pre = orbit_pre.sol(t_pre)
                    self._append_segment_samples(
                        store,
                        t_pre,
                        a_pre,
                        np.clip(e_pre, 0.0, 0.999),
                        phi_pre,
                        cg_lz[: t_pre.size],
                        ce_lz[: t_pre.size],
                    )

            post_mask = t_lz > t_event
            y_post_start = np.array([a_after, e_after, y_event[2]], dtype=float)
            if np.any(post_mask):
                orbit_post = self._solve_peters_segment(t_event, t_local_stop, y_post_start, max_step=max_step)
                t_post = t_lz[post_mask]
                if orbit_post is None:
                    a_post = np.full(t_post.size, a_after)
                    e_post = np.full(t_post.size, e_after)
                    phi_post = np.full(t_post.size, y_event[2])
                else:
                    a_post, e_post, phi_post = orbit_post.sol(t_post)
                self._append_segment_samples(
                    store,
                    t_post,
                    a_post,
                    np.clip(e_post, 0.0, 0.999),
                    phi_post,
                    cg_lz[post_mask],
                    ce_lz[post_mask],
                )
                y_current = np.asarray([a_post[-1], e_post[-1], phi_post[-1]], dtype=float)
            else:
                y_current = y_post_start

            cloud_state = (complex(cg_lz[-1]), complex(ce_lz[-1]))
            t_current = t_local_stop
            state_time = t_current
            min_event_time = t_event + max(0.5 * half_width, 4.0 * self.period_init)
            processed = dict(event)
            processed["lz_window_start_source"] = t_local_start
            processed["lz_window_stop_source"] = t_local_stop
            processed["lz_half_width_source"] = half_width
            processed["delta_high_population"] = float(delta_high)
            processed["a_before_impulse"] = float(y_event[0])
            processed["e_before_impulse"] = float(y_event[1])
            processed["a_after_impulse"] = float(a_after)
            processed["e_after_impulse"] = float(e_after)
            processed["delta_a_over_a"] = float((a_after - y_event[0]) / max(abs(y_event[0]), 1.0e-300))
            processed["delta_e"] = float(e_after - y_event[1])
            processed["lz_probability_estimate"] = float(1.0 - np.exp(-2.0 * np.pi * float(event["z_ad"])))
            processed_events.append(processed)
        else:
            if t_current < total_time:
                tail = self._solve_peters_segment(t_current, total_time, y_current, max_step=max_step)
                if tail is not None:
                    self._append_peters_cloud_segment(store, tail, t_current, total_time, cloud_state, 64)

        if not store["t"]:
            store["t"] = [0.0]
            store["a"] = [self.a_init]
            store["e"] = [self.e_init]
            store["phi"] = [0.0]
            store["cg"] = [1.0 + 0.0j]
            store["ce"] = [0.0 + 0.0j]

        t_grid = np.asarray(store["t"], dtype=float)
        order = np.argsort(t_grid)
        t_grid = t_grid[order]
        a = np.asarray(store["a"], dtype=float)[order]
        e = np.clip(np.asarray(store["e"], dtype=float)[order], 0.0, 0.999)
        phi = np.asarray(store["phi"], dtype=float)[order]
        cg = np.asarray(store["cg"], dtype=np.complex128)[order]
        ce_tilde = np.asarray(store["ce"], dtype=np.complex128)[order]
        unique = np.concatenate(([True], np.diff(t_grid) > 1.0e-14))
        t_grid, a, e, phi, cg, ce_tilde = t_grid[unique], a[unique], e[unique], phi[unique], cg[unique], ce_tilde[unique]
        omega = np.sqrt(self.G * self.M_tot / np.maximum(a, 1.0e-30) ** 3)
        eta_series = np.zeros((len(self.harmonics), t_grid.size), dtype=np.complex128)
        for idx, (a_val, e_val) in enumerate(zip(a, e)):
            eta_series[:, idx] = self._eta_vector(a_val, e_val)
        detuning_series = self.harmonics[:, None] * omega[None, :] - self.transition_omega
        closest_harmonic = self.harmonics[np.argmin(np.abs(detuning_series), axis=0)]
        overlap = np.conj(cg) * ce_tilde
        selected_final_state_rate = np.zeros_like(t_grid)
        selected_high_state_rate = np.zeros_like(t_grid)
        backreaction_gate = np.zeros_like(t_grid)
        for event in processed_events:
            mask = (t_grid >= event["lz_window_start_source"]) & (t_grid <= event["lz_window_stop_source"])
            backreaction_gate[mask] = 1.0

        orbit = {
            "solution": _TabulatedSolution(t_grid, np.vstack((a, e, phi))),
            "t": t_grid,
            "a": a,
            "e": e,
            "phi": phi,
            "omega": omega,
            "f_orb": omega / (2.0 * np.pi),
        }
        cloud = {
            "solution": _TabulatedSolution(
                t_grid,
                np.vstack((cg.real, cg.imag, ce_tilde.real, ce_tilde.imag)),
            ),
            "t": t_grid,
            "cg": cg,
            "ce_tilde": ce_tilde,
            "eta_series": eta_series,
            "detuning_series": detuning_series,
            "closest_harmonic": closest_harmonic,
            "active_harmonics": active_harmonics,
            "resonance_time": processed_events[0]["t_source"] if processed_events else t_grid[np.argmin(np.abs(detuning_series[resonance_idx]))],
            "pop_ground": np.abs(cg) ** 2,
            "pop_excited": np.abs(ce_tilde) ** 2,
            "overlap": overlap,
            "overlap_abs": np.abs(overlap),
            "selected_final_state_rate": selected_final_state_rate,
            "selected_high_state_rate": selected_high_state_rate,
            "backreaction_gate": backreaction_gate,
            "local_lz_events": processed_events,
        }
        final_events = self._build_resonance_events(orbit, cloud)
        processed_harmonics = {int(event["harmonic"]) for event in processed_events}
        resonance_events = [dict(event) for event in processed_events]
        for event in final_events:
            # In event mode, resolved crossings are the processed local LZ events.
            # Crossings introduced by the instantaneous orbital jump are interpolation
            # artifacts and should not be reinterpreted as new physical resonances.
            if int(event["harmonic"]) in processed_harmonics:
                continue
            if not event["crossed"]:
                resonance_events.append(event)
        resonance_events.sort(key=lambda item: (float(item["t_source"]), int(item["harmonic"])))
        cloud["resonance_events"] = resonance_events
        selected_events = [event for event in cloud["resonance_events"] if event["selected"]]
        selected_crossings = [event for event in selected_events if event["crossed"]]
        if selected_crossings:
            cloud["resonance_time"] = float(selected_crossings[0]["t_source"])
        elif processed_events:
            cloud["resonance_time"] = float(processed_events[0]["t_source"])
        elif selected_events:
            cloud["resonance_time"] = float(selected_events[0]["t_source"])
        return orbit, cloud

    def solve_coupled_system(self, duration_yr=1.0, secular_samples=3000):
        if self.cloud_evolution_mode in {"event", "event_lz", "event_lz_impulse", "impulse"}:
            return self.solve_event_based_system(duration_yr=duration_yr, secular_samples=secular_samples)

        total_time = duration_yr * self.yr
        max_step = min(total_time / 2500.0, 10.0 * self.period_init)
        resonance_idx, active_indices, active_harmonics = self._select_active_harmonic_indices()

        sol = solve_ivp(
            lambda t_val, y_val: self._coupled_rhs(
                t_val,
                y_val,
                active_indices,
                active_harmonics,
            ),
            (0.0, total_time),
            [self.a_init, self.e_init, 0.0, 1.0, 0.0, 0.0, 0.0],
            dense_output=True,
            max_step=max_step,
            method="DOP853",
            rtol=1.0e-12,
            atol=[1.0e-9, 1.0e-12, 1.0e-9, 1.0e-12, 1.0e-12, 1.0e-14, 1.0e-14],
        )

        t_grid = np.linspace(0.0, sol.t[-1], int(secular_samples))
        a, e, phi, cg_real, cg_imag, ce_real, ce_imag = sol.sol(t_grid)
        e = np.clip(e, 0.0, 0.999)
        omega = np.sqrt(self.G * self.M_tot / np.maximum(a, 1.0e-30) ** 3)
        cg = cg_real + 1j * cg_imag
        ce_tilde = ce_real + 1j * ce_imag

        eta_series = np.zeros((len(self.harmonics), t_grid.size), dtype=np.complex128)
        for idx, (a_val, e_val) in enumerate(zip(a, e)):
            eta_series[:, idx] = self._eta_vector(a_val, e_val)

        detuning_series = self.harmonics[:, None] * omega[None, :] - self.transition_omega
        closest_harmonic = self.harmonics[np.argmin(np.abs(detuning_series), axis=0)]
        resonance_track = detuning_series[resonance_idx]
        resonance_time = t_grid[np.argmin(np.abs(resonance_track))]
        overlap = np.conj(cg) * ce_tilde
        selected_final_state_rate = -2.0 * np.imag(eta_series[resonance_idx] * overlap)
        selected_high_state_rate = self.transition_energy_sign * selected_final_state_rate
        backreaction_gate = np.ones_like(t_grid, dtype=float)
        if self.include_orbital_backreaction:
            for idx, (omega_val, eta_vec) in enumerate(zip(omega, eta_series.T)):
                backreaction_gate[idx] = self._backreaction_gate(
                    omega_val,
                    eta_vec,
                    active_indices,
                    active_harmonics,
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
            "overlap_abs": np.abs(overlap),
            "selected_final_state_rate": selected_final_state_rate,
            "selected_high_state_rate": selected_high_state_rate,
            "backreaction_gate": backreaction_gate,
        }
        cloud["resonance_events"] = self._build_resonance_events(orbit, cloud)
        selected_events = [event for event in cloud["resonance_events"] if event["selected"]]
        selected_crossings = [event for event in selected_events if event["crossed"]]
        if selected_crossings:
            cloud["resonance_time"] = float(selected_crossings[0]["t_source"])
        elif selected_events:
            cloud["resonance_time"] = float(selected_events[0]["t_source"])
        return orbit, cloud

    def _build_resonance_events(self, orbit, cloud):
        t_grid = np.asarray(orbit["t"], dtype=float)
        detuning_series = np.asarray(cloud["detuning_series"], dtype=float)
        eta_series = np.asarray(cloud["eta_series"], dtype=np.complex128)
        return self._build_resonance_events_from_series(t_grid, detuning_series, eta_series)

    def _build_resonance_events_from_orbit(self, orbit):
        t_grid = np.asarray(orbit["t"], dtype=float)
        omega = np.asarray(orbit["omega"], dtype=float)
        detuning_series = self.harmonics[:, None] * omega[None, :] - self.transition_omega
        eta_series = np.zeros((len(self.harmonics), t_grid.size), dtype=np.complex128)
        for idx, (a_val, e_val) in enumerate(zip(orbit["a"], orbit["e"])):
            eta_series[:, idx] = self._eta_vector(a_val, e_val)
        return self._build_resonance_events_from_series(t_grid, detuning_series, eta_series)

    def _build_resonance_events_from_series(self, t_grid, detuning_series, eta_series):
        if t_grid.size < 2:
            return []

        events = []
        for idx, harmonic in enumerate(self.harmonics):
            det = np.asarray(detuning_series[idx], dtype=float)
            eta_abs = np.abs(eta_series[idx])
            finite = np.isfinite(det) & np.isfinite(t_grid)
            if np.count_nonzero(finite) < 2:
                continue

            crossing_indices = np.where((det[:-1] == 0.0) | (det[:-1] * det[1:] < 0.0))[0]
            crossed = crossing_indices.size > 0
            if crossed:
                left = int(crossing_indices[0])
                right = left + 1
                d0 = det[left]
                d1 = det[right]
                if np.isclose(d0, d1):
                    t_event = float(t_grid[left])
                else:
                    frac = -d0 / (d1 - d0)
                    t_event = float(t_grid[left] + np.clip(frac, 0.0, 1.0) * (t_grid[right] - t_grid[left]))
                closest_position = "cross"
            else:
                closest = int(np.nanargmin(np.abs(det)))
                t_event = float(t_grid[closest])
                if closest == 0:
                    closest_position = "start"
                elif closest == t_grid.size - 1:
                    closest_position = "end"
                else:
                    closest_position = "interior"

            slope_series = np.gradient(det, t_grid, edge_order=1)
            eta_event = float(np.interp(t_event, t_grid, eta_abs))
            slope_event = float(np.interp(t_event, t_grid, slope_series))
            det_event = float(np.interp(t_event, t_grid, det))
            z_ad = eta_event**2 / max(abs(slope_event), 1.0e-300)
            events.append(
                {
                    "harmonic": int(harmonic),
                    "crossed": bool(crossed),
                    "closest_position": closest_position,
                    "t_source": t_event,
                    "t_obs": t_event * (1.0 + self.z),
                    "detuning_abs": abs(det_event),
                    "eta_abs": eta_event,
                    "detuning_slope_abs": abs(slope_event),
                    "z_ad": float(z_ad),
                    "quenched": bool(z_ad >= 1.0),
                    "selected": int(harmonic) == int(self.resonance_harmonic),
                }
            )
        return events

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

    def _binary_amplitude(self, orbital_omega):
        f_orb_obs = orbital_omega / (2.0 * np.pi * (1.0 + self.z))
        f_gw_ref = 2.0 * f_orb_obs
        M_c_z = self.M_chirp * (1.0 + self.z)
        M_c_geom = self.G * M_c_z / self.c**2
        v_char = (np.pi * self.G * M_c_z * f_gw_ref / self.c**3) ** (1.0 / 3.0)
        return (4.0 * M_c_geom / self.d_L) * v_char**2

    def _cloud_amplitude(self):
        # geom_factor is the fiducial projected transition-quadrupole
        # coefficient F used with the 4G prefactor. Inclination and
        # source-angle averages are not applied to this time-domain trace.
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
        overlap_zoom = np.conj(cg_zoom) * ce_zoom

        h_binary = self._binary_strain_time_domain(a_zoom, e_zoom, phi_zoom)
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
            "h_binary": h_binary,
            "h_axion": h_axion,
            "h_total": h_binary + h_axion,
            "overlap_abs": np.abs(overlap_zoom),
            "phi": phi_zoom,
            "e": e_zoom,
            "orbital_period_res": orbital_period_res,
            "orbital_frequency_res_obs": orbital_omega_res / (2.0 * np.pi * (1.0 + self.z)),
            "resonance_time_obs": resonance_time_obs,
        }

    def _evaluate_binary_window_on_grid(self, orbit, t_zoom, resonance_time_source):
        a_zoom, e_zoom, phi_zoom = orbit["solution"].sol(t_zoom)
        h_binary = self._binary_strain_time_domain(a_zoom, e_zoom, phi_zoom)
        orbital_omega_res = np.interp(resonance_time_source, orbit["t"], orbit["omega"])
        orbital_period_res = 2.0 * np.pi / orbital_omega_res
        resonance_time_obs = resonance_time_source * (1.0 + self.z)
        t_obs = np.asarray(t_zoom, dtype=float) * (1.0 + self.z)
        zeros = np.zeros_like(h_binary)
        return {
            "t_source": np.asarray(t_zoom, dtype=float),
            "t_obs": t_obs,
            "t_rel_obs": t_obs - resonance_time_obs,
            "h_binary": h_binary,
            "h_axion": zeros,
            "h_total": h_binary,
            "overlap_abs": zeros,
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
        t_zoom = np.linspace(t_start, t_stop, int(max(16, sample_points)))
        return self._evaluate_waveform_window_on_grid(orbit, cloud, t_zoom)

    def build_waveform_window_on_reference_grid(self, orbit, cloud, reference_t_source):
        return self._evaluate_waveform_window_on_grid(orbit, cloud, np.asarray(reference_t_source, dtype=float))

    def build_waveform_window_for_observation_duration(
        self,
        orbit,
        cloud,
        duration_obs_s,
        sample_points=2400,
        center_time_source=None,
    ):
        center_time_source = cloud["resonance_time"] if center_time_source is None else float(center_time_source)
        duration_source_s = float(duration_obs_s) / (1.0 + self.z)
        half_window = 0.5 * duration_source_s
        t_start = max(0.0, center_time_source - half_window)
        t_stop = min(orbit["t"][-1], center_time_source + half_window)
        if t_stop <= t_start:
            t_zoom = np.array([center_time_source], dtype=float)
        else:
            t_zoom = np.linspace(t_start, t_stop, int(max(16, sample_points)))
        return self._evaluate_waveform_window_on_grid(orbit, cloud, t_zoom)

    def build_waveform_window_from_start(
        self,
        orbit,
        cloud,
        duration_obs_s,
        sample_points=2400,
    ):
        """Sample a waveform starting from the observation start, not from the resonance center."""
        duration_source_s = float(duration_obs_s) / (1.0 + self.z)
        t_start = orbit["t"][0]
        t_stop = min(orbit["t"][-1], t_start + duration_source_s)
        if t_stop <= t_start:
            t_zoom = np.array([t_start], dtype=float)
        else:
            t_zoom = np.linspace(t_start, t_stop, int(max(16, sample_points)))
        return self._evaluate_waveform_window_on_grid(orbit, cloud, t_zoom)

    def build_waveform_window_between(
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
            t_mid = float(np.clip(0.5 * (t_start + t_stop), orbit["t"][0], orbit["t"][-1]))
            t_zoom = np.array([t_mid], dtype=float)
        else:
            t_zoom = np.linspace(t_start, t_stop, int(max(16, sample_points)))
        return self._evaluate_waveform_window_on_grid(orbit, cloud, t_zoom)

    def _summary_window_anchor_events(self, cloud_events, pure_events):
        anchors = []

        def add_anchor(event, source, reason):
            t_source = float(event.get("t_source", np.nan))
            if not np.isfinite(t_source):
                return
            anchors.append(
                {
                    "source": source,
                    "reason": reason,
                    "harmonic": int(event.get("harmonic", -1)),
                    "selected": bool(event.get("selected", False)),
                    "crossed": bool(event.get("crossed", False)),
                    "closest_position": str(event.get("closest_position", "")),
                    "t_source": t_source,
                    "t_obs": t_source * (1.0 + self.z),
                    "z_ad": float(event.get("z_ad", np.nan)),
                    "detuning_abs": float(event.get("detuning_abs", np.nan)),
                }
            )

        cloud_events = list(cloud_events or [])
        pure_events = list(pure_events or [])
        coupled_selected_crosses = [
            event for event in cloud_events if event.get("crossed", False) and event.get("selected", False)
        ]

        for event in cloud_events:
            if event.get("crossed", False):
                add_anchor(event, "coupled", "crossing")
        for event in pure_events:
            if event.get("crossed", False) and event.get("selected", False):
                add_anchor(event, "pure_peters", "selected_crossing")

        # If the coupled selected resonance does not cross, keep its best interior
        # approach in view. This is the diagnostic needed for avoided/shifted crossings
        # such as highfre644v.
        if not coupled_selected_crosses:
            for event in cloud_events:
                if event.get("selected", False) and event.get("closest_position") == "interior":
                    add_anchor(event, "coupled", "selected_interior_closest")

        if anchors:
            return anchors

        for event in pure_events:
            if event.get("selected", False) and event.get("closest_position") == "interior":
                add_anchor(event, "pure_peters", "selected_interior_closest")
        if anchors:
            return anchors

        for event in cloud_events:
            if event.get("closest_position") == "interior":
                add_anchor(event, "coupled", "interior_closest")
        if anchors:
            return anchors

        # Last resort only: start/end closest approaches are not physical resonance
        # locations, but they are better than an arbitrary plotting interval.
        for event in cloud_events:
            if event.get("selected", False):
                add_anchor(event, "coupled", "selected_boundary_closest")
        return anchors

    def _select_orbit_summary_window(
        self,
        orbit,
        cloud,
        pure_events,
        padding_orbits=40.0,
        padding_fraction=0.15,
    ):
        anchors = self._summary_window_anchor_events(cloud.get("resonance_events", []), pure_events)
        total_start = float(orbit["t"][0])
        total_stop = float(orbit["t"][-1])
        if not anchors:
            fallback_time = float(np.clip(cloud.get("resonance_time", total_start), total_start, total_stop))
            anchors = [
                {
                    "source": "coupled",
                    "reason": "fallback_selected_track_min",
                    "harmonic": int(self.resonance_harmonic),
                    "selected": True,
                    "crossed": False,
                    "closest_position": "fallback",
                    "t_source": fallback_time,
                    "t_obs": fallback_time * (1.0 + self.z),
                    "z_ad": np.nan,
                    "detuning_abs": np.nan,
                }
            ]

        anchor_times = np.asarray([anchor["t_source"] for anchor in anchors], dtype=float)
        anchor_start = float(np.nanmin(anchor_times))
        anchor_stop = float(np.nanmax(anchor_times))
        omega_anchor = np.interp(anchor_times, orbit["t"], orbit["omega"])
        local_period = float(np.nanmax(2.0 * np.pi / np.maximum(omega_anchor, 1.0e-300)))
        anchor_span = max(anchor_stop - anchor_start, 0.0)
        padding = max(float(padding_orbits) * local_period, float(padding_fraction) * anchor_span)
        if anchor_span <= 0.0:
            padding = max(padding, 0.015 * max(total_stop - total_start, local_period))

        t_start = max(total_start, anchor_start - padding)
        t_stop = min(total_stop, anchor_stop + padding)
        if t_stop <= t_start:
            t_start = max(total_start, anchor_start - 0.5 * local_period)
            t_stop = min(total_stop, anchor_start + 0.5 * local_period)
        if t_stop <= t_start:
            t_start, t_stop = total_start, total_stop

        return {
            "t_start_source": float(t_start),
            "t_stop_source": float(t_stop),
            "t_start_obs": float(t_start * (1.0 + self.z)),
            "t_stop_obs": float(t_stop * (1.0 + self.z)),
            "anchors": anchors,
            "padding_source_s": float(padding),
        }

    def _select_first_selected_orbit_window(self, orbit, cloud, window_orbits=16.0):
        total_start = float(orbit["t"][0])
        total_stop = float(orbit["t"][-1])
        selected_crossings = [
            event
            for event in cloud.get("resonance_events", [])
            if bool(event.get("selected", False)) and bool(event.get("crossed", False))
        ]
        selected_crossings.sort(key=lambda item: float(item.get("t_source", np.inf)))

        if selected_crossings:
            event = selected_crossings[0]
            center = float(event["t_source"])
            reason = "first_selected_crossing"
        else:
            center = float(np.clip(cloud.get("resonance_time", total_start), total_start, total_stop))
            event = {
                "source": "coupled",
                "reason": "fallback_selected_track_min",
                "harmonic": int(self.resonance_harmonic),
                "selected": True,
                "crossed": False,
                "closest_position": "fallback",
                "t_source": center,
                "t_obs": center * (1.0 + self.z),
                "z_ad": np.nan,
                "detuning_abs": np.nan,
            }
            reason = "fallback_selected_track_min"

        omega_center = float(np.interp(center, orbit["t"], orbit["omega"]))
        local_period = 2.0 * np.pi / max(omega_center, 1.0e-300)
        duration = max(float(window_orbits), 1.0) * local_period
        half_duration = 0.5 * duration
        t_start = center - half_duration
        t_stop = center + half_duration

        if t_start < total_start:
            t_stop = min(total_stop, t_stop + (total_start - t_start))
            t_start = total_start
        if t_stop > total_stop:
            t_start = max(total_start, t_start - (t_stop - total_stop))
            t_stop = total_stop
        if t_stop <= t_start:
            t_start = max(total_start, center - 0.5 * local_period)
            t_stop = min(total_stop, center + 0.5 * local_period)

        anchor = {
            "source": "coupled",
            "reason": reason,
            "harmonic": int(event.get("harmonic", self.resonance_harmonic)),
            "selected": bool(event.get("selected", True)),
            "crossed": bool(event.get("crossed", False)),
            "closest_position": str(event.get("closest_position", "")),
            "t_source": float(center),
            "t_obs": float(center * (1.0 + self.z)),
            "z_ad": float(event.get("z_ad", np.nan)),
            "detuning_abs": float(event.get("detuning_abs", np.nan)),
        }
        return {
            "t_start_source": float(t_start),
            "t_stop_source": float(t_stop),
            "t_start_obs": float(t_start * (1.0 + self.z)),
            "t_stop_obs": float(t_stop * (1.0 + self.z)),
            "anchors": [anchor],
            "padding_source_s": 0.5 * float(t_stop - t_start),
            "window_mode": "first_selected_orbits",
            "window_orbits": float(window_orbits),
            "local_period_source_s": float(local_period),
        }

    def _annotate_frequency_domain_window(self, frequency_domain, window):
        frequency_domain["spectrum_window_mode"] = str(window.get("window_mode", "summary"))
        if "window_orbits" in window:
            frequency_domain["spectrum_window_orbits"] = float(window["window_orbits"])
        if "local_period_source_s" in window:
            frequency_domain["spectrum_window_local_period_source_s"] = float(window["local_period_source_s"])
        frequency_domain["spectrum_window_start_source_s"] = float(window["t_start_source"])
        frequency_domain["spectrum_window_stop_source_s"] = float(window["t_stop_source"])
        frequency_domain["spectrum_window_duration_source_s"] = float(
            window["t_stop_source"] - window["t_start_source"]
        )
        anchors = list(window.get("anchors", []))
        if anchors:
            frequency_domain["spectrum_window_anchor_reason"] = str(anchors[0].get("reason", ""))
            frequency_domain["spectrum_window_anchor_time_source_s"] = float(anchors[0].get("t_source", np.nan))
            frequency_domain["spectrum_window_anchor_harmonic"] = int(anchors[0].get("harmonic", self.resonance_harmonic))

    def build_common_start_windows(
        self,
        orbit_signal,
        cloud_signal,
        orbit_template,
        duration_obs_s,
        sample_points=2400,
    ):
        """Build start-aligned windows for the physical signal and the pure-binary template."""
        duration_source_s = float(duration_obs_s) / (1.0 + self.z)
        t_stop = min(orbit_signal["t"][-1], orbit_template["t"][-1], duration_source_s)
        if t_stop <= 0.0:
            t_zoom = np.array([0.0], dtype=float)
        else:
            t_zoom = np.linspace(0.0, t_stop, int(max(16, sample_points)))
        signal_window = self._evaluate_waveform_window_on_grid(orbit_signal, cloud_signal, t_zoom)
        template_window = self._evaluate_binary_window_on_grid(
            orbit_template,
            t_zoom,
            cloud_signal["resonance_time"],
        )
        return signal_window, template_window

    def build_zoom_waveforms(self, orbit, cloud, zoom_orbits=8, zoom_points=2400):
        return self.build_waveform_window(
            orbit,
            cloud,
            window_orbits=zoom_orbits,
            sample_points=zoom_points,
        )

    def _suggest_max_display_hz(self, f_transition_obs_hz, f_orb_res_obs_hz, freq_hz_max):
        # Keep the displayed spectrum wide enough to contain both the resonance peak
        # and the retained binary harmonic comb, instead of hard-clipping at 35 mHz.
        highest_feature_hz = max(
            float(f_transition_obs_hz),
            float(self.binary_harmonics) * float(f_orb_res_obs_hz),
        )
        return min(float(freq_hz_max), 1.25 * highest_feature_hz)

    def build_windowed_fft(self, waveform_window, pad_factor=8, tukey_alpha=0.03):
        # 棰戝煙鍒嗘瀽缁熶竴鍦ㄨ繖閲屽仛锛?        # 1. 鍏堝幓鍧囧€硷紝鎶戝埗浣庨娉勬紡
        # Use one taper consistently for the exported FFT products.
        t_obs = waveform_window["t_obs"]
        dt_obs = t_obs[1] - t_obs[0]
        n_samples = len(t_obs)
        n_fft = max(n_samples, int(pad_factor * n_samples))

        # Use a strong Kaiser taper for all exported high-frequency FFT products.
        h_binary_centered = waveform_window["h_binary"] - np.mean(waveform_window["h_binary"])
        h_axion_centered = waveform_window["h_axion"] - np.mean(waveform_window["h_axion"])
        h_total_centered = waveform_window["h_total"] - np.mean(waveform_window["h_total"])

        window_beta = 14.0
        fft_window = np.kaiser(n_samples, window_beta)
        spectrum_window = fft_window

        h_binary_tapered = h_binary_centered * fft_window
        h_axion_tapered = h_axion_centered * fft_window
        h_total_tapered = h_total_centered * fft_window

        h_binary_spectrum = h_binary_centered * spectrum_window
        h_axion_spectrum = h_axion_centered * spectrum_window
        h_total_spectrum = h_total_centered * spectrum_window

        freq_hz = np.fft.rfftfreq(n_fft, dt_obs)
        df_hz = freq_hz[1] - freq_hz[0]

        h_tilde_binary = np.fft.rfft(h_binary_tapered, n=n_fft) * dt_obs
        h_tilde_axion = np.fft.rfft(h_axion_tapered, n=n_fft) * dt_obs
        h_tilde_total = np.fft.rfft(h_total_tapered, n=n_fft) * dt_obs

        h_tilde_binary_plot = np.fft.rfft(h_binary_spectrum, n=n_fft) * dt_obs
        h_tilde_axion_plot = np.fft.rfft(h_axion_spectrum, n=n_fft) * dt_obs
        h_tilde_total_plot = np.fft.rfft(h_total_spectrum, n=n_fft) * dt_obs

        h_c_binary = 2.0 * freq_hz * np.abs(h_tilde_binary_plot)
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
            "fft_window": fft_window,
            "spectrum_window": spectrum_window,
            "tukey_alpha": tukey_alpha,
            "freq_hz": freq_hz,
            "h_tilde_binary": h_tilde_binary,
            "h_tilde_axion": h_tilde_axion,
            "h_tilde_total": h_tilde_total,
            "h_tilde_binary_plot": h_tilde_binary_plot,
            "h_tilde_axion_plot": h_tilde_axion_plot,
            "h_tilde_total_plot": h_tilde_total_plot,
            "h_c_binary": h_c_binary,
            "h_c_axion": h_c_axion,
            "h_c_total": h_c_total,
            "f_transition_obs_hz": f_transition_obs_hz,
            "f_orb_res_obs_hz": f_orb_res_obs_hz,
            "max_display_hz": max_display_hz,
        }

    def _recommended_spectrum_sample_points(
        self,
        orbit,
        t_start_source,
        t_stop_source,
        requested_points,
        samples_per_cycle=8.0,
        max_points=262144,
    ):
        duration_obs = max(0.0, (float(t_stop_source) - float(t_start_source)) * (1.0 + self.z))
        if duration_obs <= 0.0:
            return int(max(16, requested_points))

        omega_start = float(np.interp(float(t_start_source), orbit["t"], orbit["omega"]))
        omega_stop = float(np.interp(float(t_stop_source), orbit["t"], orbit["omega"]))
        max_orbital_frequency_obs = max(omega_start, omega_stop) / (2.0 * np.pi * (1.0 + self.z))
        highest_feature_hz = max(
            self.transition_omega / (2.0 * np.pi * (1.0 + self.z)),
            float(self.binary_harmonics) * max_orbital_frequency_obs,
        )
        required_points = int(np.ceil(duration_obs * highest_feature_hz * float(samples_per_cycle))) + 1
        return int(min(max(int(requested_points), required_points, 16), int(max_points)))

    def build_frequency_spectrum(self, waveform_window, pad_factor=64): # <--- pad_factor 默认值拉高到 64
         t_obs = waveform_window["t_obs"]
         dt_obs = t_obs[1] - t_obs[0]
         n_samples = len(t_obs)
        
         # 极致零填充：将原本的数组长度扩展 64 倍（后面全补 0）
         n_fft = int(pad_factor * n_samples)

         # Use the same strong taper as the exported FFT path.
         taper = np.kaiser(n_samples, 14.0)

         h_binary = (waveform_window["h_binary"] - np.mean(waveform_window["h_binary"])) * taper
         h_axion = (waveform_window["h_axion"] - np.mean(waveform_window["h_axion"])) * taper
         h_total = (waveform_window["h_total"] - np.mean(waveform_window["h_total"])) * taper

         # 执行 FFT：numpy 的 rfft 发现 n_fft > len(h_binary) 时，会自动在尾部补零
         freq_hz = np.fft.rfftfreq(n_fft, dt_obs)
         spec_binary = np.abs(np.fft.rfft(h_binary, n=n_fft))
         spec_axion = np.abs(np.fft.rfft(h_axion, n=n_fft))
         spec_total = np.abs(np.fft.rfft(h_total, n=n_fft))

         # 归一化处理
         normalization = max(np.max(spec_binary), np.max(spec_total), 1.0e-30)
        
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
            "spec_binary": spec_binary / normalization,
            "spec_axion": spec_axion / normalization,
            "spec_total": spec_total / normalization,
            "f_transition_obs_hz": f_transition_obs_hz,
            "f_orb_res_obs_hz": f_orb_res_obs_hz,
            "max_display_hz": max_display_hz,
        }

    def _extract_comb_peaks(self, frequency_domain, search_half_width=0.35):
        freq_hz = frequency_domain["freq_hz"]
        h_c_binary = frequency_domain["h_c_binary"]
        h_c_total = frequency_domain["h_c_total"]
        f_orb_res = frequency_domain["f_orb_res_obs_hz"]
        half_width_hz = search_half_width * f_orb_res

        harmonic_freqs = []
        binary_peaks = []
        total_peaks = []

        for harmonic in range(1, self.binary_harmonics + 1):
            center = harmonic * f_orb_res
            mask = (freq_hz >= center - half_width_hz) & (freq_hz <= center + half_width_hz)
            if not np.any(mask):
                continue
            harmonic_freqs.append(center)
            binary_peaks.append(np.max(h_c_binary[mask]))
            total_peaks.append(np.max(h_c_total[mask]))

        return {
            "harmonic_freqs": np.asarray(harmonic_freqs),
            "binary_peaks": np.asarray(binary_peaks),
            "total_peaks": np.asarray(total_peaks),
        }

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

    def _save_frequency_amplitude_data(self, frequency_domain, component_key, stem, component_label):
        if self.frequency_data_dir is None:
            return None

        self.frequency_data_dir.mkdir(parents=True, exist_ok=True)
        freq_hz = np.asarray(frequency_domain["freq_hz"], dtype=float)
        h_tilde = np.asarray(frequency_domain[component_key], dtype=np.complex128)
        amplitude = np.abs(h_tilde)
        valid = (
            np.isfinite(freq_hz)
            & np.isfinite(amplitude)
            & np.isfinite(h_tilde.real)
            & np.isfinite(h_tilde.imag)
        )
        table = np.column_stack(
            (
                freq_hz[valid],
                amplitude[valid],
                h_tilde.real[valid],
                h_tilde.imag[valid],
            )
        )
        header_lines = [
            f"module={self.module_stem}",
            f"component={component_label}",
            f"transition_family={self.transition_family}",
            f"eta_model={'manual_powerlaw_hansen' if self._manual_eta_ref_hz else self.eta_model}",
            f"eta_reference_hz={self.eta_ref_hz:.16e}",
            f"cloud_evolution_mode={self.cloud_evolution_mode}",
            f"resonance_harmonic={self.resonance_harmonic:d}",
            f"max_harmonic={int(self.harmonics[-1]):d}",
            f"multi_harmonic_drive={int(self.multi_harmonic_drive)}",
            f"harmonics_to_keep={self.harmonics_to_keep:d}",
            "active_harmonics="
            + ",".join(str(int(n)) for n in self._select_active_harmonic_indices()[2]),
            f"lz_window_widths={self.lz_window_widths:.16e}",
            f"backreaction_gate_mode={self.backreaction_gate_mode}",
            f"backreaction_gate_width_factor={self.backreaction_gate_width_factor:.16e}",
            f"alpha={self.alpha:.16e}",
            f"bh_spin={self.bh_spin:.16e}",
            f"cloud_mass_fraction={self.cloud_mass_fraction:.16e}",
            f"redshift={self.z:.16e}",
            f"luminosity_distance_m={self.d_L:.16e}",
            f"primary_mass_msun={self.M / self.M_sun:.16e}",
            f"secondary_mass_msun={self.M_star / self.M_sun:.16e}",
            f"eccentricity_init={self.e_init:.16e}",
            f"transition_frequency_obs_hz={frequency_domain['f_transition_obs_hz']:.16e}",
            f"fft_nyquist_hz={frequency_domain['freq_hz'][-1]:.16e}",
            f"fft_dt_obs_s={frequency_domain['dt_obs']:.16e}",
            f"fft_df_hz={frequency_domain['df_hz']:.16e}",
            f"fft_n_samples={len(frequency_domain['t_obs'])}",
            f"fft_n_fft={frequency_domain['n_fft']}",
            "fft_window=kaiser",
            "fft_window_beta=1.4000000000000000e+01",
            "fft_mean_removed=1",
            "fourier_convention=h_tilde_integral_h_exp_minus_2pi_i_f_t_dt",
            "fft_amplitude_units=strain_seconds",
        ]
        optional_metadata_keys = (
            "spectrum_window_mode",
            "spectrum_window_orbits",
            "spectrum_window_local_period_source_s",
            "spectrum_window_start_source_s",
            "spectrum_window_stop_source_s",
            "spectrum_window_duration_source_s",
            "spectrum_window_anchor_reason",
            "spectrum_window_anchor_time_source_s",
            "spectrum_window_anchor_harmonic",
        )
        for key in optional_metadata_keys:
            if key not in frequency_domain:
                continue
            value = frequency_domain[key]
            if isinstance(value, (float, np.floating)):
                header_lines.append(f"{key}={float(value):.16e}")
            else:
                header_lines.append(f"{key}={value}")
        header_lines.extend(
            [
                "frequency_frame=observer",
                "columns=frequency_hz amplitude_abs h_tilde_real h_tilde_imag",
            ]
        )
        out_path = self.frequency_data_dir / f"{stem}.txt"
        np.savetxt(out_path, table, header="\n".join(header_lines), comments="# ")
        print(f"Saved frequency-amplitude data: {out_path}")
        return out_path

    def _format_frequency(self, freq_hz):
        if freq_hz >= 1.0e3:
            return freq_hz * 1.0e-3, "kHz"
        if freq_hz < 0.1:
            return freq_hz * 1.0e3, "mHz"
        return freq_hz, "Hz"

    def _get_frequency_axis_scale(self, freq_hz_max):
        if freq_hz_max >= 1.0e3:
            return 1.0e-3, "kHz"
        if freq_hz_max < 0.1:
            return 1.0e3, "mHz"
        return 1.0, "Hz"

    def _format_frequency_band(self, f_low_hz, f_high_hz):
        scale, unit = self._get_frequency_axis_scale(max(f_low_hz, f_high_hz))
        return f_low_hz * scale, f_high_hz * scale, unit

    def _get_time_axis_scale(self, span_seconds):
        if span_seconds < 300.0:
            return 1.0, "s"
        if span_seconds < 3.0 * 3600.0:
            return 1.0 / 60.0, "min"
        if span_seconds < 7.0 * 86400.0:
            return 1.0 / 3600.0, "hr"
        return 1.0 / 86400.0, "days"

    def _format_resonance_event(self, event):
        status = "cross" if event["crossed"] else f"no-cross:{event['closest_position']}"
        tag = "[adiabatic]" if event["quenched"] else ""
        star = "*" if event["selected"] else ""
        return (
            f"n={event['harmonic']}:{event['t_obs']:.6e}s({status}),"
            f"z_ad={event['z_ad']:.3g}{tag},|det|={event['detuning_abs']:.3e}{star}"
        )

    def print_summary(self, results, elapsed_s):
        orbit = results["orbit"]
        cloud = results["cloud"]
        time_window = results["time_window"]
        spectrum_window = results.get("spectrum_window", time_window)
        summary_window = results.get("summary_window", {})
        f_transition_obs = self.transition_omega / (2.0 * np.pi * (1.0 + self.z))
        f_transition_val, f_transition_unit = self._format_frequency(f_transition_obs)
        peak_overlap = float(np.max(cloud["overlap_abs"])) if cloud["overlap_abs"].size else 0.0
        norm = np.abs(cloud["cg"]) ** 2 + np.abs(cloud["ce_tilde"]) ** 2
        peak_binary = float(np.max(np.abs(time_window["h_binary"])))
        peak_axion = float(np.max(np.abs(time_window["h_axion"])))
        ratio = peak_axion / max(peak_binary, 1.0e-300)
        e0_safe = float(np.clip(self.e_init, 0.0, 0.999))
        reduced_mass = self.M * self.M_star / self.M_tot
        orbit_energy_init = -self.G * self.M * self.M_star / (2.0 * self.a_init)
        orbit_angular_init = reduced_mass * np.sqrt(
            self.G * self.M_tot * self.a_init * max(1.0e-12, 1.0 - e0_safe * e0_safe)
        )
        energy_backreaction_ratio = abs(self.delta_E_orbit_backreaction) / max(abs(orbit_energy_init), 1.0e-300)
        angular_backreaction_ratio = abs(self.delta_L_orbit_backreaction) / max(abs(orbit_angular_init), 1.0e-300)

        print(f"Runtime: {elapsed_s:.2f} s")
        print(
            f"Transition states: |{''.join(map(str, self.transition_solver_data['initial_state']))}> -> "
            f"|{''.join(map(str, self.transition_solver_data['final_state']))}> ({self.transition_family})"
        )
        print(
            "Pipeline scope: coupled orbit-cloud waveform and FFT exports only; "
            "detector-analysis products are not generated in highfre."
        )
        print(
            f"High-frequency parameters: M={self.M / self.M_sun:.3e} Msun, "
            f"M_star={self.M_star / self.M_sun:.3e} Msun, alpha={self.alpha:.3f}, "
            f"q={self.M_star / self.M:.3e}, e_init={self.e_init:.3f}, "
            f"a_star={self.bh_spin:.3f}, Mc/M={self.cloud_mass_fraction:.3g}, "
            f"f_orb_init={self.f_orb_init:.4e} Hz"
        )
        print(
            f"Transition solver: f_obs={f_transition_val:.4f} {f_transition_unit}, "
            f"eta(A.6 @ a_init)={self.eta_ref_hz:.3e} Hz, "
            f"Gamma_decay={self.Gamma_decay_hz:.3e} Hz, "
            f"boson_mass={self.transition_solver_data['boson_mass_eV']:.3e} eV"
        )
        print(
            f"Orbital backreaction: {'enabled' if self.include_orbital_backreaction else 'disabled'}, "
            f"cloud_mode={self.cloud_evolution_mode}, gate={self.backreaction_gate_mode}, "
            f"DeltaE_cloud={self.delta_E_orbit_backreaction:.3e} J, "
            f"DeltaL_cloud={self.delta_L_orbit_backreaction:.3e} J*s"
        )
        print(
            f"Backreaction scale check: |DeltaE_cloud|/|E_orb,init|={energy_backreaction_ratio:.3e}, "
            f"|DeltaL_cloud|/|L_orb,init|={angular_backreaction_ratio:.3e}"
        )
        print(
            "Drive mode:",
            "multi-harmonic" if self.multi_harmonic_drive else "single dominant harmonic",
            f"active={','.join(str(n) for n in cloud['active_harmonics'])}",
        )
        events = list(cloud.get("resonance_events", []))
        pure_events = list(results.get("pure_resonance_events", []))
        crossing_events = [event for event in events if event["crossed"]]
        non_crossing_events = [event for event in events if not event["crossed"]]
        if crossing_events:
            print(
                "Resolved resonance crossings (obs s): "
                + ", ".join(self._format_resonance_event(event) for event in crossing_events)
            )
        else:
            print("Resolved resonance crossings (obs s): none")
        if non_crossing_events:
            print(
                "No-cross boundary/closest diagnostics in simulated window (not resonances): "
                + ", ".join(self._format_resonance_event(event) for event in non_crossing_events)
            )
        selected_pure_events = [event for event in pure_events if event["selected"]]
        if selected_pure_events:
            print(
                "Pure Peters selected-resonance check: "
                + ", ".join(self._format_resonance_event(event) for event in selected_pure_events)
            )
        local_lz_events = list(cloud.get("local_lz_events", []))
        if local_lz_events:
            print(
                "Local LZ/impulse events: "
                + ", ".join(
                    f"n={event['harmonic']}@{event['t_obs']:.6e}s,"
                    f"window=[{event['lz_window_start_source'] * (1.0 + self.z):.6e},"
                    f"{event['lz_window_stop_source'] * (1.0 + self.z):.6e}]s,"
                    f"dP_high={event['delta_high_population']:.3e},"
                    f"da/a={event.get('delta_a_over_a', np.nan):+.3e},"
                    f"de={event.get('delta_e', np.nan):+.3e},"
                    f"P_LZ~{event['lz_probability_estimate']:.3e}"
                    for event in local_lz_events
                )
            )
        print(
            f"Coherence diagnostics: max raw |cg*ce|={peak_overlap:.3e}, "
            f"state norm range=[{np.min(norm):.3e}, {np.max(norm):.3e}]"
        )
        gate = np.asarray(cloud.get("backreaction_gate", []), dtype=float)
        if gate.size:
            if self.cloud_evolution_mode in {"event", "event_lz", "event_lz_impulse", "impulse"}:
                print(
                    f"Local-event window diagnostics: sample_fraction={np.mean(gate > 0.0):.3e}, "
                    f"event_count={len(local_lz_events)}"
                )
            else:
                print(
                    f"Backreaction gate diagnostics: max={np.max(gate):.3e}, "
                    f"mean={np.mean(gate):.3e}, active_fraction(g>1e-3)={np.mean(gate > 1.0e-3):.3e}"
                )
        print(f"Zoom-window peak ratio max|h_axion|/max|h_binary| = {ratio:.3e}")
        print(
            f"Integrated source window: {orbit['t'][-1]:.6e} s "
            f"({orbit['t'][-1] / self.yr:.3e} yr), resonance t_obs={cloud['resonance_time'] * (1.0 + self.z):.6e} s"
        )
        anchors = list(summary_window.get("anchors", []))
        if anchors:
            anchor_text = ", ".join(
                f"{anchor['source']}:n={anchor['harmonic']}:{anchor['reason']}@{anchor['t_obs']:.6e}s"
                for anchor in anchors
            )
            print(f"Orbit-summary window anchors: {anchor_text}")
        print(
            f"Orbit-summary observation window: [{time_window['t_obs'][0]:.6e}, "
            f"{time_window['t_obs'][-1]:.6e}] s, samples={len(time_window['t_obs'])}"
        )
        print(
            f"FFT/export observation window: [{spectrum_window['t_obs'][0]:.6e}, "
            f"{spectrum_window['t_obs'][-1]:.6e}] s, samples={len(spectrum_window['t_obs'])}"
        )
        for label, path in results.get("export_paths", {}).items():
            if path is not None:
                print(f"Saved {label}: {path}")

    def plot_summary(self, results):
        orbit = results["orbit"]
        cloud = results["cloud"]
        time_window = results["time_window"]
        summary_window = results.get("summary_window", {})

        window_obs_start = float(time_window["t_obs"][0])
        window_obs_stop = float(time_window["t_obs"][-1])
        window_obs_span = max(window_obs_stop - window_obs_start, 0.0)
        secular_time_scale, secular_time_unit = self._get_time_axis_scale(window_obs_span)
        xlim_scaled = (window_obs_start * secular_time_scale, window_obs_stop * secular_time_scale)
        t_panel_start = window_obs_start / (1.0 + self.z)
        t_panel_stop = window_obs_stop / (1.0 + self.z)
        if t_panel_stop <= t_panel_start:
            t_panel_source = np.asarray([t_panel_start], dtype=float)
        else:
            t_panel_source = np.linspace(t_panel_start, t_panel_stop, 2400)
        a_panel, e_panel, _ = orbit["solution"].sol(t_panel_source)
        t_scaled = t_panel_source * (1.0 + self.z) * secular_time_scale
        a_au = np.asarray(a_panel, dtype=float) / self.AU
        e_panel = np.asarray(e_panel, dtype=float)
        anchor_events = list(summary_window.get("anchors", []))
        if not anchor_events:
            fallback_obs = cloud["resonance_time"] * (1.0 + self.z)
            anchor_events = [{"source": "coupled", "reason": "fallback", "t_obs": fallback_obs, "crossed": False}]
        local_lz_events = list(cloud.get("local_lz_events", []))

        def anchor_style(anchor):
            if anchor.get("source") == "pure_peters":
                return "#6f6f6f", "--", 0.55
            if anchor.get("crossed", False):
                return "k", "-", 0.70
            return "#D55E00", ":", 0.60

        def draw_absolute_anchors(ax):
            for anchor in anchor_events:
                t_obs = float(anchor.get("t_obs", np.nan))
                if not np.isfinite(t_obs):
                    continue
                color, linestyle, alpha = anchor_style(anchor)
                ax.axvline(t_obs * secular_time_scale, color=color, linestyle=linestyle, alpha=alpha)

        def draw_relative_anchors(ax, zoom_scale):
            for anchor in anchor_events:
                t_obs = float(anchor.get("t_obs", np.nan))
                if not np.isfinite(t_obs) or t_obs < window_obs_start or t_obs > window_obs_stop:
                    continue
                color, linestyle, alpha = anchor_style(anchor)
                ax.axvline((t_obs - window_obs_start) * zoom_scale, color=color, linestyle=linestyle, alpha=alpha)

        def draw_absolute_lz_windows(ax):
            for event in local_lz_events:
                start = float(event.get("lz_window_start_source", np.nan)) * (1.0 + self.z)
                stop = float(event.get("lz_window_stop_source", np.nan)) * (1.0 + self.z)
                if not np.isfinite(start) or not np.isfinite(stop):
                    continue
                left = max(start, window_obs_start) * secular_time_scale
                right = min(stop, window_obs_stop) * secular_time_scale
                if right <= left:
                    continue
                ax.axvspan(left, right, color="#6BAED6", alpha=0.14, lw=0.0)

        def draw_relative_lz_windows(ax, zoom_scale):
            for event in local_lz_events:
                start = float(event.get("lz_window_start_source", np.nan)) * (1.0 + self.z)
                stop = float(event.get("lz_window_stop_source", np.nan)) * (1.0 + self.z)
                if not np.isfinite(start) or not np.isfinite(stop):
                    continue
                left = max(start, window_obs_start)
                right = min(stop, window_obs_stop)
                if right <= left:
                    continue
                ax.axvspan((left - window_obs_start) * zoom_scale, (right - window_obs_start) * zoom_scale, color="#6BAED6", alpha=0.14, lw=0.0)

        fig = plt.figure(figsize=(7.1, 4.7), constrained_layout=True)
        grid = fig.add_gridspec(2, 1, hspace=0.14)

        ax0 = fig.add_subplot(grid[0, 0])
        draw_absolute_lz_windows(ax0)
        ax0.plot(t_scaled, a_au, color="#B23A48", alpha=0.88, lw=0.9, label=r"$a$")
        ax0.set_xlabel(f"Observer time ({secular_time_unit})")
        ax0.set_ylabel("Semi-major axis (AU)", color="#B23A48")
        ax0.tick_params(axis="y", labelcolor="#B23A48")
        ax0.set_xlim(*xlim_scaled)
        draw_absolute_anchors(ax0)
        ax0b = ax0.twinx()
        ax0.set_zorder(ax0b.get_zorder() + 1)
        ax0.patch.set_visible(False)
        ax0b.patch.set_visible(False)
        ax0b.plot(t_scaled, e_panel, color="#2C7FB8", alpha=0.82, lw=0.9, label=r"$e$")
        ax0b.set_ylabel("Eccentricity", color="#2C7FB8")
        ax0b.tick_params(axis="y", labelcolor="#2C7FB8")
        lines, labels = ax0.get_legend_handles_labels()
        lines_b, labels_b = ax0b.get_legend_handles_labels()
        ax0.legend(lines + lines_b, labels + labels_b, loc="best", framealpha=0.88, fontsize=7)

        def anchor_time_source(anchor):
            return float(anchor.get("t_source", np.nan))

        crossed_coupled = [
            anchor
            for anchor in anchor_events
            if anchor.get("source") != "pure_peters"
            and anchor.get("crossed", False)
            and np.isfinite(anchor_time_source(anchor))
        ]
        crossed_pure = [
            anchor
            for anchor in anchor_events
            if anchor.get("source") == "pure_peters"
            and anchor.get("crossed", False)
            and np.isfinite(anchor_time_source(anchor))
        ]
        interior_closest = [
            anchor
            for anchor in anchor_events
            if str(anchor.get("closest_position", "")).lower() == "interior"
            and np.isfinite(anchor_time_source(anchor))
        ]
        if crossed_coupled:
            signal_center_source = anchor_time_source(crossed_coupled[0])
        elif crossed_pure:
            signal_center_source = anchor_time_source(crossed_pure[0])
        elif interior_closest:
            signal_center_source = anchor_time_source(interior_closest[0])
        else:
            signal_center_source = float(np.clip(cloud.get("resonance_time", t_panel_start), orbit["t"][0], orbit["t"][-1]))

        signal_cycles = float(
            os.environ.get(
                "HIGHFREQ_ORBIT_SUMMARY_SIGNAL_CYCLES",
                os.environ.get("ORBIT_SUMMARY_SIGNAL_CYCLES", "100.0"),
            )
        )
        signal_center_source = float(np.clip(signal_center_source, orbit["t"][0], orbit["t"][-1]))
        omega_center = float(np.interp(signal_center_source, orbit["t"], orbit["omega"]))
        period_center = 2.0 * np.pi / max(omega_center, 1.0e-300)
        half_signal = 0.5 * max(signal_cycles, 1.0) * period_center
        samples_per_cycle = float(os.environ.get("ORBIT_SUMMARY_SAMPLES_PER_CYCLE", "64.0"))
        signal_samples = int(max(2048, min(20000, np.ceil(max(signal_cycles, 1.0) * samples_per_cycle))))
        signal_window = self.build_waveform_window_between(
            orbit,
            cloud,
            signal_center_source - half_signal,
            signal_center_source + half_signal,
            sample_points=signal_samples,
        )
        signal_obs_start = float(signal_window["t_obs"][0])
        signal_obs_stop = float(signal_window["t_obs"][-1])
        signal_obs_span = max(signal_obs_stop - signal_obs_start, 0.0)
        zoom_scale, zoom_unit = self._get_time_axis_scale(signal_obs_span)
        t_window = (signal_window["t_obs"] - signal_obs_start) * zoom_scale

        ax1 = fig.add_subplot(grid[1, 0])
        for event in local_lz_events:
            start = float(event.get("lz_window_start_source", np.nan)) * (1.0 + self.z)
            stop = float(event.get("lz_window_stop_source", np.nan)) * (1.0 + self.z)
            if not np.isfinite(start) or not np.isfinite(stop):
                continue
            left = max(start, signal_obs_start)
            right = min(stop, signal_obs_stop)
            if right <= left:
                continue
            ax1.axvspan((left - signal_obs_start) * zoom_scale, (right - signal_obs_start) * zoom_scale, color="#6BAED6", alpha=0.14, lw=0.0)
        h_axion = np.asarray(signal_window["h_axion"], dtype=float)
        overlap_display = np.asarray(signal_window["overlap_abs"], dtype=float)
        ax1.plot(t_window, h_axion, color="#5B8DB8", lw=0.55, alpha=0.48, label=r"$h_a$")
        for anchor in anchor_events:
            t_obs = float(anchor.get("t_obs", np.nan))
            if not np.isfinite(t_obs) or t_obs < signal_obs_start or t_obs > signal_obs_stop:
                continue
            color, linestyle, alpha = anchor_style(anchor)
            ax1.axvline((t_obs - signal_obs_start) * zoom_scale, color=color, linestyle=linestyle, alpha=alpha)
        ax1.set_xlabel(f"Observer time from window start ({zoom_unit})")
        ax1.set_ylabel(r"Axion strain $h_a$", color="#1F78B4")
        ax1.tick_params(axis="y", labelcolor="#1F78B4")
        ax1b = ax1.twinx()
        ax1.set_zorder(ax1b.get_zorder() + 1)
        ax1.patch.set_visible(False)
        ax1b.patch.set_visible(False)
        ax1b.plot(t_window, overlap_display, color="#D55E00", lw=1.15, alpha=0.96, label=r"$|c_i^* \tilde{c}_f|$")
        ax1b.set_ylabel(r"$|c_i^* \tilde{c}_f|$", color="#E67E22")
        ax1b.tick_params(axis="y", labelcolor="#E67E22")
        lines, labels = ax1.get_legend_handles_labels()
        lines_b, labels_b = ax1b.get_legend_handles_labels()
        ax1.legend(lines + lines_b, labels + labels_b, loc="upper right", framealpha=0.9, fontsize=7)

        for axis in fig.axes:
            axis.tick_params(axis="both", labelsize=8)
            axis.xaxis.label.set_size(9)
            axis.yaxis.label.set_size(9)

        self._save_figure(fig, f"{self.module_stem}_orbit_time_summary_{self.direction_tag}")
        if "agg" in plt.get_backend().lower():
            plt.close(fig)
        else:
            plt.show()

    def _save_time_series_data(self, waveform_window, template_window, stem, component_label):
        if self.waveform_data_dir is None:
            return None
        self.waveform_data_dir.mkdir(parents=True, exist_ok=True)
        table = np.column_stack(
            (
                waveform_window["t_source"],
                waveform_window["t_obs"],
                waveform_window["t_rel_obs"],
                waveform_window["h_binary"],
                waveform_window["h_axion"],
                waveform_window["h_total"],
                template_window["h_binary"],
                waveform_window["overlap_abs"],
                waveform_window["e"],
                waveform_window["phi"],
            )
        )
        header_lines = [
            f"module={self.module_stem}",
            f"component={component_label}",
            f"transition_family={self.transition_family}",
            f"alpha={self.alpha:.16e}",
            f"bh_spin={self.bh_spin:.16e}",
            f"cloud_mass_fraction={self.cloud_mass_fraction:.16e}",
            f"redshift={self.z:.16e}",
            f"luminosity_distance_m={self.d_L:.16e}",
            f"primary_mass_msun={self.M / self.M_sun:.16e}",
            f"secondary_mass_msun={self.M_star / self.M_sun:.16e}",
            f"transition_frequency_obs_hz={self.transition_omega / (2.0 * np.pi * (1.0 + self.z)):.16e}",
            "time_frame=source_and_observer",
            "columns=t_source_s t_obs_s t_rel_obs_s h_backreacted_binary h_axion h_total h_pure_binary_template overlap_abs eccentricity mean_anomaly",
        ]
        out_path = self.waveform_data_dir / f"{stem}.txt"
        np.savetxt(out_path, table, header="\n".join(header_lines), comments="# ")
        print(f"Saved time-series strain data: {out_path}")
        return out_path

    def run(
        self,
        duration_yr=1.0,
        secular_samples=3000,
        zoom_orbits=8,
        zoom_points=8192,
        spectrum_orbits=36,
        spectrum_points=8192,
        spectrum_pad_factor=8,
        tukey_alpha=0.03,
        spectrum_window_mode="summary",
        save_exports=True,
        **legacy_unused_options,
    ):
        orbit, cloud = self.solve_coupled_system(duration_yr=duration_yr, secular_samples=secular_samples)
        template_orbit = self.solve_orbit(duration_yr=duration_yr, secular_samples=secular_samples)
        pure_resonance_events = self._build_resonance_events_from_orbit(template_orbit)
        summary_window = self._select_orbit_summary_window(
            orbit,
            cloud,
            pure_resonance_events,
        )
        time_window = self.build_waveform_window_between(
            orbit,
            cloud,
            summary_window["t_start_source"],
            summary_window["t_stop_source"],
            sample_points=zoom_points,
        )
        if str(spectrum_window_mode).lower() in {"first_selected", "first_selected_orbits", "first_resonance"}:
            spectrum_selection_window = self._select_first_selected_orbit_window(
                orbit,
                cloud,
                window_orbits=spectrum_orbits,
            )
        else:
            spectrum_selection_window = dict(summary_window)
            spectrum_selection_window.setdefault("window_mode", "summary")
        spectrum_sample_points = self._recommended_spectrum_sample_points(
            orbit,
            spectrum_selection_window["t_start_source"],
            spectrum_selection_window["t_stop_source"],
            spectrum_points,
        )
        spectrum_window = self.build_waveform_window_between(
            orbit,
            cloud,
            spectrum_selection_window["t_start_source"],
            spectrum_selection_window["t_stop_source"],
            sample_points=spectrum_sample_points,
        )
        binary_template_time_window = self._evaluate_binary_window_on_grid(
            template_orbit,
            time_window["t_source"],
            cloud["resonance_time"],
        )
        binary_template_spectrum_window = self._evaluate_binary_window_on_grid(
            template_orbit,
            spectrum_window["t_source"],
            cloud["resonance_time"],
        )
        frequency_domain = self.build_windowed_fft(
            spectrum_window,
            pad_factor=spectrum_pad_factor,
            tukey_alpha=tukey_alpha,
        )
        binary_template_frequency_domain = self.build_windowed_fft(
            binary_template_spectrum_window,
            pad_factor=spectrum_pad_factor,
            tukey_alpha=tukey_alpha,
        )
        self._annotate_frequency_domain_window(frequency_domain, spectrum_selection_window)
        self._annotate_frequency_domain_window(binary_template_frequency_domain, spectrum_selection_window)
        spectrum = self.build_frequency_spectrum(spectrum_window, pad_factor=spectrum_pad_factor)

        export_paths = {}
        if save_exports:
            export_paths["selected time-window strain"] = self._save_time_series_data(
                time_window,
                binary_template_time_window,
                f"{self.module_stem}_strain_time_window_{self.direction_tag}",
                "selected_time_window",
            )
            export_paths["frequency-window strain"] = self._save_time_series_data(
                spectrum_window,
                binary_template_spectrum_window,
                f"{self.module_stem}_strain_frequency_window_{self.direction_tag}",
                "frequency_window",
            )
            export_paths["total frequency amplitude"] = self._save_frequency_amplitude_data(
                frequency_domain,
                "h_tilde_total",
                f"{self.module_stem}_axion_backreaction_total_frequency_amplitude",
                "axion_backreaction_total",
            )
            export_paths["axion frequency amplitude"] = self._save_frequency_amplitude_data(
                frequency_domain,
                "h_tilde_axion",
                f"{self.module_stem}_axion_frequency_amplitude",
                "axion_transition",
            )
            export_paths["pure binary frequency amplitude"] = self._save_frequency_amplitude_data(
                binary_template_frequency_domain,
                "h_tilde_binary",
                f"{self.module_stem}_pure_binary_template_frequency_amplitude",
                "pure_binary_template",
            )

        return {
            "orbit": orbit,
            "cloud": cloud,
            "template_orbit": template_orbit,
            "time_window": time_window,
            "summary_window": summary_window,
            "spectrum_selection_window": spectrum_selection_window,
            "binary_template_time_window": binary_template_time_window,
            "spectrum_window": spectrum_window,
            "binary_template_spectrum_window": binary_template_spectrum_window,
            "frequency_domain": frequency_domain,
            "binary_template_frequency_domain": binary_template_frequency_domain,
            "pure_resonance_events": pure_resonance_events,
            "spectrum": spectrum,
            "export_paths": export_paths,
            "signal_export_path": export_paths.get("total frequency amplitude"),
            "template_export_path": export_paths.get("pure binary frequency amplitude"),
        }


def run_default_highfreq_entry(simulator_cls, description, duration_yr=2.0e-8):
    start = time.time()
    simulator = simulator_cls()
    requested_duration_yr = float(os.environ.get("HIGHFREQ_DURATION_YR", str(duration_yr)))
    production_duration_yr = simulator.recommended_duration_to_cover_selected_resonance(
        initial_duration_yr=requested_duration_yr,
    )
    print(description)
    print(
        "Run mode: waveform-only high-frequency event-local LZ/impulse pipeline; "
        f"duration_yr={production_duration_yr:.3e}"
    )
    results = simulator.run(
        duration_yr=production_duration_yr,
        secular_samples=900,
        zoom_orbits=20,
        zoom_points=8192,
        spectrum_orbits=16,
        spectrum_points=8192,
        spectrum_pad_factor=4,
    )
    elapsed = time.time() - start
    simulator.print_summary(results, elapsed)
    simulator.plot_summary(results)
    return results
