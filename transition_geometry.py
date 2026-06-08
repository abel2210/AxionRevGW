from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.special import sph_harm_y


State = tuple[int, int, int]
RadialWavefunction = Callable[[State, np.ndarray], np.ndarray]


def spherical_harmonic(m: int, l: int, phi, theta):
    return sph_harm_y(l, m, theta, phi)


@dataclass(frozen=True)
class TransitionGeometry:
    initial_state: State
    final_state: State
    delta_m: int
    pattern: str
    radial_overlap: float
    waveform_geom_factor: float
    observer_projected_rss: float
    observer_plus: complex
    observer_cross: complex
    source_angle_average_factor: float
    quadrupole_matrix: np.ndarray

    def as_row(self) -> dict[str, object]:
        return {
            "initial_state": state_label(self.initial_state),
            "final_state": state_label(self.final_state),
            "delta_m": self.delta_m,
            "pattern": self.pattern,
            "radial_overlap": self.radial_overlap,
            "waveform_geom_factor": self.waveform_geom_factor,
            "observer_projected_rss": self.observer_projected_rss,
            "source_angle_average_factor": self.source_angle_average_factor,
        }


def state_label(state: State) -> str:
    return "|" + "".join(str(part) for part in state) + ">"


def _state_tuple(state) -> State:
    n, l, m = state
    return int(n), int(l), int(m)


def _quadrupole_matrix(
    initial_state: State,
    final_state: State,
    radial_overlap: float,
    theta_samples: int,
    phi_samples: int,
) -> np.ndarray:
    _, l_i, m_i = initial_state
    _, l_f, m_f = final_state
    theta = np.linspace(0.0, np.pi, int(theta_samples))
    phi = np.linspace(0.0, 2.0 * np.pi, int(phi_samples), endpoint=False)
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
            angular_integrand = (
                y_i
                * np.conj(y_f)
                * n_components[i]
                * n_components[j]
                * np.sin(theta_grid)
            )
            phi_integral = np.trapezoid(angular_integrand, phi, axis=1)
            matrix[i, j] = radial_overlap * np.trapezoid(phi_integral, theta)
    return matrix


def compute_transition_geometry(
    initial_state,
    final_state,
    radial_wavefunction: RadialWavefunction,
    *,
    overlap_max_x: float = 5.0e3,
    overlap_grid_points: int = 4096,
    observer_theta: float = math.pi / 2.0,
    observer_phi: float = 0.0,
    theta_samples: int = 180,
    phi_samples: int = 360,
) -> TransitionGeometry:
    initial_state = _state_tuple(initial_state)
    final_state = _state_tuple(final_state)

    x_grid = np.logspace(-6, np.log10(float(overlap_max_x)), max(512, int(overlap_grid_points)))
    radial_i = radial_wavefunction(initial_state, x_grid)
    radial_f = radial_wavefunction(final_state, x_grid)
    radial_overlap = float(np.trapezoid((x_grid**4) * radial_i * radial_f, x_grid))
    quadrupole_matrix = _quadrupole_matrix(
        initial_state,
        final_state,
        radial_overlap,
        theta_samples=theta_samples,
        phi_samples=phi_samples,
    )

    theta = float(observer_theta)
    phi = float(observer_phi)
    u_vec = np.array(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)],
        dtype=float,
    )
    v_vec = np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=float)
    e_plus = np.outer(u_vec, u_vec) - np.outer(v_vec, v_vec)
    e_cross = np.outer(u_vec, v_vec) + np.outer(v_vec, u_vec)
    observer_plus = complex(np.sum(quadrupole_matrix * e_plus))
    observer_cross = complex(np.sum(quadrupole_matrix * e_cross))
    observer_projected_rss = float(np.sqrt(abs(observer_plus) ** 2 + abs(observer_cross) ** 2))

    delta_m = int(final_state[2] - initial_state[2])
    i_xx = quadrupole_matrix[0, 0]
    i_yy = quadrupole_matrix[1, 1]
    i_zz = quadrupole_matrix[2, 2]
    i_xy = quadrupole_matrix[0, 1]

    if delta_m == 0:
        pattern = "axisymmetric_delta_m0"
        i_perp = 0.5 * (i_xx + i_yy)
        waveform_geom_factor = float(abs(i_zz - i_perp) / 4.0)
        source_angle_average_factor = 8.0 / 15.0
    elif abs(delta_m) == 2:
        pattern = "quadrupolar_delta_m2"
        spin2_plus = (i_xx - i_yy) + 2.0j * i_xy
        spin2_minus = (i_xx - i_yy) - 2.0j * i_xy
        waveform_geom_factor = float(max(abs(spin2_plus), abs(spin2_minus)) / 4.0)
        source_angle_average_factor = 4.0 / 5.0
    else:
        pattern = "generic_projected"
        waveform_geom_factor = observer_projected_rss
        source_angle_average_factor = 1.0

    return TransitionGeometry(
        initial_state=initial_state,
        final_state=final_state,
        delta_m=delta_m,
        pattern=pattern,
        radial_overlap=radial_overlap,
        waveform_geom_factor=waveform_geom_factor,
        observer_projected_rss=observer_projected_rss,
        observer_plus=observer_plus,
        observer_cross=observer_cross,
        source_angle_average_factor=source_angle_average_factor,
        quadrupole_matrix=quadrupole_matrix,
    )


def angle_average_factor_from_metadata(metadata: dict[str, str]) -> float:
    component = str(metadata.get("component", "")).lower()
    module = str(metadata.get("module", "")).lower()
    family = str(metadata.get("transition_family", "")).lower()
    if "pure_binary" in component or component == "pure binary template":
        return 4.0 / 5.0
    if "644" in module or family == "bohr":
        return 8.0 / 15.0
    if "211" in module or "322" in module or family in {"fine", "hyperfine"}:
        return 4.0 / 5.0
    return 4.0 / 5.0
