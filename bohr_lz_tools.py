from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def lz_probability_and_coherence(z_lz: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z_lz, dtype=float)
    probability = 1.0 - np.exp(-2.0 * np.pi * z)
    coherence = np.sqrt(np.clip(probability * (1.0 - probability), 0.0, None))
    return probability, coherence


def solve_standard_lz(
    z_lz: float,
    width_max: float = 42.0,
    samples: int = 2600,
) -> tuple[np.ndarray, np.ndarray, float]:
    coupling = np.sqrt(float(z_lz))
    tau_max = width_max * coupling

    def rhs(tau: float, y: np.ndarray) -> list[float]:
        cg = y[0] + 1j * y[1]
        ce = y[2] + 1j * y[3]
        d_cg = -1j * (0.5 * tau * cg + coupling * ce)
        d_ce = -1j * (coupling * cg - 0.5 * tau * ce)
        return [d_cg.real, d_cg.imag, d_ce.real, d_ce.imag]

    tau_values = np.linspace(-tau_max, tau_max, int(samples))
    solution = solve_ivp(
        rhs,
        (float(tau_values[0]), float(tau_values[-1])),
        [1.0, 0.0, 0.0, 0.0],
        t_eval=tau_values,
        method="DOP853",
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    cg = solution.y[0] + 1j * solution.y[1]
    ce = solution.y[2] + 1j * solution.y[3]
    coherence = np.abs(np.conj(cg) * ce)
    width_values = tau_values / max(coupling, 1.0e-300)
    return width_values, coherence, float(abs(np.conj(cg[-1]) * ce[-1]))
