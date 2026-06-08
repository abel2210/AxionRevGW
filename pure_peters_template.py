from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


G = 6.6743e-11
C = 2.99792458e8
M_SUN = 1.98847e30
MPC = 3.085677581491367e22
YEAR = 365.25 * 24.0 * 3600.0


def solve_kepler(mean_anomaly, eccentricity, max_iter=15, tol=1.0e-12):
    mean_anomaly = np.asarray(mean_anomaly, dtype=float)
    eccentricity = np.asarray(eccentricity, dtype=float)
    if eccentricity.ndim == 0:
        eccentricity = np.full_like(mean_anomaly, float(np.clip(eccentricity, 0.0, 0.999)))
    else:
        eccentricity = np.broadcast_to(np.clip(eccentricity, 0.0, 0.999), mean_anomaly.shape).astype(float)

    guess = np.where(eccentricity < 0.8, mean_anomaly, np.pi * np.ones_like(mean_anomaly))
    for _ in range(max_iter):
        residual = guess - eccentricity * np.sin(guess) - mean_anomaly
        jacobian = 1.0 - eccentricity * np.cos(guess)
        step = residual / jacobian
        guess -= step
        if np.max(np.abs(step)) < tol:
            break
    return guess


def peters_rhs(_, y, m1_kg, m2_kg):
    a, e, phi = y
    e = float(np.clip(e, 0.0, 0.999))
    m_tot = m1_kg + m2_kg
    one_minus_e2 = max(1.0e-12, 1.0 - e * e)
    prefactor = G**3 * m_tot * m1_kg * m2_kg / C**5
    dadt = (
        -(64.0 / 5.0)
        * prefactor
        / (a**3 * one_minus_e2**3.5)
        * (1.0 + (73.0 / 24.0) * e**2 + (37.0 / 96.0) * e**4)
    )
    dedt = (
        -(304.0 / 15.0)
        * e
        * prefactor
        / (a**4 * one_minus_e2**2.5)
        * (1.0 + (121.0 / 304.0) * e**2)
    )
    omega = np.sqrt(G * m_tot / a**3)
    return [dadt, dedt, omega]


def binary_strain_time_domain(semi_major_axis, eccentricity, mean_anomaly, m1_kg, m2_kg, distance_m):
    mean_anomaly = np.mod(np.asarray(mean_anomaly, dtype=float), 2.0 * np.pi)
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

    m_tot = m1_kg + m2_kg
    mean_motion = np.sqrt(G * m_tot / semi_major_axis**3)
    e_dot = mean_motion / radial_factor
    x_dot = -semi_major_axis * sin_e * e_dot
    y_dot = semi_major_axis * np.sqrt(one_minus_e2) * cos_e * e_dot

    acc_factor = -G * m_tot / radius**3
    x_ddot = acc_factor * x
    y_ddot = acc_factor * y

    reduced_mass = m1_kg * m2_kg / m_tot
    i_ddot_xx = 2.0 * reduced_mass * (x_dot**2 + x * x_ddot)
    i_ddot_yy = 2.0 * reduced_mass * (y_dot**2 + y * y_ddot)
    return (G / (C**4 * distance_m)) * (i_ddot_yy - i_ddot_xx)


def build_pure_peters_frequency_domain(
    *,
    primary_mass_msun=1.0e-3,
    secondary_mass_msun=1.0e-4,
    eccentricity_init=0.65,
    orbital_frequency_init_hz=1.5e3,
    distance_mpc=1.0e-3,
    redshift=0.0,
    window_orbits=20.0,
    sample_points=8192,
    pad_factor=4,
    window_beta=14.0,
):
    m1_kg = float(primary_mass_msun) * M_SUN
    m2_kg = float(secondary_mass_msun) * M_SUN
    distance_m = float(distance_mpc) * MPC
    f_orb_init = float(orbital_frequency_init_hz)
    omega_init = 2.0 * np.pi * f_orb_init
    a_init = (G * (m1_kg + m2_kg) / omega_init**2) ** (1.0 / 3.0)
    duration_source_s = float(window_orbits) / f_orb_init

    sol = solve_ivp(
        peters_rhs,
        (0.0, duration_source_s),
        [a_init, float(eccentricity_init), 0.0],
        args=(m1_kg, m2_kg),
        dense_output=True,
        max_step=min(duration_source_s / 2500.0, 10.0 / f_orb_init),
        rtol=1.0e-9,
        atol=[1.0e-6, 1.0e-12, 1.0e-9],
    )
    t_source = np.linspace(0.0, float(sol.t[-1]), int(sample_points))
    a, e, phi = sol.sol(t_source)
    h_binary = binary_strain_time_domain(a, e, phi, m1_kg, m2_kg, distance_m)

    t_obs = t_source * (1.0 + float(redshift))
    dt_obs = float(t_obs[1] - t_obs[0])
    n_fft = max(len(t_obs), int(pad_factor * len(t_obs)))
    window = np.kaiser(len(t_obs), float(window_beta))
    h_centered = h_binary - np.mean(h_binary)
    h_tilde = np.fft.rfft(h_centered * window, n=n_fft) * dt_obs
    freq_hz = np.fft.rfftfreq(n_fft, dt_obs)

    metadata = {
        "module": "highfre_pure_peters",
        "component": "pure_binary_template",
        "template_model": "peters",
        "redshift": f"{float(redshift):.16e}",
        "luminosity_distance_m": f"{distance_m:.16e}",
        "primary_mass_msun": f"{float(primary_mass_msun):.16e}",
        "secondary_mass_msun": f"{float(secondary_mass_msun):.16e}",
        "eccentricity_init": f"{float(eccentricity_init):.16e}",
        "orbital_frequency_init_hz": f"{f_orb_init:.16e}",
        "duration_source_s": f"{duration_source_s:.16e}",
        "window_orbits": f"{float(window_orbits):.16e}",
        "frequency_frame": "observer",
        "columns": "frequency_hz amplitude_abs h_tilde_real h_tilde_imag",
    }
    data = np.column_stack((freq_hz, np.abs(h_tilde), h_tilde.real, h_tilde.imag))
    return metadata, data


def save_pure_peters_frequency_amplitude(path: Path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata, data = build_pure_peters_frequency_domain(**kwargs)
    header = "\n".join(f"{key}={value}" for key, value in metadata.items())
    np.savetxt(path, data, header=header, comments="# ")
    print(f"Saved pure Peters frequency-amplitude data: {path}")
    return path
