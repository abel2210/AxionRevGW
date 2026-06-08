from __future__ import annotations

from pathlib import Path

import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

import highfre644
import highfre644v


BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / "figures"


def run_bohr(module):
    sim = module.EccentricResonantTidalGA(
        M_bh=0.001,
        M_star=0.00001,
        alpha=0.3,
        bh_spin=0.7,
        distance_Mpc=0.001,
        z=0.0,
        e_init=0.65,
        f_orb_init=None,
        cloud_mass_fraction=0.005,
    )
    duration_yr = sim.recommended_duration_to_cover_selected_resonance(
        initial_duration_yr=2.0e-8,
        max_duration_yr=2.0e-6,
        post_event_padding_orbits=160.0,
    )
    results = sim.run(
        duration_yr=duration_yr,
        secular_samples=900,
        zoom_orbits=20,
        zoom_points=8192,
        spectrum_orbits=16,
        spectrum_points=8192,
        spectrum_pad_factor=4,
        save_exports=False,
    )
    return sim, results


def first_event(results):
    events = list(results["cloud"].get("local_lz_events", []))
    if events:
        return events[0]
    events = list(results["cloud"].get("resonance_events", []))
    if events:
        return events[0]
    raise RuntimeError("No Bohr resonance event found.")


def relative_delta_a(results, t_values):
    orbit = results["orbit"]
    template = results["template_orbit"]
    a = np.interp(t_values, orbit["t"], orbit["a"])
    a_template = np.interp(t_values, template["t"], template["a"])
    a_res = np.interp(float(results["cloud"]["resonance_time"]), orbit["t"], orbit["a"])
    return (a - a_template) / max(abs(a_res), 1.0e-300)


def binary_phase_residual_cycles(results, t_values, t_res):
    orbit = results["orbit"]
    template = results["template_orbit"]
    phi = np.asarray(orbit["solution"].sol(t_values)[2], dtype=float)
    phi_template = np.asarray(template["solution"].sol(t_values)[2], dtype=float)
    phi_res = float(np.asarray(orbit["solution"].sol(t_res)[2], dtype=float))
    phi_template_res = float(np.asarray(template["solution"].sol(t_res)[2], dtype=float))
    return (phi - phi_template - (phi_res - phi_template_res)) / (2.0 * np.pi)


def mark_resonance(ax):
    ax.axvline(0.0, color="0.15", lw=0.85, alpha=0.78)


def add_phase_residual_inset(ax, x_ms, phase_residual, color):
    inset = ax.inset_axes([0.56, 0.13, 0.38, 0.32])
    inset.set_facecolor((1.0, 1.0, 1.0, 0.88))
    inset.axhline(0.0, color="0.45", lw=0.9, alpha=0.25)
    inset.axvline(0.0, color="0.45", lw=0.9, alpha=0.25)
    inset.plot(x_ms, phase_residual, color=color, lw=0.95)
    inset.set_xlim(-18.0, 18.0)
    inset.set_ylim(-16.5, 16.5)
    inset.set_xticks([-15, 0, 15])
    inset.set_yticks([-15, 0, 15])
    inset.set_title(r"$\Delta\Phi_{\rm bin}/2\pi$", fontsize=4.6, pad=0.6)
    inset.tick_params(axis="both", which="major", labelsize=4.3, direction="in", top=True, right=True, pad=0.5)
    for spine in inset.spines.values():
        spine.set_linewidth(0.45)
        spine.set_alpha(0.75)


def direction_signed_axion_strain(sim, cloud, t_values):
    """Evaluate h_a with the signed transition phase for direction-sensitive plots."""
    cg_r, cg_i, ce_r, ce_i = cloud["solution"].sol(t_values)
    overlap = np.conj(cg_r + 1j * cg_i) * (ce_r + 1j * ce_i)
    signed_omega = getattr(sim, "transition_energy_change_omega", sim.transition_omega)
    phase = signed_omega * t_values
    h_axion = -sim._cloud_amplitude() * (
        overlap.real * np.cos(phase) - overlap.imag * np.sin(phase)
    )
    return h_axion, np.abs(overlap)


def plot_case(
    ax_orbit,
    ax_wave,
    sim,
    results,
    title,
    orbit_color,
    show_left_labels=True,
    show_coherence_axis=True,
):
    event = first_event(results)
    t_res = float(event.get("t_source", results["cloud"]["resonance_time"]))

    orbit = results["orbit"]

    t_orbit = np.asarray(orbit["t"], dtype=float)
    x_orbit_ms = (t_orbit - t_res) * 1.0e3
    delta_a = relative_delta_a(results, t_orbit)

    half_window_s = 18.0e-3
    window = sim.build_waveform_window_between(
        orbit,
        results["cloud"],
        t_res - half_window_s,
        t_res + half_window_s,
        sample_points=12000,
    )
    t_wave = np.asarray(window["t_source"], dtype=float)
    x_wave_ms = (t_wave - t_res) * 1.0e3
    h_axion, coherence = direction_signed_axion_strain(sim, results["cloud"], t_wave)
    phase_residual = binary_phase_residual_cycles(results, t_wave, t_res)

    mark_resonance(ax_orbit)
    ax_orbit.plot(x_orbit_ms, delta_a, color=orbit_color, lw=1.25)
    ax_orbit.set_title(title, fontsize=6.1, pad=1.2)
    if show_left_labels:
        ax_orbit.set_ylabel(r"$(a-a_{\rm P})/a_{\rm res}$", fontsize=5.8, labelpad=1.0)
    else:
        ax_orbit.tick_params(axis="y", labelleft=False)
    ax_orbit.set_ylim(-0.21, 0.21)

    mark_resonance(ax_wave)
    strain_stride = max(1, int(np.ceil(len(x_wave_ms) / 1200)))
    h_axion_scaled = 1.0e27 * h_axion
    ax_wave.plot(
        x_wave_ms[::strain_stride],
        h_axion_scaled[::strain_stride],
        color="#1F78B4",
        lw=0.42,
        alpha=0.82,
        label=r"$h_a$",
    )
    if show_left_labels:
        ax_wave.set_ylabel(r"$10^{27}h_a$", fontsize=5.8, color="#1F78B4", labelpad=1.0)
    else:
        ax_wave.tick_params(axis="y", labelleft=False)
    ax_wave.tick_params(axis="y", labelcolor="#1F78B4")
    ax_wave.set_xlabel(r"$t-t_{\rm res}$ [ms]", fontsize=5.8, labelpad=1.0)

    ax_coh = ax_wave.twinx()
    ax_wave.set_zorder(ax_coh.get_zorder() + 1)
    ax_wave.patch.set_visible(False)
    ax_coh.patch.set_visible(False)
    ax_coh.plot(x_wave_ms, coherence, color="#D55E00", lw=1.05, alpha=0.96, label=r"$|c_i^\ast\tilde c_f|$")
    if show_coherence_axis:
        ax_coh.set_ylabel(r"$|c_i^\ast\tilde c_f|$", fontsize=5.8, color="#D55E00", labelpad=1.0)
        ax_coh.tick_params(axis="y", labelcolor="#D55E00", pad=1.0)
    else:
        ax_coh.tick_params(axis="y", labelright=False, right=False)
    add_phase_residual_inset(ax_wave, x_wave_ms, phase_residual, orbit_color)

    for ax in (ax_orbit, ax_wave, ax_coh):
        ax.tick_params(axis="both", which="major", labelsize=5.2, direction="in", top=True, right=True, pad=1.0)

    for ax in (ax_orbit, ax_wave):
        ax.set_xlim(-18.0, 18.0)
        ax.grid(False)


def main():
    up_sim, up_results = run_bohr(highfre644)
    down_sim, down_results = run_bohr(highfre644v)

    cm_to_inch = 1.0 / 2.54
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.6 * cm_to_inch, 7.1 * cm_to_inch),
        sharex=True,
        constrained_layout=False,
    )

    plot_case(
        axes[0, 0],
        axes[1, 0],
        up_sim,
        up_results,
        r"$|544\rangle\rightarrow|644\rangle$",
        "#B23A48",
        show_left_labels=True,
        show_coherence_axis=False,
    )
    plot_case(
        axes[0, 1],
        axes[1, 1],
        down_sim,
        down_results,
        r"$|644\rangle\rightarrow|544\rangle$",
        "#009E73",
        show_left_labels=False,
        show_coherence_axis=True,
    )

    axes[0, 0].text(0.03, 0.86, "upward", transform=axes[0, 0].transAxes, fontsize=5.8)
    axes[0, 1].text(0.03, 0.86, "downward", transform=axes[0, 1].transAxes, fontsize=5.8)
    fig.subplots_adjust(left=0.14, right=0.88, bottom=0.115, top=0.915, wspace=0.055, hspace=0.055)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "bohr_orbit_time_summary_644_pair.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out}")


if __name__ == "__main__":
    main()
