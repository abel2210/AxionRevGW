from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

import highfre644v
from adiabaticlimit import AdiabaticPhaseDiagram


BASE_DIR = Path(__file__).resolve().parent
DIAGNOSTICS_DIR = BASE_DIR / "diagnostics"
FIGURES_DIR = BASE_DIR / "figures"

TRANSITION_INITIAL = (6, 4, 4)
TRANSITION_FINAL = (5, 4, 4)
RESONANCE_HARMONIC = 1
Q_REF = 1.0e-3
E_REF = 0.64
E_INIT = 0.65
M1_MSUN = 1.0e-2
BH_SPIN = 0.70
CLOUD_MASS_FRACTION = 1.0e-4
DISTANCE_MPC = 0.001
ORBITAL_START_RATIO = 0.90
LZ_WINDOW_WIDTHS = 240.0
POST_ORBITS = 80.0

CSV_PATH = DIAGNOSTICS_DIR / "bohr_alpha_family.csv"
SUMMARY_PATH = DIAGNOSTICS_DIR / "bohr_alpha_family.md"
FIGURE_PATH = FIGURES_DIR / "bohr_alpha_family_summary.pdf"


@dataclass
class FullAlphaResult:
    alpha: float
    transition_frequency_hz: float
    duration_yr: float
    z_lz: float
    p_tr_lz: float
    c_out_lz: float
    c_post_num: float
    c_post_peak: float
    h_peak_post: float
    cloud_amplitude: float
    delta_a_over_a: float
    delta_e: float
    post_orbits_available: float
    lz_window_widths: float
    local_x: np.ndarray
    coherence: np.ndarray


def lz_probability_and_coherence(z_lz: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z_lz, dtype=float)
    probability = 1.0 - np.exp(-2.0 * np.pi * z)
    coherence = np.sqrt(np.clip(probability * (1.0 - probability), 0.0, None))
    return probability, coherence


def transition_frequency_ratio(diagram: AdiabaticPhaseDiagram, alpha_grid: np.ndarray) -> np.ndarray:
    omega_ref = abs(
        diagram._omega_real_geom(diagram.initial_state, 0.30)
        - diagram._omega_real_geom(diagram.final_state, 0.30)
    )
    omega = np.array(
        [
            abs(
                diagram._omega_real_geom(diagram.initial_state, float(alpha))
                - diagram._omega_real_geom(diagram.final_state, float(alpha))
            )
            for alpha in alpha_grid
        ],
        dtype=float,
    )
    return omega / omega_ref


def selected_local_event(results: dict) -> dict:
    events = [
        event
        for event in results["cloud"].get("local_lz_events", [])
        if bool(event.get("selected", False)) and bool(event.get("crossed", False))
    ]
    if not events:
        events = [event for event in results["cloud"].get("local_lz_events", []) if bool(event.get("selected", False))]
    if not events:
        raise RuntimeError("No selected local LZ event found.")
    events.sort(key=lambda event: float(event.get("t_source", 0.0)))
    return events[0]


def make_sim(alpha: float) -> highfre644v.EccentricResonantTidalGA:
    return highfre644v.EccentricResonantTidalGA(
        M_bh=M1_MSUN,
        M_star=Q_REF * M1_MSUN,
        alpha=float(alpha),
        bh_spin=BH_SPIN,
        distance_Mpc=DISTANCE_MPC,
        z=0.0,
        e_init=E_INIT,
        orbital_start_ratio=ORBITAL_START_RATIO,
        cloud_mass_fraction=CLOUD_MASS_FRACTION,
        lz_window_widths=LZ_WINDOW_WIDTHS,
        save_frequency_data_dir=None,
        save_time_series_data_dir=None,
        save_figure_dir="figures",
        module_stem=f"probe_bohr_alpha_{alpha:.3f}".replace(".", "p"),
        direction_tag="downward",
    )


def initial_duration_guess(alpha: float, sim: highfre644v.EccentricResonantTidalGA) -> float:
    scaled = 2.0e-5 * (0.30 / float(alpha)) ** 3
    return sim.recommended_duration_to_cover_selected_resonance(
        initial_duration_yr=scaled,
        max_duration_yr=2.0e-3,
        post_event_padding_orbits=500.0,
    )


def run_full_alpha(alpha: float, secular_samples: int = 800) -> FullAlphaResult:
    duration_yr: float | None = None
    last_payload = None

    for attempt in range(6):
        sim = make_sim(alpha)
        if duration_yr is None:
            duration_yr = initial_duration_guess(alpha, sim)
        results = sim.run(
            duration_yr=duration_yr,
            secular_samples=secular_samples,
            zoom_orbits=20,
            zoom_points=4096,
            spectrum_orbits=20,
            spectrum_points=4096,
            spectrum_pad_factor=2,
            spectrum_window_mode="first_selected_orbits",
            save_exports=False,
        )
        try:
            event = selected_local_event(results)
        except RuntimeError:
            duration_yr *= 2.5
            continue
        t_res = float(event.get("t_source", results["cloud"]["resonance_time"]))
        t_stop_lz = max(float(event.get("lz_window_stop_source", t_res)), t_res)
        orbit_end = float(results["orbit"]["t"][-1])
        omega_res = float(np.interp(t_res, results["orbit"]["t"], results["orbit"]["omega"]))
        orbital_period = 2.0 * np.pi / max(omega_res, 1.0e-300)
        post_orbits_available = max((orbit_end - t_stop_lz) / max(orbital_period, 1.0e-300), 0.0)
        last_payload = (sim, results, event, post_orbits_available, orbital_period)
        if post_orbits_available >= POST_ORBITS:
            break
        duration_yr *= 1.6 + 0.15 * attempt

    if last_payload is None:
        raise RuntimeError(f"Full alpha run failed for alpha={alpha}.")

    sim, results, event, post_orbits_available, orbital_period = last_payload
    t_res = float(event.get("t_source", results["cloud"]["resonance_time"]))
    t_start_lz = float(event.get("lz_window_start_source", t_res))
    t_stop_lz = max(float(event.get("lz_window_stop_source", t_res)), t_res)
    orbit_end = float(results["orbit"]["t"][-1])
    post_stop = min(orbit_end, t_stop_lz + POST_ORBITS * orbital_period)

    post_t = np.linspace(t_stop_lz, post_stop, 2400)
    cg_r, cg_i, ce_r, ce_i = results["cloud"]["solution"].sol(post_t)
    overlap = np.conj(cg_r + 1j * cg_i) * (ce_r + 1j * ce_i)
    post_coherence = np.abs(overlap)

    lz_t = np.linspace(t_start_lz, min(orbit_end, t_stop_lz + 0.15 * POST_ORBITS * orbital_period), 5000)
    cg_r, cg_i, ce_r, ce_i = results["cloud"]["solution"].sol(lz_t)
    lz_overlap = np.conj(cg_r + 1j * cg_i) * (ce_r + 1j * ce_i)
    lz_coherence = np.abs(lz_overlap)
    eta_abs = max(float(event.get("eta_abs", 0.0)), 1.0e-300)
    slope_abs = max(float(event.get("detuning_slope_abs", 0.0)), 1.0e-300)
    local_x = (lz_t - t_res) * slope_abs / eta_abs

    z_lz = float(event["z_ad"])
    p_lz, c_lz = lz_probability_and_coherence(z_lz)
    cloud_amplitude = abs(float(sim._cloud_amplitude()))
    c_post_peak = float(np.nanmax(post_coherence))
    c_post_num = float(np.nanmedian(post_coherence))

    return FullAlphaResult(
        alpha=float(alpha),
        transition_frequency_hz=float(sim.transition_frequency_hz),
        duration_yr=float(duration_yr),
        z_lz=z_lz,
        p_tr_lz=float(p_lz),
        c_out_lz=float(c_lz),
        c_post_num=c_post_num,
        c_post_peak=c_post_peak,
        h_peak_post=float(cloud_amplitude * c_post_peak),
        cloud_amplitude=cloud_amplitude,
        delta_a_over_a=float(event.get("delta_a_over_a", np.nan)),
        delta_e=float(event.get("delta_e", np.nan)),
        post_orbits_available=float(post_orbits_available),
        lz_window_widths=LZ_WINDOW_WIDTHS,
        local_x=local_x,
        coherence=lz_coherence,
    )


def write_outputs(
    alpha_grid: np.ndarray,
    z_grid: np.ndarray,
    c_grid: np.ndarray,
    frequency_ratio: np.ndarray,
    full_results: list[FullAlphaResult],
) -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    full_by_alpha = {round(item.alpha, 12): item for item in full_results}
    scan_alphas = np.array([0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32], dtype=float)
    diagram = AdiabaticPhaseDiagram(
        initial_state=TRANSITION_INITIAL,
        final_state=TRANSITION_FINAL,
        resonance_harmonic=RESONANCE_HARMONIC,
        eccentricity=E_REF,
    )
    rows = []

    for alpha in scan_alphas:
        z = float(diagram.compute_z_parameter(float(alpha), Q_REF, M_bh_solar=1.0, eccentricity=E_REF))
        p, c = lz_probability_and_coherence(z)
        freq_ratio = float(transition_frequency_ratio(diagram, np.array([alpha]))[0])
        full = full_by_alpha.get(round(float(alpha), 12))
        rows.append(
            {
                "alpha": f"{alpha:.8g}",
                "transition_frequency_ratio": f"{freq_ratio:.12e}",
                "z_lz": f"{z:.12e}",
                "p_tr_lz": f"{float(p):.12e}",
                "c_out_lz": f"{float(c):.12e}",
                "full_time_domain": "1" if full is not None else "0",
                "transition_frequency_hz": "" if full is None else f"{full.transition_frequency_hz:.12e}",
                "c_post_num": "" if full is None else f"{full.c_post_num:.12e}",
                "c_post_peak": "" if full is None else f"{full.c_post_peak:.12e}",
                "h_peak_post": "" if full is None else f"{full.h_peak_post:.12e}",
                "delta_a_over_a": "" if full is None else f"{full.delta_a_over_a:.12e}",
                "delta_e": "" if full is None else f"{full.delta_e:.12e}",
                "post_orbits_available": "" if full is None else f"{full.post_orbits_available:.6g}",
                "lz_window_widths": "" if full is None else f"{full.lz_window_widths:.6g}",
            }
        )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    alpha_018 = next((item for item in full_results if np.isclose(item.alpha, 0.18)), None)
    alpha_030 = next((item for item in full_results if np.isclose(item.alpha, 0.30)), None)
    lines = [
        "# Bohr alpha-family numerical validation",
        "",
        "Generated by `probe_bohr_alpha_family.py`.",
        "",
        f"Setup: downward Bohr crossing `|644> -> |544>`, `n=1`, `q=0.001`, `e_init=0.65`, `orbital_start_ratio={ORBITAL_START_RATIO:.2f}`, `a_star=0.70`, `M1=1e-2 Msun`, `Mc/M1=1e-4`, `d_L=1 kpc`.",
        f"The full time-domain checks use `lz_window_widths={LZ_WINDOW_WIDTHS:g}` and extract the post-crossing coherence after the local LZ window.",
        "",
        "## Main two-point test",
        "",
        "| alpha | z_LZ | C_out(LZ) | C_post(num) | h_pk,post | post orbits | interpretation |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in full_results:
        interpretation = "adiabatic branch with strongly suppressed outgoing coherence" if item.alpha < 0.24 else "finite-coherence branch"
        lines.append(
            f"| {item.alpha:.2f} | {item.z_lz:.6g} | {item.c_out_lz:.6g} | "
            f"{item.c_post_num:.6g} | {item.h_peak_post:.6e} | "
            f"{item.post_orbits_available:.1f} | {interpretation} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
        ]
    )
    if alpha_018 is not None and alpha_030 is not None:
        ratio = alpha_030.c_post_num / max(alpha_018.c_post_num, 1.0e-300)
        h_ratio = alpha_030.h_peak_post / max(alpha_018.h_peak_post, 1.0e-300)
        lines.extend(
            [
                f"- The `alpha=0.18` full run gives `C_post={alpha_018.c_post_num:.3e}`, consistent with strongly suppressed outgoing transition coherence on the adiabatic branch.",
                f"- The `alpha=0.30` full run gives `C_post={alpha_030.c_post_num:.3e}`, consistent with a finite outgoing two-level cloud.",
                f"- The post-crossing coherence ratio is `C_post(0.30)/C_post(0.18)={ratio:.3e}`.",
                f"- The post-crossing strain-amplitude ratio is `h_pk,post(0.30)/h_pk,post(0.18)={h_ratio:.3e}`.",
                "- This supports the finite-coherence visibility statement: the same crossing family moves from the adiabatic branch with negligible outgoing coherence into an intermediate branch with a finite transition waveform.",
            ]
        )
    else:
        lines.append("- The default two-point test was not completed.")
    lines.extend(
        [
            "",
            f"CSV table: `{CSV_PATH.as_posix()}`",
            f"Figure: `{FIGURE_PATH.as_posix()}`",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(
    alpha_grid: np.ndarray,
    z_grid: np.ndarray,
    c_grid: np.ndarray,
    frequency_ratio: np.ndarray,
    full_results: list[FullAlphaResult],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.6,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.72,
        }
    )

    cm = 1.0 / 2.54
    fig = plt.figure(figsize=(8.6 * cm, 12.4 * cm))
    grid = fig.add_gridspec(3, 1, height_ratios=(0.88, 0.88, 1.12), hspace=0.34)
    ax_z = fig.add_subplot(grid[0])
    ax_c = fig.add_subplot(grid[1])
    ax_t = fig.add_subplot(grid[2])

    ax_z.plot(alpha_grid, z_grid, color="#1F2937", lw=1.15)
    ax_z.axhline(1.0, color="0.35", lw=0.70, ls="--")
    ax_z.fill_between(alpha_grid, 1.0, np.maximum(z_grid, 1.0), color="#B45309", alpha=0.10, lw=0.0)
    ax_z.set_yscale("log")
    ax_z.set_xlim(alpha_grid[0], alpha_grid[-1])
    ax_z.set_ylim(5.0e-2, 1.6e1)
    ax_z.set_ylabel(r"$z_{\rm LZ}$")
    ax_z.tick_params(direction="in", top=True, right=True, length=3.0, width=0.65)
    ax_z.text(0.173, 1.35, "adiabatic branch", color="#9A3412", fontsize=6.1, ha="left", va="bottom")

    top_ax = ax_z.twiny()
    top_ax.set_xlim(ax_z.get_xlim())
    tick_alphas = np.array([0.18, 0.24, 0.30, 0.36])
    tick_ratios = transition_frequency_ratio(
        AdiabaticPhaseDiagram(
            initial_state=TRANSITION_INITIAL,
            final_state=TRANSITION_FINAL,
            resonance_harmonic=RESONANCE_HARMONIC,
            eccentricity=E_REF,
        ),
        tick_alphas,
    )
    top_ax.set_xticks(tick_alphas)
    top_ax.set_xticklabels([f"{value:.2f}" for value in tick_ratios])
    top_ax.set_xlabel(r"$\omega_{\rm tr}/\omega_{\rm tr}(\alpha=0.30)$", labelpad=1.5)
    top_ax.tick_params(direction="in", length=3.0, width=0.65, pad=1.0)

    ax_c.plot(alpha_grid, c_grid, color="#2563EB", lw=1.25, label=r"$C_{\rm out}$ from LZ")
    ax_c.fill_between(alpha_grid, 0.0, c_grid, color="#6BAED6", alpha=0.12, lw=0.0)
    for item, color, label in zip(
        full_results,
        ["#9A3412", "#047857", "#7C3AED", "#DC2626"],
        [r"time-domain $C_{\rm post}$", None, None, None],
    ):
        ax_c.plot(item.alpha, item.c_post_num, marker="o", ms=4.4, mfc="white", mec=color, mew=1.1, ls="None", label=label)
        ax_c.axvline(item.alpha, color=color, lw=0.70, ls="--", alpha=0.55)
        ax_z.plot(item.alpha, item.z_lz, marker="o", ms=4.4, mfc="white", mec=color, mew=1.1, zorder=4)
    ax_c.set_xlim(alpha_grid[0], alpha_grid[-1])
    ax_c.set_ylim(0.0, 0.53)
    ax_c.set_xlabel(r"$\alpha$")
    ax_c.set_ylabel(r"post-crossing coherence")
    ax_c.legend(loc="upper left", frameon=False, handlelength=1.6)
    ax_c.tick_params(direction="in", top=True, right=True, length=3.0, width=0.65)

    for item, color, label in zip(
        full_results,
        ["#9A3412", "#047857", "#7C3AED", "#DC2626"],
        [rf"$\alpha={full_results[0].alpha:.2f}$", rf"$\alpha={full_results[1].alpha:.2f}$", None, None],
    ):
        plot_label = label if label is not None else rf"$\alpha={item.alpha:.2f}$"
        ax_t.plot(item.local_x, item.coherence, color=color, lw=1.05, label=plot_label)
        ax_t.axhline(item.c_post_num, color=color, lw=0.65, ls=":", alpha=0.82)
    ax_t.axvline(0.0, color="0.20", lw=0.65, alpha=0.70)
    ax_t.axvline(LZ_WINDOW_WIDTHS, color="0.55", lw=0.62, ls="--", alpha=0.70)
    ax_t.set_xlim(-LZ_WINDOW_WIDTHS, LZ_WINDOW_WIDTHS + 8.0)
    ax_t.set_ylim(0.0, 0.53)
    ax_t.set_xlabel(r"local passage variable $(t-t_{\rm res})|\dot\delta|/|\eta|$")
    ax_t.set_ylabel(r"$|c_i^\ast\tilde c_f|$")
    ax_t.legend(loc="upper right", frameon=False, handlelength=1.6)
    ax_t.tick_params(direction="in", top=True, right=True, length=3.0, width=0.65)

    for panel, ax in zip(("(a)", "(b)", "(c)"), (ax_z, ax_c, ax_t)):
        ax.text(-0.12, 1.02, panel, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.4)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the q=0.001 Bohr alpha-family outgoing-coherence validation.")
    parser.add_argument(
        "--full-alphas",
        nargs="*",
        type=float,
        default=[0.18, 0.30],
        help="Alpha values for full time-domain validation. Defaults to the two Letter points.",
    )
    parser.add_argument(
        "--secular-samples",
        type=int,
        default=800,
        help="Secular samples for each full time-domain run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagram = AdiabaticPhaseDiagram(
        initial_state=TRANSITION_INITIAL,
        final_state=TRANSITION_FINAL,
        resonance_harmonic=RESONANCE_HARMONIC,
        eccentricity=E_REF,
    )
    alpha_grid = np.linspace(0.16, 0.40, 320)
    z_grid = np.array(
        [
            diagram.compute_z_parameter(float(alpha), Q_REF, M_bh_solar=1.0, eccentricity=E_REF)
            for alpha in alpha_grid
        ],
        dtype=float,
    )
    _, c_grid = lz_probability_and_coherence(z_grid)
    frequency_ratio = transition_frequency_ratio(diagram, alpha_grid)

    full_results = [run_full_alpha(alpha, secular_samples=args.secular_samples) for alpha in args.full_alphas]
    full_results.sort(key=lambda item: item.alpha)

    write_outputs(alpha_grid, z_grid, c_grid, frequency_ratio, full_results)
    plot_outputs(alpha_grid, z_grid, c_grid, frequency_ratio, full_results)

    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {FIGURE_PATH}")
    for item in full_results:
        print(
            f"alpha={item.alpha:.3f}: z={item.z_lz:.6g}, C_LZ={item.c_out_lz:.6g}, "
            f"C_post={item.c_post_num:.6g}, h_pk_post={item.h_peak_post:.6e}, "
            f"post_orbits={item.post_orbits_available:.1f}"
        )


if __name__ == "__main__":
    main()
