from pathlib import Path

import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

import lowfre322
from lowfre_shared import build_paper_inspired_lowfreq_profile


BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / "figures"
DIAGNOSTIC_DIR = BASE_DIR / "diagnostics"


def make_simulator(*, multi_harmonic_drive, harmonics_to_keep, module_stem):
    profile = build_paper_inspired_lowfreq_profile()
    kwargs = dict(profile.source_kwargs)
    kwargs.update(
        {
            "M_bh": 1500.0,
            "M_star": 0.5,
            "alpha": 0.25,
            "multi_harmonic_drive": bool(multi_harmonic_drive),
            "harmonics_to_keep": int(harmonics_to_keep),
            "cloud_evolution_mode": "band_gated",
            "orbital_backreaction_mode": "selected_rwa",
            "module_stem": module_stem,
        }
    )
    return lowfre322.EccentricResonantTidalGA(**kwargs), profile


def solve_variant(label, *, multi_harmonic_drive, harmonics_to_keep):
    sim, profile = make_simulator(
        multi_harmonic_drive=multi_harmonic_drive,
        harmonics_to_keep=harmonics_to_keep,
        module_stem=f"lowfre322_{label}",
    )
    orbit, cloud = sim.solve_coupled_system(
        duration_yr=profile.run_kwargs["duration_yr"],
        secular_samples=profile.run_kwargs["secular_samples"],
    )
    return sim, orbit, cloud


def interp_complex(t_new, t_old, values):
    values = np.asarray(values, dtype=np.complex128)
    real = np.interp(t_new, t_old, values.real)
    imag = np.interp(t_new, t_old, values.imag)
    return real + 1j * imag


def main():
    selected = solve_variant("selected", multi_harmonic_drive=False, harmonics_to_keep=1)
    multi3 = solve_variant("multi3", multi_harmonic_drive=True, harmonics_to_keep=3)

    _, orbit_sel, cloud_sel = selected
    _, orbit_m3, cloud_m3 = multi3

    t0 = max(float(cloud_sel["t"][0]), float(cloud_m3["t"][0]))
    t1 = min(float(cloud_sel["t"][-1]), float(cloud_m3["t"][-1]))
    t = np.linspace(t0, t1, 2400)

    ce_sel = interp_complex(t, cloud_sel["t"], cloud_sel["ce_tilde"])
    ce_m3 = interp_complex(t, cloud_m3["t"], cloud_m3["ce_tilde"])
    cg_sel = interp_complex(t, cloud_sel["t"], cloud_sel["cg"])
    cg_m3 = interp_complex(t, cloud_m3["t"], cloud_m3["cg"])

    pop_sel = np.abs(ce_sel) ** 2
    pop_m3 = np.abs(ce_m3) ** 2
    coh_sel = np.abs(np.conj(cg_sel) * ce_sel)
    coh_m3 = np.abs(np.conj(cg_m3) * ce_m3)

    a_sel = np.interp(t, orbit_sel["t"], orbit_sel["a"])
    a_m3 = np.interp(t, orbit_m3["t"], orbit_m3["a"])

    pop_scale = max(float(np.max(pop_sel)), 1.0e-300)
    coh_scale = max(float(np.max(coh_sel)), 1.0e-300)
    a_scale = max(float(np.max(np.abs(a_sel))), 1.0e-300)

    metrics = {
        "transition": "|300> -> |322>",
        "selected_active_harmonics": ",".join(str(n) for n in cloud_sel["active_harmonics"]),
        "multi3_active_harmonics": ",".join(str(n) for n in cloud_m3["active_harmonics"]),
        "max_abs_population_difference": float(np.max(np.abs(pop_m3 - pop_sel))),
        "max_fractional_population_difference": float(np.max(np.abs(pop_m3 - pop_sel)) / pop_scale),
        "max_abs_coherence_difference": float(np.max(np.abs(coh_m3 - coh_sel))),
        "max_fractional_coherence_difference": float(np.max(np.abs(coh_m3 - coh_sel)) / coh_scale),
        "max_fractional_semimajor_axis_difference": float(np.max(np.abs(a_m3 - a_sel)) / a_scale),
        "final_fractional_semimajor_axis_difference": float((a_m3[-1] - a_sel[-1]) / max(abs(a_sel[-1]), 1.0e-300)),
        "duration_years": float((t1 - t0) / (365.25 * 24.0 * 3600.0)),
    }

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    out_txt = DIAGNOSTIC_DIR / "lowfre322_selected_rwa_vs_multi3.txt"
    with out_txt.open("w", encoding="utf-8") as handle:
        for key, value in metrics.items():
            handle.write(f"{key}: {value}\n")

    years = (t - t[0]) / (365.25 * 24.0 * 3600.0)
    fig, axs = plt.subplots(3, 1, figsize=(6.8, 5.8), sharex=True, constrained_layout=True)
    axs[0].plot(years, pop_sel, label="selected RWA", lw=1.1)
    axs[0].plot(years, pop_m3, label="3 harmonics", lw=1.0, ls="--")
    axs[0].set_ylabel(r"$|c_e|^2$")
    axs[0].legend(frameon=False, fontsize=8)

    axs[1].plot(years, coh_sel, lw=1.1)
    axs[1].plot(years, coh_m3, lw=1.0, ls="--")
    axs[1].set_ylabel(r"$|c_g^\ast \tilde c_e|$")

    axs[2].plot(years, (a_m3 - a_sel) / np.maximum(np.abs(a_sel), 1.0e-300), color="#B23A48", lw=1.0)
    axs[2].set_ylabel(r"$\Delta a/a$")
    axs[2].set_xlabel("Time from start [yr]")

    for ax in axs:
        ax.tick_params(axis="both", labelsize=8, direction="in", top=True, right=True)
        ax.yaxis.label.set_size(9)
        ax.xaxis.label.set_size(9)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_fig = FIGURE_DIR / "lowfre322_selected_rwa_convergence.pdf"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_txt}")
    print(f"Saved {out_fig}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
