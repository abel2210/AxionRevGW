from __future__ import annotations

import csv
from pathlib import Path

import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

import lowfre211
import lowfre211v
import lowfre322
import lowfre322v
from lowfre_shared import build_paper_inspired_lowfreq_profile


BASE_DIR = Path(__file__).resolve().parent
DIAGNOSTIC_DIR = BASE_DIR / "diagnostics" / "rwa_convergence"
FIGURE_DIR = DIAGNOSTIC_DIR

CONVERGENCE_CASES = [
    {
        "stem": "lowfre322",
        "transition": r"$|300\rangle\to|322\rangle$",
        "module": lowfre322,
    },
    {
        "stem": "lowfre322v",
        "transition": r"$|322\rangle\to|300\rangle$",
        "module": lowfre322v,
    },
    {
        "stem": "lowfre211",
        "transition": r"$|21{-}1\rangle\to|211\rangle$",
        "module": lowfre211,
    },
    {
        "stem": "lowfre211v",
        "transition": r"$|211\rangle\to|21{-}1\rangle$",
        "module": lowfre211v,
    },
]


def make_simulator(case, *, multi_harmonic_drive, harmonics_to_keep, module_stem):
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
    return case["module"].EccentricResonantTidalGA(**kwargs), profile


def solve_variant(case, label, *, multi_harmonic_drive, harmonics_to_keep):
    sim, profile = make_simulator(
        case,
        multi_harmonic_drive=multi_harmonic_drive,
        harmonics_to_keep=harmonics_to_keep,
        module_stem=f"{case['stem']}_{label}",
    )
    orbit, cloud = sim.solve_coupled_system(
        duration_yr=profile.run_kwargs["duration_yr"],
        secular_samples=profile.run_kwargs["secular_samples"],
    )
    return sim, orbit, cloud, profile


def interp_complex(t_new, t_old, values):
    values = np.asarray(values, dtype=np.complex128)
    real = np.interp(t_new, t_old, values.real)
    imag = np.interp(t_new, t_old, values.imag)
    return real + 1j * imag


def evaluate_case(case):
    selected = solve_variant(case, "selected", multi_harmonic_drive=False, harmonics_to_keep=1)
    multi3 = solve_variant(case, "multi3", multi_harmonic_drive=True, harmonics_to_keep=3)

    _, orbit_sel, cloud_sel, profile = selected
    _, orbit_m3, cloud_m3, _ = multi3

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
        "case": case["stem"],
        "transition": case["transition"],
        "selected_active_harmonics": ",".join(str(n) for n in cloud_sel["active_harmonics"]),
        "multi3_active_harmonics": ",".join(str(n) for n in cloud_m3["active_harmonics"]),
        "max_abs_population_difference": float(np.max(np.abs(pop_m3 - pop_sel))),
        "selected_population_scale": pop_scale,
        "max_fractional_population_difference": float(np.max(np.abs(pop_m3 - pop_sel)) / pop_scale),
        "max_abs_coherence_difference": float(np.max(np.abs(coh_m3 - coh_sel))),
        "selected_coherence_scale": coh_scale,
        "max_fractional_coherence_difference": float(np.max(np.abs(coh_m3 - coh_sel)) / coh_scale),
        "max_fractional_semimajor_axis_difference": float(np.max(np.abs(a_m3 - a_sel)) / a_scale),
        "final_fractional_semimajor_axis_difference": float(
            (a_m3[-1] - a_sel[-1]) / max(abs(a_sel[-1]), 1.0e-300)
        ),
        "duration_years": float((t1 - t0) / (365.25 * 24.0 * 3600.0)),
        "alpha": 0.25,
        "observation_years": float(profile.run_kwargs["duration_yr"]),
    }
    series = {
        "t": t,
        "pop_sel": pop_sel,
        "pop_m3": pop_m3,
        "coh_sel": coh_sel,
        "coh_m3": coh_m3,
        "a_sel": a_sel,
        "a_m3": a_m3,
    }
    return metrics, series


def save_case_txt(metrics):
    out_txt = DIAGNOSTIC_DIR / f"{metrics['case']}_selected_rwa_vs_multi3.txt"
    with out_txt.open("w", encoding="utf-8") as handle:
        for key, value in metrics.items():
            handle.write(f"{key}: {value}\n")
    return out_txt


def save_summary(rows):
    fields = [
        "case",
        "transition",
        "selected_active_harmonics",
        "multi3_active_harmonics",
        "max_abs_population_difference",
        "selected_population_scale",
        "max_fractional_population_difference",
        "max_abs_coherence_difference",
        "selected_coherence_scale",
        "max_fractional_coherence_difference",
        "max_fractional_semimajor_axis_difference",
        "final_fractional_semimajor_axis_difference",
        "duration_years",
        "alpha",
    ]
    out_csv = DIAGNOSTIC_DIR / "lowfreq_selected_rwa_vs_multi3_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    out_md = DIAGNOSTIC_DIR / "lowfreq_selected_rwa_vs_multi3_summary.md"
    with out_md.open("w", encoding="utf-8") as handle:
        handle.write("# Low-frequency selected-RWA vs three-harmonic check\n\n")
        handle.write("All runs use the low-frequency reference source, alpha=0.25, ")
        handle.write("a 0.4 yr observation, selected backreaction, and compare the selected harmonic ")
        handle.write("against the nearest three harmonics in the cloud drive.\n\n")
        handle.write(
            "| case | transition | selected n | multi3 n | max d|ce|^2 | "
            "max dC | max dC/scale | max da/a |\n"
        )
        handle.write("|---|---|---:|---|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['case']} | {row['transition']} | {row['selected_active_harmonics']} | "
                f"{row['multi3_active_harmonics']} | "
                f"{row['max_abs_population_difference']:.6g} | "
                f"{row['max_abs_coherence_difference']:.6g} | "
                f"{row['max_fractional_coherence_difference']:.6g} | "
                f"{row['max_fractional_semimajor_axis_difference']:.6g} |\n"
            )
    return out_csv, out_md


def save_representative_figure(series):
    t = series["t"]
    years = (t - t[0]) / (365.25 * 24.0 * 3600.0)
    fig, axs = plt.subplots(3, 1, figsize=(6.8, 5.8), sharex=True, constrained_layout=True)
    axs[0].plot(years, series["pop_sel"], label="selected RWA", lw=1.1)
    axs[0].plot(years, series["pop_m3"], label="3 harmonics", lw=1.0, ls="--")
    axs[0].set_ylabel(r"$|c_e|^2$")
    axs[0].legend(frameon=False, fontsize=8)

    axs[1].plot(years, series["coh_sel"], lw=1.1)
    axs[1].plot(years, series["coh_m3"], lw=1.0, ls="--")
    axs[1].set_ylabel(r"$|c_g^\ast \tilde c_e|$")

    a_sel = np.asarray(series["a_sel"], dtype=float)
    a_m3 = np.asarray(series["a_m3"], dtype=float)
    axs[2].plot(years, (a_m3 - a_sel) / np.maximum(np.abs(a_sel), 1.0e-300), color="#B23A48", lw=1.0)
    axs[2].set_ylabel(r"$\Delta a/a$")
    axs[2].set_xlabel("Time from start [yr]")

    for ax in axs:
        ax.tick_params(axis="both", labelsize=8, direction="in", top=True, right=True)
        ax.yaxis.label.set_size(9)
        ax.xaxis.label.set_size(9)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_fig = FIGURE_DIR / "lowfre322_rwa_convergence.pdf"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_fig


def main():
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    representative_series = None

    for case in CONVERGENCE_CASES:
        metrics, series = evaluate_case(case)
        rows.append(metrics)
        out_txt = save_case_txt(metrics)
        print(f"Saved {out_txt}")
        if case["stem"] == "lowfre322":
            representative_series = series

    out_csv, out_md = save_summary(rows)
    print(f"Saved {out_csv}")
    print(f"Saved {out_md}")

    if representative_series is not None:
        out_fig = save_representative_figure(representative_series)
        print(f"Saved {out_fig}")

    for row in rows:
        print(
            row["case"],
            "selected=", row["selected_active_harmonics"],
            "multi3=", row["multi3_active_harmonics"],
            "pop=", f"{row['max_fractional_population_difference']:.6g}",
            "coh=", f"{row['max_fractional_coherence_difference']:.6g}",
            "a=", f"{row['max_fractional_semimajor_axis_difference']:.6g}",
        )


if __name__ == "__main__":
    main()
