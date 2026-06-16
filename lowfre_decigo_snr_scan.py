import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from pathlib import Path

# Keep BLAS/OpenMP from oversubscribing when the alpha grid is parallelized.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np

from lowfre211 import EccentricResonantTidalGA as Lowfre211
from lowfre211v import EccentricResonantTidalGA as Lowfre211v
from lowfre322 import EccentricResonantTidalGA as Lowfre322
from lowfre322v import EccentricResonantTidalGA as Lowfre322v
from lowfre_shared import build_paper_inspired_lowfreq_profile


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "snr_scan_data"
FIGURE_DIR = SCRIPT_DIR / "figures"
DETECTOR = "DECIGO"
REFERENCE_DISTANCE_KPC = 100.0
DEFAULT_DETECTOR_CURVE_KIND = "characteristic_strain"
SNR_MODEL_VERSION = "deterministic_strain_psd_v6_geometry_quadrature_decigo_n4_response"
DEFAULT_ALPHA_MIN = 0.10
DEFAULT_ALPHA_MAX = 0.30
SMALL_ALPHA_RECOMMENDED_MAX = 0.30
DEFAULT_SCAN_LOW_HZ = 2.0e-4
SNR_COLORBAR_MAX = 200.0
SNR_OVER_COLOR = "#1f2a7a"

PRESETS = {
    "211": {
        "class": Lowfre211,
        "label": r"$|21{-}1\rangle\to|211\rangle$",
        "direction": "upward",
    },
    "211v": {
        "class": Lowfre211v,
        "label": r"$|211\rangle\to|21{-}1\rangle$",
        "direction": "downward",
    },
    "322": {
        "class": Lowfre322,
        "label": r"$|300\rangle\to|322\rangle$",
        "direction": "upward",
    },
    "322v": {
        "class": Lowfre322v,
        "label": r"$|322\rangle\to|300\rangle$",
        "direction": "downward",
    },
}


def _float_env(name, default):
    value = os.environ.get(name)
    return float(default if value is None or value == "" else value)


def _int_env(name, default):
    value = os.environ.get(name)
    return int(default if value is None or value == "" else value)


def _bool_env(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _worker_count(total_jobs):
    default_workers = min(16, os.cpu_count() or 1)
    requested = _int_env("LOWFREQ_SCAN_WORKERS", default_workers)
    if requested <= 1 or total_jobs <= 1:
        return 1
    return min(requested, total_jobs)


def _detector_curve_kind():
    return os.environ.get("LOWFREQ_SCAN_DETECTOR_CURVE_KIND", DEFAULT_DETECTOR_CURVE_KIND).strip().lower()


def _scan_low_hz():
    return _float_env("LOWFREQ_SCAN_LOW_HZ", DEFAULT_SCAN_LOW_HZ)


def _extrapolate_detector_curve():
    return _bool_env("LOWFREQ_SCAN_EXTRAPOLATE_DETECTOR_CURVE", True)


def _row_matches_current_model(row):
    return (
        row.get("status") == "ok"
        and row.get("snr_model") == SNR_MODEL_VERSION
        and row.get("detector_curve_kind") == _detector_curve_kind()
    )


def _preset_list():
    raw = os.environ.get("LOWFREQ_SCAN_PRESETS", "211,211v,322,322v")
    presets = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [item for item in presets if item not in PRESETS]
    if unknown:
        raise ValueError("Unknown LOWFREQ_SCAN_PRESETS entries: " + ", ".join(unknown))
    return presets


def _alpha_grid():
    alpha_min = _float_env("LOWFREQ_SCAN_ALPHA_MIN", DEFAULT_ALPHA_MIN)
    alpha_max = _float_env("LOWFREQ_SCAN_ALPHA_MAX", DEFAULT_ALPHA_MAX)
    alpha_points = _int_env("LOWFREQ_SCAN_ALPHA_POINTS", 100)
    if alpha_min <= 0.0:
        raise ValueError("LOWFREQ_SCAN_ALPHA_MIN must be positive; alpha=0 is singular in this model.")
    if alpha_max > SMALL_ALPHA_RECOMMENDED_MAX:
        print(
            "Warning: alpha_max exceeds 0.30. "
            "This is outside the nominal small-alpha/KG benchmark regime and should be treated as diagnostic only."
        )
    return np.linspace(alpha_min, alpha_max, max(1, alpha_points))


def _distance_grid_kpc():
    d_min = _float_env("LOWFREQ_SCAN_DL_MIN_KPC", 1.0)
    d_max = _float_env("LOWFREQ_SCAN_DL_MAX_KPC", 1000.0)
    d_points = _int_env("LOWFREQ_SCAN_DL_POINTS", 80)
    if d_min <= 0.0 or d_max <= 0.0:
        raise ValueError("Distance grid endpoints must be positive.")
    return np.logspace(np.log10(d_min), np.log10(d_max), max(1, d_points))


def _source_kwargs(alpha_value, preset_cls):
    profile = build_paper_inspired_lowfreq_profile()
    kwargs = dict(profile.source_kwargs)
    kwargs.update(
        {
            "alpha": float(alpha_value),
            "distance_Mpc": REFERENCE_DISTANCE_KPC / 1000.0,
            "detector_names": (DETECTOR,),
            "detector_curve_kinds": {DETECTOR: _detector_curve_kind()},
            "module_stem": f"{preset_cls.__module__}_snrscan",
            "save_time_series_data_dir": None,
            "save_figure_dir": "figures",
        }
    )

    solver_profile = os.environ.get("LOWFREQ_SCAN_SOLVER_PROFILE")
    if solver_profile:
        kwargs["solver_profile"] = solver_profile
    for env_name, key in (
        ("LOWFREQ_SCAN_HANSEN_E_SAMPLES", "hansen_e_samples"),
        ("LOWFREQ_SCAN_HANSEN_M_SAMPLES", "hansen_M_samples"),
        ("LOWFREQ_SCAN_OVERLAP_GRID_POINTS", "overlap_grid_points"),
    ):
        if os.environ.get(env_name):
            kwargs[key] = _int_env(env_name, kwargs[key])
    return kwargs


def _run_kwargs():
    profile = build_paper_inspired_lowfreq_profile()
    run_kwargs = dict(profile.run_kwargs)
    overrides = {
        "LOWFREQ_SCAN_DURATION_YR": "duration_yr",
        "LOWFREQ_SCAN_SECULAR_SAMPLES": "secular_samples",
        "LOWFREQ_SCAN_SPECTRUM_PAD_FACTOR": "spectrum_pad_factor",
    }
    for env_name, key in overrides.items():
        if os.environ.get(env_name):
            if key.endswith("samples") or key.endswith("factor"):
                run_kwargs[key] = _int_env(env_name, run_kwargs[key])
            else:
                run_kwargs[key] = _float_env(env_name, run_kwargs[key])
    return run_kwargs


def _target_time_samples(simulator, duration_source_s):
    detector_low, detector_high = _scan_detector_band_hz(simulator)
    transition_obs_hz = simulator.transition_omega / (2.0 * np.pi * (1.0 + simulator.z))
    orbit_high_hz = simulator.binary_harmonics * simulator.f_orb_init / (1.0 + simulator.z)
    requested_high_hz = _float_env("LOWFREQ_SCAN_MAX_ANALYSIS_HZ", 0.03)
    target_high_hz = min(detector_high, requested_high_hz, max(detector_low, 2.0 * transition_obs_hz, orbit_high_hz))
    oversample = _float_env("LOWFREQ_SCAN_TIME_OVERSAMPLE", 2.5)
    minimum = _int_env("LOWFREQ_SCAN_TIME_SAMPLES_MIN", 4096)
    cap = _int_env("LOWFREQ_SCAN_TIME_SAMPLES_CAP", 750000)
    requested = int(np.ceil(2.0 * oversample * duration_source_s * target_high_hz)) + 1
    return int(max(minimum, min(cap, requested)))


def _scan_detector_band_hz(simulator):
    native_low, native_high = simulator._detector_band_hz(DETECTOR)
    low = max(float(_scan_low_hz()), 1.0e-8) if _extrapolate_detector_curve() else native_low
    return min(low, native_low), native_high


def _edge_log_slope(x_values, y_values, side):
    count = min(4, len(x_values))
    if count < 2:
        return 0.0
    if side == "left":
        x_edge = x_values[:count]
        y_edge = y_values[:count]
    else:
        x_edge = x_values[-count:]
        y_edge = y_values[-count:]
    slope, _ = np.polyfit(np.log10(x_edge), np.log10(y_edge), 1)
    return float(slope)


def _scan_detector_psd(simulator, f_hz):
    curve = simulator._load_detector_noise_curve(DETECTOR)
    if curve is None or not _extrapolate_detector_curve():
        return simulator.build_detector_psd(DETECTOR, f_hz)

    freq_curve = np.asarray(curve["freq_hz"], dtype=float)
    psd_curve = np.asarray(curve["psd"], dtype=float)
    safe_f = np.maximum(np.asarray(f_hz, dtype=float), 1.0e-8)
    log_f = np.log10(safe_f)
    log_curve_f = np.log10(freq_curve)
    log_curve_psd = np.log10(psd_curve)
    log_psd = np.interp(log_f, log_curve_f, log_curve_psd)

    below = safe_f < freq_curve[0]
    if np.any(below):
        left_slope = _edge_log_slope(freq_curve, psd_curve, "left")
        log_psd[below] = log_curve_psd[0] + left_slope * (log_f[below] - log_curve_f[0])

    above = safe_f > freq_curve[-1]
    if np.any(above):
        right_slope = _edge_log_slope(freq_curve, psd_curve, "right")
        log_psd[above] = log_curve_psd[-1] + right_slope * (log_f[above] - log_curve_f[-1])

    return np.maximum(10.0**log_psd, 1.0e-60)


def _axion_snr_from_window(simulator, waveform_window, pad_factor):
    frequency_domain = simulator.build_windowed_fft(waveform_window, pad_factor=pad_factor)
    freq_hz = np.asarray(frequency_domain["freq_hz"], dtype=float)
    detector_low, detector_high = _scan_detector_band_hz(simulator)
    mask = (freq_hz >= detector_low) & (freq_hz <= detector_high)
    if np.count_nonzero(mask) < 2:
        return {
            "snr": 0.0,
            "rho2": 0.0,
            "band_low_hz": detector_low,
            "band_high_hz": detector_high,
            "nyquist_hz": float(freq_hz[-1]),
            "df_hz": float(frequency_domain["df_hz"]),
        }

    h_axion = frequency_domain["h_tilde_axion"][mask]
    psd = _scan_detector_psd(simulator, freq_hz[mask])
    axion_norm = 4.0 * np.sum(np.abs(h_axion) ** 2 / np.maximum(psd, 1.0e-300)) * frequency_domain["df_hz"]
    rho2 = simulator._detector_snr_prefactor(DETECTOR) ** 2 * max(float(axion_norm), 0.0)
    return {
        "snr": float(np.sqrt(max(rho2, 0.0))),
        "rho2": float(rho2),
        "band_low_hz": detector_low,
        "band_high_hz": detector_high,
        "nyquist_hz": float(freq_hz[-1]),
        "df_hz": float(frequency_domain["df_hz"]),
    }


def _existing_rows(csv_path):
    rows = {}
    if not csv_path.exists():
        return rows
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                alpha = float(row["alpha"])
            except (KeyError, ValueError):
                continue
            if not _row_matches_current_model(row):
                continue
            rows[round(alpha, 12)] = row
    return rows


def _write_rows(csv_path, rows):
    fieldnames = [
        "preset",
        "alpha",
        "snr_model",
        "detector_curve_kind",
        "scan_low_hz",
        "detector_curve_extrapolated",
        "snr_ref",
        "rho2_ref",
        "transition_freq_obs_mhz",
        "peak_overlap_abs",
        "max_h_axion_ref",
        "time_samples",
        "nyquist_hz",
        "df_hz",
        "runtime_s",
        "status",
        "message",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(rows.values(), key=lambda item: float(item["alpha"]))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered_rows)


def _compute_alpha_point(preset_name, preset_cls, alpha_value, run_kwargs):
    start = time.time()
    simulator = preset_cls(**_source_kwargs(alpha_value, preset_cls))
    orbit, cloud = simulator.solve_coupled_system(
        duration_yr=run_kwargs["duration_yr"],
        secular_samples=run_kwargs["secular_samples"],
    )
    sample_points = _target_time_samples(simulator, float(orbit["t"][-1] - orbit["t"][0]))
    waveform_window = simulator.build_waveform_window_for_source_interval(
        orbit,
        cloud,
        0.0,
        float(orbit["t"][-1]),
        sample_points=sample_points,
    )
    snr_info = _axion_snr_from_window(
        simulator,
        waveform_window,
        pad_factor=int(run_kwargs["spectrum_pad_factor"]),
    )
    transition_freq_obs_mhz = simulator.transition_omega / (2.0 * np.pi * (1.0 + simulator.z)) * 1.0e3
    peak_overlap_abs = float(np.nanmax(np.asarray(cloud.get("overlap_abs", [np.nan]), dtype=float)))
    max_h_axion = float(np.nanmax(np.abs(np.asarray(waveform_window["h_axion"], dtype=float))))
    return {
        "preset": preset_name,
        "alpha": f"{alpha_value:.12g}",
        "snr_model": SNR_MODEL_VERSION,
        "detector_curve_kind": _detector_curve_kind(),
        "scan_low_hz": f"{_scan_low_hz():.16e}",
        "detector_curve_extrapolated": str(_extrapolate_detector_curve()),
        "snr_ref": f"{snr_info['snr']:.16e}",
        "rho2_ref": f"{snr_info['rho2']:.16e}",
        "transition_freq_obs_mhz": f"{transition_freq_obs_mhz:.16e}",
        "peak_overlap_abs": f"{peak_overlap_abs:.16e}",
        "max_h_axion_ref": f"{max_h_axion:.16e}",
        "time_samples": str(sample_points),
        "nyquist_hz": f"{snr_info['nyquist_hz']:.16e}",
        "df_hz": f"{snr_info['df_hz']:.16e}",
        "runtime_s": f"{time.time() - start:.3f}",
        "status": "ok",
        "message": "",
    }


def _error_row(preset_name, alpha_value, exc):
    return {
        "preset": preset_name,
        "alpha": f"{float(alpha_value):.12g}",
        "snr_model": SNR_MODEL_VERSION,
        "detector_curve_kind": _detector_curve_kind(),
        "scan_low_hz": f"{_scan_low_hz():.16e}",
        "detector_curve_extrapolated": str(_extrapolate_detector_curve()),
        "snr_ref": "nan",
        "rho2_ref": "nan",
        "transition_freq_obs_mhz": "nan",
        "peak_overlap_abs": "nan",
        "max_h_axion_ref": "nan",
        "time_samples": "0",
        "nyquist_hz": "nan",
        "df_hz": "nan",
        "runtime_s": "0",
        "status": "error",
        "message": repr(exc),
    }


def _compute_alpha_point_task(task):
    preset_name, alpha_value, run_kwargs = task
    preset_cls = PRESETS[preset_name]["class"]
    try:
        return _compute_alpha_point(preset_name, preset_cls, float(alpha_value), run_kwargs)
    except Exception as exc:  # keep long scans resumable
        return _error_row(preset_name, alpha_value, exc)


def _scan_preset(preset_name, alphas, run_kwargs, resume=True):
    csv_path = OUTPUT_DIR / f"{preset_name}_decigo_axion_snr_alpha_ref.csv"
    rows = _existing_rows(csv_path) if resume else {}
    pending = []
    for idx, alpha_value in enumerate(alphas, start=1):
        key = round(float(alpha_value), 12)
        if key in rows and rows[key].get("status") == "ok":
            print(f"[{preset_name}] alpha {alpha_value:.5f} ({idx}/{len(alphas)}) already done")
            continue
        pending.append((idx, float(alpha_value)))
    if not pending:
        return csv_path

    workers = _worker_count(len(pending))
    print(f"[{preset_name}] launching {len(pending)} alpha jobs with {workers} worker(s)")
    if workers == 1:
        for done, (idx, alpha_value) in enumerate(pending, start=1):
            print(f"[{preset_name}] alpha {alpha_value:.5f} ({idx}/{len(alphas)})")
            row = _compute_alpha_point_task((preset_name, alpha_value, run_kwargs))
            rows[round(alpha_value, 12)] = row
            _write_rows(csv_path, rows)
            _print_alpha_result(preset_name, alpha_value, row, done, len(pending))
        return csv_path

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_compute_alpha_point_task, (preset_name, alpha_value, run_kwargs)): (idx, alpha_value)
            for idx, alpha_value in pending
        }
        for done, future in enumerate(as_completed(futures), start=1):
            idx, alpha_value = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = _error_row(preset_name, alpha_value, exc)
            rows[round(alpha_value, 12)] = row
            _write_rows(csv_path, rows)
            _print_alpha_result(preset_name, alpha_value, row, done, len(pending))
    return csv_path


def _print_alpha_result(preset_name, alpha_value, row, done, total_pending):
    status = row.get("status", "unknown")
    if status == "ok":
        print(
            f"[{preset_name}] alpha {alpha_value:.5f} done "
            f"({done}/{total_pending}); SNR_ref={float(row['snr_ref']):.4g}; runtime={row['runtime_s']} s"
        )
    else:
        print(
            f"[{preset_name}] alpha {alpha_value:.5f} failed "
            f"({done}/{total_pending}): {row.get('message', '')}"
        )


def _load_snr_ref(csv_path, alphas):
    rows = _existing_rows(csv_path)
    snr_ref = np.full_like(alphas, np.nan, dtype=float)
    for idx, alpha_value in enumerate(alphas):
        row = rows.get(round(float(alpha_value), 12))
        if row is None or row.get("status") != "ok":
            continue
        snr_ref[idx] = float(row["snr_ref"])
    return snr_ref


def _write_grid_outputs(preset_name, alphas, distances_kpc, snr_grid):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT_DIR / f"{preset_name}_decigo_axion_snr_alpha_dl_grid.npz"
    np.savez(
        npz_path,
        alpha=alphas,
        distance_kpc=distances_kpc,
        snr=snr_grid,
        reference_distance_kpc=REFERENCE_DISTANCE_KPC,
        snr_model=SNR_MODEL_VERSION,
        detector_curve_kind=_detector_curve_kind(),
        scan_low_hz=_scan_low_hz(),
        detector_curve_extrapolated=_extrapolate_detector_curve(),
    )
    csv_path = OUTPUT_DIR / f"{preset_name}_decigo_axion_snr_alpha_dl_grid.csv"
    header = "distance_kpc\\alpha," + ",".join(f"{alpha:.12g}" for alpha in alphas)
    table = np.column_stack((distances_kpc, snr_grid.T))
    np.savetxt(csv_path, table, delimiter=",", header=header, comments="")
    return npz_path, csv_path


def _plot_combined(presets, alphas, distances_kpc, grids):
    import _plot_backend  # noqa: F401
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    n_panels = len(presets)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.4 * n_cols, 4.8 * n_rows), squeeze=False)
    levels = np.linspace(0.0, SNR_COLORBAR_MAX, 81)
    cmap = LinearSegmentedColormap.from_list(
        "cool_decigo_snr",
        ["#f7fbff", "#dbeaf4", "#bdd8e9", "#8fbfd8", "#5a9ec6", "#2f79b7", "#2350a1"],
        N=256,
    )
    cmap.set_over(SNR_OVER_COLOR)
    color_norm = PowerNorm(gamma=0.58, vmin=0.0, vmax=SNR_COLORBAR_MAX)

    last_image = None
    for ax, preset_name in zip(axes.ravel(), presets):
        raw_grid = grids[preset_name].T
        grid = np.ma.masked_invalid(raw_grid)
        last_image = ax.contourf(
            alphas,
            distances_kpc,
            grid,
            levels=levels,
            cmap=cmap,
            norm=color_norm,
            extend="max",
        )
        finite_grid = raw_grid[np.isfinite(raw_grid)]
        if finite_grid.size:
            grid_min = float(np.min(finite_grid))
            grid_max = float(np.max(finite_grid))
            contour_levels = [
                level
                for level in (10.0, 30.0, 50.0, 100.0, 200.0)
                if grid_min < level < grid_max
            ]
        else:
            contour_levels = []
        if contour_levels:
            low_contours = [level for level in contour_levels if level < 100.0]
            high_contours = [level for level in contour_levels if level >= 100.0]
            if low_contours:
                contours = ax.contour(
                    alphas,
                    distances_kpc,
                    raw_grid,
                    levels=low_contours,
                    colors="#111111",
                    linewidths=0.85,
                )
                ax.clabel(contours, fmt="%g", fontsize=8, colors="#111111")
            if high_contours:
                contours = ax.contour(
                    alphas,
                    distances_kpc,
                    raw_grid,
                    levels=high_contours,
                    colors="#B00020",
                    linewidths=1.05,
                )
                ax.clabel(contours, fmt="%g", fontsize=8, colors="#B00020")
        ax.axhline(REFERENCE_DISTANCE_KPC, color="#62d26f", lw=1.0, alpha=0.9)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$d_L$ [kpc]")
        ax.set_title(f"{preset_name}: {PRESETS[preset_name]['label']}")
        ax.grid(True, which="both", alpha=0.2)

    for ax in axes.ravel()[len(presets):]:
        ax.set_axis_off()
    if last_image is not None:
        colorbar = fig.colorbar(
            last_image,
            ax=axes.ravel().tolist(),
            shrink=0.92,
            pad=0.015,
            ticks=[0, 25, 50, 100, 150, 200],
        )
        colorbar.set_label(f"{DETECTOR} axion SNR")
    #fig.suptitle("Exact low-frequency axion strain SNR scan; distance is amplitude-rescaled", y=0.995)
    out_pdf = FIGURE_DIR / "lowfre_decigo_axion_snr_alpha_dl_scan.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_pdf}")


def main():
    presets = _preset_list()
    alphas = _alpha_grid()
    distances_kpc = _distance_grid_kpc()
    run_kwargs = _run_kwargs()
    resume = _bool_env("LOWFREQ_SCAN_RESUME", True)

    print(
        "DECIGO axion-only SNR scan: "
        f"presets={','.join(presets)}, alpha=[{alphas[0]:.4g},{alphas[-1]:.4g}] x {len(alphas)}, "
        f"dL=[{distances_kpc[0]:.4g},{distances_kpc[-1]:.4g}] kpc x {len(distances_kpc)}, "
        f"duration={run_kwargs['duration_yr']:.3f} yr, reference dL={REFERENCE_DISTANCE_KPC:.1f} kpc, "
        f"detector_curve_kind={_detector_curve_kind()}, scan_low={_scan_low_hz():.3g} Hz, "
        f"snr_model={SNR_MODEL_VERSION}"
    )
    grids = {}
    for preset_name in presets:
        csv_path = _scan_preset(preset_name, alphas, run_kwargs, resume=resume)
        snr_ref = _load_snr_ref(csv_path, alphas)
        snr_grid = snr_ref[:, None] * (REFERENCE_DISTANCE_KPC / distances_kpc[None, :])
        grids[preset_name] = snr_grid
        npz_path, grid_csv = _write_grid_outputs(preset_name, alphas, distances_kpc, snr_grid)
        print(f"[{preset_name}] saved grid: {npz_path}, {grid_csv}")
    _plot_combined(presets, alphas, distances_kpc, grids)


if __name__ == "__main__":
    freeze_support()
    main()
