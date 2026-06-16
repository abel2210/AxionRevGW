import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lowfre_decigo_snr_scan import (
    DETECTOR,
    PRESETS,
    REFERENCE_DISTANCE_KPC,
    SNR_MODEL_VERSION,
)
from lowfre_shared import build_paper_inspired_lowfreq_profile


SCRIPT_DIR = Path(__file__).resolve().parent
SNR_SCAN_DIR = SCRIPT_DIR / "snr_scan_data"
DIAGNOSTIC_DIR = SCRIPT_DIR / "diagnostics"
MISMATCH_PROBE_DIR = DIAGNOSTIC_DIR / "lowfre_mismatch_threshold_probe"
REPORT_MD = MISMATCH_PROBE_DIR / "lowfre_mismatch_threshold_probe.md"
REPORT_CSV = MISMATCH_PROBE_DIR / "lowfre_mismatch_threshold_probe.csv"


@dataclass(frozen=True)
class SelectedPoint:
    preset: str
    alpha: float
    distance_kpc: float
    scan_snr: float
    scan_threshold: float
    transition_freq_obs_mhz: float
    selection: str


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Pick low-frequency DECIGO SNR-scan points that just pass a requested "
            "SNR floor, then rerun the full lowfre pipeline and compare mismatch "
            "M with N/(2 rho^2)."
        )
    )
    parser.add_argument(
        "--presets",
        default="211,211v,322,322v",
        help="Comma-separated presets to probe. Default: all four low-frequency transitions.",
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=8.0,
        help=(
            "SNR floor used to select scan points. Default 8.0. "
            "For the bare necessary mismatch condition with N=13, use about 2.55."
        ),
    )
    parser.add_argument(
        "--min-line-freq-mhz",
        type=float,
        default=2.0911652313,
        help=(
            "Require the transition line to be above this observed frequency before "
            "selecting an SNR-scan point. Default is the native low-frequency edge "
            "of the DECIGO table used by the mismatch pipeline, in mHz. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--selection",
        choices=("closest-above", "max-distance", "min-alpha"),
        default="closest-above",
        help=(
            "How to choose among scan points with SNR >= snr-min. "
            "closest-above selects the smallest SNR excess over the threshold."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report selected scan points; do not run the full lowfre pipeline.",
    )
    parser.add_argument(
        "--mismatch-d",
        type=float,
        default=13.0,
        help="Effective parameter-space dimension N used in N/(2 rho^2). Default: 13.",
    )
    parser.add_argument(
        "--duration-yr",
        type=float,
        default=None,
        help="Override the lowfre run duration. Default: profile value.",
    )
    parser.add_argument(
        "--mismatch-time-samples",
        type=int,
        default=None,
        help="Override mismatch time samples. Default: profile value.",
    )
    return parser.parse_args()


def _preset_names(raw):
    names = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [item for item in names if item not in PRESETS]
    if unknown:
        raise ValueError("Unknown preset(s): " + ", ".join(unknown))
    return names


def _load_transition_freq_obs_mhz(preset, alpha):
    csv_path = SNR_SCAN_DIR / f"{preset}_decigo_axion_snr_alpha_ref.csv"
    if not csv_path.exists():
        return np.nan
    best = None
    with csv_path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                row_alpha = float(row["alpha"])
            except (KeyError, ValueError):
                continue
            if best is None or abs(row_alpha - alpha) < abs(best[0] - alpha):
                best = (row_alpha, row)
    if best is None:
        return np.nan
    try:
        return float(best[1]["transition_freq_obs_mhz"])
    except (KeyError, ValueError):
        return np.nan


def _load_transition_freq_by_alpha(preset, alphas):
    csv_path = SNR_SCAN_DIR / f"{preset}_decigo_axion_snr_alpha_ref.csv"
    if not csv_path.exists():
        return np.full_like(alphas, np.nan, dtype=float)

    rows = []
    with csv_path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append((float(row["alpha"]), float(row["transition_freq_obs_mhz"])))
            except (KeyError, ValueError):
                continue
    if not rows:
        return np.full_like(alphas, np.nan, dtype=float)

    row_alpha = np.array([item[0] for item in rows], dtype=float)
    row_freq = np.array([item[1] for item in rows], dtype=float)
    output = np.full_like(alphas, np.nan, dtype=float)
    for idx, alpha in enumerate(alphas):
        nearest = int(np.argmin(np.abs(row_alpha - alpha)))
        if abs(row_alpha[nearest] - alpha) <= 1.0e-9:
            output[idx] = row_freq[nearest]
    return output


def _select_scan_point(preset, snr_min, min_line_freq_mhz, selection):
    npz_path = SNR_SCAN_DIR / f"{preset}_decigo_axion_snr_alpha_dl_grid.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing SNR grid: {npz_path}")

    data = np.load(npz_path)
    model = str(data.get("snr_model", ""))
    if model and model != SNR_MODEL_VERSION:
        print(f"[{preset}] warning: grid model {model!r} != current {SNR_MODEL_VERSION!r}")
    alphas = np.asarray(data["alpha"], dtype=float)
    distances = np.asarray(data["distance_kpc"], dtype=float)
    snr = np.asarray(data["snr"], dtype=float)
    line_freq_mhz = _load_transition_freq_by_alpha(preset, alphas)

    valid_alpha = np.isfinite(line_freq_mhz) & (line_freq_mhz >= float(min_line_freq_mhz))
    valid = np.isfinite(snr) & (snr >= float(snr_min)) & valid_alpha[:, None]
    if not np.any(valid):
        max_snr = float(np.nanmax(snr)) if np.any(np.isfinite(snr)) else np.nan
        max_snr_in_band = (
            float(np.nanmax(snr[valid_alpha, :]))
            if np.any(valid_alpha) and np.any(np.isfinite(snr[valid_alpha, :]))
            else np.nan
        )
        raise RuntimeError(
            f"{preset}: no scan point reaches SNR >= {snr_min:g} with "
            f"line frequency >= {min_line_freq_mhz:g} mHz; "
            f"max SNR = {max_snr:.6g}, max in-band SNR = {max_snr_in_band:.6g}"
        )

    alpha_idx, distance_idx = np.where(valid)
    candidate_snr = snr[alpha_idx, distance_idx]
    candidate_alpha = alphas[alpha_idx]
    candidate_distance = distances[distance_idx]

    if selection == "closest-above":
        order_key = np.lexsort((candidate_distance, candidate_alpha, candidate_snr - snr_min))
        chosen = int(order_key[0])
    elif selection == "max-distance":
        order_key = np.lexsort((candidate_snr, candidate_alpha, -candidate_distance))
        chosen = int(order_key[0])
    elif selection == "min-alpha":
        order_key = np.lexsort((candidate_snr, -candidate_distance, candidate_alpha))
        chosen = int(order_key[0])
    else:
        raise ValueError(f"Unhandled selection rule: {selection}")

    alpha = float(candidate_alpha[chosen])
    return SelectedPoint(
        preset=preset,
        alpha=alpha,
        distance_kpc=float(candidate_distance[chosen]),
        scan_snr=float(candidate_snr[chosen]),
        scan_threshold=float(snr_min),
        transition_freq_obs_mhz=float(line_freq_mhz[alpha_idx[chosen]]),
        selection=selection,
    )


def _build_simulator(point, mismatch_d):
    preset_cls = PRESETS[point.preset]["class"]
    profile = build_paper_inspired_lowfreq_profile()
    source_kwargs = dict(profile.source_kwargs)
    source_kwargs.update(
        {
            "alpha": point.alpha,
            "distance_Mpc": point.distance_kpc / 1000.0,
            "detector_names": (DETECTOR,),
            "detector_curve_kinds": {DETECTOR: "characteristic_strain"},
            "mismatch_threshold_d": float(mismatch_d),
            "module_stem": f"{preset_cls.__module__}_snrprobe",
            "save_figure_dir": "figures/snr_threshold_probe",
            "save_time_series_data_dir": "diagnostics/lowfre_mismatch_threshold_probe/waveform_data",
            "save_mismatch_data_dir": "diagnostics/lowfre_mismatch_threshold_probe/mismatch_data",
        }
    )
    return preset_cls(**source_kwargs), profile


def _final_series_metrics(series):
    if not series:
        return {}

    obs_years = np.asarray(series.get("observation_years", []), dtype=float)
    mismatch = np.asarray(series.get("mismatch", []), dtype=float)
    rho = np.asarray(series.get("snr", []), dtype=float)
    threshold = np.asarray(series.get("distinguishability_threshold", []), dtype=float)
    raw_mismatch = np.asarray(series.get("raw_mismatch", []), dtype=float)
    finite = np.isfinite(obs_years) & np.isfinite(mismatch) & np.isfinite(rho) & np.isfinite(threshold)
    if not np.any(finite):
        return {}

    idx = np.nonzero(finite)[0][-1]
    raw_value = float(raw_mismatch[idx]) if raw_mismatch.shape == mismatch.shape and np.isfinite(raw_mismatch[idx]) else np.nan
    return {
        "observation_years": float(obs_years[idx]),
        "mismatch": float(mismatch[idx]),
        "raw_mismatch": raw_value,
        "rho_mismatch_band": float(rho[idx]),
        "threshold": float(threshold[idx]),
        "passes": bool(mismatch[idx] > threshold[idx]),
    }


def _instant_match_metrics(match):
    if not match:
        return {}
    return {
        "mismatch": float(match.get("mismatch", np.nan)),
        "rho_mismatch_band": float(match.get("snr", np.nan)),
        "threshold": float(match.get("distinguishability_threshold", np.nan)),
        "passes": bool(float(match.get("mismatch", np.nan)) > float(match.get("distinguishability_threshold", np.inf))),
        "band_low_mhz": float(match.get("analysis_band_hz", (np.nan, np.nan))[0]) * 1.0e3,
        "band_high_mhz": float(match.get("analysis_band_hz", (np.nan, np.nan))[1]) * 1.0e3,
    }


def _run_full_probe(point, args):
    simulator, profile = _build_simulator(point, args.mismatch_d)
    run_kwargs = dict(profile.run_kwargs)
    if args.duration_yr is not None:
        run_kwargs["duration_yr"] = float(args.duration_yr)
        run_kwargs["mismatch_max_years"] = min(float(args.duration_yr), float(run_kwargs["mismatch_max_years"]))
    if args.mismatch_time_samples is not None:
        run_kwargs["mismatch_time_samples"] = int(args.mismatch_time_samples)

    print(
        f"[{point.preset}] full run: alpha={point.alpha:.12g}, "
        f"dL={point.distance_kpc:.6g} kpc, scan SNR={point.scan_snr:.6g}"
    )
    start = time.time()
    results = simulator.run(**run_kwargs)
    elapsed = time.time() - start

    final_series = _final_series_metrics(results.get("mismatch_time_series", {}).get(DETECTOR, {}))
    instant_match = _instant_match_metrics(results.get("detector_match", {}).get(DETECTOR, {}))
    selected_match = _instant_match_metrics(results.get("selected_detector_match", {}).get(DETECTOR, {}))
    return {
        "elapsed_s": elapsed,
        "run_kwargs": run_kwargs,
        "final_series": final_series,
        "instant_match": instant_match,
        "selected_match": selected_match,
        "mismatch_data_path": str(results.get("mismatch_data_paths", {}).get(DETECTOR, "")),
    }


def _write_reports(points, run_results, min_line_freq_mhz):
    MISMATCH_PROBE_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "preset",
        "selection",
        "alpha",
        "distance_kpc",
        "transition_freq_obs_mhz",
        "scan_snr",
        "scan_threshold",
        "final_observation_years",
        "final_mismatch",
        "final_raw_mismatch",
        "final_rho_mismatch_band",
        "final_threshold_N_over_2rho2",
        "final_passes",
        "instant_mismatch",
        "instant_rho_mismatch_band",
        "instant_threshold_N_over_2rho2",
        "instant_passes",
        "selected_mismatch",
        "selected_rho_mismatch_band",
        "selected_threshold_N_over_2rho2",
        "selected_passes",
        "mismatch_data_path",
        "elapsed_s",
    ]
    with REPORT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            result = run_results.get(point.preset, {})
            final_series = result.get("final_series", {})
            instant = result.get("instant_match", {})
            selected = result.get("selected_match", {})
            writer.writerow(
                {
                    "preset": point.preset,
                    "selection": point.selection,
                    "alpha": f"{point.alpha:.12g}",
                    "distance_kpc": f"{point.distance_kpc:.16e}",
                    "transition_freq_obs_mhz": f"{point.transition_freq_obs_mhz:.16e}",
                    "scan_snr": f"{point.scan_snr:.16e}",
                    "scan_threshold": f"{point.scan_threshold:.16e}",
                    "final_observation_years": final_series.get("observation_years", ""),
                    "final_mismatch": final_series.get("mismatch", ""),
                    "final_raw_mismatch": final_series.get("raw_mismatch", ""),
                    "final_rho_mismatch_band": final_series.get("rho_mismatch_band", ""),
                    "final_threshold_N_over_2rho2": final_series.get("threshold", ""),
                    "final_passes": final_series.get("passes", ""),
                    "instant_mismatch": instant.get("mismatch", ""),
                    "instant_rho_mismatch_band": instant.get("rho_mismatch_band", ""),
                    "instant_threshold_N_over_2rho2": instant.get("threshold", ""),
                    "instant_passes": instant.get("passes", ""),
                    "selected_mismatch": selected.get("mismatch", ""),
                    "selected_rho_mismatch_band": selected.get("rho_mismatch_band", ""),
                    "selected_threshold_N_over_2rho2": selected.get("threshold", ""),
                    "selected_passes": selected.get("passes", ""),
                    "mismatch_data_path": result.get("mismatch_data_path", ""),
                    "elapsed_s": f"{result.get('elapsed_s', np.nan):.3f}" if result else "",
                }
            )

    lines = [
        "# Low-frequency mismatch threshold probe",
        "",
        f"- SNR scan model: `{SNR_MODEL_VERSION}`",
        f"- Selection rule: `{points[0].selection if points else 'n/a'}`",
        f"- Scan SNR floor: `{points[0].scan_threshold if points else np.nan:g}`",
        f"- Transition-line floor: `>= {min_line_freq_mhz:g} mHz`.",
        "- The scan SNR is the extrapolated transition-strain SNR from `lowfre_decigo_snr_scan.py`.",
        "- The mismatch comparison below is recomputed by the full `lowfre_shared` pipeline at the selected alpha and distance.",
        "",
        "| preset | alpha | dL [kpc] | line f [mHz] | scan SNR | final M | final rho_a | N/(2rho_a^2) | pass? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for point in points:
        final_series = run_results.get(point.preset, {}).get("final_series", {})
        lines.append(
            "| {preset} | {alpha:.6f} | {distance:.6g} | {freq:.6g} | {scan:.6g} | "
            "{mismatch} | {rho} | {threshold} | {passes} |".format(
                preset=point.preset,
                alpha=point.alpha,
                distance=point.distance_kpc,
                freq=point.transition_freq_obs_mhz,
                scan=point.scan_snr,
                mismatch=_fmt(final_series.get("mismatch")),
                rho=_fmt(final_series.get("rho_mismatch_band")),
                threshold=_fmt(final_series.get("threshold")),
                passes="yes" if final_series.get("passes") else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `final M` is the last point of the saved mismatch time series.",
            "- `final rho_a` is the axion-only perturbation SNR used internally by the conservative mismatch threshold calculation.",
            "- `N/(2rho_a^2)` uses the configured `mismatch_threshold_d` value.",
            f"- CSV report: `{REPORT_CSV}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {REPORT_MD}")
    print(f"Saved report: {REPORT_CSV}")


def _fmt(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.6g}"


def main():
    args = _parse_args()
    names = _preset_names(args.presets)
    points = []
    for preset in names:
        point = _select_scan_point(preset, args.snr_min, args.min_line_freq_mhz, args.selection)
        points.append(point)
        print(
            f"[{preset}] selected alpha={point.alpha:.12g}, dL={point.distance_kpc:.6g} kpc, "
            f"line={point.transition_freq_obs_mhz:.6g} mHz, scan SNR={point.scan_snr:.6g}"
        )

    run_results = {}
    if not args.dry_run:
        for point in points:
            run_results[point.preset] = _run_full_probe(point, args)
    _write_reports(points, run_results, args.min_line_freq_mhz)


if __name__ == "__main__":
    main()
