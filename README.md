# AxionRevGW

Code, numerical data, and figure assets supporting the paper

**Finite Coherence in Gravitational Waves from Tidally Excited Axion Clouds**

This repository contains the production scripts and derived data used to generate the figures and numerical checks reported in the article and its Appendix. It is intended as the public technical companion to the paper.

## Repository Contents

```text
.
|-- highfre_shared.py, lowfre_shared.py
|-- highfre*.py, lowfre*.py
|-- adiabaticlimit.py, bohr_lz_tools.py, bohr_tidal.py
|-- probe_bohr_*.py, validate_bohr_projection.py
|-- plot_*.py
|-- powerspectrum.py, powerspectrumv.py
|-- transition_geometry.py
|-- figures/
|-- frequency_data/
|-- waveform_data/
|-- snr_scan_data/
|-- diagnostics/
`-- benchmark_highfreq_q001/
```

### Core Modules

- `highfre_shared.py`: high-frequency eccentric binary and Bohr-transition waveform model.
- `lowfre_shared.py`: low-frequency fine and hyperfine transition model, including waveform, SNR, and mismatch utilities.
- `highfre322.py`, `highfre322v.py`, `highfre644.py`, `highfre644v.py`: high-frequency transition presets.
- `lowfre211.py`, `lowfre211v.py`, `lowfre322.py`, `lowfre322v.py`: low-frequency transition presets.
- `transition_geometry.py`: angular and radial transition-geometry factors; `list_transition_geometries.py` and `transition_geometry_factors.csv` tabulate them.
- `adiabaticlimit.py`: adiabatic two-level phase diagram, including the finite-separation Bohr coupling used by the full-model figures.
- `bohr_tidal.py`: finite-separation Newtonian potential projected onto an equal-\((\ell,m)\) Bohr pair; supplies the dimensionless coupling kernels.
- `bohr_lz_tools.py`: Landau--Zener crossing probability and outgoing-coherence utilities.
- `pure_peters_template.py`: vacuum eccentric-binary template generation.
- `remnant_rate_models.py`: effective-rate utilities for stochastic-background extensions.
- `_plot_backend.py`: non-interactive matplotlib backend selection.

### Figure and Scan Scripts

- `plot_bohr_orbit_time_summary_concise.py`: compact Bohr event time-domain summary.
- `probe_bohr_alpha_family.py`, `probe_bohr_visibility_sweep.py`: numerical support for the finite-coherence visibility figure.
- `plot_bohr_domain_visibility_map.py`: parameter-domain and waveform-normalization map.
- `probe_bohr_domain_scaling_validation.py`: scaling validation of the compact domain estimate over the declared mass-ratio range.
- `plot_bohr_full_model.py`: full-model Bohr figures (orbit-time summary, domain map, coherence map) built from the finite-separation projection; heavier than the compact probes.
- `probe_bohr_monopole_feedback.py`: two-state passage with the full allowed multipole set and continuous orbital feedback.
- `probe_bohr_nearby_levels.py`: finite bound-state screen around the selected passage.
- `validate_bohr_projection.py`: independent quadrature and cross-script checks for the selected Bohr pair.
- `compact_bohr_domain_check.py`: compact-overlap check of the displayed domain boundary.
- `plot_lowfre_resolved_diagnostics.py`: downward-transition SNR and fixed-parameter mismatch diagnostic figure.
- `plot_sgwb_remnant_rate_band.py`: rate-normalized stochastic-background spectra.
- `lowfre_decigo_snr_scan.py`: DECIGO SNR scans for low-frequency transitions.
- `probe_lowfre_mismatch_from_snr_scan.py`: mismatch probe based on selected SNR scan points.
- `lowfreq_rwa_convergence.py`: selected-harmonic RWA convergence check.
- `run_highfreq_q001_benchmarks.py`, `probe_bohr_q001_boundary.py`: mass-scaling benchmarks at fixed `q=0.01`.

### Data Directories

- `figures/`: production figure PDFs.
- `frequency_data/`: frequency-domain strain and stochastic-background spectra, including detector sensitivity curves and transition-radiation spectra for the 80/160/320-orbit spectrum windows.
- `waveform_data/`: time-domain and windowed frequency-domain waveform samples.
- `snr_scan_data/`: low-frequency DECIGO SNR grids and reference slices.
- `diagnostics/`: numerical CSV/TXT/JSON data needed by plotting scripts, with internal reports removed.
- `benchmark_highfreq_q001/`: high-frequency `q=0.01` benchmark outputs for three primary masses, summarized in `summary.md`.

Detector sensitivity inputs are stored as:

- `CE.csv`
- `DECIGO.csv`
- `ET.csv`
- `lisa.csv`

## Requirements

The scripts are plain Python and were run with Python 3.13 during the final production pass. A Python 3.10+ environment should be sufficient.

Install the required packages with:

```bash
pip install -r requirements.txt
```

Main dependencies:

- `numpy`
- `scipy` (the Bohr projection modules use `scipy.special.sph_harm_y`, which requires SciPy >= 1.15)
- `matplotlib`

Version-level changes are recorded in `CHANGELOG.md`.

## Reproducing Figures

Run scripts from the repository root.

```bash
python plot_bohr_orbit_time_summary_concise.py
python probe_bohr_alpha_family.py
python probe_bohr_visibility_sweep.py
python plot_bohr_domain_visibility_map.py
python probe_bohr_domain_scaling_validation.py
python plot_lowfre_resolved_diagnostics.py
python plot_sgwb_remnant_rate_band.py
```

The finite-coherence figure is assembled from the outputs of `probe_bohr_alpha_family.py` and `probe_bohr_visibility_sweep.py`. Generated figures are written to `figures/`.

The full-model Bohr figures are produced by:

```bash
python plot_bohr_full_model.py
```

This script also writes the `diagnostics/bohr_full_*` data products. The independent projection and screening checks are reproduced with:

```bash
python validate_bohr_projection.py
python probe_bohr_nearby_levels.py
python compact_bohr_domain_check.py
```

Some scripts can be computationally heavier because they integrate coupled orbital and cloud evolution. The supplied data directories contain the production outputs used to make the paper figures.

## Regenerating Scan Data

Low-frequency SNR scans for the two downward transitions initialized in populated superradiant levels:

```bash
python lowfre_decigo_snr_scan.py
```

Mismatch probe from selected SNR scan points:

```bash
python probe_lowfre_mismatch_from_snr_scan.py
```

The default probe uses the common source point \(\alpha=0.30\), \(d_L=100\) kpc for
`211v` and `322v`. Its CSV output records the resonance time, transition SNR,
and cumulative fixed-parameter mismatch used in the resolved-source figure.

High-frequency fixed-mass-ratio benchmarks:

```bash
python run_highfreq_q001_benchmarks.py
```

## Scope

This repository contains code, numerical data, and figures only. Manuscript drafts, cover letters, presentation slides, internal notes, technical audit reports, and review-response documents are outside the public code package.

The compact Bohr calculation is an event-local selected-crossing model. The
parameter-domain figure is a fixed-eccentricity estimate validated over
\(10^{-4}\leq q\leq10^{-2}\), rather than a population scan. For long
observations, the resolved-source comparison retains only downward transitions
that begin in populated superradiant levels; upward examples initialized in
absorptive levels require an explicit replenishment history. The reported
mismatch holds intrinsic source parameters fixed and is a waveform-deformation
diagnostic, not a parameter-optimized detection forecast.

The full-model Bohr figures use the finite-separation projection with continuous
orbital feedback for the fiducial waveform, while the accompanying parameter
maps and the Landau--Zener comparison are vacuum-sweep, two-state references.
The nearby-level screen is a finite bound-state calculation, not a continuum
treatment.

The stochastic-background scripts are retained as an optional rate-normalized
extension. Their effective event rate is an external population input rather
than a prediction of the waveform model.
