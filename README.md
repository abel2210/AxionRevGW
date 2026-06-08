# AxionRevGW

Code, numerical data, and figure assets supporting the paper

**Revival of Gravitational Waves from Tidally Excited Axion Clouds**

This repository contains the production scripts and derived data used to generate the figures and numerical checks reported in the manuscript and Supplemental Material. It is intended as a public technical companion to the paper under the APS data and code availability policy.

## Repository Contents

```text
.
|-- highfre_shared.py, lowfre_shared.py
|-- highfre*.py, lowfre*.py
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
- `transition_geometry.py`: angular and radial transition-geometry factors.
- `pure_peters_template.py`: vacuum eccentric-binary template generation.
- `remnant_rate_models.py`: effective-rate utilities for stochastic-background extensions.

### Figure and Scan Scripts

- `plot_bohr_orbit_time_summary_concise.py`: compact Bohr event time-domain summary.
- `plot_bohr_quench_revival_evidence.py`: Landau-Zener coherence evidence figure.
- `plot_bohr_domain_visibility_map.py`: parameter-domain and waveform-normalization map.
- `plot_lowfre_resolved_diagnostics.py`: low-frequency SNR and mismatch diagnostic figure.
- `plot_sgwb_remnant_rate_band.py`: rate-normalized stochastic-background spectra.
- `lowfre_decigo_snr_scan.py`: DECIGO SNR scans for low-frequency transitions.
- `probe_lowfre_mismatch_from_snr_scan.py`: mismatch probe based on selected SNR scan points.
- `lowfreq_rwa_convergence.py`: selected-harmonic RWA convergence check.
- `run_highfreq_q001_benchmarks.py`: mass-scaling benchmark at fixed `q=0.01`.

### Data Directories

- `figures/`: production figure PDFs.
- `frequency_data/`: frequency-domain strain and stochastic-background spectra, including detector sensitivity curves.
- `waveform_data/`: time-domain and windowed frequency-domain waveform samples.
- `snr_scan_data/`: low-frequency DECIGO SNR grids and reference slices.
- `diagnostics/`: numerical CSV/TXT data needed by plotting scripts, with internal reports removed.
- `benchmark_highfreq_q001/`: high-frequency `q=0.01` benchmark outputs for three primary masses.

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
- `scipy`
- `matplotlib`

## Reproducing Figures

Run scripts from the repository root.

```bash
python plot_bohr_orbit_time_summary_concise.py
python plot_bohr_quench_revival_evidence.py
python plot_bohr_domain_visibility_map.py
python plot_lowfre_resolved_diagnostics.py
python plot_sgwb_remnant_rate_band.py
```

The generated figures are written to `figures/`.

Some scripts can be computationally heavier because they integrate coupled orbital and cloud evolution. The supplied data directories contain the production outputs used to make the paper figures.

## Regenerating Scan Data

Low-frequency SNR scans:

```bash
python lowfre_decigo_snr_scan.py
```

Mismatch probe from selected SNR scan points:

```bash
python probe_lowfre_mismatch_from_snr_scan.py
```

High-frequency fixed-mass-ratio benchmarks:

```bash
python run_highfreq_q001_benchmarks.py
```

## Scope

This repository contains code, numerical data, and figures only. Manuscript drafts, cover letters, presentation slides, internal notes, and review-response documents have been removed from the public code package.
