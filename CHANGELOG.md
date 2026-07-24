# Changelog

## 2026-07-24

- Updated the finite-boundary Landau--Zener treatment and regenerated the Bohr event and coherence figures.
- Restricted the displayed Bohr domain estimate to the numerically checked range \(10^{-4}\leq q\leq10^{-2}\) and updated the reference resonance eccentricity to \(e_{\rm res}=0.6373\).
- Recomputed the resolved-source analysis for the downward \(|211\rangle\to|21{-}1\rangle\) and \(|322\rangle\to|300\rangle\) transitions, including detector-band and Nyquist checks.
- Removed the obsolete upward-transition mismatch products whose absorptive initial levels do not survive the adopted observation window without replenishment.
- Added resonance times to the compact probe output so the revised SNR--mismatch figure can be reproduced without distributing the full probe waveforms.
- Kept manuscript files, internal technical reports, and submission materials outside the public repository.

## 2026-06-17

- Synced the public production scripts and generated data products with the current `finalcode` workspace.
- Removed the obsolete `plot_bohr_visibility_prl.py` entry point; the finite-coherence figure is now supported by `probe_bohr_alpha_family.py` and `probe_bohr_visibility_sweep.py`.
- Refreshed public figure and frequency-domain data products while leaving manuscript files, internal audit notes, and private diagnostic reports outside the repository.
- Updated `README.md` to match the current public reproduction workflow.

## 2026-06-14

- Updated the public code package to match the finite-coherence interpretation used in the current manuscript.
- Replaced the old quench/revival evidence script with the outgoing-coherence visibility pipeline:
  - `bohr_lz_tools.py`
  - `probe_bohr_alpha_family.py`
  - `probe_bohr_visibility_sweep.py`
  - `plot_bohr_visibility_prl.py`
- Added the corresponding CSV diagnostics and figure outputs under `diagnostics/` and `figures/`.
- Removed stale manuscript-folder output paths from public plotting scripts; public scripts now write only inside repository data and figure directories.
- Removed stale local absolute paths from generated diagnostics.
