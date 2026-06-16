# Changelog

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
