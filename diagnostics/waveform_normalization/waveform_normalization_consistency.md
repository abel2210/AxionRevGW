# Waveform normalization consistency audit

Checks: 34
Failures: 0

## Scope

- Physical constants in high-frequency and low-frequency shared solvers.
- Cloud radius and mass normalization: r_g=GM/c^2, r_c=r_g/alpha^2, M_c=f_c M.
- Transition strain amplitude A0=4G M_c r_c^2 omega_trans^2 F/(c^4 d_L).
- Real waveform convention C(t)=Re[c_g^* c_e_tilde]cos(omega t)-Im[c_g^* c_e_tilde]sin(omega t).
- Quadrupole geometry factors for Delta m=0 and |Delta m|=2.
- Backreaction reservoir scale N_axion hbar=G M_1 M_c/(alpha c).

## Results

All waveform normalization checks passed.

CSV details: `diagnostics/waveform_normalization/waveform_normalization_consistency.csv`
