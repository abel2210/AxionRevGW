# Independent transition-geometry check

This check recomputes the radial overlap with `scipy.integrate.quad` and the angular quadrupole matrix with Gauss-Legendre integration in `mu=cos(theta)` plus a uniform periodic phi rule.  It does not call `compute_transition_geometry`.

## Source-angle averages

- axisymmetric Delta m=0: `0.5333333333332639` vs `8/15`
- quadrupolar |Delta m|=2: `0.8000000000001564` vs `4/5`

`raw_trace_abs` is reported only as a diagnostic of the non-STF matrix `I_ij=int psi_g^* psi_e r_i r_j`.  It is not expected to vanish for all transitions, and the trace part is removed by the TT projection.  The waveform check is the projected factor `F`.

| transition | pattern | radial rel diff | F rel diff | angle abs diff | raw trace abs |
|---|---|---:|---:|---:|---:|
| highfre322 upward | quadrupolar_delta_m2 | 4.957e-06 | 4.957e-06 | 0.000e+00 | 3.012e-15 |
| highfre322 downward | quadrupolar_delta_m2 | 4.957e-06 | 4.957e-06 | 0.000e+00 | 3.012e-15 |
| highfre211 upward | quadrupolar_delta_m2 | 4.957e-06 | 4.957e-06 | 0.000e+00 | 1.657e-15 |
| highfre211 downward | quadrupolar_delta_m2 | 4.957e-06 | 4.957e-06 | 0.000e+00 | 1.657e-15 |
| highfre644 upward | axisymmetric_delta_m0 | 4.957e-06 | 4.957e-06 | 0.000e+00 | 4.923e+02 |
| highfre644 downward | axisymmetric_delta_m0 | 4.957e-06 | 4.957e-06 | 0.000e+00 | 4.923e+02 |
