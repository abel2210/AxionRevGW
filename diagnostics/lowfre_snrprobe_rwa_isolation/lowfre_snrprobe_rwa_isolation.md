# Low-frequency SNR-probe RWA isolation audit

This check reruns the selected-RWA and nearest-three-harmonic cloud drive
at the actual SNR-selected probe points used in the Letter mismatch panel.
The orbital backreaction is kept in the selected-harmonic convention in
both runs, so the comparison measures neighboring-harmonic leakage in the
cloud amplitudes.

| preset | alpha | selected crossed | nearest neighbor n | min |Delta_n|/|eta_n| | min |Delta_n|/(8|eta_n|) | nonselected crossings | max dC/scale | max da/a |
|---|---:|:---:|---:|---:|---:|---|---:|---:|
| 211 | 0.289899 | yes | 3 | 229280 | 28660 | none | 0.00400978 | 1.43821e-06 |
| 211v | 0.295960 | no | 3 | 213318 | 26664.8 | none | 0.045247 | 2.99018e-07 |
| 322 | 0.265657 | yes | 3 | 64249.7 | 8031.21 | none | 0.00820299 | 8.36862e-06 |
| 322v | 0.293939 | no | 3 | 52486 | 6560.76 | none | 1.76793 | 8.57566e-08 |

Interpretation:

- `min |Delta_n|/|eta_n|` is evaluated at the selected n=4 crossing.
- The gate-width ratio divides the same detuning by `8|eta_n|`, matching the low-frequency band gate width factor.
- Values much larger than one mean neighboring harmonics are outside the local selected-harmonic resonance width at the crossing.
- Downward rows remain accumulated-waveform diagnostics in the Letter, not direct selected-transition SNR detections.

CSV: `diagnostics/lowfre_snrprobe_rwa_isolation/lowfre_snrprobe_rwa_isolation.csv`
