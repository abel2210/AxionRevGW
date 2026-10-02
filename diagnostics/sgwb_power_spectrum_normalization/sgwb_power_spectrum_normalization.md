# SGWB power-spectrum normalization audit

Frequency-amplitude exports checked: 13
Omega_GW exports checked: 6
Total issues: 0

## Frequency-amplitude exports

- highfre322_axion_backreaction_total_frequency_amplitude.txt: PASS
- highfre322_axion_frequency_amplitude.txt: PASS
- highfre322_pure_binary_template_frequency_amplitude.txt: PASS
- highfre322v_axion_backreaction_total_frequency_amplitude.txt: PASS
- highfre322v_axion_frequency_amplitude.txt: PASS
- highfre322v_pure_binary_template_frequency_amplitude.txt: PASS
- highfre644_axion_backreaction_total_frequency_amplitude.txt: PASS
- highfre644_axion_frequency_amplitude.txt: PASS
- highfre644_pure_binary_template_frequency_amplitude.txt: PASS
- highfre644v_axion_backreaction_total_frequency_amplitude.txt: PASS
- highfre644v_axion_frequency_amplitude.txt: PASS
- highfre644v_pure_binary_template_frequency_amplitude.txt: PASS
- highfre_pure_peters_frequency_amplitude.txt: PASS

## Omega_GW exports

- highfre644_axionplusbackreaction_downward_omega_gw.txt: PASS
- highfre644_axionplusbackreaction_upward_omega_gw.txt: PASS
- highfre_axionplusbackreaction_downward_omega_gw.txt: PASS
- highfre_axionplusbackreaction_upward_omega_gw.txt: PASS
- highfre_pure_binary_template_downward_omega_gw.txt: PASS
- highfre_pure_binary_template_upward_omega_gw.txt: PASS

## Interpretation

- The waveform FFT convention is h_tilde(f)=int h(t) exp(-2 pi i f t) dt.
- The SGWB conversion uses local z=0 single-event spectra; cosmological redshift enters only in the population integral.
- The source-angle factors are read from transition metadata: 8/15 for Delta m=0 Bohr spectra and 4/5 for quadrupolar fine, hyperfine, and pure-binary template spectra.
