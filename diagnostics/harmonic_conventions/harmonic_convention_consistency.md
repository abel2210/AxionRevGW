# Harmonic convention consistency audit

Checks: 49
Failures: 0

## Scope

- Hansen convention: `(a/R)^(l+1) exp(-i m f)=sum_n X_n exp(-i n M)`, implemented by averaging against `exp(+i n M)`.
- Circular limit for `|Delta m|=2`: only the `n=2` harmonic is nonzero.
- Circular limit for `Delta m=0`: the positive-frequency radial harmonics vanish.
- Finite-separation Fourier convention for the high-frequency kernel in the same circular limits.
- Active-harmonic selection for selected-RWA and nearest-harmonic multi-drive checks.
- Rotating-frame phase convention `exp[-i(n-n_r)Phi]` with the selected harmonic phase equal to unity.

## Results

All harmonic convention checks passed.

CSV details: `diagnostics/harmonic_conventions/harmonic_convention_consistency.csv`
