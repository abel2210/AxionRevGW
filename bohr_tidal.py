"""Finite-separation Newtonian potential projected onto an equal-l,m Bohr pair.

Off-diagonal coefficients use a common rephasing of one state to absorb the
overall minus sign in V. Diagonal shifts retain the physical minus sign.
All returned kernels are dimensionless; multiply by q*alpha**3*c**3/(G*M1).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import eval_legendre, sph_harm_y


class AxisymmetricBohrTide:
    def __init__(self, initial, final, radial_function, *, multipoles=None,
                 radial_samples=8192, orbital_samples=512, max_x=5000.0):
        initial, final = tuple(initial), tuple(final)
        if initial[0] == final[0] or initial[1:] != final[1:]:
            raise ValueError('An equal-l,m radial Bohr pair is required.')
        ell, m = initial[1:]
        allowed = tuple(range(0, 2*ell+1, 2))
        self.multipoles = allowed if multipoles is None else tuple(multipoles)
        if any(L not in allowed for L in self.multipoles):
            raise ValueError(f'Allowed multipoles for this pair: {allowed}')
        self.x = np.geomspace(1e-6, max_x, int(radial_samples))
        ri, rf = (radial_function(state, self.x) for state in (initial, final))
        products = np.array([ri*rf, ri*ri, rf*rf])
        u, w = np.polynomial.legendre.leggauss(max(64, 2*ell+8))
        angular_density = 2*np.pi*abs(sph_harm_y(ell, m, np.arccos(u), 0))**2
        self.angular = {}
        self.radial = {}
        for L in self.multipoles:
            self.angular[L] = float(eval_legendre(L, 0)*np.dot(w, angular_density*eval_legendre(L,u)))
            inner = cumulative_trapezoid(self.x**(L+2)*products, self.x, axis=1, initial=0)
            outer = -cumulative_trapezoid(
                (self.x**(1-L)*products)[:, ::-1], self.x[::-1], axis=1, initial=0
            )[:, ::-1]
            self.radial[L] = inner, outer
        self.mean_anomaly = np.arange(int(orbital_samples))*2*np.pi/int(orbital_samples)

    def radius_grid(self, a_over_rc, eccentricity):
        e = float(eccentricity)
        if not 0 <= e < 1:
            raise ValueError('eccentricity must be in [0,1)')
        M = self.mean_anomaly
        E = M.copy() if e < .8 else np.full_like(M,np.pi)
        for _ in range(24):
            step = (E-e*np.sin(E)-M)/(1-e*np.cos(E))
            E -= step
            if np.max(abs(step)) < 1e-13:
                break
        return float(a_over_rc)*(1-e*np.cos(E))

    def coefficients(self, a_over_rc, eccentricity, harmonics=(1,), *, return_parts=False):
        X = self.radius_grid(a_over_rc, eccentricity)
        if X.min() < self.x[0] or X.max() > self.x[-1]:
            raise ValueError('Companion radius is outside the radial integration grid.')
        phase = np.exp(1j*np.outer(np.asarray(harmonics),self.mean_anomaly))
        total = np.zeros((3, X.size))
        parts = {}
        for L in self.multipoles:
            inner, outer = self.radial[L]
            values = self.angular[L]*np.array([
                np.interp(X,self.x,inn)/X**(L+1)+X**L*np.interp(X,self.x,out)
                for inn,out in zip(inner,outer)
            ])
            total += values
            if return_parts:
                parts[L] = phase@values[0]/X.size
        offdiagonal = phase@total[0]/X.size
        diagonal = -total[1:].mean(axis=1)
        result = dict(eta=offdiagonal, diagonal=diagonal,
                      differential_diagonal=diagonal[1]-diagonal[0])
        if return_parts:
            result['parts'] = parts
            result['diagonal_harmonics'] = -phase@total[1:].T/X.size
        return result
