"""Independent quadrature and cross-script checks for the selected Bohr pair."""
from pathlib import Path
import importlib.util
import json
import numpy as np
from scipy.integrate import quad
from bohr_tidal import AxisymmetricBohrTide
from probe_bohr_monopole_feedback import make_sim
from adiabaticlimit import AdiabaticPhaseDiagram
from bohr_lz_tools import solve_standard_lz,lz_probability_and_coherence


def main():
    sim=make_sim()
    diagram=AdiabaticPhaseDiagram(eccentricity=.63)
    ref=diagram.compute_z_components(.2,1e-5,.01)
    a=ref['x_star'];scale=1e-5*.2**3*sim.c**3/(sim.G*sim.M)
    tide=AxisymmetricBohrTide((6,4,4),(5,4,4),sim._radial_wavefunction_dimensionless,orbital_samples=128)
    X=tide.radius_grid(a,.63)
    direct=np.zeros(X.size)
    def product(x):return sim._radial_wavefunction_dimensionless((6,4,4),x)*sim._radial_wavefunction_dimensionless((5,4,4),x)
    for L in tide.multipoles:
        for j,R in enumerate(X):
            inn=quad(lambda x:x**(L+2)*product(x),0,R,epsabs=1e-13,epsrel=2e-11)[0]
            out=quad(lambda x:x**(1-L)*product(x),R,np.inf,epsabs=1e-20,epsrel=2e-11)[0]
            direct[j]+=tide.angular[L]*(inn/R**(L+1)+R**L*out)
    eta_direct=scale*np.mean(direct*np.exp(1j*tide.mean_anomaly))
    eta_kernel=scale*tide.coefficients(a,.63)['eta'][0]
    eta_shared=sim._eta_vector(a*sim.r_c,.63)[0]
    zero=scale*tide.coefficients(a,0,range(1,9))['eta']
    report=dict(angular_factors=tide.angular,eta_direct_abs=abs(eta_direct),eta_kernel_abs=abs(eta_kernel),
        direct_relative_error=abs(eta_kernel/eta_direct-1),eta_shared_abs=abs(eta_shared),
        eta_diagram=ref['eta_rad_s'],cross_script_relative_error=abs(abs(eta_shared)/ref['eta_rad_s']-1),
        circular_nonzero_harmonic_max=float(max(abs(zero))),linear_passage_checks=[])
    assert report['direct_relative_error']<3e-5
    assert report['cross_script_relative_error']<3e-6
    assert report['circular_nonzero_harmonic_max']<1e-14
    for z in [1e-4,.103052584286,1.03,3.0]:
        x,c,out=solve_standard_lz(z,width_max=48,samples=101)
        exact=float(lz_probability_and_coherence(z)[1])
        report['linear_passage_checks'].append(dict(z=z,C_numeric=out,C_exact=exact,absolute_error=abs(out-exact)))
        assert abs(out-exact)<5e-5
    Path('diagnostics/bohr_projection_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
