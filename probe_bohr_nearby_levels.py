"""Finite bound-state screen around the selected Bohr passage (not a continuum calculation)."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import sph_harm_y
from probe_bohr_monopole_feedback import make_sim
from bohr_tidal import AxisymmetricBohrTide


def screen(reference_path='diagnostics/bohr_full_fiducial_down.json'):
    reference=json.loads(Path(reference_path).read_text())
    sim=make_sim(q=reference['q'],cloud_fraction=reference['cloud_fraction'],alpha=reference['alpha'])
    omega=reference['omega_res'];a=reference['a_res_over_rc'];e=.63
    tide=AxisymmetricBohrTide((6,4,4),(5,4,4),sim._radial_wavefunction_dimensionless)
    X=tide.radius_grid(a,e);M=tide.mean_anomaly
    E=M.copy()
    for _ in range(20):E-=(E-e*np.sin(E)-M)/(1-e*np.cos(E))
    phi=np.arctan2(np.sqrt(1-e*e)*np.sin(E),np.cos(E)-e)
    x=tide.x;u,w=np.polynomial.legendre.leggauss(80);theta=np.arccos(u)
    scale=sim.M_star/sim.M*sim.alpha**3*sim.c**3/(sim.G*sim.M)
    freqscale=sim.c**3/(sim.G*sim.M)
    rows=[]
    for initial in [(6,4,4),(5,4,4)]:
        ri=sim._radial_wavefunction_dimensionless(initial,x)
        yi=sph_harm_y(initial[1],initial[2],theta,0)
        wi=sim._omega_real_geom(initial)*freqscale
        for nf in range(1,11):
            for lf in range(nf):
                for mf in range(-lf,lf+1):
                    final=(nf,lf,mf)
                    if final in [(6,4,4),(5,4,4)]:continue
                    wf=sim._omega_real_geom(final)*freqscale
                    k=int(np.rint((wf-wi)/omega))
                    delta=wf-wi-k*omega
                    if abs(k)>8 or abs(delta)>100:continue
                    ms=mf-initial[2];ls=range(abs(lf-initial[1]),lf+initial[1]+1)
                    vals=np.zeros(X.size,dtype=complex)
                    rf=sim._radial_wavefunction_dimensionless(final,x)
                    yf=sph_harm_y(lf,mf,theta,0)
                    for L in ls:
                        if abs(ms)>L or (lf+initial[1]+L)%2 or (L+ms)%2:continue
                        angular=4*np.pi/(2*L+1)*np.conj(sph_harm_y(L,ms,np.pi/2,0))*2*np.pi*np.dot(w,np.conj(yf)*sph_harm_y(L,ms,theta,0)*yi)
                        prod=ri*rf
                        inn=cumulative_trapezoid(x**(L+2)*prod,x,initial=0)
                        out=-cumulative_trapezoid((x**(1-L)*prod)[::-1],x[::-1],initial=0)[::-1]
                        radial=np.interp(X,x,inn)/X**(L+1)+X**L*np.interp(X,x,out)
                        if L==1:radial-=inn[-1]/X**2  # primary-frame acceleration
                        vals+=angular*radial*np.exp(-1j*ms*phi)
                    eta=scale*abs(np.mean(vals*np.exp(1j*k*M)))
                    if eta<1e-12:continue
                    rows.append(dict(initial=initial,final=final,k=k,detuning=delta,eta=eta,
                        ratio=eta/max(abs(delta),1e-30),
                        isolated_crossing_probability_estimate=float(-np.expm1(-2*np.pi*eta**2/(abs(k)*reference['slope']))) if k else None))
    rows.sort(key=lambda d:d['ratio'],reverse=True)
    out=dict(scope='n<=10, |k|<=8, bare detuning within 100 rad/s; no continuum or multilevel evolution',rows=rows)
    Path(reference_path).with_suffix('.levels.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(dict(scope=out['scope'],rows=rows[:8]),indent=2))

if __name__=='__main__':
    import sys
    screen(*sys.argv[1:])
