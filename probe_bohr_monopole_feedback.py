"""Local two-state passage with full allowed multipoles and continuous feedback.

The projected potential includes all even L through 8 for |544> <-> |644>.
Secular diagonal shifts enter the detuning. The orbital equations are averaged
over the Kepler orbit; selected-harmonic tidal power is -s*n*Omega*N*hbar*dP_f/dt.
No artificial feedback gate or post-passage orbital impulse is applied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from bohr_tidal import AxisymmetricBohrTide
from highfre644v import EccentricResonantTidalGA


def make_sim(q=1e-5, cloud_fraction=1e-11, alpha=.20, upward=False):
    states=((5,4,4),(6,4,4)) if upward else ((6,4,4),(5,4,4))
    return EccentricResonantTidalGA(M_bh=.01,M_star=.01*q,alpha=alpha,bh_spin=.70,
        e_init=.63,cloud_mass_fraction=cloud_fraction,distance_Mpc=.001,
        initial_state=states[0],final_state=states[1],multi_harmonic_drive=False,
        backreaction_gate_mode='off')


def local_passage(sim, *, feedback=True, e_res=.63, width=12, orbital_samples=256,
                  radial_samples=8192, rtol=2e-9, multipoles=None):
    tide=AxisymmetricBohrTide(sim.transition_solver_data['initial_state'],
        sim.transition_solver_data['final_state'],sim._radial_wavefunction_dimensionless,
        multipoles=multipoles,radial_samples=radial_samples,orbital_samples=orbital_samples)
    scale=(sim.M_star/sim.M)*sim.alpha**3*sim.c**3/(sim.G*sim.M)
    bare=sim.transition_omega; sign=sim.transition_energy_sign; n=sim.resonance_harmonic
    def coupling(a,e):
        d=tide.coefficients(a/sim.r_c,e,(n,))
        eta=scale*d['eta'][0]
        splitting=bare+sign*scale*d['differential_diagonal']
        return eta,splitting
    def detuning(a,e):
        return n*np.sqrt(sim.G*sim.M_tot/a**3)-coupling(a,e)[1]
    a_bare=(sim.G*sim.M_tot/(bare/n)**2)**(1/3)
    a_res=brentq(lambda a:detuning(a,e_res),a_bare*.97,a_bare*1.03,xtol=1e-10)
    eta_res,omega_eff=coupling(a_res,e_res)
    da,de,Omega_res=sim._peters_rhs(a_res,e_res)
    h_a=a_res*1e-5;h_e=1e-6
    slope=((detuning(a_res+h_a,e_res)-detuning(a_res-h_a,e_res))/(2*h_a)*da
          +(detuning(a_res,e_res+h_e)-detuning(a_res,e_res-h_e))/(2*h_e)*de)
    if slope<=0:
        raise ValueError('This local setup requires a positive reference sweep.')
    tau=max(abs(eta_res)/slope,slope**-.5);T=width*tau
    def orbit_rhs(t,y):
        da,de,om=sim._peters_rhs(y[0]*a_res,y[1]);return [da/a_res,de]
    backward=solve_ivp(orbit_rhs,(0,-T),[1,e_res],rtol=1e-11,atol=1e-13,dense_output=True)
    start=backward.y[:,-1]
    vac=solve_ivp(orbit_rhs,(-T,T),start,rtol=1e-11,atol=1e-13,dense_output=True)
    a0,e0=start[0]*a_res,start[1];eta0,_=coupling(a0,e0);delta0=detuning(a0,e0)
    _,vectors=np.linalg.eigh([[delta0/2,eta0],[np.conj(eta0),-delta0/2]])
    incoming=vectors[:,np.argmax(abs(vectors[0]))]
    incoming*=np.exp(-1j*np.angle(incoming[0]))
    Na_hbar=sim.G*sim.M*sim.Mc_max/(sim.alpha*sim.c)
    def rhs(t,y):
        a=y[0]*a_res;e=float(y[1]);di=complex(y[2],y[3]);df=complex(y[4],y[5])
        da_p,de_p,om=sim._peters_rhs(a,e);eta,omega=coupling(a,e)
        delta=n*om-omega
        ddi=-1j*(delta/2*di+eta*df)-sim.Gamma_initial_decay*di
        ddf=-1j*(np.conj(eta)*di-delta/2*df)-sim.Gamma_decay*df
        transfer=-2*np.imag(eta*np.conj(di)*df)
        da_tid=0.0
        if feedback:
            da_tid=-2*a*a/(sim.G*sim.M*sim.M_star)*Na_hbar*sign*n*om*transfer
        de_tid=(1-e*e)/(2*a*e)*da_tid
        av=vac.sol(t)[0]*a_res;ov=np.sqrt(sim.G*sim.M_tot/av**3)
        return [(da_p+da_tid)/a_res,de_p+de_tid,ddi.real,ddi.imag,ddf.real,ddf.imag,om-ov]
    initial=[*start,incoming[0].real,incoming[0].imag,incoming[1].real,incoming[1].imag,0.0]
    sol=solve_ivp(rhs,(-T,T),initial,method='DOP853',rtol=rtol,
        atol=[1e-12,1e-12,1e-11,1e-11,1e-11,1e-11,1e-11],dense_output=True,max_step=tau/3)
    if not sol.success:raise RuntimeError(sol.message)
    ts=np.linspace(-T,T,2401);ys=sol.sol(ts);di=ys[2]+1j*ys[3];df=ys[4]+1j*ys[5]
    afin=ys[0,-1]*a_res;efin=ys[1,-1];etaf,omegaf=coupling(afin,efin);deltaf=detuning(afin,efin)
    _,v=np.linalg.eigh([[deltaf/2,etaf],[np.conj(etaf),-deltaf/2]])
    i_index=int(np.argmax(abs(v[0])));f_index=1-i_index
    end=np.array([di[-1],df[-1]]);bare_i=np.vdot(v[:,i_index],end);bare_f=np.vdot(v[:,f_index],end)
    C=float(abs(bare_i*bare_f));P=float(abs(bare_f)**2)
    z=abs(eta_res)**2/slope;survival=np.exp(-2*np.pi*z);C_lz=float(np.sqrt(survival*(-np.expm1(-2*np.pi*z))))
    valid_exit=bool(deltaf>8*max(abs(etaf),np.sqrt(slope)))
    A=abs(sim._cloud_amplitude())*(omega_eff/bare)**2
    coeff=tide.coefficients(a_res/sim.r_c,e_res,range(1,9),return_parts=True)
    dt=tau/10
    eta_minus=coupling(a_res-da*dt,e_res-de*dt)[0]
    eta_plus=coupling(a_res+da*dt,e_res+de*dt)[0]
    dm=detuning(*(backward.sol(-dt)*[a_res,1]))
    dp=detuning(*(vac.sol(dt)*[a_res,1]))
    stats=dict(q=sim.M_star/sim.M,cloud_fraction=sim.cloud_mass_fraction,alpha=sim.alpha,
        direction='upward' if sign>0 else 'downward',feedback=feedback,multipoles=list(tide.multipoles),
        e_res=e_res,a_res_over_rc=a_res/sim.r_c,omega_bare=bare,omega_res=omega_eff,
        eta_abs=float(abs(eta_res)),slope=float(slope),tau=float(tau),z=float(z),C_LZ=C_lz,
        C_out=C if valid_exit else None,C_endpoint=C,P_out=P if valid_exit else None,
        h_post_1kpc=float(A*C) if valid_exit else None,delta_final=float(deltaf),resolved_exit=valid_exit,
        delta_a_over_a=float((afin-vac.y[0,-1]*a_res)/a_res),phase_residual_cycles=float(ys[6,-1]/(2*np.pi)),
        norm_error=float(np.max(abs(abs(di)**2+abs(df)**2-1))),nfev=sol.nfev,
        eta_by_multipole={str(L):float((scale*z[0]).real) for L,z in coeff['parts'].items()},
        harmonic_separation_max=float(np.max(abs(scale*coeff['eta'][1:])/(np.arange(1,8)*Omega_res))),
        diagonal_shift=float(omega_eff-bare),width=width)
    stats.update(coupling_variation=float(abs(eta_plus-eta_minus)/(2*dt)*tau/abs(eta_res)),
        detuning_curvature=float(abs(dp+dm)/(dt*dt)*tau/slope),
        chirp_time=float(Omega_res/(-1.5*Omega_res*da/a_res)),
        damping_times_passage=float(max(sim.Gamma_initial_decay,sim.Gamma_decay)*tau),
        diagonal_modulation_over_omega=float(np.max(abs(scale*np.diff(coeff['diagonal_harmonics'],axis=1).ravel())/
            (np.arange(1,9)*Omega_res))))
    arrays=dict(t=ts,a=ys[0]*a_res,e=ys[1],a_vac=vac.sol(ts)[0]*a_res,
        coherence=abs(np.conj(di)*df),h_envelope=A*abs(np.conj(di)*df),phase_cycles=ys[6]/(2*np.pi))
    return stats,arrays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q',type=float,default=1e-5)
    parser.add_argument('--cloud-fraction',type=float,default=1e-11)
    parser.add_argument('--alpha',type=float,default=.20)
    parser.add_argument('--width',type=float,default=12);parser.add_argument('--upward',action='store_true')
    parser.add_argument('--no-feedback',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();sim=make_sim(args.q,args.cloud_fraction,alpha=args.alpha,upward=args.upward)
    stats,arrays=local_passage(sim,feedback=not args.no_feedback,width=args.width)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.with_suffix('.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
        np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
