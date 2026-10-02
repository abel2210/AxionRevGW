"""Reproduce manuscript Bohr Figs. 1--3 from the finite-separation projection.

The fiducial waveform uses continuous orbit feedback. Parameter maps and the
Landau--Zener comparison are explicitly vacuum-sweep, two-state references.
"""
from pathlib import Path
import json
import numpy as np
import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from probe_bohr_monopole_feedback import make_sim,local_passage
from adiabaticlimit import AdiabaticPhaseDiagram
from bohr_lz_tools import solve_standard_lz

ROOT=Path(__file__).resolve().parent
DIAG=ROOT/'diagnostics'
ALPHA_REF=.20
Q_REF=1e-5
E_RES=.63
MC_REF=1e-11


def coherence(z):
    survival=np.exp(-2*np.pi*np.asarray(z))
    return np.sqrt(survival*(-np.expm1(-2*np.pi*np.asarray(z))))


def run_case(name,**kwargs):
    feedback=kwargs.pop('feedback',True)
    width=kwargs.pop('width',12)
    sim=make_sim(**kwargs)
    stats,arrays=local_passage(sim,feedback=feedback,width=width)
    DIAG.mkdir(exist_ok=True)
    (DIAG/f'bohr_full_{name}.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    np.savez_compressed(DIAG/f'bohr_full_{name}.npz',**arrays)
    return stats,arrays


def style():
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
        'mathtext.fontset':'stix','font.size':8,'axes.labelsize':8,'legend.fontsize':7,
        'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.65,
        'xtick.direction':'in','ytick.direction':'in','axes.grid':False})


def save(fig,name):
    for directory in [ROOT/'figures']:
        directory.mkdir(exist_ok=True,parents=True)
        fig.savefig(directory/name,bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {name}',flush=True)


def figure_pair():
    style()
    fig,axes=plt.subplots(2,2,figsize=(7.0,4.3),sharex=True,constrained_layout=True)
    for col,up in enumerate([True,False]):
        s,a=run_case('fiducial_up' if up else 'fiducial_down',upward=up)
        x=a['t']/s['tau'];ares=make_sim().r_c*s['a_res_over_rc']
        ax=axes[0,col]
        ax.plot(x,(a['a']-a['a_vac'])/ares*1e7,color='#245f8f',lw=.8)
        ax.set_ylabel(r'$10^7\,(a-a_{\rm vac})/a_{\rm res}$')
        ax.set_title(r'$|544\rangle\to|644\rangle$' if up else r'$|644\rangle\to|544\rangle$')
        inset=ax.inset_axes([.12,.62,.36,.30])
        inset.plot(x,a['phase_cycles'],color='#714b92',lw=.9)
        inset.set_title(r'$\Delta\Phi_{\rm bin}/2\pi$',fontsize=6)
        inset.tick_params(labelsize=5)
        ax=axes[1,col]
        ax.fill_between(x,0,a['h_envelope']*1e33,color='#a8cce3',alpha=.75)
        ax.plot(x,a['h_envelope']*1e33,color='#245f8f',lw=.7)
        ax.axhline(s['h_post_1kpc']*1e33,color='#245f8f',ls='--',lw=.8)
        ax.set_ylabel(r'$h_{\rm env}\ [10^{-33}]$')
        ax.set_ylim(0,1.5);ax.set_xlabel(r'$(t-t_{\rm res,vac})/\tau_{\rm pass}$')
        twin=ax.twinx();twin.plot(x,a['coherence'],color='#b96026',lw=.65,alpha=.6)
        twin.set_ylim(0,1.5/(s['h_post_1kpc']*1e33/s['C_out']))
        twin.set_ylabel(r'$|b_i^*b_f|$')
        for row in range(2):
            axes[row,col].axvline(0,color='.4',lw=.7,ls=':')
            axes[row,col].set_xlim(-12,12)
    save(fig,'bohr_orbit_time_summary_644_pair_m1_0p01_q0001.pdf')


def figure_map():
    style()
    d=AdiabaticPhaseDiagram(eccentricity=E_RES)
    ref=d.compute_z_components(ALPHA_REF,Q_REF,.01)
    zref=float(ref['z']);cref=float(coherence(zref))
    amp=abs(make_sim()._cloud_amplitude())*(ref['omega_res']/make_sim().transition_omega)**2
    alpha=np.linspace(.10,.35,200);q=np.geomspace(1e-6,1e-3,200)
    aa,qq=np.meshgrid(alpha,q)
    zz=zref*(qq/Q_REF)*(aa/ALPHA_REF)**-5;cc=coherence(zz)
    hh=amp*cc*(aa/ALPHA_REF)**2
    fig,axes=plt.subplots(1,2,figsize=(7,2.8),constrained_layout=True)
    im=axes[0].pcolormesh(aa,qq,cc,vmin=0,vmax=.5,cmap='viridis',shading='auto',rasterized=True)
    fig.colorbar(im,ax=axes[0],label=r'$C_{\rm out}$')
    cs=axes[0].contour(aa,qq,cc,levels=[.1,.3],colors='white',linewidths=.6)
    axes[0].clabel(cs,fmt='%.1f',fontsize=6)
    axes[0].contour(aa,qq,zz,levels=[1],colors='#f29c4b',linestyles='--',linewidths=1)
    im=axes[1].pcolormesh(aa,qq,np.maximum(hh,1e-40),norm=LogNorm(1e-37,3e-33),cmap='magma',shading='auto',rasterized=True)
    fig.colorbar(im,ax=axes[1],label=r'$h_{\rm post}$ at 1 kpc',extend='min')
    for ax,title in zip(axes,['(a) Local LZ reference','(b) Fixed cloud mass']):
        ax.set_yscale('log');ax.set_xlabel(r'$\alpha$');ax.set_ylabel(r'$q=M_2/M_1$')
        ax.scatter([ALPHA_REF],[Q_REF],marker='*',s=65,c='#34d399',edgecolors='black',lw=.5)
        ax.set_title(title)
    np.savez_compressed(DIAG/'bohr_full_map.npz',alpha=alpha,q=q,z=zz,coherence=cc,h_post=hh)
    (DIAG/'bohr_full_map.json').write_text(json.dumps(dict(alpha_ref=ALPHA_REF,q_ref=Q_REF,
        e_res=E_RES,cloud_fraction=MC_REF,z_ref=zref,C_ref=cref,h_ref=amp*cref,
        scope='small-q scaling of a vacuum-sweep two-state reference; feedback and level isolation not validated over the whole map'),indent=2))
    save(fig,'bohr_domain_visibility_map_single.pdf')


def figure_lz():
    style()
    ref,a=run_case('fiducial_vac',feedback=False)
    slow,b=run_case('adiabatic_vac',q=1e-4,feedback=False,width=48)
    diagram=AdiabaticPhaseDiagram(eccentricity=E_RES)
    qs=np.geomspace(1e-7,2e-4,120)
    z=np.array([diagram.compute_z_parameter(ALPHA_REF,q,.01) for q in qs])
    fig,axes=plt.subplots(2,2,figsize=(7,4.5),constrained_layout=True)
    ax=axes[0,0];ax.semilogx(qs,coherence(z),color='#047857')
    ax.set(xlabel=r'$q$ at $\alpha=0.20$',ylabel=r'$C_{\rm out}^{\rm LZ}$',title='(a) Physical vacuum sweeps')
    for s in [ref,slow]:ax.scatter(s['q'],s['C_LZ'],s=22,color='#9a3412')
    ax=axes[1,0]
    for s,arr,color in [(ref,a,'#047857'),(slow,b,'#9a3412')]:
        ax.plot(arr['t']/s['tau'],arr['coherence'],lw=.8,color=color,label=rf"$q={s['q']:.0e}$, $z={s['z']:.3f}$")
        ax.axhline(s['C_out'],ls='--',lw=.6,color=color)
    ax.set(xlabel=r'$(t-t_{\rm res})/\tau_{\rm pass}$',ylabel=r'$h_{\rm env}/\mathcal{A}_0$',title='(b) Integrated prescribed orbits',ylim=(0,.56),xlim=(-12,12))
    ax.legend(loc='upper left',fontsize=6)
    ax=axes[0,1];zz=np.geomspace(1e-4,8,300)
    ax.semilogx(zz,coherence(zz),color='#245f8f',label=r'$C_{\rm out}$')
    ax.semilogx(zz,-np.expm1(-2*np.pi*zz),color='.5',ls='--',label=r'$P_{\rm tr}$')
    ax.set(xlabel=r'$z_{\rm LZ}$',ylabel='Asymptotic value',title='(c) Fixed strain normalization');ax.legend()
    ax=axes[1,1]
    for zz,color,label in [(3,'#9a3412','adiabatic'),(ref['z'],'#047857','finite'),(1e-4,'.4','weak')]:
        x,c,end=solve_standard_lz(zz,width_max=24,samples=2401)
        ax.plot(x,c,color=color,lw=.9,label=label)
        ax.axhline(float(coherence(zz)),ls='--',color=color,lw=.5)
    ax.set(xlabel='Dimensionless passage time',ylabel=r'$h_{\rm env}/\mathcal{A}_0$',title='(d) Controlled linear sweeps',ylim=(0,.56))
    ax.legend(loc='upper left',fontsize=6)
    save(fig,'bohr_visibility_two_group_four_panel.pdf')


if __name__=='__main__':
    figure_pair();figure_map();figure_lz()
