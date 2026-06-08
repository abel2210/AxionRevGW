import numpy as np
import _plot_backend  # noqa: F401
import matplotlib.pyplot as plt
import math
from pathlib import Path
from scipy.integrate import cumulative_trapezoid
from scipy.special import eval_genlaguerre

# 兼容不同版本的 scipy.special 球谐函数
try:
    from scipy.special import sph_harm_y
    def spherical_harmonic(m, l, phi, theta):
        return sph_harm_y(l, m, theta, phi)
except ImportError:
    from scipy.special import sph_harm
    def spherical_harmonic(m, l, phi, theta):
        return sph_harm(m, l, phi, theta)

# ==========================================
# PRL 风格高品质绘图设置
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

class AdiabaticPhaseDiagram:
    """
    轻量级、独立的绝热参数相图生成器
    仅提取共振点瞬时公式，剥离所有繁重的 ODE 积分和波形生成
    """
    def __init__(
        self,
        initial_state=(6, 4, 4),
        final_state=(5, 4, 4),
        resonance_harmonic=1,
        eccentricity=0.64,
        tidal_l=2,
    ):
        # 物理常数
        self.G = 6.6743e-11
        self.c = 2.99792458e8
        self.M_sun = 1.98847e30
        
        # 跃迁配置
        self.initial_state = initial_state
        self.final_state = final_state
        self.resonance_harmonic = resonance_harmonic
        self.eccentricity = float(eccentricity)
        self.tidal_l = int(tidal_l)
        self.radial_power = self.tidal_l + 1
        self._hansen_cache = {}
        
        # 预计算：无量纲的空间重叠积分（不依赖于 M 和 alpha，只需算一次！）
        self.mixing_overlap_data = self._precompute_mixing_overlaps()

    def _radial_wavefunction_dimensionless(self, state, x):
        n, l, _ = state
        rho = 2.0 * x / n
        normalization = (2.0 / n) ** 1.5 * math.sqrt(
            math.factorial(n - l - 1) / (2.0 * n * math.factorial(n + l))
        )
        laguerre = eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
        return normalization * np.exp(-x / n) * rho**l * laguerre

    def _compute_angular_overlap(self, initial_state, final_state):
        _, l_i, m_i = initial_state
        _, l_f, m_f = final_state
        m_star = m_f - m_i
        
        theta = np.linspace(0.0, np.pi, 320)
        phi = np.linspace(0.0, 2.0 * np.pi, 640, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        
        y_tidal = spherical_harmonic(m_star, 2, phi_grid, theta_grid)
        y_i = spherical_harmonic(m_i, l_i, phi_grid, theta_grid)
        y_f = spherical_harmonic(m_f, l_f, phi_grid, theta_grid)
        
        integrand = y_tidal * y_i * np.conj(y_f) * np.sin(theta_grid)
        phi_integral = np.trapezoid(integrand, phi, axis=1)
        full_integral = np.trapezoid(phi_integral, theta)
        return abs(full_integral)

    def _precompute_mixing_overlaps(self):
        """预计算空间重叠，极大地加速了 alpha 扫描"""
        x_grid = np.logspace(-6, 4, 2048)  # dimensionless r/r_c
        radial_i = self._radial_wavefunction_dimensionless(self.initial_state, x_grid)
        radial_f = self._radial_wavefunction_dimensionless(self.final_state, x_grid)
        
        inner_integrand = x_grid**4 * radial_i * radial_f
        outer_integrand = (radial_i * radial_f) / x_grid
        
        inner_cumulative = cumulative_trapezoid(inner_integrand, x_grid, initial=0.0)
        outer_total = np.trapezoid(outer_integrand, x_grid)
        outer_cumulative = outer_total - cumulative_trapezoid(outer_integrand, x_grid, initial=0.0)
        
        angular_overlap = self._compute_angular_overlap(self.initial_state, self.final_state)
        
        return {
            "x_grid": x_grid,
            "inner_cumulative": inner_cumulative,
            "outer_cumulative": outer_cumulative,
            "angular_overlap": angular_overlap,
        }

    def _solve_kepler(self, mean_anomaly, eccentricity, max_iter=18, tol=1.0e-13):
        eccentricity = float(np.clip(eccentricity, 0.0, 0.999))
        mean_anomaly = np.asarray(mean_anomaly, dtype=float)
        guess = np.where(eccentricity < 0.8, mean_anomaly, np.pi * np.ones_like(mean_anomaly))
        for _ in range(max_iter):
            residual = guess - eccentricity * np.sin(guess) - mean_anomaly
            jacobian = 1.0 - eccentricity * np.cos(guess)
            step = residual / jacobian
            guess -= step
            if np.max(np.abs(step)) < tol:
                break
        return guess

    def _hansen_coefficient(self, eccentricity, harmonic):
        """Same eccentric-Hansen normalization used by highfre_shared._eta_vector."""
        eccentricity = float(np.clip(eccentricity, 0.0, 0.999))
        harmonic = int(harmonic)
        cache_key = (round(eccentricity, 12), harmonic)
        if cache_key in self._hansen_cache:
            return self._hansen_cache[cache_key]
        m_star = self.final_state[2] - self.initial_state[2]
        mean_anomaly = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
        eccentric_anomaly = self._solve_kepler(mean_anomaly, eccentricity)
        radial_ratio = 1.0 - eccentricity * np.cos(eccentric_anomaly)
        cos_true = (np.cos(eccentric_anomaly) - eccentricity) / radial_ratio
        sin_true = np.sqrt(max(1.0e-14, 1.0 - eccentricity**2)) * np.sin(eccentric_anomaly) / radial_ratio
        true_anomaly = np.arctan2(sin_true, cos_true)
        base = radial_ratio ** (-self.radial_power) * np.exp(-1j * abs(m_star) * true_anomaly)
        coefficient = np.mean(np.exp(1j * harmonic * mean_anomaly) * base)
        self._hansen_cache[cache_key] = coefficient
        return coefficient

    def _eccentric_sweep_factor(self, eccentricity):
        eccentricity = float(np.clip(eccentricity, 0.0, 0.999))
        one_minus_e2 = max(1.0e-12, 1.0 - eccentricity * eccentricity)
        return (
            (1.0 + (73.0 / 24.0) * eccentricity**2 + (37.0 / 96.0) * eccentricity**4)
            / one_minus_e2**3.5
        )

    def _omega_real_geom(self, state, alpha):
        n, l, _ = state
        term1 = 1.0
        term2 = -alpha**2 / (2.0 * n**2)
        term3 = -alpha**4 / (8.0 * n**4)
        term4 = ((4.0 * l - 6.0 * n + 2.0) / (2.0 * n * (l + 1.0))) * (alpha**4 / n**3)
        return alpha * (term1 + term2 + term3 + term4)

    def compute_z_components(self, alpha_val, q_ratio, M_bh_solar=1.0, eccentricity=None):
        eccentricity = self.eccentricity if eccentricity is None else float(eccentricity)
        omega_i = self._omega_real_geom(self.initial_state, alpha_val)
        omega_f = self._omega_real_geom(self.final_state, alpha_val)
        delta_omega_geom = abs(omega_i - omega_f)
        if delta_omega_geom <= 0.0:
            return {"z": np.nan}

        primary_mass_kg = M_bh_solar * self.M_sun
        companion_mass_kg = q_ratio * primary_mass_kg
        geometric_to_si = self.c**3 / (self.G * primary_mass_kg)

        transition_omega = delta_omega_geom * geometric_to_si
        omega_orb_res = transition_omega / self.resonance_harmonic
        semi_major_axis = (self.G * (primary_mass_kg + companion_mass_kg) / omega_orb_res**2) ** (1.0 / 3.0)

        r_c_local = (self.G * primary_mass_kg / self.c**2) / alpha_val**2
        x_star = np.clip(
            semi_major_axis / r_c_local,
            self.mixing_overlap_data["x_grid"][0],
            self.mixing_overlap_data["x_grid"][-1],
        )

        i_in = np.interp(x_star, self.mixing_overlap_data["x_grid"], self.mixing_overlap_data["inner_cumulative"])
        i_out = np.interp(x_star, self.mixing_overlap_data["x_grid"], self.mixing_overlap_data["outer_cumulative"])
        i_a = self.mixing_overlap_data["angular_overlap"]

        m_omega = self.G * primary_mass_kg * omega_orb_res / self.c**3
        term_inner = q_ratio * m_omega * i_in / (alpha_val**3 * (1.0 + q_ratio))
        term_outer = (
            alpha_val**7
            * q_ratio
            * (1.0 + q_ratio) ** (2.0 / 3.0)
            * i_out
            / max(m_omega, 1.0e-30) ** (7.0 / 3.0)
        )

        eta_circular_rad_s = (3.0 * np.pi / 10.0) * i_a * abs(term_inner + term_outer) * omega_orb_res
        hansen_abs = abs(self._hansen_coefficient(eccentricity, self.resonance_harmonic))
        eta_rad_s = eta_circular_rad_s * hansen_abs

        orbital_sweep_rate = (
            omega_orb_res**2
            * (96.0 / 5.0)
            * q_ratio
            / (1.0 + q_ratio) ** (1.0 / 3.0)
            * m_omega ** (5.0 / 3.0)
        )
        circular_resonance_sweep_rate = self.resonance_harmonic * orbital_sweep_rate
        eccentric_sweep_factor = self._eccentric_sweep_factor(eccentricity)
        resonance_sweep_rate = circular_resonance_sweep_rate * eccentric_sweep_factor
        z_value = eta_rad_s**2 / max(resonance_sweep_rate, 1.0e-60)
        return {
            "z": z_value,
            "eta_circular_rad_s": eta_circular_rad_s,
            "hansen_abs": hansen_abs,
            "eta_rad_s": eta_rad_s,
            "circular_sweep_rate": circular_resonance_sweep_rate,
            "eccentric_sweep_factor": eccentric_sweep_factor,
            "resonance_sweep_rate": resonance_sweep_rate,
            "x_star": x_star,
        }

    def compute_z_parameter(self, alpha_val, q_ratio, M_bh_solar=1.0, eccentricity=None):
        return self.compute_z_components(alpha_val, q_ratio, M_bh_solar, eccentricity)["z"]

    def compute_z_parameter_legacy(self, alpha_val, q_ratio, M_bh_solar=1.0):
        """核心计算函数：获取特定参数下的 z 值"""
        # 1. 计算跃迁频率
        omega_i = self._omega_real_geom(self.initial_state, alpha_val)
        omega_f = self._omega_real_geom(self.final_state, alpha_val)
        delta_omega_geom = abs(omega_i - omega_f)
        if delta_omega_geom <= 0.0:
            return np.nan
            
        primary_mass_kg = M_bh_solar * self.M_sun
        companion_mass_kg = q_ratio * primary_mass_kg
        geometric_to_si = self.c**3 / (self.G * primary_mass_kg)
        
        transition_omega = delta_omega_geom * geometric_to_si
        omega_orb_res = transition_omega / self.resonance_harmonic
        
        # 2. 计算共振点的半长轴
        semi_major_axis = (self.G * (primary_mass_kg + companion_mass_kg) / omega_orb_res**2) ** (1.0 / 3.0)
        
        # 3. 计算耦合强度 eta
        r_c_local = (self.G * primary_mass_kg / self.c**2) / alpha_val**2
        x_star = np.clip(semi_major_axis / r_c_local, 
                         self.mixing_overlap_data["x_grid"][0], 
                         self.mixing_overlap_data["x_grid"][-1])
        
        i_in = np.interp(x_star, self.mixing_overlap_data["x_grid"], self.mixing_overlap_data["inner_cumulative"])
        i_out = np.interp(x_star, self.mixing_overlap_data["x_grid"], self.mixing_overlap_data["outer_cumulative"])
        i_a = self.mixing_overlap_data["angular_overlap"]
        
        m_omega = self.G * primary_mass_kg * omega_orb_res / self.c**3
        term_inner = q_ratio * m_omega * i_in / (alpha_val**3 * (1.0 + q_ratio))
        term_outer = (alpha_val**7 * q_ratio * (1.0 + q_ratio)**(2.0 / 3.0) * i_out) / max(m_omega, 1.0e-30)**(7.0 / 3.0)
        
        eta_rad_s = (3.0 * np.pi / 10.0) * i_a * abs(term_inner + term_outer) * omega_orb_res
        
        # 4. 计算引力波扫频速度 d\Omega/dt
        orbital_sweep_rate = omega_orb_res**2 * (96.0 / 5.0) * q_ratio / (1.0 + q_ratio)**(1.0/3.0) * m_omega**(5.0/3.0)
        resonance_sweep_rate = self.resonance_harmonic * orbital_sweep_rate
        
        # 5. 绝热参数 z
        z_value = (2.0 * eta_rad_s)**2 / max(resonance_sweep_rate, 1.0e-60)
        return z_value

def plot_adiabatic_phase_diagram():
    # 实例化轻量级生成器 (玻尔跃迁 644 -> 544)
    eccentricity_ref = 0.64
    diagram_generator = AdiabaticPhaseDiagram(
        initial_state=(6,4,4),
        final_state=(5,4,4),
        resonance_harmonic=1,
        eccentricity=eccentricity_ref,
    )
    
    # 构建高精度网格
    alpha_grid = np.linspace(0.18, 0.35, 150)
    q_grid = np.logspace(-4, -0.5, 150)
    A, Q = np.meshgrid(alpha_grid, q_grid)
    Z = np.zeros_like(A)
    
    print("Computing phase diagram over 150x150 grid... (Takes ~1 second)")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            # 质量 M 对 z 基本无影响，固定为 1.0
            Z[i, j] = diagram_generator.compute_z_parameter(A[i, j], Q[i, j], M_bh_solar=1.0)
            
    # --- 绘图逻辑 ---
    fig, ax = plt.subplots(figsize=(10, 7.5))
    
    # 填充彩色对数等高线 (RdYlBu_r 呈现红蓝对比)
    p_lz = 1.0 - np.exp(-2.0 * np.pi * Z)
    coherence = np.sqrt(np.clip(p_lz * (1.0 - p_lz), 0.0, None))
    levels = np.linspace(0.0, 0.5, 80)
    cf = ax.contourf(A, Q, coherence, levels=levels, cmap="viridis", extend="max")
    
    # 绘制 z = 1 临界边界线
    c_line = ax.contour(
        A,
        Q,
        Z,
        levels=[0.1, 1.0],
        colors=["0.92", "white"],
        linewidths=[2.0, 3.2],
        linestyles=["-", "--"],
    )
    
    # 物理区域标注
    # 1. 绝热猝灭区 (左上)
    ax.text(0.215, 1.7e-1, "Adiabatic depletion\n$(z_{\\rm LZ}\\gg1)$\n",
            color='white', fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.5'))
    
    # 2. 扫频复活区 (右下)
    ax.text(0.315, 7.0e-4, "Weak passage\n$(z_{\\rm LZ}\\ll1)$\n",
            color='white', fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(facecolor='black', alpha=0.25, edgecolor='none', boxstyle='round,pad=0.5'))
    ax.text(0.302, 1.4e-2, "finite\ncoherence",
            color='white', fontsize=14, fontweight='bold', ha='center', va='center')
    
    # 手动标注等高线
    ax.clabel(
        c_line,
        fmt={0.1: r"$z_{\rm LZ}=0.1$", 1.0: r"$z_{\rm LZ}=1$"},
        fontsize=13,
        colors="white",
        manual=[(0.305, 2.0e-3), (0.248, 8.0e-2)],
    )

    markers = [
        (0.2376, 1.0e-2, "near boundary"),
        (0.3000, 1.0e-2, r"Figs. 1,2"),
    ]
    for alpha_marker, q_marker, label in markers:
        ax.plot(alpha_marker, q_marker, marker="o", ms=6.0, mfc="none", mec="white", mew=1.4)
        ax.text(alpha_marker + 0.003, q_marker * 1.08, label, color="white", fontsize=10, va="bottom")
    
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, ticks=np.linspace(0.0, 0.5, 6))
    cbar.set_label(r'LZ coherence $\mathcal{C}_{\rm LZ}$', fontsize=16)
    
    ax.set_yscale('log')
    ax.set_xlim(0.18, 0.35)
    ax.set_ylim(1e-4, 10**-0.5)
    
    ax.set_xlabel(r'$\alpha$', fontsize=16)
    ax.set_ylabel(r'$q$', fontsize=16)
    #ax.set_title(r'Dynamical Phase Diagram of Bohr Transitions ($\Delta m=0$)', fontsize=18, pad=15)
    ax.text(
        0.182,
        1.35e-4,
        rf"$e_{{\rm res}}\simeq {eccentricity_ref:.2f}$, selected $n=1$ harmonic",
        fontsize=12,
        ha="left",
        va="bottom",
        color="white",
    )
    
    # 增加细网格线提升学术感
    ax.grid(True, which="both", ls="--", alpha=0.2)
    
    plt.tight_layout()
    output_paths = [
        Path("figures") / "adp_eccentric.pdf",
    ]
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved '{output_path}'")
    plt.show()

if __name__ == "__main__":
    plot_adiabatic_phase_diagram()
