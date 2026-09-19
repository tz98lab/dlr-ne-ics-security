#!/usr/bin/env python3
"""
================================================================================
Exp. 4 (v2): Game-Theoretic Robustness — Corollary 1 verification within a
             single network family

Corollary 1 states a *per-network* bound:
    ||V(π_D) - V(π̃_D)||_∞ ≤ L_κ ||ΔS||_F ,   L_κ = C_robust · κ(S)

The constant C_robust depends on the trained network geometry (w_out, L_φ,
R_X, ...) and is therefore NOT expected to be invariant across different
regularization levels β (v1 measured this cross-β regime).

This v2 script isolates the quantitative content of Corollary 1:
  * β is FIXED (operating point) → the network geometry statistics are
    (approximately) held fixed, so C_robust should be roughly constant;
  * only the trained core matrix S varies — across ranks r and seeds;
  * we then test whether L_κ scales proportionally with κ(S) across the
    resulting family of networks.

Output:
  * exp4_robustness.png       — two panels:
      Left : ΔV vs ||ΔS||_F (linearity per network, fixed β)
      Right: fitted L_κ vs κ(S) across the network family, with linear fit
             slope = empirical C_robust  (expect: positive, through origin)
  * results_data/exp4_results.json — all raw measurements + fit statistics
================================================================================
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, svd, norm
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. Experiment Configuration
# ==============================================================================

FIXED_BETA = 0.1          # operating point: β fixed for the whole experiment
RANKS = [10, 20, 40, 60]  # retraction ranks r (varies κ across the family)
SEEDS = [42, 7]           # independent training runs per rank
TRAIN_ITERS = 80
PERTURBATION_NORMS = [0.005, 0.01, 0.02, 0.05]

# ==============================================================================
# 1. Environment (same low-rank coupling structure as Exp. 2–3)
# ==============================================================================

class PowerSystemConfig:
    def __init__(self):
        self.n = 200
        self.m = 200
        self.ra = 3
        self.rd = 5
        self.rr = 5
        self.dt = 0.01
        self.sigma_w = 0.01
        self.x_min = 0.95
        self.x_max = 1.05
        self.ca = 0.1
        self.cd = 0.1
        self.seed = 42

class PowerSystemEnv:
    def __init__(self, cfg: PowerSystemConfig):
        self.cfg = cfg
        np.random.seed(cfg.seed)
        pos = np.random.rand(cfg.n, 2)
        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        Adj = (dist < 0.15).astype(float)
        np.fill_diagonal(Adj, 0.0)
        for i in range(1, cfg.n):
            j = np.random.randint(0, i)
            Adj[i, j] = Adj[j, i] = 1.0
        row_sums = Adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.Adj = Adj / row_sums
        self.Kpf = 0.5 * self.Adj
        self.r_coupling = 20
        self.V_coupling = np.random.randn(cfg.n, self.r_coupling) / np.sqrt(self.r_coupling)
        self.V_coupling, _ = qr(self.V_coupling, mode='economic')
        self.Ba_bar = np.random.randn(cfg.n, cfg.ra) / np.sqrt(cfg.ra)
        self.Bd_bar = np.random.randn(cfg.n, cfg.rd) / np.sqrt(cfg.rd)
        self.Wd = np.random.randn(cfg.rd, cfg.n) / np.sqrt(cfg.n)
        self.Cr = np.random.randn(cfg.rr, cfg.n) / np.sqrt(cfg.n)

    def g_power(self, x: np.ndarray) -> np.ndarray:
        z = self.V_coupling.T @ x
        diff_z = z[:, None] - z[None, :]
        nonlinear_z = np.sin(diff_z) + 0.1 * diff_z**2
        return self.Kpf @ (self.V_coupling @ nonlinear_z.sum(axis=1))

    def Ba(self, x: np.ndarray) -> np.ndarray:
        gate = 1.0 / (1.0 + np.exp(-(x - 0.5) * 10))
        return np.diag(gate) @ self.Ba_bar

    def Bd(self, x: np.ndarray) -> np.ndarray:
        logits = self.Wd @ x
        gate = np.exp(logits - logits.max())
        gate = gate / gate.sum()
        return self.Bd_bar * gate[None, :]

    def step(self, x: np.ndarray, a: np.ndarray, d: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        x_next = x + cfg.dt * self.g_power(x)
        x_next += self.Ba(x) @ a + self.Bd(x) @ d
        x_next += np.random.randn(cfg.n) * cfg.sigma_w
        return np.clip(x_next, cfg.x_min, cfg.x_max)

    def reward(self, x: np.ndarray, a: np.ndarray, d: np.ndarray) -> float:
        cfg = self.cfg
        return -np.sum((self.Cr @ x)**2) - cfg.ca*np.sum(a**2) - cfg.cd*np.sum(d**2)

    def sample_states(self, N: int) -> np.ndarray:
        samples = np.random.rand(N, self.cfg.n)
        for i in range(self.cfg.n):
            perm = np.random.permutation(N)
            samples[:, i] = (perm + samples[:, i]) / N
        return self.cfg.x_min + samples * (self.cfg.x_max - self.cfg.x_min)

# ==============================================================================
# 2. Low-Rank Value Network
# ==============================================================================

class LowRankValueNN:
    def __init__(self, n: int, m: int, r: int):
        self.n = n
        self.m = m
        self.r = r
        self.U = np.random.randn(m, r) / np.sqrt(r)
        self.V = np.random.randn(n, r) / np.sqrt(r)
        self.S = np.eye(r) * 0.1
        self.w_out = np.random.randn(m) / np.sqrt(m)
        self.U, _ = qr(self.U, mode='economic')
        self.V, _ = qr(self.V, mode='economic')

    def forward(self, x: np.ndarray) -> float:
        z = self.V.T @ x
        h = self.S @ z
        g = self.U @ h
        return float(self.w_out @ np.tanh(g))

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        Z = X @ self.V
        H = Z @ self.S.T
        G = H @ self.U.T
        return np.tanh(G) @ self.w_out

    def get_W(self) -> np.ndarray:
        return self.U @ self.S @ self.V.T

    def copy(self):
        net = LowRankValueNN(self.n, self.m, self.r)
        net.U = self.U.copy()
        net.V = self.V.copy()
        net.S = self.S.copy()
        net.w_out = self.w_out.copy()
        return net

# ==============================================================================
# 3. DLR-NE Trainer (simplified, no basis augmentation for speed)
# ==============================================================================

class DLRNETrainer:
    def __init__(self, env: PowerSystemEnv, n: int, m: int, r: int,
                 gamma: float = 0.95, lr: float = 0.02, beta: float = 0.1,
                 s_star: int = 10):
        self.env = env
        self.n = n
        self.m = m
        self.r = r
        self.gamma = gamma
        self.lr = lr
        self.beta = beta
        self.s_star = s_star
        self.value = LowRankValueNN(n, m, r)

    def spectral_regularizer(self, S: np.ndarray) -> float:
        alpha_sq = np.trace(S.T @ S) / S.shape[0]
        return np.linalg.norm(S.T @ S - alpha_sq * np.eye(S.shape[0]), 'fro')

    def grad_spectral(self, S: np.ndarray) -> np.ndarray:
        alpha_sq = np.trace(S.T @ S) / S.shape[0]
        E = S.T @ S - alpha_sq * np.eye(S.shape[0])
        R = np.linalg.norm(E, 'fro')
        if R < 1e-12:
            return np.zeros_like(S)
        return 2 * S @ E / R

    def bellman_target(self, x: np.ndarray) -> float:
        a = np.random.rand(self.env.cfg.ra)
        d = np.random.rand(self.env.cfg.rd)
        r = self.env.reward(x, a, d)
        x_next = self.env.step(x, a, d)
        return r + self.gamma * self.value.forward(x_next)

    def train_step(self, X_batch: np.ndarray) -> float:
        N = X_batch.shape[0]
        Y = np.array([self.bellman_target(x) for x in X_batch])

        Z = X_batch @ self.value.V
        H = Z @ self.value.S.T
        G = np.tanh(H @ self.value.U.T)
        v_pred = G @ self.value.w_out

        delta = v_pred - Y
        Phi_prime = 1 - G**2
        delta_h = (delta[:, None] * self.value.w_out[None, :]) * Phi_prime

        grad_w = G.T @ delta / N
        grad_S = (self.value.U.T @ delta_h.T) @ (X_batch @ self.value.V) / N
        grad_S += self.beta * self.grad_spectral(self.value.S)

        self.value.w_out -= self.lr * grad_w
        self.value.S -= self.lr * grad_S
        return np.mean(delta**2)

    def train(self, K: int = 100, N_b: int = 512) -> None:
        for k in range(K):
            X = self.env.sample_states(N_b)
            mse = self.train_step(X)
            if k % 20 == 0:
                print(f"    Train iter {k:3d}: MSE = {mse:.4f}, reg = {self.spectral_regularizer(self.value.S):.4f}")

# ==============================================================================
# 4. Policy Extractor (Greedy Nash)
# ==============================================================================

class GreedyNashPolicy:
    def __init__(self, env: PowerSystemEnv, n_action_samples: int = 15):
        self.env = env
        self.n_action_samples = n_action_samples
        self.a_grid = np.linspace(0, 1, n_action_samples)
        self.d_grid = np.linspace(0, 1, n_action_samples)

    def extract(self, x: np.ndarray, value_fn: LowRankValueNN) -> tuple:
        best_val = -np.inf
        best_pair = (np.zeros(self.env.cfg.ra), np.zeros(self.env.cfg.rd))
        for d_val in self.d_grid:
            d = np.ones(self.env.cfg.rd) * d_val
            worst_for_d = np.inf
            worst_a = None
            for a_val in self.a_grid:
                a = np.ones(self.env.cfg.ra) * a_val
                r = self.env.reward(x, a, d)
                x_next = self.env.step(x, a, d)
                q = r + 0.95 * value_fn.forward(x_next)
                if q < worst_for_d:
                    worst_for_d = q
                    worst_a = a
            if worst_for_d > best_val:
                best_val = worst_for_d
                best_pair = (worst_a, d)
        return best_pair

    def evaluate_policy(self, value_fn: LowRankValueNN, n_test: int = 200) -> float:
        X_test = self.env.sample_states(n_test)
        utilities = []
        for x in X_test:
            a, d = self.extract(x, value_fn)
            x_roll = x.copy()
            util = 0.0
            for t in range(5):
                r = self.env.reward(x_roll, a, d)
                util += (0.95**t) * r
                x_roll = self.env.step(x_roll, a, d)
                a, d = self.extract(x_roll, value_fn)
            utilities.append(util)
        return np.mean(utilities)

# ==============================================================================
# 5. Perturbation, Condition Number, Sensitivity Fit
# ==============================================================================

def inject_perturbation(value_fn: LowRankValueNN, delta_norm: float) -> LowRankValueNN:
    perturbed = value_fn.copy()
    Delta = np.random.randn(*perturbed.S.shape)
    Delta = Delta / (norm(Delta, 'fro') + 1e-12) * delta_norm
    perturbed.S += Delta
    return perturbed

def core_spectrum_stats(S: np.ndarray) -> dict:
    s = svd(S, compute_uv=False)
    alpha_sq = np.trace(S.T @ S) / S.shape[0]
    return {
        'kappa': s[0] / max(s[-1], 1e-12),
        'sigma_r': s[-1],
        'sigma_1': s[0],
        'R_spectral': norm(S.T @ S - alpha_sq * np.eye(S.shape[0]), 'fro'),
    }

def fit_L_kappa(delta_norms: list, value_deviations: list) -> tuple:
    """Origin-constrained linear fit:  L_κ = argmin_L Σ (ΔV_i - L·||ΔS||_i)² .
    Corollary 1 predicts ΔV = L_κ ||ΔS||_F (homogeneous), hence fit through origin."""
    x = np.asarray(delta_norms, dtype=float)
    y = np.asarray(value_deviations, dtype=float)
    L = float((x @ y) / (x @ x))
    y_hat = L * x
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
    return L, R2

# ==============================================================================
# 6. Main Experiment — single network at fixed β
# ==============================================================================

def run_single(rank: int, seed: int, beta: float = FIXED_BETA, K: int = TRAIN_ITERS) -> dict:
    """Train one DLR-NE network (rank r, seed) at the fixed operating point β,
    then measure its sensitivity L_κ and condition number κ(S)."""
    cfg = PowerSystemConfig()
    cfg.seed = seed
    np.random.seed(seed)
    env = PowerSystemEnv(cfg)

    print(f"\n  === r = {rank}, seed = {seed}, β = {beta} ===")
    trainer = DLRNETrainer(env, cfg.n, cfg.m, r=rank, gamma=0.95, lr=0.02, beta=beta, s_star=10)
    trainer.train(K=K, N_b=512)

    policy = GreedyNashPolicy(env, n_action_samples=15)
    U_base = policy.evaluate_policy(trainer.value, n_test=200)

    stats = core_spectrum_stats(trainer.value.S)
    print(f"  Base utility: {U_base:.4f}, κ(S) = {stats['kappa']:.2f}, "
          f"σ_r = {stats['sigma_r']:.4f}, R = {stats['R_spectral']:.4f}")

    res = {
        'rank': rank, 'seed': seed, 'beta': beta,
        'kappa': stats['kappa'], 'sigma_r': stats['sigma_r'],
        'sigma_1': stats['sigma_1'], 'R_spectral': stats['R_spectral'],
        'w_out_norm': float(norm(trainer.value.w_out)),
        'base_utility': U_base,
        'delta_norms': [], 'utility_deviations': [], 'value_deviations': [],
    }

    X_test = env.sample_states(500)
    V_base = np.array([trainer.value.forward(x) for x in X_test])

    for dn in PERTURBATION_NORMS:
        perturbed = inject_perturbation(trainer.value, dn)
        U_pert = policy.evaluate_policy(perturbed, n_test=200)
        V_pert = np.array([perturbed.forward(x) for x in X_test])
        delta_V = np.max(np.abs(V_base - V_pert))

        res['delta_norms'].append(dn)
        res['utility_deviations'].append(abs(U_base - U_pert))
        res['value_deviations'].append(delta_V)
        print(f"    ||ΔS||={dn:.3f}: ΔU={abs(U_base-U_pert):.4f}, ΔV={delta_V:.4f}")

    L_kappa, R2 = fit_L_kappa(res['delta_norms'], res['value_deviations'])
    res['L_kappa'] = L_kappa
    res['linearity_R2'] = R2
    print(f"  => L_κ = {L_kappa:.4f}  (fit R² = {R2:.4f}),  κ(S) = {stats['kappa']:.2f}")
    return res

# ==============================================================================
# 7. Visualization
# ==============================================================================

def plot_exp4(results_list: list, fit: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # ---- Left: ΔV vs ||ΔS||_F — per-network linearity at fixed β ----
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(RANKS)))
    rank_colors = {r: cmap[i] for i, r in enumerate(RANKS)}
    for res in results_list:
        label = f"$r={res['rank']}$, seed${res['seed']}$"
        ax.plot(res['delta_norms'], res['value_deviations'], 'o-', markersize=6,
                linewidth=1.8, color=rank_colors[res['rank']],
                alpha=0.55 if res['seed'] != SEEDS[0] else 1.0, label=label)
    ax.set_xlabel(r'Perturbation norm $\|\Delta S\|_F$', fontsize=12)
    ax.set_ylabel(r'Value deviation $\Delta V$', fontsize=12)
    ax.set_title(rf'Per-Network Linearity ($\beta$ = {FIXED_BETA} fixed)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ---- Right: L_κ vs κ(S) across the network family — Corollary 1 ----
    ax = axes[1]
    kappas = np.array([res['kappa'] for res in results_list])
    Ls = np.array([res['L_kappa'] for res in results_list])
    for r in RANKS:
        idx = [i for i, res in enumerate(results_list) if res['rank'] == r]
        ax.plot(kappas[idx], Ls[idx], 'o', markersize=10, color=rank_colors[r],
                label=f'$r={r}$')
    # Linear fit L_κ = C_robust · κ(S)
    C_hat, intercept = np.polyfit(kappas, Ls, 1)
    x_ref = np.linspace(kappas.min(), kappas.max(), 100)
    ax.plot(x_ref, np.polyval([C_hat, intercept], x_ref), 'k--', alpha=0.6,
            label=f'fit: $L_\\kappa$ = {C_hat:.5f}·$\\kappa$ {"+" if intercept>=0 else "−"} {abs(intercept):.4f}')
    ax.set_xlabel(r'Condition number $\kappa(S)$', fontsize=12)
    ax.set_ylabel(r'Fitted sensitivity $L_\kappa$', fontsize=12)
    ax.set_title(rf'$L_\kappa \propto \kappa(S)$ within a network family ($\beta$ = {FIXED_BETA})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    os.makedirs('results_data', exist_ok=True)
    with open('results_data/exp4_results.json', 'w') as f:
        json.dump({'config': {'fixed_beta': FIXED_BETA, 'ranks': RANKS, 'seeds': SEEDS,
                              'perturbation_norms': PERTURBATION_NORMS},
                   'runs': results_list, 'family_fit': fit}, f, indent=2)
    plt.tight_layout()
    plt.savefig('exp4_robustness.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n  [Plot saved to: exp4_robustness.png]")

# ==============================================================================
# 8. Main Entry
# ==============================================================================

def main():
    print("=" * 70)
    print("  Exp. 4 (v2): Game-Theoretic Robustness")
    print(f"  Fixed β = {FIXED_BETA}, ranks r = {RANKS}, seeds = {SEEDS}")
    print("  Tests  L_κ = C_robust · κ(S)  within a single network family")
    print("=" * 70)

    results_list = []
    for r in RANKS:
        for seed in SEEDS:
            results_list.append(run_single(rank=r, seed=seed))

    # Family-level fit: L_κ against κ(S) across all trained networks
    kappas = np.array([res['kappa'] for res in results_list])
    Ls = np.array([res['L_kappa'] for res in results_list])
    C_hat, intercept = np.polyfit(kappas, Ls, 1)
    ss_res = float(((Ls - np.polyval([C_hat, intercept], kappas)) ** 2).sum())
    ss_tot = float(((Ls - Ls.mean()) ** 2).sum())
    R2 = 1.0 - ss_res / ss_tot
    fit = {'C_robust_empirical': float(C_hat), 'intercept': float(intercept),
           'family_fit_R2': float(R2)}
    print("\n  === Family-level fit: L_κ = C·κ(S) ===")
    print(f"  C_robust (empirical) = {C_hat:.5f},  intercept = {intercept:.4f},  R² = {R2:.4f}")

    plot_exp4(results_list, fit)

    print("\n" + "=" * 70)
    print("  Exp. 4 (v2) completed successfully.")
    print("=" * 70)

if __name__ == '__main__':
    main()