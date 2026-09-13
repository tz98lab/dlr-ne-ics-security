#!/usr/bin/env python3
"""
================================================================================
Exp. 4: Game-Theoretic Robustness
Validates Corollary 1:  ||V(π_D) - V(π̃_D)||_∞ ≤ L_κ ||ΔS||_F
                         L_κ = C_robust · κ(S)
                         L_κ ≤ C_robust · exp( R(S)/(√2 σ_r(S)^2) )
================================================================================

This script:
1. Trains DLR-AVI to convergence with varying spectral regularization β.
2. Injects controlled Frobenius-norm perturbations ΔS on the core matrix.
3. Measures value-function deviation and condition number.
4. Verifies the linear relation ΔV ∝ κ(S) and the compression effect of β.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, svd, norm
import warnings
warnings.filterwarnings('ignore')

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
        """Deep copy."""
        net = LowRankValueNN(self.n, self.m, self.r)
        net.U = self.U.copy()
        net.V = self.V.copy()
        net.S = self.S.copy()
        net.w_out = self.w_out.copy()
        return net

# ==============================================================================
# 3. DLR-AVI Trainer (simplified, no basis augmentation for speed)
# ==============================================================================

class DLRAVITrainer:
    def __init__(self, env: PowerSystemEnv, n: int, m: int, r: int,
                 gamma: float = 0.95, lr: float = 0.02, beta: float = 0.01,
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

        # Forward
        Z = X_batch @ self.value.V
        H = Z @ self.value.S.T
        G = np.tanh(H @ self.value.U.T)
        v_pred = G @ self.value.w_out

        # Backward
        delta = v_pred - Y
        Phi_prime = 1 - G**2
        delta_h = (delta[:, None] * self.value.w_out[None, :]) * Phi_prime

        grad_w = G.T @ delta / N
        grad_S = (self.value.U.T @ delta_h.T) @ (X_batch @ self.value.V) / N

        # Spectral regularization
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
    def __init__(self, env: PowerSystemEnv, n_action_samples: int = 20):
        self.env = env
        self.n_action_samples = n_action_samples
        self.a_grid = np.linspace(0, 1, n_action_samples)
        self.d_grid = np.linspace(0, 1, n_action_samples)

    def extract(self, x: np.ndarray, value_fn: LowRankValueNN) -> tuple:
        """Return (a_star, d_star) for state x."""
        best_val = -np.inf
        best_pair = (np.zeros(self.env.cfg.ra), np.zeros(self.env.cfg.rd))

        for d_val in self.d_grid:
            d = np.ones(self.env.cfg.rd) * d_val
            worst_for_d = np.inf
            worst_a = None
            for a_val in self.a_grid:
                a = np.ones(self.env.cfg.ra) * a_val
                # One-step Q (single-sample MC for speed)
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

    def evaluate_policy(self, value_fn: LowRankValueNN, n_test: int = 500) -> float:
        """Average utility over test states."""
        X_test = self.env.sample_states(n_test)
        utilities = []
        for x in X_test:
            a, d = self.extract(x, value_fn)
            # Rollout for 5 steps to estimate utility
            x_roll = x.copy()
            util = 0.0
            for t in range(5):
                r = self.env.reward(x_roll, a, d)
                util += (0.95**t) * r
                x_roll = self.env.step(x_roll, a, d)
                # Re-extract policy at new state
                a, d = self.extract(x_roll, value_fn)
            utilities.append(util)
        return np.mean(utilities)

# ==============================================================================
# 5. Perturbation and Evaluation
# ==============================================================================

def inject_perturbation(value_fn: LowRankValueNN, delta_norm: float) -> LowRankValueNN:
    """Create perturbed network with ||ΔS||_F = delta_norm."""
    perturbed = value_fn.copy()
    # Random Frobenius-norm perturbation on S
    Delta = np.random.randn(*perturbed.S.shape)
    Delta = Delta / (norm(Delta, 'fro') + 1e-12) * delta_norm
    perturbed.S += Delta
    return perturbed

def condition_number(S: np.ndarray) -> float:
    s = svd(S, compute_uv=False)
    return s[0] / max(s[-1], 1e-12)

# ==============================================================================
# 6. Main Experiment
# ==============================================================================

def run_robustness_experiment(cfg: PowerSystemConfig, beta: float, K: int = 100):
    """Train DLR-AVI with given beta, then test robustness."""
    np.random.seed(cfg.seed)
    env = PowerSystemEnv(cfg)

    print(f"\n  === Training with β = {beta} ===")
    trainer = DLRAVITrainer(env, cfg.n, cfg.m, r=20, gamma=0.95, lr=0.02, beta=beta, s_star=10)
    trainer.train(K=K, N_b=512)

    # Evaluate base policy utility
    policy = GreedyNashPolicy(env, n_action_samples=15)
    U_base = policy.evaluate_policy(trainer.value, n_test=200)

    # Condition number of base S
    kappa_base = condition_number(trainer.value.S)
    print(f"  Base utility: {U_base:.4f}, Base κ(S): {kappa_base:.2f}")

    # Test perturbations
    delta_norms = [0.001, 0.005, 0.01, 0.05, 0.1]
    results = {
        'beta': beta,
        'kappa_base': kappa_base,
        'delta_norms': [],
        'utility_deviations': [],
        'value_deviations': []
    }

    for dn in delta_norms:
        perturbed = inject_perturbation(trainer.value, dn)
        U_pert = policy.evaluate_policy(perturbed, n_test=200)

        # Value function deviation (sup-norm approximation on test set)
        X_test = env.sample_states(500)
        V_base = np.array([trainer.value.forward(x) for x in X_test])
        V_pert = np.array([perturbed.forward(x) for x in X_test])
        delta_V = np.max(np.abs(V_base - V_pert))

        results['delta_norms'].append(dn)
        results['utility_deviations'].append(abs(U_base - U_pert))
        results['value_deviations'].append(delta_V)

        print(f"    ||ΔS||={dn:.3f}: ΔU={abs(U_base-U_pert):.4f}, ΔV={delta_V:.4f}")

    return results

# ==============================================================================
# 7. Visualization
# ==============================================================================

def plot_exp4(results_list: list):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Left: value deviation vs. perturbation norm
    ax = axes[0]
    for res in results_list:
        label = f"$\\beta={res['beta']:.2f}, \\kappa={res['kappa_base']:.1f}$"
        ax.plot(res['delta_norms'], res['value_deviations'], 'o-', markersize=8, linewidth=2, label=label)
    ax.set_xlabel(r'Perturbation norm $\|\Delta S\|_F$', fontsize=12)
    ax.set_ylabel(r'Value deviation $\Delta V$', fontsize=12)
    ax.set_title('Robustness: Value Deviation (Corollary 1)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Middle: condition number vs. beta
    ax = axes[1]
    betas = [res['beta'] for res in results_list]
    kappas = [res['kappa_base'] for res in results_list]
    ax.semilogy(betas, kappas, 'o-', color='#A23B72', markersize=10, linewidth=2)
    ax.set_xlabel(r'Spectral regularization weight $\beta$', fontsize=12)
    ax.set_ylabel(r'Condition number $\kappa(S)$', fontsize=12)
    ax.set_title('Condition Number vs. Regularization', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Right: fitted L_kappa vs. kappa(S)
    ax = axes[2]
    L_kappas = []
    kappa_vals = []
    for res in results_list:
        # Linear fit slope: delta_V / ||ΔS||_F
        slopes = [dv / max(dn, 1e-6) for dv, dn in zip(res['value_deviations'], res['delta_norms'])]
        L_kappa = np.mean(slopes[1:])  # exclude smallest perturbation (noisy)
        L_kappas.append(L_kappa)
        kappa_vals.append(res['kappa_base'])

    ax.plot(kappa_vals, L_kappas, 'o-', color='#2E86AB', markersize=10, linewidth=2)
    # Linear reference
    if len(kappa_vals) > 1:
        coeffs = np.polyfit(kappa_vals, L_kappas, 1)
        x_ref = np.linspace(min(kappa_vals), max(kappa_vals), 100)
        ax.plot(x_ref, np.polyval(coeffs, x_ref), 'k--', alpha=0.5, label=f'Linear fit: slope={coeffs[0]:.2f}')
    ax.set_xlabel(r'Condition number $\kappa(S)$', fontsize=12)
    ax.set_ylabel(r'Fitted $L_\kappa$', fontsize=12)
    ax.set_title(r'$L_\kappa \propto \kappa(S)$ Verification', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exp4_robustness.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n  [Plot saved to: exp4_robustness.png]")

# ==============================================================================
# 8. Main Entry
# ==============================================================================

def main():
    print("=" * 70)
    print("  Exp. 4: Game-Theoretic Robustness")
    print("  Validates Corollary 1 (Condition-Number-Controlled Robustness)")
    print("=" * 70)

    cfg = PowerSystemConfig()

    betas = [0.0, 0.01, 0.1, 1.0]
    results_list = []

    for beta in betas:
        res = run_robustness_experiment(cfg, beta=beta, K=80)
        results_list.append(res)

    plot_exp4(results_list)

    print("\n" + "=" * 70)
    print("  Exp. 4 completed successfully.")
    print("=" * 70)

if __name__ == '__main__':
    main()