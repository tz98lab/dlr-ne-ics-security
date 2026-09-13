#!/usr/bin/env python3
"""
================================================================================
Exp. 2: Convergence of DLR-AVI
Validates Theorem 2:  geometric convergence rate gamma^k + steady-state error
================================================================================

This script:
1. Implements the full Algorithm 1 (basis augmentation, inner-loop training,
   truncated-SVD retraction, spectral regularization).
2. Runs DLR-AVI for K outer iterations and records the value-function sequence.
3. Runs FC-NN-AVI (full-rank, same settings) as the convergence reference.
4. Plots relative convergence error vs. iteration on semi-log scale.
5. Varies (N_b, s*, beta) to study sensitivity.

Note: The full-rank AVI convergence value serves as V_ref. The geometric decay
slope is compared against the theoretical prediction gamma^k.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, svd
import warnings
from typing import Optional, List, Tuple
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Environment (same as Exp. 1, with low-rank coupling subspace)
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
        
        # Low-rank coupling subspace (induces physical low-rank structure)
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
# 2. Full-Rank Value Network
# ==============================================================================

class FullRankValueNN:
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.W = np.random.randn(m, n) / np.sqrt(n)
        self.w_out = np.random.randn(m) / np.sqrt(m)

    def forward(self, x: np.ndarray) -> float:
        return float(self.w_out @ np.tanh(self.W @ x))

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.W.T) @ self.w_out

# ==============================================================================
# 3. Low-Rank Value Network
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

# ==============================================================================
# 4. Bellman Target Computer
# ==============================================================================

class BellmanTarget:
    def __init__(self, env: PowerSystemEnv, gamma: float = 0.95):
        self.env = env
        self.gamma = gamma

    def compute_targets(self, X: np.ndarray, value_fn) -> np.ndarray:
        """Compute Bellman targets T V(x) for batch X using random actions."""
        N = X.shape[0]
        Y = np.zeros(N)
        for i in range(N):
            x = X[i]
            a = np.random.rand(self.env.cfg.ra)
            d = np.random.rand(self.env.cfg.rd)
            r = self.env.reward(x, a, d)
            x_next = self.env.step(x, a, d)
            Y[i] = r + self.gamma * value_fn.forward(x_next)
        return Y

# ==============================================================================
# 5. DLR-AVI (Algorithm 1)
# ==============================================================================

class DLRAVI:
    def __init__(self, env: PowerSystemEnv, n: int, m: int, r: int,
                 gamma: float = 0.95, lr: float = 0.02, beta: float = 0.01,
                 s_star: int = 10, trunc_thresh: float = 1e-6):
        self.env = env
        self.n = n
        self.m = m
        self.r = r
        self.gamma = gamma
        self.lr = lr
        self.beta = beta
        self.s_star = s_star
        self.trunc_thresh = trunc_thresh
        self.value = LowRankValueNN(n, m, r)
        self.bellman = BellmanTarget(env, gamma)

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

    def basis_augmentation(self, X_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Steps 6-7: augment basis to rank 2r."""
        value = self.value
        N = X_batch.shape[0]

        # Forward
        Z = X_batch @ value.V
        H = Z @ value.S.T
        G = H @ value.U.T
        Phi = np.tanh(G)
        Phi_prime = 1 - Phi**2

        # Bellman targets
        Y = self.bellman.compute_targets(X_batch, value)

        # Residual
        v_pred = Phi @ value.w_out
        delta = v_pred - Y

        # Gradients w.r.t. U and V
        delta_h = (delta[:, None] * value.w_out[None, :]) * Phi_prime
        grad_U = delta_h.T @ H / N
        grad_V = X_batch.T @ (delta_h @ value.U @ value.S) / N

        # Augment and orthonormalize
        U_aug = np.hstack([value.U, grad_U])
        V_aug = np.hstack([value.V, grad_V])
        U_hat, _ = qr(U_aug, mode='economic')
        V_hat, _ = qr(V_aug, mode='economic')

        # Coefficient augmentation
        S_0 = U_hat.T @ value.U @ value.S @ value.V.T @ V_hat

        return U_hat, V_hat, S_0

    def inner_loop(self, U_hat: np.ndarray, V_hat: np.ndarray, S_0: np.ndarray,
                   X_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Steps 14-19: joint optimization of (S_hat, w_out_hat)."""
        N = X_batch.shape[0]
        S_hat = S_0.copy()
        w_out_hat = self.value.w_out.copy()

        for s in range(self.s_star):
            # Forward on augmented basis
            Z = X_batch @ V_hat
            H = Z @ S_hat.T
            G = H @ U_hat.T
            Phi = np.tanh(G)
            v_pred = Phi @ w_out_hat

            # Target
            Y = self.bellman.compute_targets(X_batch, self.value)

            # Loss gradient
            delta = v_pred - Y
            Phi_prime = 1 - Phi**2
            delta_h = (delta[:, None] * w_out_hat[None, :]) * Phi_prime

            grad_w = Phi.T @ delta / N
            grad_S = (U_hat.T @ delta_h.T) @ (X_batch @ V_hat) / N

            # Spectral regularization
            grad_S += self.beta * self.grad_spectral(S_hat)

            # Update
            w_out_hat -= self.lr * grad_w
            S_hat -= self.lr * grad_S

        return S_hat, w_out_hat

    def truncate_and_retract(self, U_hat: np.ndarray, V_hat: np.ndarray, S_hat: np.ndarray) -> None:
        """Steps 22-23: truncated SVD retraction to rank r."""
        U_s, s, Vh_s = svd(S_hat, full_matrices=False)
        r_eff = min(self.r, np.sum(s > self.trunc_thresh))
        r_eff = max(1, r_eff)

        P = U_s[:, :r_eff]
        Sigma = np.diag(s[:r_eff])
        Q = Vh_s[:r_eff, :].T

        self.value.U = U_hat @ P
        self.value.V = V_hat @ Q
        self.value.S = Sigma

        if r_eff < self.r:
            pad = self.r - r_eff
            self.value.U = np.pad(self.value.U, ((0, 0), (0, pad)))
            self.value.V = np.pad(self.value.V, ((0, 0), (0, pad)))
            self.value.S = np.pad(self.value.S, ((0, pad), (0, pad)))

    def iteration(self, X_batch: np.ndarray) -> float:
        """One full outer iteration of Algorithm 1."""
        U_hat, V_hat, S_0 = self.basis_augmentation(X_batch)
        S_hat, w_out_hat = self.inner_loop(U_hat, V_hat, S_0, X_batch)
        self.truncate_and_retract(U_hat, V_hat, S_hat)
        self.value.w_out = w_out_hat
        return self.spectral_regularizer(self.value.S)

# ==============================================================================
# 6. FC-NN-AVI Baseline
# ==============================================================================

class FCNNAVI:
    def __init__(self, env: PowerSystemEnv, n: int, m: int,
                 gamma: float = 0.95, lr: float = 0.02):
        self.env = env
        self.n = n
        self.m = m
        self.gamma = gamma
        self.lr = lr
        self.value = FullRankValueNN(n, m)
        self.bellman = BellmanTarget(env, gamma)

    def iteration(self, X_batch: np.ndarray) -> None:
        """One AVI iteration with full-rank network."""
        N = X_batch.shape[0]
        Y = self.bellman.compute_targets(X_batch, self.value)

        # Forward
        G = np.tanh(X_batch @ self.value.W.T)
        v_pred = G @ self.value.w_out

        # Backprop
        delta = v_pred - Y
        Phi_prime = 1 - G**2

        grad_w = G.T @ delta / N
        grad_W = ((delta[:, None] * self.value.w_out[None, :] * Phi_prime).T @ X_batch) / N

        self.value.w_out -= self.lr * grad_w
        self.value.W -= self.lr * grad_W

# ==============================================================================
# 7. Evaluation Utilities
# ==============================================================================

def evaluate_value_error(value_fn, ref_fn, env: PowerSystemEnv, n_test: int = 2000) -> float:
    """Compute sup-norm error ||V - V_ref||_infty on test set."""
    X_test = env.sample_states(n_test)
    V_test = np.array([value_fn.forward(x) for x in X_test])
    V_ref = np.array([ref_fn.forward(x) for x in X_test])
    return np.max(np.abs(V_test - V_ref))

# ==============================================================================
# 8. Main Experiment
# ==============================================================================

def run_convergence_experiment(cfg: PowerSystemConfig, K: int = 100,
                                r: int = 20, N_b: int = 512,
                                s_star: int = 10, beta: float = 0.01,
                                lr: float = 0.02) -> dict:
    """Run DLR-AVI for K iterations and record convergence."""
    np.random.seed(cfg.seed)
    env = PowerSystemEnv(cfg)

    print(f"\n  Config: n={cfg.n}, r={r}, N_b={N_b}, s*={s_star}, beta={beta}, lr={lr}")

    # Initialize DLR-AVI
    dlr = DLRAVI(env, cfg.n, cfg.m, r, gamma=0.95, lr=lr, beta=beta,
                 s_star=s_star, trunc_thresh=1e-6)

    # Initialize FC-NN-AVI baseline (same settings, full-rank)
    fc = FCNNAVI(env, cfg.n, cfg.m, gamma=0.95, lr=lr)

    # Pre-train FC baseline for a few iterations to get a stable reference
    print("  Pre-training FC-NN-AVI baseline...")
    for it in range(50):
        X = env.sample_states(N_b)
        fc.iteration(X)
        if it % 20 == 0:
            print(f"    FC iter {it}")

    V_ref = fc.value

    # Run DLR-AVI
    print(f"  Running DLR-AVI for {K} iterations...")
    errors = []
    bellman_residuals = []
    condition_numbers = []

    for k in range(K):
        X = env.sample_states(N_b)

        # DLR-AVI step
        reg_val = dlr.iteration(X)

        # Evaluate
        err = evaluate_value_error(dlr.value, V_ref, env, n_test=1000)
        errors.append(err)

        # Bellman residual (approximate: ||T V_k - V_k||)
        X_br = env.sample_states(200)
        V_current = np.array([dlr.value.forward(x) for x in X_br])
        Y_target = dlr.bellman.compute_targets(X_br, dlr.value)
        br = np.max(np.abs(Y_target - V_current))
        bellman_residuals.append(br)

        # Condition number
        s_svd = svd(dlr.value.S, compute_uv=False)
        kappa = s_svd[0] / max(s_svd[-1], 1e-12)
        condition_numbers.append(kappa)

        if k % 20 == 0:
            print(f"    DLR iter {k:3d}: error={err:.4f}, BR={br:.4f}, kappa={kappa:.2f}, reg={reg_val:.4f}")

    return {
        'errors': np.array(errors),
        'bellman_residuals': np.array(bellman_residuals),
        'condition_numbers': np.array(condition_numbers),
        'gamma': 0.95,
        'K': K
    }

# ==============================================================================
# 9. Visualization
# ==============================================================================

def plot_exp2(results_list: list, labels: list, gamma: float = 0.95):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Left: relative convergence error
    ax = axes[0]
    for res, label in zip(results_list, labels):
        err_rel = res['errors'] / (res['errors'][0] + 1e-6)
        ax.semilogy(np.arange(res['K']), err_rel, '-o', markersize=3, label=label)
    # Theoretical envelope
    K_max = max([res['K'] for res in results_list])
    k_theory = np.arange(K_max)
    ax.semilogy(k_theory, gamma**k_theory, 'k--', linewidth=1.5, alpha=0.7, label=r'$\gamma^k$')
    ax.set_xlabel('Iteration $k$', fontsize=12)
    ax.set_ylabel(r'Relative $\|V^{(k)} - V_{\mathrm{ref}}\|_\infty$', fontsize=12)
    ax.set_title('Convergence of DLR-AVI (Theorem 2)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Middle: Bellman residual
    ax = axes[1]
    for res, label in zip(results_list, labels):
        ax.semilogy(np.arange(res['K']), res['bellman_residuals'], '-o', markersize=3, label=label)
    ax.set_xlabel('Iteration $k$', fontsize=12)
    ax.set_ylabel('Bellman residual', fontsize=12)
    ax.set_title('Bellman Residual', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Right: condition number
    ax = axes[2]
    for res, label in zip(results_list, labels):
        ax.semilogy(np.arange(res['K']), res['condition_numbers'], '-o', markersize=3, label=label)
    ax.set_xlabel('Iteration $k$', fontsize=12)
    ax.set_ylabel(r'Condition number $\kappa(S^{(k)})$', fontsize=12)
    ax.set_title('Condition Number Trajectory', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('exp2_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n  [Plot saved to: exp2_convergence.png]")

# ==============================================================================
# 10. Main Entry
# ==============================================================================

def main():
    print("=" * 70)
    print("  Exp. 2: Convergence of DLR-AVI")
    print("  Validates Theorem 2 (Geometric Convergence + Steady-State Error)")
    print("=" * 70)

    cfg = PowerSystemConfig()

    # -------------------------------------------------------------------------
    # Run three configurations to study sensitivity
    # -------------------------------------------------------------------------
    configs = [
        {'N_b': 512, 's_star': 10, 'beta': 0.01, 'label': r'$N_b=512, s^*=10, \beta=0.01$'},
        {'N_b': 128, 's_star': 10, 'beta': 0.01, 'label': r'$N_b=128, s^*=10, \beta=0.01$'},
        {'N_b': 512, 's_star': 5,  'beta': 0.01, 'label': r'$N_b=512, s^*=5, \beta=0.01$'},
        {'N_b': 512, 's_star': 10, 'beta': 0.0,  'label': r'$N_b=512, s^*=10, \beta=0$'},
    ]

    results_list = []
    for c in configs:
        print(f"\n--- Running: {c['label']} ---")
        res = run_convergence_experiment(
            cfg, K=100, r=20,
            N_b=c['N_b'], s_star=c['s_star'], beta=c['beta']
        )
        results_list.append(res)

    # Plot
    plot_exp2(results_list, [c['label'] for c in configs])

    print("\n" + "=" * 70)
    print("  Exp. 2 completed successfully.")
    print("=" * 70)

if __name__ == '__main__':
    main()