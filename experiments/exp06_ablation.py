#!/usr/bin/env python3
"""
================================================================================
Exp. 6: Ablation—Necessity of Basis Augmentation
Validates Lemma 1 + Assumption 4(e'): basis augmentation is essential
================================================================================

This script compares three algorithmic variants:
1. Full Algorithm 1: basis augmentation + inner training + truncated-SVD retraction
2. Fixed-Basis: U,V frozen after init; only S optimized (no augmentation)
3. No-Retraction: basis augmentation + inner training, but NO truncation (rank stays 2r)

Expected:
- Fixed-Basis: Bellman residual stagnates at high plateau (local subspace mismatch)
- No-Retraction: residual decays but per-iteration cost doubles, unstable
- Full: stable decay at controlled cost
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, svd
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Environment
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
# 2. Low-Rank Network
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

# ==============================================================================
# 3. Bellman Target
# ==============================================================================

class BellmanTarget:
    def __init__(self, env: PowerSystemEnv, gamma: float = 0.95):
        self.env = env
        self.gamma = gamma

    def compute(self, X: np.ndarray, value_fn) -> np.ndarray:
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
# 4. Three Variants
# ==============================================================================

class VariantFull:
    """Full Algorithm 1: basis augmentation + inner loop + truncation."""
    def __init__(self, env, n, m, r, gamma=0.95, lr=0.02, beta=0.01, s_star=10):
        self.env = env
        self.value = LowRankValueNN(n, m, r)
        self.r = r
        self.gamma = gamma
        self.lr = lr
        self.beta = beta
        self.s_star = s_star
        self.bellman = BellmanTarget(env, gamma)

    def grad_spectral(self, S):
        alpha_sq = np.trace(S.T @ S) / S.shape[0]
        E = S.T @ S - alpha_sq * np.eye(S.shape[0])
        R = np.linalg.norm(E, 'fro')
        if R < 1e-12:
            return np.zeros_like(S)
        return 2 * S @ E / R

    def iteration(self, X_batch):
        value = self.value
        N = X_batch.shape[0]
        Y = self.bellman.compute(X_batch, value)

        # Forward
        Z = X_batch @ value.V
        H = Z @ value.S.T
        G = np.tanh(H @ value.U.T)
        v_pred = G @ value.w_out

        # Residual
        delta = v_pred - Y
        Phi_prime = 1 - G**2
        delta_h = (delta[:, None] * value.w_out[None, :]) * Phi_prime

        # Basis augmentation
        grad_U = delta_h.T @ H / N
        grad_V = X_batch.T @ (delta_h @ value.U @ value.S) / N
        U_aug = np.hstack([value.U, grad_U])
        V_aug = np.hstack([value.V, grad_V])
        U_hat, _ = qr(U_aug, mode='economic')
        V_hat, _ = qr(V_aug, mode='economic')
        S_0 = U_hat.T @ value.U @ value.S @ value.V.T @ V_hat

        # Inner loop
        S_hat = S_0.copy()
        w_out_hat = value.w_out.copy()
        for _ in range(self.s_star):
            Z2 = X_batch @ V_hat
            H2 = Z2 @ S_hat.T
            G2 = np.tanh(H2 @ U_hat.T)
            v_pred2 = G2 @ w_out_hat
            delta2 = v_pred2 - Y
            Phi_prime2 = 1 - G2**2
            delta_h2 = (delta2[:, None] * w_out_hat[None, :]) * Phi_prime2
            grad_w = G2.T @ delta2 / N
            grad_S = (U_hat.T @ delta_h2.T) @ (X_batch @ V_hat) / N
            grad_S += self.beta * self.grad_spectral(S_hat)
            w_out_hat -= self.lr * grad_w
            S_hat -= self.lr * grad_S

        # Truncation
        U_s, s, Vh_s = svd(S_hat, full_matrices=False)
        r_eff = min(self.r, np.sum(s > 1e-6))
        r_eff = max(1, r_eff)
        P = U_s[:, :r_eff]
        Sigma = np.diag(s[:r_eff])
        Q = Vh_s[:r_eff, :].T
        value.U = U_hat @ P
        value.V = V_hat @ Q
        value.S = Sigma
        if r_eff < self.r:
            pad = self.r - r_eff
            value.U = np.pad(value.U, ((0,0),(0,pad)))
            value.V = np.pad(value.V, ((0,0),(0,pad)))
            value.S = np.pad(value.S, ((0,pad),(0,pad)))
        value.w_out = w_out_hat

        # Return Bellman residual
        return np.max(np.abs(delta))

class VariantFixedBasis:
    """U,V frozen after init; only optimize S and w_out."""
    def __init__(self, env, n, m, r, gamma=0.95, lr=0.02, beta=0.01):
        self.env = env
        self.value = LowRankValueNN(n, m, r)
        self.gamma = gamma
        self.lr = lr
        self.beta = beta
        self.bellman = BellmanTarget(env, gamma)

    def grad_spectral(self, S):
        alpha_sq = np.trace(S.T @ S) / S.shape[0]
        E = S.T @ S - alpha_sq * np.eye(S.shape[0])
        R = np.linalg.norm(E, 'fro')
        if R < 1e-12:
            return np.zeros_like(S)
        return 2 * S @ E / R

    def iteration(self, X_batch):
        value = self.value
        N = X_batch.shape[0]
        Y = self.bellman.compute(X_batch, value)

        # Forward (fixed U,V)
        Z = X_batch @ value.V
        H = Z @ value.S.T
        G = np.tanh(H @ value.U.T)
        v_pred = G @ value.w_out

        # Backward
        delta = v_pred - Y
        Phi_prime = 1 - G**2
        delta_h = (delta[:, None] * value.w_out[None, :]) * Phi_prime
        grad_w = G.T @ delta / N
        grad_S = (value.U.T @ delta_h.T) @ (X_batch @ value.V) / N
        grad_S += self.beta * self.grad_spectral(value.S)

        value.w_out -= self.lr * grad_w
        value.S -= self.lr * grad_S

        return np.max(np.abs(delta))

class VariantNoRetraction:
    """Basis augmentation + inner loop, but NO truncation (rank stays 2r)."""
    def __init__(self, env, n, m, r, gamma=0.95, lr=0.02, beta=0.01, s_star=10):
        self.env = env
        # Initialize at rank 2r
        self.value = LowRankValueNN(n, m, 2*r)
        self.r = r
        self.gamma = gamma
        self.lr = lr
        self.beta = beta
        self.s_star = s_star
        self.bellman = BellmanTarget(env, gamma)

    def grad_spectral(self, S):
        alpha_sq = np.trace(S.T @ S) / S.shape[0]
        E = S.T @ S - alpha_sq * np.eye(S.shape[0])
        R = np.linalg.norm(E, 'fro')
        if R < 1e-12:
            return np.zeros_like(S)
        return 2 * S @ E / R

    def iteration(self, X_batch):
        value = self.value
        N = X_batch.shape[0]
        Y = self.bellman.compute(X_batch, value)

        # Forward
        Z = X_batch @ value.V
        H = Z @ value.S.T
        G = np.tanh(H @ value.U.T)
        v_pred = G @ value.w_out

        # Residual
        delta = v_pred - Y
        Phi_prime = 1 - G**2
        delta_h = (delta[:, None] * value.w_out[None, :]) * Phi_prime

        # Basis augmentation (to rank 4r)
        grad_U = delta_h.T @ H / N
        grad_V = X_batch.T @ (delta_h @ value.U @ value.S) / N
        U_aug = np.hstack([value.U, grad_U])
        V_aug = np.hstack([value.V, grad_V])
        U_hat, _ = qr(U_aug, mode='economic')
        V_hat, _ = qr(V_aug, mode='economic')
        S_0 = U_hat.T @ value.U @ value.S @ value.V.T @ V_hat

        # Inner loop (NO truncation, keep rank 4r)
        S_hat = S_0.copy()
        w_out_hat = value.w_out.copy()
        for _ in range(self.s_star):
            Z2 = X_batch @ V_hat
            H2 = Z2 @ S_hat.T
            G2 = np.tanh(H2 @ U_hat.T)
            v_pred2 = G2 @ w_out_hat
            delta2 = v_pred2 - Y
            Phi_prime2 = 1 - G2**2
            delta_h2 = (delta2[:, None] * w_out_hat[None, :]) * Phi_prime2
            grad_w = G2.T @ delta2 / N
            grad_S = (U_hat.T @ delta_h2.T) @ (X_batch @ V_hat) / N
            grad_S += self.beta * self.grad_spectral(S_hat)
            w_out_hat -= self.lr * grad_w
            S_hat -= self.lr * grad_S

        # NO truncation: directly assign back (rank grows)
        value.U = U_hat
        value.V = V_hat
        value.S = S_hat
        value.w_out = w_out_hat

        return np.max(np.abs(delta))

# ==============================================================================
# 5. Main Experiment
# ==============================================================================

def main():
    print("=" * 70)
    print("  Exp. 6: Ablation—Necessity of Basis Augmentation")
    print("  Validates Lemma 1 + Assumption 4(e')")
    print("=" * 70)

    cfg = PowerSystemConfig()
    env = PowerSystemEnv(cfg)
    n, m, r = cfg.n, cfg.m, 10
    K = 100
    N_b = 256

    variants = {
        'Full Algorithm 1': VariantFull(env, n, m, r, s_star=10),
        'Fixed-Basis': VariantFixedBasis(env, n, m, r),
        'No-Retraction': VariantNoRetraction(env, n, m, r, s_star=10),
    }

    results = {name: [] for name in variants}

    print(f"\n  Config: n={n}, r={r}, K={K}, N_b={N_b}")
    print(f"\n  Running {len(variants)} variants...\n")

    for name, variant in variants.items():
        print(f"  --- {name} ---")
        for k in range(K):
            X = env.sample_states(N_b)
            br = variant.iteration(X)
            results[name].append(br)
            if k % 20 == 0:
                print(f"    Iter {k:3d}: BR = {br:.4f}")

    # Plot
    plot_exp6(results)

    print("\n" + "=" * 70)
    print("  Exp. 6 completed successfully.")
    print("=" * 70)

# ==============================================================================
# 6. Visualization
# ==============================================================================

def plot_exp6(results: dict):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    colors = {'Full Algorithm 1': '#2E86AB', 'Fixed-Basis': '#A23B72', 'No-Retraction': '#F18F01'}
    styles = {'Full Algorithm 1': '-o', 'Fixed-Basis': '-s', 'No-Retraction': '-^'}

    for name, brs in results.items():
        iters = np.arange(len(brs))
        ax.semilogy(iters, brs, styles[name], color=colors[name],
                    linewidth=2, markersize=5, markevery=10, label=name)

    ax.set_xlabel('Iteration $k$', fontsize=13)
    ax.set_ylabel('Bellman residual', fontsize=13)
    ax.set_title('Ablation: Necessity of Basis Augmentation (Exp. 6)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # Annotations
    ax.annotate('Stagnates:\nsubspace mismatch', xy=(80, results['Fixed-Basis'][80]),
                xytext=(60, max(results['Fixed-Basis'])*2),
                fontsize=10, color=colors['Fixed-Basis'],
                arrowprops=dict(arrowstyle='->', color=colors['Fixed-Basis']))

    ax.annotate('Unstable:\nover-parameterized', xy=(80, results['No-Retraction'][80]),
                xytext=(60, min(results['No-Retraction'])*0.5),
                fontsize=10, color=colors['No-Retraction'],
                arrowprops=dict(arrowstyle='->', color=colors['No-Retraction']))

    plt.tight_layout()
    plt.savefig('exp6_ablation.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n  [Plot saved to: exp6_ablation.png]")

if __name__ == '__main__':
    main()