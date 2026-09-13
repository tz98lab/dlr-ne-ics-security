#!/usr/bin/env python3
"""
================================================================================
Exp. 1: Low-Rank Truncation Error
Validates Theorem 1:  ||V_full - V_hat_r||_infty  vs.  EYM theoretical bound
================================================================================

This script:
1. Trains a full-rank single-hidden-layer NN on synthetic Bellman targets.
2. Performs truncated SVD on its hidden-layer weight W_full at ranks r.
3. Constructs low-rank networks V_hat_r and evaluates sup-norm error.
4. Compares measured error against the EYM theoretical prediction.
5. Plots the truncation error curve and the singular-value spectrum.

Note: The full-rank network serves as a proxy benchmark. The additive gap
||V* - V_full||_infty = epsilon_NN is independent of r and omitted from the
trend comparison, consistent with the experimental philosophy of §7.1.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, svd
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Synthetic Power-System Environment (inline for self-containment)
# ==============================================================================

class PowerSystemConfig:
    def __init__(self):
        self.n = 200              # state dimension (use 1000 for production)
        self.m = 200              # hidden width (use 1000 for production)
        self.ra = 3               # attack channels
        self.rd = 5               # defense channels
        self.rr = 5               # reward projection dim
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
        # Random geometric graph adjacency
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
         # === 低秩耦合子空间：动力学只通过 r_coupling 个主导方向传播 ===
        self.r_coupling = 20       # 耦合子空间维度（如 20 个区域联络通道）
        self.V_coupling = np.random.randn(cfg.n, self.r_coupling) / np.sqrt(self.r_coupling)
        self.V_coupling, _ = qr(self.V_coupling, mode='economic')  # 列正交
        # ================================================================
        
        # Low-rank skeletons
        self.Ba_bar = np.random.randn(cfg.n, cfg.ra) / np.sqrt(cfg.ra)
        self.Bd_bar = np.random.randn(cfg.n, cfg.rd) / np.sqrt(cfg.rd)
        self.Wd = np.random.randn(cfg.rd, cfg.n) / np.sqrt(cfg.n)
        self.Cr = np.random.randn(cfg.rr, cfg.n) / np.sqrt(cfg.n)

    def g_power(self, x: np.ndarray) -> np.ndarray:
        # 投影到低秩耦合子空间
        z = self.V_coupling.T @ x            # (r_coupling,)
        # 在低维子空间做非线性交互（物理上对应区域级耦合）
        diff_z = z[:, None] - z[None, :]
        nonlinear_z = np.sin(diff_z) + 0.1 * diff_z**2
        # 升维回 n 维：非线性效应只通过 r_coupling 个方向进入全网
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
# 2. Neural-Network Value Function Classes
# ==============================================================================

class FullRankValueNN:
    """V(x) = w_out^T tanh(W x).  W: (m, n), w_out: (m,)"""
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.W = np.random.randn(m, n) / np.sqrt(n)
        self.w_out = np.random.randn(m) / np.sqrt(m)

    def forward(self, x: np.ndarray) -> float:
        return float(self.w_out @ np.tanh(self.W @ x))

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.W.T) @ self.w_out


class LowRankValueNN:
    """V(x) = w_out^T tanh(U S V^T x).  U:(m,r), S:(r,r), V:(n,r)"""
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

    def set_from_svd(self, U_r, s_r, Vh_r, w_out):
        self.U = U_r
        self.S = np.diag(s_r)
        self.V = Vh_r.T
        self.w_out = w_out.copy()

# ==============================================================================
# 3. Training: Full-Rank Baseline
# ==============================================================================

def train_fullrank_baseline(env: PowerSystemEnv, n: int, m: int,
                            n_samples: int = 2000,
                            n_iters: int = 100,
                            lr: float = 0.02,
                            gamma: float = 0.95) -> FullRankValueNN:
    """
    Trains a full-rank network on synthetic Bellman targets.
    Target: y = r(x,a,d) + gamma * V(x') under random actions (bootstrapping).
    """
    net = FullRankValueNN(n, m)
    print(f"  Training full-rank net: n={n}, m={m}, params={net.W.size + net.w_out.size}")

    for it in range(n_iters):
        X = env.sample_states(n_samples)
        Y = np.zeros(n_samples)

        # Compute Bellman targets with current network (single-sample MC)
        for i in range(n_samples):
            x = X[i]
            a = np.random.rand(env.cfg.ra)
            d = np.random.rand(env.cfg.rd)
            r = env.reward(x, a, d)
            x_next = env.step(x, a, d)
            Y[i] = r + gamma * net.forward(x_next)

        # Batch forward
        G = np.tanh(X @ net.W.T)          # (N, m)
        V_pred = G @ net.w_out            # (N,)

        # Backprop
        delta = V_pred - Y                # (N,)
        Phi_prime = 1.0 - G**2            # (N, m)

        grad_w = G.T @ delta / n_samples
        grad_W = ((delta[:, None] * net.w_out[None, :] * Phi_prime).T @ X) / n_samples

        net.w_out -= lr * grad_w
        net.W -= lr * grad_W

        # === 诱导低秩结构：更强权重衰减 + 更频繁软阈值 ===
        # (1) 权重衰减增强50倍，更积极地压缩小权重方向
        net.W -= lr * 1e-3 * net.W
        
        # (2) 每5轮执行一次（更频繁），且从第10轮开始（更早）
        # 软阈值：每10轮压缩尾部，保护前40个主导方向
        if it % 10 == 0 and it > 20:
            U, s, Vh = svd(net.W, full_matrices=False)
            s = np.maximum(s - 0.02, 0)
            k = min(40, len(s))
            s[:k] = np.maximum(s[:k], 0.001)
            net.W = U @ np.diag(s) @ Vh
        # ================================================
        if it % 20 == 0:
            mse = np.mean(delta**2)
            print(f"    Iter {it:3d}: MSE = {mse:.4f}")

    return net

# ==============================================================================
# 4. Evaluation: Truncation Error vs. Rank
# ==============================================================================

def evaluate_truncation_error(V_full: FullRankValueNN,
                              env: PowerSystemEnv,
                              r_list: list,
                              n_test: int = 5000):
    """
    For each rank r in r_list:
      - Truncate W_full via SVD to rank r.
      - Build low-rank network V_hat_r.
      - Measure sup-norm error on test set.
      - Compute EYM error and theoretical bound.
    """
    W_full = V_full.W
    w_out = V_full.w_out
    _, s_full, _ = svd(W_full, full_matrices=False)

    # Constants for theoretical bound
    L_phi = 1.0                       # tanh is 1-Lipschitz
    R_X = env.cfg.x_max               # state-space radius
    w_norm = np.linalg.norm(w_out, 2)

    # Test set
    X_test = env.sample_states(n_test)
    V_full_test = np.array([V_full.forward(x) for x in X_test])

    results = {
        'r_list': r_list,
        'measured_error': [],
        'eym_error': [],
        'theoretical_bound': []
    }

    print(f"\n  {'Rank':>5s} | {'Measured':>12s} | {'EYM':>12s} | {'TheoryBound':>12s}")
    print("  " + "-"*55)

    for r in r_list:
        # Truncated SVD
        U_r, s_r, Vh_r = svd(W_full, full_matrices=False)
        U_r = U_r[:, :r]
        s_r = s_r[:r]
        Vh_r = Vh_r[:r, :]

        # Build low-rank network
        V_hat = LowRankValueNN(env.cfg.n, env.cfg.m, r)
        V_hat.set_from_svd(U_r, s_r, Vh_r, w_out)

        # Evaluate
        V_hat_test = np.array([V_hat.forward(x) for x in X_test])
        err_measured = np.max(np.abs(V_full_test - V_hat_test))

        # EYM error
        eym = np.sqrt(np.sum(s_full[r:]**2))

        # Theoretical bound (Theorem 1, EYM term only)
        bound = L_phi * w_norm * R_X * eym

        results['measured_error'].append(err_measured)
        results['eym_error'].append(eym)
        results['theoretical_bound'].append(bound)

        print(f"  {r:5d} | {err_measured:12.6f} | {eym:12.6f} | {bound:12.6f}")

    return results, s_full

# ==============================================================================
# 5. Visualization
# ==============================================================================

def plot_exp1(results: dict, singular_values: np.ndarray, cfg: PowerSystemConfig):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: truncation error vs. rank
    ax = axes[0]
    r_list = results['r_list']
    ax.semilogy(r_list, results['measured_error'],
                'o-', color='#2E86AB', linewidth=2, markersize=8,
                label=r'Measured $\|V_{\mathrm{full}} - \widehat{V}_r\|_\infty$')
    ax.semilogy(r_list, results['theoretical_bound'],
                's--', color='#A23B72', linewidth=2, markersize=8,
                label=r'Theory: $L_\phi \|w_{\mathrm{out}}^*\|_2 R_{\mathcal{X}}\,\epsilon_{\mathrm{EYM}}(r)$')

    ax.set_xlabel('Truncation rank $r$', fontsize=13)
    ax.set_ylabel('Error', fontsize=13)
    ax.set_title('Low-Rank Truncation Error (Theorem 1)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks(r_list)

    # Right: singular value spectrum
    ax = axes[1]
    idx = np.arange(1, len(singular_values) + 1)
    ax.semilogy(idx, singular_values, color='#F18F01', linewidth=1.5)
    ax.axvline(x=max(r_list), color='gray', linestyle='--', alpha=0.6,
               label=f'Max tested $r={max(r_list)}$')
    ax.set_xlabel('Singular value index $i$', fontsize=13)
    ax.set_ylabel(r'Singular value $\sigma_i(W_{\mathrm{full}}^*)$', fontsize=13)
    ax.set_title('Singular Value Spectrum', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('exp1_truncation_error.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n  [Plot saved to: exp1_truncation_error.png]")

# ==============================================================================
# 6. Main Entry
# ==============================================================================

def main():
    print("=" * 70)
    print("  Exp. 1: Low-Rank Truncation Error")
    print("  Validates Theorem 1 (Global Low-Rank Approximation Error Bound)")
    print("=" * 70)

    cfg = PowerSystemConfig()
    # -------------------------------------------------------------------------
    # NOTE: For quick demonstration, n=200 is used. For full-scale validation
    # matching the paper's design (n=1000, m=1000), simply set:
    #   cfg.n = 1000
    #   cfg.m = 1000
    #   n_samples = 5000
    #   n_iters = 200
    #   n_test = 10000
    # -------------------------------------------------------------------------
    print(f"\n[Config] n={cfg.n}, m={cfg.m}, ra={cfg.ra}, rd={cfg.rd}")

    np.random.seed(cfg.seed)
    env = PowerSystemEnv(cfg)

    # Step 1: Train full-rank baseline
    print("\n[Step 1] Training full-rank baseline V_full...")
    V_full = train_fullrank_baseline(
        env, cfg.n, cfg.m,
        n_samples=2000,
        n_iters=100,
        lr=0.02,
        gamma=0.95
    )

    # Step 2: Evaluate truncation errors
    print("\n[Step 2] Evaluating low-rank truncation errors...")
    r_list = [5, 10, 20, 50, 100]
    r_list = [r for r in r_list if r <= min(cfg.n, cfg.m)]
    results, s_full = evaluate_truncation_error(V_full, env, r_list, n_test=5000)

    # Step 3: Plot
    print("\n[Step 3] Generating figure...")
    plot_exp1(results, s_full, cfg)

    print("\n" + "=" * 70)
    print("  Exp. 1 completed successfully.")
    print("=" * 70)


if __name__ == '__main__':
    main()