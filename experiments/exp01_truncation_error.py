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
||V* - V_full||_infty = epsilon_NN is independent of r; as a measurable
proxy we plot its sup-norm one-step Bellman residual as the epsilon_NN
floor, consistent with the experimental philosophy of Sec. 5.
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

    def reward_batch(self, X: np.ndarray, A: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Batch version of reward(): X (N,n), A (N,ra), D (N,rd) -> (N,)"""
        cfg = self.cfg
        return -np.sum((X @ self.Cr.T)**2, axis=1) \
               - cfg.ca*np.sum(A**2, axis=1) - cfg.cd*np.sum(D**2, axis=1)

    def step_batch(self, X: np.ndarray, A: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Vectorized version of step(): X (N,n), A (N,ra), D (N,rd) -> (N,n).
        Identical operations to step(), applied along the batch axis."""
        cfg = self.cfg
        # g_power, batched through the low-rank coupling subspace
        Z = X @ self.V_coupling                                  # (N, r_coupling)
        diff = Z[:, :, None] - Z[:, None, :]                     # (N, rc, rc)
        nonlinear = np.sin(diff) + 0.1 * diff**2
        s = nonlinear.sum(axis=2)                                # (N, rc)
        x_next = X + cfg.dt * (self.Kpf @ (self.V_coupling @ s.T)).T
        # attack channel: diag(gate) @ Ba_bar @ a
        gate_a = 1.0 / (1.0 + np.exp(-(X - 0.5) * 10))           # (N, n)
        x_next += gate_a * (A @ self.Ba_bar.T)
        # defense channel: Bd_bar * softmax(W_d x), then contract with d
        logits = X @ self.Wd.T                                   # (N, rd)
        logits -= logits.max(axis=1, keepdims=True)
        gate_d = np.exp(logits)
        gate_d /= gate_d.sum(axis=1, keepdims=True)              # (N, rd)
        x_next += (self.Bd_bar[None, :, :] * (gate_d * D)[:, None, :]).sum(axis=2)
        # noise + saturation
        x_next += np.random.randn(X.shape[0], cfg.n) * cfg.sigma_w
        return np.clip(x_next, cfg.x_min, cfg.x_max)

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
                            gamma: float = 0.95,
                            wd_scale: float = 1e-3,
                            thresh_every: int = 10,
                            thresh_start: int = 20,
                            thresh_shrink: float = 0.02,
                            thresh_protect: int = 40,
                            protect_floor: float = 1e-3) -> FullRankValueNN:
    """
    Trains a full-rank network on synthetic Bellman targets.
    Target: y = r(x,a,d) + gamma * V(x') under random actions (bootstrapping).

    Low-rank induction (spectral regularization of the benchmark):
      (1) weight decay wd_scale applied every iteration;
      (2) soft-thresholding of the singular spectrum every thresh_every
          iterations after thresh_start, shrinking all singular values by
          thresh_shrink and flooring the leading thresh_protect ones at
          protect_floor so the signal head survives the shrinkage.
    """
    net = FullRankValueNN(n, m)
    print(f"  Training full-rank net: n={n}, m={m}, params={net.W.size + net.w_out.size}")

    for it in range(n_iters):
        X = env.sample_states(n_samples)

        # Bellman targets, fully vectorized (same single-sample MC protocol)
        A = np.random.rand(n_samples, env.cfg.ra)
        D = np.random.rand(n_samples, env.cfg.rd)
        R = env.reward_batch(X, A, D)
        X_next = env.step_batch(X, A, D)
        Y = R + gamma * net.forward_batch(X_next)

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

        # === spectral regularization: weight decay + soft thresholding ===
        net.W -= lr * wd_scale * net.W

        if it % thresh_every == 0 and it > thresh_start:
            U, s, Vh = svd(net.W, full_matrices=False)
            s = np.maximum(s - thresh_shrink, 0)
            k = min(thresh_protect, len(s))
            s[:k] = np.maximum(s[:k], protect_floor)
            net.W = U @ np.diag(s) @ Vh
        # =================================================================
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
                              n_test: int = 5000,
                              gamma: float = 0.95,
                              n_residual: int = 500):
    """
    For each rank r in r_list:
      - Truncate W_full via SVD to rank r (no retraining).
      - Build low-rank network V_hat_r.
      - Measure sup-norm error on test set.
      - Compute EYM error and theoretical bound.
    Also computes the epsilon_NN floor: the sup-norm one-step Bellman
    residual of the full-rank benchmark, below which further truncation
    gains are meaningless.
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
    V_full_test = V_full.forward_batch(X_test)

    # --- epsilon_NN floor: sup-norm Bellman residual of the benchmark ---
    # MC-averaged targets (M action samples per state) to remove the
    # single-sample target noise; the floor then isolates the benchmark's
    # own approximation error, which is the meaningful saturation level
    # for the truncation-error curve.
    n_res = min(n_residual, n_test)
    M = 32
    resid = np.zeros(n_res)
    for i in range(n_res):
        x = X_test[i]
        ys = np.zeros(M)
        for j in range(M):
            a = np.random.rand(env.cfg.ra)
            d = np.random.rand(env.cfg.rd)
            ys[j] = env.reward(x, a, d) + gamma * V_full.forward(env.step(x, a, d))
        resid[i] = abs(V_full_test[i] - ys.mean())
    eps_NN = float(resid.max())

    results = {
        'r_list': r_list,
        'measured_error': [],
        'eym_error': [],
        'theoretical_bound': [],
        'eps_NN': eps_NN
    }

    print(f"\n  epsilon_NN floor (Bellman residual of V_full): {eps_NN:.6f}")
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

        # Evaluate (batched forward pass)
        V_hat_test = V_hat.forward_batch(X_test)
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

    # epsilon_NN floor: sup-norm Bellman residual of the full-rank benchmark
    eps_NN = results.get('eps_NN', None)
    if eps_NN is not None:
        ax.axhline(eps_NN, color='gray', linestyle=':', linewidth=2,
                   label=r'$\epsilon_{\mathrm{NN}}$ floor (Bellman residual of $V_{\mathrm{full}}$)')

    ax.set_xlabel('Truncation rank $r$', fontsize=13)
    ax.set_ylabel('Error', fontsize=13)
    ax.set_title('Low-Rank Truncation Error (Theorem 1)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks(r_list)

    # Right: singular value spectrum
    ax = axes[1]
    idx = np.arange(1, len(singular_values) + 1)
    ax.semilogy(idx, singular_values, color='#F18F01', linewidth=1.5)
    ax.axvline(x=max(r_list), color='gray', linestyle='--', alpha=0.6,
               label=f'Max tested $r={max(r_list)}$')

    # Spectral cliff: index of the most abrupt drop in log-sigma
    if len(singular_values) > 2:
        d2 = np.diff(np.log10(singular_values), 2)
        cliff_idx = int(np.argmax(d2)) + 2          # 1-based index of the knee
        cliff_val = singular_values[cliff_idx - 1]
        ax.axvline(x=cliff_idx, color='#C0392B', linestyle=':', linewidth=2,
                   label=f'Spectral cliff at $i \\approx {cliff_idx}$')
        ax.annotate(f'$i \\approx {cliff_idx}$',
                    xy=(cliff_idx, cliff_val), xytext=(0.55, 0.35),
                    textcoords='axes fraction', fontsize=11, color='#C0392B',
                    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.2))

    ax.set_xlabel('Singular value index $i$', fontsize=13)
    ax.set_ylabel(r'Singular value $\sigma_i(W_{\mathrm{full}}^*)$', fontsize=13)
    ax.set_title('Singular Value Spectrum', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('exp1_truncation_error_v3.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("\n  [Plot saved to: exp1_truncation_error_v2.png]")

# ==============================================================================
# 6. Main Entry
# ==============================================================================

def main():
    print("=" * 70)
    print("  Exp. 1: Low-Rank Truncation Error")
    print("  Validates Theorem 1 (Global Low-Rank Approximation Error Bound)")
    print("=" * 70)

    cfg = PowerSystemConfig()
    # Production scale matching the paper (n=m=1000).
    cfg.n = 1000
    cfg.m = 1000
    print(f"\n[Config] n={cfg.n}, m={cfg.m}, ra={cfg.ra}, rd={cfg.rd}")

    np.random.seed(cfg.seed)
    env = PowerSystemEnv(cfg)

    # Step 1: Train full-rank baseline
    print("\n[Step 1] Training full-rank baseline V_full...")
    V_full = train_fullrank_baseline(
        env, cfg.n, cfg.m,
        n_samples=5000,
        n_iters=200,
        lr=0.02,
        gamma=0.95,
        # spectral regularization inducing the low-rank benchmark structure
        wd_scale=1e-2, thresh_every=5, thresh_start=30, thresh_shrink=0.05,
        thresh_protect=200, protect_floor=0.01
    )

    # Step 2: Evaluate truncation errors
    print("\n[Step 2] Evaluating low-rank truncation errors...")
    r_list = [5, 10, 20, 50, 100]
    r_list = [r for r in r_list if r <= min(cfg.n, cfg.m)]
    results, s_full = evaluate_truncation_error(V_full, env, r_list, n_test=10000)

    # Step 3: Plot
    print("\n[Step 3] Generating figure...")
    plot_exp1(results, s_full, cfg)

    print("\n" + "=" * 70)
    print("  Exp. 1 completed successfully.")
    print("=" * 70)


if __name__ == '__main__':
    main()