#!/usr/bin/env python3
"""
================================================================================
Exp. 5: Compression–Accuracy Trade-off
Explores the Pareto frontier between compression ratio and equilibrium utility
================================================================================

This script:
1. Fixes n=1000 (or 200 for demo), varies r in {5, 10, 20, 50, 100}.
2. For each r, trains a low-rank network and evaluates equilibrium policy utility.
3. Computes compression ratio: c.r. = 1 - ((m+n)r + r^2 + m) / (mn + m).
4. Compares against a full-rank FC-NN baseline.
5. Plots compression ratio and relative utility loss vs. r, annotating the sweet spot.
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
    def __init__(self, n=200):
        self.n = n
        self.m = n
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
# 2. Networks
# ==============================================================================

class FullRankValueNN:
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.W = np.random.randn(m, n) / np.sqrt(n)
        self.w_out = np.random.randn(m) / np.sqrt(m)

    def forward(self, x: np.ndarray) -> float:
        return float(self.w_out @ np.tanh(self.W @ x))

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

# ==============================================================================
# 3. Trainer (simplified fixed-basis for speed)
# ==============================================================================

class FixedBasisDLRTrainer:
    """Trains S and w_out with fixed random U, V (fast, sufficient for trade-off demo)."""
    def __init__(self, env: PowerSystemEnv, n: int, m: int, r: int,
                 gamma: float = 0.95, lr: float = 0.02, beta: float = 0.01):
        self.env = env
        self.value = LowRankValueNN(n, m, r)
        self.gamma = gamma
        self.lr = lr
        self.beta = beta

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

    def train(self, K: int = 60, N_b: int = 256) -> None:
        for k in range(K):
            X = self.env.sample_states(N_b)
            Y = np.array([self.bellman_target(x) for x in X])
            # Forward
            Z = X @ self.value.V
            H = Z @ self.value.S.T
            G = np.tanh(H @ self.value.U.T)
            v_pred = G @ self.value.w_out
            # Backward
            delta = v_pred - Y
            Phi_prime = 1 - G**2
            delta_h = (delta[:, None] * self.value.w_out[None, :]) * Phi_prime
            grad_w = G.T @ delta / N_b
            grad_S = (self.value.U.T @ delta_h.T) @ (X @ self.value.V) / N_b
            grad_S += self.beta * self.grad_spectral(self.value.S)
            self.value.w_out -= self.lr * grad_w
            self.value.S -= self.lr * grad_S

# ==============================================================================
# 4. Policy Extractor
# ==============================================================================

class GreedyNashPolicy:
    def __init__(self, env: PowerSystemEnv, n_action_samples: int = 15):
        self.env = env
        self.a_grid = np.linspace(0, 1, n_action_samples)
        self.d_grid = np.linspace(0, 1, n_action_samples)

    def extract(self, x: np.ndarray, value_fn) -> tuple:
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

    def evaluate(self, value_fn, n_test: int = 100) -> float:
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
# 5. Full-Rank Baseline Trainer
# ==============================================================================

class FullRankTrainer:
    def __init__(self, env: PowerSystemEnv, n: int, m: int,
                 gamma: float = 0.95, lr: float = 0.02):
        self.env = env
        self.value = FullRankValueNN(n, m)
        self.gamma = gamma
        self.lr = lr

    def bellman_target(self, x: np.ndarray) -> float:
        a = np.random.rand(self.env.cfg.ra)
        d = np.random.rand(self.env.cfg.rd)
        r = self.env.reward(x, a, d)
        x_next = self.env.step(x, a, d)
        return r + self.gamma * self.value.forward(x_next)

    def train(self, K: int = 60, N_b: int = 256) -> None:
        for k in range(K):
            X = self.env.sample_states(N_b)
            Y = np.array([self.bellman_target(x) for x in X])
            G = np.tanh(X @ self.value.W.T)
            v_pred = G @ self.value.w_out
            delta = v_pred - Y
            Phi_prime = 1 - G**2
            delta_h = (delta[:, None] * self.value.w_out[None, :]) * Phi_prime
            grad_w = G.T @ delta / N_b
            grad_W = (delta_h.T @ X) / N_b
            self.value.w_out -= self.lr * grad_w
            self.value.W -= self.lr * grad_W

# ==============================================================================
# 6. Compression Ratio
# ==============================================================================

def compression_ratio(n: int, m: int, r: int) -> float:
    """Returns compression ratio in [0, 1]."""
    low_rank_params = (m + n) * r + r * r + m
    full_rank_params = m * n + m
    return 1.0 - low_rank_params / full_rank_params

# ==============================================================================
# 7. Main Experiment
# ==============================================================================

def main():
    print("=" * 70)
    print("  Exp. 5: Compression–Accuracy Trade-off")
    print("  Pareto Frontier: Compression Ratio vs. Equilibrium Utility")
    print("=" * 70)

    cfg = PowerSystemConfig(n=200)  # Set to 1000 for full-scale
    env = PowerSystemEnv(cfg)
    n, m = cfg.n, cfg.m

    r_list = [5, 10, 20, 50, 100]
    r_list = [r for r in r_list if r <= min(n, m)]

    # Train full-rank baseline
    print("\n[Step 1] Training full-rank baseline...")
    fc_trainer = FullRankTrainer(env, n, m)
    fc_trainer.train(K=60, N_b=256)
    policy = GreedyNashPolicy(env, n_action_samples=15)
    U_full = policy.evaluate(fc_trainer.value, n_test=100)
    print(f"  Full-rank utility: {U_full:.4f}")

    # Train and evaluate each low-rank configuration
    results = {'r_list': [], 'compression': [], 'utility': [], 'rel_loss': []}

    print("\n[Step 2] Training low-rank configurations...")
    for r in r_list:
        print(f"\n  --- r = {r} ---")
        trainer = FixedBasisDLRTrainer(env, n, m, r, beta=0.01)
        trainer.train(K=60, N_b=256)
        U_lr = policy.evaluate(trainer.value, n_test=100)
        cr = compression_ratio(n, m, r)
        rel_loss = abs(U_full - U_lr) / (abs(U_full) + 1e-6) * 100

        results['r_list'].append(r)
        results['compression'].append(cr * 100)
        results['utility'].append(U_lr)
        results['rel_loss'].append(rel_loss)

        print(f"    Utility: {U_lr:.4f}, Compression: {cr*100:.1f}%, Rel. loss: {rel_loss:.2f}%")

    # Plot
    plot_exp5(results, U_full)

    print("\n" + "=" * 70)
    print("  Exp. 5 completed successfully.")
    print("=" * 70)

# ==============================================================================
# 8. Visualization
# ==============================================================================

def plot_exp5(res: dict, U_full: float):
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    r_list = np.array(res['r_list'])
    comp = np.array(res['compression'])
    rel_loss = np.array(res['rel_loss'])

    # Left y-axis: compression ratio
    color1 = '#2E86AB'
    ax1.set_xlabel('Truncation rank $r$', fontsize=13)
    ax1.set_ylabel('Compression ratio (%)', color=color1, fontsize=13)
    ax1.plot(r_list, comp, 'o-', color=color1, linewidth=2.5, markersize=10, label='Compression ratio')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3)

    # Right y-axis: relative utility loss
    ax2 = ax1.twinx()
    color2 = '#F18F01'
    ax2.set_ylabel('Relative utility loss (%)', color=color2, fontsize=13)
    ax2.plot(r_list, rel_loss, 's--', color=color2, linewidth=2.5, markersize=10, label='Relative utility loss')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Annotate sweet spot
    sweet_idx = np.argmin(np.abs(rel_loss - 5))  # closest to 5% loss
    ax1.annotate('Sweet spot\n' + f'$r={r_list[sweet_idx]}$, {comp[sweet_idx]:.0f}% compressed',
                 xy=(r_list[sweet_idx], comp[sweet_idx]),
                 xytext=(r_list[sweet_idx] + 15, comp[sweet_idx] - 20),
                 fontsize=11, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    # Title and legend
    plt.title('Compression–Accuracy Trade-off (Exp. 5)', fontsize=14, fontweight='bold', pad=15)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)

    plt.tight_layout()
    plt.savefig('exp5_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n  [Plot saved to: exp5_tradeoff.png]")

if __name__ == '__main__':
    main()