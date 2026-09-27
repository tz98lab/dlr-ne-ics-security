# -*- coding: utf-8 -*-
"""
Exp. 2 (v2): Convergence of DLR-AVI at n=1000, multi-seed, with Theorem-2
plateau bound estimation.

Changes vs v1:
  1. n=m=1000 (v1 used n=200).
  2. Vectorized env stepping / target computation; Bellman target computed ONCE
     per outer iteration (semi-gradient, reused across inner steps).
  3. Multi-seed (5 seeds) with mean +/- std bands.
  4. Estimates the five error sources of Lemma 2 / eq. (lem2) at the converged
     iterate of the reference config (N_b=512, s*=5, beta=0.01) and draws the
     theoretical plateau  eps_total/(1-gamma)  as a horizontal line (left panel).
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import qr
from scipy.linalg import svd
from numpy.linalg import lstsq
from typing import Tuple

# ==============================================================================
# 1. Configuration & Environment (vectorized)
# ==============================================================================

class PowerSystemConfig:
    def __init__(self, n=1000, seed=42):
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
        self.seed = seed


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

    def g_power_batch(self, X: np.ndarray) -> np.ndarray:
        Z = X @ self.V_coupling                       # (N, r_coupling)
        diff = Z[:, :, None] - Z[:, None, :]          # (N, r, r): pairs of coupling dims
        nl = np.sin(diff) + 0.1 * diff**2
        return (self.Kpf @ (self.V_coupling @ nl.sum(axis=2).T)).T

    def step_batch(self, X: np.ndarray, A: np.ndarray, D: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        x_next = X + cfg.dt * self.g_power_batch(X)
        gate_a = 1.0 / (1.0 + np.exp(-(X - 0.5) * 10))
        x_next += gate_a * (self.Ba_bar @ A.T).T          # diag(gate) Ba_bar A
        logits = X @ self.Wd.T                               # (N, rd)
        gate_d = np.exp(logits - logits.max(axis=1, keepdims=True))
        gate_d /= gate_d.sum(axis=1, keepdims=True)
        x_next += (gate_d * D) @ self.Bd_bar.T
        x_next += np.random.randn(*X.shape) * cfg.sigma_w
        return np.clip(x_next, cfg.x_min, cfg.x_max)

    def reward_batch(self, X: np.ndarray, A: np.ndarray, D: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        r = -np.sum((X @ self.Cr.T)**2, axis=1)
        return r - cfg.ca * np.sum(A**2, axis=1) - cfg.cd * np.sum(D**2, axis=1)

    def sample_states(self, N: int) -> np.ndarray:
        samples = np.random.rand(N, self.cfg.n)
        for i in range(self.cfg.n):
            perm = np.random.permutation(N)
            samples[:, i] = (perm + samples[:, i]) / N
        return self.cfg.x_min + samples * (self.cfg.x_max - self.cfg.x_min)

# ==============================================================================
# 2. Value networks
# ==============================================================================

class FullRankValueNN:
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.W = np.random.randn(m, n) / np.sqrt(n)
        self.w_out = np.random.randn(m) / np.sqrt(m)

    def forward(self, x):
        return float(self.w_out @ np.tanh(self.W @ x))

    def forward_batch(self, X):
        return np.tanh(X @ self.W.T) @ self.w_out


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

    def forward(self, x):
        return float(self.w_out @ np.tanh(self.U @ (self.S @ (self.V.T @ x))))

    def forward_batch(self, X):
        return np.tanh((X @ self.V) @ self.S.T @ self.U.T) @ self.w_out

# ==============================================================================
# 3. Bellman targets (vectorized, optional action-MC averaging)
# ==============================================================================

class BellmanTarget:
    def __init__(self, env: PowerSystemEnv, gamma: float = 0.95):
        self.env = env
        self.gamma = gamma

    def compute_targets(self, X, value_fn, k_avg: int = 1) -> np.ndarray:
        N = X.shape[0]
        Y = np.zeros(N)
        chunk = 256   # bound the (N,N,r_coupling) intermediate in g_power_batch
        for i0 in range(0, N, chunk):
            Xc = X[i0:i0 + chunk]
            n_c = Xc.shape[0]
            acc = np.zeros(n_c)
            for _ in range(k_avg):
                A = np.random.rand(n_c, self.env.cfg.ra)
                D = np.random.rand(n_c, self.env.cfg.rd)
                r = self.env.reward_batch(Xc, A, D)
                x_next = self.env.step_batch(Xc, A, D)
                acc += r + self.gamma * value_fn.forward_batch(x_next)
            Y[i0:i0 + chunk] = acc / k_avg
        return Y

# ==============================================================================
# 4. DLR-AVI with per-iteration cached target + diagnostics stash
# ==============================================================================

def _clip(g, c=100.0):
    n = np.linalg.norm(g)
    return g if n <= c else g * (c / n)

class DLRAVI:
    def __init__(self, env, n, m, r, gamma=0.95, lr=0.02, beta=0.01,
                 s_star=10, trunc_thresh=1e-6):
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
        self.diag = {}

    def spectral_regularizer(self, S):
        alpha_sq = np.trace(S.T @ S) / S.shape[0]
        return np.linalg.norm(S.T @ S - alpha_sq * np.eye(S.shape[0]), 'fro')

    def grad_spectral(self, S):
        alpha_sq = np.trace(S.T @ S) / S.shape[0]
        E = S.T @ S - alpha_sq * np.eye(S.shape[0])
        R = np.linalg.norm(E, 'fro')
        if R < 1e-12:
            return np.zeros_like(S)
        return 2 * S @ E / R

    def basis_augmentation(self, X_batch, Y):
        value = self.value
        N = X_batch.shape[0]
        Z = X_batch @ value.V
        H = Z @ value.S.T
        G = H @ value.U.T
        Phi = np.tanh(G)
        Phi_prime = 1 - Phi**2
        delta = Phi @ value.w_out - Y
        delta_h = (delta[:, None] * value.w_out[None, :]) * Phi_prime
        grad_U = _clip(delta_h.T @ H / N)
        grad_V = _clip(X_batch.T @ (delta_h @ value.U @ value.S) / N)
        U_aug = np.hstack([value.U, grad_U])
        V_aug = np.hstack([value.V, grad_V])
        U_hat, _ = qr(U_aug, mode='economic')
        V_hat, _ = qr(V_aug, mode='economic')
        S_0 = U_hat.T @ value.U @ value.S @ value.V.T @ V_hat
        return U_hat, V_hat, S_0

    def inner_loop(self, U_hat, V_hat, S_0, X_batch, Y, s_star=None):
        N = X_batch.shape[0]
        s_star = self.s_star if s_star is None else s_star
        S_hat = S_0.copy()
        w_out_hat = self.value.w_out.copy()
        Z = X_batch @ V_hat
        U_t = U_hat.T
        mS = np.zeros_like(S_hat); vS = np.zeros_like(S_hat)
        mw = np.zeros_like(w_out_hat); vw = np.zeros_like(w_out_hat)
        b1, b2, eps_adam = 0.9, 0.999, 1e-8
        for s in range(1, s_star + 1):
            Phi = np.tanh(Z @ S_hat.T @ U_t)
            v_pred = Phi @ w_out_hat
            delta = v_pred - Y
            Phi_prime = 1 - Phi**2
            delta_h = (delta[:, None] * w_out_hat[None, :]) * Phi_prime
            grad_w = Phi.T @ delta / N
            grad_S = (U_hat.T @ delta_h.T) @ Z / N
            grad_S += self.beta * self.grad_spectral(S_hat)
            mS = b1 * mS + (1 - b1) * grad_S
            vS = b2 * vS + (1 - b2) * grad_S**2
            mw = b1 * mw + (1 - b1) * grad_w
            vw = b2 * vw + (1 - b2) * grad_w**2
            S_hat -= self.lr * (mS / (1 - b1**s)) / (np.sqrt(vS / (1 - b2**s)) + eps_adam)
            w_out_hat -= self.lr * (mw / (1 - b1**s)) / (np.sqrt(vw / (1 - b2**s)) + eps_adam)
        return S_hat, w_out_hat

    def truncate_and_retract(self, U_hat, V_hat, S_hat):
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

    def iteration(self, X_batch):
        Y = self.bellman.compute_targets(X_batch, self.value, k_avg=1)
        U_hat, V_hat, S_0 = self.basis_augmentation(X_batch, Y)
        S_hat, w_out_hat = self.inner_loop(U_hat, V_hat, S_0, X_batch, Y)
        self.diag = dict(U_hat=U_hat, V_hat=V_hat, S_0=S_0.copy(),
                         S_hat=S_hat.copy(), w_out_hat=w_out_hat.copy(),
                         Y=Y.copy(), X_batch=X_batch.copy())
        self.truncate_and_retract(U_hat, V_hat, S_hat)
        self.value.w_out = w_out_hat
        return self.spectral_regularizer(self.value.S)


class FCNNAVI:
    @staticmethod
    def _clip(g, c=100.0):
        n = np.linalg.norm(g)
        return g if n <= c else g * (c / n)

    def __init__(self, env, n, m, gamma=0.95, lr=0.02):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.value = FullRankValueNN(n, m)
        self.bellman = BellmanTarget(env, gamma)

    def iteration(self, X):
        N = X.shape[0]
        Phi = np.tanh(X @ self.value.W.T)
        v_pred = Phi @ self.value.w_out
        Y = self.bellman.compute_targets(X, self.value, k_avg=1)
        delta = v_pred - Y
        Phi_prime = 1 - Phi**2
        delta_h = (delta[:, None] * self.value.w_out[None, :]) * Phi_prime
        grad_W = self._clip(delta_h.T @ X / N)
        grad_w = self._clip(Phi.T @ delta / N)
        self.value.W -= self.lr * grad_W
        self.value.w_out -= self.lr * grad_w

# ==============================================================================
# 5. Epsilon_total estimation (five sources of Lemma 2)
# ==============================================================================

def estimate_epsilon_total(dlr: DLRAVI, env: PowerSystemEnv,
                           n_eval: int = 2000, seed: int = 123):
    """Estimate the five error sources at the converged iterate.

    Returns dict with per-source terms and the plateau bound eps_total/(1-gamma).
    """
    rng = np.random.RandomState(seed)
    Xe = env.sample_states(n_eval)
    d = dlr.diag
    U_hat, V_hat, S_0, S_hat = d['U_hat'], d['V_hat'], d['S_0'], d['S_hat']
    w_hat = d['w_out_hat']
    gamma = dlr.gamma

    # Low-noise reference targets (k_avg action-MC averaging)
    Y_ref = dlr.bellman.compute_targets(Xe, dlr.value, k_avg=32)
    # Single-draw targets (algorithm protocol) -> source (v)
    Y_1 = dlr.bellman.compute_targets(Xe, dlr.value, k_avg=1)
    eps_mm = float(np.max(np.abs(Y_1 - Y_ref)))

    # Projected coordinates and covering number N_delta
    Ze = Xe @ V_hat                                  # (N, 2r)
    dist = np.linalg.norm(Ze[:, None, :] - Ze[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    knn5 = np.sort(dist, axis=1)[:, 4]
    delta_cov = float(np.mean(knn5))
    np.fill_diagonal(dist, 0.0)   # self counts as covered by its own ball
    covered = np.zeros(n_eval, dtype=bool)
    N_delta = 0
    while not covered.all():
        i = int(np.argmax(~covered))
        N_delta += 1
        covered |= dist[i] <= delta_cov

    # Source (i): basis-subspace approximation, Bridge Lemma form sqrt(N_delta)*eps_app
    Phi_aug = np.tanh((Xe @ V_hat) @ S_hat.T @ U_hat.T)
    w_ls, *_ = lstsq(Phi_aug, Y_ref)
    resid = Phi_aug @ w_ls - Y_ref
    eps_app = float(np.sqrt(np.mean(resid**2)))
    term_i = np.sqrt(N_delta) * eps_app

    # Source (ii): inner-loop suboptimality, sqrt(N_delta * eps_inner)
    S_short, w_short = dlr.inner_loop(U_hat, V_hat, S_0, Xe, Y_ref, s_star=dlr.s_star)
    S_long, w_long = dlr.inner_loop(U_hat, V_hat, S_0, Xe, Y_ref, s_star=300)
    v_short = np.tanh((Xe @ V_hat) @ S_short.T @ U_hat.T) @ w_short
    v_long = np.tanh((Xe @ V_hat) @ S_long.T @ U_hat.T) @ w_long
    gap = v_short - v_long
    eps_inner = float(np.mean(gap**2))
    term_ii = np.sqrt(N_delta * eps_inner)

    # Source (iii): sampling geometry (L_fun + L_T) * delta
    idx = rng.randint(0, n_eval, size=(20000, 2))
    dz = np.linalg.norm(Ze[idx[:, 0]] - Ze[idx[:, 1]], axis=1) + 1e-12
    lips_fun = np.abs(Y_ref[idx[:, 0]] - Y_ref[idx[:, 1]]) / dz
    L_fun = float(np.percentile(lips_fun, 97))
    eps_x = 1e-3
    lip_T = []
    for _ in range(200):
        x = Xe[rng.randint(0, n_eval)]
        u = rng.randn(dlr.n); u /= np.linalg.norm(u)
        xa = np.clip(x + eps_x * u, env.cfg.x_min, env.cfg.x_max)
        Ta = dlr.bellman.compute_targets(xa[None, :], dlr.value, k_avg=16)[0]
        Tb = dlr.bellman.compute_targets(x[None, :], dlr.value, k_avg=16)[0]
        lip_T.append(abs(Ta - Tb) / eps_x)
    L_T = float(np.percentile(lip_T, 97))
    term_iii = (L_fun + L_T) * delta_cov

    # Source (iv): retraction, L_phi * ||w_out|| * R_X * eps_retr
    s_svd = svd(S_hat, compute_uv=False)
    eps_retr = float(np.sqrt(np.sum(s_svd[dlr.r:]**2)))
    L_phi = 1.0
    W_max = float(np.linalg.norm(w_hat))
    R_X = float(np.max(np.linalg.norm(Xe, axis=1)))
    term_iv = L_phi * W_max * R_X * eps_retr

    eps_total = term_i + term_ii + term_iii + term_iv + eps_mm
    plateau = eps_total / (1 - gamma)

    return dict(term_i=term_i, term_ii=term_ii, term_iii=term_iii,
                term_iv=term_iv, term_v=eps_mm, N_delta=N_delta,
                delta_cov=delta_cov, eps_app=eps_app, eps_retr=eps_retr,
                L_fun=L_fun, L_T=L_T, W_max=W_max, R_X=R_X,
                eps_total=eps_total, plateau=plateau)

# ==============================================================================
# 6. Experiment runner
# ==============================================================================

def run_one(cfg, K, r, N_b, s_star, beta, lr=0.02, pretrain=50):
    env = PowerSystemEnv(cfg)
    fc = FCNNAVI(env, cfg.n, cfg.m, gamma=0.95, lr=lr)
    for _ in range(pretrain):
        fc.iteration(env.sample_states(N_b))
    V_ref = fc.value

    dlr = DLRAVI(env, cfg.n, cfg.m, r, gamma=0.95, lr=lr, beta=beta,
                 s_star=s_star)
    errors, residuals, kappas = [], [], []
    for k in range(K):
        X = env.sample_states(N_b)
        dlr.iteration(X)
        Xe = env.sample_states(1000)
        errors.append(float(np.max(np.abs(dlr.value.forward_batch(Xe)
                                          - V_ref.forward_batch(Xe)))))
        Xb = env.sample_states(200)
        Vb = dlr.value.forward_batch(Xb)
        Yb = dlr.bellman.compute_targets(Xb, dlr.value, k_avg=1)
        residuals.append(float(np.max(np.abs(Yb - Vb))))
        sv = svd(dlr.value.S, compute_uv=False)
        kappas.append(float(sv[0] / max(sv[-1], 1e-12)))
    out = dict(errors=np.array(errors), residuals=np.array(residuals),
               kappas=np.array(kappas))
    if s_star == 10 and beta > 0 and N_b == 512:
        out['eps_est'] = estimate_epsilon_total(dlr, env)
    return out

# ==============================================================================
# 7. Main
# ==============================================================================

def main():
    import sys, json, os
    print("=" * 70)
    print("  Exp. 2 (v2): Convergence at n=1000, multi-seed + Thm-2 plateau")
    print("=" * 70)
    K = 120
    r = 20
    LR = 0.01   # stability boundary at n=1000: lr>=0.015 diverges
    seeds = [0, 1, 2, 3, 4]
    configs = [
        dict(N_b=512, s_star=10, beta=0.01),
        dict(N_b=128, s_star=10, beta=0.01),
        dict(N_b=512, s_star=5,  beta=0.01),
        dict(N_b=512, s_star=10, beta=0.0),
        dict(N_b=64,  s_star=10, beta=0.01),
    ]

    # ---- stage mode: python exp02_convergence_v2.py run <cfg_idx> <seed> ----
    if len(sys.argv) >= 3 and sys.argv[1] == 'run':
        ci, sd = int(sys.argv[2]), int(sys.argv[3])
        c = configs[ci]
        cfg = PowerSystemConfig(n=1000, seed=sd)
        t0 = time.time()
        res = run_one(cfg, K, r, c['N_b'], c['s_star'], c['beta'], lr=LR)
        np.savez(f'exp2_v2_cfg{ci}_seed{sd}.npz',
                 errors=res['errors'], residuals=res['residuals'],
                 kappas=res['kappas'])
        if 'eps_est' in res:
            with open(f'exp2_v2_cfg{ci}_seed{sd}_eps.json', 'w') as f:
                json.dump(res['eps_est'], f)
        print(f"cfg{ci} seed{sd} done in {time.time()-t0:.0f}s  "
              f"err[-1]={res['errors'][-1]:.4f}  kappa[-1]={res['kappas'][-1]:.1f}")
        return

    # ---- plot mode: python exp02_convergence_v2.py plot ----
    agg = []
    eps_est = None
    for ci, c in enumerate(configs):
        label = f"$N_b={c['N_b']}, s^*={c['s_star']}, \\beta={c['beta']}$"
        runs = []
        for sd in seeds:
            fn = f'exp2_v2_cfg{ci}_seed{sd}.npz'
            if not os.path.exists(fn):
                print(f"MISSING {fn}"); continue
            d = np.load(fn)
            runs.append(dict(errors=d['errors'], residuals=d['residuals'],
                             kappas=d['kappas']))
            ef = f'exp2_v2_cfg{ci}_seed{sd}_eps.json'
            if os.path.exists(ef):
                with open(ef) as f:
                    eps_est = json.load(f)
        agg.append(dict(label=label, runs=runs))
    if eps_est:
        print("\n=== Epsilon_total breakdown ===")
        for k0, v0 in eps_est.items():
            print(f"  {k0}: {v0:.4e}" if isinstance(v0, float) else f"  {k0}: {v0}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ['C0', 'C1', 'C2', 'C3']
    k_axis = np.arange(K)

    ax = axes[0]
    for i, a in enumerate(agg):
        E = np.array([r0['errors'] / (r0['errors'][0] + 1e-12) for r0 in a['runs']])
        mean, std = E.mean(0), E.std(0)
        ax.semilogy(k_axis, mean, '-', color=colors[i], label=a['label'])
        ax.fill_between(k_axis, mean - std, mean + std, color=colors[i], alpha=0.2)
    ax.semilogy(k_axis, 0.95**k_axis, 'k--', lw=1.5, alpha=0.7, label=r'$\gamma^k$')
    if eps_est:
        E0 = np.mean([r0['errors'][0] for r0 in agg[0]['runs']])
        plateau_rel = eps_est['plateau'] / E0
        ax.axhline(plateau_rel, color='k', ls=':', lw=2)
        import matplotlib.ticker as mticker
        ax.text(1, plateau_rel * 0.3,
                'Thm.~2 plateau bound ' + r'$\varepsilon_{\mathrm{total}}/(1-\gamma)$'
                + f'$\\approx{eps_est["plateau"]:.1e}$ (a priori)',
                color='k', fontsize=9)
        ax.set_ylim(bottom=1e-3, top=max(2.0, plateau_rel * 3))
        print(f"\nPlateau bound (rel.): {plateau_rel:.3e}")
    ax.set_xlabel('Iteration $k$', fontsize=12)
    ax.set_ylabel(r'Relative $\|V^{(k)} - V_{\mathrm{ref}}\|_\infty$', fontsize=12)
    ax.set_title('Convergence of DLR-NE (Theorem 2)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

    ax = axes[1]
    for i, a in enumerate(agg):
        R = np.array([r0['residuals'] for r0 in a['runs']])
        mean, std = R.mean(0), R.std(0)
        ax.semilogy(k_axis, mean, '-', color=colors[i], label=a['label'])
        ax.fill_between(k_axis, mean - std, mean + std, color=colors[i], alpha=0.2)
    ax.set_xlabel('Iteration $k$', fontsize=12)
    ax.set_ylabel('Bellman residual', fontsize=12)
    ax.set_title('Bellman Residual', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

    ax = axes[2]
    for i, a in enumerate(agg):
        Ka = np.array([r0['kappas'] for r0 in a['runs']])
        mean, std = Ka.mean(0), Ka.std(0)
        ax.semilogy(k_axis, mean, '-', color=colors[i], label=a['label'])
        ax.fill_between(k_axis, mean - std, mean + std, color=colors[i], alpha=0.2)
    ax.set_xlabel('Iteration $k$', fontsize=12)
    ax.set_ylabel(r'Condition number $\kappa(S^{(k)})$', fontsize=12)
    ax.set_title('Condition Number Trajectory', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('exp2_convergence_v2.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("\n[Saved] exp2_convergence_v2.png")
    if eps_est:
        print("\n=== Epsilon_total breakdown ===")
        for k0, v0 in eps_est.items():
            print(f"  {k0}: {v0:.4e}" if isinstance(v0, float) else f"  {k0}: {v0}")


if __name__ == '__main__':
    main()
