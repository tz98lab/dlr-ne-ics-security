# DLR-NE for ICS Security Games

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Dynamical Low-Rank Approximation for Nash Equilibrium Computation in High-Dimensional ICS Security Games**
>
> Submitted to *Automatica* (Long Paper)

## Overview

This repository contains the implementation and experimental validation of the **Dynamical Low-Rank Approximate Value Iteration (DLR-NE)** algorithm for computing Nash equilibria in high-dimensional Industrial Control System (ICS) security games.

The strategic interaction between **Advanced Persistent Threats (APTs)** and **Moving Target Defenses (MTDs)** is modeled as a two-player zero-sum stochastic game over a continuous state space of dimension $n \sim 10^3$–$10^4$. The key insight is that the physical low-rank coupling of attack and defense channels (a structural property inherent to ICS network topology) enables tractable low-rank neural-network approximation of the equilibrium value function, reducing per-iteration complexity from $\mathcal{O}(n^3)$ to $\mathcal{O}(nr^2)$.

### Core Theoretical Contributions

| Theorem | Statement | Experiment |
|:---|:---|:---|
| **Theorem 1** | Global low-rank approximation error bound for the optimal value function $V^*$ | [Exp. 1](#exp1) |
| **Theorem 2** | Geometric convergence of DLR-NE with explicit steady-state error decomposition | [Exp. 2](#exp2) |
| **Theorem 3** | Per-iteration complexity $\mathcal{O}(nr^2)$ and speedup $\Theta(n/r^2)$ over full-rank baselines | [Exp. 3](#exp3) |
| **Corollary 1** | Game-theoretic robustness: equilibrium sensitivity is controlled by the condition number $\kappa(S)$ | [Exp. 4](#exp4) |

---

## Repository Structure

```
DLR-NE-ics-security/
├── src/                          # Core algorithmic modules
│   ├── environment.py            # Synthetic nonlinear power-system dynamics
│   ├── networks.py               # Low-rank and full-rank neural networks
│   ├── bellman.py                # Bellman operator and greedy Nash policy extractor
│   ├── dlra_vi.py                # Algorithm 1: DLR-NE
│   └── utils.py                  # FLOPs accounting, EYM error, metrics
│
├── experiments/                  # Experimental scripts (one per theorem)
│   ├── exp01_truncation_error.py     # Theorem 1: Low-rank truncation error
│   ├── exp02_convergence.py          # Theorem 2: Convergence of DLR-NE
│   ├── exp03_complexity.py           # Theorem 3: Computational complexity
│   ├── exp04_robustness.py           # Corollary 1: Game-theoretic robustness
│   ├── exp05_tradeoff.py             # Compression-accuracy Pareto frontier
│   └── exp06_ablation.py             # Ablation: necessity of basis augmentation
│
├── data/                         # Generated datasets (excluded from git)
├── results/                      # Figures and logs (excluded from git)
├── notebooks/                    # Prototyping and visualization
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```
## Installation

### Prerequisites

- Python >= 3.10
- NumPy, SciPy, Matplotlib, PyTorch (CPU or CUDA)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/tz98lab/DLR-NE-ics-security.git
cd DLR-NE-ics-security

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```
### Requirements

```text
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
torch>=2.0.0
```

## Experiments

All experiments are self-contained and can be run independently. Each script generates figures in the `results/figures/` directory.

> **Note on Scale**: The default configurations use $n=200$ for rapid demonstration. To reproduce the full-scale results reported in the paper ($n \sim 10^3$), modify the `PowerSystemConfig` at the top of each script (see inline comments).

### <a name="exp1"></a> Exp. 1: Low-Rank Truncation Error (Theorem 1)

Validates that the truncation error $\|V_{\mathrm{full}} - \widehat{V}_r\|_\infty$ decays with rank $r$ and aligns with the Eckart-Young-Mirsky theoretical prediction.

```bash
python experiments/exp01_truncation_error.py
```

**Output**: `results/figures/exp1_truncation_error.png`  
**Expected**: Measured error (blue circles) tracks the theoretical bound $L_\phi \|w_{\mathrm{out}}^*\|_2 R_{\mathcal{X}} \epsilon_{\mathrm{EYM}}(r)$ (purple dashed line); singular-value spectrum shows rapid decay.

### <a name="exp2"></a> Exp. 2: Convergence of DLR-NE (Theorem 2)

Validates geometric convergence with rate $\gamma$ and the explicit steady-state error neighborhood $\varepsilon_{\mathrm{total}}/(1-\gamma)$.

```bash
python experiments/exp02_convergence.py
```

**Output**: `results/figures/exp2_convergence.png`  
**Expected**: Error curves decay with slope $\approx \gamma$; larger batch size $N_b$ and inner-loop steps $s^*$ reduce the steady-state plateau.

### <a name="exp3"></a> Exp. 3: Computational Complexity (Theorem 3)

Validates the $\mathcal{O}(nr^2)$ per-iteration complexity and the $\Theta(n/r^2)$ speedup over full-rank FC-NN baselines.

```bash
python experiments/exp03_complexity.py
```

**Output**: `results/figures/exp3_complexity.png`  
**Expected**: DLR-NE scales linearly in $n$ (slope 1 in log-log), FC-NN scales quadratically (slope 2); measured speedup $\approx 20\times$ at $n=2000, r=10$.

### <a name="exp4"></a> Exp. 4: Game-Theoretic Robustness (Corollary 1)

Validates the linear relation $\|V(\cdot;\pi_D,\pi_A) - V(\cdot;\tilde{\pi}_D,\pi_A)\|_\infty \le L_\kappa \|\Delta S\|_F$ and the condition-number control $L_\kappa \propto \kappa(S)$.

```bash
python experiments/exp04_robustness.py
```

**Output**: `results/figures/exp4_robustness.png`  
**Expected**: Value deviation is linear in $\|\Delta S\|_F$ with slope decreasing in spectral regularization weight $\beta$; $\kappa(S)$ is compressed by increasing $\beta$.

### <a name="exp5"></a> Exp. 5: Compression–Accuracy Trade-off

Explores the Pareto frontier between compression ratio and equilibrium policy utility.

```bash
python experiments/exp05_tradeoff.py
```

**Output**: `results/figures/exp5_tradeoff.png`  
**Expected**: Sweet spot at $r \sim 10$–$20$ where compression $> 95\%$ and utility loss $< 5\%$.

### <a name="exp6"></a> Exp. 6: Ablation—Necessity of Basis Augmentation

Compares three variants: (i) full Algorithm 1, (ii) fixed basis, (iii) no retraction.

```bash
python experiments/exp06_ablation.py
```

**Output**: `results/figures/exp6_ablation.png`  
**Expected**: Fixed-basis variant stagnates; no-retraction variant is unstable; full Algorithm 1 achieves stable decay at controlled cost.

## Reproducibility Statement

All random seeds are fixed and reported. Each experiment script is self-contained and can be executed on a standard laptop (CPU-only) for the default $n=200$ configuration. Full-scale experiments ($n \sim 10^3$) were conducted on an Intel Xeon Gold 6248R with an NVIDIA A100 40GB GPU (used only for accelerating full-rank baselines).

Key hyperparameters:
- Discount factor: $\gamma = 0.95$
- Activation: $\tanh$ ($L_\phi = 1$)
- Batch size: $N_b = 512$ (LHS)
- Inner-loop steps: $s^* = 10$
- Spectral regularization: $\beta = 0.01$ (default)

See `src/utils.py` and individual experiment scripts for complete parameter lists.

## Citation

If you use this code, please cite:

```bibtex
@article{dlravi2026,
  title={Dynamical Low-Rank Approximation for Nash Equilibrium Computation in High-Dimensional ICS Security Games},
  journal={Automatica},
  year={2026},
  note={Submitted}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions regarding the code or experiments, please open an issue on GitHub.
For questions regarding the theoretical content, please refer to the manuscript submitted to *Automatica*.
