# Imitation Learning for Monopropellant Rocket Engine Thrust Control

A machine learning approach to replace real-time NMPC control with a lightweight neural network for monopropellant rocket engine control.

## Project Overview

**Goal:** Train a neural network via behavior cloning to imitate NMPC (Nonlinear Model Predictive Control) for a monopropellant rocket engine system.

**Why:** NMPC solves complex optimization every 50ms but is too computationally expensive for embedded deployment. A neural network can learn the same control policy and execute it in microseconds.

**NN Mapping:**
```
f(p_c, p_c_ref, kv_prev) → kv_set
```
- `p_c`: Current chamber pressure
- `p_c_ref`: Target chamber pressure
- `kv_prev`: Previous valve command
- `kv_set`: Valve command output (constrained to [0.1, 1.0])

## System Architecture

### Physical System
```
Tank (30 bar) → Pipe → Valve → Chamber → Nozzle
   p_bc         m_p,p_p   kv      p_c      exhaust
```

### Two Models

| Model | States | Purpose | Implementation |
|-------|--------|---------|-----------------|
| **Monoprop_sim** | 4-state: [m_p, p_p, kv, p_c] | Ground truth simulation | NumPy |
| **Monoprop** | 1-state: [p_c] | MPC optimization (reduced-order) | CasADi symbolic |

### Key Parameters

- **Sampling time:** h_mpc = 0.05s (50ms)
- **Prediction horizon:** 1.0s (20 steps)
- **Valve constraints:** 0.1 ≤ kv ≤ 1.0, |dkv/dt| ≤ 1.43 m³/h/s
- **Operating range:** 9–15.5 bar
- **Fluid:** Ethanol (C2H6O)

## Setup

### 1. Clone the Repository

```bash
git clone <repo-url>
cd BC_NMPC
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install numpy scipy matplotlib torch casadi CoolProp ipykernel
```

### 4. Register Jupyter Kernel (for notebooks)

```bash
python3 -m ipykernel install --user --name=venv --display-name="Python (venv)"
```

> **Note:** If `python3` resolves to the system interpreter instead of the venv, use the full path:
> ```bash
> /path/to/BC_NMPC/venv/bin/python3 scripts/<script>.py
> ```

## Quick Start

### 1. Collect Training Data
```bash
python3 scripts/collect_data.py
```
Generates 125 episodes × 80 steps = 10,000 state-action samples.
Output: `data/bc_dataset_v6.npz`

### 2. Train Neural Network
```bash
python3 scripts/train_nn.py
```
Trains a 3→5→5→1 neural network with weighted MSE loss.
Outputs:
- `results/models/policy_model.pth` — trained model weights + normalization stats
- `results/plots/training_curve_*.png` — loss curves
- `results/plots/prediction_results_*.png` — prediction accuracy

### 3. Validate Against NMPC
```bash
python3 scripts/validate_nn.py
```
Compares NN vs reactive NMPC on 3 random 5s episodes.
Output: `results/plots/validation_v6.png`

### 4. Covariate Shift Tests
```bash
# OOD reference steps (above 15.5 bar and below 9.0 bar)
python3 scripts/covariate_shift_test.py

# Measurement noise on p_c (0.1, 0.3, 0.5 bar std)
python3 scripts/covariate_shift_2.py

# Unmodelled tank pressure drop disturbance
python3 scripts/covariate_shift_3.py
```

### 5. Run Jupyter Notebooks
```bash
jupyter notebook notebooks/
```
Select the **Python (venv)** kernel when opening a notebook. If the kernel is not listed, register it first (see Setup step 4).

### 6. Run Original NMPC Demo
```bash
python3 scripts/main.py
```

## File Structure

```
BC_NMPC/
├── README.md
├── .gitignore
│
├── src/                             # Physics models and NMPC solver
│   ├── model.py                     # Monoprop_sim & Monoprop classes
│   ├── mpc.py                       # NMPC solver (CasADi/IPOPT)
│   ├── integrator.py                # RK4 numerical integration
│   ├── reference.py                 # Reference trajectory generation
│   └── plotting.py                  # Visualization utilities
│
├── scripts/                         # Behavior cloning pipeline
│   ├── collect_data.py              # Data collection (parallel, ZOH, v6)
│   ├── train_nn.py                  # Train policy network
│   ├── validate_nn.py               # Compare NN vs NMPC (5s episodes)
│   ├── covariate_shift_test.py      # OOD reference step tests
│   ├── covariate_shift_2.py         # Measurement noise tests
│   ├── covariate_shift_3.py         # Unmodelled tank pressure drop tests
│   └── main.py                      # NMPC baseline demo
│
├── notebooks/                       # Jupyter analysis (Python (venv) kernel)
│   ├── explore_dataset.ipynb        # Dataset statistics & distribution
│   └── kv_pc_mapping.ipynb          # Valve-pressure steady-state mapping
│
├── data/                            # Datasets (generated, not tracked in git)
│   └── bc_dataset_v6.npz
│
├── results/                         # Outputs (generated, not tracked in git)
│   ├── models/
│   │   └── policy_model.pth
│   └── plots/
│
├── docs/                            # References & documentation
│
└── venv/                          # Python virtual environment (not tracked in git)
```

## Neural Network Architecture

**Network Structure:** 3 → 5 → 5 → 1 (Tanh activations, sigmoid output)
- Input: normalized p_c, normalized p_c_ref, raw kv_prev
- Output: kv_set bounded to [0.1, 1.0]

**Loss Function:** Weighted MSE — active steps (|Δkv| ≥ 0.02) weighted 5×

**Training Strategy:**
- Adam optimizer, lr = 1e-3
- ReduceLROnPlateau (halve LR after 30 epochs without improvement)
- Early stopping (patience = 100 epochs, restores best weights)
- Episode-based 80/20 train/test split (prevents data leakage)

**Normalization:** p_c and p_c_ref normalized to zero mean, unit variance. Stats saved in `policy_model.pth` alongside weights.

## Dataset (v6)

- **125 episodes × 80 steps = 10,000 samples**
- ZOH step references within [9.0, 15.5] bar operating range
- Reactive NMPC expert (current reference only, no lookahead)
- Parallel collection using 8 workers (~5–10 minutes)
- Raw physical values stored (normalization done at training time)

## Dependencies

```
numpy, scipy, matplotlib
torch        # Neural network training
casadi       # NMPC optimization (CasADi/IPOPT)
CoolProp     # Fluid thermodynamic properties
ipykernel    # Jupyter notebook support
```
