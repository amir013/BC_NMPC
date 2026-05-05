# BlueROV2 Thesis Notes
## Learning-based Control For Underwater Manipulation

---

## Thesis Overview

**Title:** Master Thesis/Forschungspraxis: Learning-based Control For Underwater Manipulation
**Lab:** Environmental Robotics Lab at MIRMI, TUM
**Platform:** BlueROV2 with simple manipulator
**Contact:** Moritz Graf (moritz.graf@tum.de)

---

## Key Challenges

- Non-linear, uncertain underwater dynamics
- Coupled arm-vehicle dynamics (hard to model accurately)
- Sim-to-real gap
- Precise control and stability of manipulators

---

## Control Strategy: IL + RL

### Pipeline
```
Approximate NMPC (expert) → BC demonstrations → pretrained NN policy → RL fine-tuning on real ROV
```

### Why IL + RL (not BC alone)
- BC alone sufficient for simple, low-dimensional systems (e.g. rocket engine)
- ROV + manipulator: high-dimensional, fast coupled dynamics, model mismatch
- RL fine-tuning corrects what BC got wrong due to imperfect coupled dynamics model
- Closed-loop feedback mitigates covariate shift but not enough for manipulation tasks

### Thesis Narrative
> "The approximate NMPC model is sufficient as a BC teacher for vehicle stabilisation,
> but insufficient for coupled manipulation tasks. RL fine-tuning on the real platform
> corrects the model mismatch in the coupled dynamics, achieving robust underwater manipulation."

---

## System Modelling

### BlueROV2 Fossen Model
$$M\dot{\nu} + C(\nu)\nu + D(\nu)\nu + g(\eta) = \tau$$

| Symbol | Meaning |
|---|---|
| M | Inertia + added mass matrix |
| C | Coriolis matrix |
| D | Damping (drag) |
| g | Gravity/buoyancy |
| τ | Thruster forces |

### Parameters (already published — no full system ID needed)
- Thruster model: Blue Robotics published curves
- Inertia matrix: from CAD
- Added mass: from literature
- Drag coefficients: multiple papers available
- Full Fossen model: already in UUV Simulator

### Manipulator Coupling
- Changes centre of mass → affects buoyancy/gravity
- Reaction forces → disturbs vehicle position
- Added inertia → changes effective mass matrix
- **Quasi-static coupling** is reasonable starting point for thesis

### Control Architecture
| Subsystem | Controller |
|---|---|
| ROV position/attitude | NMPC → BC → RL fine-tune |
| Manipulator task | RL directly (reward = task completion) |

---

## NMPC

- Lab already has an approximate NMPC for the system
- Coupled dynamics not well captured — this is the core motivation for RL
- Uses CasADi/IPOPT (same as rocket engine project)
- Imperfect model is not a weakness — it is the thesis motivation

---

## Neural Network

### Architecture
- **MLP** (not ResNet — inputs are tabular state vectors, not images)
- **ReLU** activations for RL (faster convergence than Tanh)
- 2 hidden layers of 256 units (SB3 default)

```
Input (~20) → Linear(256) → ReLU → Linear(256) → ReLU → Output (6 thrusters)
```

### BC Policy
- MLP + Tanh (small, supervised training)
- May need to retrain BC with ReLU + larger network to match SAC architecture

---

## RL Algorithm

### Recommendation: SAC (Soft Actor-Critic)

| Algorithm | Type | Sample Efficiency | Stability | Verdict |
|---|---|---|---|---|
| SAC | Off-policy | High | Good | **Recommended** |
| PPO | On-policy | Low | Very stable | Good for sim only |
| TD3 | Off-policy | High | Good | Alternative to SAC |
| DDPG | Off-policy | High | Poor | Avoid |

### Why SAC
- Off-policy → replay buffer → reuses past experience
- Fewer real robot interactions needed (critical for tank time)
- Standard for continuous robot control in recent literature
- Handles continuous action spaces (thruster commands)

### SAC Networks
| Network | Purpose |
|---|---|
| Actor (policy) | Outputs action — MLP + ReLU |
| Critic (Q-function) | Estimates value — MLP + ReLU |

### Implementation
```python
from stable_baselines3 import SAC

# Start from pretrained BC weights
model = SAC("MlpPolicy", env)
model.set_parameters(bc_policy)
model.learn(total_timesteps=50_000)
```

---

## Simulation Environment

### Options
- **UUV Simulator** — ROS-based, has BlueROV2, most complete
- **Custom gym wrapper** — wrap Fossen dynamics into `gymnasium.Env` (recommended)

### Custom Gym (Recommended)
- Full control over reward shaping
- Easy domain randomisation
- No ROS dependency during training
- Fast simulation

```python
class ROVEnv(gymnasium.Env):
    def step(self, action):
        # Fossen dynamics here
        ...
    def reset(self):
        ...
```

---

## Sim-to-Real

### Strategy: Domain Randomisation
- Randomise drag, mass, thruster efficiency during RL training
- Makes policy robust to model error
- Bridges sim-to-real gap

### Online RL Fine-tuning on Real Hardware
- BC policy keeps ROV stable during fine-tuning
- No dangerous exploration
- Conservative action bounds prevent crashes
- Limited tank sessions sufficient (SAC sample efficiency)

### ROS Deployment
```
Trained SB3 policy → ROS node → BlueROV2 thrusters
```

---

## Timeline (9 months total)

### Unofficial Period (2-3 months before registration)
- Familiarise with existing NMPC implementation
- Understand coupled dynamics limitations
- Setup ROS and gym environment
- Start BC pipeline

### Official Registration (6 months)

| Month | Task |
|---|---|
| 1 | Gym wrapper + BC data collection |
| 2 | BC training + simulation validation |
| 3 | SAC RL fine-tuning in simulation |
| 4 | ROS deployment + initial hardware tests |
| 5 | Full tank experiments + robustness testing |
| 6 | Thesis writing + presentation |

### Notes
- Register officially once system is understood and working in sim
- 3-month extension available if needed (hardware experiments often take longer)
- With rocket engine background: saves ~1-2 months vs student starting from scratch

---

## Comparison: Rocket Engine vs ROV Thesis

| Factor | Rocket Engine (current) | ROV + Manipulator (thesis) |
|---|---|---|
| State dimension | 3 | 20+ |
| Control dimension | 1 | 6 + arm joints |
| Dynamics | Slow, stable | Fast, coupled, nonlinear |
| Covariate shift | Minimal (BC sufficient) | Significant (RL needed) |
| Model mismatch | Small | Large (underwater drag, coupling) |
| BC alone sufficient? | Yes | No |
| RL needed? | No | Yes |

---

## Key Insight from Rocket Engine Project

> "BC with closed-loop feedback is sufficient for low-dimensional, stable control tasks.
> For higher-dimensional systems with fast dynamics or significant model mismatch
> (e.g. 6-DOF underwater vehicles with manipulators), RL fine-tuning is necessary."

This directly motivates the ROV thesis as a natural extension.
