# Behavior Cloning for Monopropellant Rocket Engine Control
## Internship Report

---

## Table of Contents

**1. Introduction**
- 1.1 Motivation
- 1.2 Objectives
- 1.3 Report Structure

**2. System Description** *(Felix's work)*
- 2.1 Physical System Overview
- 2.2 System Parameters
- 2.3 State and Control Variables
- 2.4 Operating Constraints

**3. Mathematical Modelling** *(Felix's work)*
- 3.1 Full-Order Model (4-State)
- 3.2 Reduced-Order Model (1-State)
- 3.3 Steady-State Equations
- 3.4 Characteristic Velocity (c*)
- 3.5 Pipe Friction and Flow Model

**4. Nonlinear Model Predictive Control (NMPC)** *(Felix's work)*
- 4.1 Problem Formulation
- 4.2 Prediction Horizon and Sampling Time
- 4.3 Cost Function
- 4.4 Constraints
- 4.5 CasADi / IPOPT Implementation
- 4.6 Reactive vs Lookahead Control

**5. Behaviour Cloning**
- 5.1 Imitation Learning Overview
- 5.2 Expert Policy: Reactive NMPC
- 5.3 Neural Network Architecture
- 5.4 Input-Output Mapping
- 5.5 Data Normalisation

**6. Training Data Collection**
- 6.1 Episode Design
- 6.2 Zero-Order Hold Reference Generation
- 6.3 Operating Range and kv Bounds
- 6.4 Parallel Data Collection
- 6.5 Dataset Statistics

**7. Neural Network Training**
- 7.1 Loss Function: Weighted MSE
- 7.2 Optimiser and Learning Rate Schedule
- 7.3 Early Stopping
- 7.4 Train / Test Split Strategy

**8. Results**
- 8.1 Training Curves
- 8.2 Prediction Accuracy
- 8.3 Closed-Loop Validation vs NMPC
- 8.4 Out-of-Distribution Testing
- 8.5 Computational Performance

**9. Discussion**
- 9.1 Covariate Shift Analysis
- 9.2 Limitations of Behaviour Cloning
- 9.3 Effect of Network Size
- 9.4 kv–p_c Mapping Linearity

**10. Future Work**
- 10.1 DAgger: Dataset Aggregation
- 10.2 Reinforcement Learning Fine-Tuning
- 10.3 Bipropellant Extension
- 10.4 Hardware Deployment

**11. Conclusion**

**References**

**Appendix**
- A. Physical Parameters
- B. MPC Parameters
- C. Neural Network Hyperparameters
- D. Dataset Summary
