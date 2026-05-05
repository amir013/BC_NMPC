# -*- coding: utf-8 -*-
"""
Covariate Shift Test 3 — Unmodelled Tank Pressure Drop

Simulates fuel mass burn: tank pressure P_BC decreases over time in the real plant.
Neither NMPC nor NN knows about this — both assume P_BC = 30 bar (constant).

NMPC partially rejects it via closed-loop feedback every 50ms.
NN has no rejection mechanism — it outputs the same kv for the same (p_c, p_c_ref, kv_prev).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import Monoprop_sim, Monoprop
from mpc import NMPC

# Configuration
H_SIM         = 1e-4
H_MPC         = 0.05
T_PRED        = 1.0
N_PRED        = int(T_PRED / H_MPC)
DURATION      = 5.0
TRAJ_DURATION = DURATION + T_PRED

T_PROP = 293
FLUID  = 'C2H6O'
P_BC_0 = 30e5        # initial tank pressure [Pa]
DP     = 0.01
LP     = 0.5
VC     = 1e-3
TV     = 0.1
DTH    = 0.03

P_MIN = 9.0e5
P_MAX = 15.5e5


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=5, output_dim=1):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        raw = self.network(x)
        return 0.1 + 0.9 * torch.sigmoid(raw)


class NNController:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        self.model = PolicyNetwork()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.p_c_mean   = checkpoint['p_c_mean']
        self.p_c_std    = checkpoint['p_c_std']
        self.p_ref_mean = checkpoint['p_ref_mean']
        self.p_ref_std  = checkpoint['p_ref_std']

    def get_action(self, p_c, p_c_ref, kv_prev):
        p_c_norm     = (p_c     - self.p_c_mean)   / self.p_c_std
        p_c_ref_norm = (p_c_ref - self.p_ref_mean) / self.p_ref_std
        x = torch.FloatTensor([[p_c_norm, p_c_ref_norm, kv_prev]])
        with torch.no_grad():
            kv_set = self.model(x).item()
        max_change = 1.43 * H_MPC
        return np.clip(kv_set, kv_prev - max_change, kv_prev + max_change)


def helper(p_c_curr, sim, p_bc_assumed):
    """Compute NMPC parameters — always uses assumed (fixed) P_BC, not true plant P_BC."""
    c_star = sim.model_func(p_c_curr, a=5.71725407e-01, b=3.15515423e-01, c=1.53707137e+03)
    mdot   = p_c_curr * sim.Ath / c_star
    rho_p, eta_p, sos_p = sim.fluid_properties(FLUID, p_bc_assumed, T_PROP)
    Re     = sim.reynolds(mdot, DP, eta_p)
    zeta_p = sim.calc_zeta(Re, LP, DP, sim.Ap)
    return c_star, rho_p, eta_p, sos_p, zeta_p


def run_simulation(controller, sim, y0, pc_ref_traj, p_bc_drop_rate=0.0,
                   is_nmpc=False, mpc_obj=None, mpc_model=None):
    """
    Run closed-loop simulation with optional tank pressure drop.

    p_bc_drop_rate: rate at which true P_BC decreases [Pa/s]
                    NMPC and NN always assume P_BC = P_BC_0 (unmodelled disturbance)
    """
    N         = len(pc_ref_traj)
    sim_steps = int(np.ceil(H_MPC / H_SIM))

    states         = np.zeros((4, N * sim_steps))
    control_inputs = np.zeros(N)
    p_bc_log       = np.zeros(N)
    states[:, 0]   = y0
    kv_prev        = y0[2]

    for k in range(N - N_PRED):
        t_now    = k * H_MPC
        p_bc_true = P_BC_0 - p_bc_drop_rate * t_now   # true plant P_BC (decreasing)
        p_bc_true = max(p_bc_true, 20e5)               # floor at 20 bar (keep p_c in training range)
        sim.p_bc  = p_bc_true                          # inject into plant
        p_bc_log[k] = p_bc_true

        p_c_curr = y0[3]
        p_c_ref  = pc_ref_traj[k]

        if is_nmpc:
            # NMPC still uses P_BC_0 — it doesn't know about the drop
            ygoal = np.array([[p_c_ref] * (N_PRED + 1)])
            c_star, rho_p, eta_p, sos_p, zeta_p = helper(p_c_curr, sim, P_BC_0)
            params = [sim.Ath, c_star, P_BC_0, rho_p, zeta_p, TV]
            _, ctrl_input, _ = mpc_obj.solve(
                mpc_model.xdot_func, mpc_model.x, mpc_model.u,
                mpc_model.n_params, params,
                u0=[kv_prev], x0=np.array([p_c_curr]), xref=ygoal,
                actor_constraint={'ubg': [1], 'lbg': [0.1]},
                actor_rate={'ubg': [1.43], 'lbg': [-1.43]}
            )
            kv_set = float(ctrl_input)
        else:
            # NN also trained with P_BC_0 — no knowledge of drop
            kv_set = controller.get_action(p_c_curr, p_c_ref, kv_prev)

        control_inputs[k] = kv_set
        sim_sol = sim.integrate(y0, kv_set, sim_steps)
        states[:, 1 + k*sim_steps : 1 + (k+1)*sim_steps] = sim_sol[:, 1:]
        y0      = sim_sol[:, -1]
        kv_prev = kv_set

    sim.p_bc = P_BC_0   # restore
    return states, control_inputs, p_bc_log


if __name__ == "__main__":
    sim       = Monoprop_sim(H_SIM, TV, DP, LP, VC, DTH, FLUID, P_BC_0, T_PROP)
    mpc_model = Monoprop(1.0, TV, DP, LP, VC, DTH, FLUID, P_BC_0, T_PROP)
    mpc_model()
    mpc_obj = NMPC(N_PRED, H_MPC, np.eye(1), np.eye(1)*1e-3, np.eye(1))

    model_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'models', 'policy_model.pth')
    try:
        nn_controller = NNController(model_path)
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Train the NN first.")
        exit()

    ig = [0.5, 0.5, 25e5, 20e5, 0.7, 100, 1800]

    # Start at steady state for kv=0.60 (p_c ~ 12 bar, inside training range)
    kv_init    = 0.60
    steady_sol = sim.steady_solution(kv_init, ig).x
    p_c0       = steady_sol[3]
    y0         = np.array([steady_sol[0], steady_sol[2], steady_sol[4], p_c0])

    N      = int(TRAJ_DURATION / H_MPC) + 1
    t_traj = np.linspace(0, TRAJ_DURATION, N)

    # Fixed reference at 12 bar — inside training range
    P_REF       = 12.0e5
    pc_ref_traj = np.ones(N) * P_REF

    # Drop rates calibrated so p_c stays within training range [9, 15.5] bar
    # Reference fixed at 12 bar — disturbance causes drift but stays in-distribution
    drop_scenarios = [
        (0.0,    'No drop (baseline)'),
        (0.2e5,  'Moderate drop: 0.2 bar/s (30→29 bar over 5s)'),
        (0.5e5,  'Severe drop: 0.5 bar/s (30→27.5 bar over 5s)'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Covariate Shift Test 3 — Unmodelled Tank Pressure Drop\n'
                 'Neither NMPC nor NN knows P_BC is decreasing', fontsize=12)

    sim_steps      = int(np.ceil(H_MPC / H_SIM))
    controlled_pts = (N - N_PRED) * sim_steps
    t_sim  = np.linspace(0, DURATION, controlled_pts)
    t_ctrl = np.linspace(0, DURATION, N - N_PRED)

    for i, (drop_rate, label) in enumerate(drop_scenarios):
        print(f"\nTest {i+1}: {label}")

        s_nn,   c_nn,   pbc_nn   = run_simulation(
            nn_controller, sim, y0.copy(), pc_ref_traj, p_bc_drop_rate=drop_rate)
        s_nmpc, c_nmpc, pbc_nmpc = run_simulation(
            None, sim, y0.copy(), pc_ref_traj, p_bc_drop_rate=drop_rate,
            is_nmpc=True, mpc_obj=mpc_obj, mpc_model=mpc_model)

        rmse_nn   = np.sqrt(np.mean((s_nn[3,   :controlled_pts] - P_REF)**2)) / 1e5
        rmse_nmpc = np.sqrt(np.mean((s_nmpc[3, :controlled_pts] - P_REF)**2)) / 1e5
        print(f"  RMSE — NN: {rmse_nn:.3f} bar | NMPC: {rmse_nmpc:.3f} bar")

        # Pressure plot
        ax = axes[i, 0]
        ax.plot(t_sim, s_nn[3,   :controlled_pts]/1e5, 'b-',  label=f'NN (RMSE={rmse_nn:.3f} bar)',   lw=2)
        ax.plot(t_sim, s_nmpc[3, :controlled_pts]/1e5, 'r--', label=f'NMPC (RMSE={rmse_nmpc:.3f} bar)', lw=2)
        ax.axhline(P_REF/1e5,  color='g',      ls=':',  lw=2, label='Ref (12 bar)')
        ax.axhline(P_MAX/1e5,  color='orange',  ls='--', lw=1.5, label='P_MAX (15.5 bar)')
        ax.axhline(P_MIN/1e5,  color='orange',  ls='--', lw=1.5, label='P_MIN (9.0 bar)')
        # Show true P_BC on secondary axis
        ax2_twin = ax.twinx()
        p_bc_true = np.array([P_BC_0 - drop_rate * k * H_MPC for k in range(N - N_PRED)])
        p_bc_true = np.maximum(p_bc_true, 20e5)
        ax2_twin.plot(t_ctrl, p_bc_true/1e5, 'k:', lw=1, label='True P_BC')
        ax2_twin.set_ylabel('Tank P_BC [bar]', color='k', fontsize=8)
        ax2_twin.tick_params(axis='y', labelsize=7)
        ax.set_ylabel('Chamber Pressure [bar]')
        ax.set_title(f'Test {i+1}: {label}')
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(True)

        # Valve command plot
        ax3 = axes[i, 1]
        ax3.step(t_ctrl, c_nn[:N - N_PRED],   'b-',  label='NN kv',   lw=2, where='post')
        ax3.step(t_ctrl, c_nmpc[:N - N_PRED], 'r--', label='NMPC kv', lw=2, where='post')
        ax3.axhline(1.0, c='k', ls='--', alpha=0.5, lw=1)
        ax3.axhline(0.1, c='k', ls='--', alpha=0.5, lw=1)
        ax3.set_ylabel('kv [-]')
        ax3.set_title(f'Valve Command — Test {i+1}')
        ax3.legend(fontsize=8)
        ax3.grid(True)

    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].set_xlabel('Time [s]')

    plt.tight_layout()
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'covariate_shift_3.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved: {plot_path}")
