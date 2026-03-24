# -*- coding: utf-8 -*-
"""
Covariate Shift Test

Initializes p_c OUTSIDE training range with reference INSIDE training range.
The NN sees an OOD p_c input while being asked to track a normal reference.

Training range: p_c in [9.0, 15.5] bar
OOD initial:    p_c above 15.5 bar (high) or below 9.0 bar (low)
Reference:      fixed at 12.0 bar (center of training range)
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
H_SIM   = 1e-4
H_MPC   = 0.05
T_PRED  = 1.0
N_PRED  = int(T_PRED / H_MPC)
DURATION      = 5.0
TRAJ_DURATION = DURATION + T_PRED

T_PROP = 293
FLUID  = 'C2H6O'
P_BC   = 30e5
DP     = 0.01
LP     = 0.5
VC     = 1e-3
TV     = 0.1
DTH    = 0.03

P_MIN  = 9.0e5    # training range lower bound
P_MAX  = 15.5e5   # training range upper bound
KV_MIN = 0.405    # training range lower bound (kv)
KV_MAX = 0.841    # training range upper bound (kv)


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


def helper(p_c_curr, sim):
    c_star = sim.model_func(p_c_curr, a=5.71725407e-01, b=3.15515423e-01, c=1.53707137e+03)
    mdot   = p_c_curr * sim.Ath / c_star
    rho_p, eta_p, sos_p = sim.fluid_properties(FLUID, P_BC, T_PROP)
    Re     = sim.reynolds(mdot, DP, eta_p)
    zeta_p = sim.calc_zeta(Re, LP, DP, sim.Ap)
    return c_star, rho_p, eta_p, sos_p, zeta_p


def run_simulation(controller, sim, y0, pc_ref_traj, is_nmpc=False, mpc_obj=None, mpc_model=None):
    N         = len(pc_ref_traj)
    sim_steps = int(np.ceil(H_MPC / H_SIM))

    states         = np.zeros((4, N * sim_steps))
    control_inputs = np.zeros(N)
    states[:, 0]   = y0
    kv_prev        = y0[2]

    for k in range(N - N_PRED):
        p_c_curr = y0[3]
        p_c_ref  = pc_ref_traj[k]

        if is_nmpc:
            ygoal = np.array([[p_c_ref] * (N_PRED + 1)])
            c_star, rho_p, eta_p, sos_p, zeta_p = helper(p_c_curr, sim)
            params = [sim.Ath, c_star, P_BC, rho_p, zeta_p, TV]
            _, ctrl_input, _ = mpc_obj.solve(
                mpc_model.xdot_func, mpc_model.x, mpc_model.u,
                mpc_model.n_params, params,
                u0=[kv_prev], x0=np.array([p_c_curr]), xref=ygoal,
                actor_constraint={'ubg': [1], 'lbg': [0.1]},
                actor_rate={'ubg': [1.43], 'lbg': [-1.43]}
            )
            kv_set = float(ctrl_input)
        else:
            kv_set = controller.get_action(p_c_curr, p_c_ref, kv_prev)

        control_inputs[k] = kv_set
        sim_sol = sim.integrate(y0, kv_set, sim_steps)
        states[:, 1 + k*sim_steps : 1 + (k+1)*sim_steps] = sim_sol[:, 1:]
        y0      = sim_sol[:, -1]
        kv_prev = kv_set

    return states, control_inputs


if __name__ == "__main__":
    sim       = Monoprop_sim(H_SIM, TV, DP, LP, VC, DTH, FLUID, P_BC, T_PROP)
    mpc_model = Monoprop(1.0, TV, DP, LP, VC, DTH, FLUID, P_BC, T_PROP)
    mpc_model()
    mpc_obj = NMPC(N_PRED, H_MPC, np.eye(1), np.eye(1)*1e-3, np.eye(1))

    model_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'models', 'policy_model.pth')
    try:
        nn_controller = NNController(model_path)
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Train the NN first.")
        exit()

    ig = [0.5, 0.5, 25e5, 20e5, 0.7, 100, 1800]

    # OOD: kv_init outside [KV_MIN, KV_MAX] AND reference steps outside [P_MIN, P_MAX]
    # Each scenario: (kv_init, [(t_step, p_ref_bar), ...])
    scenarios = [
        (0.90, [(0.5, 12.0), (1.5, 20.0), (3.0, 12.0)]),   # kv OOD HIGH, ref OOD HIGH
        (0.35, [(0.5, 12.0), (1.5,  5.0), (3.0, 12.0)]),   # kv OOD LOW,  ref OOD LOW
        (0.93, [(0.5, 20.0), (2.0,  5.0), (3.5, 20.0)]),   # kv OOD HIGH, ref both OOD
    ]

    N      = int(TRAJ_DURATION / H_MPC) + 1
    t_traj = np.linspace(0, TRAJ_DURATION, N)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'Covariate Shift Test — OOD kv_init & OOD Reference Steps\n'
                 f'Training range: kv ∈ [{KV_MIN}, {KV_MAX}], p_c ∈ [{P_MIN/1e5}, {P_MAX/1e5}] bar', fontsize=12)

    for i, (kv_init, steps) in enumerate(scenarios):
        steady_sol = sim.steady_solution(kv_init, ig).x
        p_c0       = steady_sol[3]
        y0         = np.array([steady_sol[0], steady_sol[2], steady_sol[4], p_c0])

        # Build ZOH OOD reference trajectory
        pc_ref_traj = np.ones(N) * p_c0
        for t_step, p_bar in steps:
            pc_ref_traj[int(t_step / H_MPC):] = p_bar * 1e5

        kv_ood = 'OOD HIGH' if kv_init > KV_MAX else 'OOD LOW'
        print(f"Test {i+1}: kv_init={kv_init:.2f} ({kv_ood}), p_c0={p_c0/1e5:.2f} bar")

        s_nn,   c_nn   = run_simulation(nn_controller, sim, y0.copy(), pc_ref_traj)
        s_nmpc, c_nmpc = run_simulation(None, sim, y0.copy(), pc_ref_traj,
                                        is_nmpc=True, mpc_obj=mpc_obj, mpc_model=mpc_model)

        # Only plot up to controlled steps (N - N_PRED), last 1s is uncontrolled
        sim_steps      = int(np.ceil(H_MPC / H_SIM))
        controlled_pts = (N - N_PRED) * sim_steps
        t_sim = np.linspace(0, DURATION, controlled_pts)

        # Pressure plot
        ax = axes[i, 0]
        ax.plot(t_sim, s_nn[3,   :controlled_pts]/1e5, 'b-',  label='NN',   linewidth=2)
        ax.plot(t_sim, s_nmpc[3,:controlled_pts]/1e5, 'r--', label='NMPC', linewidth=2)
        ax.step(t_traj[:N - N_PRED], pc_ref_traj[:N - N_PRED]/1e5, 'g:', label='Ref', linewidth=2, where='post')
        ax.axhline(P_MAX/1e5, color='orange', ls='--', lw=1.5, label=f'P_MAX ({P_MAX/1e5} bar)')
        ax.axhline(P_MIN/1e5, color='orange', ls='--', lw=1.5, label=f'P_MIN ({P_MIN/1e5} bar)')
        ax.set_ylabel('Pressure [bar]')
        kv_ood = 'OOD HIGH' if kv_init > KV_MAX else 'OOD LOW'
        labels = ', '.join([f'{p} bar@{t}s' for t, p in steps])
        ax.set_title(f'Test {i+1}: kv_init={kv_init} ({kv_ood}), p_c0={p_c0/1e5:.1f} bar | ref: {labels}')
        ax.legend(fontsize=8)
        ax.grid(True)

        # Valve command plot
        ax2 = axes[i, 1]
        t_ctrl = np.linspace(0, DURATION, N - N_PRED)
        ax2.step(t_ctrl, c_nn[:N - N_PRED],   'b-',  label='NN kv',   linewidth=2, where='post')
        ax2.step(t_ctrl, c_nmpc[:N - N_PRED], 'r--', label='NMPC kv', linewidth=2, where='post')
        ax2.axhline(1.0, c='k', ls='--', alpha=0.5)
        ax2.axhline(0.1, c='k', ls='--', alpha=0.5)
        ax2.set_ylabel('kv [-]')
        ax2.set_title(f'Valve Command — Test {i+1}')
        ax2.legend(fontsize=8)
        ax2.grid(True)

    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].set_xlabel('Time [s]')

    plt.tight_layout()
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'covariate_shift_test.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved: {plot_path}")
