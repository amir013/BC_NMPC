# -*- coding: utf-8 -*-
"""
Covariate Shift Test 2 — Measurement Noise on p_c

Adds Gaussian noise to the chamber pressure measurement fed to both
NN and NMPC controllers. Tests three noise levels to show how the NN
degrades faster than NMPC as noise increases.

Training distribution: clean p_c measurements
OOD trigger: noisy p_c input (never seen during training)
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
P_BC   = 30e5
DP     = 0.01
LP     = 0.5
VC     = 1e-3
TV     = 0.1
DTH    = 0.03

P_MIN = 9.0e5
P_MAX = 15.5e5

# Noise levels to test (bar → Pa)
NOISE_LEVELS = [0.1e5, 0.3e5, 0.5e5]   # 0.1, 0.3, 0.5 bar std


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


def run_simulation(controller, sim, y0, pc_ref_traj, noise_std=0.0,
                   is_nmpc=False, mpc_obj=None, mpc_model=None):
    """Run simulation with optional measurement noise on p_c."""
    N         = len(pc_ref_traj)
    sim_steps = int(np.ceil(H_MPC / H_SIM))

    states         = np.zeros((4, N * sim_steps))
    control_inputs = np.zeros(N)
    states[:, 0]   = y0
    kv_prev        = y0[2]

    for k in range(N - N_PRED):
        p_c_true = y0[3]
        p_c_meas = p_c_true + np.random.normal(0, noise_std)   # noisy measurement
        p_c_ref  = pc_ref_traj[k]

        if is_nmpc:
            ygoal = np.array([[p_c_ref] * (N_PRED + 1)])
            c_star, rho_p, eta_p, sos_p, zeta_p = helper(p_c_meas, sim)
            params = [sim.Ath, c_star, P_BC, rho_p, zeta_p, TV]
            _, ctrl_input, _ = mpc_obj.solve(
                mpc_model.xdot_func, mpc_model.x, mpc_model.u,
                mpc_model.n_params, params,
                u0=[kv_prev], x0=np.array([p_c_meas]), xref=ygoal,
                actor_constraint={'ubg': [1], 'lbg': [0.1]},
                actor_rate={'ubg': [1.43], 'lbg': [-1.43]}
            )
            kv_set = float(ctrl_input)
        else:
            kv_set = controller.get_action(p_c_meas, p_c_ref, kv_prev)

        control_inputs[k] = kv_set
        sim_sol = sim.integrate(y0, kv_set, sim_steps)
        states[:, 1 + k*sim_steps : 1 + (k+1)*sim_steps] = sim_sol[:, 1:]
        y0      = sim_sol[:, -1]
        kv_prev = kv_set

    return states, control_inputs


def generate_zoh_trajectory(p_c0, kv_min=0.405, kv_max=0.841):
    """Generate ZOH reference trajectory extended by T_PRED for full 5s control."""
    sim = Monoprop_sim(H_SIM, TV, DP, LP, VC, DTH, FLUID, P_BC, T_PROP)
    ig  = [0.5, 0.5, 25e5, 20e5, 0.7, 100, 1800]

    N = int(TRAJ_DURATION / H_MPC) + 1
    t = np.linspace(0, TRAJ_DURATION, N)
    pc_ref = np.ones(N) * p_c0

    t_curr = np.random.uniform(0.3, 0.6)
    while t_curr < DURATION - 1.0:
        kv_next = np.random.uniform(kv_min, kv_max)
        sol = sim.steady_solution(kv_next, ig)
        p_next = sol.x[3] if sol.success else pc_ref[int(t_curr / H_MPC)]
        idx = np.where(t >= t_curr)[0]
        pc_ref[idx] = p_next
        t_curr += np.random.uniform(0.8, 1.5)

    return pc_ref


if __name__ == "__main__":
    np.random.seed(42)

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

    # Fixed episode — same trajectory for all noise levels for fair comparison
    kv_init    = 0.60
    steady_sol = sim.steady_solution(kv_init, ig).x
    p_c0       = steady_sol[3]
    y0         = np.array([steady_sol[0], steady_sol[2], steady_sol[4], p_c0])
    pc_ref_traj = generate_zoh_trajectory(p_c0)

    N             = len(pc_ref_traj)
    sim_steps     = int(np.ceil(H_MPC / H_SIM))
    controlled_pts = (N - N_PRED) * sim_steps
    t_sim  = np.linspace(0, DURATION, controlled_pts)
    t_ctrl = np.linspace(0, DURATION, N - N_PRED)
    t_ref  = np.linspace(0, DURATION, N - N_PRED)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Covariate Shift Test 2 — Measurement Noise on p_c', fontsize=13)

    for i, noise_std in enumerate(NOISE_LEVELS):
        noise_bar = noise_std / 1e5
        print(f"Running noise = {noise_bar:.1f} bar std...")

        np.random.seed(42)   # same noise seed per level for reproducibility
        s_nn,   c_nn   = run_simulation(nn_controller, sim, y0.copy(), pc_ref_traj,
                                        noise_std=noise_std)
        np.random.seed(42)
        s_nmpc, c_nmpc = run_simulation(None, sim, y0.copy(), pc_ref_traj,
                                        noise_std=noise_std,
                                        is_nmpc=True, mpc_obj=mpc_obj, mpc_model=mpc_model)

        # RMSE
        nn_rmse   = np.sqrt(np.mean((s_nn[3,   :controlled_pts] - np.interp(t_sim, t_ref, pc_ref_traj[:N-N_PRED]))**2)) / 1e5
        nmpc_rmse = np.sqrt(np.mean((s_nmpc[3, :controlled_pts] - np.interp(t_sim, t_ref, pc_ref_traj[:N-N_PRED]))**2)) / 1e5

        # Pressure plot
        ax = axes[i, 0]
        ax.plot(t_sim, s_nn[3,   :controlled_pts]/1e5, 'b-',  label=f'NN   (RMSE={nn_rmse:.3f} bar)',   linewidth=1.5)
        ax.plot(t_sim, s_nmpc[3, :controlled_pts]/1e5, 'r--', label=f'NMPC (RMSE={nmpc_rmse:.3f} bar)', linewidth=1.5)
        ax.step(t_ref, pc_ref_traj[:N-N_PRED]/1e5, 'g:', label='Ref', linewidth=2, where='post')
        ax.axhline(P_MAX/1e5, color='orange', ls='--', lw=1, alpha=0.7, label='P_MAX')
        ax.axhline(P_MIN/1e5, color='orange', ls='--', lw=1, alpha=0.7, label='P_MIN')
        ax.set_ylabel('Pressure [bar]')
        ax.set_title(f'Noise = {noise_bar:.1f} bar std')
        ax.legend(fontsize=8)
        ax.grid(True)

        # Valve command plot
        ax2 = axes[i, 1]
        ax2.step(t_ctrl, c_nn[:N-N_PRED],   'b-',  label='NN kv',   linewidth=1.5, where='post')
        ax2.step(t_ctrl, c_nmpc[:N-N_PRED], 'r--', label='NMPC kv', linewidth=1.5, where='post')
        ax2.axhline(1.0, c='k', ls='--', alpha=0.5)
        ax2.axhline(0.1, c='k', ls='--', alpha=0.5)
        ax2.set_ylabel('kv [-]')
        ax2.set_title(f'Valve Command — Noise = {noise_bar:.1f} bar')
        ax2.legend(fontsize=8)
        ax2.grid(True)

    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].set_xlabel('Time [s]')

    plt.tight_layout()
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'covariate_shift_2.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved: {plot_path}")
