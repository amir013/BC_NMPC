# -*- coding: utf-8 -*-
"""
Validate trained NN on the 4-state simulation (v6)
Compares NN controller vs Reactive NMPC controller
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from model import Monoprop_sim, Monoprop
from mpc import NMPC

# Configuration (must match collect_data.py v6)
H_SIM = 1e-4
H_MPC = 0.05
T_PRED = 1.0
N_PRED = int(T_PRED / H_MPC)
DURATION = 5.0
TRAJ_DURATION = DURATION + T_PRED
STEPS_PER_EP = int(DURATION / H_MPC)

T_PROP = 293
FLUID = 'C2H6O'
P_BC = 30e5
DP = 0.01
LP = 0.5
VC = 1e-3
TV = 0.1
DTH = 0.03

# Same architecture as training
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
    def __init__(self, model_path='policy_model.pth'):
        checkpoint = torch.load(model_path, weights_only=False)
        self.model = PolicyNetwork()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.p_c_mean = checkpoint['p_c_mean']
        self.p_c_std = checkpoint['p_c_std']
        self.p_ref_mean = checkpoint['p_ref_mean']
        self.p_ref_std = checkpoint['p_ref_std']

    def get_action(self, p_c, p_c_ref, kv_prev):
        """Get NN action with normalization matching training."""
        p_c_norm = (p_c - self.p_c_mean) / self.p_c_std
        p_c_ref_norm = (p_c_ref - self.p_ref_mean) / self.p_ref_std
        x = torch.FloatTensor([[p_c_norm, p_c_ref_norm, kv_prev]])
        with torch.no_grad():
            kv_set = self.model(x).item()
        # Rate limit safety
        max_change = 1.43 * H_MPC
        return np.clip(kv_set, kv_prev - max_change, kv_prev + max_change)

def helper(p_c_curr, sim):
    """Update NMPC parameters based on current pressure."""
    c_star = sim.model_func(p_c_curr, a=5.71725407e-01, b=3.15515423e-01, c=1.53707137e+03)
    mdot = p_c_curr * sim.Ath / c_star
    rho_p, eta_p, sos_p = sim.fluid_properties(FLUID, P_BC, T_PROP)
    Re = sim.reynolds(mdot, DP, eta_p)
    zeta_p = sim.calc_zeta(Re, LP, DP, sim.Ap)
    return c_star, rho_p, eta_p, sos_p, zeta_p


def run_simulation(controller, sim, y0, pc_ref_traj, is_nmpc=False, mpc_obj=None, mpc_model=None):
    """Run simulation with either NN or reactive NMPC controller."""
    N = len(pc_ref_traj)
    sim_steps = int(np.ceil(H_MPC / H_SIM))

    states = np.zeros((4, N * sim_steps))
    control_inputs = np.zeros(N)
    solve_times = []
    states[:, 0] = y0
    kv_prev = y0[2]

    for k in range(N - N_PRED):
        p_c_curr = y0[3]
        p_c_ref = pc_ref_traj[k]

        if is_nmpc:
            # Reactive NMPC: hold current reference constant
            ygoal = np.array([[p_c_ref] * (N_PRED + 1)])
            c_star, rho_p, eta_p, sos_p, zeta_p = helper(p_c_curr, sim)
            params = [sim.Ath, c_star, P_BC, rho_p, zeta_p, TV]
            t0 = time.perf_counter()
            _, ctrl_input, _ = mpc_obj.solve(
                mpc_model.xdot_func, mpc_model.x, mpc_model.u,
                mpc_model.n_params, params,
                u0=[kv_prev], x0=np.array([p_c_curr]), xref=ygoal,
                actor_constraint={'ubg': [1], 'lbg': [0.1]},
                actor_rate={'ubg': [1.43], 'lbg': [-1.43]}
            )
            solve_times.append((time.perf_counter() - t0) * 1000)
            kv_set = float(ctrl_input)
        else:
            t0 = time.perf_counter()
            kv_set = controller.get_action(p_c_curr, p_c_ref, kv_prev)
            solve_times.append((time.perf_counter() - t0) * 1000)

        control_inputs[k] = kv_set
        sim_sol = sim.integrate(y0, kv_set, sim_steps)
        states[:, 1 + k*sim_steps : 1 + (k+1)*sim_steps] = sim_sol[:, 1:]
        y0 = sim_sol[:, -1]
        kv_prev = kv_set

    return states, control_inputs, np.array(solve_times)

def generate_zoh_trajectory(p_c0, kv_min=0.405, kv_max=0.841):
    """Generate ZOH step reference trajectory extended by T_PRED for full 5s control."""
    sim = Monoprop_sim(H_SIM, TV, DP, LP, VC, DTH, FLUID, P_BC, T_PROP)
    ig = [0.5, 0.5, 25e5, 20e5, 0.7, 100, 1800]

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
    sim = Monoprop_sim(H_SIM, TV, DP, LP, VC, DTH, FLUID, P_BC, T_PROP)
    mpc_model = Monoprop(1.0, TV, DP, LP, VC, DTH, FLUID, P_BC, T_PROP)
    mpc_model()
    mpc_obj = NMPC(N_PRED, H_MPC, np.eye(1), np.eye(1)*1e-3, np.eye(1))

    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'models', 'policy_model.pth')
        nn_controller = NNController(model_path)
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Train the NN first.")
        exit()

    # Test on 3 random episodes with ZOH references
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    all_nn_times, all_nmpc_times = [], []
    for i in range(3):
        kv_init = np.random.uniform(0.40, 0.80)
        initial_guess = [0.5, 0.5, 25e5, 20e5, 0.7, 100, 1800]
        steady_sol = sim.steady_solution(kv_init, initial_guess).x
        y0 = np.array([steady_sol[0], steady_sol[2], steady_sol[4], steady_sol[3]])

        pc_ref_traj = generate_zoh_trajectory(steady_sol[3])

        s_nn, c_nn, t_nn = run_simulation(nn_controller, sim, y0.copy(), pc_ref_traj)
        s_nmpc, c_nmpc, t_nmpc = run_simulation(None, sim, y0.copy(), pc_ref_traj, is_nmpc=True, mpc_obj=mpc_obj, mpc_model=mpc_model)
        all_nn_times.extend(t_nn)
        all_nmpc_times.extend(t_nmpc)

        sim_steps      = int(np.ceil(H_MPC / H_SIM))
        controlled_pts = (len(pc_ref_traj) - N_PRED) * sim_steps
        t_sim = np.linspace(0, DURATION, controlled_pts)
        axes[i].plot(t_sim, s_nn[3,   :controlled_pts]/1e5, 'b-',  label='NN',   linewidth=2)
        axes[i].plot(t_sim, s_nmpc[3, :controlled_pts]/1e5, 'r--', label='NMPC', linewidth=2)
        axes[i].plot(np.linspace(0, DURATION, len(pc_ref_traj) - N_PRED), pc_ref_traj[:len(pc_ref_traj) - N_PRED]/1e5, 'g:', label='Ref', linewidth=2)
        axes[i].set_title(f"Validation Episode {i+1} (v6 - Reactive NMPC)", fontsize=14)
        axes[i].set_ylabel('Pressure [bar]', fontsize=13)
        axes[i].tick_params(labelsize=12)
        axes[i].legend(fontsize=12)
        axes[i].grid(True)

    axes[-1].set_xlabel('Time [s]', fontsize=13)
    plt.tight_layout()

    # Print timing statistics
    nn_times = np.array(all_nn_times)
    nmpc_times = np.array(all_nmpc_times)
    print("\n--- Inference Timing (ms) ---")
    print(f"NN   — mean: {nn_times.mean():.4f}, std: {nn_times.std():.4f}, min: {nn_times.min():.4f}, max: {nn_times.max():.4f}")
    print(f"NMPC — mean: {nmpc_times.mean():.2f}, std: {nmpc_times.std():.2f}, min: {nmpc_times.min():.2f}, max: {nmpc_times.max():.2f}")
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'validation_v6.png')
    plt.savefig(plot_path)
    print(f"Saved: {plot_path}")
