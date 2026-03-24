# -*- coding: utf-8 -*-
"""
Data Collection Script for Behavior Cloning (v6)

Features:
  - Pure ZOH step references matching Figure_1.png shape
  - kv-based steady-state targets within [9.0, 15.5] bar
  - Reactive NMPC: current reference only — deployable on real system
  - Parallel processing: each worker initializes its own CasADi objects
  - File Naming: Saves to bc_dataset_v6.npz
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time
import gc
from multiprocessing import Pool, cpu_count
from scipy.optimize import brentq

# --- Configuration ---
N_EPISODES   = 125
DURATION     = 5.0
H_MPC        = 0.05
H_SIM        = 1e-4
T_PRED       = 1.0
N_PRED       = int(T_PRED / H_MPC)
STEPS_PER_EP = int((DURATION - T_PRED) / H_MPC)

# Physical parameters
T_PROP = 293
FLUID  = 'C2H6O'
P_BC   = 30e5
DP     = 0.01
LP     = 0.5
VC     = 1e-3
TV     = 0.1
DTH    = 0.03


def _make_objects():
    """Initialize CasADi sim/mpc objects — called inside each worker process."""
    from model import Monoprop, Monoprop_sim
    from mpc import NMPC

    sim       = Monoprop_sim(H_SIM, TV, DP, LP, VC, DTH, FLUID, P_BC, T_PROP)
    mpc_model = Monoprop(1.0, TV, DP, LP, VC, DTH, FLUID, P_BC, T_PROP)
    mpc_model()
    mpc = NMPC(N_PRED, H_MPC, np.eye(1), np.eye(1)*1e-3, np.eye(1))
    return sim, mpc, mpc_model


def _get_kv_bounds():
    """Compute kv range corresponding to [9.0, 15.5] bar steady-state."""
    sim, _, _ = _make_objects()
    ig = [0.5, 0.5, 25e5, 20e5, 0.7, 100, 1800]

    def p_ss(kv):
        sol = sim.steady_solution(kv, ig)
        return sol.x[3] / 1e5

    kv_min = brentq(lambda kv: p_ss(kv) - 9.0,  0.34, 0.60)
    kv_max = brentq(lambda kv: p_ss(kv) - 15.5, 0.70, 0.95)
    return kv_min, kv_max


def _run_worker(args):
    """Worker function: runs one episode. Initializes its own CasADi objects."""
    ep_idx, kv_init, seed, kv_min, kv_max = args
    np.random.seed(seed)

    sim, mpc, mpc_model = _make_objects()
    ig = [0.5, 0.5, 25e5, 20e5, 0.7, 100, 1800]

    # --- Generate ZOH step reference ---
    N = int(DURATION / H_MPC) + 1
    t = np.linspace(0, DURATION, N)

    steady_sol = sim.steady_solution(kv_init, ig)
    p_c0 = steady_sol.x[3]
    pc_ref = np.ones(N) * p_c0

    t_curr = np.random.uniform(0.3, 0.6)
    while t_curr < DURATION - 1.0:
        kv_next = np.random.uniform(kv_min, kv_max)
        sol = sim.steady_solution(kv_next, ig)
        p_next = sol.x[3] if sol.success else pc_ref[int(t_curr / H_MPC)]
        idx = np.where(t >= t_curr)[0]
        pc_ref[idx] = p_next
        t_curr += np.random.uniform(0.8, 1.5)

    # --- Run NMPC episode ---
    y0 = np.array([steady_sol.x[0], steady_sol.x[2], steady_sol.x[4], steady_sol.x[3]])
    kv_set_prev = kv_init
    episode_data = []
    sim_steps = int(np.ceil(H_MPC / H_SIM))

    for k in range(STEPS_PER_EP):
        p_c_curr = y0[3]
        ygoal = np.array([[pc_ref[k]] * (N_PRED + 1)])   # reactive: hold current ref constant

        c_star = sim.model_func(p_c_curr, a=5.71725407e-01, b=3.15515423e-01, c=1.53707137e+03)
        mdot   = p_c_curr * sim.Ath / c_star
        rho_p, eta_p, sos_p = sim.fluid_properties(FLUID, P_BC, T_PROP)
        Re     = sim.reynolds(mdot, DP, eta_p)
        zeta_p = sim.calc_zeta(Re, LP, DP, sim.Ap)
        params = [sim.Ath, c_star, P_BC, rho_p, zeta_p, TV]

        _, ctrl_input, _ = mpc.solve(
            mpc_model.xdot_func, mpc_model.x, mpc_model.u,
            mpc_model.n_params, params,
            u0=[kv_set_prev], x0=np.array([p_c_curr]), xref=ygoal,
            actor_constraint={'ubg': [1], 'lbg': [0.1]},
            actor_rate={'ubg': [1.43], 'lbg': [-1.43]}
        )

        kv_set = float(ctrl_input)
        episode_data.append({
            'p_c':     p_c_curr,
            'p_c_ref': pc_ref[k],
            'kv_prev': kv_set_prev,
            'kv_set':  kv_set
        })

        kv_set_prev = kv_set
        sim_sol = sim.integrate(y0, kv_set, sim_steps)
        y0 = sim_sol[:, -1]

    print(f"  Episode {ep_idx+1} done ({len(episode_data)} steps)")
    return episode_data


def collect_dataset(n_episodes=125, n_workers=None):
    """Collect dataset in parallel using multiprocessing."""
    if n_workers is None:
        n_workers = 8

    os.makedirs('dataset', exist_ok=True)

    print(f"Computing kv bounds...")
    kv_min, kv_max = _get_kv_bounds()
    print(f"kv range for [9.0, 15.5] bar: [{kv_min:.4f}, {kv_max:.4f}]")

    kv_inits = np.random.uniform(0.40, 0.80, n_episodes)
    seeds    = np.random.randint(0, 100000, n_episodes)
    args     = [(i, kv_inits[i], seeds[i], kv_min, kv_max) for i in range(n_episodes)]

    print(f"Collecting {n_episodes} episodes using {n_workers} workers...")
    t_start = time.time()

    with Pool(processes=n_workers) as pool:
        results = pool.map(_run_worker, args)

    t_end = time.time()
    print(f"Collection took {t_end - t_start:.1f}s")

    all_data = [d for ep in results for d in ep]

    dataset = {
        'p_c':     np.array([d['p_c']     for d in all_data]),
        'p_c_ref': np.array([d['p_c_ref'] for d in all_data]),
        'kv_prev': np.array([d['kv_prev'] for d in all_data]),
        'kv_set':  np.array([d['kv_set']  for d in all_data]),
    }
    return dataset


if __name__ == "__main__":
    dataset = collect_dataset(n_episodes=N_EPISODES)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, 'bc_dataset_v6.npz')

    np.savez(data_path,
             p_c=dataset['p_c'],
             p_c_ref=dataset['p_c_ref'],
             kv_prev=dataset['kv_prev'],
             kv_set=dataset['kv_set'])

    print(f"\nCollection complete!")
    print(f"Total samples: {len(dataset['p_c'])}")
    print(f"Saved: {data_path}")
