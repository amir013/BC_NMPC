import numpy as np
import matplotlib.pyplot as plt

def nmpc_monoprop(states, control_inputs, timesteps, sim_steps, knot_points,
                  actor_constraint, actor_rate, h_mpc):

    t_sim = np.linspace(
        timesteps[0],
        timesteps[-1],
        states.shape[1]
    )

    # reference path
    p_c_ref = knot_points["pc_ref"]/1e5
    t_i = knot_points["time_interval"]
    t = knot_points["time"]
    p_c_ref = [p_c_ref[t_i[ii], ii] for ii in range(len(t_i))]

    # simulated state evolution
    p_c_bar = states[3, :] * 1e-5
    
    # control inputs
    u = control_inputs
    u_lb = actor_constraint["lbg"][0]
    u_ub = actor_constraint["ubg"][0]

    # control rate
    du = np.diff(u) / h_mpc

    du_lb = actor_rate["lbg"][0]
    du_ub = actor_rate["ubg"][0]
    
    # only plot arrays until index where simulation was executed
    idx_end = np.min(np.where(states[3,:]==0))
    t_end = t_sim[idx_end]
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    idx_end_ref = find_nearest(t, t_end)

    fig, axs = plt.subplots(3, 1, figsize=(6, 7), sharex=True)

    axs[0].plot(t_sim[:idx_end], p_c_bar[:idx_end], label=r"$p_c$")
    axs[0].step(t[:idx_end_ref], p_c_ref[:idx_end_ref], "--", label=r"$p_c^{ref}$", where='post')
    axs[0].set_ylabel("Pressure [bar]")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].step(t[:idx_end_ref], u[:idx_end_ref], where="post", label=r"$k_v$")
    axs[1].axhline(u_ub, linestyle="--", color="k", linewidth=0.8, label="bounds")
    axs[1].axhline(u_lb, linestyle="--", color="k", linewidth=0.8)
    axs[1].set_ylabel(r"$k_v$ [-]")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].step(t[:idx_end_ref-1], du[:idx_end_ref-1], where="post", label=r"$\dot{k}_v$")
    axs[2].axhline(du_ub, linestyle="--", color="k", linewidth=0.8, label="bounds")
    axs[2].axhline(du_lb, linestyle="--", color="k", linewidth=0.8)
    axs[2].set_ylabel(r"$\dot{k}_v$ [1/s]")
    axs[2].set_xlabel("Time [s]")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()