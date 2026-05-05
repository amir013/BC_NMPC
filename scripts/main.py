# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:40:25 2025

@author: felix
"""



import time
import numpy as np
from model import Monoprop, Monoprop_sim
from mpc import NMPC
from reference import Regulator
from plotting import nmpc_monoprop

def helper(pc):
    c_star = sim.model_func(p_c0, a=5.71725407e-01, b=3.15515423e-01, c=1.53707137e+03)
    mdot = p_c0*sim.Ath/c_star
    rho_p, eta_p, sos_p = sim.fluid_properties(fluid, p_bc, T_prop)
    Re = sim.reynolds(mdot, dp, eta_p)
    zeta_p = sim.calc_zeta(Re, lp, dp, sim.Ap)

    return c_star, rho_p, eta_p, sos_p, zeta_p

####################### fluid and geometric quantities ########################

T_prop = 293            # propellant temperature
fluid = 'C2H6O'         # propellant - chemical formula
p_bc = 30e5             # pressure boundary condition at inlet of pipe
dp = 0.01               # diameter of pipe
lp = 0.5                # length of pipe
Vc = 1e-3               # volume of combustion chamber
Tv = 0.1                # first-order decay time scale of valve dynamics
dth = 0.03              # throat diameter

################################## sim setup ##################################
h_sim = 1e-4 # integration step size for the simulation with full-order dynamics 
sim = Monoprop_sim(h_sim, Tv, dp, lp, Vc, dth, fluid, p_bc, T_prop) # instantiate simulation object

############################## initial condition ##############################

kv_set0 = 0.5           # initial valve position

initial_guess = [0.5, 0.5, 25e5, 20e5, 0.7, 100, 1800]
steady_sol = sim.steady_solution(kv_set0, initial_guess)
steady_sol = steady_sol.x

m_p0, m_v0, p_p0, p_c0, kv0, zeta_p0, c_star0 = steady_sol # states corresponding to initial valve position

y0 = np.array([m_p0, p_p0, kv0, p_c0])

############################# reduced order model #############################
h = 1 # integration step size for the reducer-order model --> can be anything, not needed

mpc_model = Monoprop(h, Tv, dp, lp, Vc, dth, fluid, p_bc, T_prop) # instantiate the reduced-order model used for the MPC
mpc_model()

xdot_func = mpc_model.xdot_func     # dynamics vector containing the ODE as a CasADi function
x = mpc_model.x                     # state vector
u = mpc_model.u                     # control input vector
n_params = mpc_model.n_params       # number of parameters that need to be fed to the ODE


################################## MPC setup ##################################

h_mpc = 0.05                    # MPC sampling time --> inverse is the control frequency
t_pred = 1                      # length of prediction horizon
N_pred = int(t_pred/h_mpc)      # prediction horizon steps

Q = np.eye(1)                   # state weight

R = np.eye(1)*1e-3              # control input weight

R_rate = np.eye(1)              # control input rate weight

mpc = NMPC(N_pred, h_mpc, Q, R, R_rate)  # instantiate MPC object

# actor constraints
actor_constraint = {'ubg': [1],     # control input constraint; Note: the unit for the kv value is [m^3/h]
                    'lbg': [0.1]}

actor_rate = {'ubg': [1.43],         # control input rate constraint [m^3/h/s]
              'lbg': [-1.43]} 

############################# numerical settings ##############################

duration = 1 + t_pred         # simulation duration
N = int(duration/h_mpc)+1
timesteps = np.linspace(0, duration, N)

sim_steps = int(np.ceil(h_mpc/h_sim)) # number of simulation steps per mpc step

####################### solution arrays and control goal ######################
states = np.zeros((4, N*sim_steps))
control_inputs = np.zeros(len(timesteps))

states[:,0] = y0

knot_points = {'t': np.array([0, 0.05, 0.3, 0.5]),      # time steps at which reference changes to a new value
               'pc': np.array([p_c0, p_c0 + 0.5e5, p_c0 - 2e5, p_c0 + 5e5])}    # value to which reference changes once a certain time is exceeded

reference = Regulator(h_mpc)
knot_points = reference.reference(duration, knot_points)
intervals = knot_points['time_interval']

kv_set_prev = kv_set0

#%%############################### main loop ##################################
t_start = time.time()

for t_index, t in enumerate(timesteps):
    if t >= duration - t_pred:
        break
    
    print(f'{t_index +1} of {len(timesteps)}')
    
    p_c0 = y0[3]
    y0_mpc = np.array([p_c0]) 
    
    ygoal = np.array([knot_points['pc_ref'][intervals[t_index], :]])
    
    c_star, rho_p, eta_p, sos_p, zeta_p = helper(y0[3])
    params = [mpc_model.Ath, c_star, p_bc, rho_p, zeta_p, Tv]

    sol, ctrl_input, stats = mpc.solve(xdot_func, x, u, n_params, params, 
                                       u0=[kv_set_prev], x0=np.array([p_c0]), xref=ygoal,
                                       actor_constraint=actor_constraint, 
                                       actor_rate=actor_rate)
    
    kv_set = float(ctrl_input)
    control_inputs[t_index] = kv_set
    kv_set_prev = kv_set

    # execute the simulation, which is considered the ground truth
    sim_sol = sim.integrate(y0, kv_set, sim_steps)

    states[:, 1 + t_index*sim_steps: 1 + (t_index+1)*sim_steps] = sim_sol[:, 1:]
    
    y0 = sim_sol[:, -1]
    
t_end = time.time()

print(f'Process took {t_end-t_start:.2f}s')


#%% plotting

nmpc_monoprop(
    states=states,
    control_inputs=control_inputs,
    timesteps=timesteps,
    sim_steps=sim_steps,
    knot_points=knot_points,
    actor_constraint=actor_constraint,
    actor_rate=actor_rate,
    h_mpc=h_mpc
)


