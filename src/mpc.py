# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:32:22 2025

@author: felix
"""



import casadi as ca
from integrator import rk4_nlp

class MPC():
    def __init__(self, Nx, h, Q, R, R_rate=None):
        self.Nx = Nx
        self.h = h
        self.Q = Q
        self.R = R
        if R_rate is not None:
            self.R_rate = R_rate
    
class NMPC(MPC):
    ### implements the NMPC with control input constraints (absolute and rate)
    def __init__(self, Nx, h, Q, R, R_rate=None):
        
        super().__init__(Nx, h, Q, R, R_rate)
        
    def solve(self, xdot_func, x, u, n_params, params, u0, x0, xref, 
              actor_constraint={}, actor_rate={}, prt_lvl="none"):
        dim_x = x.shape[0]
        dim_u = u.shape[0]
        
        p_sym = ca.SX.sym('p', n_params)
        # passing the dynamics to CasADi as a CasADi function
        F = ca.Function(
        'F',
        [x, u, p_sym],
        [rk4_nlp(xdot_func, x, u, p_sym, self.h)]
        )
        
        # decision variables
        X = ca.SX.sym('X', dim_x, self.Nx + 1)
        U = ca.SX.sym('U', dim_u, self.Nx)

        # Stack all decision variables into one vector
        w = ca.vertcat(
            ca.vec(X),
            ca.vec(U)
        )

        # objective / const function
        cost = 0
        for k in range(self.Nx):
            cost += ca.mtimes(
                [(X[:, k] - xref[:, k]).T, self.Q, (X[:, k] - xref[:, k])]
            )
            cost += ca.mtimes([U[:, k].T, self.R, U[:, k]])
        
        cost += ca.mtimes([(U[:, 0] - u0).T, self.R_rate, (U[:, 0] - u0)])
        for k in range(self.Nx-1):
            cost += ca.mtimes([(U[:, k+1].T - U[:, k].T), self.R_rate, (U[:, k+1] - U[:, k])])


        ## constraints
        g = []
        lbg = []
        ubg = []
        
        # constraints - initial condition
        g.append(X[:, 0] - x0)
        lbg.append(ca.DM.zeros(dim_x))
        ubg.append(ca.DM.zeros(dim_x))

        # constraints - dynamics
        for k in range(self.Nx):
            dyn_constraint = X[:, k+1] - F(X[:,k], U[:,k], p_sym)#(X[:, k] + F(X[:,k], U[:,k], p_sym))
            g.append(dyn_constraint)
            lbg.append(ca.DM.zeros(dim_x))
            ubg.append(ca.DM.zeros(dim_x))
            
            # constraints - control input absolute values
            if actor_constraint:
                g.append(U[:, k])
                lbg.append(actor_constraint['lbg'])
                ubg.append(actor_constraint['ubg'])

        # constraints - control input rate
        if actor_rate:
            g.append(U[:, 0])
            lbg.append([u0[ii] + actor_rate['lbg'][ii]*self.h for ii in range(len(actor_rate['lbg']))])
            ubg.append([u0[ii] + actor_rate['ubg'][ii]*self.h for ii in range(len(actor_rate['ubg']))])
            for k in range(self.Nx - 1):
                g.append(U[:, k+1] - U[:, k])
                lbg.append([rate * self.h for rate in actor_rate['lbg']])
                ubg.append([rate * self.h for rate in actor_rate['ubg']])


        g = ca.vertcat(*g)
        lbg = ca.vertcat(*lbg)
        ubg = ca.vertcat(*ubg)

        # NLP definition
        nlp = {
            'x': w,
            'f': cost,
            'g': g,
            'p': p_sym
        }

        
        # solver options
        opts = {
            'ipopt.print_level': 0 if prt_lvl == "none" else 5,
            'print_time': False,
            'error_on_fail': False,
        }
        
        # set up the solver
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # initialize the solver
        w0 = ca.DM.zeros(w.shape[0])
        
        # initialize the solver - states
        for k in range(self.Nx + 1):
            w0[k*dim_x:(k+1)*dim_x] = x0 # latest "measured" state
        
        # initialize the solver - controls
        offset_u = dim_x*(self.Nx+1)
        for k in range(self.Nx):
            w0[offset_u + k*dim_u : offset_u + (k+1)*dim_u] = u0 # previous control input
      
        # solve
        sol = solver(
            x0=w0,
            lbg=lbg,
            ubg=ubg, 
            p = params
        )
        
        # extract control input        
        offset_u = dim_x * (self.Nx + 1)
        ctrl_input = sol['x'][offset_u:offset_u + dim_u]

        stats = solver.stats()

        return sol, ctrl_input, stats