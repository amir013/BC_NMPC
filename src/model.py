# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 11:43:03 2025

@author: felix
"""



import numpy as np
import casadi as ca
from integrator import rk4_e
from scipy.optimize import root
from CoolProp.CoolProp import PropsSI as psi


class Engine:
    ''' methods required for different engine cycles '''
    
    def reynolds(self, mdot, diameter, eta):
        ''' Reynolds Number '''
        return 4*mdot/diameter/eta/np.pi
    
    def calc_zeta(self, Re, l, d, A):
        ''' Friction Factor Calculation depending on the 
            Reynolds Number given PhD Thesis Chiara Manfletti '''
        
        def friction_factor(Re, d, k=0):
            if Re <= 2320:
                lam = 64/Re
            elif Re > 2320 and Re <= 1e5:
                lam = 0.316/(Re**(0.25))
            elif Re > 1e5:
                lam_up = 10
                lam_low = 0
                err = 100
                while abs(err) > 1e-4:
                    lam_mid = np.mean([lam_low, lam_up])
                    err = 1/np.sqrt(lam_mid)+2*np.log10(2.51/Re/np.sqrt(lam_mid) + k/3.71/d)
                    if err>0:
                        lam_low = lam_mid
                    elif err<0:
                        lam_up = lam_mid
                    elif err == 0:
                        break
                lam = lam_mid
            return lam
        
        lam = friction_factor(Re, d)
        zeta = lam*l/d/2/A**2
        
        return zeta
    
    def fluid_properties(self, fluid, p, T):
        rho = psi('D','P',p,'T',T,fluid)
        eta = psi('viscosity','P',p,'T',T,fluid)
        sos = psi('speed_of_sound','P',p,'T',T,fluid)
        return rho, eta, sos
    
    def valve_resistance(self, kv, rho):
        return 3600**2*1e5/rho**2/kv**2
    
    
    
class Monoprop_sim(Engine):
    ''' implementation of the DAE system for the monoprop engine dynamics 
        this is the full-order model used for simulation '''
    
    def __init__(self, h, Tv, dp, lp, Vc, dth, fluid, p_bc, T_prop):#, param_file):
        self.h = h                  # integration step size [s]
        self.Tv = Tv                # valve time scale [s]
        
        # Geometrical quantities
        self.dp = dp
        self.Ap = dp**2/4*np.pi
        self.lp = lp
        self.Vp = self.Ap * lp
        self.Vc = Vc
        self.Ath = dth**2/4*np.pi
        
        # Fluid quantities
        self.fluid = fluid
        self.p_bc = p_bc
        self.T_prop = T_prop
        
        # with open(param_file, 'r') as f:
        #     self.params = yaml.safe_load(f)
        
    def model_func(self, x, a, b, c):
        ''' parametric polynomic function used to compute combustion properties '''
        return c + a * x**b
    
    def dynamics(self, t, y, params):
        ''' definition of four odes describing the engine dynamics
            Flow scheme: Pipe - Valve - Chamber '''
        # solution of previous time step
        # print(y)
        m_p, p_p, kv, p_c = y
        
        # control input
        kv_set = params
        
        # algebraic relations
        rho_p, eta_p, sos_p = self.fluid_properties(self.fluid, p_p, self.T_prop)
        Re = self.reynolds(m_p, self.dp, eta_p)
        zeta_p = self.calc_zeta(Re, self.lp, self.dp, self.Ap)
        Rv = self.valve_resistance(kv, rho_p)
        
        m_v = np.sqrt((p_p - p_c)/Rv)
        
        Rcc = 453
        Tcc = self.model_func(p_c, a=2.97208162, b=2.86578226e-01, c=2.22267450e+03)
        c_star = self.model_func(p_c, a=5.71725407e-01, b=3.15515423e-01, c=1.53707137e+03)
        
        m_c = p_c*self.Ath/c_star
        
        dm_p = self.Ap/self.lp*(self.p_bc-p_p-zeta_p/rho_p*m_p*abs(m_p))
        dp_p = sos_p**2/self.Vp*(m_p-m_v)
        dkv = (kv_set-kv)/self.Tv
        dp_c = Rcc*Tcc/self.Vc*(m_v-m_c)
        
        return np.array([dm_p, dp_p, dkv, dp_c])
    
    def integrate(self, x0, u, N_steps):
        ''' numerical integration of N_steps '''
        
        sol = np.zeros((len(x0), N_steps+1))
        sol[:, 0] = x0
        
        params = u
        
        for step in range(N_steps):
            
            t = self.h*step
            sol[:, step+1] = rk4_e(self.dynamics, sol[:, step], self.h, t, params)
        
        return sol

    def steady_solution(self, params, initial_guess=None):
        ''' obtain the steady solution of the system given a specified valve
            kv value '''
        rho_p, eta_p, _ = self.fluid_properties(self.fluid, self.p_bc, self.T_prop)

        self.rho_p = rho_p        

        kv_set = params

        a=5.71725407e-01
        b=3.15515423e-01
        c=1.53707137e+03
        
        # Gleichungssystem definieren
        def equations(vars):
            m_p, m_v, p_p, p_c, kv, zeta_p, c_star = vars
            
            eq_m = (self.p_bc - p_p - zeta_p / rho_p * m_p * abs(m_p))
            eq_p_p = (m_p - m_v)
            eq_kv = (kv_set - kv)
            eq_p_c = (m_v - self.Ath*p_c/c_star)
            eq_m_v = m_v - np.sqrt((p_p-p_c)/(3600**2*1e5/rho_p**2/kv**2))
            eq_zeta_p = zeta_p - self.lp/self.dp/2/self.Ap**2*0.316/((4*abs(m_p)/eta_p/np.pi/self.dp)**(0.25))
            eq_c_star = c_star - (c + a * p_c**b)
            
            return [eq_m, eq_p_p, eq_kv, eq_p_c, eq_m_v, eq_zeta_p, eq_c_star]
        
        if initial_guess is None:
            initial_guess = np.zeros(7)

        solution = root(equations, initial_guess)

        return solution
    

class Monoprop(Engine):
    ''' reduced order model where, p_c_dot = f(p, kv_set)
        this is the model used for the MPC
        reduced-order model is derived based on CasADi's AutoDiff '''
    
    def __init__(self, h, Tv, dp, lp, Vc, dth, fluid, p_bc, T_prop):#, param_file):
        self.h = h                  # integration step size [s]
        self.Tv = Tv                # valve time scale [s]
        
        # Geometrical quantities
        self.dp = dp
        self.Ap = dp**2/4*np.pi
        self.lp = lp
        self.Vp = self.Ap * lp
        self.Vc = Vc
        self.Ath = dth**2/4*np.pi
        
        # Fluid quantities
        self.fluid = fluid
        self.p_bc = p_bc
        self.T_prop = T_prop
        
        # with open(param_file, 'r') as f:
        #     self.params = yaml.safe_load(f)
        
    def __call__(self):
        ''' performs the symbolic differentiation and defines the reduced-order
            model ODE '''
                
        kv_set = ca.SX.sym('kv_set')
        pc = ca.SX.sym('pc')

        A_th = ca.SX.sym('A_th')
        c_star = ca.SX.sym('c_star')
        p_bc = ca.SX.sym('p_bc')
        rho = ca.SX.sym('rho')
        zeta = ca.SX.sym('zeta')
        T = ca.SX.sym('T')
        
        kv = ca.SX.sym('kv')

        C = A_th/c_star
        K = zeta/rho
        R = 3600**2*1e5/rho**2

        kv_eq = ca.sqrt((-R*pc**2*C**2)/(K*pc**2*C**2 - (p_bc - pc))) 

        g_eq_sym = pc*C - ca.sqrt((p_bc - pc) / (K + R/kv**2))

        dg_dkv = ca.jacobian(g_eq_sym, kv)
        dg_dpc = ca.jacobian(g_eq_sym, pc)

        dg_dkv_sub = ca.substitute(dg_dkv, kv, kv_eq)
        dg_dpc_sub = ca.substitute(dg_dpc, kv, kv_eq)

        pc_dot = - dg_dkv_sub/dg_dpc_sub * (kv_set - kv_eq) / T
        
        # nonlinear ODE
        pc_dot_func = ca.Function('pc_dot_func', [A_th, c_star, p_bc, rho, zeta, T, pc, kv_set], [pc_dot])

        self.pc_dot_func = pc_dot_func

        # linearised ode
        F = ca.vertcat(pc_dot)
        X = ca.vertcat(pc)
        U = ca.vertcat(kv_set)
        
        params = ca.vertcat(A_th, c_star, p_bc, rho, zeta, T)
        
        A = ca.jacobian(F, X)
        A_func = ca.Function('A_func', [X, U, params], [A])
        B = ca.jacobian(F, U)
        B_func = ca.Function('B_func', [X, U, params], [B])  

        self.Afunc = A_func
        self.Bfunc = B_func
        
        # for NMPC
        self.x = ca.vertcat(pc)
        self.u = ca.vertcat(kv_set)
        # xdot = F
        params = ca.vertcat(A_th, c_star, p_bc, rho, zeta, T)
        self.n_params = params.shape[0]
        
        self.xdot_func = ca.Function(
            'xdot_func',
            [self.x, self.u, params],
            [ca.vertcat(pc_dot)])
    
    def model_func(self, x, a, b, c):
        return c + a * x**b