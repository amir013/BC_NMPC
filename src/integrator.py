# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 09:10:22 2025

@author: felix
"""



def rk4_e(f, y, h, t, *args):
    # runge kutte 4th order explicit
    tk_05 = t + 0.5*h
    yk_025 = y + 0.5 * h * f(t, y, *args)
    yk_05 = y + 0.5 * h * f(tk_05, yk_025, *args)
    yk_075 = y + h * f(tk_05, yk_05, *args)
    
    return y + h/6 * (f(t, y, *args) + 2 * f(tk_05, yk_025, *args) + 2 * f(tk_05, yk_05, *args) + f(t+h, yk_075, *args))

def rk4_nlp(f, x, u, p, h):
    k1 = f(x, u, p)
    k2 = f(x + h/2 * k1, u, p)
    k3 = f(x + h/2 * k2, u, p)
    k4 = f(x + h * k3, u, p)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)