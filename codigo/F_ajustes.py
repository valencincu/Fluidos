#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:29:28 2023

@author: plopezmaggi
"""
#%%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import scipy.special as sp
import os

def chi2_reducido(x, y, func, params, *, y_err):
    """Calcula el chi^2 reducido a partir de los datos (x, y),
    la función ajustada (`func`) y los parámetros óptimos (`params`).
    """
    n_datos = y.size
    n_params = len(params)
    y_pred = func(x, *params)
    residuos = y - y_pred
    return np.sum((residuos / y_err) ** 2) / (n_datos - n_params)


#%% Modelos vortice estacionario
def burgers(r, Gamma, a):
    C = Gamma / (2*np.pi)
    return C *  (1 - np.exp(-r**2 / a**2)) / r

def rankine(r, Gamma, a):
    C = Gamma / (2*np.pi)
    return np.piecewise(r, [r < a, r >= a], [lambda x : C * x / (a**2), lambda x : C / x])

def lineal(x, m, b):
    return m*x + b


#%% Modelo disipación (ver https://link.springer.com/article/10.1007/s10409-005-0073-3)

def disipacion_potencial(r, t, omega, c, nu):
    r = r / c
    t = nu*t / c**2
    return omega * ( 1 - np.exp(-r**2 / (4*t)) ) / r

def A_rankine(n, beta):
    return -4*sp.jv(1, lamda(n)*beta) / ( lamda(n) * sp.jv(0, lamda(n)) )**2

def lamda(n):
    res = np.pi * ( n + 0.25 - 0.151982 / (4*n + 1) + 0.0151399 / ((4*n + 1)**3) - 0.245275 / ((4*n + 1)**5) )
    return res

def solucion_oseen(A, beta = 0.01, tol = 1000):
    def solucion(r, t):
        sol = 0
        for j in range(1, tol+1):
            coef = lamda(j)*beta
            sol += A(j, beta)*sp.jv(1, (coef*r))*np.exp(-(coef**2)*t)
        return beta**2 * r - sol
    return solucion

def disipacion_rankine(r, t, Gamma, c, nu, beta):
    C = Gamma / (2*np.pi)
    disipacion = solucion_oseen(A_rankine, beta = beta)
    return C * disipacion(r / c, nu*t / c**2)
    
def disipacion_contornos(r, t, omega, c, nu, r_ext):
    beta = c / r_ext
    disipacion = solucion_oseen(A_rankine, beta) 
    return omega * c**2 * ( disipacion(r / c, nu*t / c**2) - beta**2 * r ) 


#%% Colormap
def cmap(u, v, colormap):
    color = np.hypot(u, v)
    color = (color - min(color)) / (max(color) - min(color))
    return cm[colormap](color)
