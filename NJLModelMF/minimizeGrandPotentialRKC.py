#!python

import numpy as np
from scipy import optimize
from scipy.integrate import quad
from scipy.special import ellipkinc as fi, ellipeinc as ei, ellipk as k, ellipe as e, ellipj as jef
import time
from numba import complex128,float64,jit
from scipy.optimize import brute
from minimizeGrandPotentialCDW import func as funcCDW

from matplotlib.ticker import MaxNLocator

@jit((float64, float64, float64, float64), nopython=True, cache=True)
def fPV(E, d, mu, cutoff):
    return np.sqrt((E + mu)**2 + d**2) - 3*np.sqrt((np.sqrt(E**2 + cutoff**2) + mu)**2 + d**2) + 3*np.sqrt((np.sqrt(E**2 + 2*cutoff**2) + mu)**2 + d**2) \
        - np.sqrt((np.sqrt(E**2 + 3*cutoff**2) + mu)**2 + d**2)

@jit((float64, float64, float64), nopython=True, cache=True)
def E1(E, d, mu):
    return np.sqrt((E+mu)**2 + d**2)


@jit((float64, float64, float64), nopython=True, cache=True)
def E2(E, d, mu):
    return np.sqrt((E-mu)**2 + d**2)


@jit(nopython=True, cache=True)
def nut(nu):
    return 1 - nu

@jit((float64, float64, float64), nopython=True, cache=True)
def tht(E, nu, D):
    if nu == 1 or D == 0:
        return np.pi/2
    return np.arcsin(E/(np.sqrt(nut(nu))*D))

@jit((float64, float64), nopython=True, cache=True)
def th(E, D): return np.arcsin(D/E)


@jit((float64, float64, float64, float64, float64, float64, float64), nopython=True, cache=True)
def int_fact(E, nu, D, d, mu, T, cutoff):
    return E*(fPV(E, d, mu, cutoff) + fPV(E, d, -mu, cutoff) + fPV(E, 0, mu, cutoff)/2 + fPV(E, 0, -mu, cutoff)/2 \
              +2*T*np.log(1 + np.exp(-E1(E, d, mu)/T)) + T*np.log(1 + np.exp(-E1(E, 0, mu)/T))\
              +2*T*np.log(1 + np.exp(-E2(E, d, mu)/T)) + T*np.log(1 + np.exp(-E2(E, 0, mu)/T)))

#@jit((float64, float64, float64, float64, float64, float64, float64), forceobj=True, cache=True)
def integrand1(E, nu, D, d, mu, T, cutoff):
    return (ei(tht(E, nu, D), nut(nu))+(e(nu)/k(nu) - 1)*fi(tht(E, nu, D), nut(nu)))*int_fact(E, nu, D, d, mu, T, cutoff)

#@jit((float64, float64, float64, float64, float64, float64, float64), forceobj=True, cache=True)
def integrand2(E, nu, D, d, mu, T, cutoff):
    return (e(nut(nu))+(e(nu)/k(nu) - 1)*k(nut(nu)))*int_fact(E, nu, D, d, mu, T, cutoff)

#@jit((float64, float64, float64, float64, float64, float64, float64), forceobj=True, cache=True)
def integrand3(E, nu, D, d, mu, T, cutoff):
    return (ei(th(E, D), nut(nu))+(e(nu)/k(nu) - 1)*fi(th(E, D), nut(nu)) + np.sqrt((E**2 - D**2)*(E**2 - nut(nu)*D**2))/(D*E))*int_fact(E, nu, D, d, mu, T, cutoff)

#@jit((float64, float64, float64), forceobj=True, cache=True)
def Mz(z, D, nu):
    sn, cn, dn = jef(D*z, nu)[:3]
    return (D*nu*sn*cn/dn)**2

def func(x, mu, T, cutoff, G, Gd):
    D, nu, d = x[0]/cutoff, x[1], x[2]/cutoff

    if D <= 0 or nu < 0 or nu > 1:
        return 1

    CDW_res = 1 # initialize variable
    if nu > 0.9999 or D == 0:
        #print("call CDW function")
        CDW_res = funcCDW([x[0], 0, x[2]], mu, T, cutoff, G, Gd)

    if nu == 0:
        #print("call CDW function")
        return funcCDW([0, 0, x[2]], mu, T, cutoff, G, Gd)

    inf = 20 #*cutoff#np.inf
    rel = 1e-10

    integral1 = quad(integrand1, 0, np.sqrt(nut(nu))*D, args=(nu, D, d, mu, T, cutoff/cutoff), epsabs= 0, epsrel=rel, limit=1000)
    integral2 = quad(integrand2, np.sqrt(nut(nu))*D, D, args=(nu, D, d, mu, T, cutoff/cutoff), epsabs= 0, epsrel=rel, limit=1000)
    integral3 = quad(integrand3, D, inf, args=(nu, D, d, mu, T, cutoff/cutoff), epsabs= 0, epsrel=rel, limit=1000)

    L = 4*k(nu)/D

    Mint = quad(Mz, 0, L, args=(D, nu), epsabs= 0, epsrel=rel, limit=500)

    result = -2*D/(np.pi**2)*(integral1[0]+integral2[0]+integral3[0]) + 1/(4*G*L)*Mint[0] + d**2/(4*Gd)
    if result > CDW_res: return CDW_res
    return result

def find_min(mu, T, cutoff, G, Gd, max_vals, N_grid_vals):   
    # bruteforce minimization grid parameters as global variables                                                              
    N_M, N_q, N_d = N_grid_vals
    M_min, M_max = 0, max_vals[0]
    q_min, q_max = 0, max_vals[1]
    d_min, d_max = 0, max_vals[2]                                                                                                                                             
    rranges = (slice(M_min, M_max, (M_max-M_min)/N_M), slice(q_min, q_max, (q_max-q_min)/N_q), slice(d_min, d_max, (d_max-d_min)/N_d))                                            
    brute_res = brute(func, rranges, args=(mu, T, cutoff, G, Gd), finish=optimize.fmin, full_output=True)                                                                                  
    return (brute_res[0], brute_res[1])  

def main(T, mu, cutoff, G, Gd, max_vals, N_grid_vals):
    print("Main routine started...")
    start = time.time()

    # Convert arguments in natural units, except for cutoff
    minimum, pot_val = find_min(mu/cutoff, T/cutoff, cutoff, G*cutoff**2, Gd*cutoff**2, max_vals, N_grid_vals)
    print("(T, mu, [M, nu, d], pot val): ", (T, mu, minimum, pot_val))
    print("TIME: ", time.time()-start)
    return (minimum, pot_val)

if __name__ == "__main__":
    # Parameter set for Mq = 330 MeV
    # cutoff = 728.368
    # G = 6.599/k_cutoff**2

    # Parameter set for Mq = 300 MeV:
    cutoff = 757.048
    G = 6.002/cutoff**2
    fact = 0.001 # For zero diquark coupling choose a very small value, but not zero
    Gd = fact*G 
    max_vals = [301, 1, 100] # maximum values for (M, nu, d)
    N_grid_vals = [10, 10, 10]
    T, mu = 1, 305

    main(T, mu, cutoff, G, Gd, max_vals, N_grid_vals)