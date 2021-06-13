#!python

import numpy as np
from scipy.integrate import quad
from scipy import optimize
from scipy.optimize import minimize, root, basinhopping, brute, dual_annealing
import time
import os.path
from numba import complex128,float64,jit

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


@jit((float64, float64, float64, float64, float64, float64, float64), nopython=True, cache=True)
def integrand1(E, q, M, d, mu, T, cutoff):
    return E*np.sqrt((E-q)**2-M**2)*(
                                     fPV(E, d, mu, cutoff) + fPV(E, d, -mu, cutoff) + fPV(E, 0, mu, cutoff)/2 + fPV(E, 0, -mu, cutoff)/2 \
                                     +2*T*np.log(1 + np.exp(-E1(E, d, mu)/T)) + T*np.log(1 + np.exp(-E1(E, 0, mu)/T))\
                                     +2*T*np.log(1 + np.exp(-E2(E, d, mu)/T)) + T*np.log(1 + np.exp(-E2(E, 0, mu)/T)))

@jit((float64, float64, float64, float64, float64, float64, float64), nopython=True, cache=True)
def integrand2(E, q, M, d, mu, T, cutoff):
    return E*np.sqrt((E+q)**2-M**2)*(
                                     fPV(E, d, mu, cutoff) + fPV(E, d, -mu, cutoff) + fPV(E, 0, mu, cutoff)/2 + fPV(E, 0, -mu, cutoff)/2 \
                                     +2*T*np.log(1 + np.exp(-E1(E, d, mu)/T)) + T*np.log(1 + np.exp(-E1(E, 0, mu)/T))\
                                     +2*T*np.log(1 + np.exp(-E2(E, d, mu)/T)) + T*np.log(1 + np.exp(-E2(E, 0, mu)/T)))

@jit((float64, float64, float64, float64, float64, float64, float64), nopython=True, cache=True)
def integrand3(E, q, M, d, mu, T, cutoff):
    return E*(np.sqrt((E+q)**2-M**2)-np.sqrt((E-q)**2-M**2))*(
                                     fPV(E, d, mu, cutoff) + fPV(E, d, -mu, cutoff) + fPV(E, 0, mu, cutoff)/2 + fPV(E, 0, -mu, cutoff)/2 \
                                     +2*T*np.log(1 + np.exp(-E1(E, d, mu)/T)) + T*np.log(1 + np.exp(-E1(E, 0, mu)/T))\
                                     +2*T*np.log(1 + np.exp(-E2(E, d, mu)/T)) + T*np.log(1 + np.exp(-E2(E, 0, mu)/T)))


def func(x, mu, T, cutoff, G, Gd):
    """ 
    It is important to use dimensionless quantities!
    Otherwise the numerical integrator runs into trouble due to too large integrand.
    """
    M, q, d = x[0]/cutoff, x[1]/cutoff, x[2]/cutoff
    # M, q, d = [0 if el < 0 and el > -1 else el for el in [M, q, d]]
    if M < 0: 
        return 1 # cost function to prevent negative M
    # inf should be sufficiently large (~infinity in natural units), but not too large because then
    # precision errors arise
    inf = 20 
    rel = 1e-10 # relative precision
    integral1 = quad(integrand1, q+M, inf, args=(q, M, d, mu, T, cutoff/cutoff), epsabs=0, epsrel=rel, limit=1000)
    integral2 = quad(integrand2, np.abs(q-M), inf, args=(q, M, d, mu, T, cutoff/cutoff), epsabs=0, epsrel=rel, limit=1000)
    if q>M:
        integral3 = quad(integrand3, 0, q-M, args=(q, M, d, mu, T, cutoff/cutoff), epsabs=0, epsrel=rel, limit=1000)
    else: integral3 = (0, 0)

    res = -2/(2*np.pi**2)*(integral1[0]+integral2[0]+integral3[0]) + M**2/(4*G) + d**2/(4*Gd)
    if np.isnan(res): 
        print("ERROR: NAN")
        print("minimum position: ", (M, q, d))
        #print("res: ", res)
        #print("params: ", x)
        return False

    a = 1000
    if  np.abs(integral1[1]) > np.abs(integral1[0]*a*rel) or np.abs(integral2[1]) > np.abs(integral2[0]*a*rel) or np.abs(integral3[1]) > np.abs(integral3[0]*a*rel):
        print("WARNING; INTEGRATION ERROR LARGE")
        print("integral 1: ", (integral1, integral1[1]/(integral1[0]+0.1)))
        print("integral 2: ", (integral2, integral2[1]/(integral2[0]+0.1)))
        print("integral 3: ", (integral3, integral3[1]/(integral3[0]+0.1)))
    return res

def find_min(mu, T, cutoff, G, Gd, max_vals, N_grid_vals):   
    # bruteforce minimization grid parameters as global variables                                                              
    N_M, N_q, N_d = N_grid_vals
    M_min, M_max = 0, max_vals[0]
    q_min, q_max = 0, max_vals[1]
    d_min, d_max = 0, max_vals[2]                                                          
    method = 'Nelder-Mead'                                                                                     
    rranges = (slice(M_min, M_max, (M_max-M_min)/N_M), slice(q_min, q_max, (q_max-q_min)/N_q), slice(d_min, d_max, (d_max-d_min)/N_d))                                            
    brute_res = brute(func, rranges, args=(mu, T, cutoff, G, Gd), finish=optimize.fmin, full_output=True)                                                                                  
    return (brute_res[0], brute_res[1])  

def main(T, mu, cutoff, G, Gd, max_vals, N_grid_vals):
    print("Main routine started...")
    start = time.time()

    # Convert arguments in natural units, except for cutoff
    minimum, pot_val = find_min(mu/cutoff, T/cutoff, cutoff, G*cutoff**2, Gd*cutoff**2, max_vals, N_grid_vals)
    print("(T, mu, [M, q, d], pot val): ", (T, mu, minimum, pot_val))
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
    max_vals = [301, 300, 101] # maximum values for (M, q, d)
    N_grid_vals = [10, 10, 10]

    main(10, 320, cutoff, G, Gd, max_vals, N_grid_vals)