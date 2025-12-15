from .constants import *
import numpy as np
from .harmonics import *
from emri.LISA_sensitivity import *


def de_dt_PN(phi, nu, gamma, e, p):
    '''
    
    Equation 30 of Barack & Cutler (2003)
    
    '''
    M = p.M_seconds
    mu = p.mu_seconds
    Mtot = M + mu
    X = p.X
    lambda_var = p.lambda_var
    
    one_minus_e2 = max(1e-16, 1 - e**2)
    x = 2*np.pi*Mtot*nu
    x23, x83, x113 = x**(2/3), x**(8/3), x**(11/3)

    term1 = (304 + 121*e**2)*(1 - e**2)*(1 + 12*x23)
    term2 = (1/56)*x23*(8*16705 + 12*9082*e**2 - 25211*e**4)
    first_bracket = term1 - term2

    first_part = -(e/15.0)*(mu/Mtot**2)*one_minus_e2**(-3.5)*x83*first_bracket
    spin_part  =  e*(mu/Mtot**2)*X*np.cos(lambda_var)*x113*one_minus_e2**(-4)*((1364/5) + (5032/15)*e**2 + (263/10)*e**4)
    return first_part + spin_part


def dphi_dt_PN(phi, nu, gamma, e, p):
    '''
    
    Equation 27 mean anomaly 

    '''
    return 2*np.pi*nu


def dgamma_dt_PN(phi, nu, gamma, e, p):
    '''
    
    Equation 29 pericenter precession

    '''
    M = p.M_seconds
    X = p.X
    lambda_var = p.lambda_var
    
    zero_term = max(1e-16, 1 - e**2)
    first_term = 6*np.pi*nu*(2*np.pi*nu*M)**(2/3)*zero_term**(-1)
    second_term = 1 + 0.25*(2*np.pi*nu*M)**(2/3)*zero_term**(-1)*(26 - 15*e**2)
    third_term = 12*np.pi*nu*X*np.cos(lambda_var)*(2*np.pi*M*nu)*zero_term**(-1.5)
    return first_term*second_term - third_term


def dnu_dt_PN(phi, nu, gamma, e, p):
    '''
    
    Equation 28 orbital frequency evolution 

    '''
    M = p.M_seconds
    mu = p.mu_seconds
    X = p.X
    lambda_var = p.lambda_var
    
    zero_term = max(1e-16, 1 - e**2)
    omM = 2*np.pi*M*nu
    first_term = (96/(10*np.pi))*(mu/(M**3))*(omM**(11/3))*zero_term**(-4.5)
    a1 = 1 + (73/24)*e**2 + (37/96)*e**4
    b1 = (1273/336) - (2561/224)*e**2 - (3885/128)*e**4 - (13147/5376)*e**6
    c1 = (73/12) + (1211/24)*e**2 + (3143/96)*e**4 + (65/64)*e**6
    return first_term*(a1*(1 - e**2) + omM**(2/3)*b1 - omM*X*np.cos(lambda_var)*zero_term**(-0.5)*c1)


def dalpha_dt_PN(phi, nu, gamma, e, alpha, p):
    '''
    
    Equation 31 nodal precession 
    '''
    M = p.M_seconds
    X = p.X
    
    zero_term = max(1e-16, 1 - e**2)
    return 4*np.pi*nu*X*(2*np.pi*M*nu)*zero_term**(-1.5)


def f_orb_PN(n, phi, nu, gamma, e, p):
    
    return n*(nu + (2/n)*(dgamma_dt_scalar(phi, nu, gamma, e, p)/(2*np.pi)))


def dgamma_dt_scalar(phi, nu, gamma, e, p):
    '''
    
    Equation 29 pericenter precession

    '''
    M = p.M_seconds
    lambda_var = p.lambda_var
    X = p.X
    
    zero_term = np.maximum(1e-16, 1 - e**2)
    first_term = 6*np.pi*nu*(2*np.pi*nu*M)**(2/3)*zero_term**(-1)
    second_term = 1 + 0.25*(2*np.pi*nu*M)**(2/3)*zero_term**(-1)*(26 - 15*e**2)
    third_term = 12*np.pi*nu*X*np.cos(lambda_var)*(2*np.pi*M*nu)*zero_term**(-1.5)
    return first_term*second_term - third_term

def dnu_dt_scalar(phi, nu, gamma, e, p):
    '''
    
    Equation 28 orbital frequency evolution 

    '''
    M = p.M_seconds
    mu = p.mu_seconds
    lambda_var = p.lambda_var
    X = p.X
    
    zero_term = np.maximum(1e-12, 1 - e**2)
    omM = 2*np.pi*M*nu
    first_term = (96/(10*np.pi))*(mu/(M**3))*(omM**(11/3))*zero_term**(-4.5)
    a1 = 1 + (73/24)*e**2 + (37/96)*e**4
    b1 = (1273/336) - (2561/224)*e**2 - (3885/128)*e**4 - (13147/5376)*e**6
    c1 = (73/12) + (1211/24)*e**2 + (3143/96)*e**4 + (65/64)*e**6
    return first_term*(a1*(1 - e**2) + omM**(2/3)*b1 - omM*X*np.cos(lambda_var)*zero_term**(-0.5)*c1)

def deriv_PN_reduced(t, y, p):
    e, nu = y

    # compute dnu/dt and de/dt using the same formulas as the full PN system
    dnu = dnu_dt_PN(phi=0.0, nu=nu, gamma=0.0, e=e, p=p)     # or refactor dnu_dt to not need phi/gamma
    de  = de_dt_PN(phi=0.0, nu=nu, gamma=0.0, e=e, p=p)

    return [de, dnu]

def deriv_PN(t,y, p):
    
    phi, nu, gamma, e, alpha = y
    return [
        dphi_dt_PN(phi, nu, gamma, e, p),
        dnu_dt_PN(phi, nu, gamma, e, p),
        dgamma_dt_PN(phi, nu, gamma, e, p),
        de_dt_PN(phi, nu, gamma, e, p),
        dalpha_dt_PN(phi, nu, gamma, e, alpha, p),
    ]

def f_dot_PN(n, e, nu, phi, gamma, p):
    return n*dnu_dt_scalar(phi, nu, gamma, e, p)
    #return np.gradient(f_arr, time_arr)   # or n * np.gradient(nu_arr, time_arr)

def E_dot_n_PN(n, nu, e, p):
    
    return (32/5.0) * p.mu_seconds**2 * p.M_seconds**(4/3) * (2*np.pi*nu)**(10/3) * g_n_e(n, e)

def h_c_n_PN(n, e, nu, phi, gamma, p):
    return (1/(np.pi*(p.D/c))) * np.sqrt(2 * E_dot_n_PN(n, nu, e, p) / f_dot_PN(n, e, nu, phi, gamma, p))

def h_n_PN(n, e, nu, phi, gamma, p):
    # h0 with a eliminated using Kepler's law
    h0 = np.sqrt(32/5)  * (p.M_seconds**(2/3)) * p.mu_seconds / (p.D/c) \
         * (2*np.pi*nu)**(2/3)
    
    return (2/n) * np.sqrt(g_n_e(n, e)) * h0

def nu_LSO(e, p):
    M = p.M_seconds
    
    p_lso = 6.0 + 2.0 * e
    return (1.0 / (2.0 * np.pi * M)) * ((1.0 - e**2) / p_lso) ** 1.5
