import numpy as np
from .constants import M, mu

def dadt_PM(a, e):
    """
    Rate of change of semi-major axis [Equation 5.6]
    (Peters 1964)
    """
    return -(64/5) * (G**3 * M* m * (M + m)) / (c**5 * a**3 * (1 - e**2)**(3.5)) * (1 + (73/24)*e**2 + (37/96)*e**4)

def dedt_PM(a, e):
    """
    Rate of change of eccentricity [Equation 5.7]
    (Peters 1964)
    """
    return -(304/15) * (e * G**3 * M * m * (M+m)) / (c**5 * a**4 *(1-e**2)**(2.5)) * (1 + (121/304)*e**2)

def dfdt_n_PM(n, a, e):
    '''
    Direct derivative of d_orb(a)
    '''
    f = f_orb(a)
    df_da = -(3/2) * f / a
    return n * df_da * dadt(a,e)

def f_orb_PM(a):
    '''
    Keplerian orbital frequency
    Returns in seconds^-1
    '''
    return (1.0/(2.0*np.pi))*np.sqrt(G*(M+m)/a**3)

def deriv_PM(t,y):
    a, e = y
    da = dadt(a, e)
    de = dedt(a, e)
    return [da, de]

def h_n_PM(n, e, a):

    '''
    Definition for characteristic strain coming from 
    '''
    h0 = np.sqrt(32/5) * (G**2 * M * m) / (c**4 * D * a)
    return (2/n) * np.sqrt(g_n_e(n, e)) * h0
