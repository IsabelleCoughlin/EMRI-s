from scipy.interpolate import interp1d
import numpy as np
from emri.constants import *

def interpolate_PM(a_arr, e_arr, t_arr):

    f1 = interp1d(t_arr, e_arr)
    f2 = interp1d(t_arr, a_arr)

    t_interpolated = np.linspace(min(t_arr), max(t_arr), num=round(np.max(t_arr/yr)))
    e_interpolated = f1(t_interpolated)
    a_interpolated = f2(t_interpolated)

    t_to_plunge_yr_interpolated = (t_interpolated[-1] - t_interpolated) / yr
    return a_interpolated, e_interpolated, t_interpolated, t_to_plunge_yr_interpolated

def interpolate_PN(e_arr, nu_arr, phi_arr, gamma_arr, alpha_arr, t_arr, npts=1000):
    
    if t_arr[0] > t_arr[-1]:
        t_arr = t_arr[::-1]
        e_arr = e_arr[::-1]
        nu_arr = nu_arr[::-1]
        phi_arr = phi_arr[::-1]
        gamma_arr = gamma_arr[::-1]
        alpha_arr = alpha_arr[::-1]

    f1 = interp1d(t_arr, e_arr)
    f2 = interp1d(t_arr, nu_arr)
    f3 = interp1d(t_arr, phi_arr)
    f4 = interp1d(t_arr, gamma_arr)
    f5 = interp1d(t_arr, alpha_arr)
    
    num  = round(np.max(t_arr/yr))
    if num == 0:
        num = npts

    t_interpolated = np.linspace(min(t_arr), max(t_arr), num=num)
    e_interpolated = f1(t_interpolated)
    nu_interpolated = f2(t_interpolated)
    phi_interpolated = f3(t_interpolated)
    gamma_interpolated = f4(t_interpolated)
    alpha_interpolated = f5(t_interpolated)

    t_to_plunge_yr_interpolated = (t_interpolated[-1] - t_interpolated) / yr
    
    return e_interpolated, nu_interpolated, phi_interpolated, gamma_interpolated, alpha_interpolated, t_interpolated, t_to_plunge_yr_interpolated