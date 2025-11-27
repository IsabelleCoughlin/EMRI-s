import numpy as np
from .peters import *
from .LISA_sensitivity import *
from emri.PN import *

def snr2_increment(a, e, dt,p, N_harm_sum):
    M = p.M_si
    mu = p.mu_si
    SNR2 = 0.0
    for n in range(1, N_harm_sum+1):
        f_n  = n * f_orb_PM(a, M, mu)            # Hz
        h_nv = h_n_PM(n, e, a, M, mu)            # strain
        S_h  = get_sensitivity(f_n)    # PSD, 1/Hz
        S_SA = 5.0 * S_h             
        # sky-averaged PSD
        SNR2 += 2 * h_nv**2 * f_n / S_SA * dt
    return SNR2

'''

Function to compute total SNR vs time (aka just the black line)

'''

def snr_total_1yr_vs_time(a_arr, e_arr, t_arr, p, N_harm_sum):
    dt_arr = np.diff(t_arr, prepend=t_arr[0])   # seconds
    T = len(t_arr)
    SNR2_total = np.zeros(T)

    for i in range(T):
        a  = a_arr[i]
        e  = e_arr[i]
        dt = dt_arr[i]
        SNR2_total[i] = snr2_increment(a, e, dt, p, N_harm_sum)

    SNR_total = np.sqrt(np.cumsum(SNR2_total))

    t_to_plunge_yr = (t_arr[-1] - t_arr) / yr
    return t_to_plunge_yr, SNR_total


def snr_1yr_vs_time(a_arr, e_arr, t_arr,p, N_harm_show=10, N_harm_sum=1000):
    '''
    Plotting the first 10 harmonics in a variety of colors along with the total SNR from first N_harm_sum harmonics. Paper generally uses 1000
    harmonics and thus that is the default.

    '''
    
    M = p.M_si
    mu = p.mu_si

    dt_arr = np.diff(t_arr, prepend=t_arr[0])
    T = len(t_arr)

    # total SNR^2 across all harmonics
    SNR2_total = np.zeros(T)

    # per-harmonic SNR^2 (only for first N_harm_show)
    SNR2_harm = np.zeros((N_harm_show, T))

    for i in range(T):
        a = a_arr[i]
        e = e_arr[i]
        dt = dt_arr[i]

        # loop through harmonics
        for n in range(1, N_harm_sum+1):

            f_n  = n * f_orb_PM(a, M, mu)
            h_nv = h_n_PM(n, a, e, M, mu)
            S_h  = get_sensitivity(f_n)
            S_SA = 5.0 * S_h

            dSNR2 = 2 * h_nv**2 * f_n / S_SA * dt

            # add to total
            SNR2_total[i] += dSNR2

            # store FIRST N_harm_show only
            if n <= N_harm_show:
                SNR2_harm[n-1, i] += dSNR2

    # convert to cumulative SNR
    SNR_total = np.sqrt(np.cumsum(SNR2_total))

    # cumulative per-harmonic SNR
    SNR_harm = np.sqrt(np.cumsum(SNR2_harm, axis=1))

    # time to plunge in years
    t_to_plunge = (t_arr[-1] - t_arr) / yr

    return t_to_plunge, SNR_total, SNR_harm



def snr2_increment_PN(phi, nu, gamma, e, dt, p, N_harm_sum):
    SNR2 = 0.0
    for n in range(1, N_harm_sum+1):
        f_n  = f_orb_PN(n, phi, nu, gamma, e, p)            # Hz
        h_nv = h_n_PN(n, e, nu, phi, gamma, p)            # strain
        S_h  = get_sensitivity(f_n)    # PSD, 1/Hz
        S_SA = 5.0 * S_h             
        # sky-averaged PSD
        SNR2 += 2 * h_nv**2 * f_n / S_SA * dt
    return SNR2

'''

Function to compute total SNR vs time (aka just the black line)

'''

def snr_total_1yr_vs_time_PN(phi_arr, nu_arr, gamma_arr, e_arr, t_arr, p, N_harm_sum):
    dt_arr = np.diff(t_arr, prepend=t_arr[0])   # seconds
    T = len(t_arr)
    SNR2_total = np.zeros(T)

    for i in range(T):
        phi = phi_arr[i]
        nu = nu_arr[i]
        gamma = gamma_arr[i]
        e = e_arr[i]
        dt = dt_arr[i]
        SNR2_total[i] = snr2_increment(phi, nu, gamma, e, dt,p, N_harm_sum)
    
    SNR_total = np.sqrt(np.cumsum(SNR2_total))

    t_to_plunge_yr = (t_arr[-1] - t_arr) / yr
    return t_to_plunge_yr, SNR_total

def snr_1yr_vs_time_PN(phi_arr, nu_arr, gamma_arr, e_arr, t_arr, p, N_harm_show=10, N_harm_sum=1000):
    '''
    Plotting the first 10 harmonics in a variety of colors along with the total SNR from first N_harm_sum harmonics. Paper generally uses 1000
    harmonics and thus that is the default.

    '''

    dt_arr = np.diff(t_arr, prepend=t_arr[0])
    T = len(t_arr)

    # total SNR^2 across all harmonics
    SNR2_total = np.zeros(T)

    # per-harmonic SNR^2 (only for first N_harm_show)
    SNR2_harm = np.zeros((N_harm_show, T))

    for i in range(T):
        phi = phi_arr[i]
        nu = nu_arr[i]
        gamma = gamma_arr[i]
        e = e_arr[i]
        dt = dt_arr[i]

        # loop through harmonics
        for n in range(1, N_harm_sum+1):

            f_n  = f_orb_PN(n, phi, nu, gamma, e, p)
            h_nv = h_n_PN(n, e, nu, phi, gamma, p)
            S_h  = get_sensitivity(f_n)
            S_SA = 5.0 * S_h

            dSNR2 = 2 * h_nv**2 * f_n / S_SA * dt

            # add to total
            SNR2_total[i] += dSNR2

            # store FIRST N_harm_show only
            if n <= N_harm_show:
                SNR2_harm[n-1, i] += dSNR2

    # convert to cumulative SNR
    SNR_total = np.sqrt(np.cumsum(SNR2_total))

    # cumulative per-harmonic SNR
    SNR_harm = np.sqrt(np.cumsum(SNR2_harm, axis=1))

    # time to plunge in years
    t_to_plunge = (t_arr[-1] - t_arr) / yr

    return t_to_plunge, SNR_total, SNR_harm

