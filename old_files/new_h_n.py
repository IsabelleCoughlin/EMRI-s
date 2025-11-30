import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from few.waveform import GenerateEMRIWaveform
from few.utils.constants import MTSUN_SI
from few.utils.utility import get_p_at_t
from few.utils.geodesic import get_fundamental_frequencies
from few.utils.fdutils import GetFDWaveformFromFD, GetFDWaveformFromTD
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux

from scipy.interpolate import CubicSpline
from few import get_file_manager

# produce sensitivity function
traj_module = EMRIInspiral(func=KerrEccEqFlux)

# import ASD
data = np.loadtxt(get_file_manager().get_file("LPA.txt"), skiprows=1)
data[:, 1] = data[:, 1] ** 2
# define PSD function
get_sensitivity = CubicSpline(*data.T)

print("f, ASD, PSD:", data[10,0], np.sqrt(get_sensitivity(data[10,0])), get_sensitivity(data[10,0]))


# Physical constants
G   = 6.67430e-11 # 
c   = 299792458.0
Msun = 1.98847e30
pc   = 3.085677581491367e16
yr   = 365.25*24*3600

# System parameters
Mbh = 4e6 * Msun        # central black hole mass
m   = 0.05 * Msun       # brown dwarf mass
D   = 8000 * pc         # Distance to GC
Mtot = Mbh + m
mu = (Mbh*m)/Mtot # Since EMRI, probably can just be equal to m

# Initial orbital parameters
a0 = 0.002 * pc         # semi-major axis
e0 = 0.999              # initial eccentricity

# Plunge radius
plunge_radius = 8*G*Mbh/c**2

# g(n,e) from Peters / Barack & Cutler
def g_n_e(n, e):
    ne = n*e
    Jn_2 = jv(n-2, ne)
    Jn_1 = jv(n-1, ne)
    Jn   = jv(n,   ne)
    Jn1  = jv(n+1, ne)
    Jn2  = jv(n+2, ne)
    term1 = Jn_2 - 2*e*Jn_1 + (2.0/n)*Jn + 2*e*Jn1 - Jn2
    term2 = Jn_2 - 2*Jn + Jn2
    return (n**4/32.0) * (term1**2 + (1 - e**2)*term2**2 + (4.0/(3*n**2))*(Jn**2))

# Peters da/dt and de/dt (use a and e directly)
def da_dt(a, e):
    one_e2 = max(1e-16, 1 - e**2)
    return -(64/5) * G**3 * m * Mbh * Mtot / (c**5 * a**3 * one_e2**(7/2)) * (1 + (73/24)*e**2 + (37/96)*e**4)

def de_dt(a, e):
    one_e2 = max(1e-16, 1 - e**2)
    return -(304/15) * e * G**3 * m * Mbh * Mtot / (c**5 * a**4 * one_e2**(5/2)) * (1 + (121/304)*e**2)

# ODE rhs
def rhs(t, y):
    a, e = y
    return [da_dt(a, e), de_dt(a, e)]

# Plunge event
def stop_plunge(t, y):
    a, e = y
    return a*(1 - e) - plunge_radius
stop_plunge.terminal = True
stop_plunge.direction = -1

# Integrate
tmax = 1e15
# small max_step helps when eccentricity is extreme
sol = solve_ivp(rhs, [0, tmax], [a0, e0], events=stop_plunge,
                rtol=1e-9, atol=1e-12, max_step=1e11)
if sol.status == 1 and len(sol.t_events[0])>0:
    print("Plunge event at t =", sol.t_events[0][0], "s")
t_arr = sol.t
a_arr = sol.y[0]   # a is first state variable
e_arr = sol.y[1]   # e is second

# Orbital frequency
def f_orb(a):
    return (1/(2*np.pi)) * np.sqrt(G*Mtot / a**3)

def derivE(n, a, e):
    nu = f_orb(a)
    return g_n_e(n, e)*(32.0/5.0)*mu**2 * (G*Mtot)**(4.0/3.0) * (2*np.pi*nu)**(10.0/3.0)

def derivF(n, a, e):
    pre = -3/(4*np.pi)
    sqrt = (math.sqrt((a**3)/G*Mtot))/(a)
    da = da_dt(a, e)
    return pre*sqrt*da

def h_n_deriv(n, a, e):
    E_dot = derivE(n, a, e)
    f_dot = derivF(n, a, e)
    h_n = math.sqrt(2*E_dot/f_dot)/(np.pi*D)
    return h_n

def Sn_lisa(f):
    f = np.array(f, dtype=float)
    f[f <= 1e-12] = 1e-12
    S_inst = (9.18*10**(-52)/(f**4)) + (1.59*(10**(-41))) + (9.18*(10**(-38)*(f**2)))
    S_ex_gal = 4.2*(10**(-47))*(f**(-7/3))
    dN = 2*(10**(-3))*((1/f)**(11/3))
    S_gal = 2.1*(10**(-45))*((f)**(-7/3))
    S_inst_gal = min(S_inst/math.exp(-1.5/yr)*dN, S_inst + S_gal)
    return S_inst_gal + S_ex_gal

def SN_FEW(f):
    return np.sqrt(get_sensitivity(f))


def snr_harmonic(n, a, e, Tobs=1*yr):
    forb = f_orb(a)
    fn = n * forb
    h_n = ((G**2)*(m*Mbh)/((c**4)*a*D))*g_n_e(n, e)
    #return (h_n/SN_FEW(fn))**2
    S_n = get_sensitivity(fn)
    h_n_c = h_n_deriv(n, a, e)
    return h_n_c**2/(fn**2*S_n)

'''
def snr_harmonic(n, a, e, Tobs=1*yr):
    forb = f_orb(a)
    fn = n * forb
    h_n = ((G**2)*(m*Mbh)/((c**4)*a*D))*g_n_e(n, e)
    #return (h_n/SN_FEW(fn))**2
    S_n = get_sensitivity(fn)
    return h_n*math.sqrt(Tobs/S_n)
'''  

# compute SNR time-series
N = len(t_arr)
n = 10
snr_h = np.zeros((n, N))
snr_tot = np.zeros(N)

# Loop over each time?
for i in range(N):
    a, e = a_arr[i], e_arr[i]
    vals = [snr_harmonic(n, a, e) for n in range(1, n+1)] # loop over the N vals
    snr_h[:, i] = vals # Save to a value just for that time
    snr_tot[i] = np.sqrt(np.sum(np.array(vals)**2)) # Add to total SNR (black line)

time_to_plunge = (t_arr[-1] - t_arr) / yr

plt.figure(figsize=(9,6))
for n in [1,2,3,10]:
    plt.plot(time_to_plunge, snr_h[n-1,:], label=f"n={n}")
plt.plot(time_to_plunge, snr_tot, 'k', linewidth=2, label="Total")
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()
plt.xlabel("Time to plunge [yr]")
plt.ylabel("SNR in 1 yr")
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.show()
