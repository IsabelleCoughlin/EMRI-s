import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv  # Bessel functions
import matplotlib.pyplot as plt

# Physical constants
G   = 6.67430e-11
c   = 299792458.0
Msun = 1.98847e30
pc   = 3.085677581491367e16
yr   = 365.25*24*3600

# System parameters
Mbh = 4e6 * Msun        # central black hole mass
m   = 0.05 * Msun       # brown dwarf mass
D   = 8000 * pc          # Distance to the Galactic Center distance
Mtot = Mbh + m

# Initial orbital parameters
a0 = 0.002 * pc          # semi-major axis
e0 = 0.999              # initial eccentricity

# Pau Equation 1
plunge_radius = 8*G*Mbh/c**2

# Equation 58 of Barack
def g_n_e(n, e):
    ne = n*e
    Jn_2, Jn_1, Jn, Jn1, Jn2 = jv(n-2, ne), jv(n-1, ne), jv(n, ne), jv(n+1, ne), jv(n+2, ne)
    term1 = Jn_2 - 2*e*Jn_1 + (2/n)*Jn + 2*e*Jn1 - Jn2
    term2 = Jn_2 - 2*Jn + Jn2
    return (n**4/32.0) * (term1**2 + (1-e**2)*term2**2 + (4.0/(3*n**2))*Jn**2)

# A as a function of e defined in Peters
def a_of_e(e):
    first = a0*(e**(12/19)/(1-e**2))
    second = (1+(121/304)*e**2)**(870/2299)
    return first*second

# Equation 5.6 Peters
def da_dt(a, e):
    return -(64/5) * G**3 * m * Mbh * Mtot / (c**5 * a**3 * (1-e**2)**(7/2)) * (1 + (73/24)*e**2 + (37/96)*e**4)

# Equation 5.6 Peters
def de_dt(e):
    a = a_of_e(e)
    return -(304/15) * e * G**3 * m * Mbh * Mtot / (c**5 * a**4 * (1-e**2)**(5/2)) * (1 + (121/304)*e**2)

def rhs(t, y ):
    a, e = y
    return [da_dt(a,e), de_dt(a,e)]

def stop_plunge(t, y):
    a, e = y
    return a* (1 - e) - plunge_radius
stop_plunge.terminal = True
stop_plunge.direction = -1

sol = solve_ivp(rhs, [0, 1e15], [a0, e0], events=stop_plunge, rtol=1e-9, atol=1e-12)
t_arr, e_arr = sol.t, sol.y[0]
a_arr = a_of_e(e_arr)

# FIXME: Change to use compromise between radial and azimuthal
# Currently it is just a keplerian radial orbit i think
def f_orb(a):
    return (1/(2*np.pi)) * np.sqrt(G*Mtot / a**3)


def lisa_Sn(f):
    L = 2.5e9  # arm length [m]
    fstar = c/(2*np.pi*L)
    P_oms = (1.5e-11)**2 * (1 + (2e-3/f)**4)
    P_acc = (3e-15)**2 * (1 + (0.4e-3/f)**2) * (1 + (f/8e-3)**4)
    P_acc_term = P_acc/(2*np.pi*f)**4
    S_inst = (10/3) * ( (P_oms + 2*(1+np.cos(f/fstar)**2)*P_acc_term) / L**2 ) * (1 + 0.6*(f/fstar)**2)
    return S_inst


Mchirp = (m*Mbh)**(3/5) / (Mtot**(1/5))

def snr_harmonic(n, a, e, Tobs=1*yr):
    forb = f_orb(a) # Define the orbit frequency
    fn = n*forb # Harmonics
    if fn < 1e-5 or fn > 1:  # out of LISA band
        return 0.0
    h0 = (G**(5/3)/c**4) * (Mchirp**(5/3)) * (np.pi*forb)**(2/3) / D 
    hn = h0 * n**(2/3) * np.sqrt(max(0, g_n_e(n,e)))
    return hn * np.sqrt(Tobs/lisa_Sn(fn))

# loop over times
N = len(t_arr)
harmonics = 10
snr_h = np.zeros((harmonics,N))
snr_tot = np.zeros(N)

for i in range(N):
    a, e = a_arr[i], e_arr[i]
    vals = [snr_harmonic(n,a,e) for n in range(1,harmonics+1)]
    snr_h[:,i] = vals
    snr_tot[i] = np.sqrt(np.sum(np.array(vals)**2))


time_to_plunge = (t_arr[-1] - t_arr)/yr  # in years

plt.figure(figsize=(9,6))
for n in [1,2,3,10]:
    plt.plot(time_to_plunge, snr_h[n-1,:], label=f"n={n}")
plt.plot(time_to_plunge, snr_tot, 'k', linewidth=2, label="Total")

plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis() 
plt.xlabel("Time to plunge [years]")
plt.ylabel("SNR in 1 yr")
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.show()
