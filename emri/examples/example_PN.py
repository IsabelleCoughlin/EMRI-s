from emri.peters import *
from emri.integrator import *
from emri.plotting import *
from emri.constants import *
from emri.events import *
from emri.params import EMRIParams
from emri.interpolation import *
from emri.harmonics import *
from emri.LISA_sensitivity import *
from emri.PN import *

# initial conditions

# Define system parameters
p = EMRIParams(
    M_bh=1e6 * M_sun,
    mu=10 * M_sun,
    X=0.0,
    lambda_var=0.0, 
    D = 1e9*parsec_to_m
)

'''
# Starting from initial conditions
a0 = 6e-3*parsec_to_m/c         #initial semi-major axis
e0 = 0.77 
#nu0 = (1/(2*np.pi)) * np.sqrt(p.M_seconds/ a0**3)
nu0 = 0.23e-2
y0 = [0, nu0, 0, e0, 0.0]

# Starting from final lso
'''
e_LSO = 0.3
nu0 = nu_LSO(e_LSO, p)
print(nu0)
#y0 = [e_LSO, nu0]
y0 = [0, nu0, 0, e_LSO, 0]
# integrate backwards in time from t=0 (LSO)
t_span = (0.0, -10.0 * yr)
sol = integrate_trajectory(deriv_PN, y0, t_span, p=p)

# Integrate and Interpolate
#sol =  integrate_trajectory(deriv_PN, y0, (0, default_integration_time), plunge_event_PN, p)
t_arr = sol.t
phi_arr, nu_arr, gamma_arr, e_arr, alpha_arr = sol.y
e_interpolated, nu_interpolated, phi_interpolated, gamma_interpolated, alpha_interpolated, t_interpolated, t_to_plunge_yr_interpolated = interpolate_PN(e_arr, nu_arr, phi_arr, gamma_arr, alpha_arr, t_arr)

#target = 0.9
#idx = np.argmin(np.abs(e_interpolated - target))
#print("Index:", idx, "Value:", e_interpolated[idx])

# 2. Truncate arrays *starting at* this index (keep everything after)
#e_trunc = e_interpolated[idx:]
#t_trunc = t_interpolated[idx:]

# If you want to keep everything *before* the index instead:
#e_trunc = e_arr[:idx+1]
#t_trunc = t_arr[:idx+1]

#print("Truncated e:", e_interpolated)
#print("Truncated t:", t_interpolated)

#t__plunge_trunc = (t_trunc[-1] - t_trunc) / yr
#plot_e(t__plunge_trunc/(10**5), e_trunc, "PN")
#plot_e(t_to_plunge_yr_interpolated, e_interpolated)
#plot_characteristic_strain_PN(t_to_plunge_yr_interpolated, e_interpolated, nu_interpolated, phi_interpolated, gamma_interpolated, p)
#plot_characteristic_strain_PN_by_n_2(t_to_plunge_yr_interpolated, e_interpolated, nu_interpolated, phi_interpolated, gamma_interpolated, p)
#plot_SNR_PN(phi_interpolated, nu_interpolated, gamma_interpolated, e_interpolated, t_interpolated, p, 10, 1000, 1000)
'''

#plot_nu_LSO(p)

plt.figure(figsize=(7, 5))

ecc_list = [0.3]#[0.0001, 0.001, 0.005, 0.01, 0.05, 0.07, 0.10]

# choose a colormap
cmap = plt.cm.viridis
colors = cmap(np.linspace(0.1, 0.9, len(ecc_list)))

for e_LSO, color in zip(ecc_list, colors):
    nu0 = nu_LSO(e_LSO, p)
    y0 = [e_LSO, nu0]

    # integrate backwards in time from t=0 (LSO)
    t_span = (0.0, -1e2 * yr)
    sol = integrate_trajectory(deriv_PN_reduced, y0, t_span, p=p)

    t_array = sol.t / yr
    t_array = np.abs(t_array)
    e_arr, nu_arr = sol.y

    plt.plot(
        t_array,
        e_arr,
        color=color,
        lw=1.5,
        label=fr"$e_{{\rm LSO}} = {e_LSO}$"
    )
    
plt.gca().invert_xaxis()
plt.xlabel("Time before plunge [yr]")
plt.ylabel("Eccentricity")
plt.grid(alpha=0.3)
plt.title(r"Back-evolution of different final $e_{lso}$ using 2.5PN")
plt.legend(fontsize=9)
plt.show()

plt.plot(e_arr, t_array)
'''

plot_characteristic_strain_PN_by_n(
    t_to_plunge_yr_interpolated,
    e_interpolated, nu_interpolated, phi_interpolated, gamma_interpolated,
    p,
    n_list=range(1, 10),
    time_arrays=(5, 2, 1),
    t_window_yr=(0, 10)
)

#plot_analytical()
