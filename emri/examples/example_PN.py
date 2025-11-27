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

a0 = 2e-3*parsec_to_m/c         #initial semi-major axis
e0 = 0.999     

# initial conditions

# Define system parameters
p = EMRIParams(
    M_bh=4e6 * M_sun,
    mu=0.05 * M_sun,
    X=0.0,
    lambda_var=0.0
)

a0 =  2e-3*(parsec_to_m/c)
nu_0 = (1/(2*np.pi)) * np.sqrt(p.M_seconds/ a0**3)
y0 = [0, nu_0, 0, 0.999, 0.0]

# Integrate and Interpolate
sol =  integrate_trajectory(deriv_PN, y0, (0, default_integration_time), plunge_event_PN, p)
t_arr = sol.t
phi_arr, nu_arr, gamma_arr, e_arr, alpha_arr = sol.y
e_interpolated, nu_interpolated, phi_interpolated, gamma_interpolated, alpha_interpolated, t_interpolated, t_to_plunge_yr_interpolated = interpolate_PN(e_arr, nu_arr, phi_arr, gamma_arr, alpha_arr, t_arr)

#plot_e(t_to_plunge_yr_interpolated, e_interpolated)
#plot_characteristic_strain_PN(t_to_plunge_yr_interpolated, e_interpolated, nu_interpolated, phi_interpolated, gamma_interpolated, p)
plot_SNR_PN(phi_interpolated, nu_interpolated, gamma_interpolated, e_interpolated, t_interpolated, p, 10, 10, 1000)