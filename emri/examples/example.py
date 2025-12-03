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

a0 = 2e-3*parsec_to_m         #initial semi-major axis
e0 = 0.9     

# initial conditions
a0 =  2e-3*parsec_to_m
a0 = (17/(1-e0))*parsec_to_m

#nu_0 = (1/(2*np.pi)) * np.sqrt(Mbh/ a0**3)

#y0 = [0, nu_0, 0, 0.9, 0.0]

y0 = [a0, e0]

# Define system parameters
p = EMRIParams(
    M_bh=4e6 * M_sun,
    mu=0.05 * M_sun,
    X=0.0,
    lambda_var=0.0
)

# Integrate and Interpolate
sol =  integrate_trajectory(deriv_PM, y0, (0, default_integration_time), plunge_event_PM, p)
t_arr = sol.t
a_arr, e_arr = sol.y  
#a_interpolated, e_interpolated, t_interpolated, t_to_plunge_yr_interpolated = interpolate_PM(a_arr, e_arr, t_arr)

#plot_e(t_to_plunge_yr_interpolated, e_interpolated)
plot_e(t_arr, e_arr)
#plot_characteristic_strain_PM(t_to_plunge_yr_interpolated, e_interpolated, a_interpolated, p)
#plot_SNR(a_interpolated, e_interpolated, t_interpolated,p, 10, 10, 1000)