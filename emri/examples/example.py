from emri.peters import *
from emri.integrator import *
from emri.plotting import *
from emri.constants import *
from emri.events import *

from emri.params import EMRIParams
from emri.integrator import integrate
from emri.peters import deriv_PM
from emri.events import plunge_event_PM

M = 4e6 * Msun                # SMBH Mass: kg
m = 0.05 * Msun           # CO Mass: kg
mu = (M * m) / (M + m)# reduced mass
a0 = 2e-3*parsec_to_m         #initial semi-major axis
e0 = 0.999       

# Define system parameters
p = EMRIParams(
    M_bh=1e6*M_sun,
    m=1*M_sun,
    X=0.7,
    lambda_var=0.3
)

# Initial conditions
a0 = ...
e0 = ...
y0 = [a0, e0]

# Integrate
sol = integrate(
    func=deriv_PM,
    y0=y0,
    t_span=(0, 1e9),
    event=plunge_event_PM,
    params=p
)