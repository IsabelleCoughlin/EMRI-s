from emri.peters import *
from emri.integrator import *
from emri.plotting import *
from emri.constants import *

M = 4e6 * Msun                # SMBH Mass: kg
m = 0.05 * Msun           # CO Mass: kg
mu = (M * m) / (M + m)# reduced mass
a0 = 2e-3*parsec_to_m         #initial semi-major axis
e0 = 0.999       

print(M)