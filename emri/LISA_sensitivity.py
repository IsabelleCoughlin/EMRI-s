from few.waveform import GenerateEMRIWaveform
from few.utils.constants import MTSUN_SI
from few.utils.utility import get_p_at_t
from few.utils.geodesic import get_fundamental_frequencies
from few.utils.fdutils import GetFDWaveformFromFD, GetFDWaveformFromTD
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux
from scipy.interpolate import CubicSpline
from few import get_file_manager
from scipy.integrate import cumulative_trapezoid
import numpy as np

'''
Calculating FEW waveform to develop the sensitivity of LISA instrument
'''
# produce sensitivity function
traj_module = EMRIInspiral(func=KerrEccEqFlux)

# import ASD
data = np.loadtxt(get_file_manager().get_file("LPA.txt"), skiprows=1)
frequencies = data[:,0]
psd_values  = data[:, 1] ** 2
# define PSD function
get_sensitivity = CubicSpline(frequencies, psd_values)

def h_det(f):
    '''
    Return FEW Waveform sensitivity
    '''
    return np.sqrt(5*f*(get_sensitivity(f)))

def analytical_sensitivity(f):
    '''
    Analytical Waveform sensitivity
    '''
    f = np.array(f, dtype=float)
    f[f <= 1e-12] = 1e-12
    S_inst = (9.18*10**(-52)/(f**4)) + (1.59*(10**(-41))) + (9.18*(10**(-38)*(f**2)))
    S_ex_gal = 4.2*(10**(-47))*(f**(-7/3))
    dN = 2*(10**(-3))*((1/f)**(11/3))
    S_gal = 2.1*(10**(-45))*((f)**(-7/3))
    S_inst_gal = np.minimum(S_inst/math.exp(-1.5/yr)*dN, S_inst + S_gal)
    return S_inst_gal + S_ex_gal