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