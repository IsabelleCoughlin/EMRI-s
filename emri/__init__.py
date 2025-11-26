# emri/__init__.py

'''
EMRI: A package for Barack–Cutler (2003) EMRI evolution and harmonics.

Modules:
    constants   – physical constants and unit conversions
    peters      – Peters-Mathews evolution equations
    pn          – 2.5PN Barack-Cutler evolution
    harmonics   – g(n,e), h_n, h_c,n, waveform components
    integrator  – solve_ivp wrappers
    plotting    – utilities for EMRI plots
'''

from .constants import G, c, M_sun, yr, to_geo_seconds
from .peters import evo_PM, de_dt_PM, dnu_dt_PM
from .pn import pn_equations
from .harmonics import h_c_n, g_n_e
from .integrator import integrate_trajectory
from . import plotting

__all__ = [
    "G", "c", "M_sun", "yr", "to_geo_seconds",
    "evo_PM", "de_dt_PM", "dnu_dt_PM",
    "pn_equations",
    "h_c_n", "g_n_e",
    "integrate_trajectory",
    "plotting"
]