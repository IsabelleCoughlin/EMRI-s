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

from .constants import *
from .peters import *
from .harmonics import *
from .integrator import integrate_trajectory
from . import plotting

__all__ = [
    
]