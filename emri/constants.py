# constants.py
import numpy as np

c = 299792458                 # Speed of light: m/s
G = 6.6743e-11                # Gravitational Constant: m^3 / (kg s^2)
Msun = 2e30                   # Solar Masses: kg
parsec_to_m = 3e16            # meters per parsec
D = 1e3*parsec_to_m*8         # Distance to galactic center: meters
yr = 3600.0*24.0*365.0        # One year in seconds
default_integration_time = (10**9 * yr)

def mass_to_seconds(M_si):
    """Convert mass in kg → geometric seconds."""
    return G * M_si / c**3

yr = 365 * 24 * 3600
