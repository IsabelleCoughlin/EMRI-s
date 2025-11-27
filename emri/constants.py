# constants.py
import numpy as np

c = 299792458                           # Speed of light: m/s
G = 6.6743e-11                          # Gravitational Constant: m^3 / (kg s^2)
M_sun = 2e30                            # Solar Masses: kg
parsec_to_m = 3e16                      # meters per parsec
D = 1e3*parsec_to_m*8                   # Distance to galactic center: meters
yr = 3600.0*24.0*365.0                  # One year in seconds
default_integration_time = (10**9 * yr) # Default integration time: seconds
D_seconds = D/c                         # Distance to galactic center: seconds