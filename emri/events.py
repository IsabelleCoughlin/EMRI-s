import numpy as np
from .constants import G, c

def plunge_event_PM(t, y, M):
    """
    Peters–Mathews plunge event at ISCO.
    Trigger termination when pericenter equals the relativistic separatrix.

    Parameters
    ----------
    t : float
        Time (not used)
    y : array-like
        State vector [a, e]
    M : float
        Mass of MBH (SI units or geometric seconds depending on your integrator)
    """
    a, e = y

    # pericenter distance
    r_p = a * (1 - e)

    # relativistic separatrix (Barack & Cutler use p = 6 + 2e)
    rp_sep = (6 + 2*e) * (G*M / c**2) / (1 + e)

    return r_p - rp_sep

# required by solve_ivp
plunge_event_PM.terminal = True
plunge_event_PM.direction = -1


