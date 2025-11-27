import numpy as np
from .constants import G, c

def plunge_event_PM(t, y, p):
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
    rp_sep = (6 + 2*e) * (G*p.M_si / c**2) / (1 + e)

    return r_p - rp_sep

def plunge_event_PN(t, y, p):
    M = p.M_seconds
    
    phi, nu, gamma, e, alpha = y

    # ---- Semi-major axis in geometric units (seconds) ----
    # Barack & Cutler Eq. (5):  a = M^{1/3} / (2πν)^{2/3}
    a = M**(1/3) / ( (2*np.pi*nu)**(2/3) )

    # ---- Pericenter ----
    r_p = a * (1 - e)

    # ---- Geodesic separatrix (eccentric ISCO) ----
    r_p_sep = (6 + 2*e) * M / (1 + e)

    # Event function zero when plunge occurs:
    return r_p - r_p_sep

