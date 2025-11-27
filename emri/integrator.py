from scipy.integrate import solve_ivp
from .params import EMRIParams

def integrate_trajectory(func, y0, t_span, termination_event, p):
    function = lambda t, y: func(t, y, p)
    event_func = lambda t, y: termination_event(t, y, p)
    event_func.terminal = True
    event_func.direction = -1
    return solve_ivp(function, t_span, y0, events = event_func, rtol=1e-9, atol=1e-12)
