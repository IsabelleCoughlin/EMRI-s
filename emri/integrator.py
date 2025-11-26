from scipy.integrate import solve_ivp

def integrate_trajectory(func, y0, t_span, termination_event, **kwargs):
    return solve_ivp(func, t_span, y0, event = termination_event, rtol=1e-9, atol=1e-12, **kwargs)