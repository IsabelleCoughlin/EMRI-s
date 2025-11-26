from scipy.integrate import solve_ivp

def integrate_trajectory(func, M, mu y0, t_span, termination_event):
    return solve_ivp(lambda t, y: func(t, y, M, mu), t_span, y0, events = termination_event, rtol=1e-9, atol=1e-12)
