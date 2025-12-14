from scipy.integrate import solve_ivp
from .params import EMRIParams
'''
def integrate_trajectory(func, y0, t_span, termination_event, p):
    function = lambda t, y: func(t, y, p)
    event_func = lambda t, y: termination_event(t, y, p)
    event_func.terminal = True
    event_func.direction = -1
    return solve_ivp(function, t_span, y0, events = event_func, rtol=1e-9, atol=1e-12)

'''
def integrate_trajectory(func, y0, t_span, termination_event=None, p=None):
    # Wrap the ODE so it always has access to p
    function = lambda t, y: func(t, y, p)

    # If no termination event provided, just call solve_ivp normally
    if termination_event is None:
        return solve_ivp(
            function,
            t_span,
            y0,
            rtol=1e-9,
            atol=1e-12
        )

    # Otherwise, wrap the event so it includes p
    def event_func(t, y):
        return termination_event(t, y, p)

    event_func.terminal = True
    event_func.direction = -1

    return solve_ivp(
        function,
        t_span,
        y0,
        events=[event_func],   # ← must be a list!
        rtol=1e-9,
        atol=1e-12
    )