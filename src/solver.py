import numpy as np

def rk4_step(f, t, x, dt, params):
    k1 = f(t, x, params)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1, params)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2, params)
    k4 = f(t + dt, x + dt * k3, params)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def integrate(f, x0, t0, t_end, dt, params):
    t_values = np.arange(t0, t_end + dt, dt)
    x_values = np.zeros((len(t_values), len(x0)))

    x = x0.copy()

    for i, t in enumerate(t_values):
        x_values[i] = x
        x = rk4_step(f, t, x, dt, params)

        # Enforce positivity and normalization
        x = np.maximum(x, 0.0)
        x = x / np.sum(x)

    return t_values, x_values
