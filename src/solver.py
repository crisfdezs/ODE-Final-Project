import numpy as np

def rk4_step(f, t, x, dt, params):
    """
    Perform a single Runge-Kutta 4th order (RK4) step for a system of ODEs.

    This function computes the next state of the system after a timestep `dt`
    using the classical RK4 method.

    Parameters
    ----------
    f : callable
        Function that computes the right-hand side of the ODE system, 
        f(t, x, params), returning dx/dt as a NumPy array.
    t : float
        Current time.
    x : ndarray, shape (n,)
        Current state vector of the system.
    dt : float
        Time step size.
    params : dict
        Dictionary of parameters to pass to the function `f`.

    Returns
    -------
    x_next : ndarray, shape (n,)
        Estimated state vector at time t + dt.
    """
    k1 = f(t, x, params)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1, params)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2, params)
    k4 = f(t + dt, x + dt * k3, params)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate(f, x0, t0, t_end, dt, params):
    """
    Integrate a system of ODEs over a time interval using RK4.

    This function performs a full numerical integration from t0 to t_end
    using fixed timesteps `dt`, applying the `rk4_step` function at each step.
    After each step, the state vector is enforced to be non-negative
    and normalized (useful for population or probability distributions).

    Parameters
    ----------
    f : callable
        Function computing the right-hand side of the ODE system, 
        f(t, x, params), returning dx/dt as a NumPy array.
    x0 : ndarray, shape (n,)
        Initial state vector at time t0.
    t0 : float
        Initial time.
    t_end : float
        Final time of integration.
    dt : float
        Time step size.
    params : dict
        Dictionary of parameters to pass to the function `f`.

    Returns
    -------
    t_values : ndarray, shape (m,)
        Array of time points where the solution was evaluated.
    x_values : ndarray, shape (m, n)
        Array containing the state vector at each time point.

    Notes
    -----
    - The normalization step ensures that sum(x_values[i]) = 1 at each time point,
      which is appropriate for replicator dynamics or probability distributions.
    - The function assumes a fixed timestep; adaptive methods are not implemented.
    """
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
