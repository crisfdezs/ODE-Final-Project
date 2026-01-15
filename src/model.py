import numpy as np

def replicator_rhs(t, x, params):
    """
    Compute the right-hand side of the replicator equation for evolutionary dynamics.

    This function evaluates the time derivative of the strategy distribution `x` 
    according to the replicator dynamics:

        dx_i/dt = x_i * [(R * g)_i - Phi],   for i = 1, ..., n

    where:
    - R is the payoff matrix,
    - g is a (possibly time-dependent) weighting vector or function,
    - Phi = x^T * (R * g) is the average payoff of the population.

    Parameters
    ----------
    t : float
        Current time (used if `g` is time-dependent).
    x : ndarray, shape (n,)
        Current strategy distribution of the population. Each element x_i represents 
        the fraction of the population using strategy i. Should satisfy sum(x) = 1.
    params : dict
        Dictionary of model parameters, with keys:
        - "R" : ndarray, shape (n, n)
            Payoff matrix.
        - "g" : ndarray of shape (n,) or callable
            Weighting vector for strategies, or a function g(t) returning such a vector.

    Returns
    -------
    dxdt : ndarray, shape (n,)
        Time derivative of the strategy distribution x, representing the change 
        in strategy frequencies according to the replicator dynamics.

    """

    R = params["R"]

    # Allow g to be time-dependent
    if callable(params["g"]):
        g = params["g"](t)
    else:
        g = params["g"]

    Phi = np.dot(x, R * g)
    dxdt = x * (R * g - Phi)

    return dxdt
