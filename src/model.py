import numpy as np

def replicator_rhs(t, x, params):

    R = params["R"]

    # Allow g to be time-dependent
    if callable(params["g"]):
        g = params["g"](t)
    else:
        g = params["g"]

    Phi = np.dot(x, R * g)
    dxdt = x * (R * g - Phi)

    return dxdt
