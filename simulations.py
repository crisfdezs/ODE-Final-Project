import numpy as np
from model import replicator_rhs
from solver import integrate

# We define the weight of the different energy sources as close to reality as possible
x0_spain = np.array([0.038, 0.256, 0.244, 0.295, 0.167])

def baseline_scenario():

    params = {
        "R": np.array([1.0, 1.0, 0.9, 0.9, 0.6]),
        "g": np.array([0.02, 0.015, 0.03, 0.035, 0.01])
    }

    return integrate(replicator_rhs, x0_spain, 0.0, 100.0, 0.1, params)


def renewable_policy_scenario():

    params = {
        "R": np.array([1.0, 1.0, 0.9, 0.9, 0.6]),
        "g": np.array([0.01, 0.015, 0.05, 0.06, 0.01])
    }

    return integrate(replicator_rhs, x0_spain, 0.0, 100.0, 0.1, params)


def nuclear_phaseout_scenario():
    """
    Nuclear growth becomes negative after a policy shock.
    """

    R = np.array([1.0, 1.0, 0.9, 0.9, 0.6])

    def g_time(t):
        if t < 40.0:
            return np.array([0.02, 0.015, 0.03, 0.035, 0.01])
        else:
            return np.array([0.02, -0.03, 0.035, 0.04, 0.01])

    params = {
        "R": R,
        "g": g_time
    }

    return integrate(replicator_rhs, x0_spain, 0.0, 100.0, 0.1, params)
