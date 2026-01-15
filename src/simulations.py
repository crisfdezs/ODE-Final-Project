import numpy as np
from model import replicator_rhs
from solver import integrate

"""
Simulations of energy transition scenarios using replicator dynamics.

This module defines and runs numerical simulations of different energy
policy scenarios in Spain using a replicator equation model. The system
represents the share of different energy sources over time, with dynamics
governed by strategy-specific payoffs and growth rates.

The three scenarios implemented are:

1. baseline_scenario()
   -------------------
   Represents the current energy mix with constant growth rates,
   reflecting the status quo without policy interventions.

2. renewable_policy_scenario()
   ----------------------------
   Models the effect of policies that favor renewable energy growth,
   increasing the growth rates of renewable sources relative to others.

3. nuclear_phaseout_scenario()
   ----------------------------
   Models a policy shock that reduces nuclear energy growth after a
   given time (t = 40), simulating a gradual nuclear phase-out.

Each scenario uses the `integrate` function from `solver.py` to solve
the replicator ODE numerically, starting from the initial energy shares
defined in `x0_spain`.

Dependencies
------------
- numpy
- replicator_rhs (from model.py)
- integrate (from solver.py)

Returns
-------
Each scenario function returns:
- t_values : ndarray
    Array of time points where the solution was evaluated.
- x_values : ndarray
    Array containing the state vector (energy shares) at each time point.

Notes
-----
- The weight vectors `g` represent growth rates for each energy source,
  and `R` represents payoffs or relative importance.
- The normalization in `integrate` ensures that the energy shares sum to 1.
- Time is measured in arbitrary units consistent across all scenarios.
"""

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
