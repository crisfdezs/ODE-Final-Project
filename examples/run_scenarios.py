from simulations import (
    baseline_scenario,
    renewable_policy_scenario,
    nuclear_phaseout_scenario
)
from plots import plot_energy_mix

"""
Run and visualize energy transition scenarios using replicator dynamics.

This script executes three predefined energy policy scenarios for Spain
using the replicator dynamics model and visualizes the evolution of energy
shares over time. The scenarios are:

1. baseline_scenario() 
   - Current energy mix with no policy interventions.

2. renewable_policy_scenario() 
   - Simulates policies favoring renewable energy growth.

3. nuclear_phaseout_scenario() 
   - Models a policy shock reducing nuclear energy growth after a
     specified time (t = 40), simulating a nuclear phase-out.

For each scenario, the script:
- Computes the time evolution of energy shares using the `integrate` function.
- Plots the results using `plot_energy_mix`, including time series and legends.

Dependencies
------------
- simulations.py : contains the scenario functions.
- plots.py       : contains the `plot_energy_mix` function.
- numpy          : for numerical computations.
- matplotlib     : for plotting results.

Outputs
-------
- Three plots displayed sequentially, showing the evolution of energy shares
  for each scenario.
- Each plot includes all energy sources:
  ["Fossil", "Nuclear", "Wind", "Solar", "Hydro"].
- Plots are intended for visualization and interpretation in the final report.

Notes
-----
- The script does not save plots to files by default. To include figures in
  the report, plots can be saved using plt.savefig() within `plot_energy_mix`.
- Time and energy share units are consistent with the replicator dynamics model.
"""

t, x = baseline_scenario()
plot_energy_mix(t, x, "Baseline scenario")

t, x = renewable_policy_scenario()
plot_energy_mix(t, x, "Renewable support scenario")

t, x = nuclear_phaseout_scenario()
plot_energy_mix(t, x, "Nuclear phase-out scenario")

