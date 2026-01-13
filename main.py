from simulations import (
    baseline_scenario,
    renewable_policy_scenario,
    nuclear_phaseout_scenario
)
from plots import plot_energy_mix

t, x = baseline_scenario()
plot_energy_mix(t, x, "Baseline scenario")

t, x = renewable_policy_scenario()
plot_energy_mix(t, x, "Renewable support scenario")

t, x = nuclear_phaseout_scenario()
plot_energy_mix(t, x, "Nuclear phase-out scenario")

