# ODE-based Modelling of Spain’s Electricity Generation Mix

This repository contains Python code for simulating the long-term evolution of Spain’s electricity generation mix using a simplified system of ordinary differential equations (ODEs). The model focuses on five main technologies: fossil fuels, nuclear, wind, solar, and hydroelectric power. Their shares evolve over time according to **replicator dynamics**, capturing competition for a fixed total electricity generation.

---

## Repository Structure

- `model.py` – Defines the ODE right-hand side (`replicator_rhs`) for energy share dynamics.  
- `solver.py` – Implements a classical fourth-order Runge–Kutta (RK4) integrator with normalization to preserve total shares.  
- `simulations.py` – Contains functions for running different scenarios:
  - `baseline_scenario()`
  - `renewable_policy_scenario()`
  - `nuclear_phaseout_scenario()`  
- `plots.py` – Utilities for plotting the evolution of energy shares over time.  
- `main.py` – Example script that runs all scenarios and generates plots.

---

## Dependencies

- `numpy`  
- `matplotlib`
