# ODE-based Modelling of Spain's Electricity Generation Mix

This project simulates the long-term evolution of Spain's electricity generation mix using a system of ordinary differential equations (ODEs) based on **replicator dynamics** from evolutionary game theory.

The model captures how five energy technologies—fossil fuels, nuclear, wind, solar, and hydroelectric—compete for market share over a time horizon under different policy scenarios.

---

## Mathematical Model

### Replicator Dynamics

The evolution of market shares is governed by the replicator equation:

```
dx_i/dt = x_i * (f_i - φ)
```

where:
- `x_i` is the market share of technology `i` (0 ≤ x_i ≤ 1, Σx_i = 1)
- `f_i = R_i * g_i` is the "fitness" of technology `i`
- `φ = Σ x_j * f_j` is the average fitness across all technologies
- `R_i` is the resource efficiency factor
- `g_i` is the intrinsic growth rate

**Key insight**: Technologies with above-average fitness (`f_i > φ`) grow, while those below average decline. This creates a competitive dynamic where the "fittest" technologies gradually dominate.

### Parameters

| Index | Technology | R (Efficiency) | g (Baseline Growth) | Notes |
|-------|------------|----------------|---------------------|-------|
| 0 | Fossil | 1.0 | 0.02 | Mature infrastructure |
| 1 | Nuclear | 1.0 | 0.015 | High upfront costs |
| 2 | Wind | 0.9 | 0.03 | Growing rapidly |
| 3 | Solar | 0.9 | 0.035 | Fastest cost decline |
| 4 | Hydro | 0.6 | 0.01 | Geographically limited |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/crisfdezs/ODE-Final-Project.git
   cd ODE-Final-Project
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Basic Usage

Run all scenarios with interactive plots:
```bash
cd examples
python3 run_scenarios.py
```

### Available Scenarios

| Scenario | Description |
|----------|-------------|
| `baseline` | Business-as-usual: current trends continue |
| `renewable` | Strong renewable support: enhanced wind/solar growth rates |
| `nuclear_phaseout` | Nuclear phase-out: negative nuclear growth after year 40 |

---

## Repository Structure

```text
ODE-Final-Project/
├── src/
│   ├── model.py              # Defines the ODE right-hand side (replicator_rhs)
│   ├── solver.py             # Implements RK4 integrator with normalization
│   ├── simulations.py        # Functions for different scenarios:
│   │                            # baseline_scenario(), renewable_policy_scenario(), nuclear_phaseout_scenario()
│   └── plots.py              # Utilities for plotting energy shares
├── examples/
│   └── run_scenarios.py      # Script that runs all scenarios and generates plots
├── README.md
└── requirements.txt
```
---

### Module Descriptions

- **model.py**: Implements the replicator dynamics equation `dx/dt = x * (f - φ)`
- **solver.py**: Fourth-order Runge-Kutta (RK4) integrator with normalization enforcement
- **simulations.py**: Defines scenario parameters (R, g values) and initial conditions
- **plots.py**: Creates line plots of market share evolution over time
- **run_scenarios.py**: CLI interface for running simulations and exporting results
---
## Example Output

Running the baseline scenario produces a plot showing:
- Solar gaining the largest market share over time (highest growth rate)
- Wind growing steadily
- Fossil and nuclear declining relatively
- Hydro remaining stable but small (geographical constraints)

The renewable policy scenario shows accelerated growth of wind and solar, with fossil fuels declining faster due to carbon pricing effects.

---

## Interpreting Results

### Market Shares
- Values represent the fraction of total electricity generation (0.0 to 1.0)
- All shares sum to 1.0 at every timestep (conservation)

### Limitations
- Simplified model: does not capture intermittency, storage, or grid constraints
- Linear fitness function: real-world dynamics may be nonlinear
- Policy shocks are abrupt: real policies phase in gradually
- No feedback loops: electricity prices, supply chains, etc. not modeled

---

## Extending the Model

### Adding New Scenarios

Edit `simulations.py` to add new scenarios:

```python
def my_custom_scenario(t0=0.0, t_end=100.0, dt=0.1):
    """My custom energy policy scenario."""
    params = {
        "R": np.array([...]),  # Resource efficiency
        "g": np.array([...])   # Growth rates
    }
    return integrate(replicator_rhs, X0_SPAIN, t0, t_end, dt, params)
```
---

## Notes

All Python scripts include detailed docstrings explaining inputs, outputs, and assumptions.

The model assumes normalized shares for all energy sources (sum of shares = 1) and enforces non-negativity.

---

## Author

Cristina Fernández-Simal Bernard.
Final Project – Introduction to Ordinary Differential Equations, Fall 2025.

