"""
Unit tests for scenario simulations.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add the 'src' folder to Python's module search path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from simulations import (
    baseline_scenario,
    renewable_policy_scenario,
    nuclear_phaseout_scenario,
    X0_SPAIN,
    SCENARIOS
)


class TestInitialConditions:
    """Tests for initial conditions."""

    def test_x0_spain_is_normalized(self):
        """Initial Spain shares should sum to 1."""
        assert np.isclose(np.sum(X0_SPAIN), 1.0, rtol=1e-5)

    def test_x0_spain_is_positive(self):
        """Initial Spain shares should all be non-negative."""
        assert np.all(X0_SPAIN >= 0)

    def test_x0_spain_has_five_sources(self):
        """Initial state should have 5 energy sources."""
        assert len(X0_SPAIN) == 5


class TestBaselineScenario:
    """Tests for baseline scenario."""

    def test_returns_tuple(self):
        """Should return (t, x) tuple."""
        result = baseline_scenario(t_end=10.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_shapes_consistent(self):
        """Time and state arrays should have consistent shapes."""
        t, x = baseline_scenario(t_end=10.0)
        assert len(t) == x.shape[0]
        assert x.shape[1] == 5

    def test_normalization_preserved(self):
        """Shares should sum to 1 throughout."""
        t, x = baseline_scenario(t_end=50.0)
        sums = np.sum(x, axis=1)
        assert np.allclose(sums, 1.0, atol=1e-10)

    def test_positivity_preserved(self):
        """All shares should be non-negative."""
        t, x = baseline_scenario(t_end=50.0)
        assert np.all(x >= 0)

    def test_custom_parameters(self):
        """Should accept custom time parameters."""
        t, x = baseline_scenario(t0=10.0, t_end=20.0, dt=0.5)
        assert t[0] == 10.0
        assert t[-1] >= 20.0 - 0.5


class TestRenewablePolicyScenario:
    """Tests for renewable policy scenario."""

    def test_returns_tuple(self):
        """Should return (t, x) tuple."""
        result = renewable_policy_scenario(t_end=10.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_normalization_preserved(self):
        """Shares should sum to 1 throughout."""
        t, x = renewable_policy_scenario(t_end=50.0)
        sums = np.sum(x, axis=1)
        assert np.allclose(sums, 1.0, atol=1e-10)

    def test_renewable_growth_vs_baseline(self):
        """Renewables should grow faster than in baseline."""
        t_base, x_base = baseline_scenario(t_end=50.0)
        t_renew, x_renew = renewable_policy_scenario(t_end=50.0)

        # Wind (index 2) and Solar (index 3) should be higher in renewable scenario
        # at the end of simulation
        wind_idx, solar_idx = 2, 3

        assert x_renew[-1, wind_idx] > x_base[-1, wind_idx]
        assert x_renew[-1, solar_idx] > x_base[-1, solar_idx]


class TestNuclearPhaseoutScenario:
    """Tests for nuclear phase-out scenario."""

    def test_returns_tuple(self):
        """Should return (t, x) tuple."""
        result = nuclear_phaseout_scenario(t_end=10.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_normalization_preserved(self):
        """Shares should sum to 1 throughout."""
        t, x = nuclear_phaseout_scenario(t_end=80.0)
        sums = np.sum(x, axis=1)
        assert np.allclose(sums, 1.0, atol=1e-10)

    def test_nuclear_declines_after_phaseout(self):
        """Nuclear should decline after the phase-out year."""
        t, x = nuclear_phaseout_scenario(t_end=80.0, phaseout_year=40.0)

        nuclear_idx = 1

        # Find index closest to phaseout year
        phaseout_idx = np.argmin(np.abs(t - 40.0))

        # Nuclear share at end should be less than at phaseout
        assert x[-1, nuclear_idx] < x[phaseout_idx, nuclear_idx]

    def test_custom_phaseout_year(self):
        """Should accept custom phase-out year."""
        t, x = nuclear_phaseout_scenario(t_end=60.0, phaseout_year=20.0)

        # Just verify it runs without error
        assert len(t) > 0


class TestScenarioRegistry:
    """Tests for the SCENARIOS registry."""

    def test_all_scenarios_registered(self):
        """All scenarios should be in the registry."""
        assert "baseline" in SCENARIOS
        assert "renewable" in SCENARIOS
        assert "nuclear_phaseout" in SCENARIOS

    def test_registry_functions_callable(self):
        """All registered functions should be callable."""
        for name, func in SCENARIOS.items():
            assert callable(func)

    def test_registry_functions_return_correctly(self):
        """All registered functions should return (t, x) tuples."""
        for name, func in SCENARIOS.items():
            t, x = func(t_end=5.0)
            assert isinstance(t, np.ndarray)
            assert isinstance(x, np.ndarray)
            assert len(t) == x.shape[0]
