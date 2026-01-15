"""
Unit tests for the ODE solver.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add the 'src' folder to Python's module search path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from solver import rk4_step, integrate, validate_inputs


class TestValidateInputs:
    """Tests for input validation."""

    def test_valid_inputs_pass(self):
        """Valid inputs should not raise."""
        x0 = np.array([0.2, 0.3, 0.2, 0.2, 0.1])
        validate_inputs(x0, 0.0, 100.0, 0.1)  # Should not raise

    def test_negative_dt_raises(self):
        """Negative time step should raise ValueError."""
        x0 = np.array([0.2, 0.3, 0.2, 0.2, 0.1])
        with pytest.raises(ValueError, match="positive"):
            validate_inputs(x0, 0.0, 100.0, -0.1)

    def test_zero_dt_raises(self):
        """Zero time step should raise ValueError."""
        x0 = np.array([0.2, 0.3, 0.2, 0.2, 0.1])
        with pytest.raises(ValueError, match="positive"):
            validate_inputs(x0, 0.0, 100.0, 0.0)

    def test_invalid_time_range_raises(self):
        """End time before start time should raise ValueError."""
        x0 = np.array([0.2, 0.3, 0.2, 0.2, 0.1])
        with pytest.raises(ValueError, match="greater than"):
            validate_inputs(x0, 100.0, 0.0, 0.1)

    def test_negative_initial_state_raises(self):
        """Negative values in initial state should raise ValueError."""
        x0 = np.array([0.2, -0.1, 0.3, 0.3, 0.3])
        with pytest.raises(ValueError, match="negative"):
            validate_inputs(x0, 0.0, 100.0, 0.1)

    def test_non_normalized_initial_state_raises(self):
        """Initial state not summing to 1 should raise ValueError."""
        x0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Sums to 1.0
        validate_inputs(x0, 0.0, 100.0, 0.1)  # Should pass

        x0_bad = np.array([0.3, 0.3, 0.3, 0.3, 0.3])  # Sums to 1.5
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_inputs(x0_bad, 0.0, 100.0, 0.1)


class TestRk4Step:
    """Tests for single RK4 step."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        def f(t, x, params):
            return -x  # Simple decay

        x = np.array([1.0, 0.5, 0.3])
        x_new = rk4_step(f, 0.0, x, 0.1, {})
        assert x_new.shape == x.shape

    def test_exponential_decay_accuracy(self):
        """RK4 should accurately solve exponential decay."""
        def f(t, x, params):
            return -x  # dx/dt = -x, solution: x(t) = x0 * exp(-t)

        x0 = np.array([1.0])
        dt = 0.01
        t = 0.0
        x = x0.copy()

        # Take 100 steps to t=1
        for _ in range(100):
            x = rk4_step(f, t, x, dt, {})
            t += dt

        # Compare to analytical solution: exp(-1) = 0.3679
        expected = np.exp(-1.0)
        assert np.isclose(x[0], expected, rtol=1e-6)

    def test_linear_ode_accuracy(self):
        """RK4 should accurately solve linear ODE."""
        def f(t, x, params):
            return np.array([1.0])  # dx/dt = 1, solution: x = x0 + t

        x = np.array([0.0])
        dt = 0.1

        for i in range(10):
            x = rk4_step(f, i * dt, x, dt, {})

        # After 10 steps of dt=0.1, should be at t=1, x=1
        assert np.isclose(x[0], 1.0, rtol=1e-10)


class TestIntegrate:
    """Tests for the full integration function."""

    def test_output_shapes(self):
        """Output arrays should have correct shapes."""
        def f(t, x, params):
            return np.zeros_like(x)

        x0 = np.array([0.5, 0.3, 0.2])
        t, x = integrate(f, x0, 0.0, 10.0, 0.1, {})

        assert len(t) == len(x)
        assert x.shape[1] == len(x0)

    def test_normalization_preserved(self):
        """Output should always sum to 1 at each timestep."""
        from model import replicator_rhs

        x0 = np.array([0.2, 0.3, 0.2, 0.2, 0.1])
        params = {
            "R": np.array([1.0, 1.0, 0.9, 0.9, 0.6]),
            "g": np.array([0.02, 0.015, 0.03, 0.035, 0.01])
        }

        t, x = integrate(replicator_rhs, x0, 0.0, 100.0, 0.1, params)

        # Check normalization at all timesteps
        sums = np.sum(x, axis=1)
        assert np.allclose(sums, 1.0, atol=1e-10)

    def test_positivity_preserved(self):
        """Output should always be non-negative."""
        from model import replicator_rhs

        x0 = np.array([0.2, 0.3, 0.2, 0.2, 0.1])
        params = {
            "R": np.array([1.0, 1.0, 0.9, 0.9, 0.6]),
            "g": np.array([0.02, 0.015, 0.03, 0.035, 0.01])
        }

        t, x = integrate(replicator_rhs, x0, 0.0, 100.0, 0.1, params)

        # All values should be non-negative
        assert np.all(x >= 0)

    def test_initial_condition_preserved(self):
        """First row of output should match initial condition."""
        def f(t, x, params):
            return np.zeros_like(x)

        x0 = np.array([0.5, 0.3, 0.2])
        t, x = integrate(f, x0, 0.0, 10.0, 0.1, {})

        assert np.allclose(x[0], x0)

    def test_time_array_correct(self):
        """Time array should span correct range."""
        def f(t, x, params):
            return np.zeros_like(x)

        x0 = np.array([0.5, 0.5])
        t, x = integrate(f, x0, 0.0, 10.0, 0.1, {})

        assert t[0] == 0.0
        assert t[-1] >= 10.0 - 0.1  # Allow for floating point

    def test_validation_can_be_disabled(self):
        """Should be able to skip validation for performance."""
        def f(t, x, params):
            return np.zeros_like(x)

        # Invalid initial state (doesn't sum to 1)
        x0 = np.array([0.3, 0.3])

        # Should raise with validation enabled
        with pytest.raises(ValueError):
            integrate(f, x0, 0.0, 10.0, 0.1, {}, validate=True)

        # Should not raise with validation disabled
        t, x = integrate(f, x0, 0.0, 10.0, 0.1, {}, validate=False)
        assert len(t) > 0
