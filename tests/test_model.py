"""
Unit tests for the replicator dynamics model.
"""

import numpy as np
import pytest

from model import replicator_rhs


class TestReplicatorRhs:
    """Tests for the replicator_rhs function."""

    def test_output_shape(self):
        """Output should have same shape as input state."""
        x = np.array([0.2, 0.3, 0.2, 0.2, 0.1])
        params = {
            "R": np.array([1.0, 1.0, 0.9, 0.9, 0.6]),
            "g": np.array([0.02, 0.015, 0.03, 0.035, 0.01])
        }
        dxdt = replicator_rhs(0.0, x, params)
        assert dxdt.shape == x.shape

    def test_output_is_numpy_array(self):
        """Output should be a numpy array."""
        x = np.array([0.2, 0.3, 0.2, 0.2, 0.1])
        params = {
            "R": np.ones(5),
            "g": np.ones(5) * 0.02
        }
        dxdt = replicator_rhs(0.0, x, params)
        assert isinstance(dxdt, np.ndarray)

    def test_zero_sum_property(self):
        """Derivatives should sum to approximately zero (conservation)."""
        x = np.array([0.2, 0.3, 0.2, 0.2, 0.1])
        params = {
            "R": np.array([1.0, 1.0, 0.9, 0.9, 0.6]),
            "g": np.array([0.02, 0.015, 0.03, 0.035, 0.01])
        }
        dxdt = replicator_rhs(0.0, x, params)
        # Sum of derivatives should be zero (total share is conserved)
        assert np.isclose(np.sum(dxdt), 0.0, atol=1e-10)

    def test_uniform_fitness_gives_zero_derivatives(self):
        """When all technologies have equal fitness, no change occurs."""
        x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        params = {
            "R": np.ones(5),
            "g": np.ones(5) * 0.02  # All same growth rate
        }
        dxdt = replicator_rhs(0.0, x, params)
        # All derivatives should be zero when fitness is uniform
        assert np.allclose(dxdt, 0.0, atol=1e-10)

    def test_zero_share_stays_zero(self):
        """Technology with zero share should have zero derivative."""
        x = np.array([0.0, 0.3, 0.3, 0.3, 0.1])
        params = {
            "R": np.array([1.0, 1.0, 0.9, 0.9, 0.6]),
            "g": np.array([0.02, 0.015, 0.03, 0.035, 0.01])
        }
        dxdt = replicator_rhs(0.0, x, params)
        # First technology has zero share, so derivative should be zero
        assert dxdt[0] == 0.0

    def test_time_dependent_growth_rate(self):
        """Time-dependent g function should be called with current time."""
        x = np.array([0.2, 0.3, 0.2, 0.2, 0.1])

        def g_time(t):
            if t < 50:
                return np.array([0.02, 0.015, 0.03, 0.035, 0.01])
            else:
                return np.array([0.01, -0.02, 0.04, 0.05, 0.01])

        params = {
            "R": np.array([1.0, 1.0, 0.9, 0.9, 0.6]),
            "g": g_time
        }

        # Results should be different for t < 50 and t >= 50
        dxdt_before = replicator_rhs(25.0, x, params)
        dxdt_after = replicator_rhs(75.0, x, params)

        assert not np.allclose(dxdt_before, dxdt_after)

    def test_higher_fitness_grows(self):
        """Technology with above-average fitness should have positive derivative."""
        x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        params = {
            "R": np.array([1.0, 1.0, 1.0, 2.0, 1.0]),  # Solar has higher R
            "g": np.ones(5) * 0.02
        }
        dxdt = replicator_rhs(0.0, x, params)
        # Solar (index 3) should have positive derivative
        assert dxdt[3] > 0


class TestReplicatorRhsEdgeCases:
    """Edge case tests for replicator_rhs."""

    def test_single_dominant_technology(self):
        """When one technology dominates, it should remain stable."""
        x = np.array([0.99, 0.0025, 0.0025, 0.0025, 0.0025])
        params = {
            "R": np.ones(5),
            "g": np.ones(5) * 0.02
        }
        dxdt = replicator_rhs(0.0, x, params)
        # Dominant technology's derivative should be small
        assert np.abs(dxdt[0]) < 0.01

    def test_two_technology_system(self):
        """Model should work with different numbers of technologies."""
        x = np.array([0.6, 0.4])
        params = {
            "R": np.array([1.0, 1.2]),
            "g": np.array([0.02, 0.03])
        }
        dxdt = replicator_rhs(0.0, x, params)
        assert dxdt.shape == (2,)
        assert np.isclose(np.sum(dxdt), 0.0, atol=1e-10)
