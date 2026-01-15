"""
Integration tests for end-to-end simulation workflow.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from simulations import baseline_scenario, renewable_policy_scenario, nuclear_phaseout_scenario
from main import main, save_results_to_csv, run_scenario


class TestEndToEndSimulation:
    """Integration tests for complete simulation runs."""

    def test_baseline_runs_to_completion(self):
        """Baseline scenario should run 100 years without error."""
        t, x = baseline_scenario()

        assert len(t) > 900  # Should have ~1001 timesteps for dt=0.1
        assert x.shape[1] == 5
        assert np.all(x >= 0)
        assert np.allclose(np.sum(x, axis=1), 1.0, atol=1e-10)

    def test_renewable_runs_to_completion(self):
        """Renewable scenario should run 100 years without error."""
        t, x = renewable_policy_scenario()

        assert len(t) > 900
        assert x.shape[1] == 5
        assert np.all(x >= 0)
        assert np.allclose(np.sum(x, axis=1), 1.0, atol=1e-10)

    def test_nuclear_phaseout_runs_to_completion(self):
        """Nuclear phase-out scenario should run 100 years without error."""
        t, x = nuclear_phaseout_scenario()

        assert len(t) > 900
        assert x.shape[1] == 5
        assert np.all(x >= 0)
        assert np.allclose(np.sum(x, axis=1), 1.0, atol=1e-10)

    def test_all_scenarios_produce_different_results(self):
        """Different scenarios should produce different final states."""
        t1, x1 = baseline_scenario()
        t2, x2 = renewable_policy_scenario()
        t3, x3 = nuclear_phaseout_scenario()

        # Final states should be different
        assert not np.allclose(x1[-1], x2[-1])
        assert not np.allclose(x1[-1], x3[-1])
        assert not np.allclose(x2[-1], x3[-1])


class TestCsvExport:
    """Tests for CSV export functionality."""

    def test_csv_export_creates_file(self):
        """CSV export should create a valid file."""
        t, x = baseline_scenario(t_end=5.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_output.csv"
            save_results_to_csv(t, x, filepath)

            assert filepath.exists()

    def test_csv_export_content(self):
        """CSV export should have correct content."""
        t, x = baseline_scenario(t_end=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_output.csv"
            save_results_to_csv(t, x, filepath)

            with open(filepath, "r") as f:
                lines = f.readlines()

            # Should have header + data rows
            assert len(lines) > 1

            # Header should have correct columns
            header = lines[0].strip().split(",")
            assert header[0] == "time"
            assert len(header) == 6  # time + 5 energy sources


class TestCliInterface:
    """Tests for command-line interface."""

    def test_main_help(self):
        """Help should work without error."""
        # argparse exits with code 0 on --help
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_main_no_show(self):
        """Should run without showing plots."""
        # This test just verifies no error, doesn't verify plot suppression
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = main([
                "--scenario", "baseline",
                "--t-end", "5",
                "--no-show",
                "--output-dir", tmpdir,
                "--save-csv"
            ])

            assert exit_code == 0
            assert (Path(tmpdir) / "baseline.csv").exists()

    def test_main_single_scenario(self):
        """Should run single scenario."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = main([
                "--scenario", "renewable",
                "--t-end", "5",
                "--no-show",
                "--output-dir", tmpdir,
                "--save-csv"
            ])

            assert exit_code == 0
            # Only renewable should be generated
            assert (Path(tmpdir) / "renewable.csv").exists()
            assert not (Path(tmpdir) / "baseline.csv").exists()


class TestNumericalStability:
    """Tests for numerical stability over long simulations."""

    def test_long_simulation_stable(self):
        """Simulation should remain stable over 100 years."""
        t, x = baseline_scenario(t_end=100.0, dt=0.1)

        # No NaN or Inf values
        assert not np.any(np.isnan(x))
        assert not np.any(np.isinf(x))

        # All values in valid range
        assert np.all(x >= 0)
        assert np.all(x <= 1)

    def test_small_timestep_accuracy(self):
        """Smaller timestep should give similar results."""
        t1, x1 = baseline_scenario(t_end=10.0, dt=0.1)
        t2, x2 = baseline_scenario(t_end=10.0, dt=0.01)

        # Find common time points for comparison
        # Compare final states - should be similar
        assert np.allclose(x1[-1], x2[-1], rtol=0.01)  # 1% tolerance

    def test_normalization_after_many_steps(self):
        """Normalization should be exact after many steps."""
        t, x = baseline_scenario(t_end=100.0, dt=0.1)

        # Check last 100 timesteps
        final_sums = np.sum(x[-100:], axis=1)
        assert np.allclose(final_sums, 1.0, atol=1e-14)
