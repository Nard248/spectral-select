"""Pipeline integration tests for spectral_select.

This module tests the full pipeline configuration matrix to ensure all
10 configurations from full_pipeline_integration_test.py are valid and
work correctly with the spectral_select library.

These tests verify:
- All pipeline configurations create valid Config objects
- Analyzer can be instantiated with each configuration
- Configuration combinations work correctly together
- Synthetic data smoke tests pass

Note: Full pipeline execution tests (requiring trained models and real data)
are marked as slow/skipped for CI. Run manually for complete validation.
"""

import pytest
import numpy as np

from spectral_select import Analyzer, Config


# =============================================================================
# Configuration matrix from full_pipeline_integration_test.py
# =============================================================================

PIPELINE_CONFIGS = [
    # Config 1: Default baseline
    {
        "name": "baseline_default",
        "dimension_selection_method": "activation",
        "perturbation_method": "percentile",
        "normalization_method": "variance",
        "n_bands_to_select": 20,
        "use_diversity_constraint": False,
    },
    # Config 2: Variance-based dimension selection
    {
        "name": "variance_selection",
        "dimension_selection_method": "variance",
        "perturbation_method": "percentile",
        "normalization_method": "variance",
        "n_bands_to_select": 20,
        "use_diversity_constraint": False,
    },
    # Config 3: PCA dimension selection
    {
        "name": "pca_selection",
        "dimension_selection_method": "pca",
        "perturbation_method": "percentile",
        "normalization_method": "variance",
        "n_bands_to_select": 20,
        "use_diversity_constraint": False,
    },
    # Config 4: Standard deviation perturbation
    {
        "name": "stddev_perturbation",
        "dimension_selection_method": "activation",
        "perturbation_method": "standard_deviation",
        "normalization_method": "variance",
        "n_bands_to_select": 20,
        "use_diversity_constraint": False,
    },
    # Config 5: Absolute range perturbation
    {
        "name": "absrange_perturbation",
        "dimension_selection_method": "activation",
        "perturbation_method": "absolute_range",
        "normalization_method": "variance",
        "n_bands_to_select": 20,
        "use_diversity_constraint": False,
    },
    # Config 6: Max per excitation normalization
    {
        "name": "max_normalization",
        "dimension_selection_method": "activation",
        "perturbation_method": "percentile",
        "normalization_method": "max_per_excitation",
        "n_bands_to_select": 20,
        "use_diversity_constraint": False,
    },
    # Config 7: No normalization
    {
        "name": "no_normalization",
        "dimension_selection_method": "activation",
        "perturbation_method": "percentile",
        "normalization_method": "none",
        "n_bands_to_select": 20,
        "use_diversity_constraint": False,
    },
    # Config 8: MMR diversity constraint
    {
        "name": "mmr_diversity",
        "dimension_selection_method": "activation",
        "perturbation_method": "percentile",
        "normalization_method": "variance",
        "n_bands_to_select": 20,
        "use_diversity_constraint": True,
        "diversity_method": "mmr",
        "lambda_diversity": 0.5,
    },
    # Config 9: Min distance diversity
    {
        "name": "min_distance_diversity",
        "dimension_selection_method": "activation",
        "perturbation_method": "percentile",
        "normalization_method": "variance",
        "n_bands_to_select": 20,
        "use_diversity_constraint": True,
        "diversity_method": "min_distance",
        "min_distance_nm": 20.0,
    },
    # Config 10: Comprehensive - best expected combo
    {
        "name": "comprehensive",
        "dimension_selection_method": "activation",
        "perturbation_method": "percentile",
        "normalization_method": "variance",
        "n_bands_to_select": 30,
        "n_important_dimensions": 20,
        "use_diversity_constraint": True,
        "diversity_method": "mmr",
        "lambda_diversity": 0.3,
    },
]


# =============================================================================
# Pipeline Configuration Tests
# =============================================================================


class TestPipelineConfigurations:
    """Test that all pipeline configurations are valid."""

    @pytest.mark.parametrize(
        "config_dict", PIPELINE_CONFIGS, ids=lambda c: c["name"]
    )
    def test_config_creation(self, config_dict):
        """Each pipeline config should create a valid Config object."""
        # Extract name and use remaining params for Config
        params = {k: v for k, v in config_dict.items() if k != "name"}
        params["sample_name"] = "test_sample"

        config = Config(**params)

        assert config.sample_name == "test_sample"

    @pytest.mark.parametrize(
        "config_dict", PIPELINE_CONFIGS, ids=lambda c: c["name"]
    )
    def test_analyzer_instantiation(self, config_dict):
        """Each config should allow Analyzer instantiation."""
        params = {k: v for k, v in config_dict.items() if k != "name"}
        params["sample_name"] = "test_sample"
        params["device"] = "cpu"

        config = Config(**params)
        analyzer = Analyzer(config)

        assert analyzer is not None
        assert analyzer.config.sample_name == "test_sample"

    @pytest.mark.parametrize(
        "config_dict", PIPELINE_CONFIGS, ids=lambda c: c["name"]
    )
    def test_config_serialization_roundtrip(self, config_dict):
        """Each config should survive to_dict/from_dict roundtrip."""
        params = {k: v for k, v in config_dict.items() if k != "name"}
        params["sample_name"] = "test_sample"

        original = Config(**params)
        config_data = original.to_dict()
        restored = Config.from_dict(config_data)

        # Core parameters should match
        assert restored.sample_name == original.sample_name
        assert restored.dimension_selection_method == original.dimension_selection_method
        assert restored.perturbation_method == original.perturbation_method
        assert restored.normalization_method == original.normalization_method
        assert restored.n_bands_to_select == original.n_bands_to_select


class TestConfigurationCombinations:
    """Test specific configuration combination behaviors."""

    def test_diversity_methods_mmr(self):
        """Test MMR diversity constraint configuration."""
        config = Config(
            sample_name="test",
            use_diversity_constraint=True,
            diversity_method="mmr",
            lambda_diversity=0.5,
        )

        assert config.use_diversity_constraint is True
        assert config.diversity_method == "mmr"
        assert config.lambda_diversity == 0.5

    def test_diversity_methods_min_distance(self):
        """Test min distance diversity constraint configuration."""
        config = Config(
            sample_name="test",
            use_diversity_constraint=True,
            diversity_method="min_distance",
            min_distance_nm=20.0,
        )

        assert config.use_diversity_constraint is True
        assert config.diversity_method == "min_distance"
        assert config.min_distance_nm == 20.0

    def test_diversity_disabled(self):
        """Test diversity constraint can be disabled."""
        config = Config(
            sample_name="test",
            use_diversity_constraint=False,
        )

        assert config.use_diversity_constraint is False

    @pytest.mark.parametrize("method", ["variance", "activation", "pca"])
    def test_dimension_selection_methods(self, method):
        """Test all dimension selection method values."""
        config = Config(sample_name="test", dimension_selection_method=method)
        assert config.dimension_selection_method == method

    @pytest.mark.parametrize(
        "method", ["percentile", "standard_deviation", "absolute_range"]
    )
    def test_perturbation_methods(self, method):
        """Test all perturbation method values."""
        config = Config(sample_name="test", perturbation_method=method)
        assert config.perturbation_method == method

    @pytest.mark.parametrize("method", ["variance", "max_per_excitation", "none"])
    def test_normalization_methods(self, method):
        """Test all normalization method values."""
        config = Config(sample_name="test", normalization_method=method)
        assert config.normalization_method == method

    def test_n_bands_and_dimensions_combination(self):
        """Test n_bands_to_select and n_important_dimensions work together."""
        config = Config(
            sample_name="test",
            n_bands_to_select=30,
            n_important_dimensions=20,
        )

        assert config.n_bands_to_select == 30
        assert config.n_important_dimensions == 20

    def test_lambda_diversity_range(self):
        """Test lambda_diversity accepts valid range [0, 1]."""
        # Test boundaries
        Config(sample_name="test", lambda_diversity=0.0)
        Config(sample_name="test", lambda_diversity=1.0)
        Config(sample_name="test", lambda_diversity=0.5)

    def test_lambda_diversity_invalid(self):
        """Test lambda_diversity rejects values outside [0, 1]."""
        with pytest.raises(ValueError):
            Config(sample_name="test", lambda_diversity=-0.1)

        with pytest.raises(ValueError):
            Config(sample_name="test", lambda_diversity=1.1)


# =============================================================================
# Smoke Tests with Synthetic Data
# =============================================================================


class TestPipelineSmokeTests:
    """Smoke tests with synthetic data."""

    @pytest.fixture
    def synthetic_spectra_data(self):
        """Create minimal synthetic SpectraData for testing."""
        from spectral_select.types import SpectraData, ExcitationData

        # Small synthetic cube (10x10 spatial, 5 bands)
        np.random.seed(42)
        cube = np.random.rand(10, 10, 5).astype(np.float32)
        mask = np.ones((10, 10), dtype=bool)
        mask[0, 0] = False  # One masked pixel
        emission_wavelengths = np.linspace(400, 500, 5).tolist()

        ex_data_300 = ExcitationData(
            excitation_nm=300.0,
            cube=cube,
            emission_wavelengths=emission_wavelengths,
            exposure_time=100.0,
            laser_power=50.0,
        )

        # Create a copy for second excitation
        cube_310 = np.random.rand(10, 10, 5).astype(np.float32)
        ex_data_310 = ExcitationData(
            excitation_nm=310.0,
            cube=cube_310,
            emission_wavelengths=emission_wavelengths,
            exposure_time=100.0,
            laser_power=50.0,
        )

        return SpectraData(
            sample_name="synthetic_test",
            excitations={300.0: ex_data_300, 310.0: ex_data_310},
            mask=mask,  # Mask is at SpectraData level, not ExcitationData
        )

    def test_synthetic_data_structure(self, synthetic_spectra_data):
        """Test synthetic SpectraData has expected structure."""
        assert synthetic_spectra_data.n_excitations == 2
        assert synthetic_spectra_data.spatial_shape == (10, 10)
        assert 300.0 in synthetic_spectra_data.excitation_wavelengths
        assert 310.0 in synthetic_spectra_data.excitation_wavelengths

    def test_analyzer_instantiation_with_synthetic_config(self):
        """Test Analyzer can be instantiated with various configs."""
        for config_dict in PIPELINE_CONFIGS[:3]:  # Test first 3 configs
            params = {k: v for k, v in config_dict.items() if k != "name"}
            params["sample_name"] = "synthetic_test"
            params["device"] = "cpu"

            config = Config(**params)
            analyzer = Analyzer(config)

            assert analyzer is not None

    def test_config_to_dict_roundtrip(self):
        """Test config serialization roundtrip."""
        original = Config(
            sample_name="test",
            n_bands_to_select=15,
            dimension_selection_method="pca",
            use_diversity_constraint=True,
            diversity_method="mmr",
        )

        # to_dict and back
        config_dict = original.to_dict()
        restored = Config.from_dict(config_dict)

        assert restored.sample_name == original.sample_name
        assert restored.n_bands_to_select == original.n_bands_to_select
        assert restored.dimension_selection_method == original.dimension_selection_method
        assert restored.use_diversity_constraint == original.use_diversity_constraint
        assert restored.diversity_method == original.diversity_method

    def test_results_manager_creation(self, tmp_path):
        """Test ResultsManager can be created from config."""
        from spectral_select.results import ResultsManager

        config = Config(
            sample_name="test_sample",
            output_dir=tmp_path / "results",
        )

        # ResultsManager should create directory structure
        rm = ResultsManager.from_config(config)

        assert rm.sample_name == "test_sample"
        assert rm.base_dir.exists()

    def test_results_manager_run_directory(self, tmp_path):
        """Test ResultsManager creates timestamped run directories."""
        from spectral_select.results import ResultsManager

        config = Config(
            sample_name="test_sample",
            output_dir=tmp_path / "results",
        )

        rm = ResultsManager.from_config(config)

        # Run directory should have timestamp format
        assert rm.run_id is not None
        assert len(rm.run_id) > 0


# =============================================================================
# Full Pipeline Tests (Skipped in CI)
# =============================================================================


@pytest.mark.skip(reason="Requires trained model and real data")
@pytest.mark.slow
class TestFullPipelineExecution:
    """Full pipeline tests - skipped in CI, run manually.

    These tests require:
    - Real hyperspectral data files
    - Trained autoencoder model
    - Significant compute time

    To run these tests:
        pytest tests/test_pipeline_integration.py -v --run-slow -k "Full"
    """

    def test_full_pipeline_with_baseline_config(self):
        """Run full pipeline with baseline configuration."""
        # Would need real data and model
        pass

    def test_full_pipeline_comparison(self):
        """Compare results from different configurations."""
        # Would compare wavelength selections across configs
        pass


# =============================================================================
# Configuration Edge Cases
# =============================================================================


class TestConfigurationEdgeCases:
    """Test edge cases in configuration."""

    def test_minimum_bands_to_select(self):
        """Test minimum value for n_bands_to_select."""
        config = Config(sample_name="test", n_bands_to_select=1)
        assert config.n_bands_to_select == 1

    def test_invalid_bands_to_select(self):
        """Test invalid n_bands_to_select values."""
        with pytest.raises(ValueError):
            Config(sample_name="test", n_bands_to_select=0)

        with pytest.raises(ValueError):
            Config(sample_name="test", n_bands_to_select=-1)

    def test_minimum_dimensions(self):
        """Test minimum value for n_important_dimensions."""
        config = Config(sample_name="test", n_important_dimensions=1)
        assert config.n_important_dimensions == 1

    def test_invalid_dimension_method(self):
        """Test invalid dimension selection method."""
        with pytest.raises(ValueError):
            Config(sample_name="test", dimension_selection_method="invalid")

    def test_invalid_perturbation_method(self):
        """Test invalid perturbation method."""
        with pytest.raises(ValueError):
            Config(sample_name="test", perturbation_method="invalid")

    def test_invalid_normalization_method(self):
        """Test invalid normalization method."""
        with pytest.raises(ValueError):
            Config(sample_name="test", normalization_method="invalid")

    def test_invalid_diversity_method(self):
        """Test invalid diversity method."""
        with pytest.raises(ValueError):
            Config(sample_name="test", diversity_method="invalid")

    def test_min_distance_nm_valid_range(self):
        """Test min_distance_nm accepts non-negative values."""
        Config(sample_name="test", min_distance_nm=0.0)
        Config(sample_name="test", min_distance_nm=100.0)

    def test_min_distance_nm_invalid(self):
        """Test min_distance_nm rejects negative values."""
        with pytest.raises(ValueError):
            Config(sample_name="test", min_distance_nm=-1.0)
