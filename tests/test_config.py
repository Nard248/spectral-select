"""Unit tests for Config class.

Tests cover:
- Default values
- Custom value overrides
- Validation of string options
- YAML and JSON serialization round-trips
- Warning on unknown keys
"""

import warnings
from pathlib import Path

import pytest

from spectral_select import Config


class TestConfigDefaults:
    """Tests for Config default values."""

    def test_config_defaults(self):
        """Verify default values are set correctly."""
        config = Config()

        # Data configuration defaults
        assert config.sample_name == "sample"
        assert config.data_path is None
        assert config.mask_path is None
        # model_path is auto-resolved from sample_name when not explicitly set
        assert config.model_path == Path("model_output") / "sample" / "model.pth"
        assert config.output_dir is None

        # Analysis parameters defaults
        assert config.dimension_selection_method == "activation"
        assert config.n_important_dimensions == 15
        assert config.perturbation_method == "percentile"
        assert config.perturbation_magnitudes == [10, 20, 30]
        assert config.perturbation_directions == ["bidirectional"]
        assert config.normalization_method == "variance"

        # Selection parameters defaults
        assert config.n_bands_to_select == 30
        assert config.n_layers_to_extract == 10

        # Diversity constraint defaults
        assert config.use_diversity_constraint is False
        assert config.diversity_method == "mmr"
        assert config.lambda_diversity == 0.5
        assert config.min_distance_nm == 15.0

        # Output configuration defaults
        assert config.save_tiff_layers is True
        assert config.save_visualizations is True
        assert config.save_detailed_results is True

        # Technical parameters defaults
        assert config.device == "cuda"
        assert config.n_baseline_patches == 50
        assert config.patch_size == 32
        assert config.patch_stride == 16
        assert config.random_seed == 42

        # Pluggable components defaults
        assert config.classifier == "knn"
        assert config.clustering == "kmeans"
        assert config.autoencoder_architecture == "standard"
        assert config.wavelength_ranker == "perturbation"


class TestConfigCustomValues:
    """Tests for Config with custom values."""

    def test_config_custom_values(self):
        """Verify custom values override defaults."""
        config = Config(
            sample_name="Lichens_2",
            dimension_selection_method="variance",
            n_important_dimensions=20,
            perturbation_method="standard_deviation",
            n_bands_to_select=50,
            device="cpu",
            random_seed=123,
        )

        assert config.sample_name == "Lichens_2"
        assert config.dimension_selection_method == "variance"
        assert config.n_important_dimensions == 20
        assert config.perturbation_method == "standard_deviation"
        assert config.n_bands_to_select == 50
        assert config.device == "cpu"
        assert config.random_seed == 123

    def test_config_path_conversion(self, tmp_path: Path):
        """Verify string paths are converted to Path objects."""
        config = Config(
            data_path=str(tmp_path / "data.pkl"),
            mask_path=str(tmp_path / "mask.tif"),
            model_path=str(tmp_path / "model.pt"),
            output_dir=str(tmp_path / "output"),
        )

        assert isinstance(config.data_path, Path)
        assert isinstance(config.mask_path, Path)
        assert isinstance(config.model_path, Path)
        assert isinstance(config.output_dir, Path)


class TestConfigValidation:
    """Tests for Config validation."""

    def test_config_validation_dimension_method(self):
        """Invalid dimension_selection_method raises ValueError."""
        with pytest.raises(ValueError, match="dimension_selection_method must be one of"):
            Config(dimension_selection_method="invalid_method")

    def test_config_validation_perturbation_method(self):
        """Invalid perturbation_method raises ValueError."""
        with pytest.raises(ValueError, match="perturbation_method must be one of"):
            Config(perturbation_method="invalid_perturbation")

    def test_config_validation_normalization_method(self):
        """Invalid normalization_method raises ValueError."""
        with pytest.raises(ValueError, match="normalization_method must be one of"):
            Config(normalization_method="invalid_normalization")

    def test_config_validation_diversity_method(self):
        """Invalid diversity_method raises ValueError."""
        with pytest.raises(ValueError, match="diversity_method must be one of"):
            Config(diversity_method="invalid_diversity")

    def test_config_validation_device(self):
        """Invalid device raises ValueError."""
        with pytest.raises(ValueError, match="device must be one of"):
            Config(device="tpu")

    def test_config_validation_numeric_ranges(self):
        """Invalid numeric values raise ValueError."""
        with pytest.raises(ValueError, match="n_important_dimensions must be positive"):
            Config(n_important_dimensions=0)

        with pytest.raises(ValueError, match="n_bands_to_select must be positive"):
            Config(n_bands_to_select=-1)

        with pytest.raises(ValueError, match="lambda_diversity must be between 0 and 1"):
            Config(lambda_diversity=1.5)

        with pytest.raises(ValueError, match="perturbation_magnitudes cannot be empty"):
            Config(perturbation_magnitudes=[])


class TestConfigSerialization:
    """Tests for Config serialization (YAML/JSON)."""

    def test_config_yaml_roundtrip(self, tmp_path: Path):
        """to_yaml then from_yaml produces equivalent Config."""
        original = Config(
            sample_name="test_yaml",
            dimension_selection_method="pca",
            n_important_dimensions=25,
            device="mps",
        )

        yaml_path = tmp_path / "config.yaml"
        original.to_yaml(yaml_path)

        loaded = Config.from_yaml(yaml_path)

        assert loaded == original
        assert loaded.sample_name == "test_yaml"
        assert loaded.dimension_selection_method == "pca"
        assert loaded.n_important_dimensions == 25
        assert loaded.device == "mps"

    def test_config_json_roundtrip(self, tmp_path: Path):
        """to_json then from_json produces equivalent Config."""
        original = Config(
            sample_name="test_json",
            perturbation_method="absolute_range",
            n_bands_to_select=100,
            use_diversity_constraint=True,
        )

        json_path = tmp_path / "config.json"
        original.to_json(json_path)

        loaded = Config.from_json(json_path)

        assert loaded == original
        assert loaded.sample_name == "test_json"
        assert loaded.perturbation_method == "absolute_range"
        assert loaded.n_bands_to_select == 100
        assert loaded.use_diversity_constraint is True

    def test_config_to_dict(self):
        """Verify to_dict produces correct dictionary."""
        config = Config(sample_name="dict_test")
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["sample_name"] == "dict_test"
        assert "dimension_selection_method" in config_dict
        assert "device" in config_dict


class TestConfigUnknownKeys:
    """Tests for Config handling of unknown keys."""

    def test_config_unknown_keys_warning(self, tmp_path: Path):
        """from_yaml with unknown keys emits warning."""
        yaml_content = """
sample_name: test_unknown
unknown_key: some_value
another_unknown: 123
"""
        yaml_path = tmp_path / "config_unknown.yaml"
        yaml_path.write_text(yaml_content)

        with pytest.warns(UserWarning, match="Unknown configuration key 'unknown_key'"):
            Config.from_yaml(yaml_path)

    def test_config_from_dict_unknown_keys_warning(self):
        """from_dict with unknown keys emits warning."""
        data = {
            "sample_name": "test",
            "not_a_valid_key": "value",
        }

        with pytest.warns(UserWarning, match="Unknown configuration key 'not_a_valid_key'"):
            Config.from_dict(data)


class TestConfigEquality:
    """Tests for Config equality comparison."""

    def test_config_equality_same_values(self):
        """Two configs with same values are equal."""
        config1 = Config(sample_name="test", n_bands_to_select=50)
        config2 = Config(sample_name="test", n_bands_to_select=50)

        assert config1 == config2

    def test_config_equality_different_values(self):
        """Two configs with different values are not equal."""
        config1 = Config(sample_name="test1")
        config2 = Config(sample_name="test2")

        assert config1 != config2

    def test_config_equality_non_config(self):
        """Config compared to non-Config returns NotImplemented."""
        config = Config()

        assert config != "not a config"
        assert config != 123
        assert config != {"sample_name": "sample"}
