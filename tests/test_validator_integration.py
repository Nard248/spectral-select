"""Integration tests for Validator class.

Tests cover:
- Validator initialization
- fit() with cluster labels and ground truth
- score() returns ARI in valid range
- generate_report() returns markdown string
- to_json/from_json round-trip
- ground_truth property access
- load_ground_truth_from_png utility function

Uses synthetic data fixtures to avoid dependency on large data files.
"""

import json
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from PIL import Image

from spectral_select.validation import Validator, load_ground_truth_from_png
from spectral_select.types import GroundTruth


# =============================================================================
# Fixtures for Validator Tests
# =============================================================================


@pytest.fixture
def synthetic_ground_truth() -> np.ndarray:
    """Create synthetic ground truth labels (10x10 with 3 classes).

    Classes:
    - Class 0: top-left quadrant
    - Class 1: top-right quadrant
    - Class 2: bottom half
    - Background (-1): corners
    """
    labels = np.full((10, 10), -1, dtype=np.int32)

    # Class 0: top-left quadrant (rows 1-4, cols 1-4)
    labels[1:5, 1:5] = 0

    # Class 1: top-right quadrant (rows 1-4, cols 5-9)
    labels[1:5, 5:9] = 1

    # Class 2: bottom half (rows 5-9, cols 1-9)
    labels[5:9, 1:9] = 2

    return labels


@pytest.fixture
def synthetic_predictions(synthetic_ground_truth: np.ndarray) -> np.ndarray:
    """Create synthetic predictions with some errors for realistic ARI.

    Mostly correct but with ~15% errors to get an ARI around 0.6-0.8.
    """
    np.random.seed(42)
    predictions = synthetic_ground_truth.copy()

    # Add some noise - flip ~15% of valid predictions
    valid_mask = predictions >= 0
    valid_indices = np.where(valid_mask)

    n_valid = len(valid_indices[0])
    n_flip = int(n_valid * 0.15)

    flip_indices = np.random.choice(n_valid, n_flip, replace=False)

    for idx in flip_indices:
        y, x = valid_indices[0][idx], valid_indices[1][idx]
        current = predictions[y, x]
        # Flip to a different class (0, 1, or 2)
        new_class = (current + np.random.randint(1, 3)) % 3
        predictions[y, x] = new_class

    return predictions


@pytest.fixture
def synthetic_test_png(tmp_path: Path) -> Tuple[Path, np.ndarray]:
    """Create a minimal test PNG with known class colors.

    Returns tuple of (png_path, expected_labels).
    """
    # Create 5x5 image with distinct colors
    height, width = 5, 5
    img_array = np.zeros((height, width, 4), dtype=np.uint8)

    # Background: dark gray (24, 24, 24)
    img_array[0, :] = [24, 24, 24, 255]  # Top row is background
    img_array[:, 0] = [24, 24, 24, 255]  # Left column is background

    # Class 0: Red (255, 0, 0)
    img_array[1:3, 1:3] = [255, 0, 0, 255]  # 2x2 region

    # Class 1: Green (0, 255, 0)
    img_array[1:3, 3:5] = [0, 255, 0, 255]  # 2x2 region

    # Class 2: Blue (0, 0, 255)
    img_array[3:5, 1:5] = [0, 0, 255, 255]  # 2x4 region

    # Save as PNG
    img = Image.fromarray(img_array, mode='RGBA')
    png_path = tmp_path / "test_ground_truth.png"
    img.save(png_path)

    # Expected labels (before load processing)
    expected = np.array([
        [-1, -1, -1, -1, -1],
        [-1,  0,  0,  1,  1],
        [-1,  0,  0,  1,  1],
        [-1,  2,  2,  2,  2],
        [-1,  2,  2,  2,  2],
    ], dtype=np.int32)

    return png_path, expected


# =============================================================================
# Validator Initialization Tests
# =============================================================================


class TestValidatorInitialization:
    """Tests for Validator initialization."""

    def test_validator_initialization(self):
        """Validator() creates valid instance."""
        validator = Validator()

        assert validator is not None
        assert validator.is_fitted is False
        assert validator.ground_truth is None

    def test_validator_initial_state(self):
        """Validator starts in unfitted state."""
        validator = Validator()

        assert validator.is_fitted is False

        with pytest.raises(RuntimeError, match="Validator not fitted"):
            _ = validator.metrics


# =============================================================================
# Validator fit() Tests
# =============================================================================


class TestValidatorFit:
    """Tests for Validator fit method."""

    def test_validator_fit_with_ground_truth(
        self,
        synthetic_predictions: np.ndarray,
        synthetic_ground_truth: np.ndarray,
    ):
        """fit(predictions, ground_truth) stores data and computes metrics."""
        validator = Validator()

        result = validator.fit(synthetic_predictions, synthetic_ground_truth)

        # Should return self for chaining
        assert result is validator
        assert validator.is_fitted is True

    def test_validator_fit_returns_self(
        self,
        synthetic_predictions: np.ndarray,
        synthetic_ground_truth: np.ndarray,
    ):
        """fit() returns self for method chaining."""
        validator = Validator()

        result = validator.fit(synthetic_predictions, synthetic_ground_truth)

        assert result is validator

    def test_validator_fit_with_ground_truth_object(
        self,
        synthetic_predictions: np.ndarray,
        synthetic_ground_truth: np.ndarray,
    ):
        """fit() accepts GroundTruth object as well as numpy array."""
        gt_obj = GroundTruth.from_array(synthetic_ground_truth)
        validator = Validator()

        validator.fit(synthetic_predictions, gt_obj)

        assert validator.is_fitted is True

    def test_validator_fit_with_valid_mask(
        self,
        synthetic_predictions: np.ndarray,
        synthetic_ground_truth: np.ndarray,
    ):
        """fit() respects valid_mask parameter."""
        validator = Validator()

        # Create a mask that excludes part of the data
        valid_mask = np.ones_like(synthetic_ground_truth, dtype=bool)
        valid_mask[:5, :] = False  # Exclude top half

        validator.fit(synthetic_predictions, synthetic_ground_truth, valid_mask=valid_mask)

        assert validator.is_fitted is True


# =============================================================================
# Validator score() Tests
# =============================================================================


class TestValidatorScore:
    """Tests for Validator score method."""

    def test_validator_score_returns_ari(
        self,
        synthetic_predictions: np.ndarray,
        synthetic_ground_truth: np.ndarray,
    ):
        """score() returns float in [-1, 1] range."""
        validator = Validator()
        validator.fit(synthetic_predictions, synthetic_ground_truth)

        ari = validator.score()

        assert isinstance(ari, float)
        assert -1.0 <= ari <= 1.0

    def test_validator_score_positive_for_good_predictions(
        self,
        synthetic_ground_truth: np.ndarray,
    ):
        """score() returns positive ARI for mostly correct predictions."""
        validator = Validator()

        # Use ground truth as predictions (perfect match)
        validator.fit(synthetic_ground_truth, synthetic_ground_truth)
        ari = validator.score()

        # Perfect predictions should give ARI = 1.0
        assert ari == 1.0

    def test_validator_score_before_fit_raises(self):
        """score() raises RuntimeError before fit()."""
        validator = Validator()

        with pytest.raises(RuntimeError, match="Validator not fitted"):
            validator.score()


# =============================================================================
# Validator generate_report() Tests
# =============================================================================


class TestValidatorGenerateReport:
    """Tests for Validator generate_report method."""

    def test_validator_generate_report(
        self,
        synthetic_predictions: np.ndarray,
        synthetic_ground_truth: np.ndarray,
    ):
        """generate_report() returns markdown string."""
        validator = Validator()
        validator.fit(synthetic_predictions, synthetic_ground_truth)

        report = validator.generate_report()

        assert isinstance(report, str)
        assert "# Clustering Validation Report" in report
        assert "Summary Statistics" in report
        assert "Adjusted Rand Index" in report

    def test_validator_generate_report_contains_metrics(
        self,
        synthetic_predictions: np.ndarray,
        synthetic_ground_truth: np.ndarray,
    ):
        """Report contains all expected metric sections."""
        validator = Validator()
        validator.fit(synthetic_predictions, synthetic_ground_truth)

        report = validator.generate_report()

        # Check for expected sections
        assert "Purity" in report
        assert "Normalized Mutual Info" in report
        assert "Cluster-to-Ground-Truth Mapping" in report
        assert "Per-Class Metrics" in report

    def test_validator_generate_report_saves_to_file(
        self,
        synthetic_predictions: np.ndarray,
        synthetic_ground_truth: np.ndarray,
        tmp_path: Path,
    ):
        """generate_report() saves to file when output_path provided."""
        validator = Validator()
        validator.fit(synthetic_predictions, synthetic_ground_truth)

        output_path = tmp_path / "report.md"
        report = validator.generate_report(output_path=output_path)

        assert output_path.exists()
        assert output_path.read_text() == report


# =============================================================================
# Validator JSON Round-trip Tests
# =============================================================================


class TestValidatorJsonRoundtrip:
    """Tests for Validator to_json/from_json serialization."""

    def test_validator_to_json_roundtrip(
        self,
        synthetic_predictions: np.ndarray,
        synthetic_ground_truth: np.ndarray,
        tmp_path: Path,
    ):
        """to_json/from_json preserves metrics."""
        validator = Validator()
        validator.fit(synthetic_predictions, synthetic_ground_truth)

        # Save metrics to JSON
        json_path = tmp_path / "metrics.json"
        validator.to_json(json_path)

        assert json_path.exists()

        # Load and verify
        with open(json_path, "r") as f:
            data = json.load(f)

        # Check key metrics are preserved
        assert "adjusted_rand_score" in data
        assert "purity" in data
        assert "normalized_mutual_info" in data
        assert "confusion_matrix" in data
        assert "cluster_to_gt_mapping" in data

        # Values should match original
        original_metrics = validator.metrics
        assert abs(data["adjusted_rand_score"] - original_metrics.adjusted_rand_score) < 1e-6
        assert abs(data["purity"] - original_metrics.purity) < 1e-6


# =============================================================================
# Validator ground_truth Property Tests
# =============================================================================


class TestValidatorGroundTruthProperty:
    """Tests for Validator ground_truth property."""

    def test_validator_ground_truth_property(
        self,
        synthetic_predictions: np.ndarray,
        synthetic_ground_truth: np.ndarray,
    ):
        """ground_truth property returns stored labels."""
        validator = Validator()
        validator.fit(synthetic_predictions, synthetic_ground_truth)

        gt = validator.ground_truth

        assert gt is not None
        assert isinstance(gt, np.ndarray)
        # Should be flattened
        assert gt.ndim == 1
        assert len(gt) == synthetic_ground_truth.size

    def test_validator_ground_truth_before_fit(self):
        """ground_truth returns None before fit()."""
        validator = Validator()

        assert validator.ground_truth is None


# =============================================================================
# load_ground_truth_from_png Tests
# =============================================================================


class TestLoadGroundTruthFromPng:
    """Tests for load_ground_truth_from_png utility function."""

    def test_load_ground_truth_from_png(self, synthetic_test_png: Tuple[Path, np.ndarray]):
        """Function loads PNG and extracts labels."""
        png_path, expected_labels = synthetic_test_png

        gt = load_ground_truth_from_png(png_path)

        assert isinstance(gt, GroundTruth)
        assert gt.labels.shape == (5, 5)
        # Should have found 3 classes (excluding background)
        assert gt.n_classes == 3

    def test_load_ground_truth_from_png_background_handling(
        self,
        synthetic_test_png: Tuple[Path, np.ndarray],
    ):
        """Background pixels are labeled as -1."""
        png_path, expected_labels = synthetic_test_png

        gt = load_ground_truth_from_png(png_path)

        # Background pixels (dark gray) should be -1
        assert gt.labels[0, 0] == -1  # Top-left corner is background
        assert gt.labels[0, 2] == -1  # Top row is all background

    def test_load_ground_truth_from_png_file_not_found(self, tmp_path: Path):
        """Raises FileNotFoundError for missing file."""
        missing_path = tmp_path / "nonexistent.png"

        with pytest.raises(FileNotFoundError, match="PNG file not found"):
            load_ground_truth_from_png(missing_path)

    def test_load_ground_truth_from_png_with_class_colors(
        self,
        synthetic_test_png: Tuple[Path, np.ndarray],
    ):
        """Function accepts custom class_colors mapping."""
        png_path, _ = synthetic_test_png

        class_colors = {
            "Class_Red": (255, 0, 0),
            "Class_Green": (0, 255, 0),
            "Class_Blue": (0, 0, 255),
        }

        gt = load_ground_truth_from_png(
            png_path,
            class_colors=class_colors,
            color_tolerance=10,
        )

        assert isinstance(gt, GroundTruth)
        assert gt.class_names == list(class_colors.keys())


# =============================================================================
# Validator compare() Tests
# =============================================================================


class TestValidatorCompare:
    """Tests for Validator compare method (comparing multiple clustering results)."""

    def test_validator_compare_multiple_methods(
        self,
        synthetic_ground_truth: np.ndarray,
    ):
        """compare() evaluates multiple clustering results."""
        np.random.seed(42)

        # Create different prediction sets
        predictions_good = synthetic_ground_truth.copy()  # Perfect
        predictions_random = np.random.randint(0, 3, synthetic_ground_truth.shape)
        predictions_random[synthetic_ground_truth == -1] = -1  # Keep background

        results_dict = {
            "Perfect": predictions_good,
            "Random": predictions_random,
        }

        validator = Validator()
        comparison_df = validator.compare(results_dict, synthetic_ground_truth)

        assert len(comparison_df) == 2
        assert "Method" in comparison_df.columns
        assert "Purity" in comparison_df.columns
        assert "ARI" in comparison_df.columns

        # Perfect should have higher purity
        perfect_row = comparison_df[comparison_df["Method"] == "Perfect"]
        random_row = comparison_df[comparison_df["Method"] == "Random"]
        assert perfect_row["Purity"].values[0] >= random_row["Purity"].values[0]


# =============================================================================
# Validator Metrics Access Tests
# =============================================================================


class TestValidatorMetricsAccess:
    """Tests for accessing computed metrics through Validator."""

    def test_validator_get_metrics_dict(
        self,
        synthetic_predictions: np.ndarray,
        synthetic_ground_truth: np.ndarray,
    ):
        """get_metrics_dict() returns flat dictionary."""
        validator = Validator()
        validator.fit(synthetic_predictions, synthetic_ground_truth)

        metrics_dict = validator.get_metrics_dict()

        assert isinstance(metrics_dict, dict)
        assert "Purity" in metrics_dict
        assert "ARI" in metrics_dict
        assert "NMI" in metrics_dict
        assert "N_Clusters" in metrics_dict

    def test_validator_metrics_property(
        self,
        synthetic_predictions: np.ndarray,
        synthetic_ground_truth: np.ndarray,
    ):
        """metrics property returns ValidationMetrics object."""
        validator = Validator()
        validator.fit(synthetic_predictions, synthetic_ground_truth)

        metrics = validator.metrics

        assert hasattr(metrics, "adjusted_rand_score")
        assert hasattr(metrics, "purity")
        assert hasattr(metrics, "normalized_mutual_info")
        assert hasattr(metrics, "confusion_matrix")
        assert hasattr(metrics, "cluster_to_gt_mapping")
