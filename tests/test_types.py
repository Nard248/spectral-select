"""Unit tests for data types.

Tests cover:
- SpectraData: creation, shape validation, to_dict
- WavelengthBand: creation, ordering/comparison
- WavelengthResult: creation, sequential ranks validation, from_bands
- ValidationMetrics: creation, JSON round-trip
"""

from pathlib import Path
from typing import List

import numpy as np
import pytest

from spectral_select import (
    ExcitationData,
    SpectraData,
    WavelengthBand,
)
from spectral_select.types import (
    AnalysisMetrics,
    GroundTruth,
    LoadingOptions,
    ValidationMetrics,
    WavelengthResult,
)


class TestSpectraDataCreation:
    """Tests for SpectraData creation."""

    def test_spectra_data_creation(self, synthetic_spectra_data: SpectraData):
        """Basic instantiation with valid data."""
        assert synthetic_spectra_data.sample_name == "synthetic_sample"
        assert synthetic_spectra_data.n_excitations == 3
        assert synthetic_spectra_data.spatial_shape == (10, 10)
        assert synthetic_spectra_data.mask is not None
        assert synthetic_spectra_data.excitation_wavelengths == [365.0, 405.0, 450.0]

    def test_spectra_data_empty_initialization(self):
        """Allow empty excitations on initialization."""
        data = SpectraData(excitations={})
        assert data.n_excitations == 0
        assert data.spatial_shape == (0, 0)

    def test_spectra_data_shape_validation(self):
        """Mismatched cube/mask shapes raise ValueError."""
        np.random.seed(42)
        cube = np.random.rand(10, 10, 5).astype(np.float32)
        emission_wls = [500.0, 510.0, 520.0, 530.0, 540.0]

        excitation_data = ExcitationData(
            excitation_nm=365.0,
            cube=cube,
            emission_wavelengths=emission_wls,
        )

        # Mask with wrong shape
        wrong_mask = np.ones((20, 20), dtype=bool)

        with pytest.raises(ValueError, match="mask shape.*doesn't match"):
            SpectraData(
                excitations={365.0: excitation_data},
                mask=wrong_mask,
            )

    def test_spectra_data_mismatched_excitation_shapes(self):
        """Mismatched excitation cube shapes raise ValueError."""
        np.random.seed(42)
        cube1 = np.random.rand(10, 10, 5).astype(np.float32)
        cube2 = np.random.rand(15, 15, 5).astype(np.float32)  # Different spatial size

        excitations = {
            365.0: ExcitationData(
                excitation_nm=365.0,
                cube=cube1,
                emission_wavelengths=[500.0, 510.0, 520.0, 530.0, 540.0],
            ),
            405.0: ExcitationData(
                excitation_nm=405.0,
                cube=cube2,
                emission_wavelengths=[500.0, 510.0, 520.0, 530.0, 540.0],
            ),
        }

        with pytest.raises(ValueError, match="has shape.*but expected"):
            SpectraData(excitations=excitations)

    def test_spectra_data_to_dict(self, synthetic_spectra_data: SpectraData):
        """Excludes large arrays, includes metadata."""
        result_dict = synthetic_spectra_data.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["sample_name"] == "synthetic_sample"
        assert result_dict["n_excitations"] == 3
        assert result_dict["spatial_shape"] == (10, 10)
        assert result_dict["has_mask"] is True
        assert "excitation_wavelengths" in result_dict
        assert "metadata" in result_dict
        # Should NOT contain large arrays
        assert "excitations" not in result_dict  # to_dict excludes full excitation data

    def test_spectra_data_get_excitation(self, synthetic_spectra_data: SpectraData):
        """Get data for specific excitation wavelength."""
        ex_data = synthetic_spectra_data.get_excitation(365.0)

        assert isinstance(ex_data, ExcitationData)
        assert ex_data.excitation_nm == 365.0

    def test_spectra_data_get_excitation_not_found(self, synthetic_spectra_data: SpectraData):
        """KeyError when excitation not found."""
        with pytest.raises(KeyError, match="Excitation 999.0nm not found"):
            synthetic_spectra_data.get_excitation(999.0)


class TestExcitationData:
    """Tests for ExcitationData."""

    def test_excitation_data_creation(self, synthetic_excitation_data: ExcitationData):
        """Basic instantiation."""
        assert synthetic_excitation_data.excitation_nm == 365.0
        assert synthetic_excitation_data.shape == (10, 10, 5)
        assert synthetic_excitation_data.n_bands == 5
        assert synthetic_excitation_data.height == 10
        assert synthetic_excitation_data.width == 10

    def test_excitation_data_invalid_cube_dimensions(self):
        """Non-3D cube raises ValueError."""
        cube = np.random.rand(10, 10).astype(np.float32)  # 2D, not 3D

        with pytest.raises(ValueError, match="cube must be 3D"):
            ExcitationData(
                excitation_nm=365.0,
                cube=cube,
                emission_wavelengths=[500.0],
            )

    def test_excitation_data_mismatched_bands(self):
        """Mismatched emission_wavelengths length raises ValueError."""
        cube = np.random.rand(10, 10, 5).astype(np.float32)

        with pytest.raises(ValueError, match="len\\(emission_wavelengths\\)"):
            ExcitationData(
                excitation_nm=365.0,
                cube=cube,
                emission_wavelengths=[500.0, 510.0],  # Only 2, but cube has 5 bands
            )


class TestWavelengthBandCreation:
    """Tests for WavelengthBand creation."""

    def test_wavelength_band_creation(self):
        """Basic instantiation."""
        band = WavelengthBand(
            rank=1,
            excitation_nm=365.0,
            emission_nm=500.0,
            emission_band_index=0,
            influence_score=0.95,
        )

        assert band.rank == 1
        assert band.excitation_nm == 365.0
        assert band.emission_nm == 500.0
        assert band.emission_band_index == 0
        assert band.influence_score == 0.95

    def test_wavelength_band_validation_rank(self):
        """Rank < 1 raises ValueError."""
        with pytest.raises(ValueError, match="rank must be >= 1"):
            WavelengthBand(
                rank=0,
                excitation_nm=365.0,
                emission_nm=500.0,
                emission_band_index=0,
                influence_score=0.5,
            )

    def test_wavelength_band_validation_wavelengths(self):
        """Non-positive wavelengths raise ValueError."""
        with pytest.raises(ValueError, match="excitation_nm must be positive"):
            WavelengthBand(
                rank=1,
                excitation_nm=0.0,
                emission_nm=500.0,
                emission_band_index=0,
                influence_score=0.5,
            )

        with pytest.raises(ValueError, match="emission_nm must be positive"):
            WavelengthBand(
                rank=1,
                excitation_nm=365.0,
                emission_nm=-100.0,
                emission_band_index=0,
                influence_score=0.5,
            )


class TestWavelengthBandOrdering:
    """Tests for WavelengthBand ordering and comparison."""

    def test_wavelength_band_ordering(self, synthetic_wavelength_bands: List[WavelengthBand]):
        """Comparison operators work correctly (via rank)."""
        band1 = synthetic_wavelength_bands[0]  # rank=1
        band2 = synthetic_wavelength_bands[1]  # rank=2

        # WavelengthBand doesn't define __lt__ etc., but we can compare by rank
        assert band1.rank < band2.rank
        assert band2.rank > band1.rank

        # Sort by rank
        bands_copy = list(synthetic_wavelength_bands)
        sorted_bands = sorted(bands_copy, key=lambda b: b.rank)
        assert sorted_bands[0].rank == 1
        assert sorted_bands[-1].rank == 5

    def test_wavelength_band_repr(self):
        """Repr shows readable format."""
        band = WavelengthBand(
            rank=1,
            excitation_nm=365.0,
            emission_nm=500.0,
            emission_band_index=0,
            influence_score=0.85,
        )

        repr_str = repr(band)
        assert "rank=1" in repr_str
        assert "ex=365.0nm" in repr_str
        assert "em=500.0nm" in repr_str
        assert "score=0.85" in repr_str

    def test_wavelength_band_to_dict(self):
        """to_dict produces serializable dict."""
        band = WavelengthBand(
            rank=1,
            excitation_nm=365.0,
            emission_nm=500.0,
            emission_band_index=0,
            influence_score=0.85,
        )

        band_dict = band.to_dict()
        assert band_dict["rank"] == 1
        assert band_dict["excitation_nm"] == 365.0
        assert band_dict["emission_nm"] == 500.0
        assert band_dict["emission_band_index"] == 0
        assert band_dict["influence_score"] == 0.85

    def test_wavelength_band_from_dict(self):
        """from_dict creates equivalent band."""
        original = WavelengthBand(
            rank=2,
            excitation_nm=405.0,
            emission_nm=520.0,
            emission_band_index=3,
            influence_score=0.75,
        )

        band_dict = original.to_dict()
        restored = WavelengthBand.from_dict(band_dict)

        assert restored.rank == original.rank
        assert restored.excitation_nm == original.excitation_nm
        assert restored.emission_nm == original.emission_nm
        assert restored.emission_band_index == original.emission_band_index
        assert restored.influence_score == original.influence_score


class TestWavelengthResultCreation:
    """Tests for WavelengthResult creation."""

    def test_wavelength_result_creation(self, synthetic_wavelength_bands: List[WavelengthBand]):
        """Basic instantiation with valid bands."""
        metrics = AnalysisMetrics.from_bands(synthetic_wavelength_bands, total_available=100)

        result = WavelengthResult(
            sample_name="test_result",
            selected_bands=synthetic_wavelength_bands,
            metrics=metrics,
        )

        assert result.sample_name == "test_result"
        assert result.n_bands == 5
        assert result.top_band.rank == 1
        assert len(result.excitation_wavelengths) == 3  # 365, 405, 450

    def test_wavelength_result_sequential_ranks(self):
        """Non-sequential ranks raise ValueError."""
        bands = [
            WavelengthBand(rank=1, excitation_nm=365.0, emission_nm=500.0,
                          emission_band_index=0, influence_score=0.9),
            WavelengthBand(rank=3, excitation_nm=405.0, emission_nm=520.0,  # Skip rank 2!
                          emission_band_index=1, influence_score=0.8),
        ]
        metrics = AnalysisMetrics(
            total_bands_available=100,
            bands_selected=2,
            compression_ratio=50.0,
            max_influence_score=0.9,
            min_influence_score=0.8,
            mean_influence_score=0.85,
        )

        with pytest.raises(ValueError, match="Ranks must be sequential"):
            WavelengthResult(
                sample_name="test",
                selected_bands=bands,
                metrics=metrics,
            )

    def test_wavelength_result_empty_bands(self):
        """Empty bands raises ValueError."""
        metrics = AnalysisMetrics(
            total_bands_available=100,
            bands_selected=1,  # This would be invalid but we're testing empty bands check first
            compression_ratio=100.0,
            max_influence_score=0.5,
            min_influence_score=0.5,
            mean_influence_score=0.5,
        )

        with pytest.raises(ValueError, match="selected_bands cannot be empty"):
            WavelengthResult(
                sample_name="test",
                selected_bands=[],
                metrics=metrics,
            )

    def test_wavelength_result_sorts_by_rank(self):
        """Bands are sorted by rank in __post_init__."""
        bands = [
            WavelengthBand(rank=2, excitation_nm=405.0, emission_nm=520.0,
                          emission_band_index=1, influence_score=0.8),
            WavelengthBand(rank=1, excitation_nm=365.0, emission_nm=500.0,
                          emission_band_index=0, influence_score=0.9),
        ]
        metrics = AnalysisMetrics.from_bands(bands, total_available=100)

        result = WavelengthResult(
            sample_name="test",
            selected_bands=bands,
            metrics=metrics,
        )

        # Should be sorted by rank
        assert result.selected_bands[0].rank == 1
        assert result.selected_bands[1].rank == 2


class TestWavelengthResultFromBands:
    """Tests for WavelengthResult factory method and AnalysisMetrics."""

    def test_wavelength_result_from_bands(self, synthetic_wavelength_bands: List[WavelengthBand]):
        """Factory method computes metrics correctly."""
        total_available = 1000

        metrics = AnalysisMetrics.from_bands(synthetic_wavelength_bands, total_available)

        assert metrics.total_bands_available == 1000
        assert metrics.bands_selected == 5
        assert metrics.compression_ratio == 200.0  # 1000 / 5

        # Score statistics
        scores = [b.influence_score for b in synthetic_wavelength_bands]
        assert metrics.max_influence_score == max(scores)
        assert metrics.min_influence_score == min(scores)
        assert abs(metrics.mean_influence_score - sum(scores) / len(scores)) < 1e-6

    def test_analysis_metrics_validation(self):
        """AnalysisMetrics validates score ordering."""
        with pytest.raises(ValueError, match="Score ordering violated"):
            AnalysisMetrics(
                total_bands_available=100,
                bands_selected=5,
                compression_ratio=20.0,
                max_influence_score=0.5,
                min_influence_score=0.9,  # min > max!
                mean_influence_score=0.7,
            )

    def test_analysis_metrics_from_empty_bands(self):
        """from_bands with empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute metrics from empty bands list"):
            AnalysisMetrics.from_bands([], total_available=100)


class TestValidationMetricsCreation:
    """Tests for ValidationMetrics creation."""

    def test_validation_metrics_creation(self):
        """Basic instantiation."""
        confusion_matrix = np.array([[10, 2], [1, 15]])

        metrics = ValidationMetrics(
            adjusted_rand_score=0.85,
            normalized_mutual_info=0.80,
            adjusted_mutual_info=0.75,
            fowlkes_mallows_score=0.82,
            v_measure=0.78,
            homogeneity=0.80,
            completeness=0.76,
            purity=0.89,
            cluster_to_gt_mapping={0: 0, 1: 1},
            confusion_matrix=confusion_matrix,
            per_class_precision={0: 0.91, 1: 0.88},
            per_class_recall={0: 0.83, 1: 0.94},
            per_class_f1={0: 0.87, 1: 0.91},
            n_ground_truth_classes=2,
            n_predicted_clusters=2,
        )

        assert metrics.adjusted_rand_score == 0.85
        assert metrics.purity == 0.89
        assert metrics.n_ground_truth_classes == 2
        assert metrics.n_predicted_clusters == 2

    def test_validation_metrics_score_range_validation(self):
        """Invalid score ranges raise ValueError."""
        confusion_matrix = np.array([[10, 2], [1, 15]])

        with pytest.raises(ValueError, match="purity must be in"):
            ValidationMetrics(
                adjusted_rand_score=0.5,
                normalized_mutual_info=0.5,
                adjusted_mutual_info=0.5,
                fowlkes_mallows_score=0.5,
                v_measure=0.5,
                homogeneity=0.5,
                completeness=0.5,
                purity=1.5,  # Out of range!
                cluster_to_gt_mapping={},
                confusion_matrix=confusion_matrix,
                per_class_precision={},
                per_class_recall={},
                per_class_f1={},
                n_ground_truth_classes=2,
                n_predicted_clusters=2,
            )


class TestValidationMetricsJsonRoundtrip:
    """Tests for ValidationMetrics JSON round-trip."""

    def test_validation_metrics_json_roundtrip(self, tmp_path: Path):
        """to_json/from_json preserves data."""
        confusion_matrix = np.array([[10, 2, 1], [1, 15, 0], [0, 2, 12]])

        original = ValidationMetrics(
            adjusted_rand_score=0.85,
            normalized_mutual_info=0.80,
            adjusted_mutual_info=0.75,
            fowlkes_mallows_score=0.82,
            v_measure=0.78,
            homogeneity=0.80,
            completeness=0.76,
            purity=0.89,
            cluster_to_gt_mapping={0: 0, 1: 1, 2: 2},
            confusion_matrix=confusion_matrix,
            per_class_precision={0: 0.91, 1: 0.88, 2: 0.85},
            per_class_recall={0: 0.83, 1: 0.94, 2: 0.80},
            per_class_f1={0: 0.87, 1: 0.91, 2: 0.82},
            n_ground_truth_classes=3,
            n_predicted_clusters=3,
        )

        # Save and load via dict (no direct to_json method, need to use dict)
        metrics_dict = original.to_dict()

        # Write to JSON file manually
        import json
        json_path = tmp_path / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics_dict, f)

        # Load from JSON
        loaded = ValidationMetrics.from_json(json_path)

        # Verify all values preserved
        assert loaded.adjusted_rand_score == original.adjusted_rand_score
        assert loaded.normalized_mutual_info == original.normalized_mutual_info
        assert loaded.purity == original.purity
        assert loaded.n_ground_truth_classes == original.n_ground_truth_classes
        assert np.array_equal(loaded.confusion_matrix, original.confusion_matrix)
        assert loaded.per_class_f1 == original.per_class_f1


class TestLoadingOptions:
    """Tests for LoadingOptions."""

    def test_loading_options_defaults(self):
        """Default values."""
        opts = LoadingOptions()

        assert opts.cutoff_offset == 30
        assert opts.apply_rayleigh_cutoff is True
        assert opts.downscale_factor == 1
        assert opts.roi is None

    def test_loading_options_validation(self):
        """Invalid values raise ValueError."""
        with pytest.raises(ValueError, match="cutoff_offset must be >= 0"):
            LoadingOptions(cutoff_offset=-1)

        with pytest.raises(ValueError, match="downscale_factor must be >= 1"):
            LoadingOptions(downscale_factor=0)

        with pytest.raises(ValueError, match="roi row_min.*must be < row_max"):
            LoadingOptions(roi=(10, 5, 0, 10))  # row_min > row_max


class TestGroundTruth:
    """Tests for GroundTruth."""

    def test_ground_truth_from_array(self):
        """Factory method auto-generates color mapping."""
        labels = np.array([
            [-1, 0, 0],
            [1, 1, -1],
            [0, 1, 0],
        ])

        gt = GroundTruth.from_array(labels)

        assert gt.n_classes == 2  # 0 and 1 (excludes -1)
        assert gt.shape == (3, 3)
        assert -1 in gt.color_mapping
        assert 0 in gt.color_mapping
        assert 1 in gt.color_mapping

    def test_ground_truth_valid_mask(self):
        """valid_mask excludes background pixels."""
        labels = np.array([
            [-1, 0, 0],
            [1, 1, -1],
            [0, 1, 0],
        ])

        gt = GroundTruth.from_array(labels)
        valid = gt.valid_mask

        assert valid[0, 0] is np.bool_(False)  # -1 is background
        assert valid[0, 1] is np.bool_(True)   # 0 is valid
        assert valid[1, 2] is np.bool_(False)  # -1 is background
