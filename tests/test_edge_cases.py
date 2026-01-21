"""Edge case tests for spectral_select library.

This module tests boundary conditions, error handling, and unusual inputs
to ensure the library behaves correctly in edge cases.
"""

import numpy as np
import pytest

from spectral_select import (
    Config,
    ExcitationData,
    SpectraData,
    WavelengthBand,
    WavelengthResult,
    AnalysisMetrics,
    GroundTruth,
    ValidationMetrics,
)


# =============================================================================
# Config Edge Cases
# =============================================================================


class TestConfigEdgeCases:
    """Test Config with edge case inputs."""

    def test_config_minimal_bands_to_select(self):
        """Config should accept n_bands_to_select=1."""
        config = Config(sample_name="test", n_bands_to_select=1)
        assert config.n_bands_to_select == 1

    def test_config_large_bands_to_select(self):
        """Config should accept large n_bands_to_select."""
        config = Config(sample_name="test", n_bands_to_select=10000)
        assert config.n_bands_to_select == 10000

    def test_config_empty_sample_name_raises(self):
        """Config should reject empty sample_name."""
        with pytest.raises(ValueError, match="sample_name cannot be empty"):
            Config(sample_name="")

    def test_config_whitespace_only_sample_name_raises(self):
        """Config should reject whitespace-only sample_name."""
        with pytest.raises(ValueError, match="sample_name cannot be empty"):
            Config(sample_name="   ")

    def test_config_zero_bands_to_select_raises(self):
        """Config should reject n_bands_to_select=0."""
        with pytest.raises(ValueError):
            Config(sample_name="test", n_bands_to_select=0)

    def test_config_negative_bands_to_select_raises(self):
        """Config should reject negative n_bands_to_select."""
        with pytest.raises(ValueError):
            Config(sample_name="test", n_bands_to_select=-5)

    def test_config_zero_patch_size_raises(self):
        """Config should reject patch_size=0."""
        with pytest.raises(ValueError):
            Config(sample_name="test", patch_size=0)

    def test_config_very_small_patch_size(self):
        """Config should accept patch_size=1."""
        config = Config(sample_name="test", patch_size=1)
        assert config.patch_size == 1

    def test_config_zero_important_dimensions_raises(self):
        """Config should reject n_important_dimensions=0."""
        with pytest.raises(ValueError):
            Config(sample_name="test", n_important_dimensions=0)

    def test_config_empty_perturbation_magnitudes_raises(self):
        """Config should reject empty perturbation_magnitudes."""
        with pytest.raises(ValueError, match="perturbation_magnitudes cannot be empty"):
            Config(sample_name="test", perturbation_magnitudes=[])

    def test_config_negative_perturbation_magnitude_raises(self):
        """Config should reject negative perturbation magnitudes."""
        with pytest.raises(ValueError):
            Config(sample_name="test", perturbation_magnitudes=[-10, 20])

    def test_config_invalid_dimension_selection_method_raises(self):
        """Config should reject invalid dimension_selection_method."""
        with pytest.raises(ValueError, match="dimension_selection_method"):
            Config(sample_name="test", dimension_selection_method="invalid")

    def test_config_invalid_perturbation_method_raises(self):
        """Config should reject invalid perturbation_method."""
        with pytest.raises(ValueError, match="perturbation_method"):
            Config(sample_name="test", perturbation_method="invalid")

    def test_config_invalid_normalization_method_raises(self):
        """Config should reject invalid normalization_method."""
        with pytest.raises(ValueError, match="normalization_method"):
            Config(sample_name="test", normalization_method="invalid")

    def test_config_invalid_diversity_method_raises(self):
        """Config should reject invalid diversity_method."""
        with pytest.raises(ValueError, match="diversity_method"):
            Config(sample_name="test", diversity_method="invalid")

    def test_config_negative_min_distance_raises(self):
        """Config should reject negative min_distance_nm."""
        with pytest.raises(ValueError):
            Config(sample_name="test", min_distance_nm=-10.0)

    def test_config_lambda_diversity_out_of_range_raises(self):
        """Config should reject lambda_diversity outside [0, 1]."""
        with pytest.raises(ValueError):
            Config(sample_name="test", lambda_diversity=1.5)
        with pytest.raises(ValueError):
            Config(sample_name="test", lambda_diversity=-0.1)


# =============================================================================
# ExcitationData Edge Cases
# =============================================================================


class TestExcitationDataEdgeCases:
    """Test ExcitationData with edge case inputs."""

    def test_excitation_data_single_pixel(self):
        """ExcitationData should work with 1x1 spatial dimensions."""
        cube = np.array([[[0.5, 0.6, 0.7]]])  # 1x1x3
        data = ExcitationData(
            excitation_nm=365.0,
            cube=cube,
            emission_wavelengths=[500.0, 510.0, 520.0],
        )
        assert data.cube.shape == (1, 1, 3)
        assert data.n_bands == 3

    def test_excitation_data_single_band(self):
        """ExcitationData should work with single emission band."""
        cube = np.random.rand(5, 5, 1)
        data = ExcitationData(
            excitation_nm=365.0,
            cube=cube,
            emission_wavelengths=[500.0],
        )
        assert data.n_bands == 1

    def test_excitation_data_zero_excitation_raises(self):
        """ExcitationData should reject excitation_nm <= 0."""
        cube = np.random.rand(5, 5, 3)
        with pytest.raises(ValueError):
            ExcitationData(
                excitation_nm=0.0,
                cube=cube,
                emission_wavelengths=[500.0, 510.0, 520.0],
            )

    def test_excitation_data_negative_excitation_raises(self):
        """ExcitationData should reject negative excitation_nm."""
        cube = np.random.rand(5, 5, 3)
        with pytest.raises(ValueError):
            ExcitationData(
                excitation_nm=-365.0,
                cube=cube,
                emission_wavelengths=[500.0, 510.0, 520.0],
            )

    def test_excitation_data_wavelength_mismatch_raises(self):
        """ExcitationData should reject mismatched emission wavelengths."""
        cube = np.random.rand(5, 5, 3)  # 3 bands
        with pytest.raises(ValueError, match="emission_wavelengths"):
            ExcitationData(
                excitation_nm=365.0,
                cube=cube,
                emission_wavelengths=[500.0, 510.0],  # Only 2 wavelengths
            )

    def test_excitation_data_2d_cube_raises(self):
        """ExcitationData should reject 2D cube."""
        cube = np.random.rand(5, 5)  # 2D
        with pytest.raises(ValueError, match="3D"):
            ExcitationData(
                excitation_nm=365.0,
                cube=cube,
                emission_wavelengths=[500.0],
            )

    def test_excitation_data_4d_cube_raises(self):
        """ExcitationData should reject 4D cube."""
        cube = np.random.rand(5, 5, 3, 2)  # 4D
        with pytest.raises(ValueError, match="3D"):
            ExcitationData(
                excitation_nm=365.0,
                cube=cube,
                emission_wavelengths=[500.0, 510.0, 520.0],
            )

    def test_excitation_data_preserves_dtype_float32(self):
        """ExcitationData should preserve float32 dtype."""
        cube = np.random.rand(5, 5, 3).astype(np.float32)
        data = ExcitationData(
            excitation_nm=365.0,
            cube=cube,
            emission_wavelengths=[500.0, 510.0, 520.0],
        )
        assert data.cube.dtype == np.float32

    def test_excitation_data_with_nan_values(self):
        """ExcitationData should accept cubes with NaN values."""
        cube = np.random.rand(5, 5, 3)
        cube[2, 2, :] = np.nan
        data = ExcitationData(
            excitation_nm=365.0,
            cube=cube,
            emission_wavelengths=[500.0, 510.0, 520.0],
        )
        assert np.isnan(data.cube[2, 2, 0])

    def test_excitation_data_with_inf_values(self):
        """ExcitationData should accept cubes with inf values."""
        cube = np.random.rand(5, 5, 3)
        cube[0, 0, 0] = np.inf
        data = ExcitationData(
            excitation_nm=365.0,
            cube=cube,
            emission_wavelengths=[500.0, 510.0, 520.0],
        )
        assert np.isinf(data.cube[0, 0, 0])


# =============================================================================
# SpectraData Edge Cases
# =============================================================================


class TestSpectraDataEdgeCases:
    """Test SpectraData with edge case inputs."""

    def test_spectra_data_single_excitation(self):
        """SpectraData should work with single excitation."""
        cube = np.random.rand(10, 10, 5)
        ex_data = ExcitationData(
            excitation_nm=365.0,
            cube=cube,
            emission_wavelengths=[500.0, 510.0, 520.0, 530.0, 540.0],
        )
        data = SpectraData(
            excitations={365.0: ex_data},
            sample_name="single_ex",
        )
        assert data.n_excitations == 1
        assert data.excitation_wavelengths == [365.0]

    def test_spectra_data_no_mask(self):
        """SpectraData should work without mask."""
        cube = np.random.rand(10, 10, 5)
        ex_data = ExcitationData(
            excitation_nm=365.0,
            cube=cube,
            emission_wavelengths=[500.0, 510.0, 520.0, 530.0, 540.0],
        )
        data = SpectraData(
            excitations={365.0: ex_data},
            mask=None,
            sample_name="no_mask",
        )
        assert data.mask is None

    def test_spectra_data_all_false_mask(self):
        """SpectraData should accept all-False mask (no valid pixels)."""
        cube = np.random.rand(10, 10, 5)
        ex_data = ExcitationData(
            excitation_nm=365.0,
            cube=cube,
            emission_wavelengths=[500.0, 510.0, 520.0, 530.0, 540.0],
        )
        mask = np.zeros((10, 10), dtype=bool)  # All False
        data = SpectraData(
            excitations={365.0: ex_data},
            mask=mask,
            sample_name="no_valid",
        )
        assert not np.any(data.mask)

    def test_spectra_data_mask_shape_mismatch_raises(self):
        """SpectraData should reject mask with wrong shape."""
        cube = np.random.rand(10, 10, 5)
        ex_data = ExcitationData(
            excitation_nm=365.0,
            cube=cube,
            emission_wavelengths=[500.0, 510.0, 520.0, 530.0, 540.0],
        )
        mask = np.ones((8, 8), dtype=bool)  # Wrong shape
        with pytest.raises(ValueError, match="mask"):
            SpectraData(
                excitations={365.0: ex_data},
                mask=mask,
                sample_name="mismatch",
            )

    def test_spectra_data_empty_excitations_raises(self):
        """SpectraData should reject empty excitations dict."""
        with pytest.raises(ValueError, match="excitations"):
            SpectraData(
                excitations={},
                sample_name="empty",
            )

    def test_spectra_data_mixed_spatial_shapes_raises(self):
        """SpectraData should reject excitations with different spatial shapes."""
        cube1 = np.random.rand(10, 10, 5)
        cube2 = np.random.rand(8, 8, 5)  # Different shape
        ex_data1 = ExcitationData(
            excitation_nm=365.0,
            cube=cube1,
            emission_wavelengths=[500.0, 510.0, 520.0, 530.0, 540.0],
        )
        ex_data2 = ExcitationData(
            excitation_nm=405.0,
            cube=cube2,
            emission_wavelengths=[500.0, 510.0, 520.0, 530.0, 540.0],
        )
        with pytest.raises(ValueError, match="shape"):
            SpectraData(
                excitations={365.0: ex_data1, 405.0: ex_data2},
                sample_name="mixed",
            )


# =============================================================================
# WavelengthBand Edge Cases
# =============================================================================


class TestWavelengthBandEdgeCases:
    """Test WavelengthBand with edge case inputs."""

    def test_wavelength_band_zero_influence(self):
        """WavelengthBand should accept zero influence score."""
        band = WavelengthBand(
            rank=1,
            excitation_nm=365.0,
            emission_nm=500.0,
            emission_band_index=0,
            influence_score=0.0,
        )
        assert band.influence_score == 0.0

    def test_wavelength_band_negative_influence(self):
        """WavelengthBand should accept negative influence score."""
        band = WavelengthBand(
            rank=1,
            excitation_nm=365.0,
            emission_nm=500.0,
            emission_band_index=0,
            influence_score=-0.5,
        )
        assert band.influence_score == -0.5

    def test_wavelength_band_very_small_influence_repr(self):
        """WavelengthBand repr should use scientific notation for small values."""
        band = WavelengthBand(
            rank=1,
            excitation_nm=365.0,
            emission_nm=500.0,
            emission_band_index=0,
            influence_score=1.5e-08,
        )
        repr_str = repr(band)
        # Should use scientific notation for small values
        assert "1.5" in repr_str or "e-08" in repr_str.lower()

    def test_wavelength_band_very_large_influence(self):
        """WavelengthBand should accept very large influence score."""
        band = WavelengthBand(
            rank=1,
            excitation_nm=365.0,
            emission_nm=500.0,
            emission_band_index=0,
            influence_score=1e10,
        )
        assert band.influence_score == 1e10

    def test_wavelength_band_zero_rank_raises(self):
        """WavelengthBand should reject rank=0."""
        with pytest.raises(ValueError, match="rank"):
            WavelengthBand(
                rank=0,
                excitation_nm=365.0,
                emission_nm=500.0,
                emission_band_index=0,
                influence_score=0.5,
            )

    def test_wavelength_band_negative_rank_raises(self):
        """WavelengthBand should reject negative rank."""
        with pytest.raises(ValueError, match="rank"):
            WavelengthBand(
                rank=-1,
                excitation_nm=365.0,
                emission_nm=500.0,
                emission_band_index=0,
                influence_score=0.5,
            )

    def test_wavelength_band_negative_band_index_raises(self):
        """WavelengthBand should reject negative emission_band_index."""
        with pytest.raises(ValueError, match="emission_band_index"):
            WavelengthBand(
                rank=1,
                excitation_nm=365.0,
                emission_nm=500.0,
                emission_band_index=-1,
                influence_score=0.5,
            )


# =============================================================================
# GroundTruth Edge Cases
# =============================================================================


class TestGroundTruthEdgeCases:
    """Test GroundTruth with edge case inputs."""

    def test_ground_truth_single_class(self):
        """GroundTruth should work with single class (all same label)."""
        labels = np.ones((10, 10), dtype=np.int32)
        gt = GroundTruth.from_array(labels)
        assert gt.n_classes == 1

    def test_ground_truth_all_background(self):
        """GroundTruth should handle all-background array (-1 is background)."""
        labels = np.full((10, 10), -1, dtype=np.int32)  # All -1 = background
        gt = GroundTruth.from_array(labels)
        # n_classes should be 0 since all are background
        assert gt.n_classes == 0
        assert not np.any(gt.valid_mask)

    def test_ground_truth_single_pixel_class(self):
        """GroundTruth should work when a class has only 1 pixel."""
        labels = np.ones((10, 10), dtype=np.int32)
        labels[5, 5] = 2  # Single pixel of class 2
        gt = GroundTruth.from_array(labels)
        # Check n_classes is 2 (classes 1 and 2)
        assert gt.n_classes == 2

    def test_ground_truth_many_classes(self):
        """GroundTruth should work with many classes."""
        labels = np.arange(100).reshape(10, 10).astype(np.int32)
        gt = GroundTruth.from_array(labels)
        assert gt.n_classes == 100

    def test_ground_truth_with_background(self):
        """GroundTruth should correctly identify background (-1)."""
        labels = np.array([[-1, -1, 1], [1, 2, 2], [2, 1, -1]], dtype=np.int32)
        gt = GroundTruth.from_array(labels)
        # Classes 1 and 2 should be counted (not -1)
        assert gt.n_classes == 2
        # valid_mask should be False where labels == -1
        assert not gt.valid_mask[0, 0]
        assert gt.valid_mask[1, 1]

    def test_ground_truth_1d_array_raises(self):
        """GroundTruth should reject 1D array."""
        labels = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        with pytest.raises(ValueError, match="2D"):
            GroundTruth.from_array(labels)

    def test_ground_truth_3d_array_raises(self):
        """GroundTruth should reject 3D array."""
        labels = np.random.randint(0, 5, (10, 10, 3), dtype=np.int32)
        with pytest.raises(ValueError, match="2D"):
            GroundTruth.from_array(labels)


# =============================================================================
# WavelengthResult Edge Cases
# =============================================================================


class TestWavelengthResultEdgeCases:
    """Test WavelengthResult with edge case inputs."""

    def test_wavelength_result_empty_bands_metrics_raises(self):
        """AnalysisMetrics.from_bands should reject empty list."""
        with pytest.raises(ValueError, match="empty"):
            AnalysisMetrics.from_bands([], 100)

    def test_wavelength_result_minimal(self):
        """WavelengthResult should work with minimal valid data."""
        band = WavelengthBand(
            rank=1,
            excitation_nm=365.0,
            emission_nm=500.0,
            emission_band_index=0,
            influence_score=0.5,
        )
        metrics = AnalysisMetrics.from_bands([band], 100)
        result = WavelengthResult(
            sample_name="test",
            selected_bands=[band],
            metrics=metrics,
            config_snapshot={},
            method_summary={},
        )
        assert result.n_bands == 1

    def test_wavelength_result_single_band(self):
        """WavelengthResult should work with single band."""
        band = WavelengthBand(
            rank=1,
            excitation_nm=365.0,
            emission_nm=500.0,
            emission_band_index=0,
            influence_score=0.9,
        )
        metrics = AnalysisMetrics.from_bands([band], 100)
        result = WavelengthResult(
            sample_name="test",
            selected_bands=[band],
            metrics=metrics,
            config_snapshot={},
            method_summary={},
        )
        assert result.n_bands == 1


# =============================================================================
# ValidationMetrics Edge Cases
# =============================================================================


class TestValidationMetricsEdgeCases:
    """Test ValidationMetrics with edge case inputs.

    Note: ValidationMetrics is a complex dataclass with many required fields.
    These tests verify the basic validation logic works correctly.
    """

    def test_validation_metrics_perfect_scores(self):
        """ValidationMetrics should handle perfect scores."""
        metrics = ValidationMetrics(
            adjusted_rand_score=1.0,
            normalized_mutual_info=1.0,
            adjusted_mutual_info=1.0,
            fowlkes_mallows_score=1.0,
            v_measure=1.0,
            homogeneity=1.0,
            completeness=1.0,
            purity=1.0,
            cluster_to_gt_mapping={0: 0, 1: 1},
            confusion_matrix=np.array([[50, 0], [0, 50]]),
            per_class_precision={0: 1.0, 1: 1.0},
            per_class_recall={0: 1.0, 1: 1.0},
            per_class_f1={0: 1.0, 1: 1.0},
            n_ground_truth_classes=2,
            n_predicted_clusters=2,
        )
        assert metrics.adjusted_rand_score == 1.0
        assert metrics.normalized_mutual_info == 1.0

    def test_validation_metrics_boundary_scores(self):
        """ValidationMetrics should handle boundary case scores."""
        metrics = ValidationMetrics(
            adjusted_rand_score=-0.5,  # Can be negative
            normalized_mutual_info=0.0,  # Minimum
            adjusted_mutual_info=-0.1,  # Can be negative
            fowlkes_mallows_score=0.5,
            v_measure=0.0,  # Minimum
            homogeneity=0.0,  # Minimum
            completeness=0.0,  # Minimum
            purity=0.0,  # Minimum
            cluster_to_gt_mapping={},
            confusion_matrix=np.array([[25, 25], [25, 25]]),
            per_class_precision={},
            per_class_recall={},
            per_class_f1={},
            n_ground_truth_classes=2,
            n_predicted_clusters=1,
        )
        assert metrics.adjusted_rand_score == -0.5
        assert metrics.v_measure == 0.0

    def test_validation_metrics_summary_method(self):
        """ValidationMetrics summary() should work."""
        metrics = ValidationMetrics(
            adjusted_rand_score=0.8,
            normalized_mutual_info=0.7,
            adjusted_mutual_info=0.6,
            fowlkes_mallows_score=0.85,
            v_measure=0.75,
            homogeneity=0.8,
            completeness=0.7,
            purity=0.9,
            cluster_to_gt_mapping={0: 0},
            confusion_matrix=np.array([[90, 10]]),
            per_class_precision={0: 0.9},
            per_class_recall={0: 0.9},
            per_class_f1={0: 0.9},
            n_ground_truth_classes=1,
            n_predicted_clusters=1,
        )
        summary = metrics.summary()
        assert isinstance(summary, str)
        assert "ARI" in summary or "0.8" in summary


# =============================================================================
# Numeric Precision Edge Cases
# =============================================================================


class TestNumericPrecisionEdgeCases:
    """Test handling of edge cases in numeric precision."""

    def test_very_small_cube_values(self):
        """ExcitationData should handle very small values."""
        cube = np.full((5, 5, 3), 1e-15, dtype=np.float64)
        data = ExcitationData(
            excitation_nm=365.0,
            cube=cube,
            emission_wavelengths=[500.0, 510.0, 520.0],
        )
        assert np.allclose(data.cube, 1e-15)

    def test_very_large_cube_values(self):
        """ExcitationData should handle very large values."""
        cube = np.full((5, 5, 3), 1e15, dtype=np.float64)
        data = ExcitationData(
            excitation_nm=365.0,
            cube=cube,
            emission_wavelengths=[500.0, 510.0, 520.0],
        )
        assert np.allclose(data.cube, 1e15)

    def test_mixed_extreme_values(self):
        """ExcitationData should handle mix of extreme values."""
        cube = np.array([
            [[1e-15, 0.5, 1e15]],
        ])
        data = ExcitationData(
            excitation_nm=365.0,
            cube=cube,
            emission_wavelengths=[500.0, 510.0, 520.0],
        )
        assert data.cube[0, 0, 0] == pytest.approx(1e-15)
        assert data.cube[0, 0, 2] == pytest.approx(1e15)
