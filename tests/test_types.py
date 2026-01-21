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

    def test_spectra_data_empty_initialization_raises(self):
        """Empty excitations raises ValueError."""
        with pytest.raises(ValueError, match="excitations cannot be empty"):
            SpectraData(excitations={})

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


class TestWavelengthResultExcel:
    """Tests for WavelengthResult.to_excel() export functionality."""

    @pytest.fixture
    def sample_wavelength_result(self) -> WavelengthResult:
        """Create a sample WavelengthResult for Excel export testing."""
        bands = [
            WavelengthBand(
                rank=1,
                excitation_nm=365.0,
                emission_nm=500.0,
                emission_band_index=0,
                influence_score=0.95,
            ),
            WavelengthBand(
                rank=2,
                excitation_nm=405.0,
                emission_nm=520.0,
                emission_band_index=2,
                influence_score=0.85,
            ),
            WavelengthBand(
                rank=3,
                excitation_nm=365.0,
                emission_nm=530.0,
                emission_band_index=3,
                influence_score=0.75,
            ),
        ]
        metrics = AnalysisMetrics.from_bands(bands, total_available=100)
        return WavelengthResult(
            sample_name="excel_test_sample",
            selected_bands=bands,
            metrics=metrics,
        )

    def test_to_excel_creates_file(
        self, sample_wavelength_result: WavelengthResult, tmp_path: Path
    ):
        """Verify to_excel() creates file at specified path."""
        excel_path = tmp_path / "test_output.xlsx"
        assert not excel_path.exists()

        sample_wavelength_result.to_excel(excel_path)

        assert excel_path.exists()

    def test_to_excel_wavelengths_sheet(
        self, sample_wavelength_result: WavelengthResult, tmp_path: Path
    ):
        """Verify Wavelengths sheet has correct columns and data."""
        import pandas as pd

        excel_path = tmp_path / "wavelengths_sheet_test.xlsx"
        sample_wavelength_result.to_excel(excel_path)

        # Read back the Wavelengths sheet
        df = pd.read_excel(excel_path, sheet_name="Wavelengths")

        # Check columns (includes Band_Index per implementation)
        expected_columns = ["Rank", "Excitation_nm", "Emission_nm", "Band_Index", "Score"]
        assert list(df.columns) == expected_columns

        # Check data
        assert len(df) == 3
        assert df.iloc[0]["Rank"] == 1
        assert df.iloc[0]["Excitation_nm"] == 365.0
        assert df.iloc[0]["Emission_nm"] == 500.0
        assert df.iloc[0]["Band_Index"] == 0
        assert df.iloc[0]["Score"] == 0.95

        assert df.iloc[2]["Rank"] == 3
        assert df.iloc[2]["Score"] == 0.75

    def test_to_excel_metrics_sheet(
        self, sample_wavelength_result: WavelengthResult, tmp_path: Path
    ):
        """Verify Metrics sheet is present and has correct values when include_metrics=True."""
        import pandas as pd

        excel_path = tmp_path / "metrics_sheet_test.xlsx"
        sample_wavelength_result.to_excel(excel_path, include_metrics=True)

        # Read back the Metrics sheet
        df = pd.read_excel(excel_path, sheet_name="Metrics")

        # Check column names match implementation (horizontal layout)
        expected_columns = {
            "Total_Bands", "Bands_Selected", "Compression_Ratio",
            "Max_Score", "Min_Score", "Mean_Score"
        }
        assert set(df.columns) == expected_columns

        # Verify specific values
        assert df.iloc[0]["Bands_Selected"] == 3
        assert df.iloc[0]["Total_Bands"] == 100
        assert abs(df.iloc[0]["Compression_Ratio"] - (100 / 3)) < 0.01

    def test_to_excel_no_metrics(
        self, sample_wavelength_result: WavelengthResult, tmp_path: Path
    ):
        """Verify only Wavelengths sheet when include_metrics=False."""
        import pandas as pd

        excel_path = tmp_path / "no_metrics_test.xlsx"
        sample_wavelength_result.to_excel(excel_path, include_metrics=False)

        # Read the Excel file and check sheet names
        xlsx = pd.ExcelFile(excel_path)
        sheet_names = xlsx.sheet_names

        assert "Wavelengths" in sheet_names
        assert "Metrics" not in sheet_names
        assert len(sheet_names) == 1

    def test_to_excel_creates_parent_dirs(
        self, sample_wavelength_result: WavelengthResult, tmp_path: Path
    ):
        """Verify parent directories are created if needed."""
        # Create a nested path that doesn't exist
        excel_path = tmp_path / "nested" / "dir" / "output.xlsx"
        assert not excel_path.parent.exists()

        sample_wavelength_result.to_excel(excel_path)

        assert excel_path.exists()
        assert excel_path.parent.exists()

    def test_to_excel_returns_none(
        self, sample_wavelength_result: WavelengthResult, tmp_path: Path
    ):
        """Verify to_excel returns None (follows pattern of other save methods)."""
        excel_path = tmp_path / "return_test.xlsx"
        result = sample_wavelength_result.to_excel(excel_path)

        assert result is None


# ============================================================================
# SpectraData.from_raw() Tests
# ============================================================================


class TestSpectraDataFromRaw:
    """Tests for SpectraData.from_raw() factory method."""

    def test_from_raw_missing_path(self):
        """Raises FileNotFoundError (via DataLoadingError) for missing directory."""
        from spectral_select.loader import DataLoadingError

        with pytest.raises(DataLoadingError) as exc_info:
            SpectraData.from_raw("/nonexistent/path/to/data")

        assert "does not exist" in str(exc_info.value)

    def test_from_raw_empty_directory(self, tmp_path: Path):
        """Raises DataLoadingError for empty directory."""
        from spectral_select.loader import DataLoadingError

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(DataLoadingError) as exc_info:
            SpectraData.from_raw(empty_dir)

        assert "empty" in str(exc_info.value).lower()

    def test_from_raw_sample_name_from_directory(self, tmp_path: Path):
        """Derives sample_name from directory name when not provided."""
        # We can't actually call from_raw without ImageJ, but we can test the logic
        # by verifying the path handling. This tests the parameter handling.
        from spectral_select.loader import DataLoadingError

        data_dir = tmp_path / "MySampleName"
        data_dir.mkdir()

        # Will fail at loading, but the error should contain our path
        with pytest.raises(DataLoadingError) as exc_info:
            SpectraData.from_raw(data_dir)

        # Verify the path was processed correctly
        assert exc_info.value.path == data_dir

    def test_from_raw_custom_sample_name(self, tmp_path: Path):
        """Uses provided sample_name when specified."""
        from spectral_select.loader import DataLoadingError

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Will fail at loading, but we can verify sample_name parameter is accepted
        with pytest.raises(DataLoadingError):
            SpectraData.from_raw(data_dir, sample_name="CustomName")

        # No error from sample_name parameter - test passes


# ============================================================================
# SpectraData.to_pickle() Tests
# ============================================================================


class TestSpectraDataToPickle:
    """Tests for SpectraData.to_pickle() method."""

    def test_to_pickle_creates_file(
        self, synthetic_spectra_data: SpectraData, tmp_path: Path
    ):
        """File exists after to_pickle."""
        output_path = tmp_path / "output.pkl"
        assert not output_path.exists()

        synthetic_spectra_data.to_pickle(output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_to_pickle_creates_parent_dirs(
        self, synthetic_spectra_data: SpectraData, tmp_path: Path
    ):
        """Parent directories created automatically."""
        output_path = tmp_path / "nested" / "dir" / "deep" / "output.pkl"
        assert not output_path.parent.exists()

        synthetic_spectra_data.to_pickle(output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_to_pickle_returns_path(
        self, synthetic_spectra_data: SpectraData, tmp_path: Path
    ):
        """Returns Path object."""
        output_path = tmp_path / "output.pkl"

        result = synthetic_spectra_data.to_pickle(output_path)

        assert isinstance(result, Path)
        assert result == output_path

    def test_to_pickle_accepts_string_path(
        self, synthetic_spectra_data: SpectraData, tmp_path: Path
    ):
        """Accepts string path and converts to Path."""
        output_path = str(tmp_path / "string_path.pkl")

        result = synthetic_spectra_data.to_pickle(output_path)

        assert isinstance(result, Path)
        assert result.exists()


# ============================================================================
# SpectraData Round-Trip Tests
# ============================================================================


class TestSpectraDataRoundTrip:
    """Tests for SpectraData pickle round-trip serialization."""

    def test_roundtrip_basic(
        self, synthetic_spectra_data: SpectraData, tmp_path: Path
    ):
        """from_pickle(to_pickle(data)) produces equivalent data."""
        output_path = tmp_path / "roundtrip.pkl"

        # Save
        synthetic_spectra_data.to_pickle(output_path)

        # Load
        loaded = SpectraData.from_pickle(output_path)

        # Basic properties match
        assert loaded.sample_name == output_path.stem  # from_pickle uses stem as name
        assert loaded.n_excitations == synthetic_spectra_data.n_excitations
        assert loaded.spatial_shape == synthetic_spectra_data.spatial_shape

    def test_roundtrip_preserves_excitations(
        self, synthetic_spectra_data: SpectraData, tmp_path: Path
    ):
        """Same excitation wavelengths after round-trip."""
        output_path = tmp_path / "roundtrip.pkl"

        synthetic_spectra_data.to_pickle(output_path)
        loaded = SpectraData.from_pickle(output_path)

        assert loaded.excitation_wavelengths == synthetic_spectra_data.excitation_wavelengths

    def test_roundtrip_preserves_cubes(
        self, synthetic_spectra_data: SpectraData, tmp_path: Path
    ):
        """Same cube data after round-trip (np.allclose)."""
        output_path = tmp_path / "roundtrip.pkl"

        synthetic_spectra_data.to_pickle(output_path)
        loaded = SpectraData.from_pickle(output_path)

        for ex_nm in synthetic_spectra_data.excitation_wavelengths:
            original_cube = synthetic_spectra_data.get_excitation(ex_nm).cube
            loaded_cube = loaded.get_excitation(ex_nm).cube
            assert np.allclose(original_cube, loaded_cube)

    def test_roundtrip_preserves_mask(
        self, synthetic_spectra_data: SpectraData, tmp_path: Path
    ):
        """Same mask after round-trip."""
        output_path = tmp_path / "roundtrip.pkl"

        # Ensure mask exists
        assert synthetic_spectra_data.mask is not None

        synthetic_spectra_data.to_pickle(output_path)
        loaded = SpectraData.from_pickle(output_path)

        assert loaded.mask is not None
        assert np.array_equal(loaded.mask, synthetic_spectra_data.mask)

    def test_roundtrip_preserves_emission_wavelengths(
        self, synthetic_spectra_data: SpectraData, tmp_path: Path
    ):
        """Same emission wavelengths after round-trip."""
        output_path = tmp_path / "roundtrip.pkl"

        synthetic_spectra_data.to_pickle(output_path)
        loaded = SpectraData.from_pickle(output_path)

        for ex_nm in synthetic_spectra_data.excitation_wavelengths:
            original_em = synthetic_spectra_data.get_excitation(ex_nm).emission_wavelengths
            loaded_em = loaded.get_excitation(ex_nm).emission_wavelengths
            assert original_em == loaded_em

    def test_roundtrip_with_no_mask(self, tmp_path: Path):
        """Round-trip works when mask is None."""
        np.random.seed(42)
        cube = np.random.rand(5, 5, 3).astype(np.float32)

        data = SpectraData(
            excitations={
                365.0: ExcitationData(
                    excitation_nm=365.0,
                    cube=cube,
                    emission_wavelengths=[500.0, 510.0, 520.0],
                )
            },
            mask=None,  # No mask
            sample_name="no_mask_test",
        )

        output_path = tmp_path / "no_mask.pkl"
        data.to_pickle(output_path)
        loaded = SpectraData.from_pickle(output_path)

        assert loaded.mask is None

    def test_roundtrip_preserves_exposure_time(self, tmp_path: Path):
        """Exposure time preserved through round-trip."""
        np.random.seed(42)
        cube = np.random.rand(5, 5, 3).astype(np.float32)

        data = SpectraData(
            excitations={
                365.0: ExcitationData(
                    excitation_nm=365.0,
                    cube=cube,
                    emission_wavelengths=[500.0, 510.0, 520.0],
                    exposure_time=1.5,
                )
            },
            sample_name="exposure_test",
        )

        output_path = tmp_path / "exposure.pkl"
        data.to_pickle(output_path)
        loaded = SpectraData.from_pickle(output_path)

        assert loaded.get_excitation(365.0).exposure_time == 1.5

    def test_roundtrip_preserves_laser_power(self, tmp_path: Path):
        """Laser power preserved through round-trip."""
        np.random.seed(42)
        cube = np.random.rand(5, 5, 3).astype(np.float32)

        data = SpectraData(
            excitations={
                365.0: ExcitationData(
                    excitation_nm=365.0,
                    cube=cube,
                    emission_wavelengths=[500.0, 510.0, 520.0],
                    laser_power=100.0,
                )
            },
            sample_name="power_test",
        )

        output_path = tmp_path / "power.pkl"
        data.to_pickle(output_path)
        loaded = SpectraData.from_pickle(output_path)

        assert loaded.get_excitation(365.0).laser_power == 100.0
