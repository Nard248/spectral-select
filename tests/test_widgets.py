"""Tests for spectral_select.widgets module.

Tests widget helper functions and ROIWidget class management without
requiring a Jupyter environment or interactive display.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from spectral_select.widgets import (
    ROIWidget,
    create_display_image,
    path_to_mask,
)


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def sample_cube():
    """Create a simple 3D cube for testing (height, width, bands)."""
    np.random.seed(42)
    return np.random.rand(50, 60, 10).astype(np.float32)


@pytest.fixture
def mock_spectra_data():
    """Create a mock SpectraData-like object for ROIWidget testing.

    This mock provides the minimal interface required by ROIWidget
    without needing the full SpectraData class.
    """
    # Create mock excitation data
    mock_ex_data = MagicMock()
    mock_ex_data.cube = np.random.rand(50, 60, 10).astype(np.float32)
    mock_ex_data.height = 50
    mock_ex_data.width = 60

    # Create mock SpectraData
    mock_data = MagicMock()
    mock_data.excitation_wavelengths = [365.0, 405.0]
    mock_data.get_excitation.return_value = mock_ex_data

    return mock_data


# =============================================================================
# Tests for helper functions
# =============================================================================


class TestCreateDisplayImage:
    """Tests for create_display_image helper function."""

    def test_mean_projection(self, sample_cube):
        """Test default mean projection creates 2D image."""
        img = create_display_image(sample_cube, method="mean")

        assert img.ndim == 2
        assert img.shape == (50, 60)
        assert img.dtype == np.float32
        # Should be normalized to [0, 1]
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_max_projection(self, sample_cube):
        """Test maximum projection method."""
        img = create_display_image(sample_cube, method="max")

        assert img.ndim == 2
        assert img.shape == (50, 60)

    def test_single_band_extraction(self, sample_cube):
        """Test extracting a specific band."""
        img = create_display_image(sample_cube, band_index=5)

        assert img.ndim == 2
        assert img.shape == (50, 60)

    def test_single_band_out_of_range(self, sample_cube):
        """Test that invalid band_index raises IndexError."""
        with pytest.raises(IndexError):
            create_display_image(sample_cube, band_index=100)

    def test_rgb_false_color(self, sample_cube):
        """Test RGB false color output has 3 channels."""
        img = create_display_image(sample_cube, method="rgb")

        assert img.ndim == 3
        assert img.shape == (50, 60, 3)

    def test_handles_nan_values(self):
        """Test that NaN values are handled gracefully."""
        cube_with_nan = np.random.rand(10, 10, 5).astype(np.float32)
        cube_with_nan[5, 5, :] = np.nan

        img = create_display_image(cube_with_nan)

        assert not np.isnan(img).any()

    def test_handles_all_zeros(self):
        """Test cube of all zeros doesn't cause division by zero."""
        zero_cube = np.zeros((10, 10, 5), dtype=np.float32)

        img = create_display_image(zero_cube)

        assert img.shape == (10, 10)
        # Should be all zeros, normalized
        assert np.allclose(img, 0.0)


class TestPathToMask:
    """Tests for path_to_mask helper function."""

    def test_rectangle_vertices(self):
        """Test rectangle path creates expected mask."""
        # Rectangle from (10, 20) to (40, 50) in (x, y) = (col, row)
        vertices = [(10, 20), (40, 20), (40, 50), (10, 50)]

        mask = path_to_mask(vertices, shape=(100, 100))

        assert mask.shape == (100, 100)
        assert mask.dtype == bool
        # Check that pixels inside rectangle are True
        assert mask[35, 25]  # row=35, col=25 should be inside
        # Check that pixels outside are False
        assert not mask[0, 0]
        assert not mask[99, 99]

    def test_empty_path(self):
        """Test empty path returns all-False mask."""
        vertices = []

        mask = path_to_mask(vertices, shape=(50, 50))

        assert mask.shape == (50, 50)
        assert not mask.any()

    def test_triangle_path(self):
        """Test triangular path."""
        vertices = [(25, 10), (40, 40), (10, 40)]

        mask = path_to_mask(vertices, shape=(50, 50))

        assert mask.shape == (50, 50)
        # Some pixels should be selected
        assert mask.any()


# =============================================================================
# Tests for ROIWidget initialization
# =============================================================================


class TestROIWidgetInit:
    """Tests for ROIWidget initialization."""

    def test_init_with_mock_data(self, mock_spectra_data):
        """Test widget initialization with mock data."""
        widget = ROIWidget(mock_spectra_data)

        assert widget._spatial_shape == (50, 60)
        assert widget._current_class == 0
        assert widget._class_names == ["Class 0"]
        assert widget._class_labels == {}

    def test_init_with_specific_excitation(self, mock_spectra_data):
        """Test initialization with specific excitation wavelength."""
        widget = ROIWidget(mock_spectra_data, excitation=365.0)

        assert widget._excitation == 365.0

    def test_init_invalid_excitation(self, mock_spectra_data):
        """Test that invalid excitation raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            ROIWidget(mock_spectra_data, excitation=999.0)

    def test_init_invalid_tool(self, mock_spectra_data):
        """Test that invalid tool raises ValueError."""
        with pytest.raises(ValueError, match="tool must be"):
            ROIWidget(mock_spectra_data, tool="invalid")


# =============================================================================
# Tests for ROIWidget class management
# =============================================================================


class TestROIWidgetClassManagement:
    """Tests for ROIWidget multi-class management."""

    def test_add_class_auto_name(self, mock_spectra_data):
        """Test adding class with auto-generated name."""
        widget = ROIWidget(mock_spectra_data)

        new_id = widget.add_class()

        assert new_id == 1
        assert len(widget._class_names) == 2
        assert widget._class_names[1] == "Class 1"
        assert widget._current_class == 1

    def test_add_class_custom_name(self, mock_spectra_data):
        """Test adding class with custom name."""
        widget = ROIWidget(mock_spectra_data)

        new_id = widget.add_class("Lichen")

        assert widget._class_names[new_id] == "Lichen"

    def test_set_class(self, mock_spectra_data):
        """Test setting active class."""
        widget = ROIWidget(mock_spectra_data)
        widget.add_class()  # Class 1
        widget.add_class()  # Class 2

        widget.set_class(0)

        assert widget._current_class == 0

    def test_set_class_out_of_range(self, mock_spectra_data):
        """Test setting invalid class raises ValueError."""
        widget = ROIWidget(mock_spectra_data)

        with pytest.raises(ValueError, match="out of range"):
            widget.set_class(99)

    def test_rename_class(self, mock_spectra_data):
        """Test renaming a class."""
        widget = ROIWidget(mock_spectra_data)

        widget.rename_class(0, "Background")

        assert widget._class_names[0] == "Background"

    def test_rename_class_out_of_range(self, mock_spectra_data):
        """Test renaming invalid class raises ValueError."""
        widget = ROIWidget(mock_spectra_data)

        with pytest.raises(ValueError, match="out of range"):
            widget.rename_class(99, "Invalid")

    def test_n_classes_property(self, mock_spectra_data):
        """Test n_classes property."""
        widget = ROIWidget(mock_spectra_data)

        assert widget.n_classes == 1

        widget.add_class()
        assert widget.n_classes == 2

    def test_class_names_property(self, mock_spectra_data):
        """Test class_names property returns copy."""
        widget = ROIWidget(mock_spectra_data)
        widget.add_class("Test")

        names = widget.class_names
        names.append("Modified")  # Should not affect widget

        assert len(widget.class_names) == 2


# =============================================================================
# Tests for ROIWidget mask operations
# =============================================================================


class TestROIWidgetMaskOperations:
    """Tests for ROIWidget mask operations."""

    def test_get_combined_mask_empty(self, mock_spectra_data):
        """Test get_combined_mask returns all -1 when no selections."""
        widget = ROIWidget(mock_spectra_data)

        combined = widget.get_combined_mask()

        assert combined.shape == (50, 60)
        assert (combined == -1).all()

    def test_get_combined_mask_with_data(self, mock_spectra_data):
        """Test get_combined_mask with manually set class labels."""
        widget = ROIWidget(mock_spectra_data)

        # Manually set class masks
        widget._class_labels[0] = np.zeros((50, 60), dtype=bool)
        widget._class_labels[0][10:20, 10:20] = True

        widget._class_labels[1] = np.zeros((50, 60), dtype=bool)
        widget._class_labels[1][30:40, 30:40] = True

        combined = widget.get_combined_mask()

        assert combined[15, 15] == 0  # Class 0 region
        assert combined[35, 35] == 1  # Class 1 region
        assert combined[0, 0] == -1  # Background

    def test_get_class_mask(self, mock_spectra_data):
        """Test get_class_mask retrieves correct mask."""
        widget = ROIWidget(mock_spectra_data)

        widget._class_labels[0] = np.zeros((50, 60), dtype=bool)
        widget._class_labels[0][10:20, 10:20] = True

        mask = widget.get_class_mask(0)

        assert mask is not None
        assert mask[15, 15]
        assert not mask[0, 0]

    def test_get_class_mask_nonexistent(self, mock_spectra_data):
        """Test get_class_mask returns None for non-existent class."""
        widget = ROIWidget(mock_spectra_data)

        mask = widget.get_class_mask(99)

        assert mask is None

    def test_clear_current(self, mock_spectra_data):
        """Test clearing current class mask."""
        widget = ROIWidget(mock_spectra_data)
        widget._class_labels[0] = np.ones((50, 60), dtype=bool)

        widget.clear()

        assert 0 not in widget._class_labels

    def test_clear_all(self, mock_spectra_data):
        """Test clearing all class masks."""
        widget = ROIWidget(mock_spectra_data)
        widget._class_labels[0] = np.ones((50, 60), dtype=bool)
        widget._class_labels[1] = np.ones((50, 60), dtype=bool)

        widget.clear_all()

        assert len(widget._class_labels) == 0


# =============================================================================
# Tests for GroundTruth export
# =============================================================================


class TestROIWidgetGroundTruthExport:
    """Tests for ROIWidget to_ground_truth export."""

    def test_to_ground_truth_with_mock_masks(self, mock_spectra_data):
        """Test to_ground_truth creates valid GroundTruth."""
        widget = ROIWidget(mock_spectra_data)
        widget.add_class("Lichen")

        # Set masks
        widget._class_labels[0] = np.zeros((50, 60), dtype=bool)
        widget._class_labels[0][10:20, 10:20] = True

        widget._class_labels[1] = np.zeros((50, 60), dtype=bool)
        widget._class_labels[1][30:40, 30:40] = True

        gt = widget.to_ground_truth()

        assert gt.labels.shape == (50, 60)
        assert gt.n_classes == 2
        assert gt.class_names == ["Class 0", "Lichen"]

    def test_to_ground_truth_color_mapping(self, mock_spectra_data):
        """Test to_ground_truth includes correct color mapping."""
        widget = ROIWidget(mock_spectra_data)

        widget._class_labels[0] = np.zeros((50, 60), dtype=bool)
        widget._class_labels[0][10:20, 10:20] = True

        gt = widget.to_ground_truth()

        # Check color mapping has RGBA tuples
        assert -1 in gt.color_mapping  # Background
        assert 0 in gt.color_mapping  # Class 0
        assert len(gt.color_mapping[0]) == 4  # RGBA
        assert all(isinstance(v, int) for v in gt.color_mapping[0])

    def test_to_ground_truth_raises_without_roi(self, mock_spectra_data):
        """Test to_ground_truth raises ValueError when no ROIs."""
        widget = ROIWidget(mock_spectra_data)

        with pytest.raises(ValueError, match="No ROIs have been drawn"):
            widget.to_ground_truth()


# =============================================================================
# Tests for save/load mask
# =============================================================================


class TestROIWidgetSaveLoadMask:
    """Tests for ROIWidget save_mask and load_mask."""

    def test_save_load_roundtrip(self, mock_spectra_data):
        """Test saving and loading mask produces identical results."""
        widget = ROIWidget(mock_spectra_data)
        widget.add_class("Lichen")
        widget.add_class("Bark")

        # Set masks for classes 0, 1, 2
        widget._class_labels[0] = np.zeros((50, 60), dtype=bool)
        widget._class_labels[0][10:20, 10:20] = True

        widget._class_labels[1] = np.zeros((50, 60), dtype=bool)
        widget._class_labels[1][25:35, 25:35] = True

        widget._class_labels[2] = np.zeros((50, 60), dtype=bool)
        widget._class_labels[2][40:45, 40:45] = True

        original_combined = widget.get_combined_mask().copy()

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = Path(f.name)

        try:
            widget.save_mask(path)

            # Clear and reload
            widget.clear_all()
            widget.load_mask(path)

            loaded_combined = widget.get_combined_mask()

            assert np.array_equal(original_combined, loaded_combined)
        finally:
            path.unlink()

    def test_load_mask_file_not_found(self, mock_spectra_data):
        """Test load_mask raises FileNotFoundError."""
        widget = ROIWidget(mock_spectra_data)

        with pytest.raises(FileNotFoundError):
            widget.load_mask(Path("nonexistent_file.png"))

    def test_load_mask_shape_mismatch(self, mock_spectra_data):
        """Test load_mask raises ValueError on shape mismatch."""
        widget = ROIWidget(mock_spectra_data)

        # Create a mask with wrong dimensions
        from PIL import Image
        wrong_size_mask = np.zeros((100, 100), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = Path(f.name)

        try:
            Image.fromarray(wrong_size_mask, mode="L").save(path)

            with pytest.raises(ValueError, match="doesn't match"):
                widget.load_mask(path)
        finally:
            path.unlink()


# =============================================================================
# Tests for from_spectra_data factory
# =============================================================================


class TestROIWidgetFactory:
    """Tests for ROIWidget.from_spectra_data factory method."""

    def test_from_spectra_data_default(self, mock_spectra_data):
        """Test factory creates widget with default settings."""
        widget = ROIWidget.from_spectra_data(mock_spectra_data)

        assert widget._excitation == 365.0  # First available

    def test_from_spectra_data_with_excitation(self, mock_spectra_data):
        """Test factory with specific excitation."""
        widget = ROIWidget.from_spectra_data(
            mock_spectra_data,
            excitation=405.0,
        )

        assert widget._excitation == 405.0

    def test_from_spectra_data_invalid_excitation(self, mock_spectra_data):
        """Test factory raises ValueError for invalid excitation."""
        with pytest.raises(ValueError, match="not found"):
            ROIWidget.from_spectra_data(mock_spectra_data, excitation=999.0)
