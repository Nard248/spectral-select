"""Import smoke tests for spectral_select package.

These tests verify that all modules can be imported without errors,
catching circular dependencies, missing dependencies, and syntax errors.
"""
import pytest


class TestPackageImports:
    """Test top-level package imports."""

    def test_import_spectral_select(self):
        """Package should be importable."""
        import spectral_select
        assert hasattr(spectral_select, "__version__") or True  # Version optional

    def test_import_public_api(self):
        """All public API classes should be importable from package root."""
        from spectral_select import (
            Analyzer,
            Config,
            SpectraData,
            WavelengthResult,
            Validator,
            Visualizer,
            GroundTruth,
            load_ground_truth_from_png,
        )
        assert all([Analyzer, Config, SpectraData, WavelengthResult,
                   Validator, Visualizer, GroundTruth, load_ground_truth_from_png])

    def test_import_version(self):
        """Package version should be accessible."""
        from spectral_select import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0


class TestModuleImports:
    """Test individual module imports."""

    def test_import_config(self):
        """Config module should be importable."""
        from spectral_select import config
        from spectral_select.config import Config
        assert Config is not None

    def test_import_types(self):
        """Types module should be importable."""
        from spectral_select import types
        from spectral_select.types import (
            SpectraData,
            ExcitationData,
            WavelengthResult,
            WavelengthBand,
            GroundTruth,
        )
        assert all([SpectraData, ExcitationData, WavelengthResult,
                   WavelengthBand, GroundTruth])

    def test_import_analyzer(self):
        """Analyzer module should be importable."""
        from spectral_select import analyzer
        from spectral_select.analyzer import Analyzer
        assert Analyzer is not None

    def test_import_validation(self):
        """Validation module should be importable."""
        from spectral_select import validation
        from spectral_select.validation import (
            Validator,
            ValidationMetrics,
            load_ground_truth_from_png,
        )
        assert all([Validator, ValidationMetrics, load_ground_truth_from_png])

    def test_import_visualizer(self):
        """Visualizer module should be importable."""
        from spectral_select import visualizer
        from spectral_select.visualizer import Visualizer
        assert Visualizer is not None

    def test_import_results(self):
        """Results module should be importable."""
        from spectral_select import results
        from spectral_select.results import ResultsManager
        assert ResultsManager is not None

    def test_import_protocols(self):
        """Protocols module should be importable."""
        from spectral_select import protocols
        from spectral_select.protocols import (
            AutoencoderProtocol,
            ClassifierProtocol,
            ClusteringProtocol,
        )
        assert all([AutoencoderProtocol, ClassifierProtocol, ClusteringProtocol])

    def test_import_loader(self):
        """Loader module should be importable."""
        from spectral_select import loader
        from spectral_select.loader import DataLoader, DataLoadingError
        assert DataLoader is not None
        assert DataLoadingError is not None


class TestGUIImports:
    """Test GUI-related module imports.

    These tests verify that GUI modules can be imported even without
    a display or GUI backend. They test import-time behavior only.
    """

    def test_import_widgets(self):
        """Widgets module should be importable (Jupyter widgets)."""
        from spectral_select import widgets
        from spectral_select.widgets import ROIWidget
        assert ROIWidget is not None

    def test_import_widgets_helpers(self):
        """Widget helper functions should be importable."""
        from spectral_select.widgets import (
            create_display_image,
            path_to_mask,
        )
        assert create_display_image is not None
        assert path_to_mask is not None

    def test_import_widgets_class_colors(self):
        """Widget CLASS_COLORS constant should be accessible."""
        from spectral_select.widgets import ROIWidget
        assert hasattr(ROIWidget, "CLASS_COLORS")
        assert isinstance(ROIWidget.CLASS_COLORS, list)
        assert len(ROIWidget.CLASS_COLORS) > 0

    def test_import_viewer(self):
        """Viewer module should be importable (tkinter GUI)."""
        # This may fail on systems without tkinter, which is acceptable
        try:
            from spectral_select import viewer
            from spectral_select.viewer import ViewerApp
            assert ViewerApp is not None
        except ImportError as e:
            if "tkinter" in str(e).lower():
                pytest.skip("tkinter not available")
            raise

    def test_import_viewer_helpers(self):
        """Viewer helper functions should be importable."""
        try:
            from spectral_select.viewer import (
                create_rgb_image,
                detect_cube_format,
                launch_viewer,
            )
            assert create_rgb_image is not None
            assert detect_cube_format is not None
            assert launch_viewer is not None
        except ImportError as e:
            if "tkinter" in str(e).lower():
                pytest.skip("tkinter not available")
            raise


class TestCircularImportPrevention:
    """Test that circular imports are properly handled."""

    def test_no_circular_import_types_validation(self):
        """Types and validation should not have circular imports."""
        from spectral_select.types import GroundTruth
        from spectral_select.validation import Validator
        assert GroundTruth is not None
        assert Validator is not None

    def test_no_circular_import_analyzer_visualizer(self):
        """Analyzer and visualizer should not have circular imports."""
        from spectral_select.analyzer import Analyzer
        from spectral_select.visualizer import Visualizer
        assert Analyzer is not None
        assert Visualizer is not None

    def test_no_circular_import_results_config(self):
        """Results and config should not have circular imports."""
        from spectral_select.results import ResultsManager
        from spectral_select.config import Config
        assert ResultsManager is not None
        assert Config is not None

    def test_no_circular_import_loader_types(self):
        """Loader and types should not have circular imports."""
        from spectral_select.loader import DataLoader
        from spectral_select.types import SpectraData
        assert DataLoader is not None
        assert SpectraData is not None

    def test_no_circular_import_widgets_types(self):
        """Widgets and types should not have circular imports."""
        from spectral_select.widgets import ROIWidget
        from spectral_select.types import SpectraData, GroundTruth
        assert ROIWidget is not None
        assert SpectraData is not None
        assert GroundTruth is not None


class TestDependencyAvailability:
    """Test that required dependencies are available."""

    def test_numpy_available(self):
        """NumPy should be available."""
        import numpy as np
        assert hasattr(np, "ndarray")

    def test_torch_available(self):
        """PyTorch should be available."""
        import torch
        assert hasattr(torch, "Tensor")

    def test_sklearn_available(self):
        """Scikit-learn should be available."""
        from sklearn.metrics import adjusted_rand_score
        assert adjusted_rand_score is not None

    def test_matplotlib_available(self):
        """Matplotlib should be available."""
        import matplotlib
        matplotlib.use("Agg")  # Non-GUI backend
        import matplotlib.pyplot as plt
        assert plt is not None

    def test_pandas_available(self):
        """Pandas should be available."""
        import pandas as pd
        assert hasattr(pd, "DataFrame")

    def test_scipy_available(self):
        """SciPy should be available."""
        from scipy import ndimage
        assert ndimage is not None

    def test_pillow_available(self):
        """Pillow should be available."""
        from PIL import Image
        assert Image is not None

    def test_tifffile_available(self):
        """tifffile should be available."""
        import tifffile
        assert hasattr(tifffile, "imread")

    def test_openpyxl_available(self):
        """openpyxl should be available for Excel export."""
        import openpyxl
        assert hasattr(openpyxl, "Workbook")

    def test_pyyaml_available(self):
        """PyYAML should be available for config loading."""
        import yaml
        assert hasattr(yaml, "safe_load")


class TestOptionalDependencies:
    """Test optional dependencies that may not be available."""

    def test_ipywidgets_optional(self):
        """ipywidgets should be available for ROIWidget."""
        try:
            import ipywidgets
            assert ipywidgets is not None
        except ImportError:
            pytest.skip("ipywidgets not installed")

    def test_ipympl_optional(self):
        """ipympl should be available for interactive matplotlib."""
        try:
            import ipympl
            assert ipympl is not None
        except ImportError:
            pytest.skip("ipympl not installed")

    def test_tkinter_optional(self):
        """tkinter should be available for GUI viewer."""
        try:
            import tkinter
            assert tkinter is not None
        except ImportError:
            pytest.skip("tkinter not installed")
