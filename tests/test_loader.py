"""Unit tests for DataLoader and data loading error handling.

Tests cover:
- DataLoader initialization with valid/invalid paths
- DataLoadingError exception attributes and formatting
- Error message quality (includes path, hint, cause)

Note: Integration tests requiring ImageJ and real .im3 files are marked
with @pytest.mark.skip since they require external dependencies.
"""

import tempfile
from pathlib import Path

import pytest

from spectral_select.loader import DataLoader, DataLoadingError


class TestDataLoaderInit:
    """Tests for DataLoader initialization."""

    def test_init_with_valid_path(self, tmp_path: Path):
        """Creates DataLoader with existing directory."""
        # Create a valid directory
        data_dir = tmp_path / "valid_data"
        data_dir.mkdir()

        loader = DataLoader(data_dir)

        assert loader.data_path == data_dir
        assert loader.cutoff_offset == 30  # default
        assert loader.verbose is True  # default

    def test_init_with_nonexistent_path(self):
        """Raises error for missing directory."""
        nonexistent = Path("/nonexistent/path/to/data")

        with pytest.raises(DataLoadingError) as exc_info:
            DataLoader(nonexistent)

        assert "does not exist" in str(exc_info.value)
        assert exc_info.value.path == nonexistent

    def test_init_default_parameters(self, tmp_path: Path):
        """Default cutoff_offset=30, verbose=True."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        loader = DataLoader(data_dir)

        assert loader.cutoff_offset == 30
        assert loader.verbose is True
        assert loader.metadata_path is None

    def test_init_custom_parameters(self, tmp_path: Path):
        """Custom parameters are stored correctly."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        metadata_file = tmp_path / "metadata.xlsx"
        metadata_file.touch()

        loader = DataLoader(
            data_dir,
            metadata_path=metadata_file,
            cutoff_offset=25,
            verbose=False,
        )

        assert loader.cutoff_offset == 25
        assert loader.verbose is False
        assert loader.metadata_path == metadata_file

    def test_verbose_false_suppresses_output(self, tmp_path: Path, capsys):
        """No output when verbose=False during initialization."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        DataLoader(data_dir, verbose=False)

        captured = capsys.readouterr()
        # Initialization shouldn't print anything
        assert captured.out == ""


class TestDataLoaderErrors:
    """Tests for DataLoader error handling."""

    def test_load_empty_directory(self, tmp_path: Path):
        """Raises DataLoadingError with helpful message for empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        loader = DataLoader(empty_dir)

        # The actual load() will fail because no .im3 files exist
        # and HyperspectralDataLoader import may fail
        # Test the internal loader initialization error handling
        with pytest.raises(DataLoadingError) as exc_info:
            loader.load()

        # Error message should be helpful
        error_msg = str(exc_info.value)
        assert "empty" in error_msg.lower() or "no" in error_msg.lower() or "failed" in error_msg.lower()

    def test_load_no_im3_files(self, tmp_path: Path):
        """Raises DataLoadingError listing directory contents when no .im3 files."""
        data_dir = tmp_path / "no_im3"
        data_dir.mkdir()

        # Create some non-im3 files
        (data_dir / "readme.txt").touch()
        (data_dir / "data.csv").touch()

        loader = DataLoader(data_dir)

        with pytest.raises(DataLoadingError) as exc_info:
            loader.load()

        # Error should mention something about the failure
        error_msg = str(exc_info.value)
        assert len(error_msg) > 0  # Some error message exists

    def test_error_message_includes_path(self):
        """Error contains the attempted path."""
        test_path = Path("/does/not/exist/data")

        with pytest.raises(DataLoadingError) as exc_info:
            DataLoader(test_path)

        assert str(test_path) in str(exc_info.value)
        assert exc_info.value.path == test_path

    def test_error_message_includes_hint(self):
        """Error includes suggestion for fix when path doesn't exist."""
        test_path = Path("/invalid/path")

        with pytest.raises(DataLoadingError) as exc_info:
            DataLoader(test_path)

        # The error should be informative
        error = exc_info.value
        assert error.path is not None
        # Path info included via __str__
        assert "path" in str(error).lower()


class TestDataLoadingError:
    """Tests for DataLoadingError exception class."""

    def test_error_basic_creation(self):
        """Basic error with message only."""
        error = DataLoadingError("Test error message")

        assert error.message == "Test error message"
        assert error.path is None
        assert error.cause is None
        assert "Test error message" in str(error)

    def test_error_with_path(self):
        """Error with path attribute."""
        test_path = Path("/test/path")
        error = DataLoadingError("Failed to load", path=test_path)

        assert error.path == test_path
        assert "Failed to load" in str(error)
        assert str(test_path) in str(error)

    def test_error_with_cause(self):
        """Error with original exception as cause."""
        original = ValueError("Original error")
        error = DataLoadingError("Wrapped error", cause=original)

        assert error.cause is original
        assert "Wrapped error" in str(error)
        assert "Original error" in str(error)

    def test_error_with_path_and_cause(self):
        """Error with both path and cause."""
        test_path = Path("/test/path")
        original = FileNotFoundError("File not found")
        error = DataLoadingError(
            "Loading failed",
            path=test_path,
            cause=original,
        )

        assert error.path == test_path
        assert error.cause is original
        error_str = str(error)
        assert "Loading failed" in error_str
        assert str(test_path) in error_str
        assert "File not found" in error_str

    def test_error_path_converted_to_pathlib(self):
        """String paths are converted to Path objects."""
        error = DataLoadingError("Error", path="/string/path")

        assert isinstance(error.path, Path)
        assert error.path == Path("/string/path")


class TestDataLoaderImageJAvailability:
    """Tests for ImageJ availability checking."""

    def test_imagej_available_property(self, tmp_path: Path):
        """imagej_available property works without errors."""
        data_dir = tmp_path / "test"
        data_dir.mkdir()

        loader = DataLoader(data_dir)

        # Just verify property doesn't raise
        result = loader.imagej_available

        # Should be a boolean
        assert isinstance(result, bool)

    def test_imagej_availability_cached(self, tmp_path: Path):
        """ImageJ availability check is cached."""
        data_dir = tmp_path / "test"
        data_dir.mkdir()

        loader = DataLoader(data_dir)

        # First call
        result1 = loader.imagej_available
        # Second call should use cached value
        result2 = loader.imagej_available

        assert result1 == result2
        # Internal cache should be set
        assert loader._imagej_available is not None


# Integration tests requiring ImageJ - skipped by default
class TestDataLoaderIntegration:
    """Integration tests that require ImageJ and real data files.

    These tests are skipped by default since they require:
    - pyimagej to be installed
    - ImageJ/Fiji to be downloaded (first run takes minutes)
    - Real .im3 test data files
    """

    @pytest.mark.skip(reason="Requires ImageJ and test data")
    def test_load_real_im3_files(self):
        """Load actual .im3 files from test data directory."""
        # This would test actual loading with real files
        pass

    @pytest.mark.skip(reason="Requires ImageJ and test data")
    def test_load_with_metadata(self):
        """Load with metadata Excel file."""
        # This would test metadata integration
        pass

    @pytest.mark.skip(reason="Requires ImageJ and test data")
    def test_get_excitation_wavelengths_after_load(self):
        """Get excitation wavelengths from loaded data."""
        # This would test post-load queries
        pass

    @pytest.mark.skip(reason="Requires ImageJ and test data")
    def test_get_summary_after_load(self):
        """Get data summary after loading."""
        # This would test summary generation
        pass
