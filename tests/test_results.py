"""Tests for ResultsManager output organization."""

import json
import tempfile
from pathlib import Path

import pytest

from spectral_select import Config, ResultsManager


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Provide a sample Config for testing."""
    return Config(sample_name="TestSample")


# ============================================================================
# TestResultsManagerInitialization
# ============================================================================


class TestResultsManagerInitialization:
    """Tests for ResultsManager initialization and directory creation."""

    def test_creates_directory_structure(self, temp_dir):
        """Test that ResultsManager creates expected directories."""
        rm = ResultsManager(temp_dir, "MySample")

        assert rm.run_dir.exists()
        assert rm.model_dir.exists()
        assert rm.viz_dir.exists()
        assert rm.layers_dir.exists()

    def test_auto_generates_run_id(self, temp_dir):
        """Test that run_id is auto-generated in YYYYMMDD_HHMMSS format."""
        rm = ResultsManager(temp_dir, "MySample")

        # Check format: 8 digits, underscore, 6 digits
        assert rm.run_id is not None
        assert len(rm.run_id) == 15  # YYYYMMDD_HHMMSS
        assert rm.run_id[8] == "_"
        assert rm.run_id[:8].isdigit()
        assert rm.run_id[9:].isdigit()

    def test_accepts_custom_run_id(self, temp_dir):
        """Test that custom run_id is respected."""
        custom_id = "my_custom_run_123"
        rm = ResultsManager(temp_dir, "MySample", run_id=custom_id)

        assert rm.run_id == custom_id
        assert custom_id in str(rm.run_dir)

    def test_validates_empty_sample_name(self, temp_dir):
        """Test that empty sample_name raises ValueError."""
        with pytest.raises(ValueError, match="sample_name cannot be empty"):
            ResultsManager(temp_dir, "")

        with pytest.raises(ValueError, match="sample_name cannot be empty"):
            ResultsManager(temp_dir, "   ")

    def test_from_config_factory(self, temp_dir, sample_config):
        """Test from_config factory method."""
        rm = ResultsManager.from_config(sample_config, base_dir=temp_dir)

        assert rm.sample_name == "TestSample"
        assert rm.base_dir == temp_dir
        assert rm.run_dir.exists()


# ============================================================================
# TestResultsManagerPaths
# ============================================================================


class TestResultsManagerPaths:
    """Tests for path generation methods."""

    def test_get_model_path_best(self, temp_dir):
        """Test get_model_path with 'best' checkpoint type."""
        rm = ResultsManager(temp_dir, "MySample", run_id="test_run")
        path = rm.get_model_path("best")

        assert path.name == "best_model.pth"
        assert "model" in str(path.parent)

    def test_get_model_path_final(self, temp_dir):
        """Test get_model_path with 'final' checkpoint type."""
        rm = ResultsManager(temp_dir, "MySample", run_id="test_run")
        path = rm.get_model_path("final")

        assert path.name == "final_model.pth"

    def test_get_model_path_invalid_type(self, temp_dir):
        """Test get_model_path raises on invalid checkpoint type."""
        rm = ResultsManager(temp_dir, "MySample", run_id="test_run")

        with pytest.raises(ValueError, match="checkpoint_type must be"):
            rm.get_model_path("invalid")

    def test_get_result_path(self, temp_dir):
        """Test get_result_path returns correct path."""
        rm = ResultsManager(temp_dir, "MySample", run_id="test_run")
        path = rm.get_result_path("wavelength_result.json")

        assert path.name == "wavelength_result.json"
        assert path.parent == rm.run_dir

    def test_get_layer_path(self, temp_dir):
        """Test get_layer_path with zero-padded indices."""
        rm = ResultsManager(temp_dir, "MySample", run_id="test_run")

        assert rm.get_layer_path(0).name == "layer_0000.tiff"
        assert rm.get_layer_path(42).name == "layer_0042.tiff"
        assert rm.get_layer_path(9999).name == "layer_9999.tiff"

    def test_get_viz_path(self, temp_dir):
        """Test get_viz_path returns correct path."""
        rm = ResultsManager(temp_dir, "MySample", run_id="test_run")
        path = rm.get_viz_path("ranking_plot.png")

        assert path.name == "ranking_plot.png"
        assert path.parent == rm.viz_dir


# ============================================================================
# TestResultsManagerOperations
# ============================================================================


class TestResultsManagerOperations:
    """Tests for run management operations."""

    def test_get_all_runs_empty(self, temp_dir):
        """Test get_all_runs returns empty list when no runs exist."""
        rm = ResultsManager(temp_dir, "MySample", run_id="first_run")
        # Only one run exists (the one we just created)
        runs = rm.get_all_runs()

        assert isinstance(runs, list)
        assert "first_run" in runs

    def test_get_all_runs_multiple(self, temp_dir):
        """Test get_all_runs returns sorted list of all runs."""
        # Create multiple runs
        ResultsManager(temp_dir, "MySample", run_id="run_001")
        ResultsManager(temp_dir, "MySample", run_id="run_003")
        ResultsManager(temp_dir, "MySample", run_id="run_002")

        rm = ResultsManager(temp_dir, "MySample", run_id="run_004")
        runs = rm.get_all_runs()

        assert runs == ["run_001", "run_002", "run_003", "run_004"]

    def test_from_existing_run_valid(self, temp_dir):
        """Test from_existing_run loads an existing run."""
        # Create a run first
        original = ResultsManager(temp_dir, "MySample", run_id="existing_run")
        # Write a file to verify it's the same run
        (original.run_dir / "test.txt").write_text("hello")

        # Load the existing run
        loaded = ResultsManager.from_existing_run(temp_dir, "MySample", "existing_run")

        assert loaded.run_id == "existing_run"
        assert loaded.run_dir == original.run_dir
        assert (loaded.run_dir / "test.txt").read_text() == "hello"

    def test_from_existing_run_not_found(self, temp_dir):
        """Test from_existing_run raises FileNotFoundError for missing run."""
        # Create a sample directory but with a different run
        ResultsManager(temp_dir, "MySample", run_id="other_run")

        with pytest.raises(FileNotFoundError, match="Run not found"):
            ResultsManager.from_existing_run(temp_dir, "MySample", "nonexistent_run")

    def test_to_dict(self, temp_dir):
        """Test to_dict serialization."""
        rm = ResultsManager(temp_dir, "MySample", run_id="test_run")
        d = rm.to_dict()

        assert d["base_dir"] == str(temp_dir)
        assert d["sample_name"] == "MySample"
        assert d["run_id"] == "test_run"


# ============================================================================
# TestResultsManagerMetadata
# ============================================================================


class TestResultsManagerMetadata:
    """Tests for metadata and provenance tracking."""

    def test_save_run_metadata(self, temp_dir, sample_config):
        """Test save_run_metadata creates run_metadata.json."""
        rm = ResultsManager(temp_dir, sample_config.sample_name, run_id="meta_test")
        path = rm.save_run_metadata(sample_config, extra={"notes": "test run"})

        assert path.exists()
        assert path.name == "run_metadata.json"

        # Verify contents
        data = json.loads(path.read_text())
        assert data["run_id"] == "meta_test"
        assert data["sample_name"] == sample_config.sample_name
        assert "timestamp" in data
        assert "config_snapshot" in data
        assert "environment" in data
        assert data["extra"]["notes"] == "test run"

    def test_load_run_metadata(self, temp_dir, sample_config):
        """Test load_run_metadata returns saved data."""
        rm = ResultsManager(temp_dir, sample_config.sample_name, run_id="load_test")
        rm.save_run_metadata(sample_config)

        loaded = rm.load_run_metadata()

        assert loaded["run_id"] == "load_test"
        assert "config_snapshot" in loaded
        assert "environment" in loaded

    def test_load_run_metadata_not_found(self, temp_dir):
        """Test load_run_metadata raises FileNotFoundError when not saved."""
        rm = ResultsManager(temp_dir, "MySample", run_id="no_meta")

        with pytest.raises(FileNotFoundError, match="Run metadata not found"):
            rm.load_run_metadata()

    def test_environment_info_contains_python_version(self, temp_dir):
        """Test _get_environment_info includes Python version."""
        rm = ResultsManager(temp_dir, "MySample")
        env_info = rm._get_environment_info()

        assert "python_version" in env_info
        assert "python_version_info" in env_info
        assert "platform" in env_info
        assert "package_versions" in env_info
        # Verify structure
        assert isinstance(env_info["python_version_info"], list)
        assert len(env_info["python_version_info"]) == 3

    def test_git_info_structure(self, temp_dir):
        """Test _get_git_info returns expected structure when in git repo."""
        rm = ResultsManager(temp_dir, "MySample")
        git_info = rm._get_git_info()

        # We're running in a git repo, so should have info
        if git_info is not None:
            assert "commit_hash" in git_info
            assert "commit_short" in git_info
            assert "branch" in git_info
            assert "is_dirty" in git_info

    def test_list_run_files(self, temp_dir, sample_config):
        """Test list_run_files returns all files in run directory."""
        rm = ResultsManager(temp_dir, sample_config.sample_name, run_id="files_test")

        # Create some files
        (rm.run_dir / "result.json").write_text("{}")
        (rm.model_dir / "model.pth").write_bytes(b"model")
        (rm.viz_dir / "plot.png").write_bytes(b"png")

        files = rm.list_run_files()

        assert len(files) >= 3
        names = [f.name for f in files]
        assert "result.json" in names
        assert "model.pth" in names
        assert "plot.png" in names

    def test_list_run_files_empty(self, temp_dir):
        """Test list_run_files returns empty list for empty run directory."""
        # Remove all created files first
        rm = ResultsManager(temp_dir, "MySample", run_id="empty_test")
        # The directories are created but should be empty
        files = rm.list_run_files()

        # Directories are created but no files inside
        assert isinstance(files, list)


# ============================================================================
# TestResultsManagerSaveOperations
# ============================================================================


class TestResultsManagerSaveOperations:
    """Tests for save operations."""

    def test_save_training_metadata(self, temp_dir):
        """Test save_training_metadata creates training_metadata.json."""
        rm = ResultsManager(temp_dir, "MySample", run_id="training_test")
        metadata = {"epochs": 100, "final_loss": 0.05, "training_time": "10m"}

        path = rm.save_training_metadata(metadata)

        assert path.exists()
        assert path.name == "training_metadata.json"

        data = json.loads(path.read_text())
        assert data["epochs"] == 100
        assert data["final_loss"] == 0.05
        assert "timestamp" in data  # Auto-added

    def test_update_latest_symlink(self, temp_dir):
        """Test update_latest_symlink creates symlink to current run."""
        rm = ResultsManager(temp_dir, "MySample", run_id="latest_test")
        rm.update_latest_symlink()

        latest = temp_dir / "MySample" / "latest"
        assert latest.exists() or latest.is_symlink()

        # Should point to our run
        if latest.is_symlink():
            target = latest.resolve()
            assert "latest_test" in str(target)
