"""Results management for structured output organization.

This module provides the ResultsManager class that handles output directory
structure, run tracking, and artifact naming conventions for analysis runs.

Example:
    from spectral_select import ResultsManager, Config
    from pathlib import Path

    # Create from config
    config = Config(sample_name="Lichens_2")
    results = ResultsManager.from_config(config)

    # Or create directly
    results = ResultsManager(Path("results"), "Lichens_2")

    # Get paths for artifacts
    model_path = results.get_model_path("best")
    result_path = results.get_result_path("wavelength_result.json")

Directory structure created:
    results/
      {sample_name}/
        runs/
          {run_id}/
            config.json
            wavelength_result.json
            model/
              best_model.pth
              final_model.pth
              training_metadata.json
            visualizations/
            layers/
        latest -> runs/{most_recent_run_id}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .config import Config


@dataclass
class ResultsManager:
    """Manages structured output organization for analysis runs.

    This class creates and manages a hierarchical directory structure for
    organizing analysis outputs, including model checkpoints, results,
    visualizations, and intermediate layers.

    Attributes:
        base_dir: Root directory for all results (e.g., "results/").
        sample_name: Identifier for the sample being analyzed.
        run_id: Unique identifier for this run. Auto-generated if None.
        run_dir: Path to the current run directory.
        model_dir: Path to model checkpoint directory.
        viz_dir: Path to visualization output directory.
        layers_dir: Path to TIFF layer output directory.
    """

    base_dir: Path
    sample_name: str
    run_id: Optional[str] = None

    # Computed paths (set in __post_init__)
    run_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    viz_dir: Path = field(init=False)
    layers_dir: Path = field(init=False)

    # Internal flag to skip directory creation for existing runs
    _create_dirs: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        """Validate inputs and set up directory paths."""
        # Validate sample_name
        if not self.sample_name or not self.sample_name.strip():
            raise ValueError("sample_name cannot be empty or whitespace-only")

        # Ensure base_dir is a Path
        if not isinstance(self.base_dir, Path):
            object.__setattr__(self, "base_dir", Path(self.base_dir))

        # Generate run_id if not provided
        if self.run_id is None:
            object.__setattr__(self, "run_id", self._generate_run_id())

        # Set up directory paths
        sample_dir = self.base_dir / self.sample_name
        object.__setattr__(self, "run_dir", sample_dir / "runs" / self.run_id)
        object.__setattr__(self, "model_dir", self.run_dir / "model")
        object.__setattr__(self, "viz_dir", self.run_dir / "visualizations")
        object.__setattr__(self, "layers_dir", self.run_dir / "layers")

        # Create directories if this is a new run
        if self._create_dirs:
            self.ensure_directories()

    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run ID based on current timestamp.

        Returns:
            Run ID in format YYYYMMDD_HHMMSS (e.g., "20260120_143052").
        """
        now = datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")

    def ensure_directories(self) -> None:
        """Create all required directories for this run."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.layers_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, checkpoint_type: str = "final") -> Path:
        """Get path for a model checkpoint file.

        Args:
            checkpoint_type: Type of checkpoint - "best" or "final".

        Returns:
            Path to the model checkpoint file.

        Raises:
            ValueError: If checkpoint_type is not "best" or "final".
        """
        if checkpoint_type not in ("best", "final"):
            raise ValueError(
                f"checkpoint_type must be 'best' or 'final', got '{checkpoint_type}'"
            )
        return self.model_dir / f"{checkpoint_type}_model.pth"

    def get_result_path(self, filename: str) -> Path:
        """Get path for a result file in the run directory.

        Args:
            filename: Name of the result file (e.g., "wavelength_result.json").

        Returns:
            Full path to the result file.
        """
        return self.run_dir / filename

    def get_layer_path(self, layer_index: int) -> Path:
        """Get path for a TIFF layer file.

        Args:
            layer_index: Zero-based index of the layer.

        Returns:
            Path to the layer TIFF file (e.g., "layers/layer_0001.tiff").
        """
        return self.layers_dir / f"layer_{layer_index:04d}.tiff"

    def get_viz_path(self, filename: str) -> Path:
        """Get path for a visualization file.

        Args:
            filename: Name of the visualization file (e.g., "ranking_plot.png").

        Returns:
            Full path to the visualization file.
        """
        return self.viz_dir / filename

    def update_latest_symlink(self) -> None:
        """Create or update the 'latest' symlink to point to current run.

        On Windows, this creates a directory junction instead of a symlink
        if running without admin privileges.
        """
        sample_dir = self.base_dir / self.sample_name
        latest_link = sample_dir / "latest"

        # Remove existing symlink/junction if it exists
        if latest_link.exists() or latest_link.is_symlink():
            if latest_link.is_symlink():
                latest_link.unlink()
            elif os.name == "nt":
                # Windows junction - remove as directory
                try:
                    os.rmdir(latest_link)
                except OSError:
                    import shutil

                    shutil.rmtree(latest_link)

        # Create relative symlink (runs/{run_id})
        relative_target = Path("runs") / self.run_id

        try:
            latest_link.symlink_to(relative_target)
        except OSError:
            # On Windows without admin, try creating a junction
            if os.name == "nt":
                import subprocess

                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(latest_link), str(self.run_dir)],
                    check=True,
                    capture_output=True,
                )
            else:
                raise

    def save_model_checkpoint(
        self,
        model: Any,
        checkpoint_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a model checkpoint to disk.

        Args:
            model: PyTorch model or state dict to save.
            checkpoint_type: Type of checkpoint - "best" or "final".
            metadata: Optional metadata to save alongside the checkpoint.

        Returns:
            Path to the saved checkpoint file.

        Note:
            torch is imported only when this method is called to avoid
            dependency issues if torch is not installed.
        """
        import torch

        checkpoint_path = self.get_model_path(checkpoint_type)

        # Save model state dict if it's a model, otherwise save as-is
        if hasattr(model, "state_dict"):
            torch.save(model.state_dict(), checkpoint_path)
        else:
            torch.save(model, checkpoint_path)

        # Save metadata if provided
        if metadata is not None:
            metadata_path = self.model_dir / f"{checkpoint_type}_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)

        return checkpoint_path

    def save_training_metadata(self, metadata: Dict[str, Any]) -> Path:
        """Save training metadata to disk.

        Args:
            metadata: Dictionary containing training information.
                Recommended keys: epochs, final_loss, training_time,
                config_snapshot, timestamp.

        Returns:
            Path to the saved metadata file.
        """
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()

        metadata_path = self.model_dir / "training_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        return metadata_path

    def get_all_runs(self) -> List[str]:
        """Get all run IDs for this sample.

        Returns:
            Sorted list of run IDs (oldest to newest).
        """
        runs_dir = self.base_dir / self.sample_name / "runs"
        if not runs_dir.exists():
            return []

        runs = [
            d.name
            for d in runs_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        return sorted(runs)

    @classmethod
    def from_config(
        cls,
        config: "Config",
        base_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> "ResultsManager":
        """Create ResultsManager from a Config object.

        Args:
            config: Configuration object with sample_name.
            base_dir: Override base directory. If None, uses config.output_dir
                or defaults to "results/".
            run_id: Override run ID. If None, auto-generates.

        Returns:
            New ResultsManager instance.
        """
        if base_dir is None:
            if config.output_dir is not None:
                base_dir = config.output_dir
            else:
                base_dir = Path("results")

        return cls(
            base_dir=base_dir,
            sample_name=config.sample_name,
            run_id=run_id,
        )

    @classmethod
    def from_existing_run(
        cls,
        base_dir: Path,
        sample_name: str,
        run_id: str,
    ) -> "ResultsManager":
        """Load ResultsManager for an existing run.

        Args:
            base_dir: Root directory for results.
            sample_name: Sample identifier.
            run_id: Existing run ID to load.

        Returns:
            ResultsManager pointing to the existing run.

        Raises:
            FileNotFoundError: If the run directory doesn't exist.
        """
        # Check if run exists
        run_dir = Path(base_dir) / sample_name / "runs" / run_id
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Run not found: {run_dir}. "
                f"Available runs: {cls._list_runs(base_dir, sample_name)}"
            )

        # Create instance without creating directories
        instance = cls.__new__(cls)
        object.__setattr__(instance, "base_dir", Path(base_dir))
        object.__setattr__(instance, "sample_name", sample_name)
        object.__setattr__(instance, "run_id", run_id)
        object.__setattr__(instance, "_create_dirs", False)

        # Set up paths
        object.__setattr__(instance, "run_dir", run_dir)
        object.__setattr__(instance, "model_dir", run_dir / "model")
        object.__setattr__(instance, "viz_dir", run_dir / "visualizations")
        object.__setattr__(instance, "layers_dir", run_dir / "layers")

        return instance

    @staticmethod
    def _list_runs(base_dir: Path, sample_name: str) -> List[str]:
        """List available runs for a sample (helper for error messages)."""
        runs_dir = Path(base_dir) / sample_name / "runs"
        if not runs_dir.exists():
            return []
        return sorted(
            d.name for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with base_dir, sample_name, and run_id.
        """
        return {
            "base_dir": str(self.base_dir),
            "sample_name": self.sample_name,
            "run_id": self.run_id,
        }

    def __repr__(self) -> str:
        """Return readable string representation."""
        return (
            f"ResultsManager(\n"
            f"  base_dir={self.base_dir!r},\n"
            f"  sample_name={self.sample_name!r},\n"
            f"  run_id={self.run_id!r},\n"
            f"  run_dir={self.run_dir!r}\n"
            f")"
        )
