"""Ground truth validation metrics for clustering evaluation.

This module provides the Validator class for evaluating clustering results
against ground truth labels using sklearn-compatible metrics.

Example:
    from spectral_select import Validator
    from spectral_select.validation import load_ground_truth_from_png

    # Load ground truth from annotated PNG
    gt = load_ground_truth_from_png("annotations.png")

    # Evaluate clustering
    validator = Validator()
    validator.fit(cluster_labels, gt)
    print(f"ARI: {validator.score():.3f}")
    print(validator.metrics.summary())
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    confusion_matrix,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from .types import GroundTruth, ValidationMetrics

logger = logging.getLogger(__name__)


class Validator:
    """Evaluate clustering results against ground truth labels.

    Follows the sklearn estimator pattern with fit() and score() methods.
    Computes comprehensive metrics including ARI, NMI, purity, and per-class
    precision/recall/F1.

    Attributes:
        is_fitted: Whether fit() has been called.
        metrics: Computed ValidationMetrics (available after fit).

    Example:
        validator = Validator()
        validator.fit(cluster_labels, ground_truth)

        # Get primary metric
        ari = validator.score()

        # Get all metrics
        m = validator.metrics
        print(f"Purity: {m.purity:.3f}, NMI: {m.normalized_mutual_info:.3f}")
    """

    def __init__(self) -> None:
        """Initialize the Validator."""
        self._metrics: Optional[ValidationMetrics] = None

    @property
    def is_fitted(self) -> bool:
        """Check if the validator has been fitted."""
        return self._metrics is not None

    @property
    def metrics(self) -> ValidationMetrics:
        """Get computed validation metrics.

        Returns:
            ValidationMetrics instance with all computed scores.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._metrics is None:
            raise RuntimeError(
                "Validator not fitted. Call fit() first with cluster labels "
                "and ground truth."
            )
        return self._metrics

    def fit(
        self,
        cluster_labels: np.ndarray,
        ground_truth: Union[np.ndarray, GroundTruth],
        valid_mask: Optional[np.ndarray] = None,
    ) -> "Validator":
        """Compute validation metrics for clustering results.

        Args:
            cluster_labels: Predicted cluster labels (1D or 2D array).
            ground_truth: Ground truth labels as array or GroundTruth object.
                         Use -1 for background pixels.
            valid_mask: Optional boolean mask for valid pixels to evaluate.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If no valid pixels to compare.
        """
        # Extract labels array from GroundTruth if needed
        if isinstance(ground_truth, GroundTruth):
            gt_labels = ground_truth.labels
        else:
            gt_labels = ground_truth

        # Flatten arrays if 2D
        cluster_flat = cluster_labels.flatten() if cluster_labels.ndim == 2 else cluster_labels
        gt_flat = gt_labels.flatten() if gt_labels.ndim == 2 else gt_labels

        # Apply valid_mask if provided
        if valid_mask is not None:
            mask_flat = valid_mask.flatten() if valid_mask.ndim == 2 else valid_mask
            cluster_flat = cluster_flat[mask_flat]
            gt_flat = gt_flat[mask_flat]

        # Filter out background pixels (-1 in either array)
        valid_indices = (cluster_flat >= 0) & (gt_flat >= 0)
        cluster_valid = cluster_flat[valid_indices]
        gt_valid = gt_flat[valid_indices]

        if len(cluster_valid) == 0:
            raise ValueError(
                "No valid pixels to compare. Check that cluster_labels and "
                "ground_truth have matching non-background regions."
            )

        # Compute sklearn metrics
        ari = float(adjusted_rand_score(gt_valid, cluster_valid))
        nmi = float(normalized_mutual_info_score(gt_valid, cluster_valid))
        ami = float(adjusted_mutual_info_score(gt_valid, cluster_valid))
        fms = float(fowlkes_mallows_score(gt_valid, cluster_valid))
        v_meas = float(v_measure_score(gt_valid, cluster_valid))
        homog = float(homogeneity_score(gt_valid, cluster_valid))
        compl = float(completeness_score(gt_valid, cluster_valid))

        # Compute purity and cluster-to-GT mapping
        purity, mapping = self._calculate_purity_and_mapping(cluster_valid, gt_valid)

        # Build confusion matrix
        conf_mat = confusion_matrix(gt_valid, cluster_valid)

        # Compute per-class metrics
        per_class = self._calculate_per_class_metrics(conf_mat)

        # Count classes and clusters
        n_gt_classes = len(np.unique(gt_valid))
        n_clusters = len(np.unique(cluster_valid))

        # Store results
        self._metrics = ValidationMetrics(
            adjusted_rand_score=ari,
            normalized_mutual_info=nmi,
            adjusted_mutual_info=ami,
            fowlkes_mallows_score=fms,
            v_measure=v_meas,
            homogeneity=homog,
            completeness=compl,
            purity=purity,
            cluster_to_gt_mapping=mapping,
            confusion_matrix=conf_mat,
            per_class_precision=per_class["precision"],
            per_class_recall=per_class["recall"],
            per_class_f1=per_class["f1"],
            n_ground_truth_classes=n_gt_classes,
            n_predicted_clusters=n_clusters,
        )

        logger.info(
            f"Validation complete: ARI={ari:.4f}, Purity={purity:.4f}, "
            f"NMI={nmi:.4f}"
        )

        return self

    def score(self) -> float:
        """Return the primary clustering score (Adjusted Rand Index).

        This follows the sklearn estimator convention where score() returns
        a single scalar metric.

        Returns:
            Adjusted Rand Index score.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        return self.metrics.adjusted_rand_score

    def compare(
        self,
        results_dict: Dict[str, np.ndarray],
        ground_truth: Union[np.ndarray, GroundTruth],
        valid_mask: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Compare multiple clustering results against ground truth.

        Evaluates each clustering result and returns a sorted DataFrame
        with all metrics for easy comparison.

        Args:
            results_dict: Mapping from method names to cluster label arrays.
            ground_truth: Ground truth labels (shared across all methods).
            valid_mask: Optional valid pixel mask (shared).

        Returns:
            DataFrame with methods as rows, metrics as columns, sorted by purity.

        Example:
            results = {
                "KMeans": kmeans_labels,
                "HDBSCAN": hdbscan_labels,
                "Spectral": spectral_labels,
            }
            df = validator.compare(results, ground_truth)
            print(df.to_string())
        """
        comparison_rows = []

        for method_name, cluster_labels in results_dict.items():
            logger.debug(f"Evaluating {method_name}...")

            # Fit validator with this method's results
            self.fit(cluster_labels, ground_truth, valid_mask)
            m = self.metrics

            row = {
                "Method": method_name,
                "N_Clusters": m.n_predicted_clusters,
                "Purity": m.purity,
                "ARI": m.adjusted_rand_score,
                "NMI": m.normalized_mutual_info,
                "AMI": m.adjusted_mutual_info,
                "V-Measure": m.v_measure,
                "Homogeneity": m.homogeneity,
                "Completeness": m.completeness,
                "FM-Score": m.fowlkes_mallows_score,
            }
            comparison_rows.append(row)

        df = pd.DataFrame(comparison_rows)
        df = df.sort_values("Purity", ascending=False).reset_index(drop=True)

        return df

    @staticmethod
    def _calculate_purity_and_mapping(
        cluster_labels: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Tuple[float, Dict[int, int]]:
        """Calculate purity score and optimal cluster-to-GT mapping.

        Purity assigns each cluster to its majority ground truth class
        and computes the fraction of correctly assigned pixels.

        Args:
            cluster_labels: 1D array of cluster assignments.
            ground_truth: 1D array of ground truth labels.

        Returns:
            Tuple of (purity_score, cluster_to_gt_mapping).
        """
        unique_clusters = np.unique(cluster_labels)
        unique_gt = np.unique(ground_truth)

        # Build contingency table
        contingency = np.zeros((len(unique_clusters), len(unique_gt)))

        for i, cluster in enumerate(unique_clusters):
            for j, gt_class in enumerate(unique_gt):
                contingency[i, j] = np.sum(
                    (cluster_labels == cluster) & (ground_truth == gt_class)
                )

        # Assign each cluster to majority class
        cluster_to_gt: Dict[int, int] = {}
        total_correct = 0

        for i, cluster in enumerate(unique_clusters):
            best_gt_idx = int(np.argmax(contingency[i, :]))
            best_gt = unique_gt[best_gt_idx]
            cluster_to_gt[int(cluster)] = int(best_gt)
            total_correct += contingency[i, best_gt_idx]

        purity = float(total_correct / len(cluster_labels))

        return purity, cluster_to_gt

    @staticmethod
    def _calculate_per_class_metrics(
        conf_matrix: np.ndarray,
    ) -> Dict[str, Dict[int, float]]:
        """Calculate per-class precision, recall, and F1 scores.

        Args:
            conf_matrix: Confusion matrix (rows=GT, cols=predicted).

        Returns:
            Dictionary with 'precision', 'recall', 'f1' keys,
            each mapping class index to score.
        """
        n_classes = conf_matrix.shape[0]

        precision: Dict[int, float] = {}
        recall: Dict[int, float] = {}
        f1: Dict[int, float] = {}

        for i in range(n_classes):
            # True positives for this class
            tp = conf_matrix[i, i] if i < conf_matrix.shape[1] else 0

            # False positives (predicted as this class but actually other)
            col_sum = np.sum(conf_matrix[:, i]) if i < conf_matrix.shape[1] else 0
            fp = col_sum - tp

            # False negatives (actually this class but predicted as other)
            fn = np.sum(conf_matrix[i, :]) - tp

            # Compute metrics with zero-division protection
            if tp + fp > 0:
                precision[i] = float(tp / (tp + fp))
            else:
                precision[i] = 0.0

            if tp + fn > 0:
                recall[i] = float(tp / (tp + fn))
            else:
                recall[i] = 0.0

            if precision[i] + recall[i] > 0:
                f1[i] = float(
                    2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
                )
            else:
                f1[i] = 0.0

        return {"precision": precision, "recall": recall, "f1": f1}


# ============================================================================
# Ground Truth Loading Utilities
# ============================================================================


def load_ground_truth_from_png(
    png_path: Union[str, Path],
    background_colors: Optional[List[Tuple[int, int, int, int]]] = None,
    target_shape: Optional[Tuple[int, int]] = None,
    class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    color_tolerance: int = 50,
) -> GroundTruth:
    """Load ground truth labels from a colored PNG annotation file.

    Extracts pixel-wise class labels from a PNG image where different
    colors represent different classes. Background colors are mapped to -1.

    Args:
        png_path: Path to the PNG annotation file.
        background_colors: List of RGBA tuples to treat as background.
            Default: [(24, 24, 24, 255), (168, 168, 168, 255)] (dark/light gray).
        target_shape: Optional (height, width) to resize/crop to.
        class_colors: Optional mapping of class names to RGB tuples for
            tolerance-based color matching. If None, each unique color
            becomes a separate class.
        color_tolerance: Maximum Euclidean distance for color matching
            when using class_colors. Default: 50.

    Returns:
        GroundTruth instance with labels, color_mapping, and class_names.

    Raises:
        FileNotFoundError: If png_path doesn't exist.
        ValueError: If the image cannot be processed.

    Example:
        # Simple usage - each color is a class
        gt = load_ground_truth_from_png("annotations.png")

        # With predefined class colors
        gt = load_ground_truth_from_png(
            "annotations.png",
            class_colors={
                "Lichen A": (255, 0, 0),
                "Lichen B": (0, 255, 0),
                "Rock": (0, 0, 255),
            },
            color_tolerance=30,
        )
    """
    png_path = Path(png_path)
    if not png_path.exists():
        raise FileNotFoundError(f"PNG file not found: {png_path}")

    # Default background colors
    if background_colors is None:
        background_colors = [
            (24, 24, 24, 255),      # Dark gray background
            (168, 168, 168, 255),   # Light gray background
        ]

    # Load and convert to RGBA
    try:
        img = Image.open(png_path)
    except Exception as e:
        raise ValueError(f"Failed to open image {png_path}: {e}")

    if img.mode != "RGBA":
        img = img.convert("RGBA")

    img_array = np.array(img)
    original_shape = img_array.shape[:2]
    logger.debug(f"Loaded PNG with shape {original_shape}")

    # Handle resizing if needed
    if target_shape is not None and original_shape != target_shape:
        img_array = _resize_or_crop_image(
            img_array, target_shape, background_colors[0]
        )
        logger.debug(f"Resized to {img_array.shape[:2]}")

    # Find unique colors
    img_flat = img_array.reshape(-1, 4)
    unique_colors = np.unique(img_flat, axis=0)
    logger.debug(f"Found {len(unique_colors)} unique colors")

    # Filter out background colors
    def is_background(color: np.ndarray) -> bool:
        for bg in background_colors:
            dist = math.sqrt(sum((int(a) - int(b)) ** 2 for a, b in zip(color[:3], bg[:3])))
            if dist <= 30:  # Background tolerance
                return True
        return False

    foreground_colors = [
        tuple(c) for c in unique_colors if not is_background(c)
    ]
    logger.debug(f"Found {len(foreground_colors)} foreground colors")

    # Initialize output arrays
    ground_truth = np.full(img_array.shape[:2], -1, dtype=np.int32)
    color_mapping: Dict[int, Tuple[int, int, int, int]] = {-1: (0, 0, 0, 0)}
    class_names: Optional[List[str]] = None

    if class_colors is not None:
        # Tolerance-based matching to predefined classes
        class_names = list(class_colors.keys())
        class_rgb = list(class_colors.values())

        for label, (name, rgb) in enumerate(class_colors.items()):
            color_mapping[label] = (*rgb, 255)
            logger.debug(f"Class {label}: {name} -> RGB{rgb}")

        # Assign each pixel to closest class (if within tolerance)
        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                pixel = img_array[y, x]

                if is_background(pixel):
                    continue

                # Find closest class
                min_dist = float("inf")
                best_class = -1
                for label, rgb in enumerate(class_rgb):
                    dist = math.sqrt(
                        sum((int(a) - int(b)) ** 2 for a, b in zip(pixel[:3], rgb))
                    )
                    if dist < min_dist:
                        min_dist = dist
                        best_class = label

                if min_dist <= color_tolerance:
                    ground_truth[y, x] = best_class
                else:
                    warnings.warn(
                        f"Pixel at ({y}, {x}) with color {tuple(pixel[:3])} "
                        f"doesn't match any class (min distance: {min_dist:.1f})"
                    )

        # Report which classes were found
        found_classes = []
        for label, name in enumerate(class_names):
            if np.sum(ground_truth == label) > 0:
                found_classes.append(name)
        logger.info(f"Classes with pixels: {found_classes}")

    else:
        # Each unique foreground color becomes a separate class
        for label, color in enumerate(foreground_colors):
            color_mapping[label] = color
            mask = np.all(img_array == color, axis=2)
            ground_truth[mask] = label

    # Log statistics
    n_classes = len([k for k in color_mapping if k >= 0])
    n_foreground = np.sum(ground_truth >= 0)
    n_background = np.sum(ground_truth == -1)
    logger.info(
        f"Ground truth: {n_classes} classes, "
        f"{n_foreground} foreground pixels, {n_background} background"
    )

    return GroundTruth(
        labels=ground_truth,
        color_mapping=color_mapping,
        class_names=class_names,
    )


def _resize_or_crop_image(
    img_array: np.ndarray,
    target_shape: Tuple[int, int],
    fill_color: Tuple[int, int, int, int],
) -> np.ndarray:
    """Resize or crop an image to target shape.

    Centers the image and pads or crops as needed.

    Args:
        img_array: Source image as numpy array (H, W, 4).
        target_shape: Desired (height, width).
        fill_color: RGBA color for padding.

    Returns:
        Resized/cropped image array.
    """
    target_h, target_w = target_shape
    current_h, current_w = img_array.shape[:2]

    # Create output filled with background
    result = np.full((target_h, target_w, 4), fill_color, dtype=np.uint8)

    h_diff = target_h - current_h
    w_diff = target_w - current_w

    if h_diff >= 0 and w_diff >= 0:
        # Padding needed
        h_pad = h_diff // 2
        w_pad = w_diff // 2
        result[h_pad:h_pad + current_h, w_pad:w_pad + current_w] = img_array

    elif h_diff >= 0 and w_diff < 0:
        # Pad height, crop width
        h_pad = h_diff // 2
        w_crop = abs(w_diff) // 2
        result[h_pad:h_pad + current_h, :] = img_array[:, w_crop:w_crop + target_w]

    elif h_diff < 0 and w_diff >= 0:
        # Crop height, pad width
        h_crop = abs(h_diff) // 2
        w_pad = w_diff // 2
        result[:, w_pad:w_pad + current_w] = img_array[h_crop:h_crop + target_h, :]

    else:
        # Crop both dimensions
        h_crop = abs(h_diff) // 2
        w_crop = abs(w_diff) // 2
        result = img_array[h_crop:h_crop + target_h, w_crop:w_crop + target_w]

    return result
