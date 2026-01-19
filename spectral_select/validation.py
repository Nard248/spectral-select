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
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
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
