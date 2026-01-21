"""Protocol interfaces for pluggable components.

This module defines the interfaces that custom implementations must satisfy.
Use these protocols to create interchangeable components for classification,
clustering, autoencoder architectures, and wavelength ranking.

Example:
    from spectral_select.protocols import ClassifierProtocol

    class MyClassifier:
        def fit(self, X: np.ndarray, y: np.ndarray) -> "MyClassifier":
            ...
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            ...

    # MyClassifier is compatible with ClassifierProtocol
    assert isinstance(MyClassifier(), ClassifierProtocol)
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class ClassifierProtocol(Protocol):
    """Protocol for classification components.

    Classifiers should follow the scikit-learn fit/predict pattern.
    The fit method trains the classifier, and predict returns predictions.

    Example implementations: KNN, SVM, Random Forest, neural classifiers.
    """

    def fit(self, X: NDArray[np.floating[Any]], y: NDArray[np.integer[Any]]) -> "ClassifierProtocol":
        """Fit the classifier to training data.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        ...

    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.integer[Any]]:
        """Predict labels for samples.

        Args:
            X: Samples to predict, shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        ...


@runtime_checkable
class ClusteringProtocol(Protocol):
    """Protocol for clustering components.

    Clustering algorithms combine fit and predict into a single operation.

    Example implementations: K-means, DBSCAN, hierarchical, spectral clustering.
    """

    def fit_predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.integer[Any]]:
        """Fit clustering and return cluster labels.

        Args:
            X: Data to cluster, shape (n_samples, n_features).

        Returns:
            Cluster labels for each sample, shape (n_samples,).
        """
        ...


@runtime_checkable
class AutoencoderProtocol(Protocol):
    """Protocol for autoencoder architecture components.

    Autoencoders must provide encode and decode operations for
    working with latent space representations.

    Example implementations: Standard AE, Variational AE, Convolutional AE.
    """

    def encode(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Encode input data to latent space.

        Args:
            X: Input data of shape (n_samples, ...).

        Returns:
            Latent representations of shape (n_samples, latent_dim).
        """
        ...

    def decode(self, Z: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Decode latent representations to original space.

        Args:
            Z: Latent vectors of shape (n_samples, latent_dim).

        Returns:
            Reconstructed data of shape (n_samples, ...).
        """
        ...


@runtime_checkable
class WavelengthRankerProtocol(Protocol):
    """Protocol for wavelength ranking components.

    Rankers analyze the autoencoder model to rank wavelengths by importance.

    Example implementations: Perturbation-based, gradient-based, attention-based.
    """

    def rank(
        self,
        data: NDArray[np.floating[Any]],
        model: Any,
    ) -> List[Dict[str, Any]]:
        """Rank wavelengths by importance.

        Args:
            data: Spectral data to analyze.
            model: Trained autoencoder model.

        Returns:
            List of dicts with wavelength ranking information.
            Each dict should contain at minimum:
            - 'wavelength': float - the wavelength value
            - 'importance': float - importance score
            Additional fields may be included by implementations.
        """
        ...
