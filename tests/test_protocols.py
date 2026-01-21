"""Unit tests for protocol compliance.

Tests verify the runtime_checkable decorator works correctly for
protocol-based extensibility, allowing custom implementations to be
validated at runtime via isinstance() checks.

Tests cover:
- ClassifierProtocol is @runtime_checkable
- ClusteringProtocol is @runtime_checkable
- AutoencoderProtocol is @runtime_checkable
- WavelengthRankerProtocol is @runtime_checkable
- Custom class with fit/predict implements ClassifierProtocol
"""

from typing import Any, Dict, List

import numpy as np
import pytest
from numpy.typing import NDArray

from spectral_select.protocols import (
    AutoencoderProtocol,
    ClassifierProtocol,
    ClusteringProtocol,
    WavelengthRankerProtocol,
)


class TestClassifierProtocol:
    """Tests for ClassifierProtocol."""

    def test_classifier_protocol_runtime_checkable(self):
        """ClassifierProtocol is @runtime_checkable."""
        # The protocol should be usable with isinstance()
        class MockClassifier:
            def fit(
                self, X: NDArray[np.floating[Any]], y: NDArray[np.integer[Any]]
            ) -> "MockClassifier":
                return self

            def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.integer[Any]]:
                return np.zeros(len(X), dtype=np.int64)

        classifier = MockClassifier()

        # isinstance check works due to @runtime_checkable
        assert isinstance(classifier, ClassifierProtocol)

    def test_custom_classifier_implements_protocol(self):
        """Custom class with fit/predict implements ClassifierProtocol."""
        class CustomKNNClassifier:
            """A minimal KNN-like classifier for testing."""

            def __init__(self, n_neighbors: int = 3):
                self.n_neighbors = n_neighbors
                self.X_train: np.ndarray | None = None
                self.y_train: np.ndarray | None = None

            def fit(
                self, X: NDArray[np.floating[Any]], y: NDArray[np.integer[Any]]
            ) -> "CustomKNNClassifier":
                self.X_train = X
                self.y_train = y
                return self

            def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.integer[Any]]:
                # Simplified: just return most common label
                if self.y_train is None:
                    raise ValueError("Classifier not fitted")
                mode = int(np.bincount(self.y_train).argmax())
                return np.full(len(X), mode, dtype=np.int64)

        classifier = CustomKNNClassifier()

        # Should satisfy protocol
        assert isinstance(classifier, ClassifierProtocol)

        # Verify it actually works
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_train = np.array([0, 1, 0])
        X_test = np.array([[2.0, 3.0]])

        fitted = classifier.fit(X_train, y_train)
        predictions = fitted.predict(X_test)

        assert isinstance(fitted, CustomKNNClassifier)
        assert len(predictions) == 1
        assert predictions[0] == 0  # Most common in y_train

    def test_classifier_protocol_negative_case(self):
        """Class without fit/predict does NOT implement ClassifierProtocol."""
        class NotAClassifier:
            def train(self, X: np.ndarray) -> None:
                pass

        obj = NotAClassifier()
        assert not isinstance(obj, ClassifierProtocol)


class TestClusteringProtocol:
    """Tests for ClusteringProtocol."""

    def test_clustering_protocol_runtime_checkable(self):
        """ClusteringProtocol is @runtime_checkable."""
        class MockClustering:
            def fit_predict(
                self, X: NDArray[np.floating[Any]]
            ) -> NDArray[np.integer[Any]]:
                # Simple mock: assign all to cluster 0
                return np.zeros(len(X), dtype=np.int64)

        clustering = MockClustering()

        assert isinstance(clustering, ClusteringProtocol)

    def test_custom_clustering_implements_protocol(self):
        """Custom clustering class implements ClusteringProtocol."""
        class SimpleBinaryClustering:
            """Split data by mean of first feature."""

            def fit_predict(
                self, X: NDArray[np.floating[Any]]
            ) -> NDArray[np.integer[Any]]:
                mean_val = np.mean(X[:, 0])
                labels = (X[:, 0] > mean_val).astype(np.int64)
                return labels

        clustering = SimpleBinaryClustering()

        assert isinstance(clustering, ClusteringProtocol)

        # Verify functionality
        X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        labels = clustering.fit_predict(X)

        assert len(labels) == 4
        assert labels[0] == 0  # Below mean (2.5)
        assert labels[3] == 1  # Above mean

    def test_clustering_protocol_negative_case(self):
        """Class without fit_predict does NOT implement ClusteringProtocol."""
        class NotClustering:
            def cluster(self, X: np.ndarray) -> np.ndarray:
                return np.zeros(len(X))

        obj = NotClustering()
        assert not isinstance(obj, ClusteringProtocol)


class TestAutoencoderProtocol:
    """Tests for AutoencoderProtocol."""

    def test_autoencoder_protocol_runtime_checkable(self):
        """AutoencoderProtocol is @runtime_checkable."""
        class MockAutoencoder:
            def encode(
                self, X: NDArray[np.floating[Any]]
            ) -> NDArray[np.floating[Any]]:
                # Reduce to half dimensions
                return X[:, : X.shape[1] // 2]

            def decode(
                self, Z: NDArray[np.floating[Any]]
            ) -> NDArray[np.floating[Any]]:
                # Expand back (just repeat)
                return np.concatenate([Z, Z], axis=1)

        autoencoder = MockAutoencoder()

        assert isinstance(autoencoder, AutoencoderProtocol)

    def test_custom_autoencoder_implements_protocol(self):
        """Custom autoencoder class implements AutoencoderProtocol."""
        class LinearAutoencoder:
            """Simple linear projection autoencoder for testing."""

            def __init__(self, input_dim: int, latent_dim: int):
                np.random.seed(42)
                self.encoder_weights = np.random.randn(input_dim, latent_dim) * 0.1
                self.decoder_weights = np.random.randn(latent_dim, input_dim) * 0.1

            def encode(
                self, X: NDArray[np.floating[Any]]
            ) -> NDArray[np.floating[Any]]:
                return X @ self.encoder_weights

            def decode(
                self, Z: NDArray[np.floating[Any]]
            ) -> NDArray[np.floating[Any]]:
                return Z @ self.decoder_weights

        autoencoder = LinearAutoencoder(input_dim=10, latent_dim=3)

        assert isinstance(autoencoder, AutoencoderProtocol)

        # Verify functionality
        X = np.random.rand(5, 10)
        Z = autoencoder.encode(X)
        X_reconstructed = autoencoder.decode(Z)

        assert Z.shape == (5, 3)  # Latent dim
        assert X_reconstructed.shape == (5, 10)  # Original dim

    def test_autoencoder_protocol_negative_case(self):
        """Class without encode/decode does NOT implement AutoencoderProtocol."""
        class NotAutoencoder:
            def compress(self, X: np.ndarray) -> np.ndarray:
                return X

            def decompress(self, Z: np.ndarray) -> np.ndarray:
                return Z

        obj = NotAutoencoder()
        assert not isinstance(obj, AutoencoderProtocol)


class TestWavelengthRankerProtocol:
    """Tests for WavelengthRankerProtocol."""

    def test_wavelength_ranker_protocol_runtime_checkable(self):
        """WavelengthRankerProtocol is @runtime_checkable."""
        class MockRanker:
            def rank(
                self,
                data: NDArray[np.floating[Any]],
                model: Any,
            ) -> List[Dict[str, Any]]:
                # Return dummy rankings
                n_wavelengths = data.shape[-1] if data.ndim > 1 else 5
                return [
                    {"wavelength": float(i), "importance": 1.0 / (i + 1)}
                    for i in range(n_wavelengths)
                ]

        ranker = MockRanker()

        assert isinstance(ranker, WavelengthRankerProtocol)

    def test_custom_ranker_implements_protocol(self):
        """Custom wavelength ranker implements WavelengthRankerProtocol."""
        class VarianceBasedRanker:
            """Rank wavelengths by variance."""

            def rank(
                self,
                data: NDArray[np.floating[Any]],
                model: Any,  # Not used in this simple ranker
            ) -> List[Dict[str, Any]]:
                # Compute variance across samples for each wavelength
                if data.ndim == 2:
                    # Assume (samples, wavelengths)
                    variances = np.var(data, axis=0)
                else:
                    # Flatten and compute
                    variances = np.var(data.reshape(-1, data.shape[-1]), axis=0)

                # Create ranking by variance (descending)
                indices = np.argsort(variances)[::-1]
                rankings = []
                for rank_idx, wavelength_idx in enumerate(indices):
                    rankings.append({
                        "wavelength": float(wavelength_idx),
                        "importance": float(variances[wavelength_idx]),
                        "rank": rank_idx + 1,
                    })

                return rankings

        ranker = VarianceBasedRanker()

        assert isinstance(ranker, WavelengthRankerProtocol)

        # Verify functionality
        data = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 4.0, 3.0],
            [1.0, 6.0, 3.0],  # Column 1 has highest variance
        ])
        rankings = ranker.rank(data, model=None)

        assert len(rankings) == 3
        assert rankings[0]["wavelength"] == 1.0  # Highest variance column
        assert "importance" in rankings[0]

    def test_wavelength_ranker_protocol_negative_case(self):
        """Class without rank method does NOT implement WavelengthRankerProtocol."""
        class NotARanker:
            def score_wavelengths(
                self, data: np.ndarray, model: Any
            ) -> Dict[int, float]:
                return {}

        obj = NotARanker()
        assert not isinstance(obj, WavelengthRankerProtocol)


class TestProtocolCombinations:
    """Tests for classes implementing multiple protocols."""

    def test_class_can_implement_multiple_protocols(self):
        """A single class can implement multiple protocols."""
        class MultiPurposeModel:
            """A model that can classify and cluster."""

            def __init__(self):
                self.X_train: np.ndarray | None = None
                self.y_train: np.ndarray | None = None

            # ClassifierProtocol methods
            def fit(
                self, X: NDArray[np.floating[Any]], y: NDArray[np.integer[Any]]
            ) -> "MultiPurposeModel":
                self.X_train = X
                self.y_train = y
                return self

            def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.integer[Any]]:
                if self.y_train is None:
                    raise ValueError("Not fitted")
                return np.zeros(len(X), dtype=np.int64)

            # ClusteringProtocol methods
            def fit_predict(
                self, X: NDArray[np.floating[Any]]
            ) -> NDArray[np.integer[Any]]:
                return np.zeros(len(X), dtype=np.int64)

        model = MultiPurposeModel()

        # Should satisfy both protocols
        assert isinstance(model, ClassifierProtocol)
        assert isinstance(model, ClusteringProtocol)
