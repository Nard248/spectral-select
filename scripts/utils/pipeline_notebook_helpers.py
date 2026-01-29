"""
Pipeline Notebook Helpers
=========================

Helper functions for running wavelength selection and classification pipelines
in Jupyter notebooks. Provides a simpler interface than the CLI script.

Example usage:
    from scripts.utils.pipeline_notebook_helpers import (
        load_data_and_ground_truth,
        run_baseline_classification,
        run_wavelength_selection_experiment,
        create_comparison_report
    )

    # Load data
    data, ground_truth, rois = load_data_and_ground_truth(
        data_path='Data/processed/Lichens',
        mask_path='Data/processed/Lichens/ground_truth.png'
    )

    # Run baseline
    baseline_result = run_baseline_classification(data, ground_truth, rois)

    # Run experiment with wavelength selection
    result = run_wavelength_selection_experiment(
        data, ground_truth, rois,
        n_bands=30,
        data_path='Data/processed/Lichens'
    )

    # Compare results
    df = create_comparison_report([baseline_result, result])
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report
)


@dataclass
class ROIRegion:
    """Definition of a Region of Interest for training."""
    name: str
    coords: Tuple[int, int, int, int]  # (y_start, y_end, x_start, x_end)
    color: str
    class_id: int = -1


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config_name: str
    n_bands_selected: int
    n_features: int
    accuracy: float = 0.0
    f1_weighted: float = 0.0
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    cohen_kappa: float = 0.0
    confusion_matrix: np.ndarray = None
    cluster_map: np.ndarray = None
    wavelength_combinations: List[Dict] = field(default_factory=list)
    data_reduction_pct: float = 0.0


def load_data_and_ground_truth(
    data_path: str,
    mask_path: str = None,
    background_colors: List[Tuple[int, ...]] = None
) -> Tuple[Dict, np.ndarray, List[ROIRegion]]:
    """
    Load hyperspectral data and ground truth mask.

    Args:
        data_path: Path to .pkl file or directory containing .pkl/.im3 files
        mask_path: Path to PNG ground truth mask
        background_colors: List of RGBA tuples to treat as background

    Returns:
        - data: Hyperspectral data dictionary (V2 format)
        - ground_truth: 2D array with class labels (-1 for background)
        - rois: List of auto-detected ROI regions
    """
    from spectral_select import SpectraData

    data_path = Path(data_path)

    print(f"Loading from: {data_path}")

    # Load based on path type
    if data_path.suffix == '.pkl':
        spectra_data = SpectraData.from_pickle(data_path)
    elif data_path.is_dir():
        # Check for pickle files first
        pkl_files = list(data_path.glob("*.pkl"))
        im3_files = list(data_path.glob("*.im3"))

        if pkl_files:
            # Use first pickle (prefer ones with 'data' in name)
            pkl_file = pkl_files[0]
            for pf in pkl_files:
                if 'data' in pf.stem.lower() and 'cutoff' in pf.stem.lower():
                    pkl_file = pf
                    break
            print(f"  Using pickle: {pkl_file.name}")
            spectra_data = SpectraData.from_pickle(pkl_file)
        elif im3_files:
            print(f"  Loading raw .im3 files")
            spectra_data = SpectraData.from_raw(data_path)
        else:
            raise ValueError(f"No .pkl or .im3 files in {data_path}")
    else:
        raise ValueError(f"Path not found: {data_path}")

    # Convert to V2-compatible dict format
    data = {
        'excitation_wavelengths': list(spectra_data.excitations.keys()),
        'data': {},
        'metadata': spectra_data.metadata or {}
    }
    for ex_nm, ex_data in spectra_data.excitations.items():
        wavelengths = ex_data.emission_wavelengths
        if hasattr(wavelengths, 'tolist'):
            wavelengths = wavelengths.tolist()
        data['data'][str(ex_nm)] = {
            'cube': ex_data.cube,
            'wavelengths': wavelengths
        }

    # Get spatial dimensions
    first_ex = str(data['excitation_wavelengths'][0])
    height, width = data['data'][first_ex]['cube'].shape[:2]

    # Count bands
    total_bands = sum(
        len(data['data'][str(ex)]['wavelengths'])
        for ex in data['excitation_wavelengths']
    )

    print(f"Data loaded:")
    print(f"  Excitations: {len(data['excitation_wavelengths'])}")
    print(f"  Total bands: {total_bands}")
    print(f"  Spatial: {height} x {width}")

    # Load ground truth if provided
    ground_truth = None
    rois = []

    if mask_path:
        mask_path = Path(mask_path)
        if mask_path.exists():
            from PIL import Image

            img = Image.open(mask_path)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Resize if needed
            if img.size != (width, height):
                print(f"  Resizing mask from {img.size} to ({width}, {height})")
                img = img.resize((width, height), Image.NEAREST)

            img_array = np.array(img)

            if background_colors is None:
                background_colors = [
                    (0, 0, 0, 255), (255, 255, 255, 255),
                    (24, 24, 24, 255), (168, 168, 168, 255),
                    (0, 0, 0, 0)  # Transparent
                ]

            # Find unique class colors with their counts
            pixels = img_array.reshape(-1, 4)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

            # Filter: min 100 pixels to avoid anti-aliasing artifacts
            min_pixel_count = 100
            class_colors = []
            for color, count in zip(unique_colors, counts):
                if count < min_pixel_count:
                    continue
                color_tuple = tuple(int(c) for c in color)

                # Check background with total RGB difference
                is_bg = False
                for bg in background_colors:
                    rgb_diff = sum(abs(color_tuple[i] - bg[i]) for i in range(3))
                    if rgb_diff < 30:
                        is_bg = True
                        break
                if not is_bg:
                    class_colors.append(color_tuple)

            # Create ground truth map
            ground_truth = np.full((height, width), -1, dtype=int)

            for class_idx, color in enumerate(class_colors):
                mask = np.all(np.abs(img_array.astype(int) - np.array(color)) < 10, axis=2)
                ground_truth[mask] = class_idx

            print(f"\nGround truth loaded:")
            print(f"  Classes: {len(class_colors)}")
            for idx in range(len(class_colors)):
                count = np.sum(ground_truth == idx)
                print(f"    Class {idx}: {count:,} pixels ({100*count/ground_truth.size:.1f}%)")

            # Auto-detect ROIs
            rois = _auto_detect_rois(ground_truth)

    return data, ground_truth, rois


def _auto_detect_rois(ground_truth: np.ndarray) -> List[ROIRegion]:
    """Auto-detect ROI regions from ground truth."""
    colors = ['#FF0000', '#0000FF', '#00FF00', '#FFFF00', '#FF00FF',
              '#00FFFF', '#FFA500', '#800080', '#008000', '#FFC0CB']

    unique_classes = np.unique(ground_truth)
    unique_classes = unique_classes[unique_classes >= 0]

    rois = []
    for class_id in unique_classes:
        class_mask = ground_truth == class_id
        labeled, n_comp = ndimage.label(class_mask)

        if n_comp == 0:
            continue

        # Find largest component
        comp_sizes = ndimage.sum(class_mask, labeled, range(1, n_comp + 1))
        largest = np.argmax(comp_sizes) + 1
        comp_mask = labeled == largest

        y_idx, x_idx = np.where(comp_mask)
        if len(y_idx) == 0:
            continue

        # Shrink to get pure samples
        margin = 5
        y_min = max(y_idx.min() + margin, 0)
        y_max = min(y_idx.max() - margin, ground_truth.shape[0])
        x_min = max(x_idx.min() + margin, 0)
        x_max = min(x_idx.max() - margin, ground_truth.shape[1])

        # Ensure valid region
        if y_max <= y_min or x_max <= x_min:
            y_min, y_max = y_idx.min(), y_idx.max()
            x_min, x_max = x_idx.min(), x_idx.max()

        roi = ROIRegion(
            name=f'Class_{class_id}',
            coords=(y_min, y_max, x_min, x_max),
            color=colors[int(class_id) % len(colors)],
            class_id=int(class_id)
        )
        rois.append(roi)

    print(f"\nAuto-detected {len(rois)} ROI regions")
    return rois


def _build_feature_matrix(data: Dict[str, Any]) -> Tuple[np.ndarray, int, int]:
    """Build feature matrix from hyperspectral data."""
    first_ex = str(data['excitation_wavelengths'][0])
    height, width = data['data'][first_ex]['cube'].shape[:2]

    all_features = []
    for y in range(height):
        for x in range(width):
            pixel = []
            for ex in data['excitation_wavelengths']:
                cube = data['data'][str(ex)]['cube']
                pixel.extend(cube[y, x, :])
            all_features.append(pixel)

    return np.array(all_features), height, width


def run_knn_classification(
    data: Dict[str, Any],
    rois: List[ROIRegion],
    ground_truth: np.ndarray = None,
    n_neighbors: int = 5
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    """
    Run KNN classification on hyperspectral data.

    Args:
        data: Hyperspectral data dictionary
        rois: ROI regions for training
        ground_truth: Optional ground truth for metrics
        n_neighbors: Number of neighbors for KNN

    Returns:
        - cluster_map: Classification result
        - metrics: Dictionary of metrics
        - confusion: Confusion matrix (or None)
    """
    X_full, height, width = _build_feature_matrix(data)

    # Extract training from ROIs
    X_train, y_train = [], []
    for roi in rois:
        y_start, y_end, x_start, x_end = roi.coords
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                if 0 <= y < height and 0 <= x < width:
                    idx = y * width + x
                    X_train.append(X_full[idx])
                    y_train.append(roi.class_id)

    if len(X_train) == 0:
        raise ValueError("No training samples found in ROI regions")

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"  Training: {len(X_train)} samples, {len(np.unique(y_train))} classes")

    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_full_scaled = scaler.transform(X_full)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)

    predictions = knn.predict(X_full_scaled)
    cluster_map = predictions.reshape(height, width)

    # Metrics
    metrics = {'n_features': X_full.shape[1]}
    confusion = None

    if ground_truth is not None:
        valid = ground_truth >= 0
        y_true = ground_truth[valid]
        y_pred = cluster_map[valid]

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred)

        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}")

    return cluster_map, metrics, confusion


def run_baseline_classification(
    data: Dict[str, Any],
    ground_truth: np.ndarray,
    rois: List[ROIRegion]
) -> ExperimentResult:
    """
    Run baseline classification with all bands.

    Args:
        data: Hyperspectral data
        ground_truth: Ground truth labels
        rois: ROI regions

    Returns:
        ExperimentResult for baseline
    """
    print("\n" + "="*60)
    print("BASELINE: Full Data Classification")
    print("="*60)

    cluster_map, metrics, confusion = run_knn_classification(data, rois, ground_truth)

    return ExperimentResult(
        config_name='BASELINE_FULL_DATA',
        n_bands_selected=metrics['n_features'],
        n_features=metrics['n_features'],
        accuracy=metrics.get('accuracy', 0),
        f1_weighted=metrics.get('f1_weighted', 0),
        precision_weighted=metrics.get('precision_weighted', 0),
        recall_weighted=metrics.get('recall_weighted', 0),
        cohen_kappa=metrics.get('cohen_kappa', 0),
        confusion_matrix=confusion,
        cluster_map=cluster_map,
        wavelength_combinations=[],
        data_reduction_pct=0.0
    )


def run_wavelength_selection_experiment(
    data: Dict[str, Any],
    ground_truth: np.ndarray,
    rois: List[ROIRegion],
    n_bands: int = 30,
    use_diversity: bool = True,
    diversity_method: str = 'mmr',
    lambda_diversity: float = 0.5,
    data_path: str = None,
    sample_name: str = 'sample',
    baseline_n_features: int = None
) -> ExperimentResult:
    """
    Run wavelength selection experiment.

    Args:
        data: Hyperspectral data dictionary
        ground_truth: Ground truth labels
        rois: ROI regions
        n_bands: Number of bands to select
        use_diversity: Use diversity constraint
        diversity_method: Diversity method
        lambda_diversity: Diversity weight
        data_path: Path to data for wavelength selection
        sample_name: Sample name
        baseline_n_features: Baseline feature count for reduction calculation

    Returns:
        ExperimentResult
    """
    from spectral_select import Config, Analyzer, SpectraData
    import tempfile

    config_name = f"bands_{n_bands}"
    if use_diversity:
        config_name += f"_{diversity_method}_l{lambda_diversity:.1f}"

    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print("="*60)

    wavelength_combinations = []

    if data_path:
        data_path = Path(data_path)

        # Create temporary mask
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            mask = ground_truth >= 0
            np.save(f.name, mask)
            mask_path = Path(f.name)

        try:
            config = Config(
                sample_name=sample_name,
                data_path=str(data_path),
                mask_path=str(mask_path),
                n_bands_to_select=n_bands,
                use_diversity_constraint=use_diversity,
                diversity_method=diversity_method,
                lambda_diversity=lambda_diversity,
                save_visualizations=False
            )

            analyzer = Analyzer(config)

            # Load data
            if data_path.suffix == '.pkl':
                spectra_data = SpectraData.from_pickle(data_path)
            elif data_path.is_dir():
                pkl_files = list(data_path.glob("*.pkl"))
                im3_files = list(data_path.glob("*.im3"))
                if pkl_files:
                    pkl_file = pkl_files[0]
                    for pf in pkl_files:
                        if 'data' in pf.stem.lower():
                            pkl_file = pf
                            break
                    spectra_data = SpectraData.from_pickle(pkl_file)
                elif im3_files:
                    spectra_data = SpectraData.from_raw(data_path)
                else:
                    raise ValueError(f"No .pkl or .im3 files in {data_path}")
            else:
                raise ValueError(f"Invalid path: {data_path}")

            spectra_data.mask = mask
            analyzer.fit(spectra_data)

            for band in analyzer.get_selected_bands():
                wavelength_combinations.append({
                    'excitation': band.excitation_nm,
                    'emission': band.emission_nm,
                    'score': band.influence_score,
                    'rank': band.rank
                })

            print(f"  Selected {len(wavelength_combinations)} wavelengths")

            # Extract subset
            data = _extract_wavelength_subset(data, wavelength_combinations)

        finally:
            mask_path.unlink(missing_ok=True)

    # Run classification
    cluster_map, metrics, confusion = run_knn_classification(data, rois, ground_truth)

    # Calculate data reduction
    data_reduction = 0.0
    if baseline_n_features:
        data_reduction = (1 - metrics['n_features'] / baseline_n_features) * 100

    return ExperimentResult(
        config_name=config_name,
        n_bands_selected=len(wavelength_combinations) if wavelength_combinations else metrics['n_features'],
        n_features=metrics['n_features'],
        accuracy=metrics.get('accuracy', 0),
        f1_weighted=metrics.get('f1_weighted', 0),
        precision_weighted=metrics.get('precision_weighted', 0),
        recall_weighted=metrics.get('recall_weighted', 0),
        cohen_kappa=metrics.get('cohen_kappa', 0),
        confusion_matrix=confusion,
        cluster_map=cluster_map,
        wavelength_combinations=wavelength_combinations,
        data_reduction_pct=data_reduction
    )


def _extract_wavelength_subset(data: Dict, combinations: List[Dict]) -> Dict:
    """Extract wavelength subset from full data."""
    subset = {
        'excitation_wavelengths': [],
        'data': {},
        'metadata': data.get('metadata', {})
    }

    # Group by excitation
    combos_by_ex = {}
    for c in combinations:
        ex = c['excitation']
        if ex not in combos_by_ex:
            combos_by_ex[ex] = []
        combos_by_ex[ex].append(c['emission'])

    for ex in data['excitation_wavelengths']:
        ex_str = str(ex)
        wls = np.array(data['data'][ex_str]['wavelengths'])
        cube = data['data'][ex_str]['cube']

        matching = None
        for combo_ex, emissions in combos_by_ex.items():
            if abs(float(ex) - float(combo_ex)) < 1.0:
                matching = emissions
                break

        if matching is None:
            continue

        indices = []
        for em in matching:
            dists = np.abs(wls - em)
            idx = np.argmin(dists)
            if dists[idx] < 10 and idx not in indices:
                indices.append(idx)

        if indices:
            subset['data'][ex_str] = {
                'cube': cube[:, :, indices],
                'wavelengths': wls[indices].tolist()
            }
            subset['excitation_wavelengths'].append(ex)

    return subset


def run_multiple_experiments(
    data: Dict[str, Any],
    ground_truth: np.ndarray,
    rois: List[ROIRegion],
    n_bands_list: List[int] = None,
    data_path: str = None,
    sample_name: str = 'sample',
    use_diversity: bool = True,
    diversity_method: str = 'mmr'
) -> List[ExperimentResult]:
    """
    Run multiple experiments with different band counts.

    Args:
        data: Hyperspectral data
        ground_truth: Ground truth labels
        rois: ROI regions
        n_bands_list: List of band counts to test
        data_path: Path to data
        sample_name: Sample name
        use_diversity: Use diversity constraint
        diversity_method: Diversity method

    Returns:
        List of ExperimentResult objects
    """
    if n_bands_list is None:
        n_bands_list = [10, 20, 30, 40, 50]

    results = []

    # Baseline
    baseline = run_baseline_classification(data, ground_truth, rois)
    results.append(baseline)
    baseline_n_features = baseline.n_features

    # Run experiments
    for n_bands in n_bands_list:
        result = run_wavelength_selection_experiment(
            data, ground_truth, rois,
            n_bands=n_bands,
            use_diversity=use_diversity,
            diversity_method=diversity_method,
            data_path=data_path,
            sample_name=sample_name,
            baseline_n_features=baseline_n_features
        )
        results.append(result)

    return results


def create_comparison_report(
    results: List[ExperimentResult],
    output_path: str = None
) -> pd.DataFrame:
    """
    Create a comparison report DataFrame.

    Args:
        results: List of ExperimentResult
        output_path: Optional path to save report

    Returns:
        DataFrame with comparison
    """
    rows = []
    for r in results:
        rows.append({
            'Config': r.config_name,
            'Bands': r.n_bands_selected,
            'Features': r.n_features,
            'Reduction (%)': r.data_reduction_pct,
            'Accuracy': r.accuracy,
            'F1': r.f1_weighted,
            'Precision': r.precision_weighted,
            'Recall': r.recall_weighted,
            'Kappa': r.cohen_kappa
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('Accuracy', ascending=False)

    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path / 'comparison_report.csv', index=False)
        df.to_parquet(output_path / 'comparison_report.parquet', index=False)
        print(f"\nReport saved to {output_path}")

    return df


def plot_results_comparison(
    results: List[ExperimentResult],
    ground_truth: np.ndarray = None,
    rois: List[ROIRegion] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Create comparison visualization.

    Args:
        results: List of ExperimentResult
        ground_truth: Ground truth for visualization
        rois: ROI regions for colormap
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Data for plots
    features = [r.n_features for r in results]
    accuracies = [r.accuracy for r in results]
    f1s = [r.f1_weighted for r in results]
    names = [r.config_name for r in results]

    # Find baseline and best
    baseline = next((r for r in results if 'BASELINE' in r.config_name), None)
    best_idx = np.argmax(accuracies)
    best = results[best_idx]

    # 1. Accuracy vs features
    ax = axes[0, 0]
    scatter = ax.scatter(features, accuracies, s=100, c=accuracies, cmap='viridis', edgecolors='black')
    ax.scatter(features[best_idx], accuracies[best_idx],
               s=200, color='red', marker='*', zorder=5, label='Best')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs. Feature Count')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Metrics comparison
    ax = axes[0, 1]
    if baseline and best:
        metrics_names = ['Acc', 'F1', 'Prec', 'Recall']
        x = np.arange(len(metrics_names))
        width = 0.35

        baseline_vals = [baseline.accuracy, baseline.f1_weighted,
                        baseline.precision_weighted, baseline.recall_weighted]
        best_vals = [best.accuracy, best.f1_weighted,
                    best.precision_weighted, best.recall_weighted]

        ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='blue', alpha=0.7)
        ax.bar(x + width/2, best_vals, width, label='Best', color='green', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.set_ylabel('Score')
        ax.set_title('Baseline vs. Best Config')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

    # 3. F1 vs features
    ax = axes[0, 2]
    ax.scatter(features, f1s, s=100, c=f1s, cmap='plasma', edgecolors='black')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs. Feature Count')
    ax.grid(True, alpha=0.3)

    # 4-6. Classification maps
    if ground_truth is not None and rois:
        n_classes = len(rois)
        colors = [roi.color for roi in rois]
        cmap = mcolors.ListedColormap(colors)

        # Ground truth
        ax = axes[1, 0]
        ax.imshow(ground_truth, cmap=cmap, vmin=0, vmax=n_classes-1)
        ax.set_title('Ground Truth')
        ax.axis('off')

        # Baseline
        if baseline and baseline.cluster_map is not None:
            ax = axes[1, 1]
            ax.imshow(baseline.cluster_map, cmap=cmap, vmin=0, vmax=n_classes-1)
            ax.set_title(f'Baseline\nAcc: {baseline.accuracy:.4f}')
            ax.axis('off')

        # Best
        if best.cluster_map is not None:
            ax = axes[1, 2]
            ax.imshow(best.cluster_map, cmap=cmap, vmin=0, vmax=n_classes-1)
            ax.set_title(f'Best: {best.config_name[:20]}\nAcc: {best.accuracy:.4f}')
            ax.axis('off')

    plt.tight_layout()
    return fig


def print_summary(results: List[ExperimentResult]):
    """Print a summary of experiment results."""
    baseline = next((r for r in results if 'BASELINE' in r.config_name), None)
    best = max(results, key=lambda r: r.accuracy)

    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    if baseline:
        print(f"\nBaseline (full data):")
        print(f"  Features:  {baseline.n_features}")
        print(f"  Accuracy:  {baseline.accuracy:.4f}")
        print(f"  F1 Score:  {baseline.f1_weighted:.4f}")
        print(f"  Kappa:     {baseline.cohen_kappa:.4f}")

    print(f"\nBest configuration: {best.config_name}")
    print(f"  Features:  {best.n_features} ({best.data_reduction_pct:.1f}% reduction)")
    print(f"  Accuracy:  {best.accuracy:.4f}")
    print(f"  F1 Score:  {best.f1_weighted:.4f}")
    print(f"  Kappa:     {best.cohen_kappa:.4f}")

    if baseline:
        acc_change = (best.accuracy - baseline.accuracy) / baseline.accuracy * 100
        print(f"\n  Accuracy change vs baseline: {acc_change:+.2f}%")

    print("\n" + "="*70)
