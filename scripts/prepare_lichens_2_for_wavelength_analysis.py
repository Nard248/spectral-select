"""
Prepare Lichens_2 Data for Wavelength Selection Pipeline

This script:
1. Creates binary mask from the labeled PNG
2. Analyzes class distribution and locations
3. Suggests ROI regions for KNN training
4. Resizes mask to match hyperspectral data dimensions
"""

import numpy as np
from PIL import Image
from pathlib import Path
import pickle
import cv2
from scipy import ndimage

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "Data" / "processed" / "Lichens_2"
RAW_DIR = BASE_DIR / "Data" / "Raw" / "Lichens_2"

LABELED_PNG_PATH = DATA_DIR / "labeled_lichens.png"
DATA_PKL_PATH = DATA_DIR / "data_cutoff_40nm_exposure_max.pkl"
OUTPUT_MASK_PATH = DATA_DIR / "lichens_2_mask.npy"
OUTPUT_MASKED_DATA_PATH = DATA_DIR / "lichens_2_data_masked.pkl"


def load_labeled_png(png_path):
    """Load labeled PNG and analyze colors."""
    img = Image.open(png_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img)

    print(f"Labeled PNG shape: {img_array.shape}")
    return img_array


def color_distance(c1, c2):
    """Calculate Euclidean distance between two RGB colors."""
    return np.sqrt(np.sum((np.array(c1, dtype=float) - np.array(c2, dtype=float))**2))


def find_closest_class(color, class_colors, tolerance=50):
    """Find the closest class color within tolerance."""
    min_dist = float('inf')
    best_class = None

    for target_color, info in class_colors.items():
        dist = color_distance(color, target_color)
        if dist < min_dist:
            min_dist = dist
            best_class = (target_color, info, dist)

    if best_class and best_class[2] <= tolerance:
        return best_class[1]['class_id'], best_class[1]['name']
    return None, None


def extract_classes_from_png(img_array, tolerance=50):
    """
    Extract class labels from PNG based on colors with tolerance for anti-aliasing.
    Returns: class_mask (H, W), color_to_class mapping, class_info
    """
    h, w = img_array.shape[:2]

    # Define known class colors (based on the labeling tool defaults)
    # Background is black (0, 0, 0)
    CLASS_COLORS = {
        (0, 0, 0): {'class_id': -1, 'name': 'Background'},
        (255, 0, 0): {'class_id': 0, 'name': 'Class 1 (Red)'},
        (0, 255, 0): {'class_id': 1, 'name': 'Class 2 (Green)'},
        (0, 0, 255): {'class_id': 2, 'name': 'Class 3 (Blue)'},
        (255, 255, 0): {'class_id': 3, 'name': 'Class 4 (Yellow)'},
        (255, 0, 255): {'class_id': 4, 'name': 'Class 5 (Magenta)'},
        (0, 255, 255): {'class_id': 5, 'name': 'Class 6 (Cyan)'},
        (128, 128, 128): {'class_id': -2, 'name': 'Unlabeled (Gray)'},
    }

    # Create class mask
    class_mask = np.full((h, w), -1, dtype=np.int16)  # Use int16 for larger range
    class_pixel_counts = {k: 0 for k in range(-2, 6)}

    print(f"Processing {h}x{w} image with color tolerance={tolerance}...")

    # Process each pixel
    for y in range(h):
        for x in range(w):
            pixel = tuple(img_array[y, x])

            # First check exact match
            if pixel in CLASS_COLORS:
                class_id = CLASS_COLORS[pixel]['class_id']
            else:
                # Find closest class within tolerance
                class_id, _ = find_closest_class(pixel, CLASS_COLORS, tolerance)
                if class_id is None:
                    class_id = -1  # Treat as background if no match

            class_mask[y, x] = class_id
            if class_id in class_pixel_counts:
                class_pixel_counts[class_id] += 1

    # Build class_info
    class_info = {}
    print("\nClass distribution:")
    for target_color, info in CLASS_COLORS.items():
        class_id = info['class_id']
        count = class_pixel_counts.get(class_id, 0)

        if class_id >= 0 and count > 0:
            # Find bounding box
            mask = class_mask == class_id
            ys, xs = np.where(mask)
            bbox = {
                'y_min': int(ys.min()),
                'y_max': int(ys.max()),
                'x_min': int(xs.min()),
                'x_max': int(xs.max())
            }
            class_info[class_id] = {
                'name': info['name'],
                'color': target_color,
                'pixel_count': count,
                'bbox': bbox
            }

        print(f"  {info['name']}: {count:,} pixels")

    return class_mask, class_info


def create_binary_mask(class_mask):
    """Create binary mask where True = valid (non-background) pixels."""
    binary_mask = class_mask >= 0
    print(f"\nBinary mask: {np.sum(binary_mask):,} valid pixels out of {binary_mask.size:,}")
    return binary_mask


def find_object_rois(class_mask, class_info):
    """
    Find individual objects and suggest ROI regions for each class.
    Returns list of ROI definitions.
    """
    # Find connected components for each class
    all_rois = []

    print("\n" + "=" * 60)
    print("SUGGESTED ROI REGIONS FOR KNN TRAINING")
    print("=" * 60)
    print("\nFormat: {'name': 'Region N', 'coords': (y_start, y_end, x_start, x_end), 'color': '#RRGGBB'}")
    print("\n# Copy this to your pipeline configuration:")
    print("ROI_REGIONS = [")

    roi_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
    roi_index = 1

    for class_id, info in sorted(class_info.items()):
        if class_id < 0:
            continue

        # Get mask for this class
        class_mask_binary = (class_mask == class_id).astype(np.uint8)

        # Find connected components (individual objects)
        labeled, num_objects = ndimage.label(class_mask_binary)

        print(f"\n# Class {class_id}: {info['name']} - {num_objects} objects")

        # Find largest object for this class (most representative)
        best_obj = None
        best_size = 0

        for obj_id in range(1, num_objects + 1):
            obj_mask = labeled == obj_id
            size = np.sum(obj_mask)

            if size > best_size:
                best_size = size
                ys, xs = np.where(obj_mask)
                # Add some margin
                margin = 5
                best_obj = {
                    'y_min': max(0, int(ys.min()) - margin),
                    'y_max': int(ys.max()) + margin,
                    'x_min': max(0, int(xs.min()) - margin),
                    'x_max': int(xs.max()) + margin,
                    'size': size
                }

        if best_obj:
            roi_def = {
                'name': f'Region {roi_index}',
                'coords': (best_obj['y_min'], best_obj['y_max'],
                          best_obj['x_min'], best_obj['x_max']),
                'color': roi_colors[class_id % len(roi_colors)],
                'class_id': class_id,
                'class_name': info['name']
            }
            all_rois.append(roi_def)

            print(f"    {{'name': 'Region {roi_index}', 'coords': ({best_obj['y_min']}, {best_obj['y_max']}, {best_obj['x_min']}, {best_obj['x_max']}), 'color': '{roi_colors[class_id % len(roi_colors)]}'}},  # {info['name']}")
            roi_index += 1

    print("]")

    return all_rois


def resize_mask_to_data(binary_mask, data_pkl_path):
    """Resize mask to match hyperspectral data dimensions."""
    # Load data to get dimensions
    with open(data_pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Get dimensions from first excitation wavelength
    first_ex = list(data['data'].keys())[0]
    cube_shape = data['data'][first_ex]['cube'].shape
    target_h, target_w = cube_shape[0], cube_shape[1]

    current_h, current_w = binary_mask.shape

    print(f"\nResizing mask from {current_h}x{current_w} to {target_h}x{target_w}")

    # Use nearest neighbor interpolation to preserve binary values
    resized_mask = cv2.resize(
        binary_mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    print(f"Resized mask: {np.sum(resized_mask):,} valid pixels")

    return resized_mask, (target_h, target_w)


def resize_class_mask_to_data(class_mask, target_shape):
    """Resize class mask to match hyperspectral data dimensions."""
    target_h, target_w = target_shape

    # Use nearest neighbor interpolation to preserve class labels
    resized_class_mask = cv2.resize(
        class_mask.astype(np.int16),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST
    )

    return resized_class_mask


def main():
    print("=" * 60)
    print("PREPARING LICHENS_2 FOR WAVELENGTH SELECTION PIPELINE")
    print("=" * 60)

    # Check paths
    if not LABELED_PNG_PATH.exists():
        print(f"ERROR: Labeled PNG not found at {LABELED_PNG_PATH}")
        return

    if not DATA_PKL_PATH.exists():
        print(f"ERROR: Data PKL not found at {DATA_PKL_PATH}")
        return

    # Step 1: Load labeled PNG
    print("\n1. Loading labeled PNG...")
    img_array = load_labeled_png(LABELED_PNG_PATH)

    # Step 2: Extract classes
    print("\n2. Extracting class labels...")
    class_mask, class_info = extract_classes_from_png(img_array)

    # Step 3: Create binary mask
    print("\n3. Creating binary mask...")
    binary_mask = create_binary_mask(class_mask)

    # Step 4: Resize to match data dimensions
    print("\n4. Resizing mask to match hyperspectral data dimensions...")
    resized_mask, target_shape = resize_mask_to_data(binary_mask, DATA_PKL_PATH)
    resized_class_mask = resize_class_mask_to_data(class_mask, target_shape)

    # Step 5: Find ROI regions
    print("\n5. Analyzing objects and suggesting ROI regions...")
    rois = find_object_rois(resized_class_mask, class_info)

    # Step 6: Save outputs
    print("\n6. Saving outputs...")

    # Save binary mask
    np.save(OUTPUT_MASK_PATH, resized_mask)
    print(f"  Saved binary mask to: {OUTPUT_MASK_PATH}")

    # Save class mask
    class_mask_path = DATA_DIR / "lichens_2_class_mask.npy"
    np.save(class_mask_path, resized_class_mask)
    print(f"  Saved class mask to: {class_mask_path}")

    # Save class info
    import json
    class_info_path = DATA_DIR / "lichens_2_class_info.json"
    # Convert numpy types for JSON serialization
    class_info_json = {}
    for k, v in class_info.items():
        class_info_json[int(k)] = {
            'name': v['name'],
            'color': list(v['color']),
            'pixel_count': int(v['pixel_count'])
        }
    with open(class_info_path, 'w') as f:
        json.dump(class_info_json, f, indent=2)
    print(f"  Saved class info to: {class_info_path}")

    # Save ROI definitions
    roi_path = DATA_DIR / "lichens_2_roi_regions.json"
    roi_json = []
    for roi in rois:
        roi_json.append({
            'name': roi['name'],
            'coords': list(roi['coords']),
            'color': roi['color'],
            'class_id': int(roi['class_id']),
            'class_name': roi['class_name']
        })
    with open(roi_path, 'w') as f:
        json.dump(roi_json, f, indent=2)
    print(f"  Saved ROI definitions to: {roi_path}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  1. Binary mask: {OUTPUT_MASK_PATH}")
    print(f"  2. Class mask: {class_mask_path}")
    print(f"  3. Class info: {class_info_path}")
    print(f"  4. ROI regions: {roi_path}")
    print(f"\nNext step: Run the wavelength selection pipeline with these files.")


if __name__ == "__main__":
    main()
