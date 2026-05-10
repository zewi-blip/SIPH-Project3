import os
import csv
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

def extract_features_to_csv(root_dir, category, output_filename):
    """
    Category: "benign" or "malignant"
    """
    img_dir = os.path.join(root_dir, category)
    mask_dir = os.path.join(root_dir, f"{category}_mask")

    # Initialize PyRadiomics Extractor, enable 'shape2D' specifically for ultrasound
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('shape2D')  # mass shape
    extractor.enableFeatureClassByName('firstorder')  # intensity/echotexture
    extractor.enableFeatureClassByName('glcm')  # texture (Heterogeneity)

    results = []

    image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    print(f"Processing {category} images...")

    for fname in image_files:
        # Match mask
        base_name = os.path.splitext(fname)[0]
        mask_name = f"{base_name}_mask.png"

        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            continue

        sitk_img = sitk.ReadImage(img_path)

        # Needs RGB images
        if sitk_img.GetNumberOfComponentsPerPixel() > 1:
            sitk_img = sitk.VectorIndexSelectionCast(sitk_img, 0)

        sitk_mask = sitk.ReadImage(mask_path)
        # Ensure mask is binary (0 or 1)
        sitk_mask = sitk.Cast(sitk_mask > 127, sitk.sitkUInt8)

        try:
            # Extract features
            feature_vector = extractor.execute(sitk_img, sitk_mask)

            row = {"ID": base_name, "Label": category}
            for key, value in feature_vector.items():
                if not key.startswith("diagnostics"):
                    row[key] = value

            results.append(row)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    # Save to CSV
    if results:
        keys = results[0].keys()
        with open(output_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {len(results)} samples to {output_filename}")


if __name__ == "__main__":
    DATASET_ROOT = "/Users/zozo/Desktop/unimib/SIPH/Project 2"

    extract_features_to_csv(DATASET_ROOT, "benign", "benign_features.csv")
    extract_features_to_csv(DATASET_ROOT, "malignant", "malignant_features.csv")