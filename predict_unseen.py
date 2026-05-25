import os
import re
import sys
import cv2
import numpy as np
import ollama
import pandas as pd
import SimpleITK as sitk
import torch
from radiomics import featureextractor
from unet import UNet

# --- The Numpy Pickle Workaround ---
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
if "numpy._core.multiarray" not in sys.modules:
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

# Load trained segmentation model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)
checkpoint = torch.load("final_model.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()


class ImageCropper:

    def __init__(self, bbox_filename="ground_truth_boxes.npy"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        clean_name = os.path.basename(bbox_filename)
        bbox_path = os.path.join(current_dir, clean_name)

        if not os.path.exists(bbox_path):
            raise FileNotFoundError(f"File not found at: {bbox_path}")

        bbox_data = np.load(bbox_path, allow_pickle=True)
        self.bbox_map = {
            os.path.basename(item["image"]): item["boxes"] for item in bbox_data
        }
        #print(f"Loaded {len(self.bbox_map)} image bboxes successfully.")

    def crop_by_filename(self, img, mask, filename):
        name_key = os.path.basename(filename)

        if name_key not in self.bbox_map:
            print(
                f"Warning: Bounding box key '{name_key}' not found. Returning original sizes."
            )
            return img, mask

        boxes = self.bbox_map[name_key]

        x_min = int(min([b[0] for b in boxes]))
        y_min = int(min([b[1] for b in boxes]))
        x_max = int(max([b[0] + b[2] for b in boxes]))
        y_max = int(max([b[1] + b[3] for b in boxes]))

        h, w = img.shape[:2]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        # Handle grayscale mask dimensions safely
        if mask is not None:
            return (
                img[y_min:y_max, x_min:x_max],
                mask[y_min:y_max, x_min:x_max],
            )
        return img[y_min:y_max, x_min:x_max], None


def lookup_posterior_features(image_path, csv_path):
    """Looks up shadow and enhancement confidence scores from the feature CSV

    by matching the image's base ID.
    """
    base_id = os.path.splitext(os.path.basename(image_path))[0]

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["ID"] = df["ID"].astype(str)
        matched_row = df[df["ID"] == str(base_id)]

        if not matched_row.empty:
            row = matched_row.iloc[0]
            shadow_conf = row.get("shadow_conf", 0.0)
            enhance_conf = row.get("enhancement_conf", 0.0)

            if shadow_conf > 0.5 and shadow_conf > enhance_conf:
                return "Acoustic shadowing (Typical of malignant/dense structures)"
            elif enhance_conf > 0.5 and enhance_conf > shadow_conf:
                return (
                    "Acoustic enhancement (Typical of benign/fluid-filled cysts)"
                )

    return "No posterior features / No significant shadowing or enhancement"


def process_and_classify_image(
    image_path, mask_path, reference_csv, cropper_obj
):
    # --- Step 1: Read Original Files ---
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        raise FileNotFoundError(
            f"Error: Could not load image file from {image_path}"
        )

    mask_orig = (
        cv2.imread(mask_path, 0) if os.path.exists(mask_path) else None
    )

    # --- Step 2: Crop Using bounding boxes ---
    img_cropped, _ = cropper_obj.crop_by_filename(
        img_orig, mask_orig, image_path
    )

    # --- Step 3: Segmentation Pipeline (U-Net) ---
    img_resized = cv2.resize(img_cropped, (256, 256)) / 255.0
    img_tensor = (
        torch.tensor(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    )

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor)).cpu().squeeze().numpy()
    mask_np = (pred > 0.5).astype(np.uint8)

    # Scale the U-Net mask output to match the cropped image coordinates
    mask_resized = cv2.resize(
        mask_np,
        (img_cropped.shape[1], img_cropped.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    '''
    # --- Step 4: Visualize Contour ---
    contours, _ = cv2.findContours(
        mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    overlay = img_cropped.copy()
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    cv2.imshow("Predicted Contour (On Cropped Image)", overlay)
    print("Close the image window to proceed to AI Classification...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    # --- Step 5: Radiomics Feature Extraction (Using memory arrays) ---
    # Convert OpenCV image array (BGR -> Gray) directly to a SimpleITK image
    gray_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    sitk_img = sitk.GetImageFromArray(gray_cropped)
    sitk_mask = sitk.GetImageFromArray(mask_resized)
    sitk_mask.CopyInformation(sitk_img)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape2D")
    extractor.enableFeatureClassByName("firstorder")

    features = extractor.execute(sitk_img, sitk_mask)

    # Extract metrics & map descriptors
    sphericity = features["original_shape2D_Sphericity"]
    entropy = features["original_firstorder_Entropy"]
    color_mean = features["original_firstorder_Mean"]
    elongation = features.get("original_shape2D_Elongation", 1.0)

    if sphericity > 0.90:
        shape_type = "Round"
    elif sphericity > 0.82:
        shape_type = "Oval"
    else:
        shape_type = "Irregular"

    orientation = "Parallel" if elongation > 0.5 else "Non-parallel"

    if color_mean < 50:
        color_grade = "Hypoechoic (Dark)"
    elif entropy > 4.5:
        color_grade = "Complex/ Mixed (Heterogenous colors)"
    else:
        color_grade = "Isoechoic (Neutral/Mixed)"

    margin_type = (
        "Circumscribed"
        if sphericity > 0.88
        else "Not circumscribed (Indistinct/Angular)"
    )
    echo_pattern = "Homogeneous" if entropy < 3.8 else "Heterogeneous"

    # --- Step 6: Database Lookup & Execution ---
    posterior_features = lookup_posterior_features(image_path, reference_csv)
    patient_id = os.path.splitext(os.path.basename(image_path))[0]

    prompt = f"""
    System: You are an expert Radiologist AI using the ACR BI-RADS Atlas 5th Edition.
    Patient ID: {patient_id}
    Morphological Features:
    - Shape: {shape_type}
    - Margin: {margin_type}
    - Echo pattern: {echo_pattern}
    - Color grade: {color_grade}
    - Orientation: {orientation}
    - Posterior Features: {posterior_features}

    Rules :
    - Category 1: Negative
    - Category 2: Benign (Oval/Round & Parallel & Circumscribed & Enhancement & Lighter/ Isoechoic)
    - Category 3: Probably Benign, some doubts/ not definitive
    - Category 4: Suspicious (Irregular OR Not circumscribed & Non-parallel & Shadow & Dark/Mixed colors)
    - Category 5: Highly Suggestive (Irregular AND Not circumscribed & Non-parallel & Shadow & Dark/ Mixed colors)
    - Category 0: Incomplete scan or artifacts 

    Task: Assign a Category (0, 1, 2, 3, 4, or 5).
    Response Format:
    Category: [Score]
    Reasoning: [One sentence, including all important morphological features]
    """

    response = ollama.generate(model="llama3", prompt=prompt)
    return response["response"]


if __name__ == "__main__":
    # 1. Initialize Cropper Engine once
    cropper = ImageCropper("ground_truth_boxes.npy")

    # 2. Configure paths for ANY targeted file you wish to run
    img_name = "malignant (104).png"
    target_csv = "malignant_features.csv"  # matches the file type

    # Derive mask location dynamically matching your folder pattern
    mask_name = img_name.replace(".png", "_mask.png")
    mask_path_target = f"/Users/zozo/Desktop/unimib/SIPH/Project 2/malignant_mask/{mask_name}"

    try:
        # Run execution pipeline
        result = process_and_classify_image(
            image_path=img_name,
            mask_path=mask_path_target,
            reference_csv=target_csv,
            cropper_obj=cropper,
        )

        print("\n--- AI Classification Result ---")
        print(result)

    except Exception as e:
        print(f"\nExecution Error: {e}")