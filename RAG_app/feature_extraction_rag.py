import sys
import os
import json
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import segmentation_models_pytorch as smp
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import cv2
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)
model.load_state_dict(torch.load(os.path.join(_dir, "seg_model.pt"), map_location=device))
model.to(device)
model.eval()

with open(os.path.join(_dir, "bb_dict.json")) as f:
    _bb_dict = json.load(f)


def _bb_key_from_path(image_path):
    name = os.path.basename(image_path)
    m = re.search(r'(benign|malignant)[^0-9]*(\d+)', name, re.IGNORECASE)
    if m:
        return f"{m.group(1).lower()}_{int(m.group(2)):03d}"
    return None


def predict_single_image(image_path):
    # --- STEP 1: SEGMENTATION ---
    img_orig = cv2.imread(image_path)

    # Crop to bounding box if available
    bb_key = _bb_key_from_path(image_path)
    if bb_key and bb_key in _bb_dict:
        x, y, w, h = _bb_dict[bb_key]
        img_crop = img_orig[y:y + h, x:x + w]
    else:
        img_crop = img_orig

    img_resized = cv2.resize(img_crop, (256, 256)) / 255.0
    img_tensor = torch.tensor(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor)).cpu().squeeze().numpy()
    mask_np = (pred > 0.5).astype(np.uint8)

    # --- STEP 2: RADIOMICS EXTRACTION ---
    sitk_img = sitk.ReadImage(image_path)
    if sitk_img.GetNumberOfComponentsPerPixel() > 1:
        sitk_img = sitk.VectorIndexSelectionCast(sitk_img, 0)

    # Resize mask back to original image size for PyRadiomics
    mask_resized = cv2.resize(mask_np, (img_orig.shape[1], img_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    sitk_mask = sitk.GetImageFromArray(mask_resized)
    sitk_mask.CopyInformation(sitk_img)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('shape2D')
    features = extractor.execute(sitk_img, sitk_mask)

    # --- STEP 3: TRANSLATION & LLM CLASSIFICATION ---
    sphericity = features['original_shape2D_Sphericity']

    # Map to descriptors from Reference Card
    shape = "Oval" if sphericity > 0.8 else "Irregular"
    margin = "Circumscribed" if sphericity > 0.8 else "Not circumscribed"

    additional_info = f"""
    Assign BI-RADS Category for a mass with:
    - Shape: {shape} 
    - Margin: {margin} 
    - Sphericity: {sphericity:.2f}
    Return Category (0-6) and Reasoning.
    """


    return additional_info
