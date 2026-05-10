import torch
import SimpleITK as sitk
import ollama
from radiomics import featureextractor
import cv2
import numpy as np
from unet import UNet

# Load your trained segmentation model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)  # Use your existing UNet class
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()


def predict_single_image(image_path):
    # --- STEP 1: SEGMENTATION ---
    img_orig = cv2.imread(image_path)
    img_resized = cv2.resize(img_orig, (256, 256)) / 255.0
    img_tensor = torch.tensor(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor)).cpu().squeeze().numpy()
    mask_np = (pred > 0.5).astype(np.uint8)

    # --- STEP 2: RADIOMICS EXTRACTION ---
    sitk_img = sitk.ReadImage(image_path)
    if sitk_img.GetNumberOfComponentsPerPixel() > 1:
        sitk_img = sitk.VectorIndexSelectionCast(sitk_img, 0)

    # Resize mask to match original image size for PyRadiomics
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

    prompt = f"""
    Assign BI-RADS Category for a mass with:
    - Shape: {shape} 
    - Margin: {margin} 
    - Sphericity: {sphericity:.2f}
    Return Category (0-6) and Reasoning.
    """

    response = ollama.generate(model='llama3', prompt=prompt)
    return response['response']


# Usage
print(predict_single_image("malignant (173) copy.png"))