import sys
from pathlib import Path

_POSTERIOR_DIR = Path(__file__).resolve().parent.parent / "Posterior feature"
if str(_POSTERIOR_DIR) not in sys.path:
    sys.path.insert(0, str(_POSTERIOR_DIR))

import torch
from ultralytics import YOLO
import cv2
from models.refinement_net import RefinementNet
from pipeline import DetectionPipeline

YOLO_PATH = _POSTERIOR_DIR / "models" / "best.pt"
REFINE_PATH = _POSTERIOR_DIR / "models" / "refinement weights.pth"
refinement_model = RefinementNet(5)
yolo_model = YOLO(str(YOLO_PATH))
refinement_model.load_state_dict(torch.load(str(REFINE_PATH), map_location="cpu"))


def extract_posterior_features(img_path):

    pipeline = DetectionPipeline(yolo_model, refinement_model, device=None)

    image = cv2.imread(img_path)

    # Predicted box in format x_topleft, y_topleft, width, height, conf
    features = pipeline.predict(image)

    if len(features) > 1:
        features = [max(features, key=lambda f: f["shadow_score"])]

    #for fe in features:
    #    print(f'Enhancement conf:{fe["enhancement_score"]}, Shadow conf:{fe["shadow_score"]}')
    
    return features



if __name__ == "__main__":
    extract_posterior_features(str(_POSTERIOR_DIR / "benign (1).png"))
