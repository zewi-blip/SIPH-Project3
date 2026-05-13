    
import torch
from ultralytics import YOLO
import cv2
from models.refinement_net import RefinementNet
from pipeline import DetectionPipeline

YOLO_PATH = 'C:/Users/Tarek Hisham/Desktop/Posterior feature/models/best.pt'
REFINE_PATH = 'C:/Users/Tarek Hisham/Desktop/Posterior feature/models/refinement weights.pth'
refinement_model = RefinementNet(5)
yolo_model = YOLO(YOLO_PATH)
refinement_model.load_state_dict(torch.load(REFINE_PATH, map_location="cpu"))


def main():

    pipeline = DetectionPipeline(yolo_model, refinement_model, device=None)

    # read the image 
    image = cv2.imread('C:/Users/Tarek Hisham/Desktop/Posterior feature/benign (1).png')

    # Predicted box in format x_topleft, y_topleft, width, height, conf
    features = pipeline.predict(image)
    
    for fe in features:
        print(f'Enhancement conf:{fe['enhancement_score']}, Shadow conf:{fe['shadow_score']}')

if __name__ == "__main__":
    main()