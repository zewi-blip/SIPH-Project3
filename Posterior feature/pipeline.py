import torch
import numpy as np 
from posterior_features import PosteriorFeatureExtractor

class DetectionPipeline:

    def __init__(self, yolo_model, refinement_model, device):
        self.yolo = yolo_model
        self.refiner = refinement_model
        self.device = device

        self.yolo.eval()
        self.refiner.eval()
        self.extractor = PosteriorFeatureExtractor(posterior_ratio=0.5)

    # -------------------------
    # YOLO Prediction
    # -------------------------
    def run_yolo(self, image):
        results = self.yolo(image, device="cpu")[0]

        boxes = []

        for r in results.boxes:
            x1, y1, x2, y2 = r.xyxy[0].cpu().numpy()
            conf = r.conf[0].cpu().item()

            h, w = image.shape[:2]

            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            boxes.append([xc, yc, bw, bh, conf])

        return boxes

    # -------------------------
    # Refinement
    # -------------------------
    def refine_boxes(self, boxes):
        if len(boxes) == 0:
            return []

        feats = torch.tensor(boxes, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred = self.refiner(feats)

        delta = pred[:, :4]
        new_conf = torch.sigmoid(pred[:, 4])

        refined = feats[:, :4] + delta

        refined_boxes = []

        for i in range(refined.shape[0]):
            refined_boxes.append([
                refined[i, 0].item(),
                refined[i, 1].item(),
                refined[i, 2].item(),
                refined[i, 3].item(),
                new_conf[i].item()
            ])

        return refined_boxes
    
    def image_bbx(self, boxes, image_shape):
        h, w = image_shape[:2]

        converted = []  
        for b in boxes:
          xc, yc, bw, bh, conf = b
          x0 = (xc - bw/2) * w
          y0 = (yc - bh/2) * h
          x1 = (xc + bw/2) * w
          y1 = (yc + bh/2) * h  
          converted.append(np.array([x0, y0, x1, y1]))

        return converted
    # -------------------------
    # Full pipeline
    # -------------------------
    def predict(self, image):
        # YOLO
        yolo_boxes = self.run_yolo(image)

        # Refinement
        refined_boxes = self.refine_boxes(yolo_boxes)

        # Convert
        final_boxes = self.image_bbx(refined_boxes, image.shape)
        
        # Extract Posterior Feature
        all_features = [self.extractor.extract(image, box) for box in final_boxes]

        return all_features