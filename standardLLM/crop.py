
import os
import numpy as np
import cv2
import sys
sys.modules['numpy._core'] = np.core
sys.modules['numpy._core.multiarray'] = np.core.multiarray

import os
import sys
import numpy as np


class ImageCropper:
    def __init__(self, bbox_filename="ground_truth_boxes.npy"):

        if 'numpy._core' not in sys.modules:
            sys.modules['numpy._core'] = np.core
        if 'numpy._core.multiarray' not in sys.modules:
            sys.modules['numpy._core.multiarray'] = np.core.multiarray

        current_dir = os.path.dirname(os.path.abspath(__file__))
        clean_name = os.path.basename(bbox_filename)
        bbox_path = os.path.join(current_dir, clean_name)

        if not os.path.exists(bbox_path):
            raise FileNotFoundError(f"File not found at: {bbox_path}")


        bbox_data = np.load(bbox_path, allow_pickle=True)

        self.bbox_map = {
            os.path.basename(item['image']): item['boxes']
            for item in bbox_data
        }
        print(f"Loaded {len(self.bbox_map)} image bboxes successfully.")

    def crop_by_filename(self, img, mask, filename):
        name_key = os.path.basename(filename)

        if name_key not in self.bbox_map:
            return img, mask

        boxes = self.bbox_map[name_key]

        # Slicing logic
        x_min = int(min([b[0] for b in boxes]))
        y_min = int(min([b[1] for b in boxes]))
        x_max = int(max([b[0] + b[2] for b in boxes]))
        y_max = int(max([b[1] + b[3] for b in boxes]))

        h, w = img.shape[:2]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        return img[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]


cropper = ImageCropper("ground_truth_boxes.npy")

print("\n--- BBOX DATA FORMAT EXAMPLE ---")
first_filename = list(cropper.bbox_map.keys())[0]
first_boxes = cropper.bbox_map[first_filename]

print(f"Filename key: {first_filename}")
print(f"Boxes format: {first_boxes}")
print(f"Type of boxes: {type(first_boxes)}")

img_name = "malignant (9).png"
image_data = cv2.imread(img_name)

mask_path = f"/Users/zozo/Desktop/unimib/SIPH/Project 2/malignant_mask/{img_name.replace('.png', '_mask.png')}"
mask_data = cv2.imread(mask_path, 0)

if image_data is None:
    print(f"Error: Could not find or load the image file: {img_name}")
else:
    cropped_i, cropped_m = cropper.crop_by_filename(image_data, mask_data, img_name)

    cv2.imwrite("cropped_output.png", cropped_i)
    print("Cropped image saved successfully!")