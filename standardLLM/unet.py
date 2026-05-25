import os
import re
import cv2
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

DO_TRAIN = False
DO_PLOT_CURVES = False
DO_EVAL = True
DO_VISUALIZE = False
DO_EXTRACT_METRICS = False
DO_CONTOUR_OVERLAY = True

# dataset
class SegmentationDataset(Dataset):
    def __init__(self, root, size=256, limit=None, augment=False):
        self.samples = []
        self.size = size
        self.augment = augment

        mask_re = re.compile(r'^(.+?)_mask(?:_\d+)?\.png$', re.IGNORECASE)

        for cls in ["benign", "malignant"]:
            img_dir = os.path.join(root, cls)
            mask_dir = os.path.join(root, f"{cls}_mask")

            for fname in os.listdir(mask_dir):
                m = mask_re.match(fname)
                if not m:
                    continue

                base_name = m.group(1)
                img_name = f"{base_name}.png"

                img_path = os.path.join(img_dir, img_name)
                mask_path = os.path.join(mask_dir, fname)

                if os.path.exists(img_path):
                    self.samples.append((img_path, mask_path))

        if limit:
            self.samples = self.samples[:limit]

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        # resize
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        # normalize
        img = img / 255.0
        mask = (mask > 127).astype(np.float32)

        # to tensor
        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0)

        return img, mask

#augmentation for training only!
class AugmentedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, augment=False):
        self.subset = subset
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, mask = self.subset[idx]

        if self.augment:
            img = img.numpy().transpose(1, 2, 0)
            mask = mask.numpy().squeeze()

            h, w = img.shape[:2]

            # horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

            # rotation
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-35, 35)
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)

                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
                mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)


            # gaussian blur
            if np.random.rand() > 0.5:
                img = cv2.GaussianBlur(img, (5, 5), 0)

            img = torch.tensor(img).permute(2, 0, 1).float()
            mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# unet model
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


# loss + metrics
def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def dice_score(pred, target, smooth=1):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def calculate_areaPerimeter(mask):
    mask = (mask > 0).astype(np.uint8)

    area = np.sum(mask)

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    perimeter = np.sum(cv2.filter2D(mask, -1, kernel) < 8)

    return area, perimeter

def save_results(results, filename="results.csv"):
    keys = results[0].keys()
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

# training and evaluating
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    bce_fn = nn.BCEWithLogitsLoss()

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)

        bce = bce_fn(preds, masks)
        d_loss = dice_loss(preds, masks)

        loss = bce + d_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def run_training(model, train_loader, val_loader, optimizer, device):
    best_val_dice = 0
    train_losses = []
    train_dices = []
    val_dices = []

    intRange = 30

    for epoch in range(intRange):
        train_loss = train(model, train_loader, optimizer, device)

        train_dice, train_iou = evaluate(model, train_loader, device)
        val_dice, val_iou = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)

        print(f"""
        Epoch {epoch + 1}
        Train Loss: {train_loss:.4f}
        Train Dice: {train_dice:.4f}, IoU: {train_iou:.4f}
        Val   Dice: {val_dice:.4f}, IoU: {val_iou:.4f}
        """)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best model with Dice {best_val_dice:.4f}")

    return train_losses, train_dices, val_dices

def evaluate(model, loader, device):
    model.eval()
    dice_total = 0
    iou_total = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)

            dice_total += dice_score(preds, masks).item()
            iou_total += iou_score(preds, masks).item()

    dice_avg = dice_total / len(loader)
    iou_avg = iou_total / len(loader)

    return dice_avg, iou_avg


# visualization
def visualize(model, dataset, device, idx=None):
    model.eval()

    if idx is None:
        idx = np.random.randint(len(dataset))

    img, mask = dataset[idx]

    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))
        pred = torch.sigmoid(pred).cpu().squeeze().numpy()

    plt.figure(figsize=(10,3))

    plt.subplot(1, 3, 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title("GT Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(pred > 0.5, cmap='gray')
    plt.title("Prediction")

    plt.show()

def get_prediction(model, img, device):
    model.eval()
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))
        pred = torch.sigmoid(pred).cpu().squeeze().numpy()
    return (pred > 0.5).astype(np.uint8)


def predict_and_get_contours(model, img_path, mask_path, device, size=256):
    # load image + GT mask
    img = cv2.imread(img_path)
    gt_mask = cv2.imread(mask_path, 0)

    orig = img.copy()

    # preprocess image
    img_resized = cv2.resize(img, (size, size))
    img_norm = img_resized / 255.0
    img_tensor = torch.tensor(img_norm).permute(2, 0, 1).float()

    # predict
    pred_mask = get_prediction(model, img_tensor, device)

    # resize both masks back to original size
    pred_mask = cv2.resize(pred_mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_mask = cv2.resize(gt_mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_mask = (gt_mask > 127).astype(np.uint8)

    # find contours
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return orig, pred_mask, gt_mask, pred_contours, gt_contours

def predict_on_bbox_and_map_back(model, img, bbox, device, size=256):
    x, y, w, h = bbox
    H, W = img.shape[:2]

    # clip bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after clipping: {(x1, y1, x2, y2)}")

    # crop
    roi = img[y1:y2, x1:x2]

    # preprocess
    roi_resized = cv2.resize(roi, (size, size))
    roi_norm = roi_resized / 255.0
    roi_tensor = torch.tensor(roi_norm).permute(2, 0, 1).float().to(device)

    # predict
    model.eval()
    with torch.no_grad():
        pred = model(roi_tensor.unsqueeze(0))
        pred = torch.sigmoid(pred).cpu().squeeze().numpy()

    pred_mask = (pred > 0.5).astype(np.uint8)

    # resize back to bbox
    pred_mask = cv2.resize(pred_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    # place into full image
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = pred_mask

    return full_mask

def show_overlay(img, pred_mask, gt_mask, pred_contours, gt_contours):
    overlay = img.copy()

    # draw GT in green
    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)

    # draw prediction in blue
    cv2.drawContours(overlay, pred_contours, -1, (255, 0, 0), 2)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(1,3,2)
    plt.imshow(gt_mask, cmap='gray')
    plt.title("Ground Truth Mask")

    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Contours (Green=GT, Blue=Pred)")

    plt.show()

SINGLE_IMAGE_PATH = "/Users/zozo/Desktop/unimib/SIPH/Project 2_cropped/benign/benign (91).png"
SINGLE_MASK_PATH  = "/Users/zozo/Desktop/unimib/SIPH/Project 2_cropped/benign_mask/benign (91)_mask.png"

if __name__ == "__main__":
    dataset_root = "/Users/zozo/Desktop/unimib/SIPH/Project 2_cropped"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # set limit for initial training
    #LIMIT = 10
    #dataset = SegmentationDataset(dataset_root, augment=False, limit=LIMIT)
    dataset = SegmentationDataset(dataset_root, augment=False)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(42)

    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_dataset = AugmentedSubset(train_subset, augment=True)
    val_dataset = AugmentedSubset(val_subset, augment=False)
    test_dataset = AugmentedSubset(test_subset, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses, train_dices, val_dices = [], [], []

    if DO_TRAIN:
        train_losses, train_dices, val_dices = run_training(
            model, train_loader, val_loader, optimizer, device
        )
        torch.save(model.state_dict(), "final_model.pth")
    else:
        if os.path.exists("best_model.pth"):
            model.load_state_dict(torch.load("best_model.pth"))
        else:
            print("No saved model found. Train first.")

    # extracting morphological metrics: area and perimeter
    if DO_EXTRACT_METRICS:
        results = []

        for i in range(len(test_dataset)):
            img, _ = test_dataset[i]

            pred_mask = get_prediction(model, img, device)

            area, perimeter = calculate_areaPerimeter(pred_mask)

            results.append({
                "id": i,
                "area": int(area),
                "perimeter": float(perimeter)
            })

        save_results(results)

    # plotting curves for training loss and dice
    if DO_PLOT_CURVES and len(train_losses) > 0:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title("Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(train_dices, label="Train Dice")
        plt.plot(val_dices, label="Val Dice")
        plt.title("Dice Score")
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.legend()

        plt.show()


    # test evaluation
    if DO_EVAL:
        test_dice, test_iou = evaluate(model, test_loader, device)
        print(f"\nFINAL TEST → Dice: {test_dice:.4f}, IoU: {test_iou:.4f}")

    # visualization
    if DO_VISUALIZE:
        for i in range(5):
            visualize(model, test_dataset, device)

    #visualizing individual contours
    if DO_CONTOUR_OVERLAY:
        img, pred_mask, gt_mask, pred_c, gt_c = predict_and_get_contours(
            model,
            SINGLE_IMAGE_PATH,
            SINGLE_MASK_PATH,
            device
        )

        show_overlay(img, pred_mask, gt_mask, pred_c, gt_c)