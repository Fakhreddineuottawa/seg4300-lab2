import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


# 1) DATASET: Loads images + masks
class HouseDataset(Dataset):
    """
    Expects:
        train/images/*.jpg (or .png)
        train/masks/*.png
        val/images/*.jpg
        val/masks/*.png
    Each mask is a BINARY segmentation image: 0 = background, 1 = house.
    """

    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Gather file names
        self.image_files = sorted([
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.mask_files = sorted([
            f for f in os.listdir(masks_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Ensure 1:1 match
        assert len(self.image_files) == len(self.mask_files), (
            "Number of images and masks must match!"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Paths
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        msk_path = os.path.join(self.masks_dir, self.mask_files[idx])

        # Open image/mask
        image = Image.open(img_path).convert("RGB")  # shape: [3, H, W]
        mask  = Image.open(msk_path).convert("L")    # shape: [1, H, W] after transform

        # Apply transforms
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # mask is [1, H, W], convert to [H, W], then threshold => 0 or 1
        mask = mask.squeeze(0)           # remove channel dimension => [H, W]
        mask = (mask > 0.5).long()       # 0 or 1

        return image, mask


# 2) METRICS: Dice & IoU
def dice_score(pred, target, eps=1e-7):
    """
    Dice = 2 * (intersection) / (union).
    pred, target: shape [B, H, W], values in {0,1}
    """
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice

def iou_score(pred, target, eps=1e-7):
    """
    IoU = intersection / union
    pred, target: shape [B, H, W], 0 or 1
    """
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    return iou


# 3) MODEL: A Simple U-Net for Binary Segmentation
class SimpleUNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleUNet, self).__init__()

        # --- Encoder ---
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        # --- Decoder ---
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(16 + 32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(8 + 16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # encoder stage 1
        x1 = self.encoder1(x)         # [B,16,H,W]
        p1 = self.pool1(x1)           # [B,16,H/2,W/2]

        # encoder stage 2
        x2 = self.encoder2(p1)        # [B,32,H/2,W/2]
        p2 = self.pool2(x2)           # [B,32,H/4,W/4]

        # up stage 1
        u1 = self.up1(p2)             # [B,16,H/2,W/2]
        c1 = torch.cat([u1, x2], dim=1)  # => 16 + 32 = 48
        d1 = self.decoder1(c1)        # => [B,16,H/2,W/2]

        # up stage 2
        u2 = self.up2(d1)             # [B,8,H,W]
        c2 = torch.cat([u2, x1], dim=1)  # => 8 + 16 = 24
        d2 = self.decoder2(c2)        # => [B,num_classes,H,W]

        return d2


# 4) TRAINING LOOP
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # A) Transforms for images/masks
    image_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    mask_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),  # => shape [1,H,W]
    ])

    # B) Create Datasets
    train_dataset = HouseDataset(
        images_dir="./train/images",
        masks_dir ="./train/masks",
        image_transform=image_transform,
        mask_transform=mask_transform
    )
    val_dataset = HouseDataset(
        images_dir="./val/images",
        masks_dir ="./val/masks",
        image_transform=image_transform,
        mask_transform=mask_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False)

    # C) Initialize Model
    model = SimpleUNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()  # For binary classification with "2" classes
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # D) Train
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)     # [B,3,256,256]
            masks  = masks.to(device)      # [B,256,256]

            optimizer.zero_grad()
            outputs = model(images)        # => [B,2,256,256]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # E) Validate
        model.eval()
        dice_vals = []
        iou_vals  = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks  = masks.to(device)
                outputs = model(images)     # [B,2,256,256]
                preds = torch.argmax(outputs, dim=1)  # => [B,256,256]

                dice_vals.append(dice_score(preds, masks).item())
                iou_vals.append(iou_score(preds, masks).item())

        mean_dice = np.mean(dice_vals)
        mean_iou  = np.mean(iou_vals)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Loss: {epoch_loss:.4f}, Dice: {mean_dice:.4f}, IoU: {mean_iou:.4f}")

    # F) Save final model
    torch.save(model.state_dict(), "segmentation_model.pt")
    print("Training complete. Weights saved as segmentation_model.pt")


# 5) MAIN
if __name__ == "__main__":
    train_model()
