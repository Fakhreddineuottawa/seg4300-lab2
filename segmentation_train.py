import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# 1) DATASET
class HouseDataset(Dataset):
    """
    Expects:
        train/images/*.png|.jpg|.jpeg|.tif|.tiff
        train/masks/*.png|.jpg|.jpeg|.tif|.tiff
    """

    def __init__(self, images_dir, masks_dir,
                 image_transform=None, mask_transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(valid_exts)
        ])
        self.mask_files = sorted([
            f for f in os.listdir(masks_dir)
            if f.lower().endswith(valid_exts)
        ])

        # Ensure 1:1 match
        assert len(self.image_files) == len(self.mask_files), (
            f"Number of images ({len(self.image_files)}) and masks ({len(self.mask_files)}) must match!"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        msk_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")  # => shape [3,H,W] after transforms
        mask  = Image.open(msk_path).convert("L")    # => shape [1,H,W] after transforms

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = mask.squeeze(0)
        mask = (mask > 0.5).long()  # Binarize (0,1)

        return image, mask


# 2) METRICS
def dice_score(pred, target, eps=1e-7):
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice

def iou_score(pred, target, eps=1e-7):
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    return iou


# 3) MODEL
class SimpleUNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleUNet, self).__init__()
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
        x1 = self.encoder1(x)         
        p1 = self.pool1(x1)           

        x2 = self.encoder2(p1)        
        p2 = self.pool2(x2)           

        u1 = self.up1(p2)             
        c1 = torch.cat([u1, x2], dim=1)  
        d1 = self.decoder1(c1)        

        u2 = self.up2(d1)             
        c2 = torch.cat([u2, x1], dim=1)  
        d2 = self.decoder2(c2)
        return d2


# 4) TRAINING
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    mask_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    # Single dataset for both train & "val" in this script
    dataset = HouseDataset(
        images_dir="./train/images",
        masks_dir="./train/masks",
        image_transform=image_transform,
        mask_transform=mask_transform
    )
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SimpleUNet(num_classes=2).to(device)

    # Lower LR => smaller updates, more stable
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # INCREASE EPOCHS to 20
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # => [B,2,H,W]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        # Evaluate on the same data
        model.eval()
        dice_vals, iou_vals = [], []
        with torch.no_grad():
            for images, masks in data_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                dice_vals.append(dice_score(preds, masks).item())
                iou_vals.append(iou_score(preds, masks).item())

        mean_dice = np.mean(dice_vals)
        mean_iou = np.mean(iou_vals)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Loss: {epoch_loss:.4f}, Dice: {mean_dice:.4f}, IoU: {mean_iou:.4f}")

    torch.save(model.state_dict(), "segmentation_model.pt")
    print("Training complete. Weights saved as segmentation_model.pt")


if __name__ == "__main__":
    train_model()
