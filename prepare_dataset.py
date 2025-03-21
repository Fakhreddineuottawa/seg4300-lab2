import os
import cv2  # if needed, or PIL
import numpy as np
from sklearn.model_selection import train_test_split

def generate_masks_for_aerial_images(input_dir, output_dir):
    """
    Example placeholder for a script that converts aerial images to segmentation masks.
    The real logic depends on your code from "Week 7" (mask generation).
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        # Placeholder for your real house-detection logic
        # For demonstration, assume we produce a random binary mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # your segmentation approach here ...
        
        # Save mask
        mask_path = os.path.join(output_dir, filename.replace('.jpg', '_mask.png'))
        cv2.imwrite(mask_path, mask)

def split_dataset(images_dir, masks_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Splits dataset into train, val, test. 
    images_dir: path to images
    masks_dir: path to corresponding masks
    """
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    image_files.sort()
    
    # Ensure masks are sorted in the same order, or do a matching by name
    mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    mask_files.sort()
    
    # Pair images and masks to keep them matched
    pairs = list(zip(image_files, mask_files))
    
    train_pairs, test_pairs = train_test_split(pairs, test_size=(1 - train_ratio), random_state=42)
    
    # from that train set, split out a val subset
    val_size = int(val_ratio / (1 - (1 - train_ratio)) * len(train_pairs))
    val_pairs = train_pairs[:val_size]
    train_pairs = train_pairs[val_size:]
    
    # create subfolders and copy or record them in text files
    # ...
    # This is just a stub to show you how you'd split them
    print("Train size:", len(train_pairs), "Val size:", len(val_pairs), "Test size:", len(test_pairs))

if __name__ == "__main__":
    input_dir = "./aerial_images"
    output_dir = "./masks"

    generate_masks_for_aerial_images(input_dir, output_dir)
    split_dataset(input_dir, output_dir)
