"""
Preprocessing utilities for grain disease classification
Handles validation splits, augmentation, and image transformations
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Constants
MAIZE_CATEGORIES = ["0_NOR", "1_F&S", "2_SD", "3_MY", "4_AP", "5_BN", "6_HD", "7_IM"]
RICE_CATEGORIES = ["0_NOR", "1_F&S", "2_SD", "3_MY", "4_AP", "5_BN", "6_UN", "7_IM"]

CATEGORIES_MAP = {
    "maize": MAIZE_CATEGORIES,
    "rice": RICE_CATEGORIES
}



""" required to have a validation split in the specs """
def create_validation_split(grain_type, val_ratio=0.15, random_state=42, restore_original=False):
    """
    Create validation set by splitting training data.
    
    Args:
        grain_type (str): Either 'maize' or 'rice'
        val_ratio (float): Proportion of training data to use for validation (default: 0.15)
        random_state (int): Random seed for reproducibility
        restore_original (bool): If True, moves validation images back to training
    
    Returns:
        dict: Statistics about the split for each category
    """
    base_path = Path(f"./dataset/images/{grain_type}")
    train_path = base_path / "train"
    val_path = base_path / "val"
    
    categories = CATEGORIES_MAP[grain_type]
    split_stats = {}
    
    if restore_original:
        print(f"Restoring validation split for {grain_type}...")
        if not val_path.exists():
            print(f"No validation directory found for {grain_type}")
            return {}
        
        for category in categories:
            category_val = val_path / category
            category_train = train_path / category
            
            if not category_val.exists():
                continue
            
            # Move all validation images back to training
            val_images = list(category_val.glob("*.png"))
            for img in val_images:
                shutil.move(str(img), str(category_train / img.name))
            
            print(f"  {category}: Moved {len(val_images)} images back to training")
        
        # Remove validation directory
        shutil.rmtree(val_path)
        print(f"Validation split restored for {grain_type}\n")
        return {}
    
    # Create validation split
    print(f"Creating validation split for {grain_type}...")
    val_path.mkdir(exist_ok=True)
    
    for category in categories:
        category_train = train_path / category
        category_val = val_path / category
        category_val.mkdir(exist_ok=True, parents=True)
        
        # Get all images in training category
        images = list(category_train.glob("*.png"))
        
        if len(images) == 0:
            print(f"Warning: No images found in {category_train}")
            continue
        
        # Calculate validation size (at least 1 image)
        val_size = max(1, int(len(images) * val_ratio))
        
        # Split into train and validation
        train_imgs, val_imgs = train_test_split(
            images, 
            test_size=val_size, 
            random_state=random_state
        )
        
        # Move validation images
        for img in val_imgs:
            shutil.move(str(img), str(category_val / img.name))
        
        split_stats[category] = {
            'train': len(train_imgs),
            'val': len(val_imgs),
            'original': len(images)
        }
        
        print(f"  {category}: {len(train_imgs)} train, {len(val_imgs)} val (from {len(images)} total)")
    
    print(f"Validation split created for {grain_type}\n")
    return split_stats



""" this is for the neural network part, not needed for classical ML. This is just to 
 standardize the images since we're going to be using raw pixels in the NN. """
def get_augmentation_pipeline(split='train', img_size=224):
    """
    Returns augmentation pipeline for different data splits.
    
    Args:
        split (str): 'train', 'val', or 'test'
        img_size (int): Target image size (default: 224 for standard CNNs)
    
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    if split == 'train':
        return A.Compose([
            # for all splits - resize and normalize
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats

            # for training only - add augmentations and randomize it so that it doesnt end up overfitted
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),    # shifts the hue, saturation and brightness. Bc irl the pictures will be taken in diff lighting conditions
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),                                                # gaussion noise is like random pixels to match imperfections that can happen irl
            
            # NumPy array to a PyTorch tensor bc Pytorch NN use tensors not arrays
            ToTensorV2()
        ])
    else:  # validation or test - no augmentation, only normalization
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])



""" Basically same as the last one but no normalization and tensor conversion, 
its just to visualize stuff in the ntbk and to check if the augmentations are working as expected. """
def get_augmentation_pipeline_no_tensor(split='train', img_size=224):
    """
    Returns augmentation pipeline without tensor conversion (for visualization).
    
    Args:
        split (str): 'train', 'val', or 'test'
        img_size (int): Target image size
    
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    if split == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
        ])



def load_and_preprocess_image(img_path, transform=None):
    """
    Load image and apply transformations.
    
    Args:
        img_path (str or Path): Path to image file
        transform (albumentations.Compose): Transformation pipeline to apply
    
    Returns:
        numpy.ndarray or torch.Tensor: Preprocessed image
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if transform:
        transformed = transform(image=img)
        img = transformed['image']
    
    return img



def get_dataset_statistics(grain_type, splits=['train', 'val', 'test']):
    """
    Get image count statistics for all categories and splits.
    
    Args:
        grain_type (str): Either 'maize' or 'rice'
        splits (list): List of splits to analyze
    
    Returns:
        dict: Nested dictionary with statistics
    """
    categories = CATEGORIES_MAP[grain_type]
    base_path = Path(f"./dataset/images/{grain_type}")
    
    stats = {}
    
    for split in splits:
        split_path = base_path / split
        if not split_path.exists():
            continue
        
        stats[split] = {}
        
        for category in categories:
            category_path = split_path / category
            if category_path.exists():
                count = len(list(category_path.glob("*.png")))
                stats[split][category] = count
            else:
                stats[split][category] = 0
    
    return stats


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("PREPROCESSING UTILITIES")
    print("=" * 60)
    
    # Create validation splits
    print("\n1. Creating validation splits...")
    maize_stats = create_validation_split("maize", val_ratio=0.15, random_state=42)
    rice_stats = create_validation_split("rice", val_ratio=0.15, random_state=42)
    
    # Get dataset statistics
    print("\n2. Dataset statistics:")
    for grain in ["maize", "rice"]:
        print(f"\n{grain.upper()}:")
        stats = get_dataset_statistics(grain, splits=['train', 'val', 'test'])
        for split, counts in stats.items():
            total = sum(counts.values())
            print(f"  {split}: {total} images")