"""
Augment table tennis ball dataset for improved small object detection.
Creates table_tennis_ball_dataset_v5 with augmented images.
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_yolo_labels(label_path):
    """Load YOLO format labels (class cx cy w h normalized)."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append([class_id, cx, cy, w, h])
    return boxes

def save_yolo_labels(label_path, boxes):
    """Save YOLO format labels."""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, 'w') as f:
        for box in boxes:
            f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

def yolo_to_pascal(boxes, h, w):
    """Convert YOLO normalized format to Pascal VOC (x1, y1, x2, y2) for albumentations."""
    pascal_boxes = []
    for box in boxes:
        class_id, cx, cy, box_w, box_h = box
        x1 = (cx - box_w / 2) * w
        y1 = (cy - box_h / 2) * h
        x2 = (cx + box_w / 2) * w
        y2 = (cy + box_h / 2) * h
        pascal_boxes.append([x1, y1, x2, y2, int(class_id)])
    return pascal_boxes

def pascal_to_yolo(boxes, h, w):
    """Convert Pascal VOC back to YOLO normalized format."""
    yolo_boxes = []
    for box in boxes:
        x1, y1, x2, y2, class_id = box
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        box_w = (x2 - x1) / w
        box_h = (y2 - y1) / h
        yolo_boxes.append([class_id, cx, cy, box_w, box_h])
    return yolo_boxes

def get_augmentation_pipeline():
    """Create augmentation pipeline optimized for small object detection."""
    return A.Compose([
        # Spatial augmentations
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.7),
        A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), p=0.7),
        
        # Brightness/contrast/color for robust detection
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.4),
        
        # Motion blur (simulates ball movement)
        A.MotionBlur(blur_limit=3, p=0.3),
        
        # Small object specific: elastic deformations
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        
        # Gamma correction for varying lighting
        A.RandomGamma(p=0.3),
        
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.1))

def augment_dataset(source_dir, target_dir, augmentations_per_image=2):
    """
    Augment dataset with small object optimizations.
    Creates multiple augmented versions of each image.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Copy structure and non-augmented data
    for split in ['train', 'valid', 'test']:
        target_split_img = target_path / split / 'images'
        target_split_lbl = target_path / split / 'labels'
        target_split_img.mkdir(parents=True, exist_ok=True)
        target_split_lbl.mkdir(parents=True, exist_ok=True)
        
        source_split_img = source_path / split / 'images'
        source_split_lbl = source_path / split / 'labels'
        
        if not source_split_img.exists():
            continue
        
        image_files = sorted([f for f in os.listdir(source_split_img) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        total = len(image_files)
        
        print(f"\nAugmenting {split} split: {total} images...")
        
        for idx, img_file in enumerate(image_files):
            if (idx + 1) % 500 == 0:
                print(f"  Progress: {idx + 1}/{total}")
            
            img_path = source_split_img / img_file
            lbl_file = Path(img_file).stem + '.txt'
            lbl_path = source_split_lbl / lbl_file
            
            # Load image and labels
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            yolo_boxes = load_yolo_labels(str(lbl_path))
            
            if not yolo_boxes:
                # Copy original if no labels
                cv2.imwrite(str(target_split_img / img_file), image)
                if lbl_path.exists():
                    shutil.copy(lbl_path, target_split_lbl / lbl_file)
                continue
            
            pascal_boxes = yolo_to_pascal(yolo_boxes, h, w)
            
            # Save original
            cv2.imwrite(str(target_split_img / img_file), image)
            save_yolo_labels(str(target_split_lbl / lbl_file), yolo_boxes)
            
            # Generate augmented versions
            augmentation = get_augmentation_pipeline()
            for aug_idx in range(augmentations_per_image):
                augmented = augmentation(image=image, bboxes=pascal_boxes)
                aug_image = augmented['image']
                aug_boxes = augmented['bboxes']
                
                if not aug_boxes:  # Skip if augmentation removed all boxes
                    continue
                
                # Convert back to YOLO format
                aug_yolo_boxes = pascal_to_yolo(aug_boxes, h, w)
                
                # Save augmented image and labels
                aug_stem = Path(img_file).stem
                suffix = Path(img_file).suffix
                aug_name = f"{aug_stem}_aug{aug_idx + 1}{suffix}"
                
                cv2.imwrite(str(target_split_img / aug_name), aug_image)
                save_yolo_labels(str(target_split_lbl / f"{aug_stem}_aug{aug_idx + 1}.txt"), aug_yolo_boxes)
        
        print(f"  {split} complete!")
    
    # Copy data.yaml
    yaml_src = source_path / 'data.yaml'
    yaml_dst = target_path / 'data.yaml'
    if yaml_src.exists():
        shutil.copy(yaml_src, yaml_dst)
    
    print(f"\n✓ Augmented dataset created: {target_dir}")

if __name__ == "__main__":
    source_dataset = "training/table_tennis_ball_dataset"
    target_dataset = "training/table_tennis_ball_dataset_v5_augmented"
    
    print("=" * 60)
    print("DATASET AUGMENTATION FOR SMALL OBJECT DETECTION")
    print("=" * 60)
    print(f"Source: {source_dataset}")
    print(f"Target: {target_dataset}")
    print(f"Augmentations per image: 2")
    print("=" * 60)
    
    if not os.path.exists(source_dataset):
        print(f"ERROR: Source dataset not found: {source_dataset}")
        exit(1)
    
    # Clean target if exists
    if os.path.exists(target_dataset):
        print(f"\n{target_dataset} exists. Replacing...")
        shutil.rmtree(target_dataset)
    
    augment_dataset(source_dataset, target_dataset, augmentations_per_image=2)
    
    print("\nAugmentation complete!")
    print("Next steps:")
    print("1. Update data.yaml path in code if needed")
    print("2. Retrain YOLO12 with: python download_and_train.py")
