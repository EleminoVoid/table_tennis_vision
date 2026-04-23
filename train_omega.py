"""
Train YOLO12 on augmented dataset v5 for small object detection.
Output model: table_tennis_ball_yolo12_omega.pt
"""

from ultralytics import YOLO
import os
import torch

def train_omega_model():
    print("=" * 70)
    print("YOLO12 OMEGA - SMALL OBJECT DETECTION MODEL TRAINING")
    print("=" * 70)
    
    # Dataset
    dataset_path = "training/table_tennis_ball_dataset/data.yaml"
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found: {dataset_path}")
        return False
    
    print(f"\n📊 Dataset: {dataset_path}")
    
    # Load base model (prefer local checkpoints that exist in this workspace)
    base_model_candidates = [
        "models/table_tennis_ball_yolo12.pt",
        "runs/detect/table_tennis_models/ball_detection_yolo12/weights/best.pt",
        "runs/detect/table_tennis_models/ball_detection_v1/weights/best.pt",
    ]
    base_model_path = next((p for p in base_model_candidates if os.path.exists(p)), None)
    if base_model_path is None:
        print("ERROR: No base model found for Omega training")
        return False

    print(f"📦 Base model: {base_model_path}")
    model = YOLO(base_model_path)
    
    # Training config
    epochs = 100
    batch_size = 10
    imgsz = 640
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        print("ERROR: CUDA is not available in this environment. Aborting Omega training.")
        return False

    device = 0
    workers = 3
    patience = 20  # Early stopping
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   - Model: YOLO12 Nano")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Image size: {imgsz}")
    print(f"   - Device: CUDA {device} ({torch.cuda.get_device_name(device)})")
    print(f"   - Workers: {workers}")
    print(f"   - Early stopping patience: {patience}")
    
    print(f"\n🚀 Starting training...")
    print("=" * 70)
    
    # Train
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=workers,
        patience=patience,
        project="runs/detect",
        name="table_tennis_ball_yolo12_omega",
        exist_ok=False,
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        perspective=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        crop_fraction=1.0,
        val=True,
        split=None,
        save=True,
        save_period=10,
        verbose=True,
    )
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    
    # Copy best weights to models folder
    best_weights = "runs/detect/table_tennis_ball_yolo12_omega/weights/best.pt"
    if os.path.exists(best_weights):
        import shutil
        output_path = "models/table_tennis_ball_yolo12_omega.pt"
        shutil.copy(best_weights, output_path)
        print(f"\n📦 Model saved: {output_path}")
    
    return True

if __name__ == "__main__":
    train_omega_model()
