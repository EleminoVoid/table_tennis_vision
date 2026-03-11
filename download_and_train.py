"""
Download Table Tennis Ball Dataset from Roboflow and Train YOLO Model

Dataset: https://universe.roboflow.com/kevins-workspace-8zhhu/balls-per-frame/dataset/7
Images:  14384 (12581 train / 1211 val / 592 test)
Augmentation: 3x per image, ±15° rotation, ±20° hue (applied by Roboflow)

Usage:
    python download_and_train.py                         # uses built-in API key
    python download_and_train.py --api-key YOUR_KEY      # override API key
    python download_and_train.py --skip-download         # re-train on existing dataset
    python download_and_train.py --skip-train            # download only

Get your API key at: https://app.roboflow.com/settings/api
"""

import argparse
import os
import sys
import shutil

# ── Roboflow project identifiers ─────────────────────────────────────────────
RF_WORKSPACE = "kevins-workspace-8zhhu"
RF_PROJECT   = "balls-per-frame"
RF_VERSION   = 7
RF_FORMAT    = "yolov12"         # YOLOv12 format – same YAML structure as v8/v11

# Default API key for this project (override via --api-key)
DEFAULT_API_KEY = "bLl0HTnmgwktbDHzGYGc"

DATASET_DIR  = "training/table_tennis_ball_dataset"
MODEL_DIR    = "table_tennis_models"

# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(api_key: str) -> str:
    """Download dataset from Roboflow and return path to data.yaml."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow is not installed. Run: pip install roboflow")
        sys.exit(1)

    print("=" * 55)
    print("  Connecting to Roboflow …")
    print(f"  Workspace : {RF_WORKSPACE}")
    print(f"  Project   : {RF_PROJECT}")
    print(f"  Version   : {RF_VERSION}")
    print(f"  Format    : {RF_FORMAT}")
    print("=" * 55)

    rf      = Roboflow(api_key=api_key)
    project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    version = project.version(RF_VERSION)

    os.makedirs(DATASET_DIR, exist_ok=True)

    # Download – Roboflow SDK returns a Dataset object whose .location tells us
    # the exact folder it wrote to (may differ from the location= arg).
    dataset = version.download(RF_FORMAT, location=DATASET_DIR, overwrite=True)

    # Prefer the SDK-reported location, then fall back to a full tree search
    search_roots = []
    if hasattr(dataset, "location") and dataset.location:
        search_roots.append(dataset.location)
    search_roots.append(DATASET_DIR)
    search_roots.append(".")          # last-resort: whole working directory

    yaml_path = None
    for root in search_roots:
        yaml_path = _find_yaml(root)
        if yaml_path:
            break

    if yaml_path is None:
        print("ERROR: data.yaml not found after download.")
        print(f"  Searched: {search_roots}")
        sys.exit(1)

    print(f"\n✅ Dataset downloaded → {yaml_path}")

    # v6 already has a proper train/val/test split from Roboflow – only
    # call the auto-splitter as a safety net if val is somehow missing.
    _ensure_val_split(yaml_path)

    return yaml_path


def _find_yaml(root: str) -> "str | None":
    """Recursively walk root and return the first data.yaml found."""
    for dirpath, _, files in os.walk(root):
        if "data.yaml" in files:
            return os.path.join(dirpath, "data.yaml")
    return None


def _ensure_val_split(yaml_path: str, val_fraction: float = 0.15) -> None:
    """
    If the dataset has no validation images, move val_fraction of the training
    images (and their labels) into the valid/ split so YOLO can evaluate.
    """
    import random, glob

    dataset_dir = os.path.dirname(yaml_path)
    val_img_dir = os.path.join(dataset_dir, "valid", "images")
    val_lbl_dir = os.path.join(dataset_dir, "valid", "labels")

    # Count existing val images
    existing_val = glob.glob(os.path.join(val_img_dir, "*")) if os.path.isdir(val_img_dir) else []
    if existing_val:
        print(f"  Val split already present ({len(existing_val)} images). Skipping auto-split.")
        return

    train_img_dir = os.path.join(dataset_dir, "train", "images")
    train_lbl_dir = os.path.join(dataset_dir, "train", "labels")

    if not os.path.isdir(train_img_dir):
        print("  WARNING: Could not find train/images – skipping auto val-split.")
        return

    all_images = glob.glob(os.path.join(train_img_dir, "*"))
    if not all_images:
        print("  WARNING: No training images found – skipping auto val-split.")
        return

    random.seed(42)
    random.shuffle(all_images)
    n_val = max(1, int(len(all_images) * val_fraction))
    val_images = all_images[:n_val]

    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    moved = 0
    for img_path in val_images:
        fname   = os.path.basename(img_path)
        stem    = os.path.splitext(fname)[0]
        lbl_src = os.path.join(train_lbl_dir, stem + ".txt")

        shutil.move(img_path, os.path.join(val_img_dir, fname))
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, os.path.join(val_lbl_dir, stem + ".txt"))
        moved += 1

    print(f"  Auto val-split: moved {moved} images ({val_fraction*100:.0f}%) → valid/")
    print(f"  Remaining train images: {len(all_images) - moved}")


def train_model(yaml_path: str):
    """Train YOLOv12 on the downloaded dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n" + "=" * 55)
    print("  Starting YOLOv12 training …")
    print(f"  data.yaml : {yaml_path}")
    print(f"  Epochs    : 100")
    print(f"  Image size: 640")
    print(f"  Batch     : 32")
    print(f"  Cache     : disk (preprocessed cache for faster epochs)")
    print(f"  Motion blur augmentation: ON")
    print("=" * 55 + "\n")

    # Start with a pretrained YOLOv12 nano model (fastest to fine-tune)
    model = YOLO("yolo12n.pt")

    results = model.train(
        data        = yaml_path,
        epochs      = 100,
        imgsz       = 640,
        batch       = 32,        # RTX 3080 10 GB handles 32 easily for nano
        conf        = 0.05,      # low threshold – table tennis ball is small
        iou         = 0.4,
        augment     = True,
        mixup       = 0.1,
        degrees     = 10,
        scale       = 0.3,
        fliplr      = 0.5,
        mosaic      = 1.0,
        # ── Speed: cache preprocessed images to disk ─────────────
        # Epoch 1 is slower (writing cache), every subsequent epoch
        # reads from the .cache file → faster than re-reading raw images.
        # Use "ram" only if you have >20 GB free system RAM.
        cache       = "disk",
        # ── Motion blur augmentation ─────────────────────────────
        # RandAugment applies blur, sharpness, contrast shifts randomly
        # which teaches the model to handle the motion-blurred ball.
        auto_augment = "randaugment",
        # ─────────────────────────────────────────────────────────
        patience    = 15,        # early stopping
        save        = True,
        project     = MODEL_DIR,
        name        = "ball_detection_yolo12",
        exist_ok    = True,
    )

    best_weights = os.path.join(results.save_dir, "weights", "best.pt")
    dest         = "models/table_tennis_ball_yolo12.pt"

    if os.path.exists(best_weights):
        os.makedirs("models", exist_ok=True)
        shutil.copy(best_weights, dest)
        print(f"\n✅ Training complete!")
        print(f"   Best weights saved to → {dest}")
        print(f"   Full run artifacts    → {results.save_dir}")
        print(f"\n   mAP50     : {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   mAP50-95  : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    else:
        print("WARNING: best.pt not found – check training output.")

    return dest


def main():
    parser = argparse.ArgumentParser(description="Download Roboflow dataset and train table tennis ball detector")
    parser.add_argument("--api-key",       required=False, default=DEFAULT_API_KEY,
                        help="Roboflow API key (default: project key)")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use existing dataset")
    parser.add_argument("--skip-train",    action="store_true", help="Download only, do not train")
    args = parser.parse_args()

    if not args.skip_download and not args.api_key:
        parser.error("--api-key is required unless --skip-download is set")

    # ── Step 1: Download ────────────────────────────────────────────────────
    if args.skip_download:
        yaml_path = _find_yaml(DATASET_DIR)
        if yaml_path is None:
            print(f"ERROR: No data.yaml found in {DATASET_DIR}. Remove --skip-download to fetch it.")
            sys.exit(1)
        print(f"Skipping download. Using existing dataset: {yaml_path}")
    else:
        yaml_path = download_dataset(args.api_key)

    if args.skip_train:
        print("Skipping training (--skip-train). Done.")
        return

    # ── Step 2: Train ───────────────────────────────────────────────────────
    model_path = train_model(yaml_path)

    # ── Step 3: Remind user ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  NEXT STEPS")
    print("=" * 55)
    print(f"1. Test the model interactively:")
    print(f"     python test_and_tune_models.py")
    print(f"2. Run the full table tennis analysis:")
    print(f"     python table_tennis_main.py")
    print(f"3. Validate the pipeline:")
    print(f"     python validate_pipeline.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
