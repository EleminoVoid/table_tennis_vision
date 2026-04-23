# Training Workflow (Safe + Faster)

This project trains a YOLOv12 ball detector from `download_and_train.py`.

## What the script does

1. (Optional) Downloads dataset from Roboflow.
2. Finds `data.yaml` under `training/table_tennis_ball_dataset`.
3. Trains `yolo12n.pt` for 100 epochs.
4. Writes run artifacts under `runs/detect/table_tennis_models/ball_detection_yolo12`.
5. Copies best checkpoint to `models/table_tennis_ball_yolo12.pt`.

## Current training mode in this repo

Configured in `download_and_train.py`:

- `device=0` → force CUDA GPU (no silent CPU fallback).
- `batch=10` → safer VRAM usage but faster than ultra-safe batch sizes.
- `workers=3` → moderate dataloader parallelism.
- `cache=False` → avoids large disk/RAM cache spikes.
- `save_period=1` → saves each epoch (better recovery if interrupted).
- `verbose=False` + `TQDM_ASCII=1` → cleaner console/progress bars on Windows.

## Correct environment

Use `venv_tt` for CUDA training.

- `.venv` has CPU-only torch in this workspace.
- `venv_tt` has CUDA torch.

## Run commands

From repo root:

```powershell
# optional: stop old runs
Get-Process -Name python,pythonw -ErrorAction SilentlyContinue | Stop-Process -Force

# start training (reuse existing dataset)
C:/Users/312_Lab/Documents/GitHub/table_tennis_vision/venv_tt/Scripts/python.exe download_and_train.py --skip-download
```

## If training is interrupted

- Check latest logs in run folder and `runs/detect/table_tennis_models/ball_detection_yolo12/weights`.
- Because `save_period=1` is enabled, you should have recent epoch checkpoints.
- If instability continues, lower to `batch=8` and `workers=2` in `download_and_train.py`.

## Generate a Word report with Claude (best automation path)

This repo includes `generate_training_report.py`, which:

1. Reads latest training log (or a log you pass).
2. Reads current training settings from `download_and_train.py`.
3. Sends context to Claude via Anthropic API.
4. Writes a `.docx` report in `reports/`.

Setup:

```powershell
# in venv_tt (or any env with deps installed)
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY="your_key_here"
```

Where to get the key:

1. Go to `https://console.anthropic.com/`.
2. Sign in and open API Keys in the console.
3. Create a new key and copy it once.
4. Set it in your terminal session:

```powershell
$env:ANTHROPIC_API_KEY="your_new_key"
```

Optional (persist for future PowerShell sessions):

```powershell
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "your_new_key", "User")
```

Run:

```powershell
python generate_training_report.py
```

Free option (no Anthropic credits/API key required):

```powershell
python generate_training_report.py --local-only
```

Or use the launcher (it auto-falls back to local mode if `ANTHROPIC_API_KEY` is not set):

```powershell
.\generate_report.bat
```

Optional custom run:

```powershell
python generate_training_report.py --log-file training_log_safe_20260318.txt --output reports/my_report.docx
```
