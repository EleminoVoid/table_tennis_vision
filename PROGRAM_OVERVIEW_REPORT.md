# Table Tennis Vision – Program Overview Report

## 1) Purpose

This repository implements a computer-vision pipeline for table tennis video analysis.

At a high level, it:

1. Detects players and ball in video frames.
2. Detects table keypoints (for table geometry reference).
3. Projects player and ball positions onto a mini-table overlay.
4. Estimates simple rally/shot statistics (shot speed, player movement speed).
5. Renders an annotated output video.

---

## 2) Main Entry Points

### `table_tennis_main.py`
Primary full pipeline script.

Flow:
- Reads `input_videos/input_video.mp4`.
- Runs player tracking (`PlayerTracker`) and ball tracking (`TableTennisBallTracker`).
- Loads table keypoint detector (`TableLineDetector`) and predicts keypoints on first frame.
- Filters to two closest players to the table.
- Converts detections to mini-table coordinates.
- Detects ball-shot frames and computes per-shot/per-player speed stats.
- Draws bboxes, table keypoints, mini-table points, stats HUD, and frame index.
- Writes output video to `output_videos/table_tennis_output.avi`.

### `ball_tracking.py`
Ball-only analysis script (YOLOv12-focused).

Features:
- Loads a ball model from candidate paths (or `--model`).
- Optional frame sharpening (`--sharpen`) for motion blur.
- Per-frame ball detection with configurable confidence.
- Physics-aware interpolation for missing detections (linear x, quadratic y fallback).
- Draws box + center + trail and saves annotated output.

### `run_analysis.py`
Quick analysis script.

Features:
- Runs person detection (`yolov8x.pt`) and optional ball detection.
- Draws annotations with less pipeline complexity than `table_tennis_main.py`.
- Does not require keypoint model.

### `download_and_train.py`
Training utility.

Features:
- Downloads Roboflow dataset (or reuses existing with `--skip-download`).
- Trains YOLOv12 model with current safe-fast settings.
- Saves best model to `models/table_tennis_ball_yolo12.pt`.

---

## 3) Core Modules and Responsibilities

## Tracking

### `trackers/player_tracker.py`
- Uses YOLO tracking (`model.track(..., persist=True)`) to keep player IDs over frames.
- Filters detections to class `person`.
- Chooses two relevant players by nearest distance to table keypoints.

### `trackers/table_tennis_ball_tracker.py`
- Ball detector tuned for table tennis (`conf=0.10`).
- Interpolates missing ball boxes using pandas linear interpolation + back/forward fill.
- Detects likely shot/hit frames via direction-change heuristics on smoothed ball trajectory.

### `trackers/ball_tracker.py`
- Earlier generic ball-tracker variant.
- Still exported in `trackers/__init__.py` for compatibility.

## Table Geometry

### `court_line_detector/court_line_detector.py`
- ResNet50-based regression model for table keypoints.
- Outputs 8 keypoints (x,y) pairs; scales predictions from 224x224 back to original frame size.

## Mini-Table Projection

### `mini_court/mini_table_tennis.py`
- Draws a mini table overlay in a corner of each frame.
- Converts world-relative movement to mini-table coordinates using:
  - table dimensions from `constants/__init__.py`,
  - pixel↔meter conversions,
  - player-height-based scaling.
- Projects both player positions and ball position on the mini overlay.

### `mini_court/mini_court.py`
- Legacy tennis-style mini-court implementation retained for compatibility.

## Utilities

### `utils/video_utils.py`
- `read_video`: loads full video into memory as frame list.
- `save_video`: writes frames to AVI (MJPG codec, fixed 24 FPS).

### `utils/bbox_utils.py`
- Bounding-box helpers: center, foot point, bbox height, distances.

### `utils/conversions.py`
- Pixel-to-meter and meter-to-pixel conversion helpers.

### `utils/player_stats_drawer_utils.py`
- Draws a semi-transparent stats HUD with:
  - last shot speed,
  - last player speed,
  - average shot speed,
  - average player speed.

---

## 4) End-to-End Data Flow (Full Pipeline)

1. **Input video** is read to `video_frames`.
2. **Player and ball detections** are generated per frame (or loaded from stubs).
3. **Ball interpolation** fills detection gaps.
4. **Table keypoints** are predicted from first frame.
5. **Player filtering** keeps two table-adjacent players.
6. **Mini-table coordinate mapping** transforms real-frame detections to mini overlay coordinates.
7. **Shot segmentation** identifies ball shot intervals.
8. **Stats computation** estimates speed metrics using pixel→meter conversion.
9. **Rendering** overlays boxes, keypoints, mini table, trajectories, and stats.
10. **Output video** is saved.

---

## 5) Inputs, Models, and Outputs

## Typical Inputs
- Video: `input_videos/input_video.mp4`
- Table keypoint model: `models/keypoints_model.pth`
- Ball model: `models/table_tennis_ball_yolo12.pt` (or candidate fallback paths)

## Typical Outputs
- Full analysis video: `output_videos/table_tennis_output.avi`
- Ball-only video: `output_videos/ball_tracking_output.avi` (default for `ball_tracking.py`)
- Trained ball weights: `models/table_tennis_ball_yolo12.pt`

## Optional cached stubs
- `tracker_stubs/player_detections.pkl`
- `tracker_stubs/ball_detections.pkl`

---

## 6) Important Implementation Notes

- `table_tennis_main.py` currently hardcodes input path and some model paths.
- The main pipeline assumes ball detection exists for each frame after interpolation.
- Stats are derived from frame deltas using fixed FPS assumption (24 FPS).
- `video_utils.read_video` loads the entire video in memory (simple, but memory-heavy for long videos).

---

## 7) Current Risks / Technical Debt

1. **Path and model hardcoding** in entry scripts can reduce portability.
2. **Legacy + current module overlap** (`BallTracker` vs `TableTennisBallTracker`, `MiniCourt` vs `MiniTableTennis`).
3. **Fixed FPS in computations** may be inaccurate for videos with different framerate.
4. **Full-frame in-memory processing** may become a bottleneck on long/high-res videos.
5. **Heuristic shot detection** can misclassify events for unusual trajectories or occlusions.

---

## 8) How to Run (Practical)

## Full pipeline
```powershell
python table_tennis_main.py
```

## Ball-only pipeline
```powershell
python ball_tracking.py --input input_videos/input_video.mp4
```

## Quick analysis
```powershell
python run_analysis.py --input input_videos/input_video.mp4
```

## Train/retrain ball model
```powershell
python download_and_train.py --skip-download
```

---

## 9) Summary

The program is a modular, video-first CV pipeline centered on YOLO detection + geometric projection + lightweight kinematic statistics. The architecture is practical and extensible, with clear separation between tracking, geometry, rendering, and training. The biggest near-term improvements would be parameterization (paths/FPS), consolidation of legacy modules, and streaming processing for large videos.

---

## 10) Methods Used (Function/Method Inventory)

This section lists the key methods currently used by the program.

### `table_tennis_main.py`
- `main`

### `trackers/player_tracker.py` (`PlayerTracker`)
- `__init__`
- `choose_and_filter_players`
- `choose_players`
- `detect_frames`
- `detect_frame`
- `draw_bboxes`

### `trackers/table_tennis_ball_tracker.py` (`TableTennisBallTracker`)
- `__init__`
- `interpolate_ball_positions`
- `get_ball_shot_frames`
- `detect_frames`
- `detect_frame`
- `draw_bboxes`

### `trackers/ball_tracker.py` (`BallTracker`, legacy)
- `__init__`
- `interpolate_ball_positions`
- `get_ball_shot_frames`
- `detect_frames`
- `detect_frame`
- `draw_bboxes`

### `court_line_detector/court_line_detector.py` (`TableLineDetector`)
- `__init__`
- `predict`
- `draw_keypoints`
- `draw_keypoints_on_video`

### `mini_court/mini_table_tennis.py` (`MiniTableTennis`)
- `__init__`
- `convert_meters_to_pixels`
- `set_table_drawing_key_points`
- `set_table_lines`
- `set_mini_table_position`
- `set_canvas_background_box_position`
- `draw_table`
- `draw_background_rectangle`
- `draw_mini_court`
- `get_start_point_of_mini_court`
- `get_width_of_mini_court`
- `get_court_drawing_keypoints`
- `get_mini_court_coordinates`
- `convert_bounding_boxes_to_mini_court_coordinates`
- `draw_points_on_mini_court`

### `ball_tracking.py`
- `_find_model`
- `detect_ball_in_frames`
- `_sharpen_frame`
- `interpolate_detections`
- `draw_trail`
- `annotate_frame`
- `run`

### `run_analysis.py`
- `run`

### `download_and_train.py`
- `download_dataset`
- `_find_yaml`
- `_ensure_val_split`
- `train_model`
- `main`

### `utils/video_utils.py`
- `read_video`
- `save_video`

### `utils/bbox_utils.py`
- `get_center_of_bbox`
- `measure_distance`
- `get_foot_position`
- `get_closest_keypoint_index`
- `get_height_of_bbox`
- `measure_xy_distance`

### `utils/conversions.py`
- `convert_pixel_distance_to_meters`
- `convert_meters_to_pixel_distance`

### `utils/player_stats_drawer_utils.py`
- `draw_player_stats`
