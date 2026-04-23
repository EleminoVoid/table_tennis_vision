"""
Quick Table Tennis Analysis
Runs ball + player detection and saves an annotated output video.
Does NOT require the keypoint model.

Usage:
    ./venv_tt/Scripts/python.exe run_analysis.py
    ./venv_tt/Scripts/python.exe run_analysis.py --input input_videos/input_video.mp4
"""

import cv2
import os
import argparse
from collections import deque
from ultralytics import YOLO
from utils import read_video, save_video


def iou(box1, box2):
    """Calculate Intersection over Union between two boxes [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def filter_duplicate_detections(boxes, iou_threshold=0.3):
    """Remove duplicate/overlapping detections, keeping highest confidence."""
    if not boxes:
        return []
    
    # Sort by confidence descending
    sorted_boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    kept = []
    
    for current in sorted_boxes:
        is_duplicate = False
        for kept_box in kept:
            bbox1 = current[:4]
            bbox2 = kept_box[:4]
            if iou(bbox1, bbox2) > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept.append(current)
    
    return kept


def interpolate_missing_centers(centers, max_gap: int = 12):
    interpolated = list(centers)
    n = len(interpolated)

    known_indices = [i for i, c in enumerate(interpolated) if c is not None]
    if len(known_indices) < 2:
        return interpolated

    for k in range(len(known_indices) - 1):
        start_idx = known_indices[k]
        end_idx = known_indices[k + 1]
        gap = end_idx - start_idx - 1
        if gap <= 0 or gap > max_gap:
            continue

        x0, y0 = interpolated[start_idx]
        x1, y1 = interpolated[end_idx]
        for j in range(1, gap + 1):
            t = j / (gap + 1)
            x = int(round(x0 + t * (x1 - x0)))
            y = int(round(y0 + t * (y1 - y0)))
            interpolated[start_idx + j] = (x, y)

    return interpolated


def run(input_path: str, output_path: str, ball_conf: float = 0.1, show_players: bool = False, tail_length: int = 20,
        interpolate_ball: bool = True, interpolation_max_gap: int = 12):
    print(f"Reading video: {input_path}")
    frames, fps = read_video(input_path)
    print(f"  {len(frames)} frames loaded @ {fps:.1f} fps")

    # ── Ball model ──────────────────────────────────────────────────
    ball_model_candidates = [
        "models/table_tennis_ball_yolo12.pt",
        # "models/table_tennis_ball_yolo12_v4_backup.pt",
        # "models/table_tennis_ball_yolo12_v3_backup.pt",
        # "models/table_tennis_ball.pt",
        "table_tennis_models/ball_detection_v1/weights/best.pt",
        "runs/detect/table_tennis_models/ball_detection_yolo12/weights/best.pt",
        "runs/detect/table_tennis_models/ball_detection_v1/weights/best.pt",
    ]
    ball_model_path = next((p for p in ball_model_candidates if os.path.exists(p)), None)

    if ball_model_path:
        print(f"Ball model: {ball_model_path}")
        ball_model = YOLO(ball_model_path)
    else:
        print("WARNING: No ball model found — skipping ball detection")
        ball_model = None

    # ── Player model (optional) ─────────────────────────────────────
    if show_players:
        print("Player model: yolov8x (COCO, person class)")
        player_model = YOLO("yolov8x.pt")
    else:
        print("Player model: disabled (ball-focused mode)")
        player_model = None

    player_boxes_by_frame = []
    ball_boxes_by_frame = []
    ball_centers_by_frame = []

    for i, frame in enumerate(frames):
        if i % 30 == 0:
            print(f"  Detecting frame {i}/{len(frames)} ...")

        # Player detection (class 0 = person)
        frame_player_boxes = []
        if player_model is not None:
            player_results = player_model(frame, classes=[0], conf=0.5, verbose=False)
            for r in player_results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    frame_player_boxes.append((x1, y1, x2, y2, conf))

        # Ball detection
        best_ball_center = None
        frame_ball_boxes = []
        if ball_model:
            ball_results = ball_model(frame, conf=ball_conf, verbose=False)
            raw_detections = []
            for r in ball_results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    raw_detections.append((x1, y1, x2, y2, conf))
            
            # Filter duplicates: remove overlapping detections, keep highest confidence
            filtered_boxes = filter_duplicate_detections(raw_detections, iou_threshold=0.3)
            
            # Use best detection (already sorted by confidence)
            if filtered_boxes:
                x1, y1, x2, y2, conf = filtered_boxes[0]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                best_ball_center = (cx, cy)
                frame_ball_boxes = [(x1, y1, x2, y2, conf, cx, cy)]

        player_boxes_by_frame.append(frame_player_boxes)
        ball_boxes_by_frame.append(frame_ball_boxes)
        ball_centers_by_frame.append(best_ball_center)

    if interpolate_ball:
        print(f"Applying ball interpolation (max gap: {interpolation_max_gap} frames)")
        render_centers = interpolate_missing_centers(ball_centers_by_frame, max_gap=interpolation_max_gap)
    else:
        render_centers = ball_centers_by_frame

    output_frames = []
    ball_tail = deque(maxlen=max(2, tail_length))
    for i, frame in enumerate(frames):
        if i % 30 == 0:
            print(f"  Rendering frame {i}/{len(frames)} ...")

        annotated = frame.copy()

        for x1, y1, x2, y2, conf in player_boxes_by_frame[i]:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated, f"Player {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        for x1, y1, x2, y2, conf, cx, cy in ball_boxes_by_frame[i]:
            cv2.circle(annotated, (cx, cy), 8, (0, 255, 255), -1)
            cv2.circle(annotated, (cx, cy), 8, (0, 200, 200), 2)
            cv2.putText(annotated, f"Ball {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        center = render_centers[i]
        ball_tail.append(center)
        tail_points = list(ball_tail)
        for j in range(1, len(tail_points)):
            p1 = tail_points[j - 1]
            p2 = tail_points[j]
            if p1 is None or p2 is None:
                continue
            thickness = max(1, int(4 * j / len(tail_points)))
            cv2.line(annotated, p1, p2, (0, 255, 255), thickness)

        if center is not None and not ball_boxes_by_frame[i]:
            cv2.circle(annotated, center, 6, (0, 220, 255), 2)
            cv2.putText(annotated, "Ball interp", (center[0] + 8, center[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        cv2.putText(annotated, f"Frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        output_frames.append(annotated)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving output to: {output_path}")
    save_video(output_frames, output_path, fps=fps)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table Tennis Quick Analysis")
    parser.add_argument("--input", default="input_videos/input_video.mp4",
                        help="Path to input video")
    parser.add_argument("--output", default="output_videos/table_tennis_output.avi",
                        help="Path to output video")
    parser.add_argument("--ball-conf", type=float, default=0.1,
                        help="Ball detection confidence threshold (default: 0.1)")
    parser.add_argument("--show-players", action="store_true",
                        help="Enable player detection overlays (disabled by default for ball-focused output)")
    parser.add_argument("--tail-length", type=int, default=20,
                        help="Ball trail length in frames (default: 20)")
    parser.add_argument("--no-interpolate-ball", action="store_true",
                        help="Disable interpolation of missing ball detections")
    parser.add_argument("--interpolation-max-gap", type=int, default=12,
                        help="Max missing-frame gap to interpolate for ball trajectory (default: 12)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input video not found: {args.input}")
        exit(1)

    run(args.input, args.output, args.ball_conf, args.show_players, args.tail_length,
        interpolate_ball=not args.no_interpolate_ball,
        interpolation_max_gap=args.interpolation_max_gap)
