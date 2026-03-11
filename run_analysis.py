"""
Quick Table Tennis Analysis
Runs ball + player detection and saves an annotated output video.
Does NOT require the keypoint model.

Usage:
    .\venv_tt\Scripts\python.exe run_analysis.py
    .\venv_tt\Scripts\python.exe run_analysis.py --input input_videos/input_video.mp4
"""

import cv2
import os
import argparse
from ultralytics import YOLO
from utils import read_video, save_video


def run(input_path: str, output_path: str, ball_conf: float = 0.1):
    print(f"Reading video: {input_path}")
    frames = read_video(input_path)
    print(f"  {len(frames)} frames loaded")

    # ── Ball model ──────────────────────────────────────────────────
    ball_model_candidates = [
        "models/table_tennis_ball.pt",
        "table_tennis_models/ball_detection_v1/weights/best.pt",
        "runs/detect/table_tennis_models/ball_detection_v1/weights/best.pt",
    ]
    ball_model_path = next((p for p in ball_model_candidates if os.path.exists(p)), None)

    if ball_model_path:
        print(f"Ball model: {ball_model_path}")
        ball_model = YOLO(ball_model_path)
    else:
        print("WARNING: No ball model found — skipping ball detection")
        ball_model = None

    # ── Player model ────────────────────────────────────────────────
    print("Player model: yolov8x (COCO, person class)")
    player_model = YOLO("yolov8x.pt")

    output_frames = []

    for i, frame in enumerate(frames):
        if i % 30 == 0:
            print(f"  Processing frame {i}/{len(frames)} ...")

        annotated = frame.copy()

        # Player detection (class 0 = person)
        player_results = player_model(frame, classes=[0], conf=0.5, verbose=False)
        for r in player_results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated, f"Player {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        # Ball detection
        if ball_model:
            ball_results = ball_model(frame, conf=ball_conf, verbose=False)
            for r in ball_results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(annotated, (cx, cy), 8, (0, 255, 255), -1)
                    cv2.circle(annotated, (cx, cy), 8, (0, 200, 200), 2)
                    cv2.putText(annotated, f"Ball {conf:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Frame counter
        cv2.putText(annotated, f"Frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        output_frames.append(annotated)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving output to: {output_path}")
    save_video(output_frames, output_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table Tennis Quick Analysis")
    parser.add_argument("--input", default="input_videos/input_video.mp4",
                        help="Path to input video")
    parser.add_argument("--output", default="output_videos/table_tennis_output.avi",
                        help="Path to output video")
    parser.add_argument("--ball-conf", type=float, default=0.1,
                        help="Ball detection confidence threshold (default: 0.1)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input video not found: {args.input}")
        exit(1)

    run(args.input, args.output, args.ball_conf)
