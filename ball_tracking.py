"""
Ball Tracking – YOLOv12
=======================
Focused pipeline: detect the table tennis ball in every frame using a
YOLOv12 model trained on the Roboflow dataset, interpolate missing frames,
draw a motion trail, and save an annotated video.

Quick start
-----------
1. Download & train the model (one time):
       python download_and_train.py

2. Run ball tracking:
       python ball_tracking.py --input input_videos/input_video.mp4

   The output is saved to output_videos/ by default.

Arguments
---------
  --input        Path to input video  (default: input_videos/input_video.mp4)
  --output       Path to output video (default: output_videos/ball_tracking_output.avi)
  --model        Path to YOLO model   (default: models/table_tennis_ball_yolo12.pt)
  --conf         Detection confidence threshold (default: 0.10)
  --trail        Number of past positions to show as trail (default: 20)
  --no-interp    Disable interpolation of missing detections
  --sharpen      Sharpen each frame before detection (helps with motion blur)
  --sharpen-amt  Unsharp-mask amount, >1 = stronger (default: 1.5)
"""

import argparse
import os
import sys
import collections
import cv2
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Model candidates – checked in order when no --model is given
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_CANDIDATES = [
    "models/table_tennis_ball_yolo12.pt",
    "models/table_tennis_ball.pt",
    "table_tennis_models/ball_detection_yolo12/weights/best.pt",
    "table_tennis_models/ball_detection_v1/weights/best.pt",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_model(path: str | None) -> str:
    """Return model path, checking candidates when path is None."""
    if path:
        if not os.path.exists(path):
            print(f"ERROR: model not found at {path}")
            sys.exit(1)
        return path

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate

    print("ERROR: No ball model found.")
    print("  Train one first:  python download_and_train.py")
    print("  Or specify one:   --model path/to/model.pt")
    sys.exit(1)


def detect_ball_in_frames(model, frames: list, conf: float, device: str = "cpu",
                          sharpen: bool = False, sharpen_amt: float = 1.5) -> list[dict]:
    """
    Run YOLO on every frame.
    Returns a list of dicts: {1: [x1,y1,x2,y2,conf]} or {} when nothing detected.
    Class index 1 is used to match the existing TableTennisBallTracker convention.
    If sharpen=True, applies an unsharp mask before detection to counter motion blur.
    """
    detections = []
    for i, frame in enumerate(frames):
        if i % 60 == 0:
            print(f"  Detecting … frame {i}/{len(frames)}", end="\r", flush=True)

        det_frame = _sharpen_frame(frame, sharpen_amt) if sharpen else frame

        results = model.predict(det_frame, conf=conf, device=device, verbose=False)[0]

        ball_dict = {}
        best_conf = -1.0
        for box in results.boxes:
            c = float(box.conf[0])
            if c > best_conf:           # keep the highest-confidence detection
                best_conf = c
                ball_dict = {1: box.xyxy.tolist()[0] + [c]}   # [x1,y1,x2,y2,conf]

        detections.append(ball_dict)

    print()  # newline after \r progress
    return detections


def _sharpen_frame(frame, amount: float = 1.5):
    """Unsharp mask: sharpened = original + amount*(original - blurred)."""
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=2)
    return cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)


def interpolate_detections(detections: list[dict]) -> list[dict]:
    """
    Physics-aware interpolation of missing ball positions.

    - x (horizontal): linear  – constant horizontal velocity is a good model.
    - y (vertical):   quadratic – gravity bends the trajectory into a parabola.

    For each gap between two detected anchors we gather up to `_FIT_WINDOW`
    real detections on each side and fit a degree-2 polynomial to y, then
    evaluate it at the missing frame indices.  x stays linear.  If too few
    anchor points exist we fall back to linear for y as well.
    """
    _FIT_WINDOW = 5   # how many real detections on each side to use for the fit

    n = len(detections)

    # Build arrays of known centres (frame_idx, cx, cy, box_w, box_h)
    known: list[tuple] = []
    for i, d in enumerate(detections):
        info = d.get(1)
        if info:
            x1, y1, x2, y2 = info[:4]
            known.append((i, (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1))

    if not known:
        return detections

    # Start with a copy; we will fill in missing frames
    result: list[dict | None] = [d.copy() if d else None for d in detections]

    known_idx = [k[0] for k in known]

    # Walk through gaps between consecutive known detections
    for seg in range(len(known) - 1):
        fi, fcx, fcy, fw, fh = known[seg]       # gap start (frame before gap)
        ti, tcx, tcy, tw, th = known[seg + 1]   # gap end

        gap_frames = list(range(fi + 1, ti))     # frames to fill
        if not gap_frames:
            continue

        # ── Horizontal: always linear ──────────────────────────────
        cx_at = lambda f: fcx + (tcx - fcx) * (f - fi) / (ti - fi)  # noqa: E731
        w_at  = lambda f: fw  + (tw  - fw)  * (f - fi) / (ti - fi)  # noqa: E731
        h_at  = lambda f: fh  + (th  - fh)  * (f - fi) / (ti - fi)  # noqa: E731

        # ── Vertical: try quadratic fit over nearby anchor window ──
        lo = max(0, seg - _FIT_WINDOW + 1)
        hi = min(len(known), seg + _FIT_WINDOW + 1)
        window = known[lo:hi]

        if len(window) >= 3:
            wf = np.array([k[0] for k in window], dtype=float)
            wy = np.array([k[2] for k in window], dtype=float)
            try:
                coeffs  = np.polyfit(wf, wy, deg=2)
                cy_at   = lambda f, c=coeffs: float(np.polyval(c, f))  # noqa: E731
            except np.linalg.LinAlgError:
                cy_at = lambda f: fcy + (tcy - fcy) * (f - fi) / (ti - fi)  # noqa: E731
        else:
            cy_at = lambda f: fcy + (tcy - fcy) * (f - fi) / (ti - fi)  # noqa: E731

        for f in gap_frames:
            cx = cx_at(f)
            cy = cy_at(f)
            w  = w_at(f)
            h  = h_at(f)
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            result[f] = {1: [x1, y1, x2, y2, 0.0]}   # conf=0 → marks interpolated

    # Fill any remaining None slots that sit before the first or after the last
    # detection with the nearest known value (bfill / ffill).
    for i in range(n):
        if result[i] is None:
            result[i] = {}

    return result


def draw_trail(frame, trail: collections.deque, color_solid=(0, 255, 255)):
    """Draw a solid connected polyline through all trail positions."""
    pts = list(trail)
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], color_solid, 2, cv2.LINE_AA)


def annotate_frame(frame, ball_info: list | None, trail: collections.deque,
                   frame_idx: int, interpolated: bool = False):
    """Draw ball box, centre dot, trail, and overlay text."""

    # ── Trail ──────────────────────────────────────────────────────
    draw_trail(frame, trail)

    if ball_info is not None:
        x1, y1, x2, y2, conf = ball_info
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Bounding box
        box_color = (0, 180, 255) if interpolated else (0, 255, 255)
        box_label = f"Ball {'[interp]' if interpolated else f'{conf:.2f}'}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, box_label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Centre dot
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 5, (0, 200, 200), 2)

    # ── Frame counter ───────────────────────────────────────────────
    cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str, model_path: str | None,
        conf: float, trail_len: int, interpolate: bool,
        sharpen: bool = False, sharpen_amt: float = 1.5):

    # ── Load model ──────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.  Run: pip install ultralytics")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = _find_model(model_path)
    print(f"Loading model : {model_path}")
    print(f"Device        : {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
          )
    model = YOLO(model_path)

    # ── Load video ──────────────────────────────────────────────────────────
    print(f"Reading video : {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {input_path}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"  {len(frames)} frames  |  {width}×{height}  |  {fps:.1f} fps")

    # ── Detect ──────────────────────────────────────────────────────────────
    print("Running ball detection …")
    raw_detections = detect_ball_in_frames(model, frames, conf, device=device,
                                           sharpen=sharpen, sharpen_amt=sharpen_amt)

    detected = sum(1 for d in raw_detections if d)
    print(f"  Detected in {detected}/{len(frames)} frames "
          f"({100*detected/max(len(frames),1):.1f}%)")

    # ── Interpolate ─────────────────────────────────────────────────────────
    if interpolate:
        print("Interpolating missing detections …")
        detections = interpolate_detections(raw_detections)
    else:
        detections = raw_detections

    # ── Annotate & write output ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trail: collections.deque = collections.deque(maxlen=trail_len)

    print(f"Annotating & writing → {output_path}")
    for i, (frame, det) in enumerate(zip(frames, detections)):
        ball_info = det.get(1)
        was_interpolated = ball_info is not None and raw_detections[i].get(1) is None

        if ball_info is not None:
            x1, y1, x2, y2 = ball_info[:4]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            trail.append((cx, cy))

        annotated = annotate_frame(frame.copy(), ball_info, trail, i,
                                   interpolated=was_interpolated)
        writer.write(annotated)

    writer.release()
    print("Done!")
    print(f"  Output saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Table Tennis Ball Tracking with YOLOv12",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",     default="input_videos/input_video.mp4",
                        help="Input video path")
    parser.add_argument("--output",    default="output_videos/ball_tracking_output.avi",
                        help="Output video path")
    parser.add_argument("--model",     default=None,
                        help="YOLO model .pt file (auto-detected if omitted)")
    parser.add_argument("--conf",      type=float, default=0.10,
                        help="Detection confidence threshold (default: 0.10)")
    parser.add_argument("--trail",     type=int, default=20,
                        help="Ball trail length in frames (default: 20)")
    parser.add_argument("--no-interp", action="store_true",
                        help="Disable interpolation of missing detections")
    parser.add_argument("--sharpen",   action="store_true",
                        help="Sharpen frames before detection (helps with motion blur)")
    parser.add_argument("--sharpen-amt", type=float, default=1.5,
                        help="Unsharp-mask strength, >1 = stronger (default: 1.5)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input video not found: {args.input}")
        sys.exit(1)

    run(
        input_path  = args.input,
        output_path = args.output,
        model_path  = args.model,
        conf        = args.conf,
        trail_len   = args.trail,
        interpolate = not args.no_interp,
        sharpen     = args.sharpen,
        sharpen_amt = args.sharpen_amt,
    )
