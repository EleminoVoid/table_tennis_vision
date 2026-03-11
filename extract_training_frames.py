"""
Video Frame Extraction for Training Data Collection

Extracts frames from table tennis videos to build a training dataset.

Usage: python extract_training_frames.py <video_path> [frame_interval]
Example: python extract_training_frames.py input_videos/input_video.mp4 15
"""
import cv2
import os
import sys

def extract_frames(video_path, output_dir="collected_data/raw_images", frame_interval=15):
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}  |  FPS: {fps:.1f}  |  Duration: {duration:.1f}s")
    print(f"Extracting every {frame_interval} frames -> ~{total_frames // frame_interval} images")
    print(f"Output folder: {output_dir}")
    print()

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"{video_name}_frame_{frame_count:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            extracted_count += 1

            if extracted_count % 20 == 0:
                pct = frame_count / total_frames * 100 if total_frames > 0 else 0
                print(f"  {extracted_count} frames extracted ({pct:.0f}%)...")

        frame_count += 1

    cap.release()
    print(f"\nDone! {extracted_count} frames saved to '{output_dir}'")
    print(f"Next step: annotate images for ball detection using Roboflow or LabelImg.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_training_frames.py <video_path> [frame_interval]")
        print("Example: python extract_training_frames.py input_videos/input_video.mp4 15")
        sys.exit(1)

    video_path = sys.argv[1]
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    extract_frames(video_path, frame_interval=interval)

if __name__ == "__main__":
    main()
