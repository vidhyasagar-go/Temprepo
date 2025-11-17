"""
Real-time shoplifting detection demo.

Usage
-----
1) Install dependencies (preferably inside a virtual environment)::
       pip install -r requirements.txt

2) Run with webcam input::
       python main.py --input-mode webcam

3) Run with RTSP camera::
       python main.py --input-mode rtsp --rtsp-url rtsp://username:password@camera-address/stream

4) Run with local video file::
       python main.py --input-mode file --file-path path/to/video.mp4

5) Run with YouTube demo::
       python main.py --input-mode youtube --youtube-url https://www.youtube.com/watch?v=example
   The script downloads the specified YouTube clip using yt-dlp, saves it locally,
   and then processes it as a normal MP4 file.
"""
import argparse

import cv2

from config import CONFIG, Config
from object_detector import YOLODetector
from pose_detector import LANDMARK_CONNECTIONS, PoseDetector
from shoplifting_logic import ShopliftingLogic, draw_body_zones, draw_shelf_zone
from tracker import SimpleTracker
from video_utils import open_video_capture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time shoplifting detection")
    parser.add_argument("--input-mode", default=None, help="webcam|rtsp|file|youtube")
    parser.add_argument("--rtsp-url", default=None)
    parser.add_argument("--file-path", default=None)
    parser.add_argument("--youtube-url", default=None)
    return parser.parse_args()


def apply_cli_overrides(config: Config, args: argparse.Namespace):
    """Apply CLI overrides to the loaded configuration."""
    if args.input_mode:
        config.input.input_mode = args.input_mode
    if args.rtsp_url:
        config.input.rtsp_url = args.rtsp_url
    if args.file_path:
        config.input.file_path = args.file_path
    if args.youtube_url:
        config.input.youtube_url = args.youtube_url


def draw_pose(frame, pose_result):
    for connection in LANDMARK_CONNECTIONS:
        start = pose_result.landmarks.get(connection[0].name.lower())
        end = pose_result.landmarks.get(connection[1].name.lower())
        if start and end:
            cv2.line(frame, (start.x, start.y), (end.x, end.y), (0, 255, 0), 2)
    for landmark in pose_result.landmarks.values():
        cv2.circle(frame, (landmark.x, landmark.y), 4, (0, 0, 255), -1)
    cv2.putText(frame, f"Person {pose_result.person_id}", (pose_result.bounding_box[0], pose_result.bounding_box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def draw_objects(frame, tracked_objects):
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{obj.label} {obj.confidence:.2f} ID:{obj.id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0), 2)


def main():
    args = parse_args()
    config = CONFIG
    apply_cli_overrides(config, args)

    detector = YOLODetector(config)
    pose_detector = PoseDetector(config)
    tracker = SimpleTracker()
    logic = ShopliftingLogic(config)

    cap, source_desc = open_video_capture(config)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source_desc}")
    print(f"Processing source: {source_desc}")

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or cannot read frame.")
            break
        frame_index += 1

        detections = detector.infer(frame)
        tracked_objects = tracker.update(detections)

        poses = pose_detector.infer(frame)
        tracked_poses = tracker.assign_poses(poses)

        event = logic.evaluate(frame_index, frame.shape, tracked_poses, tracked_objects)
        if event:
            logic.log_event(event)
            cv2.putText(frame, "POSSIBLE SHOPLIFTING DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        draw_shelf_zone(frame, config)
        draw_objects(frame, tracked_objects)
        for pose in tracked_poses:
            draw_pose(frame, pose)
            draw_body_zones(frame, pose, config)

        cv2.imshow("Shoplifting Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
