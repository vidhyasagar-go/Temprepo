"""
Real-time shoplifting detection demo.

Setup
-----
1. Install dependencies: `pip install -r requirements.txt`
2. Optional GPU support follows Ultralytics/MediaPipe guidance.

Running
-------
* Webcam:    set `config.input.mode = "webcam"` in config.py then `python main.py`
* RTSP:      set mode to "rtsp" and provide `config.input.rtsp_url`
* File:      set mode to "file" and point `config.input.video_path` to an MP4
* YouTube:   set mode to "youtube"; the script downloads the clip via yt-dlp and processes locally.

Notes
-----
* Pose landmarks are mapped to pixel coordinates to measure distances between hands and objects.
* Hand-object proximity is computed using Euclidean distance between wrist landmarks and object centers.
* Shelf zone is defined as a rectangle (top-left and bottom-right) in pixel coordinates from config.py.
* Torso/pocket zones are computed relative to each person bounding box.
"""
from __future__ import annotations

import logging
from typing import Dict

import cv2

from config import config
from object_detector import YOLODetector
from pose_detector import PoseDetector
from shoplifting_logic import ShopliftingLogic
from tracker import SimpleTracker
from video_utils import open_capture


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.output_log, mode="a"),
    ],
)


class ShopliftingApp:
    """Main application coordinating detection, tracking, and rule logic."""

    def __init__(self) -> None:
        self.pose_detector = PoseDetector()
        self.obj_detector = YOLODetector()
        self.person_tracker = SimpleTracker()
        self.object_tracker = SimpleTracker()
        self.logic = ShopliftingLogic()

    def run(self) -> None:
        cap = open_capture()
        if not cap.isOpened():
            raise RuntimeError("Unable to open video source. Check configuration.")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if 0 < config.max_frames < frame_idx:
                break

            persons_raw = self.pose_detector.detect(frame)
            person_boxes = [p["bbox"] for p in persons_raw]
            tracked_person_boxes = self.person_tracker.update(person_boxes)
            persons: Dict[int, Dict] = {}
            for pid, (bbox, det_idx) in tracked_person_boxes.items():
                landmarks = persons_raw[det_idx]["landmarks"] if persons_raw else {}
                persons[pid] = {"bbox": bbox, "landmarks": landmarks}

            detections = self.obj_detector.detect(frame)
            object_boxes = [d["bbox"] for d in detections]
            tracked_object_boxes = self.object_tracker.update(object_boxes)
            objects: Dict[int, Dict] = {}
            for oid, (bbox, det_idx) in tracked_object_boxes.items():
                label = detections[det_idx]["label"] if detections else None
                conf = detections[det_idx]["conf"] if detections else None
                objects[oid] = {"bbox": bbox, "label": label, "conf": conf}

            new_events = self.logic.update(frame_idx, persons, objects)
            self._draw(frame, persons, objects, new_events)

            cv2.imshow("Shoplifting Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def _draw(self, frame, persons: Dict[int, Dict], objects: Dict[int, Dict], events) -> None:
        # Draw detections and pose
        self.logic.draw_overlays(frame, persons, objects)
        self.pose_detector.draw(frame, list(persons.values()))
        for oid, obj in objects.items():
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = obj.get("label", "obj")
            cv2.putText(frame, f"{label} OID {oid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Highlight suspicious events
        if events:
            cv2.putText(
                frame,
                "POSSIBLE SHOPLIFTING DETECTED",
                (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )
            for event in events:
                pdata = persons.get(event.person_id)
                if pdata:
                    x1, y1, x2, y2 = pdata["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                logging.info(
                    "[Frame %s] Person %s | Object %s | %s",
                    event.frame_idx,
                    event.person_id,
                    event.object_label,
                    event.reason,
                )


def main() -> None:
    app = ShopliftingApp()
    app.run()


if __name__ == "__main__":
    main()
