"""
YOLO-based object detector module.
Uses Ultralytics YOLOv8 for detecting retail objects such as boxes, bottles, etc.
"""
from typing import List, Tuple, Dict

import cv2
import numpy as np
try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - external dependency
    raise RuntimeError(
        "Ultralytics is required for YOLO detection. Install via `pip install ultralytics`."
    ) from exc

from config import config


class YOLODetector:
    """Wrapper around Ultralytics YOLO model for inference."""

    def __init__(self) -> None:
        self.model = YOLO(config.model.yolo_model)
        self.conf = config.model.yolo_conf_threshold
        self.iou = config.model.yolo_iou_threshold

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run object detection on a frame.

        Returns a list of dict entries with keys: id (optional), label, conf, bbox (x1,y1,x2,y2).
        """
        results = self.model.predict(source=frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        detections: List[Dict] = []
        if results.boxes is None:
            return detections
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.model.names.get(cls_id, str(cls_id))
            detections.append(
                {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": conf,
                    "label": label,
                }
            )
        return detections

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Dict], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["conf"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label}:{conf:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return frame
