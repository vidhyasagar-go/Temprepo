"""
YOLO-based object detection wrapper using the ultralytics package.

The detector returns bounding boxes, class labels, and confidence scores. These
outputs are consumed by the tracker and shoplifting logic.
"""
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import Config


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str


class YOLODetector:
    """Thin wrapper around ultralytics YOLO for object detection."""

    def __init__(self, config: Config):
        self.config = config
        self.model = YOLO(self.config.model.yolo_model_path)
        self.model.fuse()

    def infer(self, frame) -> List[Detection]:
        """Run detection on a frame and return parsed Detection objects."""
        results = self.model.predict(
            source=frame,
            conf=self.config.model.yolo_confidence,
            iou=self.config.model.yolo_iou,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections
        result = results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf.cpu().item())
            cls_id = int(box.cls.cpu().item())
            label = result.names.get(cls_id, str(cls_id))
            detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf, label=label))
        return detections
