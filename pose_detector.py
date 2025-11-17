"""
MediaPipe-based pose estimation module for extracting body landmarks.
"""
from typing import Dict, List, Tuple

import cv2
import numpy as np
try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - external dependency
    raise RuntimeError(
        "MediaPipe is required for pose estimation. Install via `pip install mediapipe`."
    ) from exc

from config import config


class PoseDetector:
    """Runs MediaPipe Pose to obtain full-body landmarks."""

    def __init__(self) -> None:
        self.pose = mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=config.model.pose_confidence,
            min_tracking_confidence=config.model.pose_confidence,
        )
        self.drawing = mp.solutions.drawing_utils
        self.pose_connections = mp.solutions.pose.POSE_CONNECTIONS

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect human pose landmarks in a frame.

        Returns list of dicts with keys: id (assigned by caller), landmarks (dict of name->(x,y)), bbox.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        persons: List[Dict] = []
        if not results.pose_landmarks:
            return persons
        h, w, _ = frame.shape
        landmarks = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = (int(lm.x * w), int(lm.y * h))
        xs = [p[0] for p in landmarks.values()]
        ys = [p[1] for p in landmarks.values()]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        persons.append({"landmarks": landmarks, "bbox": bbox})
        return persons

    def draw(self, frame: np.ndarray, persons: List[Dict]) -> np.ndarray:
        """Draw detected pose landmarks and skeleton."""
        for person in persons:
            landmark_list = [None] * len(self.pose_connections)
            for idx, (x, y) in person["landmarks"].items():
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            for start, end in self.pose_connections:
                if start in person["landmarks"] and end in person["landmarks"]:
                    cv2.line(
                        frame,
                        person["landmarks"][start],
                        person["landmarks"][end],
                        (255, 255, 0),
                        2,
                    )
            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        return frame


def get_named_landmarks(landmarks: Dict[int, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    """Map MediaPipe landmark indices to friendly names used in downstream logic."""
    names = {
        0: "nose",
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        23: "left_hip",
        24: "right_hip",
        25: "left_knee",
        26: "right_knee",
        27: "left_ankle",
        28: "right_ankle",
    }
    output: Dict[str, Tuple[int, int]] = {}
    for idx, name in names.items():
        if idx in landmarks:
            output[name] = landmarks[idx]
    return output
