"""
Pose detection wrapper built on MediaPipe Pose.

The detector returns normalized landmark coordinates plus pixel coordinates for
keypoints such as wrists, hips, and shoulders. These points are critical for
reasoning about hand-object interactions and body zones.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp

from config import Config


@dataclass
class PoseLandmark:
    name: str
    visibility: float
    x: float
    y: float


@dataclass
class PoseResult:
    person_id: int
    landmarks: Dict[str, PoseLandmark]
    bounding_box: Tuple[int, int, int, int]


class PoseDetector:
    """MediaPipe pose estimator with helper utilities."""

    def __init__(self, config: Config):
        self.config = config
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=self.config.model.pose_detection_confidence,
            min_tracking_confidence=self.config.model.pose_tracking_confidence,
        )

    def infer(self, frame) -> List[PoseResult]:
        """Run pose detection on a frame and return landmark data."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        output: List[PoseResult] = []

        if not results.pose_landmarks:
            return output

        height, width = frame.shape[:2]
        landmarks = results.pose_landmarks.landmark
        key_names = mp.solutions.pose.PoseLandmark

        named_landmarks: Dict[str, PoseLandmark] = {}
        xs, ys = [], []
        for key in key_names:
            lm = landmarks[key.value]
            x_px = int(lm.x * width)
            y_px = int(lm.y * height)
            named_landmarks[key.name.lower()] = PoseLandmark(
                name=key.name.lower(), visibility=lm.visibility, x=x_px, y=y_px
            )
            xs.append(x_px)
            ys.append(y_px)

        bbox = (
            max(min(xs), 0),
            max(min(ys), 0),
            min(max(xs), width - 1),
            min(max(ys), height - 1),
        )

        # person_id will be assigned by tracker; start with -1 placeholder
        output.append(PoseResult(person_id=-1, landmarks=named_landmarks, bounding_box=bbox))
        return output


LANDMARK_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
