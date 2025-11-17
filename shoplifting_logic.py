"""
Rule-based shoplifting detection combining pose landmarks, object tracks, and
predefined zones like shelves and body regions.
"""
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config
from pose_detector import PoseResult
from tracker import TrackedObject


@dataclass
class SuspicionEvent:
    frame_index: int
    person_id: int
    object_id: int
    object_label: str
    reason: str


class ShopliftingLogic:
    """Implements heuristics to flag potential shoplifting behavior."""

    def __init__(self, config: Config):
        self.config = config
        self.touch_counters: Dict[Tuple[int, int], int] = {}
        self.body_hold_counters: Dict[Tuple[int, int], int] = {}
        self.disappearance_counters: Dict[Tuple[int, int], int] = {}
        self.last_positions: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def _is_inside_shelf(self, point: Tuple[int, int], frame_shape) -> bool:
        height, width = frame_shape[:2]
        (sx1, sy1), (sx2, sy2) = self.config.geometry.shelf_zone
        x1, y1 = int(sx1 * width), int(sy1 * height)
        x2, y2 = int(sx2 * width), int(sy2 * height)
        x, y = point
        return x1 <= x <= x2 and y1 <= y <= y2

    def _compute_body_zones(self, pose: PoseResult) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Estimate torso and pocket rectangles from pose landmarks."""
        hips = [pose.landmarks.get(name) for name in ["left_hip", "right_hip"] if pose.landmarks.get(name)]
        shoulders = [pose.landmarks.get(name) for name in ["left_shoulder", "right_shoulder"] if pose.landmarks.get(name)]
        if not hips or not shoulders:
            return (0, 0, 0, 0), (0, 0, 0, 0)

        hip_x = int(np.mean([h.x for h in hips]))
        hip_y = int(np.mean([h.y for h in hips]))
        shoulder_x = int(np.mean([s.x for s in shoulders]))
        shoulder_y = int(np.mean([s.y for s in shoulders]))
        torso_width = int(abs(shoulders[0].x - shoulders[-1].x) * (1 + self.config.geometry.torso_zone_padding))
        torso_height = abs(shoulder_y - hip_y)
        x1 = shoulder_x - torso_width // 2
        x2 = shoulder_x + torso_width // 2
        y1 = shoulder_y - int(self.config.geometry.torso_zone_padding * torso_height)
        y2 = hip_y + int(self.config.geometry.torso_zone_padding * torso_height)
        torso_zone = (x1, y1, x2, y2)

        pocket_y1 = hip_y
        pocket_y2 = hip_y + int(self.config.geometry.pocket_zone_height * torso_height)
        pocket_zone = (x1, pocket_y1, x2, pocket_y2)
        return torso_zone, pocket_zone

    def _point_in_rect(self, point: Tuple[int, int], rect: Tuple[int, int, int, int]) -> bool:
        x, y = point
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def evaluate(
        self,
        frame_index: int,
        frame_shape,
        poses: List[PoseResult],
        objects: List[TrackedObject],
    ) -> Optional[SuspicionEvent]:
        """Assess the current frame for suspicious behaviors."""
        wrist_names = ["left_wrist", "right_wrist"]
        object_centers = {obj.id: ((obj.bbox[0] + obj.bbox[2]) // 2, (obj.bbox[1] + obj.bbox[3]) // 2) for obj in objects}

        for pose in poses:
            torso_zone, pocket_zone = self._compute_body_zones(pose)
            wrists = [pose.landmarks[name] for name in wrist_names if name in pose.landmarks]
            for obj in objects:
                center = object_centers[obj.id]
                for wrist in wrists:
                    distance = float(np.linalg.norm(np.array([wrist.x, wrist.y]) - np.array(center)))
                    key = (pose.person_id, obj.id)

                    # Touch detection near shelf
                    if self._is_inside_shelf(center, frame_shape) and distance < self.config.geometry.max_hand_object_distance:
                        self.touch_counters[key] = self.touch_counters.get(key, 0) + 1
                    else:
                        self.touch_counters[key] = max(self.touch_counters.get(key, 0) - 1, 0)

                    # Track object leaving shelf after touch
                    if self.touch_counters.get(key, 0) >= self.config.logic.min_frames_touch and not self._is_inside_shelf(center, frame_shape):
                        self.body_hold_counters[key] = self.body_hold_counters.get(key, 0) + 1
                    else:
                        self.body_hold_counters[key] = max(self.body_hold_counters.get(key, 0) - 1, 0)

                    # Check proximity to torso/pocket
                    if self._point_in_rect(center, torso_zone) or self._point_in_rect(center, pocket_zone):
                        self.disappearance_counters[key] = self.disappearance_counters.get(key, 0) + 1
                    else:
                        self.disappearance_counters[key] = max(self.disappearance_counters.get(key, 0) - 1, 0)

                    self.last_positions[key] = center

                    # Trigger event when object stays near torso/pocket after leaving shelf
                    if (
                        self.body_hold_counters.get(key, 0) >= self.config.logic.min_frames_body_hold
                        or self.disappearance_counters.get(key, 0) >= self.config.logic.disappearance_frames
                    ):
                        return SuspicionEvent(
                            frame_index=frame_index,
                            person_id=pose.person_id,
                            object_id=obj.id,
                            object_label=obj.label,
                            reason="Object removed from shelf and moved towards torso/pocket",
                        )
        return None

    def log_event(self, event: SuspicionEvent):
        """Append event information to the log file and print to console."""
        message = (
            f"{datetime.datetime.now().isoformat()} | frame {event.frame_index} | "
            f"person {event.person_id} | object {event.object_id} ({event.object_label}) | {event.reason}"
        )
        print(message)
        with open(self.config.logic.log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def draw_shelf_zone(frame, config: Config):
    """Draw the configured shelf rectangle on the frame."""
    h, w = frame.shape[:2]
    (sx1, sy1), (sx2, sy2) = config.geometry.shelf_zone
    x1, y1 = int(sx1 * w), int(sy1 * h)
    x2, y2 = int(sx2 * w), int(sy2 * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
    cv2.putText(frame, "Shelf Zone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)


def draw_body_zones(frame, pose: PoseResult, config: Config):
    """Visualize torso and pocket zones for debugging."""
    torso_zone, pocket_zone = ShopliftingLogic(config)._compute_body_zones(pose)
    cv2.rectangle(frame, (torso_zone[0], torso_zone[1]), (torso_zone[2], torso_zone[3]), (255, 0, 255), 1)
    cv2.rectangle(frame, (pocket_zone[0], pocket_zone[1]), (pocket_zone[2], pocket_zone[3]), (0, 255, 255), 1)
