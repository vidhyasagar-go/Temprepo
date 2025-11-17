"""
Rule-based fusion of pose landmarks, object detections, and shelf zones to flag potential shoplifting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from config import config
from pose_detector import get_named_landmarks


@dataclass
class SuspiciousEvent:
    frame_idx: int
    person_id: int
    object_label: str
    reason: str


@dataclass
class PersonState:
    near_object_frames: int = 0
    near_body_frames: int = 0
    last_seen_frame: int = 0
    last_object_label: str = ""
    object_disappeared_near_body: bool = False
    last_object_box: Tuple[int, int, int, int] | None = None


class ShopliftingLogic:
    """Encapsulates rule-based detection of suspicious interactions."""

    def __init__(self) -> None:
        self.events: List[SuspiciousEvent] = []
        self.person_states: Dict[int, PersonState] = {}
        self.shelf_top_left = config.zones.shelf_top_left
        self.shelf_bottom_right = config.zones.shelf_bottom_right

    def _is_in_shelf(self, bbox: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        sx1, sy1 = self.shelf_top_left
        sx2, sy2 = self.shelf_bottom_right
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        return sx1 <= center[0] <= sx2 and sy1 <= center[1] <= sy2

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _torso_zone(self, person_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1
        zone_height = int(height * config.zones.torso_zone_factor)
        mid_y = int((y1 + y2) / 2)
        return (x1, mid_y - zone_height // 2, x2, mid_y + zone_height // 2)

    def _pocket_zone(self, person_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1
        zone_height = int(height * config.zones.pocket_zone_factor)
        hip_y = y1 + int(0.75 * height)
        return (x1, hip_y - zone_height // 2, x2, hip_y + zone_height // 2)

    def update(
        self,
        frame_idx: int,
        persons: Dict[int, Dict],
        objects: Dict[int, Dict],
    ) -> List[SuspiciousEvent]:
        """Update state with current frame detections and return new suspicious events."""
        new_events: List[SuspiciousEvent] = []
        for pid, pdata in persons.items():
            state = self.person_states.setdefault(pid, PersonState())
            state.last_seen_frame = frame_idx
            named = get_named_landmarks(pdata["landmarks"])
            wrists = [named.get("left_wrist"), named.get("right_wrist")]
            torso_zone = self._torso_zone(pdata["bbox"])
            pocket_zone = self._pocket_zone(pdata["bbox"])

            # Check hand-object proximity within shelf
            for obj in objects.values():
                if obj.get("label") is None:
                    continue
                if wrists[0] is None and wrists[1] is None:
                    continue
                hand_close = False
                for wrist in wrists:
                    if wrist is None:
                        continue
                    dist = self._distance(
                        wrist,
                        (
                            (obj["bbox"][0] + obj["bbox"][2]) // 2,
                            (obj["bbox"][1] + obj["bbox"][3]) // 2,
                        ),
                    )
                    if dist < config.thresholds.hand_object_distance:
                        hand_close = True
                        break
                if hand_close and self._is_in_shelf(obj["bbox"]):
                    state.near_object_frames += 1
                    state.last_object_label = obj["label"]
                    state.last_object_box = obj["bbox"]
                    break
            else:
                state.near_object_frames = max(0, state.near_object_frames - 1)

            # Monitor object proximity to torso/pocket
            suspected_obj = state.last_object_box
            if suspected_obj:
                obj_center = (
                    (suspected_obj[0] + suspected_obj[2]) // 2,
                    (suspected_obj[1] + suspected_obj[3]) // 2,
                )
                torso_x1, torso_y1, torso_x2, torso_y2 = torso_zone
                pocket_x1, pocket_y1, pocket_x2, pocket_y2 = pocket_zone
                in_torso = torso_x1 <= obj_center[0] <= torso_x2 and torso_y1 <= obj_center[1] <= torso_y2
                in_pocket = pocket_x1 <= obj_center[0] <= pocket_x2 and pocket_y1 <= obj_center[1] <= pocket_y2
                if in_torso or in_pocket:
                    state.near_body_frames += 1
                else:
                    state.near_body_frames = max(0, state.near_body_frames - 1)
            else:
                state.near_body_frames = max(0, state.near_body_frames - 1)

            # Determine disappearance near body (object id missing but was near body recently)
            tracked_labels = [o.get("label") for o in objects.values()]
            if state.last_object_label and state.near_body_frames > 0 and state.last_object_label not in tracked_labels:
                state.object_disappeared_near_body = True

            if (
                state.near_object_frames >= config.thresholds.smoothing_window
                and state.near_body_frames >= config.thresholds.frames_near_body
            ) or state.object_disappeared_near_body:
                event = SuspiciousEvent(
                    frame_idx=frame_idx,
                    person_id=pid,
                    object_label=state.last_object_label or "unknown",
                    reason="Object removed from shelf and moved close to torso/hip region",
                )
                new_events.append(event)
                state.near_object_frames = 0
                state.near_body_frames = 0
                state.object_disappeared_near_body = False
        self.events.extend(new_events)
        return new_events

    def draw_overlays(self, frame, persons: Dict[int, Dict], objects: Dict[int, Dict]) -> None:
        """Draw helper overlays including shelf zone and torso/pocket zones."""
        cv2.rectangle(frame, self.shelf_top_left, self.shelf_bottom_right, (255, 0, 255), 2)
        for pid, pdata in persons.items():
            torso_zone = self._torso_zone(pdata["bbox"])
            pocket_zone = self._pocket_zone(pdata["bbox"])
            cv2.rectangle(frame, (torso_zone[0], torso_zone[1]), (torso_zone[2], torso_zone[3]), (0, 165, 255), 1)
            cv2.rectangle(frame, (pocket_zone[0], pocket_zone[1]), (pocket_zone[2], pocket_zone[3]), (0, 0, 255), 1)
            cv2.putText(
                frame,
                f"PID {pid}",
                (pdata["bbox"][0], pdata["bbox"][1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
        for oid, obj in objects.items():
            x1, y1, x2, y2 = obj["bbox"]
            cv2.putText(frame, f"OID {oid}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
