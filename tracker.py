"""
Simple IoU-based tracker for associating detections across frames.

This tracker is intentionally lightweight to avoid heavy dependencies like
DeepSORT. It maintains persistent IDs for objects and persons using bounding box
IoU matching.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from object_detector import Detection
from pose_detector import PoseResult


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two bounding boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass
class TrackedObject:
    id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str


class SimpleTracker:
    """Maintain IDs for detections across frames using IoU matching."""

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks: Dict[int, Tuple[int, int, int, int]] = {}

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        assigned_ids: List[TrackedObject] = []
        unmatched_tracks = set(self.tracks.keys())

        for det in detections:
            best_iou = 0.0
            best_track = None
            for track_id, track_bbox in self.tracks.items():
                score = _iou(det.bbox, track_bbox)
                if score > best_iou:
                    best_iou = score
                    best_track = track_id
            if best_iou >= self.iou_threshold and best_track is not None:
                self.tracks[best_track] = det.bbox
                unmatched_tracks.discard(best_track)
                assigned_ids.append(
                    TrackedObject(id=best_track, bbox=det.bbox, confidence=det.confidence, label=det.label)
                )
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = det.bbox
                assigned_ids.append(
                    TrackedObject(id=new_id, bbox=det.bbox, confidence=det.confidence, label=det.label)
                )

        for track_id in unmatched_tracks:
            self.tracks.pop(track_id, None)

        return assigned_ids

    def assign_poses(self, poses: List[PoseResult]) -> List[PoseResult]:
        """Assign IDs to pose results using bounding box IoU against tracked boxes."""
        updated: List[PoseResult] = []
        for pose in poses:
            best_iou = 0.0
            best_id = None
            for track_id, track_bbox in self.tracks.items():
                score = _iou(pose.bounding_box, track_bbox)
                if score > best_iou:
                    best_iou = score
                    best_id = track_id
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = pose.bounding_box
            updated.append(
                PoseResult(person_id=best_id, landmarks=pose.landmarks, bounding_box=pose.bounding_box)
            )
        return updated
