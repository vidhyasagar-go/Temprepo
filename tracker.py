"""
Simple nearest-neighbor tracker for persisting IDs of detections across frames.
"""
from typing import Dict, List, Tuple


class Track:
    """Represents a tracked entity with bbox and centroid."""

    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int]) -> None:
        self.id = track_id
        self.bbox = bbox
        self.missed = 0

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


class SimpleTracker:
    """Lightweight tracker based on centroid distance and IoU matching."""

    def __init__(self, max_missed: int = 10, iou_threshold: float = 0.3) -> None:
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def _iou(self, box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def update(self, detections: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[Tuple[int, int, int, int], int]]:
        """Assign detections to existing tracks and return mapping track_id->(bbox, detection_index)."""
        assignments: Dict[int, Tuple[Tuple[int, int, int, int], int]] = {}
        used_tracks = set()
        used_dets = set()

        track_ids = list(self.tracks.keys())
        for det_idx, det in enumerate(detections):
            best_id = None
            best_score = 0.0
            for tid in track_ids:
                if tid in used_tracks:
                    continue
                score = self._iou(det, self.tracks[tid].bbox)
                if score > best_score:
                    best_score = score
                    best_id = tid
            if best_id is not None and best_score >= self.iou_threshold:
                self.tracks[best_id].bbox = det
                self.tracks[best_id].missed = 0
                assignments[best_id] = (det, det_idx)
                used_tracks.add(best_id)
                used_dets.add(det_idx)

        for det_idx, det in enumerate(detections):
            if det_idx in used_dets:
                continue
            track = Track(self.next_id, det)
            self.tracks[self.next_id] = track
            assignments[self.next_id] = (det, det_idx)
            self.next_id += 1

        to_delete = []
        for tid, track in self.tracks.items():
            if tid not in assignments:
                track.missed += 1
            if track.missed > self.max_missed:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]
        return assignments
