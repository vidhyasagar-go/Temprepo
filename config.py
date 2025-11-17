"""
Configuration settings for the shoplifting detection pipeline.
Update values here to change input source, model paths, thresholds, and
geometric regions like the shelf zone.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class InputConfig:
    """Configuration for video input selection."""
    input_mode: str = "youtube"  # Options: "webcam" | "rtsp" | "file" | "youtube"
    rtsp_url: str = "rtsp://username:password@camera-address/stream"
    file_path: str = "sample_store.mp4"
    youtube_url: str = "https://www.youtube.com/watch?v=HhZ6M4SBfZk"
    youtube_download_path: str = "downloaded_youtube.mp4"


@dataclass
class ModelConfig:
    """Paths and thresholds for detection models."""
    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence: float = 0.35
    yolo_iou: float = 0.45
    pose_detection_confidence: float = 0.5
    pose_tracking_confidence: float = 0.5


@dataclass
class GeometryConfig:
    """Geometric regions and distance thresholds for behavioral rules."""
    # Shelf zone expressed as normalized (0-1) coordinates of top-left and bottom-right
    shelf_zone: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.1, 0.3), (0.9, 0.6))
    # Body zone multipliers relative to hip and shoulder landmarks
    torso_zone_padding: float = 0.2
    pocket_zone_height: float = 0.25
    max_hand_object_distance: float = 75.0  # pixels


@dataclass
class LogicConfig:
    """Thresholds for smoothing and event triggering."""
    min_frames_touch: int = 3
    min_frames_body_hold: int = 10
    disappearance_frames: int = 8
    log_path: str = "events.log"


@dataclass
class Config:
    input: InputConfig = field(default_factory=InputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    logic: LogicConfig = field(default_factory=LogicConfig)


CONFIG = Config()
