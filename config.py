"""
Configuration module defining model paths, thresholds, and input sources.
"""
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class InputConfig:
    """Configuration for the chosen input source."""
    mode: str = "youtube"  # Options: webcam | rtsp | file | youtube
    rtsp_url: str = "rtsp://username:password@ip-address:554/stream"
    video_path: str = "sample.mp4"  # Used when mode == "file"
    youtube_url: str = (
        "https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID"  # Replace with a store-like clip
    )


@dataclass
class ModelConfig:
    """Configuration for models used in detection."""
    yolo_model: str = "yolov8n.pt"
    yolo_conf_threshold: float = 0.4
    yolo_iou_threshold: float = 0.45
    pose_confidence: float = 0.5


@dataclass
class ZoneConfig:
    """Geometric definitions for shelf and body-related zones."""
    shelf_top_left: Tuple[int, int] = (50, 150)
    shelf_bottom_right: Tuple[int, int] = (600, 400)
    # Relative factors for defining torso/hip zones around the person box
    torso_zone_factor: float = 0.5  # fraction of person box height centered at hips
    pocket_zone_factor: float = 0.25  # relative area around hips for pockets


@dataclass
class LogicThresholds:
    """Threshold values for determining suspicious behavior."""
    hand_object_distance: float = 80.0  # pixels
    frames_near_body: int = 12
    disappearance_grace: int = 6
    smoothing_window: int = 4


@dataclass
class AppConfig:
    """Aggregate configuration for the application."""
    input: InputConfig = field(default_factory=InputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    thresholds: LogicThresholds = field(default_factory=LogicThresholds)
    draw_landmarks: bool = True
    output_log: str = "events.log"
    save_downloads_to: str = "downloads"
    max_frames: int = -1  # -1 processes the full video


config = AppConfig()
