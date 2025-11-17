"""
Video input utilities for the shoplifting detection demo.

Functions here open different sources (webcam, RTSP, file, YouTube) and return
an initialized cv2.VideoCapture object. The YouTube path will download the
video before returning the capture handle.
"""
import os
import subprocess
from typing import Tuple

import cv2

from config import Config


def open_video_capture(config: Config) -> Tuple[cv2.VideoCapture, str]:
    """Open the video source defined in the configuration.

    Returns
    -------
    capture: cv2.VideoCapture
        OpenCV capture instance for the chosen source.
    source_description: str
        Description of the input for logging.
    """
    mode = config.input.input_mode.lower()
    if mode == "webcam":
        cap = cv2.VideoCapture(0)
        return cap, "webcam:0"
    if mode == "rtsp":
        cap = cv2.VideoCapture(config.input.rtsp_url)
        return cap, f"rtsp:{config.input.rtsp_url}"
    if mode == "file":
        cap = cv2.VideoCapture(config.input.file_path)
        return cap, f"file:{config.input.file_path}"
    if mode == "youtube":
        path = download_youtube_video(
            url=config.input.youtube_url,
            output_path=config.input.youtube_download_path,
        )
        cap = cv2.VideoCapture(path)
        return cap, f"youtube:{config.input.youtube_url} -> {path}"
    raise ValueError(f"Unsupported input mode: {mode}")


def download_youtube_video(url: str, output_path: str) -> str:
    """Download a YouTube video using yt-dlp.

    The video is stored locally so it can be processed like any other MP4 file.
    If the file already exists it will be reused to avoid repeated downloads.
    """
    if os.path.exists(output_path):
        return output_path

    # Use yt-dlp via subprocess to avoid import-time overhead when not needed.
    command = [
        "yt-dlp",
        "-f",
        "best[ext=mp4]/best",
        "-o",
        output_path,
        url,
    ]
    subprocess.run(command, check=True)
    return output_path
