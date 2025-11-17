"""
Utility helpers for opening different video sources and downloading YouTube clips.
"""
import os
import pathlib
import cv2

try:
    from yt_dlp import YoutubeDL
except ImportError:  # pragma: no cover - optional dependency
    YoutubeDL = None

from config import config


def open_capture():
    """Return an OpenCV VideoCapture based on configured input mode."""
    mode = config.input.mode
    if mode == "webcam":
        return cv2.VideoCapture(0)
    if mode == "rtsp":
        return cv2.VideoCapture(config.input.rtsp_url)
    if mode == "file":
        return cv2.VideoCapture(config.input.video_path)
    if mode == "youtube":
        path = download_youtube_video(config.input.youtube_url)
        return cv2.VideoCapture(path)
    raise ValueError(f"Unsupported input mode: {mode}")


def download_youtube_video(url: str) -> str:
    """Download a YouTube video to a local file using yt-dlp.

    The downloaded file is stored in the configured downloads directory and returned for processing.
    """
    download_dir = pathlib.Path(config.save_downloads_to)
    download_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(download_dir / "%(title)s.%(ext)s")

    if YoutubeDL is None:
        raise RuntimeError("yt-dlp is required for YouTube downloads. Install via `pip install yt-dlp`.")

    ydl_opts = {
        "format": "best[ext=mp4]/bestvideo+bestaudio",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = ydl.prepare_filename(info)
    return downloaded_path
