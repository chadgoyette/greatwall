#!/usr/bin/env python3
"""
Video player for RGB LED matrices driven by rpi-rgb-led-matrix.

Decodes MP4 (and similar) files using ffmpeg, resizes them to the configured
panel geometry, and streams frames to the matrix. Videos are played in order
and looped until interrupted.
"""
import argparse
import glob
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image
from rgbmatrix import RGBMatrix, RGBMatrixOptions


DEFAULT_VIDEO_PATTERN = os.path.expanduser("~/videos/*")
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")
DEFAULT_CHAIN_LAYOUT = "vertical"


def ensure_ffmpeg_available():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH. Install it with `sudo apt install ffmpeg`.")


def list_videos(pattern_or_dir: str) -> list[str]:
    path = Path(os.path.expanduser(pattern_or_dir))
    if path.is_dir():
        videos = sorted(
            str(p) for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        )
    else:
        videos = sorted(glob.glob(str(path)))

    if not videos:
        raise FileNotFoundError(f"No video files found for pattern {pattern_or_dir}")
    return videos


def get_video_fps(video_path: str) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None
    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None

    if not output:
        return None

    if "/" in output:
        num, denom = output.split("/")
        try:
            fps = float(num) / float(denom)
        except (ValueError, ZeroDivisionError):
            fps = None
    else:
        try:
            fps = float(output)
        except ValueError:
            fps = None

    if fps is not None and math.isfinite(fps) and fps > 0:
        return fps
    return None


def build_matrix(rows: int, cols: int, chain: int, parallel: int, brightness: int) -> RGBMatrix:
    options = RGBMatrixOptions()
    options.rows = rows
    options.cols = cols
    options.chain_length = chain
    options.parallel = parallel
    options.hardware_mapping = "regular"
    options.panel_type = "FM6126A"
    options.brightness = brightness
    options.disable_hardware_pulsing = True
    options.gpio_slowdown = 4
    return RGBMatrix(options=options)


def build_ffmpeg_command(video_path: str, width: int, height: int, mode: str, zoom: float) -> list[str]:
    zoom = max(0.1, zoom)
    scaled_w = max(1, int(round(width * zoom)))
    scaled_h = max(1, int(round(height * zoom)))

    if mode == "fit":
        if zoom <= 1.0:
            scale_filter = (
                f"scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=decrease"
            )
            pad_or_crop = f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        else:
            scale_filter = (
                f"scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=increase"
            )
            pad_or_crop = f"crop={width}:{height}"
        filter_chain = f"{scale_filter},{pad_or_crop}"

    elif mode == "fill":
        scale_filter = (
            f"scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=increase"
        )
        pad_or_crop = f"crop={width}:{height}"
        filter_chain = f"{scale_filter},{pad_or_crop}"

    else:  # stretch
        scale_filter = f"scale={scaled_w}:{scaled_h}"
        if zoom >= 1.0:
            pad_or_crop = f"crop={width}:{height}"
        else:
            pad_or_crop = f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        filter_chain = f"{scale_filter},{pad_or_crop}"

    return [
        "ffmpeg",
        "-loglevel", "error",
        "-i", video_path,
        "-vf", filter_chain,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]


def render_frame_to_canvas(
    canvas,
    frame: Image.Image,
    offset_x: int,
    offset_y: int,
    layout: str,
    panel_cols: int,
    panel_rows: int,
    chain_count: int,
    parallel_count: int,
) -> None:
    if layout == "horizontal":
        canvas.SetImage(frame, offset_x, offset_y)
        return

    if parallel_count != 1:
        raise RuntimeError("Vertical chain layout currently requires --parallel 1.")
    if offset_y != 0:
        raise RuntimeError("Vertical chain layout currently requires --offset-y 0.")

    hardware_width = panel_cols * chain_count
    hardware_height = panel_rows * parallel_count
    composed = Image.new("RGB", (hardware_width, hardware_height))

    for idx in reversed(range(chain_count)):
        y0 = idx * panel_rows
        y1 = min(y0 + panel_rows, frame.height)
        if y0 >= frame.height or y0 >= y1:
            continue
        slice_img = frame.crop((0, y0, frame.width, y1))
        panel_canvas = Image.new("RGB", (panel_cols, panel_rows))
        panel_canvas.paste(slice_img, (offset_x, 0))
        composed.paste(panel_canvas, (idx * panel_cols, 0))

    canvas.SetImage(composed, 0, 0)


def stream_video(
    matrix: RGBMatrix,
    video_path: str,
    virtual_width: int,
    virtual_height: int,
    mode: str,
    fps_override: Optional[float],
    min_frame_delay: float,
    offset_x: int,
    offset_y: int,
    zoom: float,
    layout: str,
    panel_cols: int,
    panel_rows: int,
    chain_count: int,
    parallel_count: int,
) -> None:
    cmd = build_ffmpeg_command(video_path, virtual_width, virtual_height, mode, zoom)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("Failed to open ffmpeg stdout.")

    fps = fps_override or get_video_fps(video_path) or 30.0
    frame_duration = max(1.0 / fps, min_frame_delay)
    frame_size = virtual_width * virtual_height * 3

    canvas = matrix.CreateFrameCanvas()
    frame_count = 0
    playback_start = time.monotonic()

    try:
        while True:
            frame_bytes = process.stdout.read(frame_size)
            if len(frame_bytes) < frame_size:
                break

            frame = Image.frombuffer("RGB", (virtual_width, virtual_height), frame_bytes, "raw", "RGB", 0, 1)
            canvas.Clear()
            render_frame_to_canvas(
                canvas,
                frame,
                offset_x,
                offset_y,
                layout,
                panel_cols,
                panel_rows,
                chain_count,
                parallel_count,
            )
            canvas = matrix.SwapOnVSync(canvas)
            frame_count += 1

            target_time = frame_count * frame_duration
            elapsed = time.monotonic() - playback_start
            delay = target_time - elapsed
            if delay > 0:
                time.sleep(delay)
    finally:
        process.stdout.close()
        process.wait()


def main():
    parser = argparse.ArgumentParser(description="Play videos on a RGB matrix using ffmpeg.")
    parser.add_argument("--pattern", default=DEFAULT_VIDEO_PATTERN,
                        help="Glob or directory of videos to play (default: ~/videos/*)")
    parser.add_argument("--brightness", type=int, default=50, help="Matrix brightness (0-100)")
    parser.add_argument("--rows", type=int, default=32, help="Rows per panel")
    parser.add_argument("--cols", type=int, default=64, help="Columns per panel")
    parser.add_argument("--chain", type=int, default=2, help="Number of chained panels")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel panels")
    parser.add_argument("--chain-layout", choices=("horizontal", "vertical"), default=DEFAULT_CHAIN_LAYOUT,
                        help="Interpret chained panels as a horizontal strip (library default) or as a vertical stack.")
    parser.add_argument("--mode", choices=("fit", "fill", "stretch"), default="fit",
                        help="Scale strategy: fit (letterbox), fill (crop), or stretch")
    parser.add_argument("--fps", type=float, default=None, help="Override video FPS")
    parser.add_argument("--min-delay", type=float, default=0.005,
                        help="Minimum frame delay to avoid busy looping (seconds)")
    parser.add_argument("--offset-x", type=int, default=0, help="Horizontal offset when drawing the frame")
    parser.add_argument("--offset-y", type=int, default=0, help="Vertical offset when drawing the frame")
    parser.add_argument("--zoom", type=float, default=1.0, help="Zoom factor (>1 enlarges, <1 shrinks before padding/cropping)")
    parser.add_argument("--no-loop", action="store_false", dest="loop",
                        help="Play each video once and exit (default: loop forever)")
    parser.set_defaults(loop=True)

    args = parser.parse_args()

    ensure_ffmpeg_available()

    videos = list_videos(args.pattern)
    print(f"Found {len(videos)} video(s): {[Path(v).name for v in videos]}")

    chain_layout = args.chain_layout
    panel_rows = args.rows
    panel_cols = args.cols
    chain_count = max(1, args.chain)
    parallel_count = max(1, args.parallel)

    if chain_layout == "vertical":
        if parallel_count != 1:
            parser.error("Vertical chain layout currently supports --parallel 1.")
        if args.offset_y != 0:
            parser.error("Use --offset-y 0 when --chain-layout vertical (bottom panel is remapped through chained columns).")
        virtual_width = panel_cols
        virtual_height = panel_rows * chain_count
    else:
        virtual_width = panel_cols * chain_count
        virtual_height = panel_rows * parallel_count

    matrix = build_matrix(panel_rows, panel_cols, chain_count, parallel_count, args.brightness)

    try:
        while True:
            for video in videos:
                print(f"Playing {video}")
                stream_video(
                    matrix,
                    video,
                    virtual_width,
                    virtual_height,
                    args.mode,
                    args.fps,
                    args.min_delay,
                    args.offset_x,
                    args.offset_y,
                    args.zoom,
                    chain_layout,
                    panel_cols,
                    panel_rows,
                    chain_count,
                    parallel_count,
                )
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        matrix.Clear()


if __name__ == "__main__":
    main()
