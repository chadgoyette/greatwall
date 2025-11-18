#!/usr/bin/env python3
"""
Animated GIF player for RGB LED matrices driven by rpi-rgb-led-matrix.

Loads GIF files, resizes them to the configured virtual canvas, and plays each
animation in sequence. After the last GIF finishes, playback loops from the top
until interrupted.
"""

import argparse
import glob
import os
import time
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageSequence
from rgbmatrix import RGBMatrix, RGBMatrixOptions


DEFAULT_GIF_PATTERN = os.path.expanduser("~/gifs/*")
SUPPORTED_GIF_EXTENSIONS = (".gif",)
DEFAULT_CHAIN_LAYOUT = "vertical"


def list_gifs(pattern_or_dir: str) -> List[Path]:
    """Expand a pattern or directory into a sorted list of GIF file paths."""
    path = Path(os.path.expanduser(pattern_or_dir))
    if path.is_dir():
        gifs = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_GIF_EXTENSIONS
        )
    else:
        gifs = sorted(
            Path(p) for p in glob.glob(str(path))
            if Path(p).suffix.lower() in SUPPORTED_GIF_EXTENSIONS
        )

    if not gifs:
        raise FileNotFoundError(f"No GIF files matched pattern {pattern_or_dir}")
    return gifs


def resize_frame_to_canvas(frame: Image.Image, width: int, height: int, scale: float) -> Image.Image:
    """Fit a frame onto a black canvas while preserving aspect ratio."""
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 255))

    scale = max(0.1, scale)
    target_w = max(1, int(width * scale))
    target_h = max(1, int(height * scale))
    ratio = min(target_w / frame.width, target_h / frame.height)
    new_w = max(1, int(frame.width * ratio))
    new_h = max(1, int(frame.height * ratio))

    resized = frame.resize((new_w, new_h), Image.LANCZOS)
    offset_x = (width - new_w) // 2
    offset_y = (height - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y), resized)
    return canvas.convert("RGB")


def render_virtual_frame(
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
    """Blit a virtual canvas to the physical matrix layout."""
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


def load_gif_frames(
    gif_path: Path,
    width: int,
    height: int,
    scale: float,
    min_delay: float,
) -> List[Tuple[Image.Image, float]]:
    """Decode a GIF into sized RGB frames and their display durations in seconds."""
    frames: List[Tuple[Image.Image, float]] = []
    with Image.open(gif_path) as gif:
        if getattr(gif, "is_animated", False) is False or getattr(gif, "n_frames", 1) < 1:
            raise ValueError(f"{gif_path} does not contain animation frames.")

        for frame in ImageSequence.Iterator(gif):
            frame_rgba = frame.convert("RGBA")
            sized = resize_frame_to_canvas(frame_rgba, width, height, scale)
            duration_ms = frame.info.get("duration", gif.info.get("duration", 100))
            duration = max(min_delay, max(duration_ms, 10) / 1000.0)
            frames.append((sized, duration))

    if not frames:
        raise ValueError(f"Failed to decode frames from {gif_path}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Play animated GIFs sequentially on an RGB matrix.")
    parser.add_argument("--pattern", default=DEFAULT_GIF_PATTERN,
                        help="Glob or directory of GIFs to play (default: ~/gifs/*)")
    parser.add_argument("--brightness", type=int, default=50, help="Matrix brightness (0-100)")
    parser.add_argument("--rows", type=int, default=32, help="Rows per panel")
    parser.add_argument("--cols", type=int, default=64, help="Columns per panel")
    parser.add_argument("--chain", type=int, default=2, help="Number of chained panels")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel panels")
    parser.add_argument("--chain-layout", choices=("horizontal", "vertical"), default=DEFAULT_CHAIN_LAYOUT,
                        help="Interpret chained panels as a horizontal strip (library default) or vertical stack.")
    parser.add_argument("--offset-x", type=int, default=0, help="Horizontal offset for the virtual frame")
    parser.add_argument("--offset-y", type=int, default=0, help="Vertical offset for the virtual frame")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor when fitting frames (1.0 fits, >1 zooms)")
    parser.add_argument("--min-delay", type=float, default=0.02, help="Minimum frame delay in seconds")
    parser.add_argument("--pause", type=float, default=0.5,
                        help="Pause between GIFs in seconds (set to 0 for continuous playback)")
    args = parser.parse_args()

    chain_layout = args.chain_layout
    panel_rows = args.rows
    panel_cols = args.cols
    chain_count = max(1, args.chain)
    parallel_count = max(1, args.parallel)

    if chain_layout == "vertical":
        if parallel_count != 1:
            parser.error("Vertical chain layout currently supports --parallel 1.")
        if args.offset_y != 0:
            parser.error("Use --offset-y 0 when --chain-layout vertical.")
        virtual_width = panel_cols
        virtual_height = panel_rows * chain_count
    else:
        virtual_width = panel_cols * chain_count
        virtual_height = panel_rows * parallel_count

    matrix_options = RGBMatrixOptions()
    matrix_options.rows = panel_rows
    matrix_options.cols = panel_cols
    matrix_options.chain_length = chain_count
    matrix_options.parallel = parallel_count
    matrix_options.hardware_mapping = "regular"
    matrix_options.panel_type = "FM6126A"
    matrix_options.gpio_slowdown = 4
    matrix_options.disable_hardware_pulsing = True
    matrix_options.brightness = args.brightness

    matrix = RGBMatrix(options=matrix_options)

    gif_paths = list_gifs(args.pattern)
    print(f"Found {len(gif_paths)} GIF(s): {[p.name for p in gif_paths]}")

    playlist = []
    for path in gif_paths:
        try:
            frames = load_gif_frames(path, virtual_width, virtual_height, args.scale, args.min_delay)
            playlist.append((path.name, frames))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Skipping {path}: {exc}")

    if not playlist:
        raise RuntimeError("No playable GIFs were loaded.")

    canvas = matrix.CreateFrameCanvas()
    try:
        while True:
            for name, frames in playlist:
                print(f"Playing {name} ({len(frames)} frame(s))")
                for frame_img, delay in frames:
                    frame_start = time.monotonic()
                    canvas.Clear()
                    render_virtual_frame(
                        canvas,
                        frame_img,
                        args.offset_x,
                        args.offset_y,
                        chain_layout,
                        panel_cols,
                        panel_rows,
                        chain_count,
                        parallel_count,
                    )
                    canvas = matrix.SwapOnVSync(canvas)
                    elapsed = time.monotonic() - frame_start
                    sleep_time = max(args.min_delay, delay - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                if args.pause > 0:
                    time.sleep(args.pause)
    except KeyboardInterrupt:
        pass
    finally:
        matrix.Clear()


if __name__ == "__main__":
    main()
