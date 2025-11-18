#!/usr/bin/env python3
"""
Image scrolling demo for 64x64 LED matrix built from two 64x32 panels.

Loads PNG files from an images/ directory, scrolls each one in from the right,
holds it centered for a few seconds, scrolls out to the left, and repeats
forever until interrupted.
"""
import argparse
import glob
import time
from pathlib import Path
import os

from PIL import Image
from rgbmatrix import RGBMatrix, RGBMatrixOptions


# === Default configuration ===
DEFAULT_IMAGE_GLOB = os.path.expanduser("~/images/*")
SCROLL_SPEED_PX = 1          # pixels per frame
FRAME_DELAY_SEC = 0.01       # time between frames in seconds
HOLD_TIME_SEC = 5.0          # dwell time while centered
DISPLAY_WIDTH = 64
DISPLAY_HEIGHT = 64
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
CHAIN_LAYOUT = "vertical"     # how chained panels are arranged virtually
PANEL_ROWS = 32
PANEL_COLS = 64
CHAIN_COUNT = 2
PARALLEL_COUNT = 1


def build_matrix(brightness: int, rows: int, cols: int, chain: int, parallel: int) -> RGBMatrix:
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


def load_images(pattern_or_dir: str, scale: float):
    path = Path(os.path.expanduser(pattern_or_dir))
    if path.is_dir():
        files = sorted(
            str(p) for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        files = sorted(glob.glob(str(path)))

    images = []
    for filename in files:
        try:
            img = Image.open(filename).convert("RGB")
        except Exception as exc:
            print(f"Skipping {filename}: {exc}")
            continue

        # Resize while keeping aspect ratio, then paste into canvas
        img = resize_to_canvas(img, DISPLAY_WIDTH, DISPLAY_HEIGHT, scale)
        images.append((Path(filename).name, img))

    if not images:
        raise FileNotFoundError(f"No images matched pattern {pattern_or_dir}")
    return images


def resize_to_canvas(img: Image.Image, width: int, height: int, scale: float) -> Image.Image:
    canvas = Image.new("RGB", (width, height))
    img_copy = img.copy()

    scale = max(0.1, scale)
    target_w = max(1, int(width * scale))
    target_h = max(1, int(height * scale))
    ratio = min(target_w / img_copy.width, target_h / img_copy.height)
    new_w = max(1, int(img_copy.width * ratio))
    new_h = max(1, int(img_copy.height * ratio))
    img_copy = img_copy.resize((new_w, new_h), Image.LANCZOS)

    offset_x = (width - new_w) // 2
    offset_y = (height - new_h) // 2
    canvas.paste(img_copy, (offset_x, offset_y))
    return canvas


def render_virtual_image(canvas, image: Image.Image, x_offset: int):
    if CHAIN_LAYOUT == "horizontal":
        canvas.SetImage(image, x_offset, 0)
        return

    if PARALLEL_COUNT != 1:
        raise ValueError("Vertical chain layout currently requires --parallel 1.")

    hardware_width = PANEL_COLS * CHAIN_COUNT
    hardware_height = PANEL_ROWS * PARALLEL_COUNT
    frame = Image.new("RGB", (hardware_width, hardware_height))

    for idx in reversed(range(CHAIN_COUNT)):
        y0 = idx * PANEL_ROWS
        y1 = min(y0 + PANEL_ROWS, image.height)
        if y0 >= image.height or y0 >= y1:
            continue
        panel_slice = image.crop((0, y0, image.width, y1))
        panel_canvas = Image.new("RGB", (PANEL_COLS, PANEL_ROWS))
        panel_canvas.paste(panel_slice, (x_offset, 0))
        frame.paste(panel_canvas, (idx * PANEL_COLS, 0))

    canvas.SetImage(frame, 0, 0)


def scroll_image(matrix: RGBMatrix, image: Image.Image):
    canvas = matrix.CreateFrameCanvas()
    start_x = DISPLAY_WIDTH
    center_x = (DISPLAY_WIDTH - image.width) // 2
    end_x = -image.width

    # scroll in
    for x in range(start_x, center_x - 1, -SCROLL_SPEED_PX):
        canvas.Clear()
        render_virtual_image(canvas, image, x)
        canvas = matrix.SwapOnVSync(canvas)
        time.sleep(FRAME_DELAY_SEC)

    # hold centered
    hold_until = time.time() + HOLD_TIME_SEC
    while time.time() < hold_until:
        canvas.Clear()
        render_virtual_image(canvas, image, center_x)
        canvas = matrix.SwapOnVSync(canvas)
        time.sleep(FRAME_DELAY_SEC)

    # scroll out
    for x in range(center_x, end_x - 1, -SCROLL_SPEED_PX):
        canvas.Clear()
        render_virtual_image(canvas, image, x)
        canvas = matrix.SwapOnVSync(canvas)
        time.sleep(FRAME_DELAY_SEC)


def main():
    global SCROLL_SPEED_PX, HOLD_TIME_SEC, FRAME_DELAY_SEC, DISPLAY_WIDTH, DISPLAY_HEIGHT
    global CHAIN_LAYOUT, PANEL_ROWS, PANEL_COLS, CHAIN_COUNT, PARALLEL_COUNT

    parser = argparse.ArgumentParser(description="Scroll PNG images across a 64x64 LED matrix.")
    parser.add_argument("--pattern", default=DEFAULT_IMAGE_GLOB,
                        help="Glob or directory of images (default: ~/images/*)")
    parser.add_argument("--brightness", type=int, default=50, help="Matrix brightness 0-100")
    parser.add_argument("--speed", type=int, default=SCROLL_SPEED_PX, help="Pixels per frame while scrolling")
    parser.add_argument("--hold", type=float, default=HOLD_TIME_SEC, help="Seconds to hold the centered image")
    parser.add_argument("--delay", type=float, default=FRAME_DELAY_SEC, help="Delay between frames in seconds")
    parser.add_argument("--rows", type=int, default=32, help="Rows per panel")
    parser.add_argument("--cols", type=int, default=64, help="Columns per panel")
    parser.add_argument("--chain", type=int, default=2, help="Number of chained panels (width)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel panels (height)")
    parser.add_argument("--chain-layout", choices=("horizontal", "vertical"), default="vertical",
                        help="Interpret chained panels horizontally (default from library) or as a vertical stack.")
    parser.add_argument("--scale", type=float, default=1.0, help="Image scale factor (1.0 fits, >1 zooms in)")
    args = parser.parse_args()

    SCROLL_SPEED_PX = max(1, args.speed)
    HOLD_TIME_SEC = max(0, args.hold)
    FRAME_DELAY_SEC = max(0.001, args.delay)
    CHAIN_LAYOUT = args.chain_layout
    PANEL_ROWS = args.rows
    PANEL_COLS = args.cols
    CHAIN_COUNT = max(1, args.chain)
    PARALLEL_COUNT = max(1, args.parallel)

    if CHAIN_LAYOUT == "vertical":
        if PARALLEL_COUNT != 1:
            parser.error("Vertical chain layout currently supports --parallel 1. Adjust wiring or run with --chain-layout horizontal.")
        DISPLAY_WIDTH = PANEL_COLS
        DISPLAY_HEIGHT = PANEL_ROWS * CHAIN_COUNT
    else:
        DISPLAY_WIDTH = PANEL_COLS * CHAIN_COUNT
        DISPLAY_HEIGHT = PANEL_ROWS * PARALLEL_COUNT

    images = load_images(args.pattern, args.scale)
    print(f"Loaded {len(images)} image(s): {[name for name, _ in images]}")

    matrix = build_matrix(args.brightness, args.rows, args.cols, args.chain, args.parallel)

    try:
        while True:
            for name, img in images:
                print(f"Displaying {name}")
                scroll_image(matrix, img)
    except KeyboardInterrupt:
        pass
    finally:
        matrix.Clear()


if __name__ == "__main__":
    main()
