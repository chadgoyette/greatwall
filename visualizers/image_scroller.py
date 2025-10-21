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
DISPLAY_HEIGHT = 32
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


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


def scroll_image(matrix: RGBMatrix, image: Image.Image):
    canvas = matrix.CreateFrameCanvas()
    start_x = DISPLAY_WIDTH
    center_x = (DISPLAY_WIDTH - image.width) // 2
    end_x = -image.width

    # scroll in
    for x in range(start_x, center_x - 1, -SCROLL_SPEED_PX):
        canvas.Clear()
        canvas.SetImage(image, x, 0)
        canvas = matrix.SwapOnVSync(canvas)
        time.sleep(FRAME_DELAY_SEC)

    # hold centered
    hold_until = time.time() + HOLD_TIME_SEC
    while time.time() < hold_until:
        canvas.Clear()
        canvas.SetImage(image, center_x, 0)
        canvas = matrix.SwapOnVSync(canvas)
        time.sleep(FRAME_DELAY_SEC)

    # scroll out
    for x in range(center_x, end_x - 1, -SCROLL_SPEED_PX):
        canvas.Clear()
        canvas.SetImage(image, x, 0)
        canvas = matrix.SwapOnVSync(canvas)
        time.sleep(FRAME_DELAY_SEC)


def main():
    global SCROLL_SPEED_PX, HOLD_TIME_SEC, FRAME_DELAY_SEC, DISPLAY_WIDTH, DISPLAY_HEIGHT

    parser = argparse.ArgumentParser(description="Scroll PNG images across a 64x64 LED matrix.")
    parser.add_argument("--pattern", default=DEFAULT_IMAGE_GLOB,
                        help="Glob or directory of images (default: ~/images/*)")
    parser.add_argument("--brightness", type=int, default=50, help="Matrix brightness 0-100")
    parser.add_argument("--speed", type=int, default=SCROLL_SPEED_PX, help="Pixels per frame while scrolling")
    parser.add_argument("--hold", type=float, default=HOLD_TIME_SEC, help="Seconds to hold the centered image")
    parser.add_argument("--delay", type=float, default=FRAME_DELAY_SEC, help="Delay between frames in seconds")
    parser.add_argument("--rows", type=int, default=32, help="Rows per panel")
    parser.add_argument("--cols", type=int, default=64, help="Columns per panel")
    parser.add_argument("--chain", type=int, default=1, help="Number of chained panels (width)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel panels (height)")
    parser.add_argument("--scale", type=float, default=1.0, help="Image scale factor (1.0 fits, >1 zooms in)")
    args = parser.parse_args()

    SCROLL_SPEED_PX = max(1, args.speed)
    HOLD_TIME_SEC = max(0, args.hold)
    FRAME_DELAY_SEC = max(0.001, args.delay)
    DISPLAY_WIDTH = args.cols * args.chain
    DISPLAY_HEIGHT = args.rows * args.parallel

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
