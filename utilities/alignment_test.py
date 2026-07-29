#!/usr/bin/env python3
"""
Alignment test for dual FM6126A 64x32 LED panels (vertical chain).
Displays five 20x20 white squares: one in each corner and one in the center.
Run with: sudo python3 alignment_test.py
"""
import time

from PIL import Image, ImageDraw
from rgbmatrix import RGBMatrix, RGBMatrixOptions

PANEL_ROWS = 32
PANEL_COLS = 64
CHAIN_COUNT = 2
DISPLAY_WIDTH = 64
DISPLAY_HEIGHT = 64

SQUARE_SIZE = 20


def build_matrix():
    options = RGBMatrixOptions()
    options.rows = PANEL_ROWS
    options.cols = PANEL_COLS
    options.chain_length = CHAIN_COUNT
    options.parallel = 1
    options.hardware_mapping = "regular"
    options.panel_type = "FM6126A"
    options.brightness = 100
    options.disable_hardware_pulsing = True
    options.gpio_slowdown = 4
    options.multiplexing = 0
    return RGBMatrix(options=options)


def virtual_to_hardware(virtual_img):
    """Convert 64x64 virtual image to 128x32 hardware image for vertical chain."""
    hardware_width = PANEL_COLS * CHAIN_COUNT
    hardware_height = PANEL_ROWS
    frame = Image.new("RGB", (hardware_width, hardware_height))
    for idx in reversed(range(CHAIN_COUNT)):
        y0 = idx * PANEL_ROWS
        y1 = y0 + PANEL_ROWS
        panel_slice = virtual_img.crop((0, y0, PANEL_COLS, y1))
        frame.paste(panel_slice, ((CHAIN_COUNT - 1 - idx) * PANEL_COLS, 0))
    return frame


def main():
    matrix = build_matrix()
    canvas = matrix.CreateFrameCanvas()

    img = Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    s = SQUARE_SIZE
    # Top-left
    draw.rectangle([0, 0, s - 1, s - 1], fill=(255, 255, 255))
    # Top-right
    draw.rectangle([DISPLAY_WIDTH - s, 0, DISPLAY_WIDTH - 1, s - 1], fill=(255, 255, 255))
    # Bottom-left
    draw.rectangle([0, DISPLAY_HEIGHT - s, s - 1, DISPLAY_HEIGHT - 1], fill=(255, 255, 255))
    # Bottom-right
    draw.rectangle([DISPLAY_WIDTH - s, DISPLAY_HEIGHT - s, DISPLAY_WIDTH - 1, DISPLAY_HEIGHT - 1], fill=(255, 255, 255))
    # Center
    cx = (DISPLAY_WIDTH - s) // 2
    cy = (DISPLAY_HEIGHT - s) // 2
    draw.rectangle([cx, cy, cx + s - 1, cy + s - 1], fill=(255, 255, 255))

    hardware = virtual_to_hardware(img)
    canvas.SetImage(hardware, 0, 0)
    matrix.SwapOnVSync(canvas)

    print("Alignment Test — 5x 20x20 white squares (corners + center)")
    print("Ctrl+C to exit")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        matrix.Clear()
        print("\nDone.")


if __name__ == "__main__":
    main()
