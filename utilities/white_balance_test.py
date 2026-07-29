#!/usr/bin/env python3
"""
White balance test for dual FM6126A 64x32 LED panels (vertical chain).
Shows solid white at 100%, then cycles through T-E-S-T letters full screen.
Run with: sudo python3 white_balance_test.py
"""
import time

from PIL import Image, ImageDraw, ImageFont
from rgbmatrix import RGBMatrix, RGBMatrixOptions

PANEL_ROWS = 32
PANEL_COLS = 64
CHAIN_COUNT = 2
DISPLAY_WIDTH = 64
DISPLAY_HEIGHT = 64

HOLD_SECONDS = 5.0
LETTERS = list("TEST")

# White-balance adjustment. Tweak these 0-255 values to correct the color.
# e.g. lower BLUE if the panels look too cold/blue, lower RED if too warm.
#
COLOR = (225, 255, 255)  # (R, G, B)

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


def display_image(matrix, canvas, virtual_img):
    """Send a 64x64 virtual image to the display."""
    hardware = virtual_to_hardware(virtual_img)
    canvas.SetImage(hardware, 0, 0)
    return matrix.SwapOnVSync(canvas)


def make_solid_white():
    """Create a full white 64x64 image."""
    return Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), COLOR)


def make_letter(letter):
    """Create a 64x64 image with a single letter centered as large as possible."""
    img = Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Find the largest font size that fits the display
    font = None
    for size in range(120, 10, -1):
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except OSError:
            f = ImageFont.load_default()
        bbox = f.getbbox(letter)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= DISPLAY_WIDTH and h <= DISPLAY_HEIGHT:
            font = f
            break

    if font is None:
        font = ImageFont.load_default()

    bbox = font.getbbox(letter)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (DISPLAY_WIDTH - w) // 2 - bbox[0]
    y = (DISPLAY_HEIGHT - h) // 2 - bbox[1]
    draw.text((x, y), letter, fill=COLOR, font=font)
    return img


def main():
    matrix = build_matrix()
    canvas = matrix.CreateFrameCanvas()

    print("White Balance Test — Ctrl+C to exit")
    print(f"Hold: {HOLD_SECONDS}s per step")
    print(f"Sequence: SOLID WHITE -> T -> E -> S -> T -> repeat")
    print()

    try:
        while True:
            # Solid white
            print(">> SOLID WHITE")
            canvas = display_image(matrix, canvas, make_solid_white())
            time.sleep(HOLD_SECONDS)

            # Letters
            for letter in LETTERS:
                print(f">> {letter}")
                canvas = display_image(matrix, canvas, make_letter(letter))
                time.sleep(HOLD_SECONDS)

    except KeyboardInterrupt:
        pass
    finally:
        matrix.Clear()
        print("\nDone.")


if __name__ == "__main__":
    main()
