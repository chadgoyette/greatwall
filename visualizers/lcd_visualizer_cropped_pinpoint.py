#!/usr/bin/env python3
import time
import math
import numpy as np
import pyaudio
import pygame
from pygame import gfxdraw

# ——— Configuration ———
# Update these values based on your setup
config = {
    # Audio / FFT
    "SOURCE_NAME": "null_monitor",  # Audio source name (check your system for the correct monitor source)
    "RATE": 44100,                  # Audio sample rate
    "CHUNK": 2048,                  # Audio chunk size
    "BANDS": 64,                    # Number of frequency bands
    "LOW_FREQ": 50,                 # Low frequency cutoff
    "HIGH_FREQ": 6000,              # High frequency cutoff

    # Display grid size (matches your LED matrix)
    "ROWS": 32,                     # Number of rows in the matrix
    "COLS": 64,                     # Number of columns in the matrix

    # Physical screen measurements (measure your LCD's active display area with a ruler)
    # REPLACE THESE WITH YOUR ACTUAL MEASUREMENTS IN MILLIMETERS
    # Can be floats for fine-tuning (e.g., 154.5, 155.5)
    "SCREEN_WIDTH_MM": 217,       # Width of the screen's active area in mm
    "SCREEN_HEIGHT_MM": 137,      # Height of the screen's active area in mm

    # Alignment offsets (in mm) to fine-tune dot positioning
    # Adjust these in small increments (e.g., 0.5, -0.5) if dots are misaligned
    "OFFSET_X_MM": 0.0,             # Horizontal offset in mm
    "OFFSET_Y_MM": 0.0,             # Vertical offset in mm

    # LED pitch (spacing between centers of LEDs)
    "LED_PITCH_MM": 4,              # Desired pitch in mm (5mm for your case)

    # Timing
    "FPS": 25.0,                    # Frames per second
    "FADE_TIME": 0.5,               # Fade time in seconds
    "FADE_FLOOR": 0.0001,           # Fade floor value

    # Dynamics
    "PEAK_DECAY": 0.9,              # Peak decay factor
    "MAX_DECAY": 0.95,              # Max decay factor

    # Mirror mode: "inverted" or "identical"
    "MIRROR_MODE": "inverted",      # Mirror mode for the visualization

    # Color: base RGB values (0–255)
    "COLOR": (255, 255, 255),       # Base color for the dots

    # Additional for dot scaling (to mimic LED expand/shrink)
    "MIN_V_FOR_DRAW": 0.01,         # Minimum brightness to draw a dot (avoids tiny faint dots)
}

# ——— Derived constants ———
# These are calculated from the config; no need to change
FADE_FACTOR = math.exp(
    math.log(config["FADE_FLOOR"]) / (config["FADE_TIME"] * config["FPS"])
)
base_r, base_g, base_b = config["COLOR"]

# ——— Pygame setup ———
# Initialize Pygame and get screen info
pygame.init()
info = pygame.display.Info()
screen_w, screen_h = info.current_w, info.current_h
screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

# ——— Calculate cell spacing and visible grid based on physical pitch ———
# These ensure the on-screen dots match the 5mm pitch; cropping if needed
pixels_per_mm_x = screen_w / config["SCREEN_WIDTH_MM"]
pixels_per_mm_y = screen_h / config["SCREEN_HEIGHT_MM"]
cell_spacing_x = pixels_per_mm_x * config["LED_PITCH_MM"]  # Keep as float for precision
cell_spacing_y = pixels_per_mm_y * config["LED_PITCH_MM"]  # Keep as float for precision

# Visible grid size (use floor division for integer counts)
num_cols_visible = min(config["COLS"], int(screen_w / cell_spacing_x))
num_rows_visible = min(config["ROWS"], int(screen_h / cell_spacing_y))

# Starting matrix indices for centering the crop
start_col = (config["COLS"] - num_cols_visible) // 2
start_row = (config["ROWS"] - num_rows_visible) // 2

# Position of the first dot center for centering on screen
first_center_x = (screen_w - (num_cols_visible - 1) * cell_spacing_x) / 2 + pixels_per_mm_x * config["OFFSET_X_MM"]
first_center_y = (screen_h - (num_rows_visible - 1) * cell_spacing_y) / 2 + pixels_per_mm_y * config["OFFSET_Y_MM"]

# Max radius in pixels, corresponding to 2mm diameter (1mm radius)
max_radius = int(min(pixels_per_mm_x, pixels_per_mm_y) * 1 + 0.5)  # 1mm radius for 2mm diameter
if max_radius < 1:
    max_radius = 1

# ——— Audio Input Setup ———
# Set up PyAudio for audio capture
pa = pyaudio.PyAudio()
def find_input_device(name):
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if name in info['name'] and info['maxInputChannels'] > 0:
            return i
    return None

dev_index = find_input_device(config["SOURCE_NAME"])
if dev_index is None:
    print("Error: monitor source not found.")
    exit(1)

stream = pa.open(
    format=pyaudio.paInt16,
    channels=2,
    rate=config["RATE"],
    input=True,
    input_device_index=dev_index,
    frames_per_buffer=config["CHUNK"]
)

# ——— Buffers ———
# Buffers for frame and peak data
frame_buf   = np.zeros((config["ROWS"], config["COLS"]), dtype=np.float32)
peak_buf    = np.zeros(config["BANDS"],          dtype=np.float32)
running_max = 1e-6

# ——— Main Loop ———
try:
    print("Visualizer running. Press Ctrl+C or ESC to exit.")
    running = True
    while running:
        t0 = time.time()
        for evt in pygame.event.get():
            if evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE:
                running = False

        # 1. Read audio & mono
        raw = stream.read(config["CHUNK"], exception_on_overflow=False)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        mono = (samples[0::2] + samples[1::2]) * 0.5

        # 2. FFT & band magnitudes
        windowed = mono * np.hanning(len(mono))
        fft_vals = np.abs(np.fft.rfft(windowed))
        freqs    = np.fft.rfftfreq(config["CHUNK"], 1.0 / config["RATE"])
        bins     = np.logspace(
            np.log10(config["LOW_FREQ"]),
            np.log10(config["HIGH_FREQ"]),
            config["BANDS"] + 1
        )
        mags = np.array([
            fft_vals[(freqs>=bins[i]) & (freqs<bins[i+1])].mean()
            if ((freqs>=bins[i]) & (freqs<bins[i+1])).any() else 0.0
            for i in range(config["BANDS"])
        ], dtype=np.float32)

        # 3. Normalize
        peak = mags.max()
        running_max = max(running_max * config["MAX_DECAY"], peak)
        norm = np.clip(mags / running_max, 0.0, 1.0)

        # 4. Fade
        frame_buf *= FADE_FACTOR

        # 5. Draw bars into buffer
        x_bins = np.linspace(0, config["COLS"] - 1, config["BANDS"])
        interp = np.interp(np.arange(config["COLS"]), x_bins, norm)
        half   = config["ROWS"] // 2
        for x in range(config["COLS"]):
            h = int(interp[x] * half)
            top    = half - 1 - h
            bottom = (half + h) if config["MIRROR_MODE"]=="inverted" else (half + h)
            frame_buf[top:half, x]      = 1.0
            frame_buf[half:bottom+1, x] = 1.0

        # 6. Peak caps
        peak_buf = np.maximum(peak_buf, interp)
        peak_buf *= config["PEAK_DECAY"]
        for x in range(config["COLS"]):
            cap_h = int(peak_buf[int(x * config["BANDS"] / config["COLS"])] * half)
            frame_buf[half - 1 - cap_h, x] = 1.0
            frame_buf[half + cap_h,     x] = 1.0

        # 7. Render (only the visible cropped portion, centered)
        # Dots scale in size based on brightness (expand at full bright, shrink on fade)
        screen.fill((0,0,0))
        fb_uint = (frame_buf * 255).astype(np.uint8)
        for vis_y in range(num_rows_visible):
            matrix_y = start_row + vis_y
            for vis_x in range(num_cols_visible):
                matrix_x = start_col + vis_x
                v = fb_uint[matrix_y, matrix_x] / 255.0
                if v > config["MIN_V_FOR_DRAW"]:
                    dot_radius = max(1, int(max_radius * v))
                    color = (int(base_r * v),
                             int(base_g * v),
                             int(base_b * v))
                    px = int(first_center_x + vis_x * cell_spacing_x + 0.5)
                    py = int(first_center_y + vis_y * cell_spacing_y + 0.5)
                    # anti-aliased filled circle
                    gfxdraw.filled_circle(screen, px, py, dot_radius, color)
                    gfxdraw.aacircle(screen, px, py, dot_radius, color)

        pygame.display.flip()

        # 8. Frame rate
        dt = time.time() - t0
        clock.tick(config["FPS"])

except KeyboardInterrupt:
    pass

# ——— Cleanup ———
stream.stop_stream()
stream.close()
pa.terminate()
pygame.quit()