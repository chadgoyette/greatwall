#!/usr/bin/env python3
import time
import math
import numpy as np
import pyaudio
import pygame
from pygame import gfxdraw

# ——— Configuration ———
config = {
    # Audio / FFT
    "SOURCE_NAME": "null_monitor",
    "RATE": 44100,
    "CHUNK": 2048,
    "BANDS": 64,
    "LOW_FREQ": 50,
    "HIGH_FREQ": 6000,

    # Display
    "ROWS": 32,
    "COLS": 64,

    # Timing
    "FPS": 25.0,
    "FADE_TIME": 0.5,
    "FADE_FLOOR": 0.0001,

    # Dynamics
    "PEAK_DECAY": 0.9,
    "MAX_DECAY": 0.95,

    # Mirror mode: "inverted" or "identical"
    "MIRROR_MODE": "inverted",

    # Color: base RGB values (0–255)
    "COLOR": (255, 255, 255),
}

# ——— Derived constants ———
FADE_FACTOR = math.exp(
    math.log(config["FADE_FLOOR"]) / (config["FADE_TIME"] * config["FPS"])
)
base_r, base_g, base_b = config["COLOR"]

# ——— Pygame setup ———
pygame.init()
info = pygame.display.Info()
screen_w, screen_h = info.current_w, info.current_h
screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

# calculate cell size to fit 64×32 grid
cell_w = screen_w // config["COLS"]
cell_h = screen_h // config["ROWS"]
cell_size = min(cell_w, cell_h)
# center the grid
offset_x = (screen_w - cell_size * config["COLS"]) // 2
offset_y = (screen_h - cell_size * config["ROWS"]) // 2
radius = cell_size // 2 - 1

# ——— Audio Input Setup ———
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

        # 7. Render
        screen.fill((0,0,0))
        fb_uint = (frame_buf * 255).astype(np.uint8)
        for y in range(config["ROWS"]):
            for x in range(config["COLS"]):
                v = fb_uint[y, x] / 255.0
                if v > 0:
                    color = (int(base_r * v),
                             int(base_g * v),
                             int(base_b * v))
                    px = offset_x + x * cell_size + cell_size//2
                    py = offset_y + y * cell_size + cell_size//2
                    # anti-aliased filled circle
                    gfxdraw.filled_circle(screen, px, py, radius, color)
                    gfxdraw.aacircle(screen, px, py, radius, color)

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
