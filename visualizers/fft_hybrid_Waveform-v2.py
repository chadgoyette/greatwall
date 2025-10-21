#!/usr/bin/env python3
import time
import math
import numpy as np
import pyaudio
from rgbmatrix import RGBMatrix, RGBMatrixOptions

# ——— Configuration ———
config = {
    # Audio / FFT
    "SOURCE_NAME": "null_monitor",
    "RATE": 44100, #the audio sampling rate in Hz (44 100 Hz is CD quality)
    "CHUNK": 2048, #how many samples you grab per FFT frame (power of two; 2048 gives ~21 ms time resolution)
    "BANDS": 64, #how many log-spaced frequency bins you split your spectrum into (64 gives one column per band)
    "LOW_FREQ": 50,  #the bottom and top frequencies (in Hz) of your log-spaced bins—everything between these bounds is analyzed
    "HIGH_FREQ": 6000,

    # Display
    "ROWS": 32,
    "COLS": 64,

    # Timing
    "FPS": 25.0,  #Frames per second—how many times per second the visualizer updates
    "FADE_TIME": 0.5,  #How many seconds it takes for a drawn pixel to fade down to the “floor” level
    "FADE_FLOOR": 0.0001, #The relative brightness (0–1) a pixel reaches after FADE_TIME seconds of fading

    # Dynamics
    "PEAK_DECAY": 0.9, #A higher PEAK_DECAY (e.g. 0.97 → 0.99) makes the caps drop more slowly, leaving them visible longer.
    "MAX_DECAY": 0.95, #A higher MAX_DECAY (e.g. 0.98 → 0.995) means the normalization baseline decays more slowly, so if you get a really loud passage, the display will stay globally scaled to that level longer—your bars will look smaller in quieter parts.

    # Mirror mode: "inverted" or "identical"
    #"MIRROR_MODE": "identical",
    "MIRROR_MODE": "inverted",
    
    # Color: base RGB values (0–255)
    "COLOR": (255, 255, 255),  # white; change to e.g. (0,200,255) for teal
}

# ——— Derived constants ———
FADE_FACTOR = math.exp(
    math.log(config["FADE_FLOOR"]) / (config["FADE_TIME"] * config["FPS"])
)
base_r, base_g, base_b = config["COLOR"]

# ——— LED Matrix Setup ———
options = RGBMatrixOptions()
options.rows = config["ROWS"]
options.cols = config["COLS"]
options.chain_length = 1
options.hardware_mapping = 'regular'
options.brightness = 50
options.disable_hardware_pulsing = True
options.panel_type = "FM6126A"
options.gpio_slowdown = 4
options.multiplexing = 0

matrix = RGBMatrix(options=options)
canvas = matrix.CreateFrameCanvas()

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
    print("Visualizer running. Press Ctrl+C to exit.")
    while True:
        t0 = time.time()

        # 1. Read audio & convert to mono
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

        mags = np.zeros(config["BANDS"], dtype=np.float32)
        for i in range(config["BANDS"]):
            mask = (freqs >= bins[i]) & (freqs < bins[i+1])
            mags[i] = fft_vals[mask].mean() if mask.any() else 0.0

        # 3. Normalize magnitudes
        peak = mags.max()
        running_max = max(running_max * config["MAX_DECAY"], peak)
        norm = np.clip(mags / running_max, 0.0, 1.0)

        # 4. Fade previous frame
        frame_buf *= FADE_FACTOR

        # 5. Draw inverted, filled curve
        x_bins = np.linspace(0, config["COLS"] - 1, config["BANDS"])
        interp = np.interp(np.arange(config["COLS"]), x_bins, norm)
        half   = config["ROWS"] // 2

        for x in range(config["COLS"]):
            h = int(interp[x] * half)
            top    = half - 1 - h
            bottom = half + h if config["MIRROR_MODE"] == "inverted" else half + h
            frame_buf[top:half, x]      = 1.0
            frame_buf[half:bottom+1, x] = 1.0

        # 6. Peak caps
        peak_buf = np.maximum(peak_buf, interp)
        peak_buf *= config["PEAK_DECAY"]
        for x in range(config["COLS"]):
            cap_h = int(peak_buf[int(x * config["BANDS"] / config["COLS"])] * half)
            frame_buf[half - 1 - cap_h, x] = 1.0
            frame_buf[half + cap_h,     x] = 1.0

        # 7. Render with color scaling
        canvas.Clear()
        fb_uint = (frame_buf * 255).astype(np.uint8)
        for y in range(config["ROWS"]):
            for x in range(config["COLS"]):
                v = fb_uint[y, x] / 255.0  # normalized intensity
                if v > 0:
                    r = int(base_r * v)
                    g = int(base_g * v)
                    b = int(base_b * v)
                    canvas.SetPixel(x, y, r, g, b)
        canvas = matrix.SwapOnVSync(canvas)

        # 8. Frame rate control
        dt = time.time() - t0
        time.sleep(max(0, (1.0 / config["FPS"]) - dt))

except KeyboardInterrupt:
    pass

# ——— Cleanup ———
stream.stop_stream()
stream.close()
pa.terminate()
matrix.Clear()
