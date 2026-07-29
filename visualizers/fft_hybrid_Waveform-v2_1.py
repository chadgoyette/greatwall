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

    # Display geometry
    "PANEL_ROWS": 32,    # physical rows on a single HUB75 panel
    "PANEL_COLS": 64,    # physical columns on a single panel
    "CHAIN_LENGTH": 2,   # number of panels daisy-chained horizontally
    "PARALLEL_COUNT": 1, # number of panels stacked vertically via HUB75 parallel connectors
    "CHAIN_LAYOUT": "vertical",  # "horizontal" = treat chain as wide display, "vertical" = virtual vertical stack
    "ROTATION_DEG": 0,   # rotate output: 0, 90, 180, or 270 (90/270 require a square virtual matrix)

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
    # Per-panel white balance multipliers (R, G, B), left-to-right in chain order.
    # For 2 panels: [(panel_1_r, panel_1_g, panel_1_b), (panel_2_r, panel_2_g, panel_2_b)]
    "PANEL_WHITE_BALANCE": [(.849, 1.0, 1.0), (1.0, 1.0, 1.0)],
}

# ——— Derived constants ———
PANEL_ROWS = config["PANEL_ROWS"]
PANEL_COLS = config["PANEL_COLS"]
CHAIN_LENGTH = max(1, config.get("CHAIN_LENGTH", 1))
PARALLEL_COUNT = max(1, config.get("PARALLEL_COUNT", 1))
CHAIN_LAYOUT = config.get("CHAIN_LAYOUT", "horizontal").lower()

if CHAIN_LAYOUT not in ("horizontal", "vertical"):
    raise ValueError("CHAIN_LAYOUT must be 'horizontal' or 'vertical'.")

if CHAIN_LAYOUT == "horizontal":
    VIRTUAL_ROWS = PANEL_ROWS * PARALLEL_COUNT
    VIRTUAL_COLS = PANEL_COLS * CHAIN_LENGTH
else:
    if PARALLEL_COUNT != 1:
        raise ValueError("CHAIN_LAYOUT 'vertical' currently requires PARALLEL_COUNT = 1.")
    VIRTUAL_ROWS = PANEL_ROWS * CHAIN_LENGTH
    VIRTUAL_COLS = PANEL_COLS * PARALLEL_COUNT

HARDWARE_ROWS = PANEL_ROWS * PARALLEL_COUNT
HARDWARE_COLS = PANEL_COLS * CHAIN_LENGTH

FADE_FACTOR = math.exp(math.log(config["FADE_FLOOR"]) / (config["FADE_TIME"] * config["FPS"]))
base_r, base_g, base_b = config["COLOR"]
raw_wb = config.get("PANEL_WHITE_BALANCE", [(1.0, 1.0, 1.0)])
if len(raw_wb) < CHAIN_LENGTH:
    raw_wb = list(raw_wb) + [raw_wb[-1]] * (CHAIN_LENGTH - len(raw_wb))
panel_white_balance = np.array(raw_wb[:CHAIN_LENGTH], dtype=np.float32)
if panel_white_balance.shape != (CHAIN_LENGTH, 3):
    raise ValueError("PANEL_WHITE_BALANCE must be a list of (R, G, B) tuples.")
panel_white_balance = np.clip(panel_white_balance, 0.0, 2.0)
panel_base_rgb = np.clip(
    np.rint(panel_white_balance * np.array([base_r, base_g, base_b], dtype=np.float32)),
    0,
    255,
).astype(np.uint16)
intensity = np.arange(256, dtype=np.uint16)
panel_rgb_lut = np.empty((CHAIN_LENGTH, 256, 3), dtype=np.uint8)
for p in range(CHAIN_LENGTH):
    pr, pg, pb = panel_base_rgb[p]
    panel_rgb_lut[p, :, 0] = ((intensity * pr) // 255).astype(np.uint8)
    panel_rgb_lut[p, :, 1] = ((intensity * pg) // 255).astype(np.uint8)
    panel_rgb_lut[p, :, 2] = ((intensity * pb) // 255).astype(np.uint8)

# ——— LED Matrix Setup ———
options = RGBMatrixOptions()
options.rows = PANEL_ROWS
options.cols = PANEL_COLS
options.chain_length = CHAIN_LENGTH
options.parallel = PARALLEL_COUNT
options.hardware_mapping = 'regular'
options.brightness = 50
options.disable_hardware_pulsing = True
options.panel_type = "FM6126A"
options.gpio_slowdown = 4
options.multiplexing = 0

ROTATION = config.get("ROTATION_DEG", 0) % 360
if ROTATION in (90, 270) and VIRTUAL_ROWS != VIRTUAL_COLS:
    raise ValueError("ROTATION_DEG of 90/270 requires a square matrix layout.")

matrix = RGBMatrix(options=options)
canvas = matrix.CreateFrameCanvas()

def rotate_buffer(buf: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return buf
    if rotation == 90:
        return np.rot90(buf, k=-1)
    if rotation == 180:
        return np.rot90(buf, k=2)
    if rotation == 270:
        return np.rot90(buf, k=1)
    return buf

def map_virtual_to_hardware(buf: np.ndarray, layout: str) -> np.ndarray:
    """Map a virtual canvas onto the actual panel arrangement."""
    if layout == "horizontal":
        if buf.shape != (HARDWARE_ROWS, HARDWARE_COLS):
            raise ValueError("Virtual buffer size does not match hardware layout.")
        return buf

    # layout == "vertical": treat chained panels as a vertical stack
    if PARALLEL_COUNT != 1:
        raise ValueError("Vertical chain layout currently supports PARALLEL_COUNT = 1.")

    stripes = CHAIN_LENGTH
    if buf.shape[0] < stripes * PANEL_ROWS or buf.shape[1] != PANEL_COLS:
        raise ValueError("Virtual buffer dimensions do not match vertical chain expectations.")

    hardware = np.zeros((HARDWARE_ROWS, HARDWARE_COLS), dtype=buf.dtype)
    for stripe in range(stripes):
        y0 = stripe * PANEL_ROWS
        y1 = y0 + PANEL_ROWS
        slice_buf = buf[y0:y1, :PANEL_COLS]
        dest_x = (stripes - 1 - stripe) * PANEL_COLS
        hardware[:, dest_x:dest_x + PANEL_COLS] = slice_buf
    return hardware

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

# ——— FFT precompute ———
window = np.hanning(config["CHUNK"]).astype(np.float32)
freqs = np.fft.rfftfreq(config["CHUNK"], 1.0 / config["RATE"])
bins = np.logspace(
    np.log10(config["LOW_FREQ"]),
    np.log10(config["HIGH_FREQ"]),
    config["BANDS"] + 1
)
band_masks = [(freqs >= bins[i]) & (freqs < bins[i + 1]) for i in range(config["BANDS"])]
x_bins = np.linspace(0, VIRTUAL_COLS - 1, config["BANDS"])

# ——— Buffers ———
frame_buf   = np.zeros((VIRTUAL_ROWS, VIRTUAL_COLS), dtype=np.float32)
peak_buf    = np.zeros(config["BANDS"], dtype=np.float32)
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
        windowed = mono * window
        fft_vals = np.abs(np.fft.rfft(windowed))

        mags = np.zeros(config["BANDS"], dtype=np.float32)
        for i in range(config["BANDS"]):
            mask = band_masks[i]
            mags[i] = fft_vals[mask].mean() if mask.any() else 0.0

        # 3. Normalize magnitudes
        peak = mags.max()
        running_max = max(running_max * config["MAX_DECAY"], peak)
        norm = np.clip(mags / running_max, 0.0, 1.0)

        # 4. Fade previous frame
        frame_buf *= FADE_FACTOR

        # 5. Draw inverted, filled curve
        interp = np.interp(np.arange(VIRTUAL_COLS), x_bins, norm)
        half   = VIRTUAL_ROWS // 2

        for x in range(VIRTUAL_COLS):
            h = int(interp[x] * half)
            if config["MIRROR_MODE"] == "inverted":
                top = max(0, half - 1 - h)
                bottom = min(VIRTUAL_ROWS - 1, half + h)
                frame_buf[top:half, x] = 1.0
                # Single center baseline: bottom side starts only when h > 0.
                if h > 0:
                    frame_buf[half:bottom, x] = 1.0
            else:
                top = max(0, half - 1 - h)
                bottom = min(VIRTUAL_ROWS - 1, half + h)
                frame_buf[top:half, x] = 1.0
                frame_buf[half:bottom + 1, x] = 1.0

        # 6. Peak caps
        peak_buf = np.maximum(peak_buf, interp)
        peak_buf *= config["PEAK_DECAY"]
        for x in range(VIRTUAL_COLS):
            cap_idx = int(x * config["BANDS"] / VIRTUAL_COLS)
            cap_h = int(peak_buf[cap_idx] * half)
            if config["MIRROR_MODE"] == "inverted":
                cap_top = max(0, half - 1 - cap_h)
                frame_buf[cap_top, x] = 1.0
                if cap_h > 0:
                    cap_bottom = min(VIRTUAL_ROWS - 1, half + cap_h - 1)
                    frame_buf[cap_bottom, x] = 1.0
            else:
                frame_buf[max(0, half - 1 - cap_h), x] = 1.0
                frame_buf[min(VIRTUAL_ROWS - 1, half + cap_h), x] = 1.0

        # 7. Render with color scaling
        render_buf = rotate_buffer(frame_buf, ROTATION)
        mapped_buf = map_virtual_to_hardware(render_buf, CHAIN_LAYOUT)

        canvas.Clear()
        fb_uint = (mapped_buf * 255).astype(np.uint8)
        rows, cols = mapped_buf.shape
        for panel_idx in range(CHAIN_LENGTH):
            x0 = panel_idx * PANEL_COLS
            x1 = min(cols, x0 + PANEL_COLS)
            lut = panel_rgb_lut[panel_idx]
            for y in range(rows):
                for x in range(x0, x1):
                    v = fb_uint[y, x]
                    if v > 0:
                        r, g, b = lut[v]
                        canvas.SetPixel(x, y, int(r), int(g), int(b))
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
