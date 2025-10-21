import time
import math
import numpy as np
import pyaudio
from rgbmatrix import RGBMatrix, RGBMatrixOptions

# Configuration
config = {
    # Audio / FFT
    "SOURCE_NAME": "null_monitor",
    "RATE": 44100,
    "CHUNK": 2048,
    "BANDS": 64,

    # Display
    "ROWS": 32,
    "COLS": 64,

    # Framerate & fade
    "FPS": 25.0,
    "FADE_TIME": 0.5,         # seconds until a pixel fades to FADE_FLOOR
    "FADE_FLOOR": 0.0001,     # brightness floor after FADE_TIME

    # Peak caps
    "PEAK_DECAY": 0.9,        # per-frame multiplier for peak hold

    # Normalization
    "MAX_DECAY": 0.95,        # per-frame multiplier for running max

    # Mirroring: “inverted” or “identical”
    "MIRROR_MODE": "inverted",
}

# Derived parameters
FADE_FACTOR = math.exp(
    math.log(config["FADE_FLOOR"]) / (config["FADE_TIME"] * config["FPS"])
)

# Matrix Setup
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

# Audio Setup
p = pyaudio.PyAudio()

def get_index(name):
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if name in dev['name'] and dev['maxInputChannels'] > 0:
            return i
    return None

index = get_index(config["SOURCE_NAME"])
if index is None:
    print("Monitor source not found!")
    exit(1)

stream = p.open(format=pyaudio.paInt16,
                channels=2,
                rate=config["RATE"],
                input=True,
                input_device_index=index,
                frames_per_buffer=config["CHUNK"])

# Buffers
frame_buf = np.zeros((config["ROWS"], config["COLS"]), dtype=np.float32)
peak_buf = np.zeros(config["BANDS"], dtype=np.float32)
running_max = 1e-6

print("Visualizing... Ctrl+C to stop.")
try:
    while True:
        start_time = time.time()

        # 1. Read audio & mono conversion
        data = stream.read(config["CHUNK"], exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        samples = (samples[0::2] + samples[1::2]) / 2

                # 2. FFT & log-spaced banding (safe against empty bins)
        windowed = samples * np.hanning(len(samples))
        fft = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(config["CHUNK"], 1.0 / config["RATE"])
        bins = np.logspace(np.log10(50), np.log10(8000), config["BANDS"] + 1)

        mags = np.zeros(config["BANDS"], dtype=np.float32)
        for i in range(config["BANDS"]):
            mask = (freqs >= bins[i]) & (freqs < bins[i+1])
            if mask.any():
                mags[i] = fft[mask].mean()
            else:
                mags[i] = 0.0  # no data in this band
                
        # Prevent any NaNs sneaking through
        mags = np.nan_to_num(mags, nan=0.0, posinf=0.0, neginf=0.0)


        # 3. Normalize
        current_peak = mags.max()
        running_max = max(running_max * config["MAX_DECAY"], current_peak)
        norm = mags / running_max
        norm = np.clip(norm, 0.0, 1.0)

        # 4. Fade frame buffer
        frame_buf *= FADE_FACTOR

        # 5. Draw filled inverted curve
        #   For each x, interpolate norm across all columns
        x_coords = np.linspace(0, config["COLS"] - 1, config["BANDS"])
        interp = np.interp(np.arange(config["COLS"]), x_coords, norm)
        half = config["ROWS"] // 2
        for x in range(config["COLS"]):
            height = int(interp[x] * half)
            top_y = half - 1 - height
            bottom_y = half + height if config["MIRROR_MODE"] == "inverted" else half + height
            # fill between top_y and half-1
            frame_buf[top_y:half, x] = 1.0
            # fill between half and bottom_y
            frame_buf[half:bottom_y+1, x] = 1.0

        # 6. Update & draw peak caps
        peak_buf = np.maximum(peak_buf, interp)
        peak_buf *= config["PEAK_DECAY"]
        for x in range(config["COLS"]):
            cap_h = int(peak_buf[int(x * config["BANDS"] / config["COLS"])] * half)
            # top cap
            frame_buf[half - 1 - cap_h, x] = 1.0
            # bottom cap
            frame_buf[half + cap_h, x] = 1.0

        # 7. Render to matrix
        #    Map [0,1]→[0,255] white
        canvas.Clear()
        fb_uint = (frame_buf * 255).astype(np.uint8)
        for y in range(config["ROWS"]):
            for x in range(config["COLS"]):
                val = int(fb_uint[y, x])
                if val:
                    canvas.SetPixel(x, y, val, val, val)
        canvas = matrix.SwapOnVSync(canvas)

        # 8. Frame timing
        elapsed = time.time() - start_time
        sleep = max(0, (1.0 / config["FPS"]) - elapsed)
        time.sleep(sleep)

except KeyboardInterrupt:
    pass

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
matrix.Clear()
