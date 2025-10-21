import pyaudio
import numpy as np
from rgbmatrix import RGBMatrix, RGBMatrixOptions

# === CONFIG ===
AUDIO_SOURCE = 'null_monitor'
CHUNK = 1024
RATE = 48000
AMPLITUDE_SCALE = 40        # Lower = shorter waves
SMOOTHING = 0.9              # 0.9 = smooth, 0.5 = snappy
WAVE_COLOR = (255, 255, 255) # RGB 215,214,214=warm white
DISPLAY_WIDTH = 64
DISPLAY_HEIGHT = 32

# === MATRIX SETUP ===
options = RGBMatrixOptions()
options.rows = 32
options.cols = 64
options.chain_length = 1
options.hardware_mapping = 'regular'
options.brightness = 50
options.panel_type = "FM6126A"
options.gpio_slowdown = 4
options.disable_hardware_pulsing = True
matrix = RGBMatrix(options=options)
canvas = matrix.CreateFrameCanvas()

# === AUDIO SETUP ===
p = pyaudio.PyAudio()

def get_device_index(name):
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if name in dev['name'] and dev['maxInputChannels'] > 0:
            return i
    return None

index = get_device_index(AUDIO_SOURCE)
if index is None:
    print("Audio input not found.")
    exit(1)

stream = p.open(format=pyaudio.paInt16, channels=2, rate=RATE,
                input=True, input_device_index=index, frames_per_buffer=CHUNK)

# === DRAW LOOP ===
print("Running... Ctrl+C to stop.")
prev_wave = np.zeros(DISPLAY_WIDTH)

try:
    while True:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        mono = (data[0::2] + data[1::2]) / 2

        # Downsample and normalize
        interp = np.interp(np.linspace(0, len(mono), DISPLAY_WIDTH), np.arange(len(mono)), mono)
        interp /= np.max(np.abs(interp) + 1e-6)
        wave = interp * AMPLITUDE_SCALE + DISPLAY_HEIGHT // 2

        # Smooth with low-pass filter
        smoothed = SMOOTHING * prev_wave + (1 - SMOOTHING) * wave
        prev_wave = smoothed

        # Draw thin continuous line
        canvas.Clear()
        for x in range(DISPLAY_WIDTH - 1):
            y1 = int(np.clip(smoothed[x], 0, DISPLAY_HEIGHT - 1))
            y2 = int(np.clip(smoothed[x + 1], 0, DISPLAY_HEIGHT - 1))

            # Draw line between y1 and y2
            if y1 == y2:
                canvas.SetPixel(x, y1, *WAVE_COLOR)
            else:
                step = 1 if y2 > y1 else -1
                for y in range(y1, y2 + step, step):
                    canvas.SetPixel(x, y, *WAVE_COLOR)

        canvas = matrix.SwapOnVSync(canvas)

except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    matrix.Clear()
