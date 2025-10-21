import pyaudio
import numpy as np
from scipy.interpolate import interp1d
from rgbmatrix import RGBMatrix, RGBMatrixOptions

# === CONFIGURATION ===
MATRIX_ROWS = 32
MATRIX_COLS = 64
BRIGHTNESS = 50

RATE = 44100         # Audio sample rate
CHUNK = 1024          # Samples per audio frame (controls refresh rate)
NUM_BUCKETS = 12       # Number of frequency bands to visualize (crests)

SOURCE_NAME = 'null_monitor'

# === SETUP MATRIX ===
options = RGBMatrixOptions()
options.rows = MATRIX_ROWS
options.cols = MATRIX_COLS
options.chain_length = 1
options.hardware_mapping = 'regular'
options.brightness = BRIGHTNESS
options.disable_hardware_pulsing = True
options.panel_type = "FM6126A"
options.gpio_slowdown = 4
options.multiplexing = 0

matrix = RGBMatrix(options=options)
canvas = matrix.CreateFrameCanvas()

# === SETUP AUDIO ===
p = pyaudio.PyAudio()

def get_index(name):
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if name in dev['name'] and dev['maxInputChannels'] > 0:
            return i
    return None

device_index = get_index(SOURCE_NAME)
if device_index is None:
    print("Monitor source not found!")
    exit(1)

stream = p.open(format=pyaudio.paInt16,
                channels=2,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK)

# === MAIN LOOP ===
print("Standing wave visualizer running...")
try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        mono = (samples[0::2] + samples[1::2]) / 2

        # Apply window & FFT
        window = np.hanning(len(mono))
        mono *= window
        fft = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(len(mono), 1 / RATE)

        # Define bucket edges (log scale)
        edges = np.logspace(np.log10(50), np.log10(8000), NUM_BUCKETS + 1)
        amplitudes = []
        for i in range(NUM_BUCKETS):
            band = (freqs >= edges[i]) & (freqs < edges[i + 1])
            power = np.mean(fft[band]) if np.any(band) else 0
            amplitudes.append(power)

        # Normalize and smooth
        amplitudes = np.log1p(amplitudes)
        norm = amplitudes / (np.max(amplitudes) + 1e-6)
        heights = norm * (MATRIX_ROWS // 2 - 1)

        # Center-aligned waveform points
        x_vals = np.linspace(0, MATRIX_COLS - 1, NUM_BUCKETS)
        y_vals = MATRIX_ROWS // 2 - heights

        # Interpolate to full matrix width
        interp = interp1d(x_vals, y_vals, kind='cubic')
        xs = np.arange(MATRIX_COLS)
        ys = interp(xs)
        ys = np.clip(ys, 0, MATRIX_ROWS - 1).astype(int)

        # Draw
        canvas.Clear()
        for x, y in zip(xs, ys):
            canvas.SetPixel(x, y, 255, 255, 255)
        canvas = matrix.SwapOnVSync(canvas)

except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    matrix.Clear()
