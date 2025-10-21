import pyaudio
import numpy as np
from rgbmatrix import RGBMatrix, RGBMatrixOptions, graphics

# Matrix setup (use your working params)
options = RGBMatrixOptions()
options.rows = 32
options.cols = 64
options.chain_length = 1
options.hardware_mapping = 'regular'  # Or 'adafruit-hat' if needed
options.brightness = 50
options.disable_hardware_pulsing = True
options.panel_type = "FM6126A"
options.gpio_slowdown = 4
options.multiplexing = 0
matrix = RGBMatrix(options = options)
canvas = matrix.CreateFrameCanvas()

p = pyaudio.PyAudio()
SOURCE_NAME = 'null_monitor'  # From your .asoundrc
RATE = 44100
CHUNK = 2048  # For FFT, power of 2
max_val = 1e-6

def get_index(name):
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if name in dev['name'] and dev['maxInputChannels'] > 0:
            return i
    return None

index = get_index(SOURCE_NAME)
if index is None:
    print("Monitor source not found!")
    exit(1)

stream = p.open(format=pyaudio.paInt16, channels=2, rate=RATE, input=True, input_device_index=index, frames_per_buffer=CHUNK)

print("Visualizing... Ctrl+C to stop.")
try:
    while True:
        data = stream.read(CHUNK)
        data_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)  # Stereo to mono avg
        data_np = (data_np[0::2] + data_np[1::2]) / 2

        # Window and FFT
        window = np.hanning(len(data_np))
        data_np *= window
        fft = np.abs(np.fft.rfft(data_np))
        freqs = np.fft.rfftfreq(CHUNK, 1/RATE)

        # Bin into 64 bars (log spaced for music)
        #bins = np.logspace(np.log10(20), np.log10(RATE/2), 65)  # 20Hz to Nyquist, 64 bins
        # To something more constrained (e.g., 50 Hz to 8 kHz):
        bins = np.logspace(np.log10(50), np.log10(8000), 65)
        bar_heights = np.zeros(64)
        for i in range(64):
            mask = (freqs >= bins[i]) & (freqs < bins[i+1])
            if np.any(mask):
                bar_heights[i] = np.mean(fft[mask])
        #bar_heights = np.log10(bar_heights + 1e-10)  # Log scale
        current_max = np.max(bar_heights)
        max_val = max(max_val * 0.95, current_max)  # slow decay
        #bar_heights = np.clip((bar_heights - np.min(bar_heights)) / (np.max(bar_heights) - np.min(bar_heights)) * 32, 0, 32)  # Scale to 32 rows
        bar_heights = np.clip((bar_heights / max_val) * 32, 0, 32)
        # Draw bars on matrix (green, simple)
        canvas.Clear()
        for x in range(64):
            height = int(bar_heights[x])
            for y in range(height):
                canvas.SetPixel(x, 32 - y - 1, 255, 255, 255)  # Adjust RGB values as needed
        canvas = matrix.SwapOnVSync(canvas)
except KeyboardInterrupt:
    pass

stream.stop_stream()
stream.close()
p.terminate()
matrix.Clear()
