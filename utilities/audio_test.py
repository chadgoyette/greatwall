import pyaudio
import numpy as np

p = pyaudio.PyAudio()
SOURCE_NAME = 'null_monitor'  # We'll check if this matches any

print("Available input devices:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['maxInputChannels'] > 0:
        print(f"Index {i}: {dev['name']} (host: {dev['hostApi']})")

def get_index(name):
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if name in dev['name'] and dev['maxInputChannels'] > 0:
            return i
    return None

index = get_index(SOURCE_NAME)
if index is None:
    print("Monitor source not found! Try 'default' or 'pulse' as SOURCE_NAME, or use a listed index manually.")
    exit(1)

stream = p.open(format=pyaudio.paInt16, channels=2, rate=44100, input=True, input_device_index=index, frames_per_buffer=1024)

print("Capturing... Ctrl+C to stop.")
try:
    while True:
        data = stream.read(1024)
        data_np = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(data_np**2))
        print(f"Volume: {rms:.2f}")
except KeyboardInterrupt:
    pass

stream.stop_stream()
stream.close()
p.terminate()
