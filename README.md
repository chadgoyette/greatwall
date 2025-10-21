# Raspotify Audio + LED Matrix Visualizers

This project turns a Raspberry Pi into a Spotify Connect endpoint that also drives a 64×32 RGB LED matrix (and optional HDMI display) with real‑time audio visualizers. Raspotify streams audio into PulseAudio, where it is split between the 3.5 mm headphone jack and a null sink that feeds the Python visualizer scripts.

```
Spotify Client  →  librespot (raspotify.service)
                     │
                     ▼
           PulseAudio "combined" sink
                ├─► Analog audio (bcm2835 Headphones)
                └─► Null sink monitor (null_monitor) ──► PyAudio visualizers
```

The repository contains several visualization modes (FFT bars, waveforms, LCD/pygame display, etc.) as well as the systemd units and PulseAudio configuration that keep everything running unattended.

---

## Components at a Glance

- **PulseAudio configuration**
  - `/etc/pulse/default.pa` loads a `nullsink` and a `combined` sink that mirrors audio to the Pi headphone jack and to the null sink monitor (see `configs/pulse-default.pa` for the snippet).
  - `~/.asoundrc` exposes the monitor as an ALSA/PortAudio device called `null_monitor` so the Python scripts can subscribe with PyAudio (template in `configs/asoundrc`).
  - The per-user service `pulseaudio.service` (systemd user unit) is enabled and should always be running; the helper script `run-visualizer.sh` exports `XDG_RUNTIME_DIR` so that PulseAudio is reachable even when started by the root systemd unit.

- **Raspotify (`systemd/user/raspotify.service`)**
  - Wraps `/usr/bin/librespot` and reads options from `/etc/default/raspotify`.
  - The options include `--backend pulseaudio --device combined`, so Spotify playback is delivered straight into the PulseAudio combined sink.
  - The unit now includes `Requires=pulseaudio.service` and `After=pulseaudio.service`, guaranteeing PulseAudio is ready before the player launches.

- **Visualizer supervisor (`scripts/run-visualizer.sh`, `systemd/system/visualizer.service`)**
  - The shell script polls `pactl list sink-inputs` for a librespot stream.
  - When audio is present it spawns `python3 /home/admin/fft_hybrid_Waveform-v2.py`; when silent it tears the process down.
  - Environment bootstrapping (`XDG_RUNTIME_DIR`, `PULSE_RUNTIME_PATH`) has been added so it works from a root-owned systemd service.
  - Override the target script or interpreter by exporting `VIZ_SCRIPT`, `PYTHON_BIN`, or `VIZ_PATTERN` before starting the service.

- **Python visualizers (`visualizers/`)**
  - `fft_hybrid_Waveform-v2.py` – primary 64×32 matrix FFT visualizer with falloff caps.
  - `fft_hybrid_Waveform.py`, `fft_waveform_visualizer.py`, `waveform_visualizer.py`, `visualizer.py` – alternate matrix modes.
  - `lcd_visualizer.py` and variants – FFT display rendered via pygame on an HDMI-connected LCD.
  - `image_scroller.py` – scrolls PNG posters across the 64×64 array for font/art previews.

- **Third-party library**: `rpi-rgb-led-matrix/` – compiled `rgbmatrix` Python bindings for driving the HUB75 panels.

---

## Prerequisites

1. **Hardware**
   - Raspberry Pi (3B+ or newer recommended).
   - HUB75 RGB panel (64×32) plus an RGB matrix bonnet/HAT compatible with `rpi-rgb-led-matrix`.
   - Optional HDMI LCD for the pygame visualizer.

2. **System packages**
   ```bash
   sudo apt update
   sudo apt install raspotify pulseaudio python3 python3-pip python3-pyaudio \
       python3-numpy python3-pygame git build-essential pkg-config libfreetype6-dev
   ```

3. **LED matrix library**
   ```bash
   cd ~/rpi-rgb-led-matrix
   sudo apt install python3-dev python3-venv
   make build-python
   sudo make install-python
   ```

4. **Python dependencies**
   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install numpy pygame pyaudio
   ```

---

## PulseAudio Configuration

PulseAudio is the hub of the signal routing:

- `/etc/pulse/default.pa` (system-wide) contains:
  ```pa
  load-module module-null-sink sink_name=nullsink
  load-module module-combine-sink sink_name=combined slaves=nullsink,alsa_output.platform-fe00b840.mailbox.stereo-fallback
  set-default-sink combined
  ```
  This creates the `combined` sink that mirrors to the headphone DAC and the `nullsink`.

- `~/.asoundrc` exposes the monitor as ALSA devices:
  ```ini
  pcm.null_monitor {
      type pulse
      device nullsink.monitor
  }

  ctl.null_monitor {
      type pulse
      device nullsink.monitor
  }
  ```

- Validate after login:
  ```bash
  systemctl --user status pulseaudio
  pactl info
  pactl list sinks short
  ```
  You should see `combined` as the default sink and `combined.monitor`/`nullsink.monitor` as sources.

If PulseAudio ever fails to start because of a stale PID file, run:
```bash
pulseaudio -k
systemctl --user restart pulseaudio
```

---

## Raspotify Configuration

`/etc/default/raspotify` is already set up:

```bash
DEVICE_NAME="Greatwall"
BACKEND_NAME="pulseaudio"
OPTIONS="--backend pulseaudio --device combined --name Greatwall"
```

The custom user service lives at `~/.config/systemd/user/raspotify.service`:

```ini
[Unit]
Description=Raspotify (Spotify Connect Client)
Documentation=https://github.com/dtcooper/raspotify
Requires=pulseaudio.service
After=pulseaudio.service

[Service]
ExecStart=/usr/bin/librespot $OPTIONS
EnvironmentFile=-/etc/default/raspotify
Restart=always
RestartSec=10
Environment=HOME=/home/admin

[Install]
WantedBy=default.target
```

Reload and restart if you change anything:
```bash
systemctl --user daemon-reload
systemctl --user restart raspotify
```

Check status/logs:
```bash
systemctl --user status raspotify
journalctl --user -u raspotify -f
```

---

## Visualizer Service

`/usr/local/bin/run-visualizer.sh` supervises the Python process. Key features:

- Exports `XDG_RUNTIME_DIR`/`PULSE_RUNTIME_PATH` so `pactl` works in a root-owned systemd unit.
- Waits until PulseAudio is ready before polling for librespot streams.
- Starts/stops `fft_hybrid_Waveform-v2.py` based on whether Spotify audio is present.
- Cleans up the Python process on exit.

The systemd wrapper (`/etc/systemd/system/visualizer.service`):

```ini
[Unit]
Description=Auto-start FFT Hybrid Waveform Visualizer when Spotify audio plays
After=network.target sound.target

[Service]
Type=simple
ExecStart=/usr/local/bin/run-visualizer.sh
Restart=always
User=admin

[Install]
WantedBy=multi-user.target
```

After modifying the script or unit:
```bash
sudo systemctl daemon-reload
sudo systemctl restart visualizer.service
sudo systemctl enable visualizer.service   # optional: autostart on boot
sudo systemctl status visualizer.service
```

Inspect the log for the helper script:
```bash
sudo journalctl -u visualizer.service -f
```

---

## Running the Visualizers Manually

All scripts expect the LED matrix hardware to be connected and the `rgbmatrix` Python bindings available.

Examples:

```bash
python3 visualizers/fft_hybrid_Waveform-v2.py         # default FFT matrix visualizer
python3 visualizers/waveform_visualizer.py            # smooth waveform line
python3 visualizers/visualizer.py                     # log-spaced FFT bars
python3 visualizers/lcd_visualizer.py                 # pygame LCD mirror
python3 visualizers/image_scroller.py                # scroll PNGs from images/
```

`image_scroller.py` looks for image files in `~/images` by default (PNG/JPG/BMP/GIF; override with `--pattern` pointing at a directory or glob), automatically scales them to the active matrix size, scrolls each in from the right, pauses for 5 seconds, and scrolls off to the left; it loops until stopped. Adjust panel geometry with `--rows`, `--cols`, `--chain`, and `--parallel` (defaults target a single 64×32 panel; set `--parallel 2` for two panels stacked to 64×64). Use `--scale 1.25` (for example) to zoom the artwork beyond the panel bounds while keeping it centered.

Most scripts look up the input device named `null_monitor`; if you change the sink name in PulseAudio, update `config["SOURCE_NAME"]` (or `SOURCE_NAME` constants) accordingly.

To list available devices:
```bash
python3 - <<'PY'
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(i, info["name"], info["maxInputChannels"], info["maxOutputChannels"])
PY
```

---

## Deployment Workflow

- Copy the desired visualizer from `visualizers/` to the host (or run in-place).
- Install/update `scripts/run-visualizer.sh` to `/usr/local/bin/` and ensure it is executable (`chmod +x`).
- Deploy the unit files from `systemd/` to their respective systemd directories and reload the daemons.
- Keep `pulseaudio.service` and `raspotify.service` running in your user session when testing.
- Stop the systemd visualizer before running scripts manually to avoid conflicts:
  ```bash
  sudo systemctl stop visualizer.service
  python3 fft_hybrid_Waveform-v2.py
  sudo systemctl start visualizer.service
  ```
- For rapid iteration on LED output, run scripts with `sudo` only if your HAT requires elevated GPIO access. The current configuration with `rgbmatrix` works as non-root when your user is in the `gpio` group.

---

## Troubleshooting

- **`pactl` fails with “Connection refused”**
  - Ensure `pulseaudio.service` is running: `systemctl --user restart pulseaudio`.
  - Confirm `XDG_RUNTIME_DIR=/run/user/1000` if running commands via `sudo`.

- **No audio on the LED matrix but headphones work**
  - Verify the null sink monitor is live: `pactl list sources short`.
  - Make sure `OPTIONS` in `/etc/default/raspotify` still points to `--device combined`.
  - Restart the visualizer supervisor: `sudo systemctl restart visualizer.service`.

- **Python cannot find `null_monitor`**
  - Confirm the alias exists: `aplay -L | grep null_monitor`.
  - Ensure `~/.asoundrc` matches the sink name (`nullsink.monitor`).

- **Raspotify not visible in Spotify app**
  - Check `systemctl --user status raspotify` for errors (bad credentials, etc.).
  - Verify the Pi has network connectivity and the zeroconf port (default 5353) is open.

- **PulseAudio complains about HDMI profiles** – harmless for this setup; the headphone jack still works. The messages can be ignored.

---

## Repository Layout (key items)

```
visualizers/                 # Python LED matrix & HDMI visualizer scripts
scripts/run-visualizer.sh    # Supervisor that starts/stops fft_hybrid_Waveform-v2.py
systemd/user/raspotify.service
systemd/system/visualizer.service
configs/asoundrc             # ALSA template pointing at the null sink monitor
configs/pulse-default.pa     # PulseAudio snippet defining nullsink + combined sink
requirements.txt             # Python dependencies for the visualizers
README.md                    # This guide
```

---

## Quick Reference Commands

```bash
# PulseAudio
systemctl --user status pulseaudio
systemctl --user restart pulseaudio

# Raspotify
systemctl --user restart raspotify
systemctl --user status raspotify

# Visualizer supervisor
sudo systemctl restart visualizer.service
sudo systemctl status visualizer.service

# Audio routing checks
pactl info
pactl list sink-inputs
pactl list sinks short
```

With these pieces in place the Spotify stream is delivered to the analog output for speakers while simultaneously driving the LED matrix visualizers via the null sink monitor. Share this document with other developers to get them from fresh Pi to pulsing LEDs quickly.
