#!/bin/bash
# run-visualizer.sh – launches FFT Hybrid Waveform when Spotify audio is present

set -u -o pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VIZ_SCRIPT="${VIZ_SCRIPT:-/home/admin/fft_hybrid_Waveform-v2.py}"
VIZ_PATTERN="${VIZ_PATTERN:-$VIZ_SCRIPT}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-2}"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$UID}"
export PULSE_RUNTIME_PATH="${PULSE_RUNTIME_PATH:-$XDG_RUNTIME_DIR/pulse}"
PATH="/usr/bin:/bin:$PATH"

log() {
  echo "[run-visualizer] $*" >&2
}

wait_for_pulseaudio() {
  local attempts=0
  until pactl info >/dev/null 2>&1; do
    if (( attempts == 0 )); then
      log "Waiting for PulseAudio to come up..."
    elif (( attempts % 5 == 0 )); then
      log "Still waiting for PulseAudio (attempt ${attempts})..."
    fi
    ((attempts++))
    sleep 1
  done
  if (( attempts > 0 )); then
    log "PulseAudio is ready."
  fi
}

start_visualizer() {
  if ! pgrep -f "$VIZ_PATTERN" >/dev/null; then
    log "Starting visualizer."
    "$PYTHON_BIN" "$VIZ_SCRIPT" &
  fi
}

stop_visualizer() {
  if pgrep -f "$VIZ_PATTERN" >/dev/null; then
    log "Stopping visualizer process."
    pkill -f "$VIZ_PATTERN"
  fi
}

trap stop_visualizer EXIT

wait_for_pulseaudio

while true; do
  if ! pactl info >/dev/null 2>&1; then
    wait_for_pulseaudio
    sleep "$CHECK_INTERVAL_SEC"
    continue
  fi

  if pactl list sink-inputs 2>/dev/null | grep -qi "librespot"; then
    start_visualizer
  else
    stop_visualizer
  fi

  sleep "$CHECK_INTERVAL_SEC"
done
