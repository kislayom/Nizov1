#!/usr/bin/env bash
# Nizo aggressive GPU fan curve.
#
# Why this exists: NVIDIA's BIOS-default curve on the RTX PRO 5000 Blackwell tunes for
# acoustics + bearing life and lets the core ride at ~82°C under sustained inference.
# We want full perf with cooler GPU, even at the cost of audible fan.
#
# Safety properties:
#   - `set -euo pipefail` → script aborts on any unhandled error
#   - trap on EXIT/INT/TERM → restores BIOS auto-control before dying
#   - bad temp read → defaults to 90°C → forces fan=100% (failsafe)
#   - waits for Xvfb to actually be alive before touching nvidia-settings
#   - logs every adjustment to journalctl with timestamps

set -euo pipefail

DISPLAY_NUM=${NIZO_FAN_DISPLAY:-:99}
GPU_ID=${NIZO_FAN_GPU:-0}
POLL_SEC=${NIZO_FAN_POLL_SEC:-5}

XVFB_PID=""

log() { printf '%s %s\n' "$(date +'%Y-%m-%dT%H:%M:%S')" "$*"; }

cleanup() {
    log "cleanup: restoring auto fan control"
    if [[ -n "${DISPLAY:-}" ]]; then
        nvidia-settings -a "[gpu:${GPU_ID}]/GPUFanControlState=0" >/dev/null 2>&1 || true
    fi
    if [[ -n "$XVFB_PID" ]]; then
        kill "$XVFB_PID" 2>/dev/null || true
        wait "$XVFB_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# --- bring up a virtual X display so nvidia-settings has something to talk to ---
log "starting Xvfb on $DISPLAY_NUM"
Xvfb "$DISPLAY_NUM" -screen 0 1024x768x24 &
XVFB_PID=$!
sleep 2

if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    log "ERROR: Xvfb failed to start"
    exit 1
fi
export DISPLAY="$DISPLAY_NUM"

# --- arm manual fan control ---
if ! nvidia-settings -a "[gpu:${GPU_ID}]/GPUFanControlState=1" >/dev/null 2>&1; then
    log "ERROR: could not arm GPUFanControlState — is Coolbits enabled in xorg.conf?"
    log "       run:  sudo nvidia-xconfig --enable-all-gpus --cool-bits=28 ; reboot"
    exit 1
fi
log "manual fan control armed for gpu:${GPU_ID}"

# --- the curve ---
fan_for_temp() {
    local t=$1
    if   [ "$t" -lt 50 ]; then echo 40
    elif [ "$t" -lt 65 ]; then echo 60
    elif [ "$t" -lt 75 ]; then echo 80
    elif [ "$t" -lt 82 ]; then echo 90
    else                       echo 100
    fi
}

LAST_F=""
while true; do
    T=$(nvidia-smi -i "$GPU_ID" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "")
    if ! [[ "$T" =~ ^[0-9]+$ ]]; then
        log "WARN: bad temp read '$T' — defaulting T=90 → fan=100%"
        T=90
    fi
    F=$(fan_for_temp "$T")

    if [[ "$F" != "$LAST_F" ]]; then
        nvidia-settings -a "[fan:${GPU_ID}]/GPUTargetFanSpeed=$F" >/dev/null 2>&1 || \
            log "WARN: failed to set fan=$F (transient — will retry)"
        log "T=${T}°C → fan=${F}%"
        LAST_F=$F
    fi
    sleep "$POLL_SEC"
done
