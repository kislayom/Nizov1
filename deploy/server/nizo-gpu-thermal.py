#!/usr/bin/env python3
"""Nizo GPU fan governor — hold the core under a temperature ceiling using the MINIMUM fan
speed needed, and nothing else. GPU fan only; the power limit is never touched.

Why
---
This governs the GPU's TEMPERATURE. It does NOT (and must not) replace the 250W power cap —
those solve two different problems and BOTH are kept:
  * Power cap (nizo-gpucap, 250W): prevents the box dropping off the network under sustained
    load. That drop is power/electrical (PSU headroom) — it reproduced at 300W even while this
    governor kept the GPU cool, so a fan can't fix it.
  * This fan governor: prevents thermal CLOCK throttling. Within the 250W envelope the lazy
    BIOS fan curve (30->50%) still lets the core ride to ~79C and throttle clocks
    (2512->2317 MHz); we hold it lower with the minimum fan instead.
Closed loop on core temperature: nudge ONLY the GPU fan up as the core climbs toward the
target, ease it down when there's headroom, settle on the quietest speed under the ceiling.
Never touches the power limit, never touches any chassis/CPU fan (NVML exposes only the GPU's
own fan), restores the BIOS auto curve on exit.

Method (same as nizo-mon.py): NVML `nvmlDeviceSetFanSpeed_v2` — no Coolbits / Xvfb /
nvidia-settings. Must run as root.

Env:
  NIZO_GPU_TARGET_C   setpoint to hold the core near, default 71 (gives margin under ~75)
  NIZO_GPU_FAN_FLOOR  lowest fan % the loop will command, default 35
  NIZO_GPU_POLL_SEC   poll interval seconds, default 2
"""
from __future__ import annotations

import os
import signal
import sys
import time
import warnings

warnings.filterwarnings("ignore")
try:
    import pynvml as nv
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"pynvml/nvidia-ml-py not importable: {e}\n")
    sys.exit(1)

# Setpoint sits a few degrees under the 75C ceiling. Thermal momentum means the core keeps
# rising for a moment after the fan ramps, so holding at ~71 keeps transient peaks under 75.
# The loop still eases to the floor when idle, so this is "minimum fan that reliably holds
# under 75", not a fixed high speed.
TARGET_C = int(os.getenv("NIZO_GPU_TARGET_C", "71"))
FAN_FLOOR = int(os.getenv("NIZO_GPU_FAN_FLOOR", "35"))
POLL_SEC = float(os.getenv("NIZO_GPU_POLL_SEC", "2"))


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main() -> None:
    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByIndex(0)
    num_fans = nv.nvmlDeviceGetNumFans(h)

    def set_fan(pct: int) -> None:
        for i in range(num_fans):
            try:
                nv.nvmlDeviceSetFanSpeed_v2(h, i, pct)
            except Exception as e:
                log(f"WARN set fan {i}={pct} failed: {e}")

    def restore() -> None:
        log("restoring BIOS auto fan curve")
        for i in range(num_fans):
            try:
                nv.nvmlDeviceSetDefaultFanSpeed_v2(h, i)
            except Exception:
                pass

    stop = {"v": False}

    def on_sig(signum, _frame):
        stop["v"] = True

    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)

    # Seed from the current auto fan so the first move is smooth, not a jump.
    try:
        fan = max(FAN_FLOOR, nv.nvmlDeviceGetFanSpeed(h))
    except Exception:
        fan = FAN_FLOOR
    set_fan(fan)
    last_set = fan

    log(f"fan governor up (GPU fan only): target~{TARGET_C}C floor={FAN_FLOOR}% "
        f"fans={num_fans} poll={POLL_SEC}s — power limit left untouched")
    try:
        while not stop["v"]:
            t = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
            # Asymmetric loop: ease the fan DOWN gently when there's headroom (quiet), but ramp
            # UP increasingly hard as the core nears 75C so a sudden load never breaches the
            # ceiling. Steady state settles at the quietest fan that holds ~TARGET_C.
            if t >= 74:
                fan = min(100, fan + 20)          # hard guard: slam so the core never reaches 75
            elif t >= 72:
                fan = min(100, fan + 6)           # braking zone, catch it well before the ceiling
            elif t >= TARGET_C:                   # 71: gentle hold-up
                fan = min(100, fan + 2)
            elif t >= TARGET_C - 2:               # 69-70: deadband — hold, don't hunt
                pass
            else:
                fan = max(FAN_FLOOR, fan - 1)     # real headroom (<69): ease down for quiet

            if fan != last_set:
                set_fan(fan)
                # Log only meaningful moves to keep the journal readable.
                if abs(fan - last_set) >= 3 or fan in (FAN_FLOOR, 100):
                    log(f"core {t}C -> fan {fan}%")
                last_set = fan

            time.sleep(POLL_SEC)
    finally:
        restore()


if __name__ == "__main__":
    main()
