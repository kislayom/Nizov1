#!/usr/bin/env python3
"""
Nizo aggressive GPU fan curve — NVML edition (no X required).

Why this file: on a truly headless server, the nvidia-settings path doesn't work
(Xvfb doesn't load the nvidia X driver). NVML talks to the kernel module directly
and works on workstation cards (Quadro / RTX A / RTX PRO series).

Operational properties:
  - atexit + signal handlers restore BIOS auto-control before dying.
  - On any temperature read failure, defaults to T=90°C → fan=100% (failsafe).
  - On any fan-set failure (driver doesn't support it), exits cleanly without
    leaving the card in a half-controlled state.
  - Logs each adjustment with timestamp; intended for foreground-with-tail use
    during testing, or systemd `journalctl -u <unit> -f` later.

Stop:
  - Ctrl-C in foreground
  - `pkill -f nizo-fan-curve.py`
  - `kill <pid>` (any of TERM/INT/HUP — all trapped)
"""
from __future__ import annotations

import argparse
import atexit
import signal
import sys
import time
from datetime import datetime

try:
    import pynvml
except ImportError:
    sys.stderr.write("pynvml not installed. Run:  pip install --user pynvml\n")
    sys.exit(2)


def log(msg: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


CURVE = [
    # (temp_c_below, fan_percent)
    (50,  40),
    (65,  60),
    (75,  80),
    (82,  90),
    (999, 100),
]


def fan_for_temp(t: int) -> int:
    for ceil, fan in CURVE:
        if t < ceil:
            return fan
    return 100


_handle = None
_num_fans = 0
_armed = False


def restore_auto() -> None:
    """Return all fans to driver-managed (BIOS curve) control."""
    global _armed
    if not _armed:
        return
    try:
        for i in range(_num_fans):
            pynvml.nvmlDeviceSetFanControlPolicy(
                _handle, i, pynvml.NVML_FAN_POLICY_TEMP_CONTINOUS_SW
            )
        log("restored auto fan control")
    except pynvml.NVMLError as e:
        log(f"WARN: could not restore auto control: {e}")
    finally:
        _armed = False
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def on_signal(signum, _frame):
    log(f"received signal {signum}, exiting")
    sys.exit(0)


def main():
    global _handle, _num_fans, _armed

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--poll", type=float, default=5.0, help="seconds between checks")
    ap.add_argument("--dry-run", action="store_true", help="log decisions, never set fan")
    args = ap.parse_args()

    pynvml.nvmlInit()
    _handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    name = pynvml.nvmlDeviceGetName(_handle)
    if isinstance(name, bytes):
        name = name.decode()
    drv = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(drv, bytes):
        drv = drv.decode()
    _num_fans = pynvml.nvmlDeviceGetNumFans(_handle)
    log(f"device gpu:{args.gpu}={name}  driver={drv}  fans={_num_fans}")

    if _num_fans == 0:
        log("ERROR: no controllable fans reported by NVML")
        sys.exit(3)

    # Probe: try setting fan[0] to its current value. If this fails, the card
    # doesn't support manual control via NVML and we exit BEFORE pretending to.
    cur = pynvml.nvmlDeviceGetFanSpeed_v2(_handle, 0)
    log(f"probe: current fan[0]={cur}%, attempting manual control...")
    try:
        if not args.dry_run:
            pynvml.nvmlDeviceSetFanSpeed_v2(_handle, 0, cur)
            _armed = True
    except pynvml.NVMLError as e:
        log(f"ERROR: NVML fan control unsupported on this card/driver: {e}")
        sys.exit(4)

    atexit.register(restore_auto)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)

    last_f = -1
    while True:
        try:
            t = pynvml.nvmlDeviceGetTemperature(_handle, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError as e:
            log(f"WARN: temp read failed ({e}) → defaulting T=90 → fan=100")
            t = 90

        f = fan_for_temp(int(t))

        if f != last_f:
            log(f"T={t}°C → fan={f}%  ({'DRY-RUN' if args.dry_run else 'apply'})")
            if not args.dry_run:
                try:
                    for i in range(_num_fans):
                        pynvml.nvmlDeviceSetFanSpeed_v2(_handle, i, f)
                except pynvml.NVMLError as e:
                    log(f"WARN: fan set failed: {e}")
            last_f = f

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
