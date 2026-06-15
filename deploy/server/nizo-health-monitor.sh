#!/usr/bin/env bash
# Nizo health monitor — captures the state leading UP TO a hard drop, so the recurring
# "box falls off the network under load" events become diagnosable instead of a mystery.
#
# The drops observed are full network losses (ping/SSH "host is down"), not service crashes —
# that points at a thermal shutdown, a power event, or a kernel hang, none of which a systemd
# Restart= can help with. This logs GPU + CPU temps, load, memory, and the kernel ring buffer
# tail every INTERVAL seconds to a persistent file on local disk. After the next drop + reboot,
# `tail` the log: a temp climbing into the 90s°C right before the gap = thermal; a clean cutoff
# with no warning = likely power; an OOM/panic in dmesg = software.
#
# Install as a systemd unit (nizo-health.service) or run under tmux. Append-only, ~1 line/cycle,
# so it's cheap to leave running forever.
#
# Env: NIZO_HEALTH_LOG (default /mnt/ai-models/logs/health.log), NIZO_HEALTH_INTERVAL (default 30)
set -uo pipefail

LOG="${NIZO_HEALTH_LOG:-/mnt/ai-models/logs/health.log}"
INTERVAL="${NIZO_HEALTH_INTERVAL:-30}"
mkdir -p "$(dirname "$LOG")"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# One-time header so a fresh boot is obvious in the log (the line after a gap = it came back).
echo "===== nizo-health start $(ts) (boot id $(cat /proc/sys/kernel/random/boot_id 2>/dev/null)) =====" >> "$LOG"

while true; do
  gpu=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  # CPU temp: prefer k10temp/coretemp via sensors; fall back to the hottest hwmon input.
  cpu=$(sensors 2>/dev/null | grep -iE 'Tctl|Tdie|Package id 0|Core 0' | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
  [ -z "$cpu" ] && cpu=$(awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
  load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)
  mem=$(free -m | awk '/Mem:/{print $3"/"$2"MB"}')
  echo "$(ts) | gpu(temp,util,W,memMB)=$gpu | cpuC=${cpu:-?} | load=$load | mem=$mem" >> "$LOG"

  # Capture any fresh kernel warnings/errors (thermal, OOM, hardware) since they're the smoking gun.
  dmesg --since "${INTERVAL}s ago" --level=err,warn,crit,alert,emerg 2>/dev/null \
      | grep -iE 'thermal|temperature|throttl|oom|killed process|hardware error|mce|gpu has fallen|Xid' \
      | sed "s/^/$(ts) [dmesg] /" >> "$LOG"

  sleep "$INTERVAL"
done
