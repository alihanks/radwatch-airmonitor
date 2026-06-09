#!/bin/bash
# syshealth_logger.sh — lightweight, journald-independent system-health logger for
# diagnosing freezes.
#
# Context: the 2026-06-07 freeze left almost no evidence because journald stopped
# writing ~8 hours before the box fully locked (see docs/runbook.md §7a/§7e). This
# logger does not depend on journald: it is a small resident loop that appends one
# line of vitals every INTERVAL seconds, reading mostly from /proc (in-memory, no
# disk I/O) so it has the best possible chance of recording the final moments before
# a hang — especially the disk-I/O-stall signature we suspect, where 'pblk' (procs
# blocked in uninterruptible sleep) climbs while CPU sits idle.
#
# Fields per line (space-separated; a '#' header is written on each (re)start):
#   ts        ISO-8601 local timestamp
#   load1/5/15  load averages (/proc/loadavg)
#   prun      procs running           (/proc/stat)
#   pblk      procs BLOCKED (D state)  (/proc/stat)  <- I/O-stall signal
#   memMB     MemAvailable, MB         (/proc/meminfo) <- falling = memory pressure
#   swapMB    SwapFree, MB             (/proc/meminfo) <- falling = swap pressure
#   tempC     CPU temperature, °C      (hwmon), or "?"
#   root%     root filesystem use%     (timeout-guarded df), or "?" if disk stalled
#   topproc   highest-RSS process name:MB (timeout-guarded ps), or "?"
#
# Usage:
#   bash maintenance/syshealth_logger.sh             # run the loop (foreground)
#   bash maintenance/syshealth_logger.sh --once      # print one sample and exit
#   sudo bash maintenance/syshealth_logger.sh --install    # install + start systemd service
#   sudo bash maintenance/syshealth_logger.sh --uninstall  # stop + remove the service
#   bash maintenance/syshealth_logger.sh --status    # service state + tail the log
#
# Overridable via env: SYSHEALTH_LOG (default /var/log/syshealth.log),
#                      SYSHEALTH_INTERVAL (default 30 seconds).

INTERVAL="${SYSHEALTH_INTERVAL:-30}"
LOGFILE="${SYSHEALTH_LOG:-/var/log/syshealth.log}"
MAXSIZE=$((5 * 1024 * 1024))     # rotate at 5 MB
SERVICE_NAME="syshealth-logger"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

require_root() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "This action needs root. Re-run with sudo." >&2
        exit 1
    fi
}

# Build one sample line. Reads /proc pseudo-files (no disk I/O) for the critical
# fields; the two disk-touching extras (df, ps) are timeout-guarded so a stalling
# disk can't hang the logger — they just record "?" for that tick.
sample_line() {
    local ts
    printf -v ts '%(%Y-%m-%dT%H:%M:%S)T' -1

    # /proc/loadavg: "0.12 0.08 0.05 1/234 5678"
    local l1 l5 l15 procfield rest
    read -r l1 l5 l15 procfield rest < /proc/loadavg
    local prun="${procfield%/*}"

    # /proc/stat: instantaneous procs_running / procs_blocked
    local pblk="?" prun2=""
    local key val junk
    while read -r key val junk; do
        case "$key" in
            procs_running) prun2="$val" ;;
            procs_blocked) pblk="$val" ;;
        esac
    done < /proc/stat
    [ -n "$prun2" ] && prun="$prun2"

    # /proc/meminfo: MemAvailable / SwapFree (kB)
    local memkb=0 swapkb=0 k v u
    while read -r k v u; do
        case "$k" in
            MemAvailable:) memkb="$v" ;;
            SwapFree:)     swapkb="$v" ;;
        esac
    done < /proc/meminfo
    local memMB=$((memkb / 1024))
    local swapMB=$((swapkb / 1024))

    # CPU temperature from hwmon (k10temp on AMD). millidegrees -> °C.
    local tempC="?" t raw
    for t in /sys/class/hwmon/hwmon*/temp1_input; do
        [ -r "$t" ] || continue
        read -r raw < "$t" 2>/dev/null || continue
        tempC=$(( raw / 1000 ))
        break
    done

    # Root FS usage — df touches the FS; guard so a stalled disk records "?".
    local rootpct
    rootpct="$(timeout 3 df -P / 2>/dev/null | awk 'NR==2{print $5}')"
    [ -z "$rootpct" ] && rootpct="?"

    # Highest-RSS process — ps reads /proc; guard it too.
    local topproc
    topproc="$(timeout 3 ps -eo rss=,comm= --sort=-rss 2>/dev/null | awk 'NR==1{printf "%s:%dMB", $2, $1/1024}')"
    [ -z "$topproc" ] && topproc="?"

    printf '%s %s %s %s %s %s %s %s %s %s %s\n' \
        "$ts" "$l1" "$l5" "$l15" "$prun" "$pblk" "$memMB" "$swapMB" "$tempC" "$rootpct" "$topproc"
}

rotate_if_needed() {
    [ -f "$LOGFILE" ] || return 0
    local sz
    sz=$(stat -c '%s' "$LOGFILE" 2>/dev/null || echo 0)
    if [ "$sz" -gt "$MAXSIZE" ]; then
        mv -f "$LOGFILE" "${LOGFILE}.old"
    fi
}

run_loop() {
    mkdir -p "$(dirname "$LOGFILE")"
    printf '# syshealth start %s interval=%ss\n# ts load1 load5 load15 prun pblk memMB swapMB tempC root%% topproc\n' \
        "$(date -Is 2>/dev/null)" "$INTERVAL" >> "$LOGFILE"
    while true; do
        rotate_if_needed
        sample_line >> "$LOGFILE"
        sleep "$INTERVAL"
    done
}

install_service() {
    require_root
    local script_path
    script_path="$(readlink -f "$0")"
    cat > "$SERVICE_PATH" <<EOF
[Unit]
Description=Lightweight system-health logger (freeze diagnosis) — see docs/runbook.md §7e
After=multi-user.target

[Service]
Type=simple
Environment=SYSHEALTH_LOG=${LOGFILE}
Environment=SYSHEALTH_INTERVAL=${INTERVAL}
ExecStart=/bin/bash ${script_path}
Restart=always
RestartSec=5
Nice=10

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now "${SERVICE_NAME}.service"
    echo "Installed and started ${SERVICE_NAME}.service"
    echo "  script:   $script_path"
    echo "  log:      $LOGFILE  (every ${INTERVAL}s)"
    echo "  tail it:  tail -f $LOGFILE"
}

uninstall_service() {
    require_root
    systemctl disable --now "${SERVICE_NAME}.service" 2>/dev/null || true
    rm -f "$SERVICE_PATH"
    systemctl daemon-reload
    echo "Removed ${SERVICE_NAME}.service (log file $LOGFILE left in place)."
}

show_status() {
    systemctl status "${SERVICE_NAME}.service" --no-pager 2>/dev/null || echo "(service not installed)"
    echo
    echo "=== last 15 log lines ($LOGFILE) ==="
    tail -15 "$LOGFILE" 2>/dev/null || echo "(no log yet)"
}

case "${1:-}" in
    --once)      sample_line ;;
    --install)   install_service ;;
    --uninstall) uninstall_service ;;
    --status)    show_status ;;
    "")          run_loop ;;
    *)
        echo "Unknown option: $1" >&2
        echo "Usage: $0 [--once|--install|--uninstall|--status]" >&2
        exit 1
        ;;
esac
