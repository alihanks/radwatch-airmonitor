#!/bin/bash
# setup_watchdog.sh — configure automatic recovery so the dosenet server reboots
# itself when it hangs, instead of sitting wedged for days.
#
# Context: after the 2026-05 SRSO soft-lockup freezes were worked around with the
# spec_rstack_overflow=off kernel parameter (see docs/runbook.md §7a), the server
# hung AGAIN on 2026-06-07 with a different, silent signature: journald stopped
# logging hours before the box fully froze, and nothing was written to explain it.
# It stayed wedged ~2 days until a manual power-cycle. Root cause is still under
# investigation (an aging-HDD I/O stall is the leading hypothesis). Regardless of
# cause, a watchdog turns a multi-day outage into a ~20-second one and keeps the
# pipeline running.
#
# What this configures:
#   1. A hardware watchdog device (sp5100_tco on this AMD board; softdog fallback),
#      loaded at boot and driven by systemd's RuntimeWatchdogSec. If systemd (PID 1)
#      stops petting the device, the hardware resets the machine.
#   2. Reboot-on-panic sysctls (kernel.panic, kernel.panic_on_oops) so kernel
#      panics/oopses also auto-reboot instead of halting.
#
# See docs/runbook.md §7e for the full write-up and the procedure to TEST that the
# auto-reboot actually fires — an untested watchdog is not a safety net.
#
# Usage:
#   sudo bash maintenance/setup_watchdog.sh           # apply configuration
#   bash maintenance/setup_watchdog.sh --check        # report current state only

set -e

WATCHDOG_TIMEOUT="20s"     # systemd pets the device every WATCHDOG_TIMEOUT/2
MODULES_FILE="/etc/modules-load.d/watchdog.conf"
SYSTEMD_CONF="/etc/systemd/system.conf"
SYSCTL_FILE="/etc/sysctl.d/60-hang-recovery.conf"

report() {
    echo
    echo "=== Watchdog device ==="
    if [ -e /dev/watchdog ]; then
        echo "/dev/watchdog present"
        local wd
        for wd in /sys/class/watchdog/watchdog*; do
            [ -e "$wd" ] || continue
            local id="(unknown)" to="(unknown)"
            [ -r "$wd/identity" ] && id="$(cat "$wd/identity")"
            [ -r "$wd/timeout" ]  && to="$(cat "$wd/timeout")s"
            echo "  $(basename "$wd"): identity=$id timeout=$to"
        done
    else
        echo "/dev/watchdog NOT present (no watchdog driver loaded yet)"
    fi

    echo
    echo "=== systemd RuntimeWatchdog ==="
    systemctl show -p RuntimeWatchdogUSec 2>/dev/null || true

    echo
    echo "=== Reboot-on-panic sysctls ==="
    local key
    for key in kernel.panic kernel.panic_on_oops; do
        printf '  %s = %s\n' "$key" "$(sysctl -n "$key" 2>/dev/null || echo '?')"
    done

    echo
    echo "=== Loaded watchdog modules ==="
    lsmod | grep -E "sp5100_tco|iTCO_wdt|softdog" || echo "  (none loaded)"
}

if [ "${1:-}" = "--check" ]; then
    report
    exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "This script must run as root to load modules and edit systemd config." >&2
    echo "Re-run as: sudo bash maintenance/setup_watchdog.sh" >&2
    exit 1
fi

echo "--- Before ---"
report

# 1. Pick a watchdog driver. Prefer the chipset hardware watchdog; it resets the
#    box via independent silicon even on a true hard lockup. softdog is a last
#    resort because it depends on the kernel still running.
echo
echo "--- Selecting watchdog driver ---"
WD_MODULE=""
if grep -q "AuthenticAMD" /proc/cpuinfo; then
    if modprobe sp5100_tco 2>/dev/null && [ -e /dev/watchdog ]; then
        WD_MODULE="sp5100_tco"
        echo "Loaded hardware watchdog: sp5100_tco"
    fi
fi
if [ -z "$WD_MODULE" ] && grep -q "GenuineIntel" /proc/cpuinfo; then
    if modprobe iTCO_wdt 2>/dev/null && [ -e /dev/watchdog ]; then
        WD_MODULE="iTCO_wdt"
        echo "Loaded hardware watchdog: iTCO_wdt"
    fi
fi
if [ -z "$WD_MODULE" ]; then
    echo "WARNING: no hardware watchdog available; falling back to softdog."
    echo "         softdog relies on the kernel still running, so it may NOT catch"
    echo "         a true hard lockup (the 2026-06-07 freeze looked like one)."
    echo "         Check whether the BIOS exposes a watchdog/WDT setting, and that"
    echo "         sp5100_tco didn't log 'Watchdog hardware is disabled' (dmesg)."
    modprobe softdog 2>/dev/null || true
    WD_MODULE="softdog"
fi

if [ ! -e /dev/watchdog ]; then
    echo "ERROR: still no /dev/watchdog after loading $WD_MODULE. Cannot continue." >&2
    exit 1
fi

# Persist the module so it loads on every boot.
echo
echo "--- Persisting module load ($MODULES_FILE) ---"
echo "$WD_MODULE" > "$MODULES_FILE"
echo "Wrote $MODULES_FILE: $WD_MODULE"

# 2. Tell systemd to arm and pet the watchdog.
echo
echo "--- Configuring systemd RuntimeWatchdogSec=$WATCHDOG_TIMEOUT ($SYSTEMD_CONF) ---"
if grep -qE '^[[:space:]]*#?[[:space:]]*RuntimeWatchdogSec=' "$SYSTEMD_CONF"; then
    sed -i -E "s|^[[:space:]]*#?[[:space:]]*RuntimeWatchdogSec=.*|RuntimeWatchdogSec=$WATCHDOG_TIMEOUT|" "$SYSTEMD_CONF"
else
    sed -i "/^\[Manager\]/a RuntimeWatchdogSec=$WATCHDOG_TIMEOUT" "$SYSTEMD_CONF"
fi
echo "Set RuntimeWatchdogSec=$WATCHDOG_TIMEOUT"
echo "Applying via 'systemctl daemon-reexec' (arms the watchdog now, no reboot needed)..."
systemctl daemon-reexec

# 3. Reboot-on-panic sysctls.
echo
echo "--- Configuring reboot-on-panic sysctls ($SYSCTL_FILE) ---"
cat > "$SYSCTL_FILE" <<'EOF'
# Auto-reboot on kernel panic/oops so the box recovers instead of halting.
# Installed by maintenance/setup_watchdog.sh — see docs/runbook.md §7e.
kernel.panic = 10
kernel.panic_on_oops = 1

# Optional: also reboot if any task is stuck uninterruptible (D state) > timeout.
# This directly targets the suspected disk-I/O-stall freeze (procs_blocked climbing),
# but it can false-fire during legitimate very-heavy I/O. Enable by uncommenting,
# then watch syshealth.log for spurious reboots before trusting it.
#kernel.hung_task_panic = 1
#kernel.hung_task_timeout_secs = 120
EOF
sysctl -p "$SYSCTL_FILE"
echo "Wrote and applied $SYSCTL_FILE"

echo
echo "--- After ---"
report

cat <<EOF

Watchdog configured. The box will now:
  - reset within ~$WATCHDOG_TIMEOUT if systemd (PID 1) stops responding (hardware watchdog), and
  - reboot ~10s after a kernel panic/oops (sysctls).

IMPORTANT — verify it actually fires before trusting it. See docs/runbook.md §7e
for the safe test (induce a panic/hang in a maintenance window and confirm the box
reboots itself). Also pair this with the health logger so the next freeze leaves a
trail:  sudo bash maintenance/syshealth_logger.sh --install

Re-check state any time:
  bash maintenance/setup_watchdog.sh --check
EOF
