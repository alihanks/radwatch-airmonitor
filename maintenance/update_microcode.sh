#!/bin/bash
# update_microcode.sh — apply CPU microcode updates and report SRSO mitigation state.
#
# Context: in 2026-05 the dosenet server began hanging with kernel soft-lockups whose
# stack traces all passed through the AMD SRSO (Speculative Return Stack Overflow)
# return-thunk mitigation. dmesg showed:
#   "Speculative Return Stack Overflow: IBPB-extending microcode not applied!"
# The motherboard BIOS dates to 2017 and never delivered the matching microcode, so
# the kernel fell back to a software mitigation that has known soft-lockup issues on
# smp_call_function / TLB flushes. The proper fix is to update the CPU microcode.
# Linux delivers microcode updates via the amd64-microcode / intel-microcode packages
# loaded at boot through the initramfs, independent of the BIOS.
#
# See docs/runbook.md §7 for the full diagnosis and follow-up workarounds (kernel
# parameter, BIOS update) if this script doesn't resolve the issue.
#
# Usage:
#   sudo bash maintenance/update_microcode.sh           # apply updates
#   bash maintenance/update_microcode.sh --check        # report state only (no root needed for most fields)

set -e

CHECK_ONLY=0
if [ "${1:-}" = "--check" ]; then
    CHECK_ONLY=1
fi

report() {
    echo
    echo "=== CPU & current microcode revision ==="
    grep -m1 "model name" /proc/cpuinfo || true
    grep -m1 "microcode" /proc/cpuinfo || true

    echo
    echo "=== Microcode load messages from this boot ==="
    if [ "$(id -u)" -eq 0 ]; then
        dmesg | grep -iE "microcode" | tail -10 || echo "(no microcode lines in dmesg)"
    else
        echo "(dmesg is root-only on this system; rerun with sudo to see these lines)"
    fi

    echo
    echo "=== SRSO mitigation status ==="
    local srso_status=""
    if [ -r /sys/devices/system/cpu/vulnerabilities/spec_rstack_overflow ]; then
        srso_status="$(cat /sys/devices/system/cpu/vulnerabilities/spec_rstack_overflow)"
        echo "$srso_status"
    else
        echo "(spec_rstack_overflow not exposed; CPU may be unaffected or kernel too old)"
    fi

    echo
    echo "=== Installed microcode packages ==="
    dpkg -l 2>/dev/null | grep -E "amd64-microcode|intel-microcode" || echo "(none installed)"

    # Verdict: interpret the SRSO status in plain English so the operator knows
    # whether to expect a fix from another microcode round, a BIOS update, or
    # the spec_rstack_overflow=off kernel-param workaround.
    if [ -n "$srso_status" ]; then
        echo
        echo "=== Verdict ==="
        case "$srso_status" in
            *"no microcode"*)
                echo "Kernel has fallen back to the Safe RET software mitigation because no"
                echo "IBPB-extending microcode is loaded. For Zen 1/2 (e.g. Ryzen 1xxx/2xxx),"
                echo "AMD never published such microcode -- your microcode revision is already"
                echo "at the ceiling amd64-microcode can deliver, and a BIOS update will not"
                echo "change that either."
                echo
                echo "If the system is hanging with srso_return_thunk in the soft-lockup stack,"
                echo "the practical fix is to disable the software mitigation via kernel"
                echo "parameter. See docs/runbook.md section 7a for the GRUB edit."
                ;;
            *"Mitigation: Safe RET"*|*"Mitigation: IBPB"*)
                echo "SRSO is fully mitigated (software + microcode). No further action needed."
                ;;
            *Vulnerable*)
                echo "CPU is reported vulnerable to SRSO with no mitigation applied. Investigate"
                echo "kernel boot parameters (mitigations=off?) and /etc/default/grub."
                ;;
            *"Not affected"*)
                echo "This CPU is not affected by SRSO; no mitigation needed."
                ;;
        esac
    fi
}

if [ "$CHECK_ONLY" -eq 1 ]; then
    report
    exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "This script must run with sudo to install packages and rebuild initramfs." >&2
    echo "Re-run as: sudo bash maintenance/update_microcode.sh" >&2
    exit 1
fi

# Pick the right microcode package for this CPU vendor.
if grep -q "AuthenticAMD" /proc/cpuinfo; then
    PACKAGES="amd64-microcode"
elif grep -q "GenuineIntel" /proc/cpuinfo; then
    PACKAGES="intel-microcode"
else
    echo "Unknown CPU vendor; bailing out." >&2
    grep -m1 "vendor_id" /proc/cpuinfo >&2 || true
    exit 1
fi

echo "Detected CPU vendor; will install/upgrade: $PACKAGES"

echo "--- Before ---"
report

echo
echo "--- Applying updates ---"
apt update
apt install -y $PACKAGES
update-initramfs -u

echo
echo "--- After (kernel will only see the new microcode after the next reboot) ---"
report

cat <<EOF

Next steps:
  1. Reboot:           sudo reboot
  2. After reboot:     bash maintenance/update_microcode.sh --check
  3. Look for:
       - microcode revision in /proc/cpuinfo higher than before
       - SRSO mitigation line no longer mentions "microcode not applied"
       - dmesg no longer shows "IBPB-extending microcode not applied!"
  4. If symptoms persist, see docs/runbook.md §7 for the spec_rstack_overflow=off
     kernel-parameter workaround and the longer-term BIOS update path.
EOF
