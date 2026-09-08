#!/usr/bin/env bash

set -uo pipefail

print_section() {
  printf '\n%s\n' "$1"
}

printf '%s\n' \
  "================================" \
  " BIG DATA CLUSTER - NODE INFO" \
  "================================"

print_section "Hostname"
hostname

print_section "Operating system"
if [[ -r /etc/os-release ]]; then
  source /etc/os-release
  printf '%s\n' "${PRETTY_NAME:-Unknown}"
else
  printf '%s\n' "Unknown"
fi

print_section "CPU"
lscpu | grep -E 'Model name|CPU\(s\)|Core\(s\)|Thread'

print_section "Memory"
free -h

print_section "Root filesystem"
df -h /

print_section "Java"
if command -v java >/dev/null 2>&1; then
  java -version 2>&1 | head -n 1
else
  printf '%s\n' "Not installed"
fi

print_section "Python"
if command -v python3 >/dev/null 2>&1; then
  python3 --version
else
  printf '%s\n' "Not installed"
fi

print_section "SSH service"
if systemctl is-active --quiet ssh 2>/dev/null; then
  printf '%s\n' "active (ssh)"
elif systemctl is-active --quiet sshd 2>/dev/null; then
  printf '%s\n' "active (sshd)"
else
  printf '%s\n' "inactive or unavailable"
fi