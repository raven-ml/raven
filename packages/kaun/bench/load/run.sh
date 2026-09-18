#!/bin/sh
# Runs every configuration of the load benchmark, one process each, and prints
# a Markdown table row per configuration: the phases' wall times, as printed by
# bench_load.exe, and the process's peak memory.
#
# usage: packages/kaun/bench/load/run.sh [bench_load.exe arguments, e.g. --repo R]
set -eu
root=$(cd "$(dirname "$0")/../../../.." && pwd)
exe="$root/_build/default/packages/kaun/bench/load/bench_load.exe"
(cd "$root" && dune build packages/kaun/bench/load/bench_load.exe)

# [peak out command...] runs the command with its output in [out] and prints
# the process's peak memory: macOS reports the peak memory footprint in bytes,
# GNU time the maximum resident set size in KB.
peak() {
  out=$1
  shift
  if /usr/bin/time -l true 2>/dev/null; then
    /usr/bin/time -l "$@" >"$out" 2>"$log"
    awk '/peak memory footprint/ { printf "%.2f GB", $1 / 1e9 }' "$log"
  else
    /usr/bin/time -v "$@" >"$out" 2>"$log"
    awk -F: '/Maximum resident set size/ { printf "%.2f GB", $2 * 1024 / 1e9 }' "$log"
  fi
}

log=$(mktemp)
trap 'rm -f "$log"' EXIT
echo "| dtype | device | phases | peak memory |"
echo "| --- | --- | --- | --- |"
for dtype in "" float32; do
  for device in "" CPU METAL; do
    [ "$device" = METAL ] && [ "$(uname)" != Darwin ] && continue
    args=""
    [ -n "$dtype" ] && args="$args --dtype $dtype"
    [ -n "$device" ] && args="$args --device $device"
    out=$(mktemp)
    # shellcheck disable=SC2086
    mem=$(peak "$out" "$exe" "$@" $args) || {
      cat "$log" >&2
      exit 1
    }
    echo "| ${dtype:-as stored} | ${device:-none} | $(cat "$out") | $mem |"
    rm -f "$out"
  done
done
