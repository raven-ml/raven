#!/usr/bin/env bash
# memcap.sh COMMAND...: runs COMMAND with its memory capped. A test that
# leaks or blows up is killed, and dune names it; without the cap the runner
# runs out of memory and dies without a report.
#
# The cap is on resident memory. A cap on address space (ulimit -v) would
# break AddressSanitizer, whose shadow reserves terabytes, and rig's NV over
# PCI, which reserves a range of about 1 TiB.
#
# Linux: COMMAND runs in a cgroup whose processes together may hold 12 GiB,
# swap excluded; past that the kernel kills the largest of them. The
# heaviest tests, nx's test_bit and rig's test_submit, peak at 2.8 GB
# resident, test_bit at 3.3 GB under AddressSanitizer's allocator. dune runs
# four at a time, and four of the heaviest hold about 9 GB together. The
# cap leaves 3 GB of the runner's 16 GB to the runner and the system.
#
# macOS: the kernel enforces no limit on resident memory, so a watchdog
# kills any process of COMMAND whose resident set passes 4 GiB, checking
# every second. The cap is per process: the runner has 7 GB, which three
# concurrent heavy tests can pass on their own.
#
# Windows: a cap needs a job object, which no tool on the runner sets up.
# COMMAND runs uncapped.

set -eu

linux_cap_gib=12
macos_cap_gib=4

case "$(uname -s)" in
  Linux)
    cg=/sys/fs/cgroup/memcap
    sudo mkdir -p "$cg"
    echo $((linux_cap_gib << 30)) | sudo tee "$cg/memory.max" >/dev/null
    echo 0 | sudo tee "$cg/memory.swap.max" >/dev/null
    echo $$ | sudo tee "$cg/cgroup.procs" >/dev/null

    kills() { awk '$1 == "oom_kill" { print $2 }' "$cg/memory.events"; }
    before=$(kills)
    status=0
    "$@" || status=$?
    killed=$(($(kills) - before))
    if [ "$killed" -gt 0 ]; then
      echo "memcap: killed $killed process(es) past $linux_cap_gib GiB" >&2
    fi
    exit "$status"
    ;;

  Darwin)
    cap_kib=$((macos_cap_gib << 20))

    # Prints the pid, resident KiB and name of each descendant of $1 whose
    # resident set passes $2 KiB.
    over() {
      ps -A -o pid=,ppid=,rss=,comm= | awk -v root="$1" -v cap="$2" '
        { parent[$1] = $2; rss[$1] = $3; name[$1] = $4 }
        END {
          below[root] = 1
          do {
            grew = 0
            for (p in parent)
              if (!(p in below) && (parent[p] in below)) { below[p] = 1; grew = 1 }
          } while (grew)
          for (p in below)
            if (p != root && rss[p] > cap) print p, rss[p], name[p]
        }'
    }

    watch() {
      while sleep 1; do
        over $$ "$cap_kib" | while read -r pid kib name; do
          echo "memcap: killing $name ($pid) at $((kib >> 10)) MiB, past $macos_cap_gib GiB" >&2
          kill -KILL "$pid" 2>/dev/null || true
        done
      done
    }

    watch &
    watcher=$!
    status=0
    "$@" || status=$?
    kill "$watcher"
    exit "$status"
    ;;

  *)
    exec "$@"
    ;;
esac
