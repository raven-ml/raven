#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Runs test_rxe.exe on a Soft-RoCE device: loads rdma_rxe if it is not loaded,
# adds the device rxe_rig over NETDEV, runs the suite, checks that the queue
# pairs the suite's killed child made are gone, then deletes the device and
# unloads the module if this script loaded it. It lives outside runtest
# because it needs root to make the device; the suite itself skips without
# one.
#
# The suite runs as the user who ran sudo, if any, under a locked-memory limit
# of 1 MiB, so that registration meets the limit an unprivileged process has:
# root's CAP_IPC_LOCK ignores it.
#
# TREE is a checkout in which dev/rig/test/mlx5/uverbs/test_rxe.exe is built.
#
# usage: sudo rxe.sh NETDEV TREE

set -eu

[ $# -eq 2 ] || {
  echo "usage: sudo rxe.sh NETDEV TREE" >&2
  exit 2
}
netdev=$1
dir=$2/_build/default/dev/rig/test/mlx5/uverbs
link=rxe_rig

[ "$(id -u)" -eq 0 ] || {
  echo "rxe.sh: run as root" >&2
  exit 2
}
[ -x "$dir/test_rxe.exe" ] || {
  echo "rxe.sh: $dir/test_rxe.exe is not built" >&2
  exit 2
}

loaded=0
if ! grep -q '^rdma_rxe ' /proc/modules; then
  modprobe rdma_rxe
  loaded=1
fi

cleanup() {
  rdma link delete "$link" 2>/dev/null || true
  if [ "$loaded" -eq 1 ]; then modprobe -r rdma_rxe || true; fi
}
trap cleanup EXIT

rdma link add "$link" type rxe netdev "$netdev"

# The device's queue pairs, the kernel's own among them.
qps() {
  rdma resource show qp link "$link/1" | grep -c lqpn || true
}
before=$(qps)

status=0
cd "$dir"
if [ -n "${SUDO_USER:-}" ]; then
  runuser -u "$SUDO_USER" -- sh -c 'ulimit -l 1024 && ./test_rxe.exe' || status=$?
else
  sh -c 'ulimit -l 1024 && ./test_rxe.exe' || status=$?
fi

after=$(qps)
if [ "$after" -ne "$before" ]; then
  echo "rxe.sh: $link holds $after queue pairs after the suite, $before before" >&2
  status=1
fi
exit "$status"
