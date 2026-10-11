#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Times a job between this machine and AGENT (lan.ml says what): the floors'
# echo runs on AGENT over ssh, and rig run starts the controller here and an
# agent there. TREE is a checkout at the same path on both machines. lan.sh
# builds lan.exe and rig in it on both, and both machines run that rig.
# AGENT's login PATH must find dune.
#
# usage: lan.sh AGENT TREE

set -eu
agent=$1
tree=$2
lan=packages/rig/bench/remote/lan.exe
rig=packages/rig/bin/main.exe

# Timings come from release builds. A dune server running in the tree would
# take the build and keep its own profile, so its notice stops the run.
release() {
  log=$("$@" 2>&1) || { printf '%s\n' "$log" >&2; exit 1; }
  case $log in *'forwarded to a running Dune instance'*)
    echo "lan.sh: $tree needs a release build, in a tree without a running dune" >&2
    exit 1
  esac
}
release sh -c "cd '$tree' && dune build --profile release ./$lan ./$rig"
release ssh "$agent" "exec \"\$SHELL\" -lc 'cd $tree && dune build --profile release ./$lan ./$rig'" </dev/null

ssh "$agent" "timeout 300 $tree/_build/default/$lan floors 47301" </dev/null &
timeout 300 "$tree/_build/default/$rig" run --rig "$tree/_build/default/$rig" \
  --on "$(hostname),$agent" -- "$tree/_build/default/$lan" controller "$agent" 47301
wait $!
