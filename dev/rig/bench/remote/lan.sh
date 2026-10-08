#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Times a link between two machines (lan.ml says what): lan.exe's agent runs
# on AGENT and its controller on CONTROLLER, each over ssh from this machine.
# It lives outside the bench suite because a suite runs on one machine, and
# this needs two.
#
# TREE is a checkout, at the same path on both machines, in which
# dev/rig/bench/remote/lan.exe is built in the release profile. The agent
# and the controller run under a 300 s timeout, the agent besides its own
# idle rule. The script fails if a lan.exe outlives the run on either
# machine.
#
# usage: lan.sh AGENT CONTROLLER TREE [PORT]

set -eu

[ $# -ge 3 ] || {
  echo "usage: lan.sh AGENT CONTROLLER TREE [PORT]" >&2
  exit 2
}
agent=$1
controller=$2
exe=$3/_build/default/dev/rig/bench/remote/lan.exe
port=${4:-47300}
key=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')

printf '%s\n' "$key" | ssh "$agent" "timeout 300 $exe agent $port" &
ssh_pid=$!
status=0
printf '%s\n' "$key" |
  ssh "$controller" "timeout 300 $exe controller $agent $port" || status=$?
wait "$ssh_pid" || status=$?

# The brackets keep pgrep from matching the shell that runs it.
left=$(ssh "$controller" 'pgrep -f "lan[.]exe controller" || true')
far=$(ssh "$agent" 'pgrep -f "lan[.]exe agent" || true')
if [ -n "$left$far" ]; then
  echo "lan.sh: lan.exe left running: on $controller [$left]," \
    "on $agent [$far]" >&2
  exit 1
fi
exit "$status"
