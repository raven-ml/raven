#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Times a job between two machines (lan.ml says what): agent.exe and lan.exe's
# echo run on AGENT and lan.exe's controller on CONTROLLER, each over ssh from
# this machine. It lives outside the bench suite because a suite runs on one
# machine, and this needs two.
#
# TREE is a checkout, at the same path on both machines, in which
# dev/rig/bench/remote/agent.exe and lan.exe are built in the release
# profile. Every process runs under a 300 s timeout. The script fails if one
# outlives the run on either machine.
#
# usage: lan.sh AGENT CONTROLLER TREE [PORT]

set -eu

[ $# -ge 3 ] || {
  echo "usage: lan.sh AGENT CONTROLLER TREE [PORT]" >&2
  exit 2
}
agent=$1
controller=$2
dir=$3/_build/default/dev/rig/bench/remote
port=${4:-47300}
key=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')
out=$(mktemp)
trap 'rm -f "$out"' EXIT

ssh "$agent" "timeout 300 $dir/lan.exe floors $((port + 1))" </dev/null &
floors_pid=$!
printf '%s\n' "$key" |
  ssh "$agent" "timeout 300 $dir/agent.exe 0.0.0.0 $port" >"$out" &
agent_pid=$!

# The agent prints its port once it listens.
n=0
until [ -s "$out" ] || [ $n -ge 300 ]; do
  sleep 0.1
  n=$((n + 1))
done

status=0
printf '%s\n' "$key" |
  ssh "$controller" "timeout 300 $dir/lan.exe controller $agent $port" ||
  status=$?
wait "$agent_pid" || status=$?
wait "$floors_pid" || status=$?

# The brackets keep pgrep from matching the shell that runs it.
left=$(ssh "$controller" 'pgrep -f "lan[.]exe controller" || true')
far=$(ssh "$agent" 'pgrep -f "lan[.]exe floors|agent[.]exe" || true')
if [ -n "$left$far" ]; then
  echo "lan.sh: left running: on $controller [$left], on $agent [$far]" >&2
  exit 1
fi
exit "$status"
