A machine's session that does not end once rig run closed its input, as
when ssh hangs on a network that went away. rig run kills that ssh after
its bound, and waits for the machine as for one that does not answer.
Here b's half is stopped while the job runs, and c's agent killed.
Hold the runner until c's half has reported and exited, so that a delayed
half cannot miss the runner's cleanup bound. Its zombie stays visible
until the runner resumes and reaps it.

  $ . support/env.sh
  $ host a 127.0.0.1
  $ host b 127.0.0.1
  $ host c ::1
  $ agent_of() { cat machines/$1/rig-agent-$(id -u).lock; }
  $ await_state() {
  >   i=0
  >   until ps -o stat= -p "$1" | grep -q "^[[:space:]]*$2"; do
  >     i=$((i + 1))
  >     if [ "$i" -ge 600 ]; then
  >       echo "process $1 did not reach state $2:" >&2
  >       ps -o pid,ppid,stat,command -p "$1" >&2
  >       return 1
  >     fi
  >     sleep 0.1
  >   done
  > }

  $ rig run --on a,b,c -- ./support/ctl.exe wait-first >out 2>err &
  $ run=$!
  $ ./support/await out joined
  $ b=$(cat machines/b/pid); c=$(cat machines/c/pid)
  $ kill -STOP $run $b
  $ await_state $run T
  $ await_state $b T
  $ kill -KILL $(agent_of c)
  $ await_state $c Z
  $ ./support/await err 'Fatal error:'
  $ kill -CONT $run
  $ wait $run
  $ cat out
  attempt 1
  joined
  attempt 2
  copied 2
  $ cat err
  Fatal error: exception Failure("the job failed")
  rig: job failed: the agent on c was killed by SIGKILL
  rig: b does not answer; waiting for it
  rig: b answers; restarting the job (1 of 3)
