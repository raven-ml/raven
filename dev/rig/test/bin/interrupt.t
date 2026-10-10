rig run, told to stop, ends the job and dies by the signal it got, so
that a shell sees it stopped. rig run keeps a signal that was ignored when
it started ignored, as a background job's SIGINT, so these tests start it
through support/defaults.exe, with the three signals at their defaults, as
from a terminal. Each sends the signal once the program joined its job.

  $ . support/env.sh
  $ host a 127.0.0.1
  $ host b 127.0.0.1
  $ host c ::1
  $ agent_of() { cat machines/$1/rig-agent-$(id -u).lock; }
  $ stop() {
  >   rm -f attempts
  >   ./support/defaults.exe rig run --on a,b -- ./support/ctl.exe wait >out 2>err &
  >   run=$!
  >   ./support/await out joined
  >   kill -$1 $run
  >   wait $run 2>/dev/null
  > }

Control-C sends SIGINT.

  $ stop INT
  [130]
  $ cat err
  rig: interrupted; ending the job
  $ kill -0 $(cat machines/b/pid) 2>/dev/null
  [1]

A closed terminal, or a lost ssh session to the first machine, sends
SIGHUP.

  $ stop HUP
  [129]
  $ cat err
  rig: interrupted; ending the job
  $ kill -0 $(cat machines/b/pid) 2>/dev/null
  [1]

  $ stop TERM
  [143]
  $ cat err
  rig: interrupted; ending the job
  $ kill -0 $(cat machines/b/pid) 2>/dev/null
  [1]

rig run killed by SIGKILL ends nothing itself. Each machine's session
sees its input end, and the machine ends its agent; the program loses its
agents, and its job fails.

  $ rm attempts
  $ rig run --on a,b,c -- sh -c './support/ctl.exe wait; echo "the program exited $?" >status' >out 2>err &
  $ run=$!
  $ ./support/await out joined
  $ machines="$(cat machines/b/pid) $(agent_of b) $(cat machines/c/pid) $(agent_of c)"
  $ kill -KILL $run
  $ wait $run 2>/dev/null
  [137]
  $ ./support/await status "the program"
  $ cat status
  the program exited 2
  $ cat err
  Fatal error: exception Failure("the job failed")
  $ ./support/gone $machines

The program gets the signal first, and its agents serve it while it ends:
time to save its work.

  $ rm -f attempts out
  $ rig run --on a,b -- ./support/ctl.exe save >out 2>err &
  $ run=$!
  $ ./support/await out joined
  $ kill -TERM $run
  $ wait $run 2>/dev/null
  [143]
  $ cat out
  attempt 1
  joined
  copied 1
  $ cat err
  rig: interrupted; ending the job
