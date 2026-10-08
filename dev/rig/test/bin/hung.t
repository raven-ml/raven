A machine's session that does not end once rig run closed its input, as
when ssh hangs on a network that went away. rig run kills that ssh after
its bound, and waits for the machine as for one that does not answer.
Here b's half is stopped while the job runs, and c's agent killed.

  $ . support/env.sh
  $ host a 127.0.0.1
  $ host b 127.0.0.1
  $ host c ::1
  $ agent_of() { pgrep -P $(cat machines/$1/pid); }

  $ rig run --on a,b,c -- ./support/ctl.exe wait-first >out 2>err &
  $ run=$!
  $ ./support/await out joined
  $ kill -STOP $(cat machines/b/pid)
  $ kill -KILL $(agent_of c)
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
