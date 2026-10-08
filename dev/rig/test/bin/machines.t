Machines that fail under a running job. rig run's output goes to err, the
program's to out; each test waits for the program to join its job before it
breaks a machine.

  $ . support/env.sh
  $ host a 127.0.0.1
  $ host b 127.0.0.1
  $ host c ::1
  $ agent_of() { pgrep -P $(cat machines/$1/pid); }

An agent killed: its machine's half says so, and that death is the cause,
whatever the program saw of it first.

  $ rig run --on a,b,c -- ./support/ctl.exe wait-first >out 2>err &
  $ run=$!
  $ ./support/await out joined
  $ kill -KILL $(agent_of b)
  $ wait $run
  $ cat out
  attempt 1
  joined
  attempt 2
  copied 2
  $ cat err
  Fatal error: exception Failure("the job failed")
  rig: job failed: the agent on b was killed by SIGKILL
  rig: restarting the job (1 of 3)

A machine's session lost, here its half killed. The agent ends with its
half, and the job learns of it through its connections. The agent's
machine does not answer until rig run reaches it again, and its new agent
finds no other agent there.

  $ rm attempts
  $ rig run --on a,b,c -- ./support/ctl.exe wait-first >out 2>err &
  $ run=$!
  $ ./support/await out joined
  $ kill -KILL $(cat machines/b/pid)
  $ wait $run
  $ cat out
  attempt 1
  joined
  attempt 2
  copied 2
  $ cat err
  Fatal error: exception Failure("the job failed")
  rig: job failed: b: closed its connection
  rig: b does not answer; waiting for it
  rig: b answers; restarting the job (1 of 3)

The same, with the old agent stopped instead: the job learns of it from
its silence, and b's new agent waits until the old one ends.

  $ rm attempts
  $ rig run --on a,b,c -- ./support/ctl.exe wait-first >out 2>err &
  $ run=$!
  $ ./support/await out joined
  $ old=$(agent_of b)
  $ kill -STOP $old
  $ kill -KILL $(cat machines/b/pid)
  $ ./support/await err "rig: b runs another agent"
  $ kill -KILL $old
  $ wait $run
  $ cat out
  attempt 1
  joined
  attempt 2
  copied 2
  $ cat err
  Fatal error: exception Failure("the job failed")
  rig: job failed: b: silent for 10 s
  rig: b does not answer; waiting for it
  rig: b runs another agent of this user; waiting for it to end
  rig: b answers; restarting the job (1 of 3)

A machine that stays down is tried again every 5 seconds until it
answers; only its first try's errors come out.

  $ rm attempts
  $ rig run --on a,b,c -- ./support/ctl.exe wait-first >out 2>err &
  $ run=$!
  $ ./support/await out joined
  $ touch machines/b/down
  $ kill -KILL $(agent_of b)
  $ ./support/await err "rig: b does not answer"
  $ rm machines/b/down
  $ wait $run
  $ cat out
  attempt 1
  joined
  attempt 2
  copied 2
  $ cat err
  Fatal error: exception Failure("the job failed")
  rig: job failed: the agent on b was killed by SIGKILL
  b: ssh: connect to host b port 22: Connection refused
  rig: b does not answer; waiting for it
  rig: b answers; restarting the job (1 of 3)

What an agent or ssh writes on its standard error comes out after its
machine's name.

  $ rm attempts
  $ touch machines/c/down
  $ rig run --on a,b,c -- ./support/ctl.exe copy
  c: ssh: connect to host c port 22: Connection refused
  rig: c: ssh exited with status 255 before its agent listened
  [123]
  $ rm machines/c/down
