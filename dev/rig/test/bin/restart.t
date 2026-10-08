Jobs that fail, and start again. support/ctl.exe counts its attempts.

  $ . support/env.sh
  $ host a 127.0.0.1
  $ host b 127.0.0.1
  $ host c ::1

A program killed once its job started fails the job. Its agents see their
connections close, and the cause rig run names is the death no process
reported. The job starts again, with fresh agents, and ends in order.

  $ rig run --on a,b,c -- ./support/ctl.exe kill-first
  attempt 1
  rig: job failed: ./support/ctl.exe was killed by SIGKILL
  rig: restarting the job (1 of 3)
  attempt 2
  copied 2

A job that fails four times in a row with one cause is not started again.

  $ rm attempts
  $ rig run --on a,b -- ./support/ctl.exe kill
  attempt 1
  rig: job failed: ./support/ctl.exe was killed by SIGKILL
  rig: restarting the job (1 of 3)
  attempt 2
  rig: job failed: ./support/ctl.exe was killed by SIGKILL
  rig: restarting the job (2 of 3)
  attempt 3
  rig: job failed: ./support/ctl.exe was killed by SIGKILL
  rig: restarting the job (3 of 3)
  attempt 4
  rig: job failed: ./support/ctl.exe was killed by SIGKILL
  rig: the job failed 4 times in a row with this cause; giving up
  [123]

The count restarts at 1 when the cause changes. Here the program dies at
its first attempt, and at its second its job does not start: it holds a
key that is not the job's. The third ends in order.

  $ rm attempts
  $ cat >spoiled <<'EOS'
  > #!/bin/sh
  > if [ "$(cat attempts 2>/dev/null)" = 1 ]; then
  >   export RIG_REMOTE_KEY=0000000000000000000000000000000000000000000000000000000000000000
  > fi
  > exec ./support/ctl.exe kill-first
  > EOS
  $ chmod +x spoiled
  $ rig run --on a,b -- ./spoiled
  attempt 1
  rig: job failed: ./spoiled was killed by SIGKILL
  rig: restarting the job (1 of 3)
  attempt 2
  ctl.exe: b: the dialing end does not know the job's key
  rig: job failed: b: the dialing end does not know the job's key
  rig: restarting the job (1 of 3)
  attempt 3
  copied 1

A machine whose agent fails at every restart fails each attempt with its
reason, after the machine's name, until rig run gives up. Here b's agent
is killed twice, then b's lock becomes a directory before the third
start: a new cause, counted from 1.

  $ rm attempts
  $ agent_of() { pgrep -P $(cat machines/$1/pid); }
  $ rig run --on a,b -- ./support/ctl.exe wait >out 2>err &
  $ run=$!
  $ ./support/await out joined
  $ kill -KILL $(agent_of b)
  $ ./support/await out "attempt 2"
  $ ./support/await out joined
  $ rm machines/b/rig-agent-$(id -u).lock
  $ mkdir machines/b/rig-agent-$(id -u).lock
  $ kill -KILL $(agent_of b)
  $ wait $run
  [123]
  $ cat out
  attempt 1
  joined
  attempt 2
  joined
  $ sed -e "s|$PWD|PWD|" -e "s|-$(id -u)[.]|-UID.|" err
  Fatal error: exception Failure("the job failed")
  rig: job failed: the agent on b was killed by SIGKILL
  rig: restarting the job (1 of 3)
  Fatal error: exception Failure("the job failed")
  rig: job failed: the agent on b was killed by SIGKILL
  rig: restarting the job (2 of 3)
  rig: job failed: b: PWD/machines/b/rig-agent-UID.lock is no regular file
  rig: restarting the job (1 of 3)
  rig: job failed: b: PWD/machines/b/rig-agent-UID.lock is no regular file
  rig: restarting the job (2 of 3)
  rig: job failed: b: PWD/machines/b/rig-agent-UID.lock is no regular file
  rig: restarting the job (3 of 3)
  rig: job failed: b: PWD/machines/b/rig-agent-UID.lock is no regular file
  rig: the job failed 4 times in a row with this cause; giving up
