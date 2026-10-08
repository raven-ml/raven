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
