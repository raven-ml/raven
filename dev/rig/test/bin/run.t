Jobs that end in order, on three machines: this one, a, and b and c,
played by this machine through support/ssh. Each test starts from no
attempt.

  $ . support/env.sh
  $ host a 127.0.0.1
  $ host b 127.0.0.1
  $ host c ::1
  $ fresh() { rm -f attempts; }
  $ gone() { for m in b c; do kill -0 $(cat machines/$m/pid) 2>/dev/null && echo "$m's agent runs"; done; true; }

The program uses every machine, closes the job, and rig run exits with its
status, writing nothing. Every agent is gone once rig run exits.

  $ rig run --on a,b,c -- ./support/ctl.exe copy
  attempt 1
  copied 2
  $ gone

A program that exits once its job started closes the job: rig run exits
with its status.

  $ fresh
  $ rig run --on a,b,c -- ./support/ctl.exe exit 3
  attempt 1
  [3]
  $ gone

So does a program that raises: a bug ends the job.

  $ fresh
  $ rig run --on a,b,c -- ./support/ctl.exe raise
  attempt 1
  Fatal error: exception Failure("ctl raised")
  [2]
  $ gone

The program finds the agents in --on order and the key in its
environment, and once its job started, the variables are gone from it.

  $ fresh
  $ rig run --on a,b,c -- ./support/ctl.exe env
  attempt 1
  agents b c, key of 64 characters
  variables gone: true

A program that exits before starting its job does so on its own: rig run
says so and exits with its status, without starting it again.

  $ fresh
  $ rig run --on a,b,c -- ./support/ctl.exe early 124
  attempt 1
  rig: ./support/ctl.exe exited with status 124 before starting the job
  [124]
  $ gone

  $ rig run --on a,b -- true
  rig: true exited with status 0 before starting the job

A program killed before starting its job exits as a shell reports it,
128 + N for signal N.

  $ rig run --on a,b,c -- sh -c 'kill -KILL $$'
  rig: sh was killed by SIGKILL before starting the job
  [137]
  $ gone

A program that cannot be run fails the start.

  $ rig run --on a,b -- ./nowhere
  rig: ./nowhere: No such file or directory
  [123]

A program's child is no process of the job: its exit neither ends nor
fails the job.

  $ fresh
  $ rig run --on a,b,c -- ./support/ctl.exe fork
  attempt 1
  copied 2
  $ gone

A program that the program runs is not launched: it finds none of the
variables.

  $ fresh
  $ rig run --on a,b,c -- ./support/ctl.exe exec 'env | grep -c ^RIG_REMOTE_'
  attempt 1
  0
  copied 2
  $ gone
