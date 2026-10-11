A signal rig run handles wakes it even when it comes just before rig
blocks, with nothing else to wake it. support/gap.exe sends the signal
between the OCaml runtime's last look at its pending signals and the
system call that blocks, then bounds the wait at one second.

A child's end:

  $ ./support/gap.exe chld
  woke: child ended
  killed by SIGKILL

An interrupt:

  $ ./support/gap.exe term
  woke: interrupted
  SIGTERM
