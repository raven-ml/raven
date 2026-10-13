A launch whose processes each hold more than 1024 descriptors: every
pipe, socket and child of rig run, its agents and its program has a
number past 1100, where select takes none. support/high.exe holds the
descriptors and passes them on.

  $ . support/env.sh
  $ host a 127.0.0.1
  $ host b 127.0.0.1
  $ ./support/high.exe rig run --on a,b -- ./support/ctl.exe copy
  attempt 1
  copied 1
