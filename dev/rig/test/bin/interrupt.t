rig run, told to stop, ends the job and dies by the signal it got, so
that a shell sees it stopped. A shell runs a command in the background
with SIGINT ignored, which rig run keeps; the test sends SIGTERM, which
takes the same path. rig run runs under a subshell that waits for it,
whose notice of the signal goes nowhere.

  $ . support/env.sh
  $ host a 127.0.0.1
  $ host b 127.0.0.1

  $ (rig run --on a,b -- ./support/ctl.exe wait-first >out 2>err &
  >  echo $! >rig.pid
  >  wait $!) 2>/dev/null &
  $ run=$!
  $ ./support/await out "attempt 1"
  $ kill -TERM $(cat rig.pid)
  $ wait $run
  [143]
  $ cat err
  rig: interrupted; ending the job
  $ kill -0 $(cat machines/b/pid) 2>/dev/null
  [1]
