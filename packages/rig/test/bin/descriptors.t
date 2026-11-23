A launch whose processes each hold more than 1024 descriptors: every
pipe, socket and child of rig run, its agents and its program has a
number past 1100, where select takes none. support/high.exe holds the
descriptors and passes them on. Where the hard limit of descriptors is
below the 4096 high.exe asks for, the launch is not tried, so the run's
output is compared inside the test.

  $ . support/env.sh
  $ host a 127.0.0.1
  $ host b 127.0.0.1
  $ cat >copied <<EOF
  > attempt 1
  > copied 1
  > exit 0
  > EOF
  $ hard=$(ulimit -H -n)
  $ if [ "$hard" = unlimited ] || [ "$hard" -ge 4096 ]; then
  >   ./support/high.exe rig run --on a,b -- ./support/ctl.exe copy >out 2>&1
  >   echo "exit $?" >>out
  >   diff copied out
  > fi
