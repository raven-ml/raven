A system that refuses rig run a socket on the loopback, as a sandbox can:
rig run says its agent cannot listen there, with the system's reason, and
exits 123. Nothing is started.

  $ . support/env.sh
  $ host a 127.0.0.1
  $ host b 127.0.0.1
  $ sandbox-exec -p '(version 1)(allow default)(deny network-bind)' \
  >   rig run --on a,b -- true
  rig: a: the agent cannot listen here: binding 127.0.0.1: Operation not permitted
  [123]
  $ ls machines/a
  hostname
