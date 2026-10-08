The first start of a job, through the ssh of support/ssh. Every machine
of these tests is this one: machine M is the directory machines/M, and
machines/M/hostname, its HostName for ssh -G, an address of this machine.
While machines/M/down exists, M does not answer.

These starts fail before the program runs: rig run says where, ends what
it started, and exits 123.

  $ . support/env.sh
  $ host a 127.0.0.1
  $ host b 127.0.0.1
  $ host c 127.0.0.1

The first machine is this one. A first machine whose HostName is not an
address of this machine is misuse.

  $ host far 192.0.2.1
  $ rig run --on far,b -- true
  rig: --on: 'far' is not this machine; name this machine first
  Try 'rig run --help'.
  [124]
  $ test -e machines/b/pid
  [1]

A machine whose HostName does not resolve here fails the start before any
ssh session.

  $ host x nowhere.invalid
  $ rig run --on a,x -- true
  rig: x: nowhere.invalid does not resolve here
  [123]
  $ test -e machines/x/pid
  [1]

A machine that does not answer ssh fails the start. ssh's message comes
out after the machine's name.

  $ touch machines/b/down
  $ rig run --on a,b -- true
  b: ssh: connect to host b port 22: Connection refused
  rig: b: ssh exited with status 255 before its agent listened
  [123]

An IPv6 machine is written in brackets. ssh gets the address bare; rig
run's lines name the machine as --on wrote it.

  $ mkdir -p machines/::1
  $ touch machines/::1/down
  $ rig run --on 'a,[::1]' -- true
  [::1]: ssh: connect to host ::1 port 22: Connection refused
  rig: [::1]: ssh exited with status 255 before its agent listened
  [123]

The agents of the other machines, started before one failed, are gone
when rig run exits.

  $ rig run --on a,c,b -- true
  b: ssh: connect to host b port 22: Connection refused
  rig: b: ssh exited with status 255 before its agent listened
  [123]
  $ kill -0 $(cat machines/c/pid) 2>/dev/null
  [1]
