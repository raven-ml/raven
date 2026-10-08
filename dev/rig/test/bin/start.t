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

An agent that fails before it listens fails the start, its reason after
the machine's name. Here d's lock is a directory.

  $ host d 127.0.0.1
  $ mkdir machines/d/rig-agent-$(id -u).lock
  $ rig run --on a,c,d -- true 2>err
  [123]
  $ sed -e "s|$PWD|PWD|" -e "s|-$(id -u)[.]|-UID.|" err
  rig: d: PWD/machines/d/rig-agent-UID.lock is no regular file
  $ kill -0 $(cat machines/c/pid) 2>/dev/null
  [1]

A program whose job does not start fails the start: rig run names the
reason the program gave. Here the program holds a key that is not the
job's.

  $ zeros=0000000000000000000000000000000000000000000000000000000000000000
  $ rig run --on a,c -- sh -c "RIG_REMOTE_KEY=$zeros exec ./support/ctl.exe copy"
  attempt 1
  ctl.exe: c: the dialing end does not know the job's key
  rig: the job did not start: c: the dialing end does not know the job's key
  [123]
  $ kill -0 $(cat machines/c/pid) 2>/dev/null
  [1]

Each machine's rig must be this one's version. A machine that runs
another fails the start. machines/M/bin comes first on M's PATH, so M
runs the rig there.

  $ mkdir machines/c/bin
  $ cat >machines/c/bin/rig <<'EOS'
  > #!/bin/sh
  > echo rig-agent 0.9
  > exec cat >/dev/null
  > EOS
  $ chmod +x machines/c/bin/rig
  $ rig run --on a,c -- true 2>err
  [123]
  $ sed "s/ $(rig --version)\$/ VERSION/" err
  rig: c: runs rig 0.9; this is rig VERSION

A shell that writes lines before rig starts, as a .bashrc that echoes,
does not disturb the session: rig skips what comes before its greeting.

  $ cat >machines/c/bin/rig <<EOS
  > #!/bin/sh
  > echo "Welcome to c"
  > exec $(command -v rig) "\$@"
  > EOS
  $ rm attempts
  $ rig run --on a,c -- ./support/ctl.exe copy
  attempt 1
  copied 1

After the greeting, a line that is none of an agent's breaks the session,
quoted.

  $ cat >machines/c/bin/rig <<EOS
  > #!/bin/sh
  > echo "rig-agent $(rig --version)"
  > echo "Welcome to c"
  > exec cat >/dev/null
  > EOS
  $ rig run --on a,c -- true
  rig: c: wrote "Welcome to c"
  [123]
