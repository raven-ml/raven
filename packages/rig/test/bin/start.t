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

Each machine's rig must be this one's: its version, and the version of
the protocol between a job's processes, which an agent names first. A
machine that runs another fails the start. machines/M/bin comes first on
M's PATH, so M runs the rig there.

  $ session=$(rig agent 127.0.0.1:0 </dev/null | sed -n 's/^rig-agent //p')
  $ echo "$session" | sed -E "s/^$(rig --version) protocol [0-9]+\$/VERSION protocol N/"
  VERSION protocol N
  $ mkdir machines/c/bin
  $ cat >machines/c/bin/rig <<'EOS'
  > #!/bin/sh
  > echo rig-agent 0.9
  > exec cat >/dev/null
  > EOS
  $ chmod +x machines/c/bin/rig
  $ rig run --on a,c -- true 2>err
  [123]
  $ sed "s/ $session\$/ SESSION/" err
  rig: c: runs rig 0.9; this is rig SESSION

Builds of one version that speak other protocols fail the start too.

  $ cat >machines/c/bin/rig <<EOS
  > #!/bin/sh
  > echo "rig-agent $(rig --version) protocol 0"
  > exec cat >/dev/null
  > EOS
  $ rig run --on a,c -- true 2>err
  [123]
  $ sed -e "s/ $session\$/ SESSION/" -e "s/ $(rig --version) / VERSION /" err
  rig: c: runs rig VERSION protocol 0; this is rig SESSION

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

rig run passes each --firmware to every agent, in order, each directory
one argument whatever its characters. Here c's rig records its
arguments.

  $ cat >machines/c/bin/rig <<EOS
  > #!/bin/sh
  > printf '%s\n' "\$@" >$PWD/args
  > exec $(command -v rig) "\$@"
  > EOS
  $ rm attempts
  $ rig run --firmware 'd i r' --firmware="it's \$HOME \"x\"" --on a,c -- ./support/ctl.exe copy
  attempt 1
  copied 1
  $ cat args
  agent
  --firmware
  d i r
  --firmware
  it's $HOME "x"
  127.0.0.1:0

After the greeting, a line that is none of an agent's breaks the session,
quoted.

  $ cat >machines/c/bin/rig <<EOS
  > #!/bin/sh
  > echo "rig-agent $session"
  > echo "Welcome to c"
  > exec cat >/dev/null
  > EOS
  $ rig run --on a,c -- true
  rig: c: wrote "Welcome to c"
  [123]

A machine whose login shell does not find rig fails the start, saying
so. Here y's ~/.profile leaves rig off its PATH; the shell's own message
differs between systems.

  $ host y 127.0.0.1
  $ echo 'PATH=/usr/bin:/bin' >machines/y/.profile
  $ rig run --on a,y -- true 2>err
  [123]
  $ grep '^rig' err
  rig: y: rig is not on y's PATH; add it in ~/.profile

--rig names the rig each other machine runs: a name the login shell
finds on its PATH, or a path. Here y runs this rig, off its PATH.

  $ rm attempts
  $ rig run --rig "$(command -v rig)" --on a,y -- ./support/ctl.exe copy
  attempt 1
  copied 1

A machine where --rig names nothing it can run fails the start, saying
so. Shells exit 126 or 127 for a path they cannot run.

  $ rig run --rig rig-old --on a,y -- true 2>err
  [123]
  $ grep '^rig' err
  rig: y: rig-old is not on y's PATH; add it in ~/.profile
  $ rig run --rig /nowhere/rig --on a,y -- true 2>err
  [123]
  $ grep '^rig' err
  rig: y: its shell cannot run /nowhere/rig

A machine ssh cannot read the configuration of fails the start, after
ssh's message.

  $ host z 127.0.0.1
  $ touch machines/z/unconfigured
  $ rig run --on a,z -- true
  /etc/ssh/ssh_config: line 1: Bad configuration option: z
  /etc/ssh/ssh_config: terminating, 1 bad configuration options
  rig: z: ssh -G exited with status 255
  [123]

A supervisor that blocks SIGCHLD does not keep rig run from seeing its
children end. One session closes its output, then waits before exiting.
The other writes two lines in separate polls: once rig has forwarded the
second, it has read the first session's EOFs. Only that session's exit can
then wake rig.

  $ host closed 127.0.0.1
  $ host held 127.0.0.1
  $ mkdir machines/closed/bin machines/held/bin
  $ mkfifo closed_exit held_step
  $ cat >machines/closed/bin/rig <<'EOS'
  > #!/bin/sh
  > read key
  > exec >/dev/null 2>&1
  > echo ready >closed_ready
  > read finish <closed_exit
  > exit 255
  > EOS
  $ cat >machines/held/bin/rig <<'EOS'
  > #!/bin/sh
  > read key
  > read step <held_step
  > echo first >&2
  > read step <held_step
  > echo second >&2
  > read finish
  > EOS
  $ chmod +x machines/closed/bin/rig machines/held/bin/rig
  $ ./support/sig.exe block CHLD rig run --on a,closed,held -- true >out 2>err &
  $ run=$!
  $ ./support/await closed_ready ready
  $ echo step >held_step
  $ ./support/await err 'held: first'
  $ echo step >held_step
  $ ./support/await err 'held: second'
  $ echo exit >closed_exit
  $ wait $run
  [123]
  $ cat err
  held: first
  held: second
  rig: closed: ssh exited with status 255 before its agent listened
  $ kill -0 $(cat machines/held/pid) 2>/dev/null
  [1]
