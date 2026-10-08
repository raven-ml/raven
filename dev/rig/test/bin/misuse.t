A command line rig cannot read names what is wrong, quoting what was
typed, points at the page that says how to write it, and exits 124.
Nothing is started.

  $ . support/env.sh
  $ rig
  rig: no command; the commands are run and agent
  Try 'rig --help'.
  [124]

  $ rig launch
  rig: unknown command 'launch'; the commands are run and agent
  Try 'rig --help'.
  [124]

  $ rig --on a,b -- true
  rig: unknown command '--on'; the commands are run and agent
  Try 'rig --help'.
  [124]

rig run needs its machines, two or more, each named once.

  $ rig run -- true
  rig: --on is missing
  Try 'rig run --help'.
  [124]

  $ rig run --on
  rig: --on needs its machines
  Try 'rig run --help'.
  [124]

  $ rig run --on a -- true
  rig: --on names one machine, expected two or more
  Try 'rig run --help'.
  [124]

  $ rig run --on a,b,a -- true
  rig: --on names 'a' twice
  Try 'rig run --help'.
  [124]

  $ rig run --on a,,b -- true
  rig: --on has an empty name
  Try 'rig run --help'.
  [124]

  $ rig run --on a,b --on c,d -- true
  rig: --on is given twice
  Try 'rig run --help'.
  [124]

A machine is a name or an address, an IPv6 address in brackets as
everywhere in rig, and takes no port.

  $ rig run --on a,fd00::2 -- true
  rig: --on: 'fd00::2': write an IPv6 address in brackets, as [ADDRESS]
  Try 'rig run --help'.
  [124]

  $ rig run --on a,b:22 -- true
  rig: --on: 'b:22': a machine takes no port
  Try 'rig run --help'.
  [124]

  $ rig run --on 'a,[fd00::2]:22' -- true
  rig: --on: '[fd00::2]:22': a machine takes no port
  Try 'rig run --help'.
  [124]

  $ rig run --on 'a,[fd00::2' -- true
  rig: --on: '[fd00::2' is no machine
  Try 'rig run --help'.
  [124]

  $ rig run --on 'a,[]' -- true
  rig: --on: '[]' is no machine
  Try 'rig run --help'.
  [124]

  $ rig run --on 'a,[fd00::2],[fd00::2]' -- true
  rig: --on names '[fd00::2]' twice
  Try 'rig run --help'.
  [124]

and a program after --.

  $ rig run --on a,b true
  rig: -- is missing before the program
  Try 'rig run --help'.
  [124]

  $ rig run --on a,b --
  rig: no program after --
  Try 'rig run --help'.
  [124]

  $ rig run --on a,b --verbose -- true
  rig: unknown option '--verbose'
  Try 'rig run --help'.
  [124]

--on=MACHINES is --on MACHINES.

  $ rig run --on=a -- true
  rig: --on names one machine, expected two or more
  Try 'rig run --help'.
  [124]

  $ rig run --on= -- true
  rig: --on needs its machines
  Try 'rig run --help'.
  [124]

Arguments after -- are the program's, --help among them, so rig run reads
none of them.

  $ rig run --on a -- prog --help
  rig: --on names one machine, expected two or more
  Try 'rig run --help'.
  [124]

rig agent needs one address, HOST:PORT, its port a number from 0 to 65535.

  $ rig agent
  rig: the address is missing
  Try 'rig agent --help'.
  [124]

  $ rig agent 127.0.0.1
  rig: '127.0.0.1' is no HOST:PORT
  Try 'rig agent --help'.
  [124]

  $ rig agent :7000
  rig: ':7000' is no HOST:PORT
  Try 'rig agent --help'.
  [124]

  $ rig agent 127.0.0.1:http
  rig: the port of '127.0.0.1:http' is no number from 0 to 65535
  Try 'rig agent --help'.
  [124]

  $ rig agent 127.0.0.1:65536
  rig: the port of '127.0.0.1:65536' is no number from 0 to 65535
  Try 'rig agent --help'.
  [124]

  $ rig agent 127.0.0.1:-1
  rig: the port of '127.0.0.1:-1' is no number from 0 to 65535
  Try 'rig agent --help'.
  [124]

  $ rig agent 127.0.0.1:0 127.0.0.1:1
  rig: unexpected argument '127.0.0.1:1'
  Try 'rig agent --help'.
  [124]

  $ rig agent --port 0
  rig: unknown option '--port'
  Try 'rig agent --help'.
  [124]

An IPv6 address goes in brackets, as in a URI: its colons would otherwise
run into the port's.

  $ rig agent ::1:0
  rig: '::1:0': write an IPv6 address in brackets, as [ADDRESS]:PORT
  Try 'rig agent --help'.
  [124]

  $ rig agent fd00::2
  rig: 'fd00::2': write an IPv6 address in brackets, as [ADDRESS]:PORT
  Try 'rig agent --help'.
  [124]

  $ rig agent '[::1]'
  rig: '[::1]' is no HOST:PORT
  Try 'rig agent --help'.
  [124]

  $ rig agent '[::1:0'
  rig: '[::1:0' is no HOST:PORT
  Try 'rig agent --help'.
  [124]

  $ rig agent '[]:0'
  rig: '[]:0' is no HOST:PORT
  Try 'rig agent --help'.
  [124]

  $ rig agent '[::1]:http'
  rig: the port of '[::1]:http' is no number from 0 to 65535
  Try 'rig agent --help'.
  [124]
