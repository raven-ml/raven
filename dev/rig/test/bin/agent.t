rig agent alone, its input held open through a named pipe as rig run's
ssh session holds it. The lock between agents lives in TMPDIR.

  $ . support/env.sh
  $ export TMPDIR=$PWD/tmp
  $ mkdir tmp
  $ key=0123456789abcdef0123456789abcdef
  $ show() { sed -E -e 's/:[0-9]+$/:PORT/' -e "s/^rig-agent $(rig --version)\$/rig-agent VERSION/" "$1"; }

An agent first names rig's version, then reads the key, listens
at a port the system chooses, and says where.

  $ mkfifo in1
  $ rig agent 127.0.0.1:0 <in1 >out1 &
  $ first=$!
  $ exec 3>in1
  $ echo $key >&3
  $ ./support/await out1 listening
  $ show out1
  rig-agent VERSION
  listening 127.0.0.1:PORT

A second agent of the user on this machine waits while the first runs,
and says so.

  $ mkfifo in2
  $ rig agent 127.0.0.1:0 <in2 >out2 3>&- &
  $ second=$!
  $ exec 4>in2
  $ echo $key >&4
  $ ./support/await out2 waiting
  $ show out2
  rig-agent VERSION
  waiting

When the first agent's input ends, it ends: it writes nothing more, exits
123, and the second listens.

  $ exec 3>&-
  $ wait $first
  [123]
  $ show out1
  rig-agent VERSION
  listening 127.0.0.1:PORT
  $ ./support/await out2 listening
  $ show out2
  rig-agent VERSION
  waiting
  listening 127.0.0.1:PORT

Once the second ends too, an agent listens at once, with no wait.

  $ exec 4>&-
  $ wait $second
  [123]
  $ mkfifo in3
  $ rig agent 127.0.0.1:0 <in3 >out3 &
  $ third=$!
  $ exec 5>in3
  $ echo $key >&5
  $ ./support/await out3 listening
  $ show out3
  rig-agent VERSION
  listening 127.0.0.1:PORT
  $ exec 5>&-
  $ wait $third
  [123]

An IPv6 address is written in brackets, and the agent says where it
listens the same way.

  $ mkfifo in7
  $ rig agent '[::1]:0' <in7 >out7 &
  $ seventh=$!
  $ exec 9>in7
  $ echo $key >&9
  $ ./support/await out7 listening
  $ show out7
  rig-agent VERSION
  listening [::1]:PORT
  $ exec 9>&-
  $ wait $seventh
  [123]

A key outside 16 to 4096 bytes, or no key, fails the agent before it
listens.

  $ mkfifo in4
  $ rig agent 127.0.0.1:0 <in4 >out4 &
  $ fourth=$!
  $ exec 6>in4
  $ echo short >&6
  $ wait $fourth
  [123]
  $ exec 6>&-
  $ show out4
  rig-agent VERSION
  failed the key has 5 bytes, outside 16 to 4096

  $ rig agent 127.0.0.1:0 </dev/null >out8
  [123]
  $ show out8
  rig-agent VERSION
  failed no key on standard input

A line longer than any key is refused once it is, its length given, and
read no further: here the rest of its standard input is left for cat.

  $ head -c 4097 /dev/zero | tr '\0' k >long
  $ echo >>long
  $ rig agent 127.0.0.1:0 <long >out-long
  [123]
  $ show out-long
  rig-agent VERSION
  failed the key has 4097 bytes, outside 16 to 4096

  $ head -c 10000 /dev/zero | tr '\0' k >longer
  $ (rig agent 127.0.0.1:0 >out-longer; cat | wc -c | tr -d ' ') <longer
  5903
  $ show out-longer
  rig-agent VERSION
  failed the key has 4097 bytes, outside 16 to 4096

An address that does not resolve fails the agent.

  $ mkfifo in5
  $ rig agent nowhere.invalid:0 <in5 >out5 &
  $ fifth=$!
  $ exec 7>in5
  $ echo $key >&7
  $ wait $fifth
  [123]
  $ exec 7>&-
  $ show out5
  rig-agent VERSION
  failed nowhere.invalid does not resolve

The lock is a file of the user's own. Another file in its place, here a
directory, fails the agent.

  $ mkdir other
  $ mkdir other/rig-agent-$(id -u).lock
  $ mkfifo in6
  $ TMPDIR=$PWD/other rig agent 127.0.0.1:0 <in6 >out6 &
  $ sixth=$!
  $ exec 8>in6
  $ echo $key >&8
  $ wait $sixth
  [123]
  $ exec 8>&-
  $ show out6 | sed -e "s|$PWD|PWD|" -e "s|-$(id -u)[.]|-UID.|"
  rig-agent VERSION
  failed PWD/other/rig-agent-UID.lock is no regular file

The agent's last line says how its job ended. The controller here is
support/ctl.exe, given by hand what rig run gives a program: the agents,
the key, and a report, the file report.

  $ port() { sed -n 's/^listening .*:\([0-9]*\)$/\1/p' "$1"; }
  $ ctl() { rm -f attempts; RIG_REMOTE_AGENTS=b=127.0.0.1:$(port $1) RIG_REMOTE_KEY=$key$key RIG_REMOTE_REPORT=3 ./support/ctl.exe $2 3>report; }

A job its controller closes ends in order: the agent writes closed and
exits 0.

  $ mkfifo in10
  $ rig agent 127.0.0.1:0 <in10 >out10 &
  $ agent=$!
  $ exec 3>in10
  $ echo $key$key >&3
  $ ./support/await out10 listening
  $ ctl out10 copy
  attempt 1
  copied 1
  $ wait $agent
  $ show out10
  rig-agent VERSION
  listening 127.0.0.1:PORT
  closed
  $ cat report
  started
  closed
  $ exec 3>&-

A job that fails, here as its controller dies, ends the agent with its
reason: failed, and 123.

  $ mkfifo in11
  $ rig agent 127.0.0.1:0 <in11 >out11 &
  $ agent=$!
  $ exec 3>in11
  $ echo $key$key >&3
  $ ./support/await out11 listening
  $ ctl out11 kill 2>/dev/null
  attempt 1
  [137]
  $ wait $agent
  [123]
  $ show out11
  rig-agent VERSION
  listening 127.0.0.1:PORT
  failed controller: closed its connection
  $ exec 3>&-

An agent killed reports nothing: rig agent says how it died, and exits
123. The controller learns of the death through its job.

  $ mkfifo in12
  $ rig agent 127.0.0.1:0 <in12 >out12 &
  $ agent=$!
  $ exec 3>in12
  $ echo $key$key >&3
  $ ./support/await out12 listening
  $ ctl out12 wait >ctlout 2>ctlerr &
  $ controller=$!
  $ ./support/await ctlout joined
  $ kill -KILL $(pgrep -P $agent)
  $ wait $agent
  [123]
  $ show out12
  rig-agent VERSION
  listening 127.0.0.1:PORT
  died killed by SIGKILL
  $ wait $controller
  [2]
  $ cat report
  started
  failed b: closed its connection
  $ exec 3>&-
