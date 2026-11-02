With RUNE_JIT_DEBUG=1, a program over values cut across two devices
reports why it evaluates. The calls that donate nothing, that donate their
argument and that gather their cut argument whole on each device compile
and report nothing.

  $ RUNE_JIT_DEBUG=1 ./declines.exe 2>&1 | grep uncompiled
  [1]

An iterate whose stop lies on another device than the program's, its
first argument's, copies the stop there before each trip, and reports it.

  $ RUNE_JIT_DEBUG=1 ./declines.exe 2>&1 | grep stop
  rune.jit: Rune.iterate' moves its stop, 1 byte, from declines-0 to declines-1 each trip
