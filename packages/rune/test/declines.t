With RUNE_JIT_DEBUG=1, a program over values cut across two devices
reports why it evaluates: a program that gathers its cut argument whole
on each device, since a compiled call would hold all it gathers and
closes at once. The calls that donate nothing and that donate their
argument compile and report nothing.

  $ RUNE_JIT_DEBUG=1 ./declines.exe 2>&1 | grep uncompiled
  rune.jit: evaluates its program uncompiled: Nx.place gathers an argument cut across devices: a compiled call holds all it gathers and closes at once

An iterate whose stop lies on another device than the program's, its
first argument's, copies the stop there before each trip, and reports it.

  $ RUNE_JIT_DEBUG=1 ./declines.exe 2>&1 | grep stop
  rune.jit: Rune.iterate' moves its stop, 1 byte, from declines-0 to declines-1 each trip
