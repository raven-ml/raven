With RUNE_JIT_DEBUG=1, a program over values cut across two devices
reports why it evaluates: a program that gathers its cut argument whole
on each device, since a compiled call would hold all it gathers and
closes at once. The calls that donate nothing and that donate their
argument compile and report nothing.

  $ RUNE_JIT_DEBUG=1 ./declines.exe 2>&1 | grep uncompiled
  rune.jit: evaluates its program uncompiled: Nx.place gathers an argument cut across devices: a compiled call holds all it gathers and closes at once
