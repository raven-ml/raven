With RUNE_JIT_DEBUG=1, a program over values cut across two devices
reports why it evaluates: a call that donates an argument, since such a
program lends no memory, and a program that gathers its cut argument
whole on each device, since it would hold what it gathers between
calls. The call that donates nothing compiles and reports nothing.

  $ RUNE_JIT_DEBUG=1 ./declines.exe 2>&1 | grep uncompiled
  rune.jit: evaluates a variant uncompiled: a donated argument over several devices: a program over them lends no memory
  rune.jit: evaluates its program uncompiled: Nx.place gathers an argument cut across devices: a program over them holds what it gathers between calls
