Programs over values on two software devices. Each call reports whether
its program compiled or ran evaluated (Rune.evaluations) and whether it
gave eager's values. Over several devices a program compiles unless it
holds what rune.mli's Evaluated programs lists, such as a check in a
loop's step or stop or a switch's branch, or an addition to a total in a
branch or a remat's function, and its first call to find it declined
prints why under RUNE_JIT_DEBUG, naming the construct: a check names the
innermost loop's step or stop or switch's branch that holds it, a remat
between them included, and an addition the construct that holds it. The
first call of a compiled program prints each move between devices: the
two that gather a cut argument whole on each device, and the stop of an
iterate, which lies on another device than the program's, its first
argument's, and moves there before each trip.

  $ RUNE_JIT_DEBUG=1 ./declines.exe 2>&1 | grep -e values -e uncompiled -e moves
  a cut argument: compiled, eager's values
  a cut argument donated: compiled, eager's values
  rune.jit: Nx.place moves 8 bytes from declines-1 to declines-0
  rune.jit: Nx.place moves 8 bytes from declines-0 to declines-1
  a cut argument placed whole on each device: compiled, eager's values
  rune.jit: evaluates its program uncompiled: a check in Rune.switch's branch over several devices
  a check inside a switch's branch: evaluated, eager's values
  rune.jit: evaluates its program uncompiled: a check in a scan's step over several devices
  a check inside a scan's step: evaluated, eager's values
  rune.jit: evaluates its program uncompiled: a check in Rune.switch's branch over several devices
  a check inside a remat inside a switch's branch: evaluated, eager's values
  rune.jit: evaluates its program uncompiled: a total in Rune.switch over several devices
  a total inside a switch's branch: evaluated, eager's values
  rune.jit: evaluates its program uncompiled: a total in Rune.remat over several devices
  a total inside a remat's function: evaluated, eager's values
  rune.jit: evaluates its program uncompiled: a check in an iterate's stop over several devices
  a check inside an iterate's stop: evaluated, eager's values
  rune.jit: Rune.iterate' moves its stop, 1 byte, from declines-0 to declines-1 each trip
  an iterate whose stop lies on another device: compiled, eager's values
