With RUNE_JIT_DEBUG=1, a call reports each retrace with the first difference
from the previous key, and each consumed leaf: the result it lent its storage
to, and whether it was reused or copied.

  $ RUNE_JIT_DEBUG=1 CACHEDB=$PWD/cache ./reports.exe
  rune.jit: 0.0 -> result 0 reused
  rune.jit: 0.1 -> result 1 reused
  rune.jit: retrace: 0.0: shape [8] here, shape [4] in the previous key
  rune.jit: 0.0 -> result 0 reused
  rune.jit: 0.1 -> result 1 reused
  rune.jit: 0.0 -> result 0 copied
  rune.jit: 0.1 -> result 1 reused
