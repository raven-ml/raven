With RUNE_JIT_DEBUG=1, a call reports each retrace with the first difference
from the previous key (a shape, a view's strides, where its run of storage
starts), and each consumed leaf: the result it lent its storage to, and
whether it was reused or copied.

  $ RUNE_JIT_DEBUG=1 CACHEDB=$PWD/cache ./reports.exe
  rune.jit: 0.0 -> result 0 reused
  rune.jit: 0.1 -> result 1 reused
  rune.jit: retrace: 0.0: shape [8] here, shape [4] in the previous key
  rune.jit: 0.0 -> result 0 reused
  rune.jit: 0.1 -> result 1 reused
  rune.jit: 0.0 -> result 0 copied
  rune.jit: 0.1 -> result 1 reused
  rune.jit: retrace: 0: strides [1; 2] here, strides [3; 1] in the previous key
  rune.jit: retrace: 0: shape [4] here, shape [2; 3] in the previous key
  rune.jit: retrace: 0: 1 element into its run here, 0 elements into its run in the previous key
  rune.jit: 0 consumed, lent to no result
  rune.jit: retrace: BEAM=0 NOOPT=true profiled=false here, BEAM=0 NOOPT=false profiled=false in the previous key

Without RUNE_JIT_DEBUG, a call reports nothing.

  $ CACHEDB=$PWD/cache ./reports.exe
