With RUNE_JIT_DEBUG=1, each kernel a search optimises prints the width it
searches at, the function's or the settings' (rune.mli's Search). In
search.ml's order: the function given width 1; none for the function
given none; none for width 0 under BEAM=1; the function given width 2,
once, though called again under BEAM=3 and JITBEAM=3; the three kernels
of a gradient's programs derived from a function given width 1; none for
a function given width 1 inlined in one given none; and the four kernels
of a concatenation given width 2, one a piece. The pieces run as one
launch, which rune.mli does not state: the case stays because a search
once skipped a launch's kernels.

  $ RUNE_JIT_DEBUG=1 CACHEDB=$PWD/cache ./search.exe 2>&1 | grep 'searched a kernel'
  rune.jit: searched a kernel at width 1
  rune.jit: searched a kernel at width 2
  rune.jit: searched a kernel at width 1
  rune.jit: searched a kernel at width 1
  rune.jit: searched a kernel at width 1
  rune.jit: searched a kernel at width 2
  rune.jit: searched a kernel at width 2
  rune.jit: searched a kernel at width 2
  rune.jit: searched a kernel at width 2
