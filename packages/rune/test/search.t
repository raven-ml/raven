With RUNE_JIT_DEBUG=1, a compiled call reports each kernel a search
optimises, at the width the function was given, or the settings'. In
search.ml's order: the function given width 1; none for the function
given none; none for width 0 under BEAM=1; the function given width 2,
once, though called again under BEAM=3 and JITBEAM=3; the three
kernels of a gradient's programs derived from a function given width 1;
and none for a function given width 1 inlined in one given none.

  $ RUNE_JIT_DEBUG=1 CACHEDB=$PWD/cache ./search.exe 2>&1 | grep searched
  rune.jit: searched a kernel at width 1
  rune.jit: searched a kernel at width 2
  rune.jit: searched a kernel at width 1
  rune.jit: searched a kernel at width 1
  rune.jit: searched a kernel at width 1
