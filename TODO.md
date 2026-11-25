# todo

- rig, talon: find what makes rig's image drops and two-domain launches, and
  talon's window law, slow enough to time out on slow opam CI hosts, and fix.
- nx: add a metal backend.
- nx: port eig's QR split at the iteration limit to nx.cuda and nx.amd.
- nx: fix test_spread's 1-ulp `std` (replay `r1:AQAAAICAgKiAgIDw3wEAAA:75a726be`).
- lz: support big-endian (s390x), proven on opam CI's s390x workers.
- ymir: fit F-1 itself, so Omega_nu keeps full precision near y = 2^-7.
- hugin: cap ticks on continuous scales.
- talon: accept musl's symlink-loop message in test_tz's expect output.
- rig: make `Machine.functions` leave out or refuse an `ops.functions` name
  that is no bus address.
- rig: state in rig_metal.mli the two refusals of a GPU without residency
  sets.

## next

- rune: cut eager per-op cost (norn 0.64 µs an equation, jera ~30 ms an eager
  solve).
- tolk: stop the register budget overcharging contractions 2.5-4x.
- tolk: decide which kernels get lanes: LLVM's vectorizer, not tolk, gives
  gammainccinv f16 its 2-4x today, and upcasting every such kernel makes
  binomial's first call 33 s.
- nx: give Prog a bounded loop, so fixed-round kernels (samplers,
  gammainccinv's Halley steps) shrink by their round count. After the lanes
  decision, if the big kernels are still the slowest.
- tolk: replace our C renderer in Tolk with LLVM IR, which can be compiled both by
  loading `libLLVM`, or with `clang -x ir`.
