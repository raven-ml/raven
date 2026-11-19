# alda

Discrete-time signals over nx: windows, framing, short-time transforms,
spectra and filterbanks. It lives in `dev/` and is never published (no opam
package, no CHANGES entry).

## Boundaries

- alda depends on nx, and on rune for a traced loop (`Rune.scan`) and the
  derivatives it states itself (`Rune.custom_jvp`, `Rune.custom_transpose`);
  on nothing else. It never calls `Rune.jit`: the caller compiles.
- A signal is a plain `Nx.t` with its time axis `?axis`; the rate is
  `~rate:float`. No signal record, units or phantom rate.
- A capability alda lacks in nx (an extension rule, a transform, a kernel) is
  built in nx, never worked around here.
- Features are `[…; frames; bins]`: framing moves the time axis to the
  second-to-last place and appends the frame's samples.
- Windows and filterbanks are host values computed at float64 by nx
  operations and rounded once. Compositions (log-mel, MFCC, dB floors,
  clamps, Welch's mean) are guide code, never values of the library.

## Decisions

- A stateful operation is the whole function plus a module with `State`,
  `start`, `feed` and `finish`; the whole call is one feed. Each states its
  chunked-against-whole law in `alda.mli`.
- Overlap-add adds frame by frame from `+0` with `Nx.add`, never through a
  fold of the kernel library, so its stream and the whole call share bits.
- Overlap-add is the adjoint of framing under every edge, the extension's
  terms folded back onto the samples they read with their coefficients.
  `istft` divides it by the folded envelope, the least-squares inverse only
  where each position reads one sample, so a grid's type carries its edge and
  `istft` takes `[< invertible ] Frames.t`.
- A stream's state holds no counter that changes per chunk, so that a loop
  can carry it. Where the end needs a length, `finish` takes only what the
  fed frames leave open (`~extra`).
- IIR sections run in blocks of 4096 samples from each call's start, the
  DF-I state injected into a block's first two samples, so chunks at
  multiples of 4096 give one call's bits. The derivative is a
  `Rune.custom_jvp` whose transpose runs the recursion backward, so the
  first derivative never differentiates the poles.
- Weighted sums (FIR, resampling) are one-channel `Nx.conv` reductions or
  `Nx.sum`s of one output's products, never `matmul` or a contraction, so
  an output's bits depend on its window alone.
- FFT convolution sizes its blocks from the kernel alone, so the whole
  call equals its stream.

## Traps

- The mel product (`Nx.matmul` by a bank) rounds by the number of frames
  per call, so streamed features equal the whole call's only within one
  rounding of the product.
- Kaldi's and librosa's lifters, logs and floors differ: compositions in the
  guide name which one they follow.
- The cascade's first derivative differentiates no pole; higher ones
  differentiate the blocked evaluation and may be NaN at an exact double
  pole. A tangent map that called the rule again would recurse forever
  under `grad` (tracing a `Rune.custom_jvp` traces its tangent map); the
  fix is a transpose-aware linear solve in rune, a rune design question.
- A resampling stream's first feed moves its state from `start` to arrays
  and emits fewer outputs: inside `Rune.scan` feed the first chunk outside.
- SciPy's `sosfiltfilt` pad reads coefficient values (zeros in `b2`, `a2`);
  alda's cannot, so odd orders differ from SciPy near the ends.
