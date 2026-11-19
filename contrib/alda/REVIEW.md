# Reviewing alda

## Design

- A loop or kernel of alda's own where nx operations compose the result;
  rune can't differentiate, batch or compile what alda hides. A capability
  nx lacks is built in nx.
- A composition (log-mel, MFCC, a dB floor, Welch's mean) made a value of
  the library, where guide code shows it.
- An operation with memory across samples without its `State`, `start`,
  `feed` and `finish` and its chunked-against-whole law, or with a state
  that changes shape per chunk, which a loop cannot carry.
- A call to `Rune.jit`: the caller compiles.

## Semantics

References: the papers and programs each section of `alda.mli` cites, and
the generator of each golden under `test/gen/`.

Failure here is a signal off by a sample or a scale: a frame that starts a
sample late, an edge that reads the wrong sample, a spectrum scaled by the
wrong window sum, a stream whose output differs from the whole call's.

- Edges on signals shorter than a frame or a kernel, and each rule's
  coefficients under overlap-add.
- Streams split at every boundary, empty and one-sample chunks included,
  against the whole call: bit for bit where the law says bits.
- Float32: a window, bank or weight rounded once, and each sum within its
  stated bound.
- Derivatives at a double pole, at a band edge and at a zero signal.
