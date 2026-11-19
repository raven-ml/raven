# Alda

Signal processing for OCaml over discrete-time signals, built on
[Nx](../../packages/nx/). It has windows, frame grids and their streams,
short-time transforms and their least-squares inverse, spectral scaling,
filterbanks such as the mel bank, FIR and IIR filters with their classical
designs, FFT convolution and band-limited resampling. Every
operation is a composition of nx operations, so rune differentiates,
batches and compiles it, and an operation with memory across samples also
runs over a signal given in chunks, with a stated law against the whole
call.

alda is a contrib package: it builds and tests against raven's main,
releases on its own schedule and carries no API stability guarantee.
`opam install alda` installs it; `opam install raven` does not.

- `lib/alda.mli` is the API and states every rule.
- `doc/design.md` explains the design to a reader new to raven.
- `doc/` shows compositions: Whisper's log-mel frontend, MFCC, a streamed
  log-mel, Whisper's resampling, a learned equaliser streamed in a scan.
- `test/` holds the laws and the cases against reference implementations.
