# omr's design

This page explains omr's design to a reader who knows signal processing
and OCaml but not raven. It gives the foundation, then the operations
built on it today, whose interfaces may still change.
[`omr.mli`](../lib/omr.mli) states each rule in full, in the section
named.

## The foundation

### Signals are arrays

A signal is an `Nx.t` whose samples lie along its time axis, `?axis`, the
last by default. Every other axis indexes independent signals, so a
`[n; c]` array read from a file is `c` channels along axis 0. There is no
signal record. The rate is `~rate:float`, samples per unit of time, passed
only where meaning needs it, as to a mel bank. Frequencies are in cycles
per that unit. (The module preamble.)

### Every operation is an nx composition

Each operation is a composition of nx operations, so it computes where its
operands lie, eagerly on the host and compiled on a GPU, and rune
differentiates (`Rune.grad`), batches (`Rune.vmap`) and compiles
(`Rune.jit`) it. The caller compiles; omr never calls `Rune.jit`.

The cost is that every algorithm must be array operations. An IIR section
runs as associative scans over blocks of 4096 samples, whose rounding is
stated against a sequential loop's, and framing copies each frame. Values
made from numbers alone (windows, filterbanks, filter designs) are host
arrays computed at float64 and rounded once, with the same bits on every
machine, and the caller places them.

### Windows, frames and the short-time transform

A window of `n` points samples a shape on `[0, 1]`, symmetric about `1/2`,
at `i / P`, with `P = n` (periodic, for spectral analysis) or `P = n - 1`
(symmetric, for filter design). From two points on it is centred on its
point `P / 2`.

A frame grid is an edge, ends `(lo, hi)` saying how far it reads before
and past the signal, a window length and a hop. Its frame `m` starts at
position `m hop - lo`. The edge rules (`Omr.edge`) read zeros, the end
sample repeated or the signal mirrored; framing, FIR filtering and
resampling take them, and zero-phase filtering takes `Nx.extension`, which
adds a periodic rule. Centred, left-aligned and right-aligned grids, and
those of common front ends, differ only in ends and edge. A frame's
instant, the position its value stands for, is its weighting's centre,
`m hop - lo + P / 2` under a window of two points or more. Framing moves
the time axis to the second-to-last place and appends each frame's
samples, so features are
`[…; frames; bins]` and meet a filterbank `[bins; bands]` in `Nx.matmul`
without a transpose.

The short-time transform is `Nx.rfft` of each frame times the window,
unscaled. Its inverse, `istft`, is the least-squares signal for given
spectra. It divides the overlap-add of the windowed inverse transforms by
that of the squared window, and returns a signal from its own spectra
wherever a frame weighs it. Overlap-add is framing's adjoint. The
grid's type keeps its edge, so `istft` takes only the edges it inverts.
(The Windows, Frames and Short-time transforms sections.)

### Streams are values

An operation with memory across samples also runs over a signal given in
chunks. Its module has `State`, `start`, `feed`, which takes a state and
the next chunk and gives the next state and the outputs the chunk
completes, and `finish`, which gives the outputs the end completes. A
state is a structure (`Nx.Ptree.S`), so it is placed, saved or carried
through `Rune.scan` as any structure is. Parameters such as a grid stay
outside it.

Each stream states its law, which compares the joined outputs of any split
into chunks with the whole call. Computed eagerly on a kernel library
whose results do not depend on the extent of a call, as nx's CPU library's
do not, framing, the transform, overlap-add, FIR filtering, FFT
convolution and resampling give the whole call's bits. IIR filtering gives
them for chunks of multiples of 4096 samples. Under `Rune.jit` the laws
hold within rune's numerics. A mel product rounds as its matrix kernel
does, so streamed features agree within one rounding of it. (The Streams
section.) Below, frames fed in two chunks equal the whole call's.

```ocaml
let grid = Omr.Frames.v ~edge:`Symmetric_whole ~ends:(3, 3) ~window:8 ~hop:2
let x = Nx.init Nx.float32 [| 50 |] (fun i -> Float.of_int i.(0))

let () =
  let s, a = Omr.Frames.feed grid Omr.Frames.start (Nx.slice [ R (0, 17) ] x) in
  let s, b = Omr.Frames.feed grid s (Nx.slice [ R (17, 50) ] x) in
  let c = Omr.Frames.finish grid s in
  let streamed = Nx.concatenate ~axis:0 [ a; b; c ] in
  assert (Nx.to_array streamed = Nx.to_array (Omr.Frames.frames grid x))
```

Latency is stated as a release rule. An output leaves the feed after which
every sample it reads is known, so a frame leaves with its last sample,
about `window / 2` samples after its instant.

### Errors

An operation raises `Invalid_argument` at the call that misuses it.
Filtering reads no coefficient's value, so an unstable section's outputs
grow, and `Sos.stable` is a predicate the caller checks. A design may read
the host values it takes, as `Taps.minimum_phase` refuses a negative gain.

### Required semantics

Conventions that programs choose differently are required arguments: a
window's sampling, a grid's ends and edge, a mel scale and a bank's
normalisation. Other choices have defaults that their value's doc states,
such as the last axis, `fir`'s causal ends, `zero_phase`'s extension and
ends, and a mel bank's band from zero to half the rate. The guide writes
common defaults as explicit arguments ([Defaults as
arguments](index.md#defaults-as-arguments)).

### Accuracy statements

The interface states what holds for every input, never a measurement. A
law is bit for bit where the computation allows it; elsewhere it states a
bound or the inputs where accuracy degrades. FFT convolution has a
normwise bound, a section with poles near `z = ±1` needs float64, a
lowpass kernel's specification is certified on its taps, and an equiripple
design is within 1% of the least error.

## Operations built on it

These exist today and may change.

- **Spectral scaling and filterbanks:** `power_spectrum`, `psd`, and banks
  as dense arrays applied by `Nx.matmul`, the mel bank among them.
- **FIR filters:** `fir` is a one-channel `Nx.conv`, so an output's bits
  depend on its window alone.
- **IIR filters:** a cascade is a float array `[sections; 6]`, so cookbook
  sections are differentiable in their parameters, and `sos`'s first
  derivative never differentiates a pole. `Iir.meet` finds a design's
  least order.
- **FFT convolution:** overlap-save whose block size depends on the kernel
  alone, so the whole call equals its stream.
- **Resampling:** kernels as plain data, and `Kernel.lowpass` the least
  half-width, from Kaiser's estimate up, whose taps meet a specification.
- **Tap designs:** least squares, equiripple with a certified least
  length, minimum phase and Savitzky-Golay.

## Decided, not yet built

These designs are decided and await implementation:

- the analytic signal, cross spectra, autoregressive models and DPSS
  tapers;
- peaks that trace under `Rune.jit`, sampling at arbitrary positions,
  phase unwrapping, more windows and octave bands;
- constant-Q transforms with atoms as data, streamed with a stated
  latency, an exact inverse, and Morlet wavelets;
- Bark and ERB-rate scales beside mel, and gammatone filters;
- guide compositions for deltas, MFCC variants, PCEN, onset strength,
  loudness, Griffin-Lim and the phase vocoder;
- converting frames and durations to times, in the guide.

Proposed beyond these: a latency in samples for every stream with a rule
for composing them, a cascade per signal in `sos`, and a matrix product in
nx whose bits do not depend on the number of frames in a call.
