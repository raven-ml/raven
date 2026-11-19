(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Discrete-time signals: windows, frames, short-time transforms, spectra,
    filterbanks, filters, FFT convolution and resampling, as compositions of nx
    operations.

    A {e signal} is an [Nx.t] whose {e samples} [x(0)], …, [x(n - 1)] lie along
    one of its axes, its {e time axis}. Every other axis indexes independent
    signals. A [[n; c]] array read along axis [0] is [c] signals of [n] samples,
    and each operation computes on each of them alike. An operation takes the
    time axis as [?axis], the last by default; a negative axis counts from the
    end.

    A signal's {e rate} is its number of samples per unit of time, passed as
    [~rate], a positive finite float. Frequencies are in cycles per that unit,
    so at [~rate:16_000.] samples a second they are in hertz; {!Iir.analog}
    alone takes values of the Laplace variable, where frequency [f] lies at
    [2πi f]. A perceptual scale such as the mel scale ({!Mel}) is defined in
    hertz, so an operation that names one takes the second as the unit. An
    operation in samples alone, such as framing, takes no rate.

    Unless a value's doc says otherwise:
    - an operation computes where its operands lie and at their dtype, so that
      rune differentiates, batches and compiles it as it does the nx operations
      it is made of;
    - a value made from numbers alone, such as a window, is a host value
      computed at float64 by nx operations and rounded once into its dtype, with
      the same bits on every machine; the caller places it with {!Nx.place};
    - an operation raises [Invalid_argument] at the call that misuses it, such
      as one whose [axis] is not an axis of its signal.

    {1:streams Streams}

    An operation with memory across samples, such as framing, also computes over
    a signal given in consecutive {e chunks} along its time axis, as a
    microphone delivers them. Such an operation [op] has a module [Op] with:
    - [State], the type of the state between chunks, a structure ({!Nx.Ptree.S})
      so that a state is placed, saved and carried through a loop as any
      structure is;
    - [start], the state before any sample, which holds no array;
    - [feed], which takes a state and the next chunk and gives the next state
      and the outputs the chunk completes;
    - [finish], where the end of the signal changes outputs, which gives the
      outputs that the end completes.

    An operation whose state may begin elsewhere than at rest, as a filter's may
    begin from initial conditions, also has a constructor for that state.
    Parameters, such as a frame grid or a window, are passed to every call and
    stay outside the state. Every call on one stream must take the same
    parameters and the same [axis], and chunks whose other axes are those of the
    first.

    [op x] is [start], one [feed] of [x], then [finish], their outputs joined
    along the time axis. Each operation states its {e law}, how the joined
    outputs of any split of a signal into chunks, empty ones included, compare
    with [op x]. Framing copies samples, a transform reads each frame alone and
    overlap-add adds in an order it states, so for these the law is bit for bit,
    computed eagerly on a kernel library whose results do not depend on the
    extent of a call, as [Nx_cpu]'s do not; under [Rune.jit] it holds within the
    bounds of rune's numerics. *)

(** {1:windows Windows}

    A {e window} of [n] points is a host array whose point [i] is [f (i / P)],
    for a {e shape} [f] on \[0, 1\] symmetric about [1/2]. A {e periodic} window
    takes [P = n], so that it holds one period of [f] without the point that
    would open the next (Harris, "On the Use of Windows for Harmonic Analysis
    with the Discrete Fourier Transform", Proc. IEEE 66(1), 1978, §II), as
    spectral analysis needs. A {e symmetric} window takes [P = n - 1], so that
    point [i] equals point [n - 1 - i], as filter design needs. Under both, a
    window of one point is [[1.]] and a window of no point is empty.

    Each constructor computes at float64 and rounds once into [dt]. Its point
    [i] has the bits of point [P - i] modulo [n], so a window of [n ≥ 2] points
    is {e centred} on its point [P / 2]; a window of one point is centred on
    that point, [0]. Each raises [Invalid_argument] if [n] is negative. *)

module Window : sig
  (** The type for the two samplings of a window's shape. *)
  type sampling = Periodic  (** [P = n]. *) | Symmetric  (** [P = n - 1]. *)

  val cosine_sum :
    (float, 's) Nx.dtype ->
    sampling ->
    float array ->
    int ->
    (float, 's, Nx.host) Nx.t
  (** [cosine_sum dt s a n] is the window of the shape
      [f u = Σ_k (-1){^k} a.(k) cos (2πku)], its terms added in order of [k].

      Raises [Invalid_argument] also if [a] is empty. *)

  val hann :
    (float, 's) Nx.dtype -> sampling -> int -> (float, 's, Nx.host) Nx.t
  (** [hann dt s n] is [cosine_sum dt s [| 0.5; 0.5 |] n]. *)

  val hamming :
    (float, 's) Nx.dtype -> sampling -> int -> (float, 's, Nx.host) Nx.t
  (** [hamming dt s n] is [cosine_sum dt s [| 0.54; 0.46 |] n]. *)

  val blackman :
    (float, 's) Nx.dtype -> sampling -> int -> (float, 's, Nx.host) Nx.t
  (** [blackman dt s n] is [cosine_sum dt s [| 0.42; 0.5; 0.08 |] n]. *)

  val kaiser :
    (float, 's) Nx.dtype ->
    sampling ->
    beta:float ->
    int ->
    (float, 's, Nx.host) Nx.t
  (** [kaiser dt s ~beta n] is the window of the shape
      [f u = I₀ (β √(1 - (2u - 1){^2})) / I₀ β], where [I₀] is the modified
      Bessel function of the first kind of order 0. Its points are finite at
      every [β], computed as a ratio of {!Nx.i0e}.

      Raises [Invalid_argument] also if [beta] is negative or not finite. *)

  val gaussian :
    (float, 's) Nx.dtype ->
    sampling ->
    sigma:float ->
    int ->
    (float, 's, Nx.host) Nx.t
  (** [gaussian dt s ~sigma n] is the window whose point [i] is
      [exp (-((i - P / 2) / σ){^2} / 2)], with [σ] in samples.

      Raises [Invalid_argument] also if [sigma] is not positive and finite. *)
end

(** {1:framing Frames}

    A {e frame grid} cuts a signal into {e frames} of [window] consecutive
    samples, one every [hop]. Its {e ends} [(lo, hi)] are how far it reads
    before the signal's first sample and past its last, and its {e edge} is the
    rule that reads the signal there. Frame [m ≥ 0] holds positions [m hop - lo]
    to [m hop - lo + window - 1] of the signal read under the edge, and the grid
    has every frame that ends by position [n + hi - 1], as many as
    {!Nx.sliding_window} cuts from the signal extended by [lo] samples before it
    and [hi] after it.

    Point [c] of frame [m], for a real [c] in \[0, window - 1\], lies at
    position [m hop - lo + c] of the signal. A value computed from a frame under
    a weighting centred on its point [c], as a short-time transform computes
    one, stands for the frame at that position, the frame's {e instant}. A
    window of {!Window} of two points or more is centred on its point [P / 2]
    ({!section-windows}), so an instant is a sample when [P] is even and lies
    halfway between two samples when [P] is odd. With [lo = window / 2], frame
    [m]'s instant is [m hop] under a window of one point, a periodic window of
    even [window] or a symmetric one of odd [window], [m hop + 1/2] under a
    periodic one of odd [window] above [1] and [m hop - 1/2] under a symmetric
    one of even [window]. With [hi = window / 2] such a grid has [n / hop + 1]
    frames for an even [window], and with [hi = window / 2 - hop] one fewer.
    Ends [(0, 0)] keep the frames that lie within the signal.

    Framing replaces the time axis by the frames, moved to the second-to-last
    place, and appends a frame's samples as the last axis; the other axes keep
    their order. A [[b; n]] signal gives [[b; frames; window]], and a [[n; c]]
    signal along axis [0] gives [[c; frames; window]], each channel a mono
    signal's frames. *)

type edge =
  [ `Zero
  | `Replicate
  | `Symmetric_whole
  | `Symmetric_half
  | `Antisymmetric_whole ]
(** The type for framing's edge rules: {!Nx.extension}'s rules, each read as
    {!Nx.extend} reads it, but [`Periodic]. A periodic grid with [lo > 0] reads
    the signal's last samples in its first frame, which a stream receives last.
    A whole signal is framed periodically by {!Nx.sliding_window} of
    {!Nx.extend}[ `Periodic]. *)

module Frames : sig
  type +'e t = private { edge : 'e; ends : int * int; window : int; hop : int }
  (** The type for frame grids read under an edge of type ['e]. A grid's type
      keeps the edge it was made with, so that {!istft} takes only the grids it
      inverts; [(g :> edge t)] forgets it, as a list of grids of several edges
      needs. *)

  val v :
    edge:([< edge ] as 'e) -> ends:int * int -> window:int -> hop:int -> 'e t
  (** [v ~edge ~ends ~window ~hop] is the grid of those parts.

      Raises [Invalid_argument] if [window] or [hop] is below [1] or an end is
      negative. *)

  val count : [< edge ] t -> int -> int
  (** [count g n] is the number of frames of a signal of [n] samples,
      [(n + lo + hi - window) / hop + 1], or [0] where [n + lo + hi < window].

      Raises [Invalid_argument] if [n] is negative. *)

  val frames :
    ?axis:int -> [< edge ] t -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
  (** [frames g x] is the [count g n] frames of [x] along [axis]:
      {!Nx.sliding_window}[ ~window ~step:hop] of [x] extended by [(lo, hi)]
      under [edge], a copy of [x]'s samples.

      Raises [Invalid_argument] if [axis] is not an axis of [x], if [x] has no
      sample along it and [edge] other than [`Zero] reads past an end, or as
      {!Nx.extend} does. *)

  (** {2:framing_stream Framing a stream}

      Framing streams. A frame comes out of the [feed] after which every sample
      it reads is known, once every frame before it is out. A frame that reads
      before the signal's start waits for the samples its edge reads there,
      [lo + 1] under [`Symmetric_whole]. The frames that read past the end come
      out of [finish]. Once the frames that read before the start are out, a
      chunk of [k hop] samples gives [k] frames and leaves the state's shapes
      unchanged. The frames of every split, joined along the frames' axis, equal
      {!frames}' bit for bit. *)

  module State : Nx.Ptree.S
  (** The state between chunks: the samples later frames read. *)

  val start : ('a, 'd) State.t
  (** [start] is the state before any sample. *)

  val feed :
    ?axis:int ->
    [< edge ] t ->
    (('v, 's, 'd) Nx.t, 'd) State.t ->
    ('v, 's, 'd) Nx.t ->
    (('v, 's, 'd) Nx.t, 'd) State.t * ('v, 's, 'd) Nx.t
  (** [feed g s x] is the state after the samples [x], the next along [axis],
      and the [[…; k; window]] frames they complete, [k ≥ 0].

      Raises [Invalid_argument] if [axis] is not an axis of [x], or if [x]'s
      other axes are not those of the chunks before it. *)

  val finish :
    [< edge ] t -> (('v, 's, 'd) Nx.t, 'd) State.t -> ('v, 's, 'd) Nx.t
  (** [finish g s] is the frames left once the signal has ended.

      Raises [Invalid_argument] if no chunk was fed (a stream that may end
      before its first sample feeds an empty chunk first), and as {!frames} does
      on the whole signal. *)

  (** {2:overlap_adding Overlap-add}

      {e Overlap-add} is the adjoint of framing. It maps [count g n] frames
      [[…; frames; window]] to a signal of [n] samples, sample [i] the sum of
      [c] times what the frames hold at every position that reads [x(i)] with
      coefficient [c] under the grid's edge, the extension's positions included.
      Every rule reads one sample with coefficient [1] at a position but
      [`Antisymmetric_whole], whose position [-i] reads [2 x(0) - x(i)], so
      [x(0)] with coefficient [2] and [x(i)] with [-1], and whose positions past
      a further fold read integer multiples of samples. Under [`Zero] it drops
      what the frames hold outside the signal.

      It adds in a stated order. First the frames are added along the extended
      signal frame by frame from frame [0], each position's sum starting from
      [+0]. Then sample [i] is the sum, from [+0], of [c] times those sums at
      every position that reads [x(i)] with coefficient [c], in order of
      position.

      Its stream gives a sample once no later frame reaches it and no position
      past the signal's end can read it, whatever the signal's length. Such a
      position reads one of the last [hi + 1] samples under [`Symmetric_whole]
      and [`Antisymmetric_whole], [hi] under [`Symmetric_half] and [1] under
      [`Replicate]. It adds in the same order, so the samples of every split of
      the frames, joined along the time axis, equal {!overlap_add}'s bit for
      bit. A whole call makes [⌈window / hop⌉] passes over the signal. *)

  val overlap_add :
    ?axis:int ->
    [< edge ] t ->
    int ->
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t
  (** [overlap_add g n f] is the overlap-add of the frames [f] to [n] samples
      along [axis] of the result, its other axes [f]'s but the last two.

      Raises [Invalid_argument] if [n] is negative, if [f] does not hold
      [count g n] frames of [window] samples, or if [axis] is not an axis of the
      result. *)

  module Overlap_add : sig
    module State : Nx.Ptree.S
    (** The state between chunks: the sums later frames still reach and the
        samples held back. *)

    val start : ('a, 'd) State.t
    (** [start] is the state before any frame. *)

    val feed :
      ?axis:int ->
      [< edge ] t ->
      ((float, 's, 'd) Nx.t, 'd) State.t ->
      (float, 's, 'd) Nx.t ->
      ((float, 's, 'd) Nx.t, 'd) State.t * (float, 's, 'd) Nx.t
    (** [feed g s f] is the state after the frames [f], the next
        [[…; k; window]], and the samples they complete along [axis]. Once the
        samples that the start's extension reads are out, [k] frames give
        [k hop] samples and leave the state's shapes unchanged.

        Raises [Invalid_argument] as {!overlap_add} does, or if [f]'s other axes
        are not those of the frames before it. *)

    val finish :
      ?axis:int ->
      [< edge ] t ->
      extra:int ->
      ((float, 's, 'd) Nx.t, 'd) State.t ->
      (float, 's, 'd) Nx.t
    (** [finish g ~extra s] is the rest of the signal whose frames were fed, of
        [(m - 1) hop + window - lo - hi + extra] samples after [m ≥ 1] frames
        and of [extra] samples after none. The frames fix a signal's length but
        for [extra], [0 ≤ extra < hop] after frames and
        [0 ≤ extra < window - lo - hi] after none. A signal of [n] samples has
        [extra = (n + lo + hi - window) mod hop], or [n] where
        [n + lo + hi < window]. A stream whose signal ends with its last frame
        takes [~extra:0].

        Raises [Invalid_argument] if no chunk was fed (a stream that may end
        before its first frame feeds an empty chunk first), if [extra] is
        outside its range, or if the length it gives is negative. *)
  end
end

(** {1:short_time Short-time transforms}

    The {e short-time transform} of a signal under a frame grid [g] and a window
    [win] of [g]'s [window] points is the {!Nx.rfft} of each frame multiplied by
    [win], unscaled. A signal of [n] samples gives [Frames.count g n] frames of
    [window / 2 + 1] bins, [[…; frames; bins]]. A frame transformed at more
    points, with zeros after it, is {!Nx.rfft}[ ~s] of the windowed frames.
    Zeros on both sides of a shorter taper are the taper {!Nx.pad}ded to
    [window].

    The transform has no stream of its own. Its stream is framing's
    ({!Frames.feed}) followed by the same product and {!Nx.rfft} of each frame,
    whose bits do not depend on the frames beside it on a kernel library such as
    [Nx_cpu].

    The {e inverse} of spectra [z] is the signal [x] of [n] samples that
    minimises the sum of squares of the difference between [z] and [x]'s
    transform over the full spectrum, of least norm among those. Each frame is
    inverted by {!Nx.irfft} at [window] points and multiplied by [win]; their
    overlap-add ({!Frames.overlap_add}) is divided by the {e envelope}, the
    overlap-add of [win{^2}] over the same frames. A sample whose envelope is
    zero, which no frame weighs, is [0.]. So the inverse of a signal's transform
    is the signal, to rounding, at every sample of positive envelope. The
    division is the least-squares solution where each position of the extended
    signal reads one sample, so the inverse takes the {!invertible} grids. *)

type invertible = [ `Zero | `Replicate | `Symmetric_whole | `Symmetric_half ]
(** The type for the edges whose positions each read one sample: {!edge} but
    [`Antisymmetric_whole], whose positions before the start read two, which
    couples them in the least-squares equations. *)

val stft :
  ?axis:int ->
  (Complex.t, 'c) Nx.dtype ->
  [< edge ] Frames.t ->
  win:(float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (Complex.t, 'c, 'd) Nx.t
(** [stft dt g ~win x] is [Nx.rfft dt (Nx.mul (Frames.frames ~axis g x) win)].

    Raises [Invalid_argument] if [win] is not [[window]], and as
    {!Frames.frames} does. *)

val istft :
  ?axis:int ->
  (float, 'r) Nx.dtype ->
  [< invertible ] Frames.t ->
  win:(float, 'r, 'd) Nx.t ->
  int ->
  (Complex.t, 'c, 'd) Nx.t ->
  (float, 'r, 'd) Nx.t
(** [istft dt g ~win n z] is the inverse of the spectra [z], [n] samples along
    [axis] of the result, its other axes [z]'s but the last two.

    Raises [Invalid_argument] if [win] is not [[window]], if [z] does not hold
    [Frames.count g n] frames of [window / 2 + 1] bins, and as
    {!Frames.overlap_add} does. *)

(** The inverse as a stream. It runs the overlap-adds of the windowed frames and
    of the envelope as streams ({!Frames.Overlap_add}) and divides each sample
    by its envelope as it comes out, so every split gives {!istft}'s bits. *)
module Istft : sig
  module State : Nx.Ptree.S
  (** The state between chunks: two overlap-add states. *)

  val start : ('a, 'd) State.t
  (** [start] is the state before any spectrum. *)

  val feed :
    ?axis:int ->
    (float, 'r) Nx.dtype ->
    [< invertible ] Frames.t ->
    win:(float, 'r, 'd) Nx.t ->
    ((float, 'r, 'd) Nx.t, 'd) State.t ->
    (Complex.t, 'c, 'd) Nx.t ->
    ((float, 'r, 'd) Nx.t, 'd) State.t * (float, 'r, 'd) Nx.t
  (** [feed dt g ~win s z] is the state after the spectra [z], the next
      [[…; k; bins]], and the samples they complete along [axis].

      Raises [Invalid_argument] as {!istft} does, or if [z]'s other axes are not
      those of the spectra before it. *)

  val finish :
    ?axis:int ->
    [< invertible ] Frames.t ->
    extra:int ->
    ((float, 'r, 'd) Nx.t, 'd) State.t ->
    (float, 'r, 'd) Nx.t
  (** [finish g ~extra s] is the rest of the signal whose spectra were fed, of
      the length {!Frames.Overlap_add.finish} gives it, and raises as that does.
  *)
end

(** {1:scaling Spectral scaling}

    These scale [p], the squared magnitudes [re{^2} + im{^2}] of the one-sided
    spectra of [n]-point transforms of frames multiplied by a window [win],
    [[…; n / 2 + 1]], where [n], [win]'s length by default, is at least [win]'s
    length and the frames are followed by zeros. Bin [0] and, for an even [n],
    bin [n / 2] each stand for one frequency; every other bin stands for a
    positive and a negative one and counts twice. So [c_k] is [1] for the first
    and [2] for the others (Heinzel, Rüdiger and Schilling, "Spectrum and
    spectral density estimation by the Discrete Fourier transform (DFT)", 2002,
    eqs. 23 and 24). Each multiplies [p] by [c_k] and divides by its scale once.
    Each raises [Invalid_argument] if [win] is not a 1-d array of at least one
    point, [n] is below its length, or [p]'s last axis is not [n / 2 + 1]. *)

val psd :
  ?n:int ->
  rate:float ->
  win:(float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t
(** [psd ~rate ~win p] is the one-sided power spectral density, bin [k]
    [c_k p_k / (rate Σ win{^2})], in the signal's squared unit per cycle per
    unit of time. Its mean over frames is Welch's estimate.

    Raises [Invalid_argument] also if [rate] is not positive and finite. *)

val power_spectrum :
  ?n:int ->
  win:(float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t
(** [power_spectrum ~win p] is the one-sided power spectrum, bin [k]
    [c_k p_k / (Σ win){^2}]. A sinusoid of amplitude [A] at a bin's frequency
    reads [A{^2} / 2] there. *)

(** {1:banks Filterbanks}

    A {e filterbank} is a dense array [[k; b]] whose column [j] weighs [k]
    points, such as the bins of a one-sided spectrum. [Nx.matmul p bank] of
    power spectra [[…; k]] gives each frame's [b] band energies. A matrix
    product rounds as its kernel library's contraction does, which may depend on
    the number of frames in a call. So features computed from a stream of frames
    equal the whole call's within one rounding of the product, and may differ
    from it in the last bit. *)

(** The type for triangle heights. *)
type norm =
  | Unit_peak  (** Height [1] at the centre. *)
  | Unit_area  (** Area [1] along the axis the triangles lie on. *)

val triangles :
  norm ->
  at:(float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t
(** [triangles nm ~at e] is the [[k; b]] filterbank of [b] triangles at the [k]
    points [at], for [b + 2] edges [e]: band [j] at point [p] is
    [max 0 (min r f)] for the rise [r = (p - e(j)) / (e(j+1) - e(j))] and the
    fall [f = (e(j+2) - p) / (e(j+2) - e(j+1))], under [Unit_area] times
    [2 / (e(j+2) - e(j))]. Points and edges lie on one axis of the caller's
    choice, such as hertz; triangles linear in mel are [triangles] of mel points
    and mel edges ({!Mel.of_hz}). It is nx operations of [at] and [e],
    differentiable in both. Edges must increase strictly; a band whose edges do
    not divides by zero.

    Raises [Invalid_argument] if [at] or [e] is not 1-d or [e] has fewer than
    [3] elements. *)

(** Mel scales and mel filterbanks. A {e mel scale} maps a frequency in hertz to
    a pitch in mel, so its unit of time is the second. *)
module Mel : sig
  (** The type for mel scales. *)
  type scale =
    | Slaney
        (** [3f / 200] below 1000 Hz and [15 + 27 ln (f / 1000) / ln 6.4] from
            1000 Hz (Slaney, {e Auditory Toolbox}, version 2, Interval Research
            Corporation technical report 1998-010). *)
    | Htk  (** [1127 ln (1 + f / 700)], the HTK toolkit's. *)

  val of_hz : scale -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [of_hz sc f] is the pitch in mel of [f] hertz on [sc]. *)

  val to_hz : scale -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [to_hz sc m] is the frequency in hertz of [m] mel on [sc], the inverse of
      {!of_hz}. *)

  val edges :
    (float, 's) Nx.dtype ->
    scale ->
    low:float ->
    high:float ->
    int ->
    (float, 's, Nx.host) Nx.t
  (** [edges dt sc ~low ~high k] is [k] frequencies in hertz from [low] to
      [high], both included, evenly spaced on [sc].

      Raises [Invalid_argument] if [k < 2] or [0 ≤ low < high] fails with [high]
      finite. *)

  val bank :
    (float, 's) Nx.dtype ->
    scale ->
    norm ->
    rate:float ->
    n:int ->
    ?low:float ->
    ?high:float ->
    int ->
    (float, 's, Nx.host) Nx.t
  (** [bank dt sc nm ~rate ~n b] is the [[n / 2 + 1; b]] filterbank
      {!triangles}[ nm] at the bin frequencies [k rate / n] of {!Nx.rfft} of [n]
      samples at [rate] samples a second, on [edges sc ~low ~high (b + 2)].
      [low] defaults to [0.] and [high] to [rate /. 2.]. Whisper's banks are
      [bank dt Slaney Unit_area ~rate:16_000. ~n:400 b] for [b] of [80] and
      [128].

      Raises [Invalid_argument] if [rate] is not positive and finite, [n < 2],
      [b < 1], [0 ≤ low < high ≤ rate / 2] fails, or a band weighs no bin. *)
end

(** {1:fir_filters FIR filters} *)

val fir :
  ?axis:int ->
  ?edge:edge ->
  ?ends:int * int ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t
(** [fir ~edge ~ends h x] convolves [x] along [axis] with the [m] taps [h]: with
    [x̃] the signal extended by [lo] samples before it and [hi] after it as
    [edge] reads them ({!Nx.extend}), output [j] is [Σₖ hₖ x̃(j + m - 1 - k)],
    [n + lo + hi - m + 1] outputs or none. [edge] defaults to [`Zero] and
    [ends = (lo, hi)] to [(m - 1, 0)]: the causal filter from rest,
    [yⱼ = Σₖ hₖ x(j - k)], [n] outputs. [((m - 1) / 2, m / 2)] keeps [n] outputs
    centred on an odd [h].

    Each output is one reduction of its [m] products, as {!Nx.conv} of one
    channel computes it, so its bits depend on its window and [h] alone. Floats
    and complex numbers; differentiable in [h] and [x].

    Raises [Invalid_argument] if [h] is not 1-d or is empty, an end is negative,
    [axis] is not an axis of [x], or as {!Frames.frames} does when [x] has no
    sample. *)

(** Streams of {!fir}: framing's stream ({!Frames.feed}) with a window of [m]
    and a hop of [1], each window reduced against [h]. Every call takes the same
    [axis], [edge], [ends] and [h]. The outputs of every split, joined, equal
    {!fir}'s bit for bit. *)
module Fir : sig
  module State : Nx.Ptree.S
  (** The state between chunks: the samples later outputs read. *)

  val start : ('a, 'd) State.t
  (** [start] is the state before any sample. *)

  val feed :
    ?axis:int ->
    ?edge:edge ->
    ?ends:int * int ->
    ('v, 's, 'd) Nx.t ->
    (('v, 's, 'd) Nx.t, 'd) State.t ->
    ('v, 's, 'd) Nx.t ->
    (('v, 's, 'd) Nx.t, 'd) State.t * ('v, 's, 'd) Nx.t
  (** [feed h s x] is the state after the chunk [x] and every output whose
      window [x] completes.

      Raises [Invalid_argument] as {!fir} and {!Frames.feed} do. *)

  val finish :
    ?axis:int ->
    ?edge:edge ->
    ?ends:int * int ->
    ('v, 's, 'd) Nx.t ->
    (('v, 's, 'd) Nx.t, 'd) State.t ->
    ('v, 's, 'd) Nx.t
  (** [finish h s] is the outputs that read past the end.

      Raises [Invalid_argument] as {!fir} and {!Frames.finish} do. *)
end

(** {1:iir IIR filters}

    A {e cascade} is a float array [[sections; 6]] whose row [s] is
    [b0 b1 b2 a0 a1 a2], the {e section}
    [H(z) = (b0 + b1 z{^-1} + b2 z{^-2}) / (a0 + a1 z{^-1} + a2 z{^-2})]. The
    cascade is the sections' product, section [0] first. A row is used divided
    by its [a0], rounded once into its dtype; an [a0] of zero gives infinities
    or NaN. A section computes the recursion
    [yₙ = b0 xₙ + b1 xₙ₋₁ + b2 xₙ₋₂ - a1 yₙ₋₁ - a2 yₙ₋₂] on those rows (direct
    form I), its numerator's three products added in that order. Its {e state}
    after a sample is its last two inputs and its last two outputs; a cascade's
    is its first section's last two inputs and each section's last two outputs,
    the initial conditions of what follows. Nothing reads a coefficient's value:
    an unstable section gives growing outputs ({!Sos.stable}).

    {2:iir_evaluation Evaluation}

    A section runs in {e blocks} of 4096 samples from the start of each {!sos}
    call or {!Sos.feed}, the last block shorter. The state the block before left
    enters a block through its first two samples, as the recursion defines them
    there. A block of fewer than three samples is then the recursion itself. A
    longer one is computed from rest as two first-order recursions over complex
    numbers, [(1 - p z{^-1}){^-1}] then [(1 - q z{^-1}){^-1}], each by
    {!Nx.associative_scan}: [p] is the root of [z{^2} + a1 z + a2] of the larger
    modulus, from [a1{^2} - 4 a2] rounded once, and [q = a2 / p], or [0] where
    [p] is. The blocks follow each other in order, by {!Rune.scan}. So chunks
    whose lengths, all but the last, are multiples of 4096 give one call's bits,
    eagerly on a kernel library whose results do not depend on the extent of a
    call, and under [Rune.jit] within rune's numerics; other splits agree within
    the rounding of one evaluation.

    The derivative with respect to the signal, the coefficients and the state is
    the transfer function's, [d(v / A) = (dv - dA y) / A] for the numerator's
    output [v] and the denominator [A]: one more recursion forward, and in
    reverse mode one recursion backward in time. This first derivative
    differentiates no pole, so it is regular where poles coincide. Higher
    derivatives differentiate the outputs the first one reads, which the
    evaluation computes through its poles: at an exact double pole they may be
    NaN.

    {2:iir_accuracy Accuracy}

    A sequential recursion's rounding grows with the section's impulse
    response's absolute sum against the absolute values of the terms each step
    adds. The blocked evaluation's grows as if [|a1|] in those terms were
    [|p| + |q|]: where the two differ, as for real poles near both [1] and [-1],
    it can be many times a sequential evaluation's. A section whose poles lie
    close to the unit circle near [z = 1] or [z = -1], a narrow band near zero
    or near half the rate, is ill-conditioned: rounding its coefficients to
    [float32] already changes its outputs, and its gradients far more. Design
    and run such filters, coefficients included, in [float64]. *)

val sos :
  ?axis:int ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t
(** [sos c x] is [x] filtered along [axis] by the cascade [c] from rest,
    {!Sos.feed}'s output from {!Sos.start}. Of [x]'s shape.

    Raises [Invalid_argument] if [c] is not [[sections; 6]] with a section at
    least, or if [axis] is not an axis of [x]. *)

(** Streams of {!sos}. A stream starts at rest or from a state {!Sos.steady}
    gives, and each {!Sos.feed} continues from the state the last left. A filter
    has no [finish]: its last feed gives every output. The outputs of a split
    whose chunks but the last are multiples of 4096 samples long are the whole
    call's, bit for bit ({!section-iir_evaluation}). *)
module Sos : sig
  module State : sig
    (** The type for a cascade's states. *)
    type ('a, 'd) t =
      | Rest  (** Every past sample zero. *)
      | At of { inputs : 'a; outputs : 'a }
          (** [inputs] is [[lead…; 2]], the first section's last input then the
              one before; [outputs] is [[lead…; sections; 2]], each section's
              last output then the one before; [lead…] is the signal's shape
              without its time axis. *)

    val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
    (** [walk c s] walks [inputs] then [outputs], nothing in [Rest]. *)
  end

  val start : ('a, 'd) State.t
  (** [start] is [Rest]. *)

  val steady :
    (float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t ->
    ((float, 's, 'd) Nx.t, 'd) State.t
  (** [steady c u] is the state of [c] under an input held at [u], [[lead…]],
      forever: inputs [u], and section [s]'s outputs [u] times
      [Π_{t ≤ s} (b0 + b1 + b2) / (a0 + a1 + a2)] of the rows [t] divided by
      their [a0] and rounded, as the recursion uses them, each sum added in that
      order. A section with a pole at [z = 1] gives infinities or NaN.
      [steady c (Nx.zeros …)] is the rest state as arrays, for a {!Rune.scan}'s
      [init].

      Raises [Invalid_argument] as {!sos} does on [c]. *)

  val feed :
    ?axis:int ->
    (float, 's, 'd) Nx.t ->
    ((float, 's, 'd) Nx.t, 'd) State.t ->
    (float, 's, 'd) Nx.t ->
    ((float, 's, 'd) Nx.t, 'd) State.t * (float, 's, 'd) Nx.t
  (** [feed c s x] is the state after the chunk [x] and [x] filtered along
      [axis] from [s]. [c] may change between feeds: the next output is then the
      recursion with the new rows on the samples the state holds. A section
      whose rows change at every sample is a {!Rune.scan} of feeds of one sample
      each, which costs one step of the recursion a sample; near the unit circle
      such a section is best run in [float64] (Yu et al., "Differentiable
      All-pole Filters for Time-varying Audio Systems", DAFx 2024, §5.1).

      Raises [Invalid_argument] as {!sos} does, or if [s]'s arrays do not have
      the shapes {!State.t} states for [c] and [x]. *)

  val stable : (float, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [stable c] is, for each row of [c], whether both poles of its section lie
      inside the unit circle: [|a2 / a0| < 1] and [|a1 / a0| < 1 + a2 / a0] of
      the row divided by [a0], the reflection-coefficient test of order two.
      [[sections]].

      Raises [Invalid_argument] as {!sos} does on [c]. *)
end

val zero_phase :
  ?axis:int ->
  ?edge:Nx.extension ->
  ?ends:int * int ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t
(** [zero_phase c x] is [x] filtered along [axis] by the cascade [c] forward
    then backward, a filter of gain [|H|{^2}] and zero phase: [x] extended by
    [ends] as [edge] reads it ({!Nx.extend}), filtered from {!Sos.steady} at its
    first sample, reversed, filtered from {!Sos.steady} at the reversal's first
    sample, reversed, and cut to [x]'s span. [edge] defaults to
    [`Antisymmetric_whole], [ends] to [(p, p)] with [p = 3 (2 S + 1)] for [S]
    sections. The result depends on both through the starts' transients, which
    decay into the signal as the cascade's impulse response does: for a cascade
    whose response outlasts the signal, a narrow band near zero, that reaches
    its middle.

    These are SciPy's [sosfiltfilt] defaults but one: SciPy takes
    [p = 3 (2 S + 1 - z)], [z] the lesser of the numbers of rows whose [b2] and
    whose [a2] are zero, which reads coefficients' values. An odd-order design
    has [z = 1], so SciPy's result differs from this one wherever those
    transients reach, the ends and, for a slow cascade, the middle too, unless
    [ends] is [(6 S, 6 S)]. A short signal is extended by as many folds as
    [ends] needs.

    Raises [Invalid_argument] as {!sos} does, if an end is negative, or if [x]
    has no sample and [edge] other than [`Zero] extends it. *)

(** {2:biquads Cookbook sections} *)

(** The sections of the W3C Audio EQ Cookbook (Bristow-Johnson,
    {{:https://www.w3.org/TR/audio-eq-cookbook/}Audio EQ Cookbook}), each the
    bilinear transform of an analog prototype prewarped at [freq]. Each takes
    arrays of one broadcast shape [[…]] and gives rows [[…; 6]], computed by nx
    operations and so differentiable in each parameter: a [[3]] array of
    frequencies gives a cascade of three sections. [freq] is in cycles per the
    rate's unit of time, [q] is the cookbook's Q and [gain] its dBgain. With
    [w = 2π freq / rate], [α = sin w / (2 q)] and [A = 10{^gain/40}], each gives
    [b0 b1 b2 a0 a1 a2] as below. Nothing checks a parameter: outside
    [0 < freq < rate / 2] and [q > 0] a row is the formula's value.

    For {!Biquad.notch} and {!Biquad.bandpass} the gain is half its peak power
    (-3 dB) at two frequencies [Δw] radians a sample apart with
    [tan (Δw / 2) = α]. A notch of -3 dB width [b] in cycles per the rate's unit
    of time so has [q = sin w / (2 tan (π b / rate))]; a quality factor defined
    as [freq / b] is another number for the same notch. *)
module Biquad : sig
  val lowpass :
    rate:float ->
    freq:(float, 's, 'd) Nx.t ->
    q:(float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t
  (** [lowpass ~rate ~freq ~q] is
      [(1 - cos w) / 2, 1 - cos w, (1 - cos w) / 2, 1 + α, -2 cos w, 1 - α]. *)

  val highpass :
    rate:float ->
    freq:(float, 's, 'd) Nx.t ->
    q:(float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t
  (** [highpass ~rate ~freq ~q] is
      [(1 + cos w) / 2, -(1 + cos w), (1 + cos w) / 2, 1 + α, -2 cos w, 1 - α].
  *)

  val bandpass :
    rate:float ->
    freq:(float, 's, 'd) Nx.t ->
    q:(float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t
  (** [bandpass ~rate ~freq ~q] is [α, 0, -α, 1 + α, -2 cos w, 1 - α], a peak
      gain of [1]. *)

  val notch :
    rate:float ->
    freq:(float, 's, 'd) Nx.t ->
    q:(float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t
  (** [notch ~rate ~freq ~q] is [1, -2 cos w, 1, 1 + α, -2 cos w, 1 - α]. *)

  val allpass :
    rate:float ->
    freq:(float, 's, 'd) Nx.t ->
    q:(float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t
  (** [allpass ~rate ~freq ~q] is
      [1 - α, -2 cos w, 1 + α, 1 + α, -2 cos w, 1 - α]. *)

  val peaking :
    rate:float ->
    freq:(float, 's, 'd) Nx.t ->
    q:(float, 's, 'd) Nx.t ->
    gain:(float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t
  (** [peaking ~rate ~freq ~q ~gain] is
      [1 + α A, -2 cos w, 1 - α A, 1 + α / A, -2 cos w, 1 - α / A]: a boost and
      a cut of one [freq] and [q] multiply to [1]. Its poles coincide where
      [q A = 1/2]. *)

  val low_shelf :
    rate:float ->
    freq:(float, 's, 'd) Nx.t ->
    q:(float, 's, 'd) Nx.t ->
    gain:(float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t
  (** [low_shelf ~rate ~freq ~q ~gain] is, with [r = 2 √A α]:
      [A ((A + 1) - (A - 1) cos w + r)], [2 A ((A - 1) - (A + 1) cos w)],
      [A ((A + 1) - (A - 1) cos w - r)], [(A + 1) + (A - 1) cos w + r],
      [-2 ((A - 1) + (A + 1) cos w)], [(A + 1) + (A - 1) cos w - r]. *)

  val high_shelf :
    rate:float ->
    freq:(float, 's, 'd) Nx.t ->
    q:(float, 's, 'd) Nx.t ->
    gain:(float, 's, 'd) Nx.t ->
    (float, 's, 'd) Nx.t
  (** [high_shelf ~rate ~freq ~q ~gain] is, with [r = 2 √A α]:
      [A ((A + 1) + (A - 1) cos w + r)], [-2 A ((A - 1) + (A + 1) cos w)],
      [A ((A + 1) + (A - 1) cos w - r)], [(A + 1) - (A - 1) cos w + r],
      [2 ((A - 1) - (A + 1) cos w)], [(A + 1) - (A - 1) cos w - r]. *)
end

(** {2:designs Classical designs} *)

(** Classical designs, computed at float64 on the host by nx operations and
    rounded once into the dtype asked, as a window is. Frequencies are in cycles
    per the rate's unit of time and lie in [(0, rate / 2)].

    A design takes its family's analog prototype, of poles and zeros normalised
    at [1] rad per unit of time, moves it to the band whose edges are prewarped
    to [(rate / π) tan (π f / rate)], and maps it by the bilinear transform
    [s = 2 rate (1 - z{^-1}) / (1 + z{^-1})], so that each edge falls where the
    family places it. The transform keeps each prototype's gain and phase and
    moves them in frequency: the response at [f] is the prototype's at
    [Ω = 2 rate tan (π f / rate)] radians per unit of time, and its group delay,
    in units of time, the prototype's there times [1 + tan{^2} (π f / rate)]. A
    delay flat in the prototype, as [Bessel]'s, so gains a relative error of
    about [(π f / rate){^2}] at [f], whatever the order. The poles and zeros
    then pair into sections as SciPy's [zpk2sos] pairs them by default: the pole
    nearest the unit circle goes in the last section, with its conjugate or the
    nearest real pole, and with the zeros nearest it; the next nearest in the
    section before. The gain is spread over the [S] sections: each numerator is
    multiplied by the [S]-th root of its magnitude, the first by its sign too,
    so that a design of high order or narrow band, whose gain underflows
    float64, has no row of zeros. Every pole of a design lies inside the unit
    circle. Near [z = ±1] see {!section-iir_accuracy}.

    Zeros and poles a caller holds enter through two functions that share this
    pairing: {!Iir.analog} takes an analog filter's, through the transform, and
    {!Iir.digital} a digital filter's, as an all-pole model or a published
    design gives them. A cascade is the one representation of an IIR filter;
    zeros and poles are arguments only. *)
module Iir : sig
  (** The type for bands, by their edges: frequencies for {!design}, transitions
      for {!meet}. *)
  type 'e band =
    | Lowpass of 'e
    | Highpass of 'e
    | Bandpass of 'e * 'e
    | Bandstop of 'e * 'e

  type transition = { pass : float; stop : float }
  (** The type for a transition between bands: the pass band's edge and the stop
      band's edge beside it. *)

  (** The type for families. [ripple] is the pass band's ripple and [reject] the
      stop band's floor below the pass band, both in decibels. *)
  type family =
    | Butterworth  (** Maximally flat; its edges are its -3 dB points. *)
    | Chebyshev1 of { ripple : float }
        (** Equiripple in the pass band, whose edges end it. *)
    | Chebyshev2 of { reject : float }
        (** Equiripple in the stop band, whose edges start it. *)
    | Elliptic of { ripple : float; reject : float }
        (** Equiripple in both bands, the pass band's edges ending it: the least
            order for a given transition (Orfanidis, "Lecture Notes on Elliptic
            Filter Design", Rutgers University, 2006). Its prototype's Jacobi
            elliptic functions are computed by Landen's transformations, each
            modulus carried with its complement. *)
    | Bessel
        (** Maximally flat group delay at zero frequency (Thomson, "Delay
            Networks having Maximally Flat Frequency Characteristics", Proc. IEE
            Part III 96(44), 1949), its edges its -3 dB points. The prototype's
            poles are the reciprocals of the zeros of the Bessel polynomial of
            the order, scaled so that its gain at [1] is [1 / √2]. A low pass's
            delay at zero frequency is [ω₃ / Ω] for its edge's [Ω], where [ω₃],
            the -3 dB frequency of the prototype whose delay at zero frequency
            is [1], grows with the order as about [√((2N - 1) ln 2)]. Its step
            response barely overshoots for an edge well below the rate, and many
            times as much near a fifth of the rate. *)

  val design :
    (float, 's) Nx.dtype ->
    rate:float ->
    order:int ->
    family ->
    float band ->
    (float, 's, Nx.host) Nx.t
  (** [design dt ~rate ~order family band] is the cascade of [family]'s
      prototype of [order] moved to [band]: [⌈order / 2⌉] sections for a low or
      high pass, [order] for a band pass or stop. An odd order's real pole
      shares a section with the pole at [0] that pairing adds.

      Raises [Invalid_argument] if [rate] is not positive and finite,
      [order < 1], an edge is outside [(0, rate / 2)], a band's edges do not
      increase, or a ripple or [reject] is not positive and finite; [Failure] if
      the zeros it computes for [Bessel] fail the identities of the polynomial's
      coefficients [Σ zₖ = -1] and [Σ 1 / zₖ = -N (N + 1) / 2]. *)

  val meet :
    rate:float ->
    ripple:float ->
    reject:float ->
    [ `Butterworth | `Chebyshev1 | `Chebyshev2 | `Elliptic ] ->
    transition band ->
    int * family * float band
  (** [meet ~rate ~ripple ~reject kind spec] is [(order, family, band)], the
      least order of [kind] whose {!design} keeps its gain within [ripple] dB
      below its peak over the pass band and at least [reject] dB below it over
      the stop band, the bands bounded by [spec]'s transitions and by [0] and
      [rate / 2] (Orfanidis, "Lecture Notes on Elliptic Filter Design", 2006).
      The pass edges are met exactly, a band stop's at its stricter side, and
      the excess goes to the stop band; [band] is where [kind] places its edges
      to do so. This holds in exact arithmetic. [design Nx.float64] holds it to
      within what rounding its rows moves the gain, which grows as poles near
      [z = ±1]. Rounded into float32, such rows can miss the specification by
      decibels or put a pole on the unit circle; design them at float64 and
      check {!Sos.stable}.

      Raises [Invalid_argument] if [rate] is not positive and finite, an edge is
      outside [(0, rate / 2)], a transition's pass edge is not on its pass
      band's side or a band pass's or band stop's lower transition is not below
      its upper, or [ripple] or [reject] is not positive and finite or [reject]
      is not above [ripple]. *)

  val digital :
    (float, 's) Nx.dtype ->
    zeros:Complex.t list ->
    poles:Complex.t list ->
    gain:float ->
    (float, 's, Nx.host) Nx.t
  (** [digital dt ~zeros ~poles ~gain] is the cascade of
      [gain Π (1 - zᵢ z{^-1}) / Π (1 - pⱼ z{^-1})], paired as {!design} pairs.
      The shorter list is padded with values at [0], and an odd count once more
      in each, so a value at [0] adds no delay; the cascade has
      [⌈max nz np / 2⌉] sections, one section [gain 0 0 1 0 0] when both lists
      are empty. Poles may lie anywhere: one on or outside the unit circle gives
      a row {!Sos.stable} reports.

      Raises [Invalid_argument] if a complex value lacks its conjugate, within
      100 units in the last place. *)

  val analog :
    (float, 's) Nx.dtype ->
    rate:float ->
    zeros:Complex.t list ->
    poles:Complex.t list ->
    gain:float ->
    (float, 's, Nx.host) Nx.t
  (** [analog dt ~rate ~zeros ~poles ~gain] is the cascade of the analog filter
      [gain Π (s - zᵢ) / Π (s - pⱼ)] by the bilinear transform without
      prewarping, paired as {!design} pairs. [s] is the Laplace variable, in
      reciprocal units of time. Frequency [f] lies at [s = 2πi f], and a real
      pole [-1 / τ] decays with time constant [τ]. A zero at infinity, one per
      pole beyond the zeros, maps to [z = -1]. The transform maps
      [s = 2i rate tan (π f / rate)] to frequency [f], so a caller that needs
      the response at [f] exact uses that value in place of [2πi f]. [analog] is
      {!digital} of the transform's zeros, poles and gain. It refuses a pole
      whose real part is not negative so that, as for {!design}, every pole of
      the result lies inside the unit circle.

      Raises [Invalid_argument] if [rate] is not positive and finite, there is
      no pole, there are more zeros than poles, a pole's real part is not
      negative, a zero lies at [s = 2 rate], which the transform maps to
      [z = ∞], or a complex value lacks its conjugate. *)
end

(** {1:fft_convolution FFT convolution} *)

val fft_convolve :
  axes:int list ->
  ends:(int * int) list ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t
(** [fft_convolve ~axes ~ends x k] is {!fir}'s convolution along every axis of
    [axes] at once, [x] extended with zeros by [ends], a pair per axis, computed
    by Fourier transforms. [k] has [x]'s rank: along [axes] its extents [m] are
    the kernel's, along the other axes [1] or [x]'s, a kernel per slice. Along
    an axis of [axes] the result has [n + lo + hi - m + 1] outputs or none.
    Floats use real transforms, complex numbers complex ones.

    Along each axis of [axes] it computes by overlap-save, from output [0], in
    blocks of [N - m + 1] outputs, each the last [N - m + 1] points of the
    circular convolution of [N] points of the extended signal with [k], the last
    block's points past the signal zeros. [N] is the least power of two of at
    least [m] points that minimises [N log₂ N / (N - m + 1)], the transforms'
    work per output. [N] depends on [m] alone, so a block's bits depend on its
    points of [x], on [k] and on the shapes along [axes], on a kernel library
    whose transforms do not depend on their batch, and the whole call equals its
    stream ({!Fft_convolve}) bit for bit. The price is the block: a short signal
    against a long kernel still transforms blocks of about ten times the kernel
    along each axis, where {!fir} costs less.

    The error is normwise, unlike {!fir}'s. Barring overflow and underflow, to
    first order, each slice's result is within
    [(2√2 ε + √2 ε' + 4) u √(c N') ‖x‖₂ ‖k‖₂] of the exact convolution of
    floats, and [(2 ε + ε' + 2√2) u √(c N') ‖x‖₂ ‖k‖₂] of complex numbers:
    [‖x‖₂] and [‖k‖₂] are the slice's norms, [N'] the points of a block, the
    product of the [N]s, [ε] and [ε'] the constants of the forward and inverse
    transforms of a block in the bound [‖ŷ - y‖₂ ≤ ε u √N' ‖x‖₂] the kernel
    library states, [c] the product over [axes] of [⌈N / (N - m + 1)⌉], the
    blocks that read one point at most, and [u] the unit roundoff. An output
    much smaller than [‖x‖₂ ‖k‖₂] may lose every digit, and a NaN or an infinity
    in a block's points of [x], or in [k], reaches every output of that block.
    Differentiable in [x] and [k].

    Raises [Invalid_argument] if [axes] repeats or names an axis [x] lacks,
    [ends] does not have a pair per axis or has a negative end, [k] does not
    have [x]'s rank, an extent of at least [1] along [axes] and [1] or [x]'s
    along the others, or the dtype is neither float nor complex. *)

(** The stream of {!fft_convolve} along one of its axes, [axis]; the other axes
    are whole in every chunk. A feed gives every block whose points it
    completes, [finish] the blocks that read past the end. Every call takes the
    same [axis], [axes], [ends] and [k], and the outputs of every split, joined,
    equal {!fft_convolve}'s bit for bit. *)
module Fft_convolve : sig
  module State : Nx.Ptree.S
  (** The state between chunks: the points later blocks read. *)

  val start : ('a, 'd) State.t
  (** [start] is the state before any chunk. *)

  val feed :
    axis:int ->
    axes:int list ->
    ends:(int * int) list ->
    ('v, 's, 'd) Nx.t ->
    (('v, 's, 'd) Nx.t, 'd) State.t ->
    ('v, 's, 'd) Nx.t ->
    (('v, 's, 'd) Nx.t, 'd) State.t * ('v, 's, 'd) Nx.t
  (** [feed ~axis ~axes ~ends k s x] is the state after the chunk [x] and the
      outputs of every block it completes along [axis].

      Raises [Invalid_argument] as {!fft_convolve} does, if [axis] is not among
      [axes], or if [x]'s other axes are not those of the chunks before it. *)

  val finish :
    axis:int ->
    axes:int list ->
    ends:(int * int) list ->
    ('v, 's, 'd) Nx.t ->
    (('v, 's, 'd) Nx.t, 'd) State.t ->
    ('v, 's, 'd) Nx.t
  (** [finish ~axis ~axes ~ends k s] is the outputs of the blocks that read past
      the end.

      Raises [Invalid_argument] as {!feed} does, or if no chunk was fed (a
      stream that may end before its first sample feeds an empty chunk first).
  *)
end

(** {1:kernels Kernels}

    A {e kernel} [k] is a function of a position [τ] in samples of its
    {e frame}, zero outside [\[-h, h)], [h] its {e half-width}. FIR design
    samples it at the integers; resampling samples it at each output's offsets,
    its frame the lower of the two rates. A kernel is plain data: two equal
    kernels are the same kernel. {!Kernel.eval} computes its values by nx
    operations, at float64 for {!Kernel.taps} and resampling's weights, as a
    window's are. *)
module Kernel : sig
  (** The type for the windows of a windowed sinc of half-width [h]. *)
  type window =
    | Kaiser of float
        (** [I₀(β √(1 - (τ/h){^2})) / I₀(β)] for the given [β], where [I₀] is
            the modified Bessel function of the first kind of order 0. It is
            evaluated as its power series [Σₖ cₖ q{^k}] in [q = 1 - (τ/h){^2}],
            [cₖ = (β{^2}/4){^k} / ((k!){^2} I₀(β))] computed at float64, ending
            before the first term past the largest whose [cₖ] is below float64's
            unit roundoff: a polynomial in [τ], smooth at every order. *)
    | Hann  (** [cos{^2}(π τ / 2h)]. *)
    | Lanczos  (** [sinc (τ / h)]. *)

  (** The type for kernels. *)
  type t =
    | Box  (** [1] on [\[-½, ½)]. *)
    | Triangle  (** [1 - |τ|] on [\[-1, 1)]. *)
    | Cubic of float
        (** Keys' cubic of parameter [a] on [\[-2, 2)]:
            [(a + 2)|τ|{^3} - (a + 3)|τ|{^2} + 1] to [|τ| = 1], then
            [a|τ|{^3} - 5a|τ|{^2} + 8a|τ| - 4a]. *)
    | Sinc of { cutoff : float; half_width : float; window : window }
        (** [2c sinc(2cτ) w(τ)] on [\[-h, h)], [c] the [cutoff] in cycles per
            sample of the frame, [h] the [half_width], [w] the [window] and
            [sinc x = sin(πx) / (πx)]. *)

  val half_width : t -> float
  (** [half_width k] is [k]'s [h]. *)

  val eval : t -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
  (** [eval k τ] is [k] at each [τ], [0] outside [\[-h, h)]: nx operations in
      [τ]'s dtype, where [τ] lies. It is differentiable in [τ]; where [k] has a
      kink or a jump, as [Triangle] at [0] and [Box] or a [Sinc] at [-h], the
      derivative is the one from the right. *)

  val taps : (float, 's) Nx.dtype -> t -> (float, 's, Nx.host) Nx.t
  (** [taps dt k] is [k] at the integers in [\[-h, h)], and at [h] too for a
      [Sinc] of an integer [h], scaled to sum [1] and rounded once into [dt]: a
      linear-phase low pass for {!fir} when [k] is even. A high pass is the unit
      impulse less a low pass of odd length; a band pass the difference of two
      low passes of one length. Each low pass sums to [1], so such a high or
      band pass has gain [0] at zero frequency, to rounding, and elsewhere the
      difference of its terms' gains; nothing rescales it. *)

  val lowpass : rate:float -> pass:float -> stop:float -> reject:float -> t
  (** [lowpass ~rate ~pass ~stop ~reject] is a Kaiser-windowed [Sinc] of cutoff
      [(pass + stop) / (2 rate)] whose {!taps} keep their gain within
      [δ = 10{^-reject/20}] of [1] up to [pass] and of [0] from [stop],
      frequencies in cycles per the rate's unit of time. [β] is Kaiser's:
      [0.1102 (A - 8.7)] above 50 dB, [0.5842 (A - 21){^0.4} + 0.07886 (A - 21)]
      from 21 dB, [0] below, for [A = reject]. Its half-width is the least
      integer [h] that meets the specification from Kaiser's estimate
      [⌈(A - 7.95) / (4.57 Δω)⌉] up, [Δω = 2π (stop - pass) / rate]. The
      specification is certified, not sampled: on a grid of at least 64 points a
      tap and at the band edges, the gain [A] and its first two derivatives
      bound the error [e], [|A - 1|] in the pass band and [|A|] in the stop
      band, between neighbouring points by Taylor's theorem,
      [e(p) + |A'(p)| s + |A''(p)| s{^2} / 2 + M₃ s{^3} / 6] within [s] of a
      point [p], [M₃ = 8π{^3} Σ |tⱼ| |j|{^3}] over the taps [tⱼ] at distance [j]
      from the centre, with an allowance for rounding; an interval the bound
      does not clear is halved, up to 40 times, and a half-width whose bands it
      cannot clear is lengthened. Memory is linear in the taps.

      The certificate covers [taps Nx.float64 k]. Rounding the taps into a
      narrower dtype moves the gain by up to that dtype's unit roundoff times
      [Σ |tⱼ|], which in float32 passes [δ] at high rejection.

      Raises [Invalid_argument] unless [rate] is positive and finite,
      [0 < pass < stop ≤ rate / 2] and [reject] is positive and finite, or if no
      half-width up to four times the estimate meets the specification, as at a
      [reject] past float64's precision. *)
end

(** {1:tap_designs Tap designs}

    A {e linear-phase} filter of [m] taps is {e symmetric}, tap [j] equal to tap
    [m - 1 - j], or {e antisymmetric}, tap [j] its negative. Its response at [f]
    is [e{^-iπ f (m - 1) / rate}] times a real {e amplitude} [A(f)], times [i]
    if antisymmetric. The exponential is a delay of [(m - 1) / 2] samples, whole
    for an odd [m]. An antisymmetric filter's amplitude is [0] at zero frequency
    and, for an odd [m], at [rate / 2]; a symmetric filter's is [0] at
    [rate / 2] for an even [m]. A differentiator (amplitude [2π f]) and a
    Hilbert transformer (amplitude [1]) are antisymmetric.

    A design approximates on each of its {e bands} a target amplitude linear
    from [fst gain] at [lo] to [snd gain] at [hi], frequencies in cycles per the
    rate's unit of time, and measures the error there in units of the band's
    tolerance, linear from [fst tol] at [lo] to [snd tol] at [hi]. Between bands
    the amplitude is free. Bands lie in [[0, rate / 2]], each with [lo ≤ hi] and
    both tolerances positive and finite, in increasing order without overlap.
    Designs compute at float64 on the host and round once into the dtype asked;
    their taps are never rescaled. Each design from bands raises
    [Invalid_argument] if [rate] is not positive and finite, the bands are empty
    or break these rules, or a band asks a gain farther than its tolerance from
    [0] at a frequency where the symmetry and the length force [0]. *)
module Taps : sig
  (** The type for the symmetries of linear-phase taps. *)
  type symmetry = Symmetric | Antisymmetric

  type band = {
    lo : float;
    hi : float;
    gain : float * float;
    tol : float * float;
  }
  (** The type for bands. A pass band of gain 1 is
      [{ lo; hi; gain = (1., 1.); tol = (t, t) }], a stop band has [(0., 0.)], a
      differentiator per unit of time [(2π lo, 2π hi)], and a relative error [r]
      of it, for [lo > 0], the tolerance [(2π r lo, 2π r hi)]. *)

  val least_squares :
    (float, 's) Nx.dtype ->
    rate:float ->
    symmetry ->
    band list ->
    int ->
    (float, 's, Nx.host) Nx.t
  (** [least_squares dt ~rate s bands m] is [m] taps of symmetry [s] that
      minimise the sum over the bands of the integral of the squared error in
      units of the tolerance; their error is smallest on average and largest
      near the band edges. The integrals are exact for a band's constant target
      and tolerance, and accurate to their quadrature's convergence for sloped
      ones (Da Fies and Vianello, "Trigonometric Gaussian quadrature on
      subintervals of the period", ETNA 39, 2012). The result's integral exceeds
      the least by a rounding error relative to the solution's size. Where the
      bands leave the taps underdetermined, as a lone band does at many taps,
      the taps are one such minimiser.

      Raises [Invalid_argument] as the section says, if [m < 1], or if a band
      has [lo = hi]. *)

  val equiripple :
    (float, 's) Nx.dtype ->
    rate:float ->
    symmetry ->
    band list ->
    int ->
    (float, 's, Nx.host) Nx.t
  (** [equiripple dt ~rate s bands m] is [m] taps of symmetry [s] designed for
      the least largest error over the bands, in units of the tolerance, Parks
      and McClellan's minimax design, which Remez' exchange computes as Filip
      gives it ("A robust and scalable implementation of the Parks-McClellan
      algorithm for designing FIR filters", ACM TOMS 43(1), 2016). The least
      error's taps have an error of one magnitude, alternating in sign, at one
      more frequency than the taps have independent values. On the design's
      final reference the result's own error alternates in sign, so by de la
      Vallée Poussin's theorem its least magnitude there, [δ], is at most the
      least error. The result's largest error [E], found by a search apart from
      the reference, has [(E - δ) / E ≤ 0.01], Filip's convergence test (§7.1),
      so it is within 1% of the least possible. {!meet} certifies a length.

      Raises [Invalid_argument] as the section says, or if [m < 1]; [Failure] if
      the bands hold fewer frequencies, besides those where the taps' gain is
      forced to [0], than one more than the taps have independent values (as
      single-frequency bands can), if on the final reference the result's error
      does not alternate in sign at increasing frequencies, or if [E] exceeds
      [δ] by more than 1% of [E], as when a reference is too ill-conditioned for
      float64. *)

  val meet : rate:float -> odd:bool -> symmetry -> band list -> int
  (** [meet ~rate ~odd s bands] is the least length [m], odd if [odd] and even
      otherwise, at which [equiripple Nx.float64 ~rate s bands m] keeps each
      band's error within its tolerance at every frequency of the band. A length
      is accepted when the bound {!Kernel.lowpass} states clears every band on
      its taps, the target's slope entering its first-derivative term, the
      tolerance taken at its least over each interval, and the taps' distances
      from the centre, whole or half, its [M₃]. It is the least when the final
      reference of [m - 2]'s design has an alternating error above the tolerance
      by more than the rounding allowance, which proves that no taps of that
      length meet the bands.

      Raises [Invalid_argument] as {!equiripple} does, or once the bound's
      rounding allowance alone reaches a band's tolerance, past which no length
      can be certified, or if a band's target is its tolerance away from [0]
      where the taps' gain is forced to [0], which no length can be certified to
      meet; [Failure] as {!equiripple} does at a length, or naming a length that
      is neither certified nor proved too short. *)

  val minimum_phase :
    (float, 's) Nx.dtype ->
    length:int ->
    (float, 'r, Nx.host) Nx.t ->
    (float, 's, Nx.host) Nx.t
  (** [minimum_phase dt ~length g] is [length] taps of a near minimum-phase
      filter whose gain approximates [g], given at [k rate / n] for the
      [n / 2 + 1] values of [g] on the grid of an [n]-point {!Nx.rfft}, [n]
      even, computed at float64 by the real cepstrum on [n] points (Smith,
      "Spectral Audio Signal Processing", Minimum Phase and Causal Cepstra).
      Given {!Nx.magnitude} of the transform of [m] taps [h] and [~length:m], it
      approximates [h] with each zero outside the unit circle moved to its
      conjugate reciprocal. A gain below [u] times the largest, [u] float64's
      unit roundoff, is taken as that.

      On the [n] points the untruncated result's gain is [g]. Between them it
      differs by an error absolute in the gain that falls as [n] grows,
      geometrically for zeros off the circle and slowly for zeros on it or a
      gain of [0] over a band, where a small [n] can leave zeros outside the
      circle; check the result between the grid's points.

      Raises [Invalid_argument] if [length < 1] or [length > n], [g] is not 1-d
      with at least two values, a value of [g] is negative or not finite, or
      every value is [0]. *)

  val savitzky_golay :
    (float, 's) Nx.dtype ->
    window:int ->
    order:int ->
    deriv:int ->
    at:float ->
    (float, 's, Nx.host) Nx.t
  (** [savitzky_golay dt ~window ~order ~deriv ~at] is the [window] taps that
      give the [deriv]-th derivative, at position [at] of a window, of the
      polynomial of degree [order] fitted in least squares to the window's
      samples (Savitzky and Golay, "Smoothing and Differentiation of Data by
      Simplified Least Squares Procedures", Anal. Chem. 36(8), 1964), with
      respect to the sample index. {!fir} of them gives at output [j] that
      derivative read [at] samples after the first sample output [j] reads.
      Positions run from [0] to [window - 1], and a fractional [at] reads the
      fit between samples. Times [rate{^deriv}] it is per the rate's unit of
      time; a [deriv] above [order] gives zeros.

      Raises [Invalid_argument] if [order] or [deriv] is negative,
      [window <= order], or [at] is not finite. *)
end

(** {1:resampling Resampling}

    To {e resample} by [by = (l, m)] is to evaluate a band-limited
    reconstruction of [x] at [l] new samples for every [m] old ones; the ratio
    is reduced by its greatest common divisor, which changes no output. [n]
    samples give [⌈n l / m⌉]. Output [j] lies at input position [c_j], on one of
    two {e grids}: [`Points], sample [i] the instant [i], [c_j = j m / l];
    [`Cells], sample [i] the cell [\[i, i + 1)] seen at its centre,
    [c_j = (j + ½) m / l - ½]. With [s = max 1 (m / l)] and [k] the kernel of
    half-width [h],

    [y_j = Σᵢ x(i) k((c_j - i) / s) / s]

    over the [i] with [-s h ≤ c_j - i < s h]: when shrinking, the kernel
    stretches to the lower rate, which places its cutoff there. A sample outside
    [x] reads as the edge says ({!Nx.extend}); [`Renormalise] drops it, and
    scales every output's weights to sum [1].

    The weights are computed at float64 on the host from exact positions and
    rounded once into [x]'s dtype, beside [x]. Each output is one reduction of
    its products, as {!Nx.conv} of one channel computes it, or under
    [`Renormalise] where it reads past an end as {!Nx.sum} adds, so its bits
    depend on its inputs and its weights alone. The result is linear in [x],
    differentiable in it; the ratio, grid and kernel fix shapes and weights and
    have no derivative. Floats and complex numbers.

    {2:resample_stream Its stream}

    A feed emits each output once every input it reads is known, the extension's
    at the start included, in order; [finish] gives the outputs that read past
    the end. The outputs of every split, joined, equal {!resample}'s bit for
    bit. Past the first outputs, a chunk of [b m] samples gives [b l] outputs
    and leaves the state's shape unchanged, which {!Rune.scan} needs to carry
    it, and another length changes it, which a scan's trace refuses. The first
    feed differs: it moves the state from {!Resample.start}, which holds no
    array, to one that does, and gives fewer outputs, the first ones waiting for
    the samples past them that they read. So a stream inside a scan feeds its
    first chunk before the scan, whose [init] is that feed's state: at 48 kHz to
    16 kHz in 10 ms chunks, the first chunk of 480 samples gives fewer than 160
    outputs and every later one 160. *)

(** Resamplings and their streams. *)
module Resample : sig
  type t
  (** The type for resamplings: a reduced ratio, a grid, an edge, a kernel and
      their weights. *)

  val v :
    grid:[ `Points | `Cells ] ->
    edge:[ edge | `Renormalise ] ->
    kernel:Kernel.t ->
    by:int * int ->
    t
  (** [v ~grid ~edge ~kernel ~by] computes the weights of the [l] phases, [l]
      the reduced [by]'s first part, at most [⌈2 s h⌉] each at float64 and a
      second copy under [`Renormalise], so its memory and time grow with [l].

      Raises [Invalid_argument] unless both parts of [by] are positive. *)

  val count : t -> int -> int
  (** [count r n] is [⌈n l / m⌉], the outputs of [n] samples. *)

  val weights : t -> (int * float array) array
  (** [weights r] is, for each output [j < l], the offset of its first input
      from [⌊c_j⌋] and its weights at float64, as {!resample} applies them away
      from the ends; output [j + l] reads [m] inputs later with the same
      weights. *)

  module State : Nx.Ptree.S
  (** The state between chunks: the samples later outputs read and where the
      next output lies among them. *)

  val start : ('a, 'd) State.t
  (** [start] is the state before any sample. *)

  val feed :
    ?axis:int ->
    t ->
    (('v, 's, 'd) Nx.t, 'd) State.t ->
    ('v, 's, 'd) Nx.t ->
    (('v, 's, 'd) Nx.t, 'd) State.t * ('v, 's, 'd) Nx.t
  (** [feed r s x] is the state after the chunk [x] and the outputs it completes
      along [axis].

      Raises [Invalid_argument] if [axis] is not an axis of [x] or [x]'s other
      axes are not those of the chunks before it. *)

  val finish :
    ?axis:int -> t -> (('v, 's, 'd) Nx.t, 'd) State.t -> ('v, 's, 'd) Nx.t
  (** [finish r s] is the outputs that read past the end, up to {!count}.

      Raises [Invalid_argument] if no chunk was fed (a stream that may end
      before its first sample feeds an empty chunk first), or if the signal has
      no sample, an output reads past an end and the edge is an extension other
      than [`Zero]. *)
end

val resample : ?axis:int -> Resample.t -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
(** [resample r x] is [x] resampled along [axis]: {!Resample.start}, one
    {!Resample.feed}, {!Resample.finish}, joined.

    Raises [Invalid_argument] as {!Resample.feed} and {!Resample.finish} do. *)
