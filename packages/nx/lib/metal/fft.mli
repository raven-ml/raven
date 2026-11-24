(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The plans of the Fourier transforms.

    A line of [n] points runs as an FFT of [l1] points: [n] where [n] is
    7-smooth, its only prime factors 2, 3, 5 and 7; [n - 1] for Rader's
    convolution; [m] for Bluestein's. That FFT runs as {e steps}, each an FFT of
    at most {!most_points} points in one group of threads: one step where [l1]
    fits, else a step for each factor of a balanced split. A step's passes and
    their radices are a function of its points alone, so a line's arithmetic is
    a function of [n] and whether it is a halved real line, whatever its layout,
    batch or kernel.

    A plan holds no table and makes no pipeline: it names what the call makes
    when it runs. *)

(** {1:steps Steps} *)

val most_points : int
(** [most_points] is 4096, the most points of a step: a group's 32 KiB of
    threadgroup memory holds 4096 [complex64] points. *)

type step = private {
  l : int;  (** The points. *)
  radices : int array;  (** The passes' radices, in order. *)
  p : int;  (** The points a thread holds. *)
  tl : int;  (** The threads a line takes. *)
  threads : int;  (** [tl] in whole SIMD groups, at least 32. *)
}
(** The type for steps. *)

val step : int -> step
(** [step l] is the step of [l] points:
    - [radices]: an 8 for each three of [l]'s factors 2, one power of two for
      the 2s left, then its 7s, its 5s and its 3s, where [tl] at [p = 8] is then
      at most 256; else a 16 for each four of its factors 2, then the same. As
      [8 8 8 2] for 1024, [8 5 5 5] for 1000, [16 16 16] for 4096 and
      [16 16 3 3] for 2304;
    - [p]: 8 where its 2s are taken as 8s, else 16;
    - [tl]: the most over passes of radix [R] of [⌈(l / R) / ⌊p / R⌋⌉], so that
      each of a pass's butterflies has a thread, 1 for [l = 1];
    - [threads]: [tl] rounded up to 32, a SIMD group, at least 32. No step takes
      more than 288.

    Raises [Invalid_argument] if [l] is not 7-smooth or is past {!most_points}.
*)

val factors : int -> step array
(** [factors n] is the steps of a long line of [n] points: for [q] from 2 up,
    [n]'s primes dealt largest first, each to the least factor so far, the first
    [q] whose factors each fit a step, largest first. A factor takes at most
    [7 n{^1/q}] points, at most 1792 where [n{^1/q} <= 256], so
    [q <= ⌈log₂ n / 8⌉].

    Raises [Invalid_argument] if [n] is not 7-smooth or fits a step. *)

(** {1:groups Groups}

    A group holds {!group_memory} bytes of threadgroup memory, where a step
    exchanges its points between passes. *)

val group_memory : int
(** [group_memory] is {!Plan.group_memory}, 32 KiB. *)

val plane : step -> int
(** [plane s] is the bytes of a line's complex plane: 8 bytes a point, padded by
    one point in 16 where [s.l] is a power of two of at most 2048. *)

val rows : step -> int
(** [rows s] is the most lines a group of [s] takes where each line's points lie
    one element apart: as many as fill 256 threads, [⌊256 / s.tl⌋], fewer where
    their planes would pass {!group_memory}, and at least one. *)

val side_most : int
(** [side_most] is 2048, the most points of a step whose lines run side by side.
*)

val side_threads : int
(** [side_threads] is 512, the threads of a group whose lines run side by side.
*)

val side_step : int -> step
(** [side_step l] is the step lines of [l] points run side by side: [step l]
    where a group of 512 threads takes at least 3 of its lines, else [step l]'s
    radices at [p = 16], its [tl] and [threads] as {!step} gives them, with the
    same bits.

    Raises [Invalid_argument] if [l] is not 7-smooth or is past {!side_most}. *)

val side : int -> int
(** [side l] is the most lines a group takes side by side, for a step of [l]
    points: [min ⌊8192 / l⌋ ⌊512 / tl⌋], [tl] {!side_step}[ l]'s, so that their
    float planes, [4 l] bytes a line, fit {!group_memory}. It is at least 3 for
    every [l] up to {!side_most}.

    Raises [Invalid_argument] if [l] is not 7-smooth or is past {!side_most}. *)

val column_lines : int
(** [column_lines] is 16, the adjacent lines a group of a column step takes
    ({!c2c}): their same point, 128 bytes of [complex64], a cache line where the
    first starts on one. *)

val columns_stride_most : int
(** [columns_stride_most] is 8192, the most bytes between a line's consecutive
    points on a side of a column step ({!c2c}). The step's 128-byte segments of
    one point fall in rows that far apart, and at a few times this stride the
    column step runs slower than the transposes. It chooses between plans of the
    same bits, so it bounds speed alone. *)

(** {1:lines Lines} *)

(** The type for how a line's FFT runs. *)
type how =
  | Smooth  (** Its own points. *)
  | Rader  (** Rader's convolution, of [n - 1] points. *)
  | Bluestein of int  (** Bluestein's convolution, of this many points. *)

type line = private {
  n : int;  (** The points. *)
  real : bool;  (** Whether it is a halved real line. *)
  how : how;
  l1 : int;  (** The points of its FFT. *)
  steps : step array;  (** Its FFT's steps, in order. *)
}
(** The type for lines. *)

val line : int -> real:bool -> line
(** [line n ~real] is the plan of a line of [n] points, a halved real one's
    where [real]. It runs:
    - [Smooth] where [n] is 7-smooth;
    - [Rader] where [n] is a prime, [n - 1] is 7-smooth and fits a step, and the
      line is not [real];
    - otherwise [Bluestein m], [m] the 7-smooth length from [2n - 1] to
      [2 (2n - 1)] of the least cost, the least of equal cost. Its cost is [m]
      times the sum of its steps' radices' weights: 4 for 2, 7.3 for 3, 7 for 4,
      10.4 for 5, 12.8 for 7, 10 for 8 and 13.3 for 16. Costs compare as
      computed in floats, each the radices' weights summed step by step, pass by
      pass, then times [m]. A new [n] weighs each 7-smooth number of its window,
      838 near [2{^32}].

    The plans of the {!kept_lines} lines last planned are kept, on the OCaml
    heap, for every domain and device.

    Raises [Invalid_argument] if [n] is not positive. *)

val kept_lines : int
(** [kept_lines] is 64. *)

val bluestein : int -> int
(** [bluestein n] is the [m] a Bluestein line of [n] points takes, as {!line}
    states. It does not decrease as [n] grows: from [n] to [n + 1] the
    candidates lose their least and gain greater ones. *)

val least_takers : unit -> (int * int) list
(** [least_takers ()] is, for each [m] past {!most_points} that {!bluestein}
    gives some [n <= 2{^32}], [(m, n)] with [n] the least such, in increasing
    order. *)

(**/**)

val windows : int array -> int array
(** [windows ns] is [Array.map bluestein ns], from one sweep of the 7-smooth
    numbers, for [ns] positive and increasing. For tests. *)

(**/**)

(** {1:calls Calls}

    A call transforms along each of its axes, the last first: the first from the
    operand into [dst], each later one in place on [dst]. Along an axis, each
    launch moves lines of points from one {e side}, where it reads, to another,
    where it writes: the operand, [dst] or the workspace. How a step moves its
    points depends on its two sides alone; its passes stay its points'.

    A plan holds no direction. Every table is the forward transform's, and an
    inverse runs each launch's forward operation with each point the launch
    loads and each it stores conjugated, a negation of its imaginary part, but
    in a [Transpose] that copies bits: [Plain], from [Complex] to [Complex] or
    [Real], or from [Real] to [Real]. Conjugating twice is exact, so an inverse
    c2c computes the conjugate of the forward transform of its conjugated
    operand, bit for bit, the signs of zeros included. For odd [n], a c2r equals
    the real parts of the inverse c2c of its completed bins,
    [X{_ n-k} = conj X{_ k}], with bin 0's imaginary part [-0], bit for bit; for
    even [n], it is computed on a halved line, and the signs of its zeros are
    that computation's: its [Prepass] forms [Z'{_ k}], the conjugate of the
    points whose inverse FFT is the c2r, from the conjugated bins, and the
    halved line's forward FFT of [Z'] conjugated holds the result's pairs. *)

(** The type for how a side lays out its lines, by its strides. A side is
    {e wide} where a line's points span past [2{^31} - 1] elements, past what a
    step's 32-bit point offset reaches, or past a test's bound ({!bound_wide}).
*)
type access =
  | Rows  (** A line's points lie one element apart, either way. *)
  | Columns
      (** Lines next to each other in the batch's innermost axis lie one element
          apart, either way, so the same point of adjacent lines is adjacent. A
          batch of one line is never [Columns]. *)
  | Scattered  (** Neither. *)

(** The type for the buffers a side lies in. *)
type buffer =
  | Operand
  | Dst
  | Workspace of int  (** The workspace, from this byte. *)

type side = private {
  buffer : buffer;
  first : int;  (** The position of line 0's point 0, in elements. *)
  ps : int;  (** The stride between a line's points, in elements. *)
  dims : int array;
      (** The lines' extents, in C order, the batch's axes merged where both
          sides lay them out as one run and extents of 1 dropped; one extent at
          least. *)
  strides : int array;  (** The stride between lines along each of [dims]. *)
  points : int;  (** The points of a line. *)
  width : int;
      (** The bytes of an element: 4 for a [float32] line's, 8 for a [complex64]
          one's. *)
  base : int;
      (** The byte of a {!Plan.line} at which its buffer's first byte lies, [β]:
          the operand's or [dst]'s as the call states it, [0] for the workspace.
      *)
  access : access;
}
(** The type for sides. Line [(i{_0}, …)] of a side starts at position
    [first + Σ i{_k} strides.(k)], exactly, past [2{^32}] elements too. *)

(** The type for how a step's group takes its lines. *)
type path =
  | Contiguous
      (** {!group_lines} lines a group, each line's points read and written in
          order, in their [lines × tl] threads in whole SIMD groups, at least
          32. *)
  | Side_by_side
      (** {!group_lines} lines a group, their planes interleaved, so that a
          [Columns] side moves the same point of all of them at once: in
          {!side_threads} threads where the group takes {!side} lines, else in
          their [lines × tl] threads in whole SIMD groups. *)

val group_lines : path:path -> step -> int -> int
(** [group_lines ~path s n] is the lines a group of [s] takes on [path] for a
    side of [n >= 1] lines: {!rows}[ s] or {!side}[ s.l], or the least power of
    two at least [n] where that is fewer, so that a call of few lines runs no
    empty slots. No bit depends on it. *)

(** The type for the parts of a column step ({!c2c}): the step of 4096 points in
    two launches through the workspace, each group on {!column_lines} adjacent
    lines. The step's passes are 0, 1 and 2, its plane positions [x < 4096]
    (kernels.h). *)
type part =
  | First_pass
      (** Pass 0: for each [t < 256], the DFT of 16 of the points [t + 256 r],
          [r < 16], each value [r] written to point [256 r + t] of the
          workspace's line, which the step's pass 0 writes to position
          [16 t + r]. *)
  | Last_passes
      (** Passes 1 and 2, which never mix positions of different value mod 16:
          from point [256 m + p] of the workspace's line as position [16 p + m],
          the step's operations on each class [m < 16], written as the step
          writes its result. *)

(** The type for what a transpose reads, a point at a time. *)
type input =
  | Complex  (** A [complex64] point, 8 bytes, or a pair of floats. *)
  | Real  (** One float, its imaginary part [+0]. *)
  | Bins
      (** A c2r's [⌊n / 2⌋ + 1] bins, completed to [n] points by
          [X{_ n-k} = conj X{_ k}], [0 < k <= ⌊n / 2⌋], bin 0's imaginary part
          read as [+0]. *)

(** The type for what a transpose multiplies its points by, between reading and
    writing them. A product [z w] is
    [(fma z.re w.re (-(z.im w.im)), fma z.re w.im (z.im w.re))] in [float32],
    each operation rounded once to nearest. Each reads a subnormal operand as a
    zero of its sign, and gives a zero of the exact result's sign where that
    result is nonzero and below [2{^-126}], the least normal number, in
    magnitude. *)
type scale =
  | Plain  (** Nothing. *)
  | Twiddle  (** [w{_ ab}{^ rc}] at row [r], column [c] of [a] rows of [b]. *)
  | Chirp_in
      (** Point [j < n] by Bluestein's [chirp{_ j}], zeros for [n <= j < m]. *)
  | Kernel
      (** Bin [k] by the kernel's spectrum, divided by [m], then the conjugate.
      *)
  | Chirp_out
      (** The conjugate, then [chirp{_ k}]; only the first [n] points are
          written. *)

(** The type for what a transpose writes. *)
type output =
  | Complex  (** Each point, 8 bytes, or a pair of floats. *)
  | Half  (** An r2c's bins [k <= ⌊n / 2⌋] alone. *)
  | Real  (** A c2r's real parts alone, a float each. *)

(** The type for what a launch does. *)
type work =
  | Step of {
      line : line;  (** The line whose FFT the step is part of. *)
      step : step;
          (** The step it runs: {!side_step}'s of its points side by side. *)
      path : path;
      lines : int;  (** The lines a group takes. *)
    }
  | Column_step of { line : line; step : step; part : part }
      (** A part of a column step of [line], whose only step is [step]. Its
          [First_pass] followed by its [Last_passes] writes each line the bits
          the step's kernel writes from the same points. *)
  | Transpose of {
      line : line;  (** The line it moves the points of. *)
      split : int * int;
          (** [(a, b)]: each run of [a b] points of a line, [a] rows of [b]
              points, is written as [b] rows of [a]; [(1, n)] moves it
              unchanged. *)
      wide : bool;  (** Whether it forms offsets in 64 bits, for a wide side. *)
      input : input;
      scale : scale;
      output : output;
    }
  | Split of { wide : bool }
      (** An r2c's long halved line: the [h] complex points of its FFT, in place
          on [dst], into its [h + 1] bins. [wide] as a [Transpose]'s. Its
          [float32] operations round and flush as a {!scale}'s product's. *)
  | Prepass of { wide : bool }
      (** A c2r's long halved line: its [h + 1] bins into the [h] complex points
          its FFT takes, in the workspace. Its pair [k] with the forward root
          [w = w{_ 2h}{^ k}], from bins [a = X{_ k}] and [b = X{_ h-k}], their
          imaginary parts [+0] at [k = 0], with [E = a + conj b],
          [D = a - conj b], [U = fma w.re D.im (w.im D.re)] and
          [V = fma (-w.re) D.re (w.im D.im)], writes
          [Z'{_ k} = (E.re + U, E.im + V)], and
          [Z'{_ h-k} = (E.re - U, V - E.im)] where [0 < k] and [2k <> h]; an
          inverse conjugates [a] and [b] as it loads them, then sets their
          imaginary parts [+0] at [k = 0], and conjugates what it writes
          ({!calls}). [wide] as a [Transpose]'s. Its [float32] operations round
          and flush as a {!scale}'s product's. *)

type launch = private {
  work : work;
  src : side;
  dst : side;
  first_group : int;  (** The block its first group takes. *)
  groups : int;  (** Its groups, each a block, from [first_group]. *)
  threads : int;  (** The threads of a group. *)
  shared : int;  (** The threadgroup memory bytes of a group. *)
  cost : Plan.cost;
  seconds : float;  (** {!Plan.seconds} of [cost], one group resident a core. *)
}
(** The type for launches. A step's blocks are its lines, [lines] to a block; a
    transpose's [(a, b)] are tiles of 32 columns [c < b] by 32 rows [r < a] of
    the [b] rows of [a] it writes, each run's in turn, a run's by column, then
    by row, and a move [(1, n)]'s tiles of 32 of its lines by 32 of their
    points, the lines in order as the rows of one run; a [Split]'s or a
    [Prepass]'s are 1024 of a line's pairs of bins [(k, h - k)],
    [0 <= k <= h / 2], in groups of 256 threads with no threadgroup memory. *)

type call = private {
  launches : launch array;  (** In order. *)
  workspace : int;  (** The bytes of workspace the call takes. *)
  seconds : float;  (** The launches' together. *)
}
(** The type for calls. *)

val c2c :
  bases:int * int ->
  axes:int array ->
  Nx_array.Layout.t ->
  dst:Nx_array.Layout.t ->
  call option
(** [c2c ~bases ~axes x ~dst] is the plan of a complex transform along [axes]
    from an operand laid out as [x] into [dst], their buffers' first bytes at
    bytes [bases] of a {!Plan.line} ({!line_offset}), or [None] where one of its
    lines is Bluestein's with an [m] past {!bluestein_most} or takes more than
    [2{^32}] real points, the most its steps are planned for, whatever
    {!Plan.allowed}[ ()]: a call that declines plans no launch. A call with no
    element plans no launch.

    Along an axis of [n] points, of {!line}[ n ~real:false], whose FFT's sides
    are the operand's or [dst]'s, by their {!access}:
    - a line of one step with both sides [Rows] takes the [Contiguous] step;
    - one of one step of at most {!side_most} points, [n - 1] for Rader's and
      [m] for Bluestein's, with no side [Scattered] or wide, takes the
      [Side_by_side] step where its {!field-call.seconds} are at most those of
      the transposes below;
    - one of 4096 points, 7-smooth, whose sides are [complex64], [Columns] and
      not wide, [d] lines one element apart along the innermost axis of their
      batch, [d >= ]{!column_lines}, takes the column step where the bytes
      between a line's consecutive points, [|ps| w] on either side, are at most
      {!columns_stride_most}, and so are the workspace's [8 d], since [dst]'s
      [d] lines lie within its point stride: a [First_pass] from the operand's
      lines into the workspace, its line of run [i] and of place [c < d] in the
      run at element [4096 d i + c], its points [d] apart, then [Last_passes]
      from it into [dst]'s. Its {!field-call.seconds} are at most 0.51 of the
      transposes' below, so it takes no comparison: its parts move at most
      [4 ⌈d / 16⌉] MiB a run of [d] lines, two 128-byte lines a segment, at most
      0.42 of the [1.125 d] MiB the transposes charge, 288 bytes a point with a
      cache line for each point of a columns side; its operations, 76 a point
      against their 78 and more, and its two chains, over 16 groups or more, add
      at most 0.09 of the transposes' bytes' time;
    - one of one step otherwise moves each side that is not [Rows] through a
      [Transpose] [(1, n)] to or from the workspace, and takes the [Contiguous]
      step on the rest;
    - a long line of steps [s{_ 0}], …, [s{_ q-1}] runs in two levels,
      [n = s{_ 0} M]: a [Transpose] [(s{_ 0}, M)], the step on rows of [s{_ 0}],
      a [Twiddle] [Transpose] [(M, s{_ 0})], the [M]-point transform of each run
      of [M] by the same scheme, or the last step, and a [Transpose]
      [(s{_ 0}, M)] into natural order: [3 (q - 1)] transposes and [q] steps,
      each in place on rows of its points. The first transpose reads the lines
      the axis reads and the last writes [dst]'s; the others alternate, back
      from the last, between the workspace and [dst], where it is C-contiguous
      and holds the line's points, else a second workspace line, which also
      takes the first where it would read and write one buffer. A Bluestein line
      runs two stages of these on [m] points, each transpose from one of two
      workspace lines of [m] to the other: the first reads [Chirp_in], stage 1's
      last scales by its [Kernel], stage 2 starts where stage 1 ends and its
      last writes [Chirp_out] into [dst].

    A [Transpose] is [wide] where a side it reads or writes is wide. The
    workspace takes at most [3 m] points of [8] bytes a line of the axis that
    takes most, [m] its FFT's points.

    A launch's cost is {!Plan.cost}'s counts at one group resident a core, which
    every pipeline holds, since planning makes none, with a barrier at 4 links.
    A step's chain is a dependent read, then each pass's butterfly and two
    barriers, four side by side, and past the first pass its twiddle's product,
    2 links, and the issue of a thread's [⌈(l / r) / tl⌉ (r - 1)] twiddle loads,
    [r - 1] for each of its butterflies, [0.15] links a SIMD group of the group
    each; the chain rounds up to whole links. Its bytes are its groups' and, for
    a step, its roots', [8] bytes each, read once: the first half of an even
    step's points, whose second half is the first negated, and all of an odd
    one's. A group's side moves its lines' points at their bytes on [Rows] and a
    cache line, [128] bytes, a point off them, but for a [Columns] side side by
    side: its [g] lines lie in runs of [r = min g d] adjacent lines of [d] a
    run, [⌈g / r⌉] runs and one more where neither of [g] and [d] divides the
    other, each charged the lines [⌈(o + w r) / 128⌉] that [w r] bytes touch
    from [o], the worst byte of a line at which one starts. A run's segment
    starts at its buffer's {!field-side.base} plus its lowest element's bytes,
    plus multiples of [gcd (128, ps w, S{_ k} w, gcd (g, d) w)] bytes, [S{_ k}]
    the side's strides past the innermost, [w] its element's bytes, so [o] is
    that gcd's [q] less than 128 past the start's residue mod [q]. A transpose's
    group moves its tile's points, each read and written at its bytes on a
    [Rows] side and a cache line, [128] bytes, an element off rows, and [8]
    bytes more a point where it scales by a chirp or a kernel. In a [Twiddle]
    transpose each 32 of them at column [c], the lanes of a SIMD group along
    [r], are charged the high table's [⌈8 (⌊31 c / 1024⌋ + 2) / 128⌉ + 1] lines
    their entries [(r c) / 1024] may touch, [128] bytes each: where rows fill
    their tiles, [8] bytes a point at [b = 90], about [10.1] at 1024 and [21.8]
    at 4096, and [36] at column 4095. Its launch reads the [8] KiB low table and
    its chain counts the product's 4 links. A column step's work takes, for each
    run of [d] lines, [⌈d / 16⌉] blocks of 16 lines, block [b] from line [16 b],
    16 groups a block: in a [First_pass] group [k] takes the threads [t] of
    [⌊t / 16⌋ = k], in [Last_passes] the class [k]; groups lie by run, then by
    [k], then by block. Each of a group's two sides moves 256 segments, its
    lines' same point, each charged the lines [⌈(o + 128) / 128⌉] from [o], a
    side-by-side group's worst start at [g = d = 16]; a last block of fewer
    lines lies within those bytes. A [Last_passes] launch reads the step's
    roots. A group's operations are [5 · 4096 · 4] for a [First_pass]'s DFTs,
    [5 · 4096 · 8 + 6 · 4096 · 2] for [Last_passes]' DFTs and products, and 2 a
    point for an inverse's flips, each of 256 threads charged its share rounded
    up. A [First_pass]'s chain is a dependent read and a butterfly of 16; a
    [Last_passes]' a dependent read and the step's passes 1 and 2, one barrier
    between them. Its {!field-launch.seconds} are within {!Plan.allowed}[ ()]:
    longer work splits into launches of ranges of blocks, each charged its own,
    every one but a work's last holding more than [Plan.allowed () - s], [s] the
    seconds of the work's heaviest group, since the group that did not fit adds
    at most [s].

    Raises [Invalid_argument] if [bases] are not bytes of a {!Plan.line}, if
    [axes] is not strictly increasing axes of [x], if [x] and [dst] are not of
    one shape, or if a single group of a launch takes past {!Plan.allowed}[ ()].
*)

val line_offset : Rig.Buffer.t -> int
(** [line_offset b] is the byte of a {!Plan.line} at which [b]'s first byte
    lies, in the addresses of its device's work ({!Rig.Buffer.aligned}): [0]
    after one test where it lies on a line, as a created buffer does, else from
    seven, one a bit.

    Raises as {!Rig.Buffer.aligned} does. *)

val r2c :
  bases:int * int ->
  axes:int array ->
  Nx_array.Layout.t ->
  dst:Nx_array.Layout.t ->
  call option
(** [r2c ~bases ~axes x ~dst] is the plan of a real transform along [axes] from
    a [float32] operand laid out as [x] into [complex64] [dst], [n / 2 + 1] bins
    along the last of [axes], [n] [x]'s extent there, or [None] as {!c2c}.

    The last axis runs first, from [x] into [dst], by {!c2c}'s paths: a line of
    even [n] as the halved line {!line}[ (n / 2) ~real:true], of odd [n] as
    {!line}[ n ~real:false], its points real. A halved line in a group holds its
    [n / 2] points in its plane, and its first thread stores bin [n / 2], from
    bin 0's pair, outside it; a long one's FFT writes its [n / 2] points into
    [dst], and a [Split] makes its bins there. A [Transpose] of a line in a
    group's operand moves its floats, [Real] to [Real]. A long line's first
    reads the operand: [Real] for an odd line, [Complex] for a halved one, whose
    points are pairs of floats, [8] bytes a point on rows and [256] off them; an
    odd line's last writes [Half]. With [n = 0], each line takes a step of
    {!line}[ 1 ~real:false] that loads nothing and stores one bin of [+0], and
    no other axis runs. Each earlier axis then runs as {!c2c}'s, in place on
    [dst].

    Raises as {!c2c} does, and if [dst]'s shape is not [x]'s with [n / 2 + 1]
    along the last of [axes]. *)

val c2r :
  bases:int * int ->
  axes:int array ->
  Nx_array.Layout.t ->
  dst:Nx_array.Layout.t ->
  call option
(** [c2r ~bases ~axes x ~dst] is the plan of the transform from [complex64] bins
    laid out as [x], [n / 2 + 1] along the last of [axes], into [float32] [dst],
    [n] [dst]'s extent there, or [None] as {!c2c}.

    With more than one axis, the others run first, as {!c2c}'s, from [x] into a
    C-contiguous copy of [x] in the workspace from its first byte, and the last
    from the copy; with one, the last from [x]. It runs into [dst] by {!c2c}'s
    paths: a line of even [n] as the halved line {!line}[ (n / 2) ~real:true],
    of odd [n] as {!line}[ n ~real:false], completed from its bins as it loads
    them. A halved line in a group holds its [n / 2] points in its plane, and
    its first thread loads bin [n / 2] beside bin 0, outside it; a long one's
    [Prepass] makes its [n / 2] points in the workspace before its FFT. A
    [Transpose] of a line in a group's [dst] moves its floats, [Real] to [Real].
    A long line's last writes [dst]: [Real] for an odd line, [Complex] for a
    halved one, pairs as {!r2c}'s; an odd line's first reads [Bins]. The
    workspace takes the copy and, past it, what {!c2c} states.

    Raises as {!c2c} does, and if [x]'s shape is not [dst]'s with [n / 2 + 1]
    along the last of [axes]. *)

(**/**)

val bluestein_most : int
(** [bluestein_most] is [2{^20}], the most points of a Bluestein [m] a line
    plans with. *)

val bluestein_points : int
(** [bluestein_points] is [n{_ B} = 524,288], the most points of a Bluestein
    line whose [m] is at most {!bluestein_most}. *)

val bound_wide : int -> unit
(** [bound_wide n] is nx_metal.mli's [bound_fft_wide]: a side is wide where a
    line's points span past [n] elements. *)

val divisor : int -> int64
(** [divisor d] is [d]'s divisor for a kernel's division of a 64-bit unsigned
    [n]: [m = ⌊2{^64} (2{^l} - d) / d⌋ + 1] modulo [2{^64}] for [l = ⌈log₂ d⌉],
    so that [n / d = (t + (n - t) / 2) / 2{^l - 1}], [t] the high half of [m n],
    for [d >= 2]; [d = 1] divides nothing. For tests.

    Raises [Invalid_argument] if [d < 1]. *)

val flush_scale : float
(** [flush_scale] is [c = 13.29], the factor of the flushing term [c n 2{^-126}]
    nx_metal.mli adds to the bound of a line: [√(2n) Σ_ℓ f_ℓ ‖R_ℓ‖ / n] over the
    line's layers [ℓ], rounded up from its least upper bound over every sequence
    of passes, which seven passes of 3 nearly reach (12.98 at 2187) and passes
    of 16 do not (7.52 at 4096). [f_ℓ] is the flushes a layer's part takes at
    most, and [‖R_ℓ‖] a bound on the norm of the line's arithmetic that remains
    after it. The layers are the loads, then each pass's, a long line's steps'
    passes in order: past the first, its product by roots, a long line's step's
    first by the twiddles of its transpose, [f = 2]; then its DFT's levels,
    [f = 1], each a sum, an [fma] or a product by a constant. A DFT of [2{^k}]
    points is [k] levels of 2-point sums, some split in two and one of 16
    points' a product by roots; one of an odd prime [p] is its sums and
    differences, [(p - 1)/2] levels of [fma] chains and a last sum. [‖R_ℓ‖] is
    [√(n/(s r))] for the passes past the layer's, [s] the points before its pass
    and [r] its radix, times the remaining DFT's norm within the pass:
    [√(2{^j})] for [j] levels of 2-point sums, [√p] after an odd DFT's sums and
    differences, [√2] after a chain's level. A step of one point is its load and
    one product by [1]. Moves between steps compute nothing. Each pass's terms
    scale by [1/√s], so the sum over passes converges whatever their number. It
    bounds a halved real line's too, of [n = 2h] points: an r2c's, its FFT's
    factor over [√2], the split's norm at most [√2], and the split's layers', of
    [4 (h/2 + 1)] parts at most and a norm of at most 1 after each; a c2r's, its
    FFT's over 2, and the pre-pass's layers', each times [√h], the FFT's norm,
    and the pre-pass's that remains, at most 2. Both fall with [h] past their
    FFT's factor. For tests. *)

val rader_flush_scale : float
(** [rader_flush_scale] is [c = 42], {!flush_scale}'s factor for a Rader line,
    odd real or complex, of [n] points: from its loads, carried by the DFT's
    norm [√n]; the first FFT's error, carried to [y_0] and through the kernel
    and the second FFT, of norm [√m max |B_k|], at most [√(n / m)] as Gauss's
    sums give [|B_k| <= √n / m]; [y_0]'s sum; the kernel's product, [f = 2],
    carried by the second FFT; that FFT's error; and the sums with [x_0]:
    rounded up from its greatest over the class, each FFT's factor its step's.
    For tests. *)

val bluestein_flush_scale : int -> float
(** [bluestein_flush_scale n] is [c = 76 √n + 72], {!flush_scale}'s factor for a
    Bluestein line, odd real, halved or complex, of [n] points by an FFT of [m]:
    from its loads and chirp's products, carried by the DFT's norm [√n]; the
    first FFT's error, carried through the kernel and the second FFT, of norm
    [√m max |B_k|], at most [√(2n - 1)] since [Σ |B_k|²] is [(2n - 1) / m]; the
    kernel's product, carried by the second FFT; that FFT's error; and the
    chirp's last products. With each FFT's factor at most {!flush_scale} and
    [m < 4n], that is at most [4 · 13.29 √(2n) + 3√2 + 8√2 + 4 · 13.29 + 2√2],
    rounded up. For tests. *)

val c2c_moved :
  bases:int * int ->
  axes:int array ->
  Nx_array.Layout.t ->
  dst:Nx_array.Layout.t ->
  call option
(** [c2c_moved ~bases ~axes x ~dst] is {!c2c}'s plan with no column step: a line
    that would take it takes the transposes. For tests, which hold the column
    step's {!field-call.seconds} to its bound against theirs.

    Raises as {!c2c} does. *)

val runs : call -> bool
(** [runs c] is [true] iff {!emit} takes every launch of [c]: each step is one
    it has a kernel for, on sides it reads. *)

val key : launch -> (Kernels.t * int array) option
(** [key l] is the key {!emit} launches for [l], a specialised step's function
    and its constants ([Kernels.fft_fc_*]), or [None] where it launches a kernel
    of its own: a [Step]'s, but the [Contiguous] step of 4096 points of a
    complex line on complex sides, the only step of a 7-smooth line or one of a
    long line's: [Contiguous], [fft_step_p8] where [step.p = 8], else
    [fft_step_p16]; [Side_by_side], [fft_side_p8] where [step.p = 8], else
    [fft_side_p16]; of {!group_lines} lines a group; with its points, threads a
    line, lines a group, passes and radices, [0] past them, and what its sides
    hold, from their widths and points: a halved line's lone step's split where
    it reads floats and pre-pass where it writes them; an odd line's floats or
    bins in, and bins up to [n / 2] or floats out; nothing in where it reads no
    point; and the line's convolution where it runs in its group, Rader's or
    Bluestein's, else none, a long Bluestein line's steps FFTs alone. A key is a
    function of the step's points, the path, its lines a group and those alone.
*)

val emit :
  ?wide:bool ->
  Plan.builder ->
  inverse:bool ->
  place:(buffer -> int * int) ->
  launch ->
  unit
(** [emit ~wide b ~inverse ~place l] adds [l], a [Transpose]'s, a [Split]'s, a
    [Prepass]'s, a [Step]'s or a [Column_step]'s launch, to [b]: its kernel, of
    64-bit offsets where [wide] (defaults to the work's), over [l]'s groups,
    each side at [place]'s slot and byte, the constants its scale or its roots
    read, the forward transform's, and its sides' axes past what its words hold
    as the call's own bytes ({!Plan.Uploaded}), which take no kept constant.
    Where [inverse], each point it loads and each it stores is conjugated
    ({!calls}), but in a [Transpose] that copies bits. A step's kernel reads the
    forward roots of its points, the first half, as its inner table, a halved
    line's the roots [w{_ 2h}{^ k}], [k <= h / 2], of one table, each rounded
    once, and a convolution's tables, Rader's kernel and logarithms or
    Bluestein's kernel and chirp, with the line's [n]. For tests.

    Raises [Invalid_argument] if [l] is a [Step]'s whose lines a group are not
    {!rows}[ step] ([Contiguous]) or {!side}[ step.l] ([Side_by_side]), or a
    power of two below it: the steps with a kernel and a test that runs it; or
    if [l] is a [Column_step] whose [step] is not of 4096 points of radices
    [16 16 16], the only step of a complex 7-smooth line; or if a step's or a
    column step's side is wide, its point offsets past [2{^31} - 1] elements, or
    a column step's is not [complex64]. *)
