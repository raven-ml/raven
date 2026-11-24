(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Contractions on the matrix units: how a call cuts into tiles, ranges of k
    and launches (nx_metal.mli's Contraction).

    The plan reads a call's logical extents and its operands' dtype alone: its
    tile, its ranges and so its bits never depend on a layout. A layout picks a
    tile's loader, which only sets how many groups a core holds, and with it
    where launches cut, which changes no bit. *)

(** {1:tiles Tiles} *)

(** The type for the dtype a tile stages its operands in, whose values each
    convert to [float32] exactly. A loader reads a weight of another dtype and
    stages its elements' casts (nx_metal.mli's Contraction). [I8] stages an
    integer call's bytes, of integers of 8 bits or fewer or booleans, as
    [float16]: its tiles sum each block of 256 terms at [float32], below
    [2{^24}], so exactly, and add the blocks in 64-bit integers. *)
type stage = F32 | F16 | Bf16 | I8

(** The type for tiles of outputs: 64 x 64, 32 x 32 and 16 x 64, and a [Rows]
    call's grouped tiles, 32 x 64 and 16 x 64, whose rows are of one group in
    any order. *)
type tile = T64 | T32 | T16 | G32 | G16

val rows : tile -> int
(** [rows t] is [t]'s rows of outputs. *)

val cols : tile -> int
(** [cols t] is [t]'s columns of outputs. *)

val depth : stage -> tile -> int
(** [depth s t] is the terms a tile stages a step: 32 for 16 x 64 tiles, at
    [float16] and at [I8], 16 otherwise. *)

val bound : rows:int -> buckets:int -> int -> int
(** [bound ~rows ~buckets bm] is [⌈rows / bm⌉ + min buckets rows - 1], or [0]
    for no rows: the most tile rows of [bm] rows that [rows] rows in [buckets]
    buckets take, a tile row's rows of one bucket. A bucket of [c] rows takes
    [⌈c / bm⌉], and at most [min buckets rows] hold rows. *)

val shared : nk:bool -> stage -> tile -> int
(** [shared ~nk s t] is the threadgroup memory bytes of a group: [a]'s staged
    rows of {!depth} terms and [b]'s of {!cols} columns, or of {!depth} terms
    where [nk], each row padded by 16 bytes. *)

val threads : int
(** [threads] is 128, the threads of a tile's group: four SIMD-groups. *)

val spread : stage -> tile -> int
(** [spread s t] is [G]: 32 times the groups of [t] a core holds, by
    {!Plan.holds} of [b]'s [k] x [n] form's {!shared}, a constant of the
    library. *)

(** {1:plans Plans} *)

type extents = {
  batch : int;  (** The batch elements. *)
  m : int;  (** The rows: [a]'s free extents multiplied. *)
  n : int;
      (** The columns: [b]'s free extents multiplied, twice that for {!complex}.
      *)
  k : int;  (** The contracted extent, twice that for {!complex}. *)
  stage : stage;
  nk : bool;
      (** Whether the tiles read [b] of the plan's rows and columns
          ({!t.exchanged}) along [k] ([n] x [k]) rather than along [n]. *)
  complex : bool;
      (** Whether the call is a [complex64] one's real form: [float32] rows of
          [n] columns and [k] terms, [a]'s parts its terms and [b]'s 2 x 2
          blocks of parts its terms and columns. *)
  source : int;
      (** The bytes of an operand's element a loader reads, of the wider
          operand, [0] where the tiles read their stage's own dtype. *)
}
(** The type for a call as the plan reads it. *)

type launch = {
  first : int;  (** The launch's first tile in the call's order. *)
  tiles : int;  (** Its tiles. *)
  chunk : int;
      (** The chunk of each range it takes: its terms from [chunk * length] past
          the range's start. *)
  length : int;
      (** A chunk's terms, [0] for whole ranges. The last chunk of a range takes
          the rest of its terms, which may be fewer, or none. *)
  seconds : float;  (** Its {!Plan.seconds}. *)
}
(** The type for a launch of tiles, each a group for each range of k. *)

type t = {
  exchanged : bool;
      (** Whether the tiles compute [yᵀ = bᵀ aᵀ]: where [n <= 16 < m] and the
          call is not {!extents.complex}. *)
  rows_out : int;  (** The tiles' rows: [m], or [n] where exchanged. *)
  cols_out : int;  (** The tiles' columns. *)
  tile : tile;
  ranges : int;  (** [S]. *)
  spans : int;  (** [r = ⌈k / 32⌉]. *)
  tiles_m : int;  (** The tiles along the rows, [⌈rows_out / rows tile⌉]. *)
  tiles_n : int;
  swizzle : int;
      (** The log of the tile rows a block of the call's order holds: 2 where
          [tiles_m] and [tiles_n] are at least 8, else 0. *)
  padded : int;
      (** The call's order's tiles a batch element: [⌈tiles_m / 2^swizzle⌉]
          blocks of [tiles_n 2^swizzle], of which a tile past [tiles_m] computes
          nothing. *)
  resident : int;
      (** The groups a core holds: {!Plan.holds} of the layout's {!shared}.
          Metal reports no more than a group's 128 threads for a tile's
          pipeline, whose attribute bounds them, so it tells nothing of the
          groups its registers let a core hold; the bench checks the estimate on
          the M1 Max. *)
  launches : launch array;
      (** In order: tile windows, each window's chunks in increasing k. Empty
          where [k = 0]. *)
  seconds : float;  (** The launches' {!launch.seconds} added. *)
}
(** The type for plans. Range [s] of an output takes the spans of 32 terms from
    [⌊s r / S⌋] to below [⌊(s + 1) r / S⌋]. Tile [i] of the call's order is of
    batch element [i / padded]; within it, the [q]th of [padded], block
    [q / (tiles_n 2^swizzle)], column [(q mod (tiles_n 2^swizzle)) / 2^swizzle]
    and row [2^swizzle × block + q mod 2^swizzle]. *)

val exchanges : m:int -> n:int -> bool
(** [exchanges ~m ~n] is [n <= 16 < m]: whether a call of [m] rows and [n]
    columns computes [yᵀ = bᵀ aᵀ], so that its tiles' rows are few. *)

val big : int
(** [big] is 96: a call takes 64 x 64 tiles where it has at least that many of
    them. *)

val plan : extents -> t option
(** [plan e] is the plan of [e], [None] where neither a launch of one tile over
    whole ranges nor one over chunks of 32 terms fits {!Plan.allowed}[ ()]:
    + [exchanged] where {!exchanges} and [e] is not {!extents.complex};
    + tiles of 16 x 64 where the tiles' rows are at most 16; else 64 x 64 where
      the call has at least {!big} of them, [batch] times the rows' and columns'
      tiles of 64; else 32 x 32;
    + [S = 1] where the call's tiles [T], [batch tiles_m tiles_n], are none or
      at least half of [G = spread]; else
      [max 1 (min 64 (min ⌊r / 8⌋ ⌈G / T⌉))];
    + launches of whole ranges, each of the most tiles that keep its estimate
      within {!Plan.allowed}[ ()] and are a multiple of
      [lcm (G, tiles_n 2^swizzle)], the last taking the rest; where one such
      unit already exceeds it, windows of one unit, each over chunks of the most
      terms, a multiple of 32, that keep it within, in increasing k; where one
      unit's 32 terms exceed it, the same in units of [tiles_n 2^swizzle] tiles,
      then of one tile.

    Only the launches depend on [e.nk] and [e.source]. It allocates its launches
    and touches no device. *)

val grouped : extents -> groups:int -> t option
(** [grouped e ~groups] is the plan of a [Rows] call of [e], [b] of [groups]
    groups, its rows those of every batch element in [batch groups + 1] buckets,
    its last of the rows of no group, as {!plan} but for:
    + never [exchanged];
    + tiles of 32 x 64 ({!G32}) where [m >= 16 groups], else 16 x 64 ({!G16});
    + [tiles_m] the {!bound} of [batch m] rows in those buckets, of one batch
      element, [padded] being [tiles_m tiles_n] and [swizzle] [0];
    + [T] for [S] is [padded];
    + a tile's estimate reading its own slice of [b].

    Neither the ids nor their order enter it. *)

val estimate :
  extents ->
  t ->
  bandwidth:float ->
  cores:int ->
  op_rate:float ->
  launch ->
  float
(** [estimate e p ~bandwidth ~cores ~op_rate l] is [l]'s estimate by
    {!Plan.estimate} with these rates, its wave of [cores] cores, for the bench,
    which checks each launch against the M1 Max's. *)

(** {1:vector Vector kernels} *)

(** A call of few rows over a weight of a key: an output's [r = ⌈k / 32⌉] spans
    of 32 terms go to [S] lanes (nx_metal.mli's vector association). A
    threadgroup takes a column group for one pass of [pass] rows; the grid's x
    is the passes, so a column group's passes run together and share its weight
    in the caches. An n x k threadgroup of {!threads} threads holds [32 / S]
    columns of [width] a lane in each SIMD-group. A k x n one, of a weight of
    columns contiguous, of {!kn_threads}, holds column blocks of [S / rho']
    SIMD-groups, [rho' = min] {!rho}[ S], each SIMD-group [rho'] spans of
    [32 / rho'] lanes of [width] columns. *)
module Vector : sig
  (** The type for the keys of the wider operand: [float32], [float16] and
      [bfloat16] unscaled, [float4_e2m1fn] by [Exponent] codes, and [int8] by
      [Value]s of [float16] or [float32]. *)
  type key =
    | Dense_f32
    | Dense_f16
    | Dense_bf16
    | E2m1_by_exponent
    | Int8_by_value

  val bound : key -> rows:bool -> int
  (** [bound key ~rows] is the most rows of a call over a weight of [key], a
      [Rows] call's where [rows]: 11 for [E2m1_by_exponent] and 3 for
      [Int8_by_value], 16 for both in a [Rows] call; 2 for [Dense_f16] and
      [Dense_bf16], 1 for [Dense_f32]. *)

  val terms : key -> rows:bool -> int
  (** [terms key ~rows] is the most terms of a call over a weight of [key], a
      [Rows] call's where [rows]: [2{^23}] for [Dense_f32], [2{^22}] for
      [Dense_f16] and [Dense_bf16], [2{^19}] for [E2m1_by_exponent] and
      [Int8_by_value], [2{^18}] in a [Rows] call. One column group at every row
      count to {!bound}, n x k or k x n, plans within {!Plan.budget} at each, so
      {!plan} finds launches for every call they take. *)

  val columns : int
  (** [columns] is 256, the least columns of a vector call. *)

  val threads : int
  (** [threads] is 128, an n x k threadgroup's: four SIMD-groups. *)

  val kn_threads : int
  (** [kn_threads] is 256, a k x n threadgroup's: eight SIMD-groups. *)

  val rho : int
  (** [rho] is 4, a k x n SIMD-group's most spans at once, so that its lanes
      read a term's 128 bytes of columns together. *)

  type t = {
    kn : bool;  (** Whether of the k x n kernels. *)
    rows : int;  (** The outputs' rows a batch element. *)
    cols : int;  (** Their columns. *)
    lanes : int;  (** [S]: the most power of two at most [r] and 32. *)
    spans : int;  (** [r]. *)
    pass : int;
        (** [R], a pass's rows: 1 at one row and in a [Rows] call; n x k, 2 for
            [Dense_f16] and [Dense_bf16], else 1; k x n, 4 for
            [E2m1_by_exponent] and [Int8_by_value], else 1. No bit depends on
            it. *)
    width : int;
        (** A lane's columns: n x k, 1 for [Dense_f32], 4 for
            [E2m1_by_exponent], else 2; k x n, 16 bytes of a plain key's, 8 of a
            scaled key's. *)
    groups : int;  (** The column groups a batch element. *)
    resident : int;  (** The groups a core holds: {!Plan.holds}. *)
    launches : launch array;
        (** In order, each of [tiles] column groups from [first] of the call's
            [batch groups], over every pass and whole spans. Empty where a count
            is 0. *)
    seconds : float;  (** The launches' {!launch.seconds} added. *)
  }
  (** The type for vector plans. Column group [q] is of batch element
      [q / groups]. *)

  val passes : t -> int
  (** [passes v] is [⌈rows / pass⌉]. *)

  val shared : t -> int
  (** [shared v] is a group's threadgroup memory bytes: a k x n one's 4 a sum of
      each thread, at most 8 sums, else 0. *)

  val plan : extents -> key:key -> rows:bool -> kn:bool -> t option
  (** [plan e ~key ~rows ~kn] is the vector plan of [e] over a weight of [key],
      of a [Rows] call where [rows], of the k x n kernels where [kn], whose
      launches each take the most column groups that keep its estimate within
      {!Plan.allowed}[ ()], the last the rest; [None] where one column group
      exceeds it. [e]'s [stage], [nk], [complex] and [source] do not enter it,
      and [kn] only its launches, never [lanes] or [spans].

      Raises [Invalid_argument] where [kn] and [key] is scaled. *)

  val weights : key -> extents -> t -> tiles:int -> Plan.cost
  (** [weights key e v ~tiles] is a launch's cost of [tiles] column groups: each
      pass reads its columns' weight at DRAM's rate, which errs high where the
      caches keep it for the column group's other passes. *)

  val estimate :
    key ->
    extents ->
    t ->
    bandwidth:float ->
    cores:int ->
    op_rate:float ->
    launch ->
    float
  (** [estimate key e v ~bandwidth ~cores ~op_rate l] is [l]'s estimate by
      {!Plan.estimate}, for the bench. *)

  (**/**)

  val bound_terms : int -> unit
  (** [bound_terms k] makes {!terms} at most [k], so that tests reach it with
      small arrays; [max_int], the initial bound, restores it. It applies to
      calls planned after it returns. For tests.

      Raises [Invalid_argument] if [k] is not positive. *)
end

(** The type for how a call's terms and columns fill its tiles:
    - [Whole]: [k] is a multiple of the tile's {!depth} and the columns a
      multiple of its {!cols};
    - [Wide]: otherwise, with at least {!cols} columns;
    - [Narrow]: fewer than {!cols} columns. *)
type fit = Whole | Wide | Narrow

val fit : extents -> t -> fit
(** [fit e p] is how [e]'s terms and [p]'s columns fill [p]'s tiles. *)

val kernel : stage -> nk:bool -> tile -> fit -> Kernels.t
(** [kernel s ~nk t f] is the GPU function of tile [t] at [s] for calls that
    fill it as [f], its [b] read [n] x [k] where [nk]. With [b] read [k] x [n],
    a [Whole] call takes a function that bounds no read; a [Wide] one, one whose
    last tile column starts [cols t] columns before the last, so that it reads
    no column past [n]; a [Narrow] one, one that reads [b]'s columns past [n] as
    its last. At [I8] and for grouped tiles, [Whole] and [Wide] take the same
    function. With [b] read [n] x [k], one function bounds every read, whatever
    [f].

    Raises [Invalid_argument] for a grouped tile at [I8]. *)

val complex_kernel : nk:bool -> tile -> fit -> Kernels.t
(** [complex_kernel ~nk t f] is the GPU function of the complex tile [t], a
    {!kernel} at {!F32} whose [b] is [complex64], read [n] x [k] where [nk], and
    staged as the real 2 x 2 blocks of {!extents.complex}. Its functions by [f]
    are {!kernel}'s. *)

(** {1:calls Calls} *)

val converted : Nx_array.any -> Nx_array.any -> stage
(** [converted a b] is the stage of a [float32] call of [a] and [b] not of one
    staged dtype: {!F16} where [float16] holds every value of both dtypes, else
    {!Bf16} where [bfloat16] does, else {!F32}. *)

val source_width : Nx_array.any -> int
(** [source_width x] is the bytes of [x]'s element as a loader reads it: its
    dtype's, or [1] for a dtype narrower than a byte, which is widened first. *)

type recast = {
  a : Nx_array.Dtype.any option;
      (** The dtype into which a [float32] call casts [a] first, [None] for [a]
          read as it is. *)
  b : Nx_array.Dtype.any option;  (** The same of [b]. *)
  temp : bool;
      (** Whether [dst] does not merge its batch, row and column axes each into
          one run of strides. *)
}
(** The type for a call that runs in pieces ({!call}). *)

(** The type for a call's plan: [Planned p], its plan or [None] for a call of no
    output; or [Recast r], a call to run in pieces. *)
type planned = Planned of Plan.t option | Recast of recast

val largest_outputs : int
(** [largest_outputs] is [2{^31} - 1]: a piece's outputs at most, so that a
    [complex64] piece's real form has fewer than [2{^32}]. *)

val most_outputs : int Atomic.t
(** [most_outputs] holds the outputs a call computes at most, from 1 to
    {!largest_outputs}, which it holds first: a call of more but a [Contracting]
    one is [Recast] ({!call}). *)

type band = {
  a_band : int;  (** The bytes [a]'s band takes at most, [0] for none. *)
  b_band : int;  (** The bytes [b]'s band takes at most, [0] for none. *)
}
(** The type for the bands of a piece's operands ({!call}). *)

val bandable :
  contracting:(int * int) iarray ->
  Nx_kernel.operand ->
  Nx_kernel.operand ->
  bool * bool
(** [bandable ~contracting a b] is whether a line of [a], and of [b], may be
    read through a band ({!call}): its terms, along the axes [contracting]
    pairs, make one run of strides. *)

val vectored :
  whole:int * int * int ->
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  segments:
    (int64, Nx_array.Dtype.int64_elt) Nx_array.t Nx_kernel.segments option ->
  Nx_array.Dtype.any option * Nx_array.Dtype.any option ->
  Nx_kernel.operand ->
  Nx_kernel.operand ->
  bool
(** [vectored ~whole ~batch ~contracting ~segments (ca, cb) a b] is [true] iff
    the vector kernels take a [float32] call of [whole]'s batch, rows and
    columns of [a] cast into [ca] and [b] into [cb], [None] for an operand read
    as it is: the call of {!Recast}'s casts. *)

val call :
  ?whole:int * int * int ->
  ?band:band ->
  ops:Nx_array.any array ->
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  segments:
    (int64, Nx_array.Dtype.int64_elt) Nx_array.t Nx_kernel.segments option ->
  acc:Nx_array.Dtype.any ->
  dst:Nx_array.any ->
  init:Nx_array.any option ->
  Nx_kernel.operand ->
  Nx_kernel.operand ->
  (planned, Nx_array.answer) result
(** [call ~whole ~ops ~batch ~contracting ~segments ~acc ~dst ~init a b] plans
    the contraction {!Nx_kernel.S.contract} states, its slots naming the call's
    operands [ops] as the door numbers them ({!Plan.slot}): [dst], then the
    arrays it reads. It plans copies of the operands and of [init] whose layouts
    the tiles do not read, an integer call's widening of operands narrower than
    a byte into bytes, the launches of {!plan}, and the finish, which adds the
    ranges' sums in order, then [init] as the accumulator holds it, and casts
    the sum into [dst], where the tiles do not store into [dst] themselves.

    A [float32] call that would stage an unscaled operand not of its stage's
    dtype through no loader, its tiles holding products, is [Recast]: each such
    operand is to be cast into the stage's dtype, [float32] for a scaled call,
    and the call of the casts planned, which is never [Recast]. So is a call but
    a [Contracting] one whose [dst] does not merge its batch, row and column
    axes each into one run of strides, [temp]: each piece is to compute into a
    C-contiguous temporary of [dst]'s dtype, which merges them; and one of more
    outputs than {!most_outputs} holds, each piece to compute at most so many.
    No array is read.

    An integer [acc] takes [a] and [b] of integers of 8 bits or fewer or
    booleans on the tiles at [I8]; other operands, wider integers and floats, on
    the SIMT kernel, a thread an output and a range of k, which reads each
    element at its own dtype. Both sum in 64-bit integers, which wrap, and the
    finish wraps the sum into [acc] and casts it into [dst].

    A [Rows] call first launches contract_buckets, which gives each row its
    bucket and, for the tiles of {!grouped}, lists the rows by bucket and each
    tile row's bucket and rows; an integer one takes the SIMT kernel, each row
    reading its bucket. A row of no group has no products.

    A [Contracting] call first launches contract_compact, which lists each batch
    element's contracted indices by group, in increasing order within a group,
    and takes the SIMT kernel at every accumulator: an output sums its group's
    indices through the list, in the ranges its group's count gives under the
    call's cap, and the finish adds that group's ranges alone. A group without
    indices has no products.

    It is [Ok (Planned None)] for a call of no output, and [Error Declined] for
    a call of an [acc] of another dtype than [float32], [complex64] and the
    integers, a scale beside an integer or [complex64] [acc], into [complex64]
    an [a], [b] or [dst] of another dtype, a [float64] or [complex128] anywhere,
    a scale's included, a [Contracting] call's [dst] whose groups with its
    batch, row and column axes do not each merge into one run of strides,
    extents past [2{^32}] buckets, a [Contracting] call's outputs or contracted
    indices, no plan, or a prepass past a launch.

    [whole] is the batch, rows and columns of a call of which this one is a
    piece, its outputs a box of the call's ([a]'s rows, [b]'s columns and the
    batch cut, its terms whole): the exchange, the tile, [S] and whether the
    vector kernels take it are the call's, so each output has the call's bits.
    It defaults to the call's own.

    An operand of [band] is one line of a cast operand, a row of [a] or a column
    of [b] with its terms, of one batch element: the call reads its cast through
    a band in its workspace of at most that many bytes, which, before each chunk
    of the tiles' launches, holds the chunk's terms of each range, cast into the
    stage's dtype as {!Recast}'s casts are. The tiles then launch in chunks
    always, of at most the terms the band holds, a span at least, so each output
    keeps the bits of the call of the casts. A call with [band] is one
    {!vectored} is [false] of, casts no operand but the banded ones, and bands
    only lines {!bandable} allows: it raises [Invalid_argument] for a banded
    line whose terms make several runs of strides.

    nx.metal has checked the call: its shapes fit and [acc] accumulates. *)

(** {1:routines The matrix routines' products}

    A routine plans its products into its own call: their launches join the
    builder's, in order, and take pieces of its workspace. *)

type matrix
(** The type for a product's matrices: an array's, or one in the call's
    workspace. *)

val array : int -> Nx_array.any -> matrix
(** [array k x] is [x]'s elements through its layout, [x] on the buffer of the
    call's operand [k], the first operand on it ({!Plan.slot}). *)

val operand : int -> Nx_array.Dtype.any -> Nx_array.Layout.t -> matrix
(** [operand k dt l] is the matrix of [dt] elements through [l] on the buffer of
    the call's operand [k], as {!array} names it: a planner's view of an operand
    of which it holds the facts alone ({!Operands}). *)

val work : at:int -> Nx_array.Dtype.any -> Nx_array.Layout.t -> matrix
(** [work ~at dt l] is the matrix of [dt] elements through [l] in the workspace
    of the builder it is planned into, [l]'s offset counted in elements from
    byte [at], a piece {!Plan.take} gave. *)

type options = {
  init_first : bool;
      (** The first range of k sums from [y], the others from [+0]. *)
  neg : bool;  (** [a]'s values negated. *)
  conj_a : bool;  (** [a] conjugated, for a complex [acc]. *)
  conj_b : bool;  (** [b] conjugated, for a complex [acc]. *)
  tri : [ `Lower | `Upper ] option;
      (** Where [Some], the outputs outside that triangle, the diagonal inside
          it, are left as they are; tiles that hold none of the triangle compute
          nothing. *)
  ranges : int option;
      (** [Some s]: each output's [k'] terms, [k' = 2k] for a complex [acc] (its
          real terms), in [S = min s ⌈k' / 32⌉] ranges, range [i] taking the
          spans of 32 terms from [⌊i r / S⌋] to below [⌊(i + 1) r / S⌋],
          [r = ⌈k' / 32⌉], whatever the tile. [None] takes {!plan}'s count,
          which depends on the batch, so a routine whose bits must follow each
          matrix's extents alone passes [Some]. *)
}
(** The type for what a routine's product asks beyond {!Nx_kernel.S.contract}.
*)

val plain : options
(** [plain] asks nothing more: a contraction's association. *)

val ranges : rows:int -> cols:int -> k:int -> int
(** [ranges ~rows ~cols ~k] is the count of ranges for products of [rows] x
    [cols] outputs of [k] terms ([k'] for a complex [acc]), a function of these
    extents alone: with [t = ⌈rows / 64⌉ ⌈cols / 64⌉] and [r = ⌈k / 32⌉], [1]
    where [2t >= 192] or [t = 0], else [max 1 (min (min 64 ⌊r / 8⌋) ⌈192 / t⌉)]:
    {!plan}'s rule for one matrix of tiles of 64 x 64 at [float32]'s spread. *)

exception Decline
(** Raised by {!product} and {!copy} where a step finds no launches: one of its
    groups passes the budget ({!Plan.allowed}), or its outputs pass one launch's
    bounds. The call they plan for declines. *)

val product :
  Plan.builder ->
  options ->
  acc:Nx_array.Dtype.any ->
  init:bool ->
  y:matrix ->
  matrix ->
  matrix ->
  unit
(** [product b o ~acc ~init ~y a m] plans [y <- op(a) op(m)], or
    [y <- y + op(a) op(m)] where [init], into [b]: [a] of shape
    [batch; rows; k], [m] of [batch; k; cols], [y] of [batch; rows; cols], all
    of [acc]'s dtype, [float32] or [complex64], and of any layouts, [op] [o]'s
    negation and conjugations. It takes the tiles at {!F32}, never the vector
    kernels. Each output's sum is a fused multiply-add chain in the ranges
    [o.ranges] gives, else {!plan}'s; with [o.init_first] its first range starts
    from [y], else [y] is added after the ranges' sums, which are added in
    increasing k. A subnormal element of [a], [m] or a [y] the sums read is read
    as a zero of its sign. With no term, [y] is left as it is where [init], else
    [+0]. It plans nothing for a [y] of no element. [y] shares no byte with [a]
    or [m].

    An operand the tiles do not read where it lies is copied into a piece of the
    workspace by {!copy} first: an [a] whose terms are not contiguous, an [m]
    whose columns and terms are not, and a [complex64] [y] whose columns are
    not. The complex tiles read a [complex64] [m] as its real blocks, with [o]'s
    sign flips. [o.init_first] at more than one range takes the ranges' planes
    and a copy of [y] into the first. The pieces are given back after.

    Raises [Invalid_argument] if [acc] or an operand's dtype is another, and
    {!Decline} if a step finds no launches. *)

val copy : Plan.builder -> src:matrix -> dst:matrix -> unit
(** [copy b ~src ~dst] plans the copy of [src]'s elements into [dst], of [src]'s
    dtype, [float32] or [complex64], and shape, of rank 3, both of any layouts,
    into [b]; a subnormal element, or part, is copied as a zero of its sign.

    Raises [Invalid_argument] if [src] is of another dtype or rank or [dst] of
    another dtype or shape, and {!Decline} if a group of [NX_CCOPY_THREADS]
    elements of a row takes longer than a launch allows or [src] has 2{^ 32}
    rows or more over its batch. *)

(**/**)

val bound_big : int -> unit
(** [bound_big n] makes {!plan} give 64 x 64 tiles to a call of at least [n] of
    them, in place of {!big}, so that a test's call of a few tiles runs the 64 x
    64 tile's kernels. Only the choice of tile changes: the plan's other fields
    follow {!plan}'s rule from that tile, and the launches run the same kernels.
    [bound_big big] restores the default.

    Raises [Invalid_argument] unless [1 <= n <= big]. *)

val divisor : int -> int
(** [divisor d] is the divisor word of [d] that a loader divides by (kernels.h's
    tiles' words): a multiplier [m] in its low 32 bits and shifts [s1] and [s2]
    in bits 32-39 and 40-47, such that [u / d = (t + ((u - t) lsr s1)) lsr s2],
    [t] the high 32 bits of [u m], for every [u] below [2{^32}] (Granlund and
    Montgomery, "Division by invariant integers using multiplication", 1994,
    figure 4.1).

    Raises [Invalid_argument] if [d] is not in \[[1];[2{^32} - 1]\]. *)

type steps = { sz : int; sj : int; sn : int; rn : int; sk : int; rk : int }
(** The type for a loader's scale of [b]: its strides in elements along [b]'s
    batch ([sz]), its groups ([sj]), its columns ([sn]) and its terms ([sk]),
    and a block's elements along the columns ([rn]) and the terms ([rk]). *)

val scale_steps :
  nk:bool ->
  ?axis:int ->
  Nx_array.Layout.t ->
  Nx_array.Layout.t ->
  int array array ->
  steps option
(** [scale_steps ~nk ?axis l sl groups] is the steps of a scale laid out as [sl]
    of a [b] laid out as [l], over [b]'s axes [groups] (batch, staged rows,
    inner), its staged rows its columns where [nk], and a [Rows] call's axis of
    groups [axis]. It is [None] where a run does not fit (nx_metal.mli's
    loaders), a step or a block's elements lie outside signed 32 bits, or [b]'s
    terms number [2{^32}] or more, and the call then reads [b]'s products with
    its scales from workspace. *)
