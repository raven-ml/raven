(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gathers, scatters, folds, assemblies, sorts and numberings as tolk
    graphs.

    Each function takes nodes as {!Lower_arith} holds elements and computes what
    {!Nx_kernel.S} states of its operation. Every shape must be static.

    {b Numerics.} These bounds implement the indexing rows of
    {!Rune.section-numerics}. A gather, a scatter by [Set], an assembly and a
    sort move elements, so they give the bits of nx's kernels, a NaN's sign and
    payload included. So does a scatter where no update lands. A scatter by
    [Max] or [Min] gives nx's bits except at a NaN result, whose sign and
    payload are unspecified. A scatter by [Add] wraps integers exactly. Where
    [n] updates [u1], …, [un] land on an element [x], the float sum of [x] and
    those updates adds in the carrier ({!Nx_kernel.carrier}) in an
    association {!scatter} chooses, and rounds once.
    Before that rounding it is within [γ(n) (|x| + Σ |ui|)] of the exact sum,
    where [γ] is as {!Nx_kernel.section-bounds} defines it for the carrier. Each
    part of a complex sum is such a sum. The sum is [-0] where [x] and every
    [ui] are [-0], and [+0] for other terms whose sum is zero. A fold of [m]
    terms [ti] sums them as a scatter's [Add] does, within [γ(m - 1) Σ |ti|]. It
    is [-0] where every [ti] is [-0], and [+0] where it has no term. It rounds a
    narrow float once, from [float32], while nx's kernel rounds each of its
    additions. *)

open Tolk

val gather : Nx_array.Dtype.any -> axis:int -> Ops.t -> Ops.t -> Ops.t
(** [gather d ~axis idx x] is [x], a value of [d], read at the [int64] positions
    [idx] along [axis], as {!Nx_kernel.S.gather} states. It has [idx]'s shape,
    and zero bits where a position lies outside the axis. Each element is one
    indexed load. *)

(** The storage a scatter combines into. *)
type dst =
  | Filled of (Ops.t -> Ops.t)
      (** [Filled f]: [f v], where [v] is [into]'s parts flat in C order, is
          storage of [v]'s dtype and element count that holds [v], such as a
          buffer once [v] is stored into it, an {!Ops.after} over the store. *)
  | Holding of Ops.t
      (** [Holding s]: [s] is storage of [into]'s parts flat in C order that
          holds them already, such as a loop's carry that [into] reads. The
          caller guarantees that neither [idx] nor [updates] reads [s]: the
          scatter's kernel reads them while it writes [s]. *)

val scatter :
  ?dst:dst ->
  host:bool ->
  unique:bool ->
  Renderer.t ->
  Ops.device ->
  Nx_array.Dtype.any ->
  Nx_kernel.combine ->
  axis:int ->
  idx:Ops.t ->
  updates:Ops.t ->
  Ops.t ->
  Ops.t
(** [scatter ~dst ~host ~unique ren device d combine ~axis ~idx ~updates
    into] is [into], a value of [d], with [updates] combined by [combine] at
    the positions [idx] holds along [axis], as {!Nx_kernel.S.scatter}
    states, for a program on [device], whose kernels compile for the host's
    processors where [host] holds ({!Tolk_engine.target}) and are rendered
    by [ren]. [unique] is the caller's promise that no two updates target
    one element: where two do, the result there is unspecified. Without it,
    under [Set], the last update in C order of [updates] wins.

    For [n] positions and [k] updates along [axis], nx's kernel stores [n + k]
    elements. The scatter combines the updates in a kernel of its own, into
    storage that holds [into], each thread walking its updates in C order:
    [n + k] work at each index of the other axes where [host] holds, and at
    most [n + r k] elsewhere, for [r <= min n k] runs of positions, as many as
    keep [ren]'s [stream_threads] busy; [n] of it fills the storage, none
    where [dst] is [Holding]. Where the [m] elements of [into] are fewer than
    [stream_threads], threads take [h] chunks of about [√k] consecutive
    updates, [h m <= stream_threads], each chunk combined apart into storage
    of its own, then the chunks in their order: [h m] more work, and up to
    two more kernels, where the chunks shorten each thread's walk by more
    than those kernels cost ({!plan}). An element then takes [Set]'s last
    update and [Max]'s or [Min]'s extreme as in C order, and [Add]'s sum of
    each chunk's updates in C order, the chunks' sums in order. A narrow
    float's [Add] sums in a [float32] copy of [into] and marks where updates
    land in a byte per element of [into].

    On a GPU, updates that target distinct elements take no walk: a thread
    for each combines it into its element, [n + k] work. That holds under
    [unique], for every combine but a narrow float's [Add], and for a [Set]
    of rows, positions an expansion repeats along every axis after [axis],
    where the walk would take one chunk and within the bounds {!election}
    states. Such a [Set] first elects its rows: in a kernel or two, each
    row's position is compared with those of the later rows at its index of
    the axes before [axis], [p k{^2}] comparisons for [p] such indices, and
    each row a later one overwrites is dropped. Each target then takes one
    update, the last in C order, as the walk gives it. A chunked walk is
    about as short as the election; below the bounds the walk is shorter
    than the election's launch, and above them the comparisons outnumber
    the walk's steps. The host always walks.

    Other combines combine into [dst]'s storage, and the result is that
    storage once combined. The scatter writes over it, so no value but the
    result may lie in it once the scatter's kernel runs: it is fresh storage,
    a donated argument's that the result takes, or a carry's that the result
    replaces, each value that reads the argument or carry as it was reading it
    before. [dst] defaults to [Filled] by a fresh buffer that [into] is stored
    into.

    Raises {!Lower_arith.Unlowered} for a boolean [Add]. *)

val fold :
  Nx_array.Dtype.any -> Nx_kernel.pad -> int iarray -> Ops.t -> Ops.t
(** [fold d pad shape x] is the adjoint, applied to [x], of a [Padded] load by
    [pad] from a value of [d] and [shape], as {!Nx_kernel.S.fold} states. [x]
    has the load's result shape. Its work is [O(size·(l·step + n))] at each
    index of the other axes, for [l] windows of [size] elements [step] apart
    over an axis of [n].

    Raises {!Lower_arith.Unlowered} for a boolean [d]. *)

val assemble :
  ('v, 's) Nx_array.Dtype.t ->
  'v ->
  int iarray ->
  (Nx_array.Move.range iarray * Ops.t) list ->
  Ops.t
(** [assemble dt fill shape pieces] is the value of [dt] and [shape] made of
    [pieces], as {!Nx_kernel.S.assemble} states. Its element at an index is that
    of the last piece whose region holds the index, or [fill] where none does.
    Each piece selects its elements over the earlier ones, making [p N] selects
    for [p] pieces of [m] elements in all into [N] elements, each over every
    piece's node at that element: a kernel computes each piece at every element
    unless tolk folds its mask away. nx's kernel stores [N + m]. For no piece,
    or one over its background (the fill, or a first piece whose region is all
    of [shape]) that reads storage as it lies, each element loads at most two
    values and computes only its background. *)

val sort :
  Nx_array.Dtype.any ->
  axis:int ->
  descending:bool ->
  k:int option ->
  Ops.t ->
  Ops.t * Ops.t
(** [sort d ~axis ~descending ~k x] is the elements of [x], a value of [d],
    sorted stably along [axis], then their [int64] positions, as
    {!Nx_kernel.S.sort} states; with [Some k], the first [k] of each slice.
    Each slice sorts by a bitonic network over its keys packed above their
    positions, one pass per 32 bits of key: [O(n log² n)] work and about
    [log² n / 2] kernels a pass for [n] elements, two passes for a 64-bit
    dtype or [complex64], four for [complex128]. A first [k] of keys of at most
    32 bits, when the whole array's [k n²] comparisons are few, compares every
    pair instead: two kernels for its positions. Otherwise, with [k] at most
    the network's [s(s+1)/2] compare steps, [s] the least with [n <= 2{^s}],
    it takes [k] rounds, each the least key past the last round's: [O(k n)]
    work and a kernel a round for its positions, two where tolk splits a long
    slice's reduction.

    Raises {!Lower_arith.Unlowered} for an axis of more than [2{^31}]
    elements. *)

val unique :
  host:bool -> Renderer.t -> Ops.device -> size:int -> Ops.t -> Ops.t array
(** [unique ~host ren device ~size keys] is the numbering of the rows of the
    [int64] node [keys], of [[n; w]], as {!Nx_kernel.S.unique} states: [ids],
    [first] and [counts] of [size] entries, and [count], 0-d, for a program
    on [device] whose kernels compile for the host's processors where [host]
    holds and are rendered by [ren]. It sorts the rows stably by {!sort}, a
    sort of [n] [uint64] words for each column, then scatters by {!scatter}:
    [O(w n log² n)] work.

    Raises {!Lower_arith.Unlowered} for more than [2{^31}] rows. *)

(**/**)

val plan :
  host:bool -> Renderer.t -> p:int -> n:int -> q:int -> k:int -> int * int
(** [plan ~host ren ~p ~n ~q ~k] is the runs of positions and the chunks of
    updates that {!scatter} takes for [into] of [p·n·q] elements, an axis of
    [n] between [p] and [q], under [p·k·q] updates. *)

val election :
  Renderer.t -> p:int -> n:int -> q:int -> k:int -> int option
(** [election ren ~p ~n ~q ~k] is [Some c] where a GPU's [Set] {!scatter} of
    rows into [into] of [p·n·q] elements, an axis of [n] between [p] and
    [q], under [k] rows of [q] updates at each of the [p] indices, elects its
    rows, and [None] where it walks. It elects where {!plan} takes one chunk,
    from 16 rows to [2{^30}] comparisons [p k{^2}]. Each row's [k]
    comparisons are cut in [c] parts, compared in one kernel and reduced in
    a second, where that shortens a thread's [k] comparisons by more than
    the second kernel's launch costs. *)
