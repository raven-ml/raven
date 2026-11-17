(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Staging a tensor core's operands through shared memory, and a decoded
    operand through a panel on the host ({!section-panels}).

    A tensor core ({!Op.Wmma}) reads each operand as fragments of a tile: the
    box of addresses its index spans over the workgroup's threads and the
    thread's lanes, at each trip of the loop of its sum. Read from global
    memory, each warp loads its own fragments in a few bytes at a time, and
    warps that share a tile load it again. Staged, every thread of the
    workgroup copies the tile into memory the workgroup shares, [16] bytes at a
    time, once a trip, and the cores read their fragments back from it:

    {v
      global  ──copy, 16 bytes a thread──▶  shared, swizzled  ──▶  fragments
    v}

    Staging moves values and never rounds them: a staged kernel computes the
    bits of the kernel of the same optimisations unstaged.

    {b Two stages.} Where the target copies asynchronously
    ({!Renderer.t.async_copy}), the copy of the next trip's tile overlaps this
    trip's cores, in a second slot:

    {v
      before the loop:  copy chunk 0 into slot 0;  barrier
      trip k:           copy chunk k + 1 into slot (k + 1) mod 2
                        read slot k mod 2, compute;  barrier
    v}

    One barrier a trip retires every read of slot [k mod 2] before trip [k + 1]
    overwrites it, and publishes chunk [k + 1] before it is read. The barriers
    are the ones code generation places between a store to shared memory and
    the reads after it. The loop's first copy runs before each run of the
    loop, at each trip of every serial loop around it. A tile that steps
    through several loops, as an operand summed over heads and then a head's
    width does, steps through them as one loop of trips counted outermost
    first, the trip after an inner loop's last being the next outer trip's
    first, where one node closes them all, their extents are constant and
    that node closes no other loop inside the outermost; it copies in one
    stage otherwise.
    Without asynchronous copies a single stage copies each trip's tile at the
    start of the trip. *)

val depth : ?used:int -> Renderer.t -> int list -> int
(** [depth ~used ren tiles] is the number of stages a core whose staged
    operands' tiles hold [tiles] bytes takes on [ren], beside [used] bytes
    (default [0]) of shared memory other cores take: [2] where [ren] copies
    asynchronously and two stages fit {!Renderer.t.shared_max}, [1] where one
    does, and [0], staging nothing, where neither does, where [tiles] is empty
    or a tile holds less than one copy of [16] bytes, where [ren] has no local
    threads or no shared memory, or where {!Setting.stage} is off. *)

val operands : Renderer.t -> Ops.t -> Ops.t
(** [operands ren sink] is the kernel [sink], whose optimisations are applied,
    with each tensor core's operands staged ({!depth}) that read global memory
    through an index whose terms that read the workgroup's threads and the
    thread's lanes form a dense box of strided dimensions: each element read
    once, and none of those terms reading another range. One dimension of
    the box may be gathered rows: a term [c · t] whose [t] reads the threads
    and lanes only through one load of global memory, a row map, whose own
    index's terms form one dimension of the box. Each copy loads its row's
    entry of the map and reads that row, so the operand's rows are gathered
    once a workgroup; such a tile copies in one stage, and is staged only
    where several warps read it: where a local range of the workgroup does
    not move the operand's index. A gate on such an
    operand, its index's validity or a selection of [0], copies [0] where it
    fails. An operand computed elementwise from loads of global memory, and
    not {!decoded}, is staged where some of its loads that read the threads
    and lanes read such a box, and each that reads none reads fewer of the
    threads' and lanes' elements than the box of most elements. The loads
    whose tiles step through the loops of its tile of most elements are
    copied, where those are the loops of each operand that is a load, and the
    others stay loads, as a per-head scale stays while the core's loop runs
    within a head, and as a norm's scale and shift stay beside its normalised
    rows where their gate over a padded axis reads lanes their own box does
    not hold; the operand computes on the copies as it did on the loads. Rows
    gathered through a map of row indices, which hold no box and read at
    least as many of the threads' and lanes' elements as the map's, keep
    their operand unstaged.

    A {!decoded} operand is staged where [ren] copies asynchronously
    ({!Renderer.t.async_copy}) and each of its loads is copied or held. A
    load is copied where every copy of its tile a trip is one asynchronous
    copy of [16] bytes: its innermost dimension contiguous in whole copies,
    and every other term of its address and its storage's start multiples of
    [16] bytes, the index read simplified. A load whose copies a trip are not
    so is held where its one loop moves it by at most a line of [128] bytes
    over all its trips: its tile over the threads, the lanes and that loop,
    one run of whole copies, is copied once before the loop into shared
    memory, which the operand reads at each trip. A tile is held only where
    the operand has copied loads and each is staged, so that none is left to
    the L1 that shared memory shares an SM's store with, and in the shared
    memory every core's stages leave. The first trip's reads wait for the
    held copies at the barrier of the stages' first copies.

    An operand whose terms mix the box's ranges with others otherwise stays
    unstaged, as do the cores of a kernel that holds other shared memory or
    reduces across threads. *)

val admits : Renderer.t -> Ops.t -> bool
(** [admits ren sink] is [true] iff {!operands} stages every operand of each
    tensor core of [sink] on [ren], a computed operand through the loads it
    copies, tiles permitting ({!depth}): the question
    the heuristic asks before it gives a core the staged tile
    ({!Heuristic.hand_coded_optimizations}). *)

val steps : Renderer.t -> Ops.t -> k:Ops.t -> int
(** [steps ren sink ~k] is the least count of steps of the range [k] a trip
    at which each row of the tiles {!operands} copies from global memory
    covers a sector of [32] bytes, the unit in which an asynchronous copy
    reads global memory, where [k] steps the row along its innermost
    dimension, [k] keeps more than one trip and the cores' stages keep their
    depth ({!depth}). It divides [k]'s extent, and is [1] where no tile
    reads a row so, no such count exists, or [ren] copies no tile
    asynchronously ({!Renderer.t.async_copy}).
    A row of [16] bytes a trip, as MXFP4's codes are at [32] values a trip,
    reads each sector at two trips. The heuristic upcasts the core's [K] by
    it where the core keeps the unstaged splits
    ({!Heuristic.hand_coded_optimizations}). *)

val warp_tile : int
(** [warp_tile] is the rows and the columns of the outputs a warp of a staged
    core holds at most ({!tile}). *)

val tile :
  Renderer.t ->
  Ops.t ->
  Ops.wmma ->
  core_threads:int ->
  workgroups:int option ->
  sizes:int * int * int ->
  axes:Ops.t * Ops.t * Ops.t ->
  in_run:(int -> bool) ->
  (Opt.t list * bool) option
(** [tile ren sink core ~core_threads ~workgroups ~sizes:(n, m, k) ~axes
    ~in_run] is the optimisations of a tensor core's staged tile, splits and
    a pad ({!Opt.t.Split}, {!Opt.t.Padto}) whose [axis] is [0], [1] or [2]
    for the core's N, M and K, in the order to apply them, and whether the
    core has two stages and the kernel keeps [core_threads / 2] threads on
    some tile, which an unknown count of workgroups does not show, when
    {!admits} and {!depth} stage the core; [None] otherwise, as for a core
    reading gathered rows, whose warps the core does not yet split. [n], [m] and [k] are the sizes
    of those axes as the core leaves them, [0] when unknown, [axes] their
    ranges, and [workgroups] the kernel's, if known. A warp holds
    {!warp_tile} by {!warp_tile} outputs; a workgroup takes 2 by 2 warps
    where [M] and [N] divide by 2, one on each of the four schedulers of an
    NVIDIA SM since Volta, and [K] is unrolled to 32 elements a trip where it
    divides. The stages [s] are
    those {!operands} copies: {!depth} of the tiles its loads copy, at their
    types, beside the stages of the kernel's earlier cores, and one where the
    tiles step through no loop or loops that are no nest, or an address
    gathers.
    While the kernel keeps fewer than [core_threads / s] threads, the warp
    tile halves, [M] then [N], down to half of {!warp_tile} a side: a
    pipeline overlaps a warp's copies with its cores, so it needs fewer warps
    in flight. Where neither the whole warp tile's upcast of [N] nor its
    half divides [n], and {!Setting.tc_opt} is [2], [N] is first padded to a
    multiple of the whole tile's ({!Opt.t.Padto}): its padded columns
    multiply zeros into outputs never stored, and every plan divides it.

    Where [ren] states its compute units and the bytes a unit keeps in
    flight ({!Renderer.t.units}, {!Renderer.t.in_flight}), the core has two
    stages and [workgroups] is known, the tile follows them. Past that plan,
    where its accumulators take at most 320 bytes a thread and so are
    blocked into running sums, the warp tile halves on, [M] then [N] in
    turn, down to the core's own, and the plan whose busiest unit copies the
    fewest bytes a trip is taken: [⌈g / units⌉] of the plan's [g]
    workgroups, each copying its rows of each operand. [K] is then unrolled
    to the least depth from 32 elements at which those copies reach
    [in_flight], or the deepest where none does, among the depths that keep
    the stages and the blocks of the sum's running sums, where [K] is its
    sum's innermost range; it stays at 32 otherwise. Each output so sums
    the same terms in the same order whatever the plan and the units, and
    [K] is never split among a workgroup's warps.

    [in_run u] is whether [u] of the core's tiles of rows read one value of
    an operand: [M]'s upcast falls by one tile, and its local split is left
    out, while it does not hold, so that a run of three tiles takes one
    warp. *)

(**/**)

val layout : itemsize:int -> cols:int -> Ops.t -> Ops.t -> Ops.t
(** [layout ~itemsize ~cols col row] is the offset in a staged tile of the
    element at [col] in [row], rows of [cols] elements of [itemsize] bytes:
    each row's [16]-byte chunks permuted by an XOR with bits of [row], so that
    the [16]-byte chunks of eight consecutive rows at one column, and eight
    consecutive chunks of the tile in row order, fall on eight different
    [16]-byte slots of a [128]-byte line of shared memory's banks. *)

(**/**)

(** {1:panels Decoded panels on the host}

    A product with an operand decoded from integer codes, as a quantised
    matrix's is, decodes each value in every tile of outputs that reads it.
    The rows a tile holds share a decode, but each trip of an output loop the
    operand does not read decodes the value again: 64 rows in tiles of 8 decode
    each value 8 times. Staged, the values the operand takes within one trip of
    the output loops it reads, its panel, are decoded once into a buffer on
    the stack, and every trip of the other output loops reads them back, as a
    packed panel of a matrix is reused across the rows of the other (Low et
    al., "Analytical Modeling is Enough for High Performance BLIS", ACM TOMS,
    §4.3.2):

    {v
      for each trip of the loops the operand reads:      (its columns)
        for its sum's ranges and lanes:  panel <- decode
        for each trip of the other loops:                (the row tiles)
          sum the product, reading the panel
    v} *)

val decoded : Ops.t -> bool
(** [decoded o] is [true] iff [o] converts integer codes into floats, by a cast
    or a bitcast, outside any access's address: integers that a load of
    integer storage gives, through integer arithmetic alone. A float's bits
    read as an integer, as a bfloat16's sign is selected, are no codes. *)

val decodes : Renderer.t -> Ops.t -> Ops.t
(** [decodes ren sink] is the kernel [sink], whose optimisations are applied,
    with its decoded operand's panel staged where [ren] targets the host
    (["CPU"]). It stages where exactly one operand [o] of [sink]'s sums of
    products is {!decoded}, every output loop (a weak range no sum closes) has
    a constant bound, one of them of at least two trips does not read [o], and
    [o]'s other ranges are its sum's ranges or upcast lanes of constant
    extents, of which [o]'s values take at most 64 KiB: the panel lies on the
    stack of the thread that runs the kernel.

    Staged, [o]'s values over those ranges are stored, before the output loops
    that do not read [o] run, into a buffer of registers' address space
    ({!Dtype.Reg}), in rows of its sum's ranges, its lanes together, and [o]
    reads them back; the output loops that read [o] take the first of the
    output loops' axes, in their order, so that they run outside the others.
    Staging moves values and never rounds them, and the output loops' trips
    are independent: a staged kernel computes the bits of the kernel of the
    same optimisations unstaged. [sink] is returned as it is where
    {!Setting.stage} is off. *)
