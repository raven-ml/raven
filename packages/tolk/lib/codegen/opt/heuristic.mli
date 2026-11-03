(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Hand-coded kernel optimisations.

    Rules of thumb that pick a kernel's optimisations ({!Opt.t}) from its shape
    and accesses, without measuring it. A reduce axis is one a reduction closes;
    the upcastable axes are the global, local and weak axes of a known size
    that no reduction closes, and the unrollable axes those that one does. An
    unroll is an upcast of a reduce axis. *)

val hand_coded_optimizations : Postrange.Scheduler.t -> Postrange.Scheduler.t
(** [hand_coded_optimizations k] is a copy of [k] with optimisations applied;
    [k] itself is left as it is. The first of these that applies decides:

    - {b streaming}, below, when the reuse of the streamed operand is at most
      32, the contraction sums at least 256 values an output, or a number of
      them not known, and it does at most the renderer's
      {!Renderer.t.per_byte_first} multiply-adds per byte of the buffers the
      streamed operand reads: the product of its output and
      contracted axes' sizes, against the bytes of each buffer once, or a
      number not known, as when a size is symbolic or the operand reads a
      buffer that is not a parameter;
    - {b tensor cores}, when the setting {!Setting.use_tc} is positive and [k]
      reduces over one axis, or {!Setting.tc_opt} admits more: the first of the
      three candidate axes on which a tensor core applies ({!Opt.Tc}), then
      upcasts of its [M] and [N] axes by the first of 5, 4, 3 and 2 that divides
      them, an upcast of [N] by 2 more while a lane of the core then holds at
      most 160 outputs, and a local split of [N] by the largest of 4 and 2 that
      divides it and leaves at least 256 workgroups. Each upcast applies only
      when it leaves at least [16384 / t] workgroups, [t] the threads of the
      core, a workgroup's, and each split when the count is not known. A split
      gives each workgroup more of the outputs and the kernel fewer workgroups:
      an upcast leaves fewer threads to keep loads in flight, and the local
      split, which keeps the threads, fewer workgroups to share among the
      compute units. A larger tile loads fewer operands for each step of the
      core, until its outputs fill a thread's registers. The floors move the
      order of a sum, so they are one for every device of a target, never read
      from a device: 16384 threads and 256 workgroups are the bounds of
      [sm_89]'s parts, which every target takes.
      A core whose operands are staged through shared memory ({!Stage}),
      where {!Stage.depth} stages its largest tile, takes a tile of its own
      instead of these splits and their floors, whose local split shares each
      copied tile among its warps:
      each warp holds 64 by 64 outputs, by upcasts of powers of two, a
      workgroup 2 by 2 warps where [M] and [N] divide by 2 past them, and [K]
      is unrolled to 32 elements a trip where it divides. While the kernel
      would keep fewer than [16384 / s] threads, [s] the stages, the warp
      tile halves, [M] then [N], down to 32 by 32: a pipeline overlaps a
      warp's copies with its cores, so fewer warps keep the loads in flight.
      Its warp's rows, and its workgroup's, stay within a run of the rows
      below, [M]'s upcast halved and its local split left out while they do
      not.
      Where the operands read [M] only as its quotient by
      [d], as blocks of [d] rows read their expert's matrix, and the core's
      tile times [M]'s upcast does not divide [d], [M] is upcast instead by
      the first of 5, 4, 3 and 2 that divides it and whose warp, the tile
      times the upcast, divides [d], if one does: a warp whose rows cross
      blocks loads each block's matrix.
      With {!Setting.tc_min_globals}, [M] is upcast and [N] split locally
      whatever the workgroups, and [N] is upcast only while that many
      workgroups remain;
    - {b streaming}, when the reuse is at most 32, or the streamed operand is
      decoded, whatever its reuse;
    - {b grouping}: when the upcastable axes hold at most 2048 elements, the
      first of the first three reduce axes that splits by 16 into local threads
      from the top; nothing more is applied then.

    {b Streaming} applies on a renderer with local indices and shared memory,
    when {!Setting.mv} is on, to a kernel where no range's bound reads an axis
    and that has no call or custom code ({!Postrange.Scheduler.opaque}), to a
    sum over [k]'s first reduce of a product of
    two computations of accesses with no reduce. Each element of an operand
    feeds as many outputs as its reuse: the product of the sizes of the global
    axes it does not read, times [d] for each it reads only as its quotient by
    [d]. The operand of least reuse is streamed, or of two of equal reuse the
    one laid out along a contracted axis. Its main access is the one with an
    axis of unit stride, a term of its index, whose terms of an axis the
    reduce contracts span the most bytes. When that axis [u] is contracted:
    {ul
     {- with reuse, the axes it comes from are upcast, each by its share of the
        largest divisor of the reuse up to 16; a reuse of [f] outputs along
        one axis the operand does not read takes [ceil (f / 16)] passes
        instead, of as many outputs each, the axis padded where they do not
        divide it;}
     {- [u] splits into the largest of 32, 16, 8, 4 and 2 local threads that
        divides it, and when they take [u] whole and are fewer than 32, the
        contracted axis of the next stride in the main access into as many
        more as make 32 or divide it;}
     {- the first global axis that the streamed operand reads, the other does
        not, and 4 divides, or else one both read, splits by 4: into local
        threads without reuse, into upcast lanes with it;}
     {- what the threads leave of [u], or else the axis of the next stride, is
        unrolled by its largest divisor up to 8, or 4 with reuse;}
     {- each contracted axis that no split has taken, other than [u], and
        along which the other operand is read at unit stride, is unrolled
        whole where its values there fit one load of 16 bytes and 32 lanes,
        stepping the other operand by its stride along [u], load across
        more than a 128-byte line: a byte's two MXFP4 values multiply a
        float32 operand at [2 b] and [2 b + 1], which a thread then loads
        at once.}}
    These two unrolls are skipped where their split is refused.
    The rows layout declines, and the next rule applies, when no global axis
    that the streamed operand reads other than as a quotient divides by 4,
    when [u] takes one thread, or when a split it asks for is refused.
    When that axis is a global axis, after the upcast of the reuse, it splits
    into 32 local threads and 2 upcast lanes, and the first reduce axis into
    the least power of two of local threads that makes 32768 threads, at most
    32, or the largest that divides it; with few outputs fewer threads take
    the global axis, down to 8, and more the reduce. It does not apply when
    more than 32768 outputs are left after the upcast. Like the tensor
    core's floor, 32768 threads is [sm_89]'s bound, fixed for every target.

    {b Decodes.} On a GPU, an operand decoded from integers, one whose value
    converts integers it reads into floats, that reads every output axis but
    one, along which its [R] outputs read it at one index, as every row of a
    product reads its matrix, is decoded at most [ceil (R / 16)] times a
    value when the kernel streams or takes a tensor core whose upcast of
    those rows applies, as it does wherever the streamed layout applies: a
    thread's upcast or a core's rows with their upcast hold up to 16 of the
    outputs, and each pass over the others decodes it again. The other rules
    share only their upcast of 4 rows, so a decoded operand streams past a
    reuse of 32. A core whose floor refuses the upcast decodes it once a tile
    of its rows: [ceil (R / 8)] times on a core of 8 rows, as Metal's. A core
    whose fragment holds each value of the operand in several lanes
    ({!Tc.fragment}), as RDNA3's holds its [B] in two, decodes it that many
    times more.

    On the host, which streams nothing, a decoded operand's rows share its
    decodes before the other rules apply. Along an output axis, [f] outputs
    read each value of a decoded operand at one index: the axis's size where
    the operand does not read it, [d] where it reads it only as its quotient
    by [d]. Each such axis, later axes first, is upcast by the largest amount
    up to 16 that divides the axis and one such operand's [f], and keeps the
    kernel within the host's budget below (its accumulators and its
    conversions). Where the rows' stores lie apart, each row's accumulators
    take registers of their own: a file of 16 registers shares a decode over
    at most 8 rows, one of 32 over 16. A host that does not know its
    registers takes at most 4.

    Otherwise, in order:

    - masked axes, those a selection's condition runs in, of size at most 7, are
      upcast whole, while their product stays at most 49;
    - while the upcastable axes hold at least 1024 elements and fewer than 32
      lanes are upcast, one more axis is upcast by 3 or 4, chosen among axes
      some access does not index while indexing every upcast axis, least indexed
      and of least stride first;
    - on a renderer with local indices, a contraction summed in 4 bytes or
      fewer whose loop upcast two axes, each by 4, is blocked when an operand
      does not index each, 2 divides each, and what is left of them once
      halved holds a SIMD group of 32 threads in powers of two: each axis is upcast again by 2, so that
      a thread holds 8 by 8 outputs, the last unrollable axis is unrolled by 4
      when 4 divides it, and the workgroup takes at most 32 threads;
    - the last unrollable axis is unrolled whole if of size at most 32 (and a
      second one if both are at most 3), or else by 4, when at most 4 lanes are
      upcast or no reduce axis is, and fewer than 64;
    - if nothing is upcast, the last upcastable axis is upcast by 4 when 4
      divides it;
    - for a renderer with local indices, global and weak axes of a constant size
      that every call and custom code reads
      ({!Postrange.Scheduler.unread_by_opaque}) become local: up to three
      axes, those some access does not index first,
      split by the first of 32 (axis [0] only), 16, 8, 4, 3 and 2 that divides
      them while the workgroup stays within 128 threads, or 32 when blocked.

    The host is the [CPU] device; a host vector spans 64 bytes, two registers
    of AVX2. A kernel of the host without a reduce, whose lanes the host computes
    as vectors, takes in place of the upcasts after its masked axes one upcast
    of its last output axis that 2 divides, by the largest power of two, at most
    16, that keeps its lanes within a vector of the widest type it reads, writes
    or computes, and again while they have room. It takes none when its stored
    values hold 5 operations or more for each operation of their longest chain,
    or when it computes a type the target emulates.

    A kernel of the host with a reduce, for a renderer that knows its vector
    registers ({!Renderer.t.registers}), counts its lanes in registers
    ({!held_registers}): its upcasts after the masked and shared ones, and its
    unroll, take only what keeps the accumulators of its hottest loop within
    half the registers and each conversion within 64 lanes, those of the upcast
    axes its value reads that hold distinct values ([n] lanes of an axis read as
    a quotient by [d] hold [n / gcd n d], and a reduce's lanes are summed before
    it); and a file of no fewer registers, each no narrower, takes no fewer
    output lanes. These upcasts apply however few elements the upcastable axes
    hold. The first upcasts the axis its stores write at unit stride, when every
    access that reads the axis reads it at unit stride, by the largest power of
    two, at most 16, that keeps the lanes within a vector of the accumulators'
    type, divides the axis and fits; the upcasts by 3 or 4 follow while they
    fit, whatever the lanes; when none is left, one more, by 2, of an output
    axis that an access reading a reduce axis does not index, while it fits. The
    unroll takes what fits while the lanes upcast fill fewer registers than the
    target has, and next to a wholly unrolled axis of at most 3, unrolls the
    next reduce axis by the largest power of two that keeps the unrolled lanes
    within a vector of the reduce's type. *)

val held_registers : Postrange.Scheduler.t -> Renderer.registers -> int
(** [held_registers k file] is the registers of [file] that the accumulators of
    [k]'s hottest loop take. A register of [file.bytes] holds as many lanes of a
    reduce's type as fit, at 4 bytes for a type narrower than 4, which C
    computes as an [int] or a [float], and no more than [k]'s stores write at
    consecutive addresses.

    The hottest loop is the loop of the reduce with the most iterations around
    its body: its own loops and those of the reduces it runs inside. A reduce of
    at most 32 iterations belongs to the body around it, the hottest reduce's
    loop, or the output loop when no reduce is longer. The loop holds, for each
    output lane, one accumulator of each reduce it runs inside or shares a loop
    with and of each shorter reduce that runs inside it. An unroll that keeps a
    reduce's loop holds nothing more; a reduce unrolled whole holds each
    unrolled lane while another reduce of the loop still loops, and nothing once
    none does. A kernel without a reduce holds nothing. *)

(**/**)

val hand_coded_with :
  core_threads:int -> Postrange.Scheduler.t -> Postrange.Scheduler.t
(** [hand_coded_with ~core_threads k] is {!hand_coded_optimizations} with the
    tensor core's floor of 16384 threads made [core_threads]. *)
