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
    unroll is an upcast of a reduce axis. A bound, stride, tile or divisor
    that is a constant of a type other than the weak ones, a cast of a weak
    one ({!Ops.is_literal}), is read as its value, so the optimisations do not
    depend on the integer type of a kernel's ranges and index.

    A kernel holding a range of size 0, which does no work along it, takes no
    optimisation. Otherwise the first of these that applies decides:

    - {{!section-streaming}streaming}, when the reuse of the streamed operand
      is at most 32, the contraction sums at least 256 values an output, or a
      number not known, and does at most the renderer's
      {!Renderer.t.per_byte_first} multiply-adds per byte of the buffers the
      streamed operand reads, or a number not known: the product of its
      output and contracted axes' sizes against the bytes of each buffer once,
      unknown when a size is symbolic or the operand reads a buffer that is
      not a parameter. A reuse past 16 streams only when no tensor core of
      two stages ({!Stage.tile}) keeps its threads, since the operand is then
      read in several passes;
    - a {{!section-cores}tensor core}, when the setting {!Setting.use_tc} is
      positive, on any number of reduce axes: the core takes one of them, and
      the others stay loops around it, unless an unstaged core's operand
      reads one inside the core's [K] ({{!section-cores}below});
    - streaming, when the reuse is at most 32 or the streamed operand is
      decoded, whatever its reuse;
    - the {{!section-general}general rules}, preceded on the host by the
      {{!section-decodes}decodes}.

    The streaming guards are arithmetic on the kernel's sizes and index
    terms, so a kernel that passes them takes the layout, unless a core takes
    a reuse past 16 first, and only a tensor core is tried. On the host, a renderer that knows its vector registers
    ({!Renderer.t.registers}) bounds the general rules with
    {{!section-budget}the budget}.

    {1:streaming Streaming}

    Streaming applies on a renderer with local indices and shared memory, when
    {!Setting.mv} is on, to a kernel where no range's bound reads an axis and
    that has no call, and no code in the target's language
    ({!Postrange.Scheduler.opaque}). It takes the first of the kernel's
    reduces that sums a product of two computations of accesses, neither
    reading a reduce that runs in a range the sum closes: a row's expert that
    a reduce over the experts finds runs in the row alone, and is computed
    once for the row. An operand's reuse is the product of the sizes of the
    global axes it does not read, times [d] for each it reads only as its
    quotient by [d]. The operand of least reuse is streamed, or of two of
    equal reuse the one laid out along a contracted axis. Its main access is
    the one with an axis of unit stride, a term of its index, whose terms of
    an axis the reduce contracts span the most bytes. No layout applies, and
    the kernel falls to the next rule, when it has no such access.

    When that axis [u] is contracted, the kernel takes the rows layout, when
    [u] splits into more than one thread and a global axis that the streamed
    operand reads other than as a quotient divides by 4. A split that
    {!Postrange.Scheduler.apply_opt} refuses is skipped.
    {ul
     {- With reuse, the axes it comes from are upcast, each by its share of
        the largest divisor of the reuse up to 16; a reuse of [f] outputs
        along one axis the operand does not read takes [ceil (f / 16)] passes
        instead, of as many outputs each, the axis padded where they do not
        divide it.}
     {- [u] splits into the largest of 32, 16, 8, 4 and 2 local threads that
        divides it, and when they take [u] whole and are fewer than 32, the
        contracted axis of the next stride in the main access into as many
        more as make 32 or divide it.}
     {- The first global axis that the streamed operand reads, the other does
        not and 4 divides, or else one both read, splits by 4: into local
        threads without reuse, into upcast lanes with it.}
     {- What the threads leave of [u], or else the axis of the next stride,
        is unrolled by its largest divisor up to 8, or 4 with reuse.}
     {- Each contracted axis that no split has taken, other than [u], and
        along which the other operand is read at unit stride, is unrolled
        whole where its values there fit one load of 16 bytes and 32 lanes,
        stepping the other operand by its stride along [u], load across more
        than a 128-byte line.}}

    When that axis is a global axis, the kernel takes the columns layout: after
    the upcast of the reuse, the axis splits into 32 local threads, fewer down
    to 8 with few outputs, and 2 upcast lanes, and the first reduce axis into
    the least power of two of local threads that makes 32768 threads, at most
    what a workgroup of 1024 threads leaves beside those across the axis, or
    the largest that divides it. It does not apply when more than 32768
    outputs are left after the upcast, or when the axis does not divide by the
    local threads and lanes; 32768 threads is [sm_89]'s bound, fixed for every
    target as the tensor core's floor is.

    {1:cores Tensor cores}

    A tensor core applies to the first of the three candidate axes on which
    {!Opt.Tc} does. Its [M] and [N] axes are upcast by the first of 5, 4, 3
    and 2 that divides them, and [N] splits locally by the largest of 4 and 2
    that divides it and leaves at least 256 workgroups. Each upcast applies
    only when it leaves at least [16384 / t] workgroups, [t] the threads of
    the core, a workgroup's, and each split when the count is not known. The
    floors move the order of a sum, so they are one for every device of a
    target: 16384 threads and 256 workgroups, the bounds of [sm_89]'s parts,
    which every target takes. With {!Setting.tc_min_globals}, [M] is upcast
    and [N] split locally whatever the workgroups, and [N] upcast only while
    that many workgroups remain.

    After those splits, each other reduce axis of [n] values that an operand
    reads at unit stride, and the core's own [K] at stride [n], is unrolled
    whole where [n] of the operand's values fit a 16-byte load: the operand
    reads the two axes as one run, so the values a thread's fragment holds
    along [K] become adjacent. A decoded 4-bit operand splits the reduce into
    bytes and a byte's two codes, which the other operand reads at [2b + c]:
    the core takes the bytes, and the codes unrolled make one load where two
    trips read the values two apart.

    A core whose operands are staged ({!Stage.tile}) takes the staged tile
    instead of those splits and floors. Where the operands read [M] only as
    its quotient by [d], as blocks of [d] rows read their expert's matrix,
    and the core's tile times [M]'s upcast does not divide [d], [M] is upcast
    by the first of 5, 4, 3 and 2 that divides it and whose warp divides
    [d], if one does: a warp whose rows cross blocks loads each block's
    matrix. The staged tile keeps its warps' and workgroups' rows within such
    a run by halving [M]'s upcast and leaving out its local split.

    {b Decodes.} On a GPU, a value of an operand decoded from integers, whose
    [R] outputs along one axis read it at one index, is decoded at most
    [ceil (R / 16)] times when the kernel streams or takes a core whose
    upcast of those rows applies, since a thread's upcast or a core's rows
    hold up to 16 of the outputs. A core whose floor refuses the upcast
    decodes it once a tile of its rows, [ceil (R / 8)] times on a core of 8
    rows, and a core whose fragment holds each value of the operand in
    several lanes ({!Tc.fragment}) that many times more.

    {1:general The general rules}

    A kernel that neither streams nor takes a core takes the rules below in
    order. The first, grouping, ends the sequence when it applies.

    - {b Grouping}: when the upcastable axes hold at most 2048 elements, or,
      in a kernel that is not a contraction and each of whose reduces sums
      at least 256 values an output, fewer than the renderer's
      [stream_threads], 16 local threads take consecutive indices of one of
      the first three reduce axes, the first that splits by 16 of those a
      load reads at unit stride, then of the others. Where no load reads the axis the group takes at
      unit stride but one reads an output axis so, 16 threads first take
      consecutive indices of the first such axis whose split leaves the group
      one, and are the workgroup's fastest ([lidx0]). Consecutive threads of
      a SIMD group then read consecutive addresses where a load allows it.
    - Masked axes, those a selection's condition runs in, of size at most 7,
      are upcast whole, while their product stays at most 49.
    - While the upcastable axes hold at least 1024 elements and fewer than 32
      lanes are upcast, one more axis is upcast by 3 or 4, chosen among axes
      some access does not index while indexing every upcast axis, least
      indexed and of least stride first, 3 before 4 of one axis; 4 before 3
      for a contraction that the next rule would block once its two output
      axes some operand does not index are upcast by 4, and whose blocked
      threads, a 64th of its outputs, number at least [core_threads].
    - On a renderer with local indices, a contraction summed in 4 bytes or
      fewer whose loop upcast two axes, each by 4 and each leaving part of
      its axis, is blocked when an operand does not index each, 2 divides
      each, and what is left of them once halved holds a SIMD group of 32
      threads in powers of two: each axis is upcast again by 2, the last
      unrollable axis is unrolled by 4 when 4 divides it, and the workgroup
      takes at most 32 threads. Where one of the two upcasts took its axis
      whole, the output axis that the same accesses index takes its place, as
      the width of a projection's heads takes the heads'.
    - The last unrollable axis is unrolled whole if of size at most 32 (and
      a second one if both are at most 3), or else by 4, when at most 4
      lanes are upcast or no reduce axis is, and fewer than 64.
    - If nothing is upcast, the last upcastable axis is upcast by 4 when 4
      divides it.
    - On a renderer with local indices, global and weak axes of a constant
      size that take threads ({!Postrange.Scheduler.threadless}) become
      local: up to three axes, those some access does not index first, split
      by the first of 32 (axis [0] only), 16, 8, 4, 3 and 2 that divides them
      while the workgroup stays within 128 threads, or 32 when blocked. A
      blocked contraction keeps that split where it lays a power of two of at
      least 2 threads along the later of its two axes; else 16 or 8 threads,
      whichever pads it less, lie along the later axis and the rest along the
      other, each axis padded to its threads ({!Opt.t.Padto}), unless one takes
      no threads or a pad is refused.

    {1:host The host}

    The host is the [CPU] device; a host vector spans 64 bytes, two registers
    of AVX2.

    {2:decodes Decodes on the host}

    On the host, which streams nothing, a decoded operand's rows share its
    decodes before the general rules. Along an output axis, [f] outputs read
    each value of a decoded operand at one index: the axis's size where the
    operand does not read it, [d] where it reads it only as its quotient by
    [d]. Each such axis, later axes first, is upcast by the largest amount up
    to 16 that divides the axis and one such operand's [f] and keeps the
    kernel within {{!section-budget}the budget}. A host that does not know
    its registers takes at most 4.

    {2:budget The budget}

    A kernel of the host with a reduce, for a renderer that knows its vector
    registers ({!Renderer.t.registers}), counts its lanes in registers
    ({!held_registers}): its upcasts after the masked and shared ones, and
    its unroll, take only what keeps its accumulators within half the
    registers, each conversion within 64 lanes, those of the upcast axes
    its value reads that hold distinct values ([n] lanes of an axis read as
    a quotient by [d] hold [n / gcd n d], and a reduce's lanes are summed
    before it), and an iteration within 512 lanes, the product of its upcast
    axes, the unrolled ones among them; an output upcast counts too the lanes
    an unroll of the last unrollable axis would take, the axis whole when it
    holds at most 32, 4 otherwise. A file of no fewer registers, each no
    narrower, takes no fewer output lanes. These upcasts apply however few
    elements the upcastable axes hold.

    The first upcasts the axis its stores write at unit stride, when every
    access that reads the axis reads it at unit stride, by the largest power
    of two, at most 16, that keeps the lanes within a vector of the
    accumulators' type, divides the axis and fits; the upcasts by 3 or 4
    follow while they fit, whatever the lanes; when none is left, one more,
    by 2, of an output axis that an access reading a reduce axis does not
    index, while it fits. The unroll takes what fits while the lanes upcast
    fill fewer registers than the target has, and next to a wholly unrolled
    axis of at most 3, unrolls the next reduce axis by the largest power of
    two that keeps the unrolled lanes within a vector of the reduce's type.

    A kernel of the host without a reduce takes in place of the upcasts after
    its masked axes one upcast of its last output axis that 2 divides, by the
    largest power of two, at most 16, that keeps its lanes within a vector
    of the widest type it reads, writes or computes, and again while they
    have room. It takes none when its stored values hold 5 operations or
    more for each operation of their longest chain, or when it computes a
    type the target emulates. *)

val hand_coded_optimizations : Postrange.Scheduler.t -> Postrange.Scheduler.t
(** [hand_coded_optimizations k] is a copy of [k] with the optimisations these
    rules pick; [k] itself is left as it is. *)

val core_fits :
  Renderer.t ->
  dtype_in:Dtype.t ->
  dtype_out:Dtype.t ->
  rows:int ->
  outputs:int ->
  block:int ->
  bool
(** [core_fits ren ~dtype_in ~dtype_out ~rows ~outputs ~block] is [true] iff a
    contraction of [dtype_in] operands of one free axis each, of extents [rows]
    and [outputs], accumulated in [dtype_out] and summed over [block] of them
    at a time, leaves streaming ({!section-streaming}) to take a tensor core of
    [ren]: the lesser of [rows] and [outputs] is past the reuse that streams
    first, {!Setting.use_tc} is [1], and a core of those dtypes, the one
    {!Setting.tc_select} names if any (none where [ren] has no such core), a
    float32 core of CUDA only under {!Setting.allow_tf32}, has its [K] dividing
    [block] and its [M] and [N] dividing [rows] and [outputs], in either role.
    A caller that writes a contraction asks it to choose a form for the layout
    the kernel will take. *)

val core_rows :
  Renderer.t ->
  dtype_in:Dtype.t ->
  dtype_out:Dtype.t ->
  outputs:int ->
  block:int ->
  int option
(** [core_rows ren ~dtype_in ~dtype_out ~outputs ~block] is the fewest rows of
    a contraction as {!core_fits} takes it, of [outputs] outputs, that a
    tensor core of [ren] takes with the rows along its [M]: the least multiple
    of a core's [M] past the reuse that streams first ({!section-streaming}),
    the least over the cores {!core_fits} would try whose [N] divides
    [outputs]. It is [None] where no such core is tried or [outputs] is not
    past that reuse. {!core_fits} holds of these rows. A caller that cuts rows
    into blocks asks it for the smallest block a core takes with the rows
    along its [M]. *)

val held_registers : Postrange.Scheduler.t -> Renderer.registers -> int
(** [held_registers k file] is the registers of [file] that [k]'s reduces'
    accumulators take: one accumulator for each output lane of each reduce. A
    register of [file.bytes] holds as many lanes of a reduce's type as fit, at
    4 bytes for a type narrower than 4, which C computes as an [int] or a
    [float], and no more than [k]'s stores write at consecutive addresses as
    their indices read now: a term a later split nested, as
    [(a * 4 + b) * 3 + c], is no consecutive run, so the count can exceed
    what the budget judged before that split. A kernel without a reduce holds
    nothing. *)

(**/**)

val hand_coded_with :
  core_threads:int -> Postrange.Scheduler.t -> Postrange.Scheduler.t
(** [hand_coded_with ~core_threads k] is {!hand_coded_optimizations} with the
    renderer's [core_threads] made [core_threads]. *)
