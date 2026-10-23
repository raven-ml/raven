(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The eager engine: operations on values, computed now by their set's kernels,
    or held as constants.

    {!run} applies an operation's rule ({!Prim.results}) before it allocates or
    calls anything. An operation whose every operand is a constant, a creation
    included, gives constants; [Place] and [Check] compute. Otherwise it
    computes each constant operand where the route reads it, places operands the
    route moves, and allocates each result at the route's placement. Each device
    of that placement then computes its window of the results by the set's
    kernels, over each operand's window there ({!Route.window}); a program that
    reads coordinates adds the window's first index to them. Where the route
    computes by partials ({!Route.Partials}), {!Partial} computes it
    instead. [Done] is the result; a refusal raises
    through {!Nx_array.refused} naming [by]. A decline raises
    [Invalid_argument] naming [by], the kernels, the operation (a one-node
    map's kind, a routine's name, what a reduction or a scan computes, as
    ["Reduce Sum and Moments"]), the dtypes, the device and the move the
    program makes, [Nx.place] onto a set whose kernels compute it: nx
    computes no case on the kernels' behalf and moves no data between
    devices on its own. A result
    shares an operand's memory only as its view, or where the operand was
    donated. Movements and bitcasts are views where a
    layout expresses them, and a copy, then a view, otherwise. [Check] reduces
    its flags where they lie, to whether all hold and the first failing index.
    One flag is its own reduction: its element, on the device of the window
    that holds it, as a 0-d view, and index [0]; a pending one holds the
    flag's memory as it holds its data's.
    On the host it reads them and raises at once; on another set it holds them
    and its data, unread, on the calling domain's list of the set's pending
    checks, which the domain's systhreads share, until a {!wait} on the set
    drains them. There, a device of this machine whose memory the host does
    not address copies whether all hold, in its order, into [Pinned] memory
    ({!Rig.Buffer.kind}), which both address: one more copy submitted per
    check, which nothing waits for. On another machine's device the host
    reads it through a copy, which follows that device's later work, so
    settling such a check waits for the work submitted there; no rig read
    avoids that yet. The check records its result's device's submitted value
    ({!Rig.submitted}). Issuing a check on the set first settles its earlier
    checks whose result is written: on a device the new check computes on,
    once the device has reported that value ({!Rig.signaled}), which commits
    no work and waits for none; on another, which may get no more work, as
    {!Rig.Buffer.reached} finds it, which commits that device's work. A
    passing one goes and a failing one keeps only its exception. A device
    reports its work at most its driver's batch of values late while work is
    submitted on it, so the checks no wait follows hold the data of the work
    in flight and of that batch; on a domain whose submits stop, they hold it
    until a wait. A failing check's elements are read through a copy where
    the host does not address them. The new check is computed and kept only if no failure of
    the set is known then, and a raise while settling leaves it unissued. A
    [Place] onto the host from another set is a {!wait} on that set first.

    Over devices of one architecture, a result's bits are those of the same
    operation on one device, by the window law ({!Route.window}) or by
    {!Partial}, wherever the operands lie. *)

val run : by:string -> 'r Value.prim -> 'r
(** [run ~by op] is [op]'s meaning where no interpretation reaches it.

    Raises [Invalid_argument] naming [by] for a traced operand: its
    interpretation receives the operation first ({!Eval.eval}). *)

val apply1 :
  by:string ->
  Nx_kernel.Prog.op1 ->
  ('w, 'r) Value.dtype ->
  ('v, 's, 'd) Value.t ->
  ('w, 'r, 'd) Value.t

val apply2 :
  by:string ->
  Nx_kernel.Prog.op2 ->
  ('w, 'r) Value.dtype ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t ->
  ('w, 'r, 'd) Value.t

val apply3 :
  by:string ->
  Nx_kernel.Prog.op3 ->
  ('a, 'b, 'd) Value.t ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t
(** [applyN ~by k dt x …] is the one-node map [k] over [x …], of one shape,
    with result dtype [dt] (the second operand's for [apply3]), where no
    interpretation reaches it. For live arrays at physically one placement
    whose set has kernels, where [k] takes their dtypes, it builds no
    operation: it allocates the result, sharing the first operand's layout
    when that is C-contiguous at offset 0 and of the result's dtype, and calls
    the kernels' [map] once, with the program {!Nx_kernel.Prog.of_node} makes
    of [k], made once per kind and dtypes. A decline raises as in {!run}, and
    a refusal naming [by]. Otherwise it is {!run} of the operation
    ({!Prim.op1} to {!Prim.op3}), whose rule raises on operands it does not
    take; a traced operand raises as in {!run}. *)

val at : 'd Devices.placement -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [at p c] is the constant [c] computed at [p] for an operation that reads it:
    each device computes its window, whole along each axis [c]'s operation keeps
    whole ({!Route.keeps}), and keeps its window in memory of its own. [c] keeps its results at [p]: it computes once per
    placement within a domain; across domains, at most once per domain whose
    first use races, every result equal bit for bit and the first one stored
    kept. It takes no lock. A movement or a bitcast of a constant keeps nothing:
    it is its operand read at [p], viewed. The constants [c] is computed from
    are taken from their own results where an operation read them at the
    placement [c] reads them, and are otherwise computed for this alone, once
    each, and dropped. A kernel's refusal or decline raises here, naming the
    function that made the constant. A value that is not a constant is returned
    as it is. *)

val wait : by:string -> ('v, 's, 'd) Value.t -> unit
(** [wait ~by x] waits for the work [x]'s arrays depend on, then drains the
    checks the calling domain issued on [x]'s set: it reads them in issue order
    up to the first that fails, which raises its exception, and drops the later
    ones unread. Each drained check's hold on its data ends then
    ({!Rig.Claim.release}). A constant has no set: it returns.

    Raises [Invalid_argument] naming [by] for a traced [x]. *)

val on_host : by:string -> ('v, 's, 'd) Value.t -> ('v, 's) Nx_array.t
(** [on_host ~by x] is an array on the host holding [x]'s elements: a constant
    computed there, a concrete value placed there ({!Place.value}), over [x]'s
    own array where it is the host's. No interpretation receives it, and it
    settles no check ({!wait}).

    Raises [Invalid_argument] naming [by] for a traced [x], and what
    {!Place.value} raises. *)

val contract :
  by:string ->
  order:Nx_kernel.order ->
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  acc:Nx_array.Dtype.any ->
  ('v, 's) Value.dtype ->
  ('a, 'b, 'd) Value.t ->
  ('c, 'e, 'd) Value.t ->
  ('v, 's, 'd) Value.t option ->
  ('v, 's, 'd) Value.t option
(** [contract ~by ~order ~batch ~contracting ~acc dt a b init] is [Some] the
    contraction [Value.Contract] of [a] and [b] from [init] into [dt], in
    [order], where no interpretation reaches it, its operands are live arrays
    on one device at physically one placement whose set has kernels, its pairs
    fit them: it builds no operation and allocates the C-contiguous result of
    the shape {!Nx_kernel.Shape.contract} gives. [None] otherwise, for the
    operation to decide. A refusal of the kernels raises naming [by], and a
    decline as in {!run}. *)
