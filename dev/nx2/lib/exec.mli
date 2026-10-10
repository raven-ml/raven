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
    route moves, allocates each result at the route's placement, and calls the
    set's kernels once per device. [Done] is the result; a refusal raises
    through {!Nx_array.refused} naming [by]; a map, an assembly, a gather, a
    scatter or a sort of a sub-byte dtype, or a scatter whose targets may
    repeat, that a device's kernels decline runs as its expansion
    ({!Expand.run}) on that device, over its own operands. Any other decline
    raises naming the kernels, the kind, the dtypes, the device and the move
    the program makes, [Nx.place] onto a set whose kernels compute it: no data
    moves between devices on its own. The decline of a sort, a transform, a
    routine, a fold or a map with a padded load also says that its expansion
    is not available yet. Movements and
    bitcasts are views where a layout expresses them, and a copy, then a view,
    otherwise. [Check] raises its exception from the first failing index, read
    on the host.

    A program that reads coordinates computes each device's window with that
    window's first index added to them. *)

val run : by:string -> 'r Value.prim -> 'r
(** [run ~by op] is [op]'s meaning where no interpretation reaches it.

    Raises [Invalid_argument] naming [by] for a traced operand: its
    interpretation receives the operation first ({!Eval.eval}). *)

val apply1 :
  slow:
    (by:string ->
    Nx_kernel.Prog.op1 ->
    ('w, 'r) Value.dtype ->
    ('v, 's, 'd) Value.t ->
    ('w, 'r, 'd) Value.t) ->
  by:string ->
  Nx_kernel.Prog.op1 ->
  ('w, 'r) Value.dtype ->
  ('v, 's, 'd) Value.t ->
  ('w, 'r, 'd) Value.t

val apply2 :
  slow:
    (by:string ->
    Nx_kernel.Prog.op2 ->
    ('w, 'r) Value.dtype ->
    ('v, 's, 'd) Value.t ->
    ('v, 's, 'd) Value.t ->
    ('w, 'r, 'd) Value.t) ->
  by:string ->
  Nx_kernel.Prog.op2 ->
  ('w, 'r) Value.dtype ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t ->
  ('w, 'r, 'd) Value.t

val apply3 :
  slow:
    (by:string ->
    Nx_kernel.Prog.op3 ->
    ('a, 'b, 'd) Value.t ->
    ('v, 's, 'd) Value.t ->
    ('v, 's, 'd) Value.t ->
    ('v, 's, 'd) Value.t) ->
  by:string ->
  Nx_kernel.Prog.op3 ->
  ('a, 'b, 'd) Value.t ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t
(** [applyN ~slow ~by k dt x …] is the one-node map [k] over [x …], of one
    shape, with result dtype [dt] (the second operand's for [apply3]), where no
    interpretation reaches it. For operands on one device at physically one
    placement, and constants beside them, read there once [k] takes the
    operands' dtypes and shapes, it builds no operation: it allocates the
    result, sharing the first operand's layout when that is C-contiguous at
    offset 0 and of the result's dtype, and calls the set's kernel, which checks
    the dtypes and shapes. Otherwise, and where the kernel declines or refuses
    them, it is [slow ~by k dt x …], the operation built and evaluated, whose
    rule raises. *)

val at : 'd Devices.placement -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [at p c] is the constant [c] computed at [p] for an operation that reads it:
    a map, a creation included, at [p] itself, each device its window; any other
    operation whole on each device of [p]'s set, each device keeping its window
    in memory of its own. [c] keeps its results at [p]: it computes once per
    placement within a domain; across domains, at most once per domain whose
    first use races, every result equal bit for bit and the first one stored
    kept. It takes no lock. A movement or a bitcast of a constant keeps nothing:
    it is its operand read at [p], viewed. The constants [c] is computed from
    are taken from their own results where an operation read them at the
    placement [c] reads them, and are otherwise computed for this alone, once
    each, and dropped. A kernel's refusal or decline raises here, naming the
    function that made the constant. A value that is not a constant is returned
    as it is. *)

val donate : by:string -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [donate ~by x] is a handle over [x]'s memory that one operation reads
    ({!Value.Donated}): [x] lives until that operation is called, then dies with
    the handle, naming the operation. An elementwise operation writes its result
    into the memory, where rig holds it exclusive (no other value reads it) and
    it has the result's dtype and C-contiguous layout; the memory is consumed
    either way where no other value reads it. A movement that maps elements one
    to one passes a new handle on instead. A constant or a traced [x] is [x].

    Raises [Invalid_argument] naming [by] if [x] is dead. *)

val live : ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [live x] is a handle's arrays as a live value, for an interpretation that
    reads it without consuming it; any other value as it is. *)

val read : ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [read c] is [c] computed on the host's device into memory of its own: a
    value of every set computed for this library to read, its placement the
    host's whatever ['d]. It never reaches a function of nx's own as an operand.
*)

val on_host : by:string -> ('v, 's, 'd) Value.t -> ('v, 's) Nx_array.t
(** [on_host ~by x] is an array on the host holding [x]'s elements: a constant
    computed there, a concrete value placed there ({!Place.value}), over [x]'s
    own array where it is the host's. No interpretation receives it.

    Raises [Invalid_argument] naming [by] for a traced [x], and what
    {!Place.value} raises. *)

val contract :
  by:string ->
  Nx_kernel.Spec.contract Nx_kernel.Spec.t ->
  ('v, 's) Value.dtype ->
  ('a, 'b, 'd) Value.t ->
  ('c, 'e, 'd) Value.t ->
  ('v, 's, 'd) Value.t option ->
  ('v, 's, 'd) Value.t option
(** [contract ~by spec dt a b init] is [Some] the contraction [spec] of [a] and
    [b] from [init] where no interpretation reaches it, its operands are live
    arrays on one device at physically one placement whose set has kernels, they
    fit [spec], and the kernels compute it: it builds no operation and allocates
    the C-contiguous result. [None] otherwise, for the operation to decide. *)
