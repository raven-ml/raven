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
    through {!Nx_array.refused} naming [by]; a map the kernels decline runs as
    one kernel call per node on each device, over its own operands, and a kernel
    a node needs that the kernels decline raises naming the kernels, the kind,
    the dtypes and the device. Movements and bitcasts are views where a layout
    expresses them, and a copy, then a view, otherwise. [Check] raises its
    exception from the first failing index, read on the host.

    A program that reads coordinates computes each device's window with that
    window's first index added to them. *)

val run : by:string -> 'r Value.prim -> 'r
(** [run ~by op] is [op]'s meaning where no interpretation reaches it. *)

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
(** [applyN ~by k dt x …] is {!run} of the one-node map [k] over [x …] with
    result dtype [dt] (the second operand's for [apply3]). For operands of one
    shape on one device at physically one placement it builds no operation: it
    checks the kind's dtypes, allocates the result, sharing the first operand's
    layout when that is C-contiguous at offset 0 and of the result's dtype, and
    calls the set's kernel. *)

val at : 'd Devices.placement -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t
(** [at p c] is the constant [c] computed at [p]: a map, a creation included, at
    [p] itself, each device its window; any other operation whole on each device
    of [p]'s set, each device keeping its window. Its operands are computed the
    same way. A node, however many values or chains share it, computes once per
    placement within a domain; across domains, at most once per domain whose
    first use races, every result equal bit for bit and the first one stored
    kept. It takes no lock. A kernel's refusal or decline raises here, naming
    the function that made the constant. At {!Devices.anywhere} it computes on
    the host. A value that is not a constant is returned as it is. *)
