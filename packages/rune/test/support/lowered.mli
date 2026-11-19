(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Operations lowered to tolk, compiled and run on the host. *)

val run :
  ?local:bool ->
  ?cores:(string * Tolk.Tc.t list) list ->
  (unit -> ('v, 's, 'd) Nx.t) ->
  'v array
(** [run f] is the elements, in C order, of [f ()] computed by lowering each
    operation [f] applies ({!Rune.Lower.op}) into one tolk graph, compiled and
    run on the host. A value [f] did not compute is bound as storage that holds
    its elements, read through its layout where it is a view of an array's
    storage. [local] (default [true]) is the lowering's [local]: with
    [~local:false] it lowers as for a device whose work is not this process's
    code, though the graph still runs on the host. [cores] are the tensor
    cores the schedule takes each device to have
    ({!Tolk.Schedule.create_linear_with_vars}), by default the host's own
    ({!Tolk.Heuristic.val-cores} of its renderer): given others for ["CPU"], the
    host's schedule keeps apart the products such cores take, though its
    kernels run without them. *)

val map : (Tolk.Ops.t -> Tolk.Ops.t) -> (float, 's, 'd) Nx.t -> float array
(** [map f x] is the elements, in C order, of [f] applied to the node of
    [x]'s elements, a [float32] or [float64] array, compiled and run on the
    host. [f] keeps the node's shape and type. *)

val work : (unit -> ('v, 's, 'd) Nx.t) -> int * int
(** [work f] is the arithmetic operations, and the bytes loaded and stored,
    that the kernels of {!run}'s schedule for [f] are estimated to perform
    ({!Tolk.Realize.estimate_uop}), summed over its calls.

    Raises [Invalid_argument] if the schedule runs calls in a loop. *)

val kernels :
  ?cores:(string * Tolk.Tc.t list) list -> (unit -> ('v, 's, 'd) Nx.t) -> int
(** [kernels ~cores f] is the number of kernels of {!run}'s schedule for
    [f], given [cores] as {!run} is. *)

val sources :
  target:Tolk.Helpers.Target.t ->
  (unit -> ('v, 's, 'd) Nx.t) ->
  (string * Tolk.Ops.t * string) list
(** [sources ~target f] is each kernel of the schedule that computes [f ()]
    on [target]'s device, its tensor cores those of [target]'s renderer
    ({!Tolk.Heuristic.val-cores}), lowered as for a device whose work is not this
    process's code: its name, the sink it was compiled from and the source
    [target]'s renderer writes for it, made without the target's toolchain
    ({!Tolk.Cstyle.without_compiler}), in the order the schedule calls
    them. *)

val program : target:Tolk.Helpers.Target.t -> Tolk.Ops.t -> Tolk.Ops.t option
(** [program ~target ast] is the program {!sources} linearized for [target]
    from the kernel's sink [ast] it gave, as {!Tolk.Codegen.linearize} gives
    it, or [None] where it gave no [ast] for [target]. *)

val derivative_kernels :
  target:Tolk.Helpers.Target.t ->
  (unit -> ('v, 's, 'd) Nx.t) ->
  (string * Tolk.Ops.t) list option list
(** [derivative_kernels ~target f] is, for each [Nx.attention] [f ()]
    computes, in turn, the kernels of its derivative's transpose that
    {!Rune.jit} would compile on [target]'s device, at the call's own [o]
    and [lse] and cotangents of ones, scheduled as {!sources} schedules
    them, or [None] where the derivative takes its default, and so has no
    kernels of attention's own ([Rune.attention_transposed]). Each is its
    sink's name ({!Tolk.Ops.kernel_info}), which a kernel of attention's
    own takes from its lowering, and its sink, unlinearized, in the order
    the schedule calls them. *)
