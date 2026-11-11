(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Operations lowered to tolk, compiled and run on the host. *)

val run : ?local:bool -> (unit -> ('v, 's, 'd) Nx.t) -> 'v array
(** [run f] is the elements, in C order, of [f ()] computed by lowering each
    operation [f] applies ({!Rune.Lower.op}) into one tolk graph, compiled and
    run on the host. A value [f] did not compute is bound as storage that holds
    its elements, read through its layout where it is a view of an array's
    storage. [local] (default [true]) is the lowering's [local]: with
    [~local:false] it lowers as for a device whose work is not this process's
    code, though the graph still runs on the host. *)

val work : (unit -> ('v, 's, 'd) Nx.t) -> int * int
(** [work f] is the arithmetic operations, and the bytes loaded and stored,
    that the kernels of {!run}'s schedule for [f] are estimated to perform
    ({!Tolk.Realize.estimate_uop}), summed over its calls.

    Raises [Invalid_argument] if the schedule runs calls in a loop. *)

val kernels : (unit -> ('v, 's, 'd) Nx.t) -> int
(** [kernels f] is the number of kernels of {!run}'s schedule for [f]. *)

val sources :
  target:Tolk.Helpers.Target.t ->
  (unit -> ('v, 's, 'd) Nx.t) ->
  (string * Tolk.Ops.t * string) list
(** [sources ~target f] is each kernel of the schedule that computes [f ()]
    on [target]'s device, lowered as for a device whose work is not this
    process's code: its name, the sink it was compiled from and the source
    [target]'s renderer writes for it, made without the target's toolchain
    ({!Tolk.Cstyle.without_compiler}), in the order the schedule calls
    them. *)
