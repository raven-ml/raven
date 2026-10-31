(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Operations lowered to tolk, compiled and run on the host. *)

val run : (unit -> ('v, 's, 'd) Nx.t) -> 'v array
(** [run f] is the elements, in C order, of [f ()] computed by lowering each
    operation [f] applies ({!Rune.Lower.op}) into one tolk graph, compiled and
    run on the host. A value [f] did not compute is bound as storage that holds
    its elements, read through its layout where it is a view of an array's
    storage. *)

val work : (unit -> ('v, 's, 'd) Nx.t) -> int * int
(** [work f] is the arithmetic operations, and the bytes loaded and stored,
    that the kernels of {!run}'s schedule for [f] are estimated to perform
    ({!Tolk.Realize.estimate_uop}), summed over its calls.

    Raises [Invalid_argument] if the schedule runs calls in a loop. *)
