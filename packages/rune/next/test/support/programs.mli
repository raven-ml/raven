(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Traced values compiled for the host and run. *)

open Rune_next

val compiled : Lower.scope -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [compiled s y] is [y] as {!Traces.value} gives it, computed by its graph
    scheduled, compiled for the host and run. The graph must schedule to one
    kernel.

    Raises [Invalid_argument] if it schedules to several, and as the compiler
    does. *)

val kernels : ('a, 'b) Nx.t -> Tolk_next.Ops.t
(** [kernels y] is the sink of the kernels, in order, that the schedule storing
    the traced [y] into a new buffer on the host runs. *)
