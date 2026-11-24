(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Attention's plans (nx_metal.mli's Attention). *)

val keeps_all : Nx_kernel.Prog.t -> bool
(** [keeps_all p] is [true] where [p]'s output 2 is the constant [true], so that
    every visible pair is kept. A launch reads only outputs 0 and 1: nx.metal
    declines a program of any other output 2. *)

val bound_offsets : int -> unit
(** [bound_offsets n] is nx_metal.mli's [bound_offsets]. *)

val plan :
  holds:(Kernels.t -> int) ->
  Nx_kernel.Prog.t ->
  scale:float ->
  paged:bool ->
  lengths:bool ->
  Operands.t ->
  (Plan.t option, Nx_array.answer) result
(** [plan ~holds p ~scale ~paged ~lengths xs] plans the attention by [p] of the
    call of facts [xs]: destinations [o] and [lse], then loads [q], [k] and [v],
    the table where [paged], the lengths where [lengths], and [p]'s loads. It is
    [Ok None] for a call of no row and [Error Declined] where a unit's estimate
    exceeds {!Plan.allowed} or Loop declines a temporary of [p]'s.

    nx.metal has checked the call: its dtypes, [p] and its shapes fit, and its
    accumulator is [float32]. *)
