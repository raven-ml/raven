(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dispatch: an operation's one entry, from nx's functions to its meaning.

    An operation goes to the interpretation {!Interp.receiver} gives, each
    constant operand first computed at the placement where the operation's route
    reads it ({!Exec.at}). An operation over values of every set alone reaches
    the interpretation with them as they are: there is no placement to compute
    them at. A check's data is read on the host. An operation no interpretation reaches is
    {!Exec.run}'s. *)

open Value

val eval : by:string -> 'r prim -> 'r
(** [eval ~by op] is [op]'s meaning under the rule. Errors name [by]. *)

val apply1 :
  by:string ->
  Nx_kernel.Prog.op1 ->
  ('w, 'r) dtype ->
  ('v, 's, 'd) t ->
  ('w, 'r, 'd) t

val apply2 :
  by:string ->
  Nx_kernel.Prog.op2 ->
  ('w, 'r) dtype ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('w, 'r, 'd) t

val apply3 :
  by:string ->
  Nx_kernel.Prog.op3 ->
  ('a, 'b, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [applyN ~by k dt x …] is {!eval} of the one-node map [k] ({!Prim.op1} to
    {!Prim.op3}). With no live [Extent] on the calling domain and every operand
    an [Array], it is {!Exec.apply1} to {!Exec.apply3}, which build no operation
    on their fast path. *)

val place :
  by:string -> 'e Devices.placement -> ('v, 's, 'd) t -> ('v, 's, 'e) t
(** [place ~by p x] is {!eval} of [Place (p, x)]. With no live [Extent] on the
    calling domain, a value on one device already at [p] answers at once, over
    its array. *)

val expand : interpretation -> by:string -> 'r prim -> 'r option
(** [expand i ~by op] is {!Expand.run} over {!eval}, with [i] not running: an
    optional case as core operations, which reach [i] again. *)
