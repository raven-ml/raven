(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dispatch: an operation's one entry, from nx's functions to its meaning.

    An operation goes to the interpretation {!Interp.receiver} gives. An
    [Extent] receiver first receives the operations of each formula operand its
    rule did not return, rebuilt at the operand's brand, in dependency order,
    each node once per operand; the operation reads its results. Other operands
    reach the rule as they are. An operation no interpretation reaches is
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
    {!Prim.op3}). With no live [Extent] on the calling domain and no traced
    operand, it is {!Exec.apply1} to {!Exec.apply3}, which build no operation
    on their fast path. *)

val contract :
  by:string ->
  batch:(int * int) array ->
  contracting:(int * int) array ->
  acc:Nx_array.Dtype.any ->
  ('v, 's) dtype ->
  ('a, 'b, 'd) t ->
  ('c, 'e, 'd) t ->
  ('v, 's, 'd) t option ->
  ('v, 's, 'd) t
(** [contract ~by ~batch ~contracting ~acc dt a b init] is {!eval} of the
    contraction [Contract] of [a] and [b] from [init] into [dt]. With
    no live [Extent] on the calling domain, it is {!Exec.contract} where that
    answers. *)

val place :
  by:string -> 'e Devices.placement -> ('v, 's, 'd) t -> ('v, 's, 'e) t
(** [place ~by p x] is {!eval} of [Place (p, x)]. With no live [Extent] on the
    calling domain, a value on one device already at [p] answers at once, over
    its array. *)

val expand : interpretation -> by:string -> 'r prim -> 'r option
(** [expand i ~by op] is {!Expand.run} over {!eval}, with [i] not running: an
    optional case as core operations, which reach [i] again. *)
