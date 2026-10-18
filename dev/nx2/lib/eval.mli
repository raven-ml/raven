(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dispatch: an operation's one entry, from nx's functions to its meaning.

    Each interpretation's extent runs under a handler of its own. An operation
    that a live [Extent] or a traced operand may send to an interpretation
    performs one effect, which the innermost handler whose interpretation
    reaches it answers: by running the rule in its clause, on the handler's
    parent fiber, outside the interpretation and every handler inside it, with
    the extent suspended until the rule returns. The receiver first refuses a
    traced operand of another interpretation that is not around it. An
    operation no handler answers is {!Exec.run}'s, and raises where an operand
    is traced.

    An [Extent] receiver first receives the operations of each formula operand
    its rule did not return, rebuilt at the operand's brand, in dependency
    order, each node once per operand; the operation reads its results. The
    operation is the final consumer of its donated handles, a movement too: it
    kills their donors ({!Exec.consume}), and each handle reaches the rule as
    its arrays. Other operands reach the rule as they are.

    One atomic counts the live [Extent] interpretations of the process, on
    every domain: a continuation may resume on any domain, so no domain's own
    count could tell an extent is absent. With the count at zero, an operation
    without a traced operand reads it and nothing else. The count is a hint
    that is never low. While any [Extent] lives, on any domain, every operation
    of every domain performs the dispatch effect, which costs it a perform and
    changes no result; a continuation dropped inside an extent keeps the count
    high for the life of the process. *)

open Value

val interpret :
  name:string ->
  reach ->
  ('r. interpretation -> by:string -> 'r prim -> 'r) ->
  (interpretation -> 'a) ->
  'a
(** [interpret ~name reach rule f] is [f i] under [i]'s handler, [i] a new
    interpretation that gives the operations it reaches the meaning
    [rule i ~by op]. [i] is live, and counted if [reach] is [Extent], until [f]
    returns or raises. *)

val traced :
  interpretation -> ('v, 's, 'd) form -> ('v, 's, 'd) payload -> ('v, 's, 'd) t
(** [traced i f p] is a value of form [f] that [i] owns, keeping [p]. *)

val payload : interpretation -> ('v, 's, 'd) t -> ('v, 's, 'd) payload option
(** [payload i x] is [Some p] iff [i] owns [x], [p] what it keeps. *)

val owner : ('v, 's, 'd) t -> interpretation option
(** [owner x] is the interpretation that owns [x], if [x] is traced. *)

val later : interpretation -> interpretation -> bool
(** [later a b] is [true] iff [a] started after [b], on any domains. Of two
    interpretations around one operation, the later is inside the other. *)

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
    {!Prim.op3}). With no live [Extent] and no traced operand, it is
    {!Exec.apply1} to {!Exec.apply3}, which build no operation on their fast
    path. *)

val contract :
  by:string ->
  shape:int array ->
  rows:int ->
  columns:int ->
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  acc:Nx_array.Dtype.any ->
  ('v, 's) dtype ->
  ('a, 'b, 'd) t ->
  ('c, 'e, 'd) t ->
  ('v, 's, 'd) t option ->
  ('v, 's, 'd) t
(** [contract ~by ~shape ~rows ~columns ~batch ~contracting ~acc dt a b init]
    is {!eval} of the contraction [Contract] of [a] and [b] from [init] into
    [dt], whose result has [shape] and whose rows and columns are [rows] and
    [columns] ({!Prim.contract_order_of}), as its caller planned them from
    operands and an [acc] it checked against the contraction's rule. With no
    live [Extent], it is {!Exec.contract} where that
    answers. *)

val place :
  by:string -> 'e Devices.placement -> ('v, 's, 'd) t -> ('v, 's, 'e) t
(** [place ~by p x] is {!eval} of [Place (p, x)]. With no live [Extent], a
    value on one device already at [p] answers at once, over
    its array. *)

val expand : interpretation -> by:string -> 'r prim -> 'r option
(** [expand i ~by op] is {!Expand.run} over {!eval} under [i]'s handler again:
    an optional case as core operations, which reach [i] again.

    Raises [Invalid_argument] naming [by] and [i] if [i] returned. *)
