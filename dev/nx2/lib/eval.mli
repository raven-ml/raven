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
    traced operand of another interpretation that is not around it, but one
    it adopts, which is around by construction. On another domain than the
    one it was installed on, an adopting interpretation's handler answers an
    operation it reaches only where its parent is around too, and refuses it
    otherwise. An
    operation no handler answers is {!Exec.run}'s, and raises where an operand
    is traced.

    An [Extent] receiver first receives the operations of each formula operand
    its rule did not return, rebuilt at the operand's brand, in dependency
    order, each node once per operand; the operation reads its results. Each
    donated handle reaches a rule as its arrays, lent ({!Donate.lent}), and
    its donor lives on: only {!Exec} consumes handles. Other operands reach
    the rule as they are.

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
    returns or raises.

    Raises [Invalid_argument] naming [name] and [p], with [Adopting p], if [p]
    is an [Extent], returned, or is not around the caller. *)

val traced :
  interpretation -> ('v, 's, 'd) form -> ('v, 's, 'd) payload -> ('v, 's, 'd) t
(** [traced i f p] is a value of form [f] that [i] owns, keeping [p]. *)

val payload : interpretation -> ('v, 's, 'd) t -> ('v, 's, 'd) payload option
(** [payload i x] is [Some p] iff [i] owns [x], [p] what it keeps. *)

val owner : ('v, 's, 'd) t -> interpretation option
(** [owner x] is the interpretation that owns [x], if [x] is traced. *)

val quiet : unit -> bool
(** [quiet ()] is [true] iff the count of live [Extent] interpretations is
    zero: none is live, on any domain. *)

val once :
  by:string ->
  'd any list ->
  (('v 's 'e. ('v, 's, 'e) t -> ('v, 's, 'e) t) -> 'a) ->
  'a
(** [once ~by xs k] is [k read], [k] the function [by] that reads its
    operands [xs] in several operations. [by] consumes the handles among [xs]
    when called ({!Donate.consume}), and [read] gives [xs] as they were then,
    each handle and each of their donors claimed for the call
    ({!Donate.read}): the donors are dead from the start, whatever [k]
    does, [k]'s operations write into none of their memory, and an
    interpretation that receives one of them reads its arrays lent. Two reads
    of one donor share its memory, so [k] may return both. Once [k] returns or
    raises, the claimed values are live, and the memory returns to its device
    once no value holds it. Raises [Invalid_argument] before [k] as
    {!Donate.consume} does, naming [by]. *)

val once_value :
  by:string ->
  'd any list ->
  (('v 's 'e. ('v, 's, 'e) t -> ('v, 's, 'e) t) -> ('w, 'q, 'd) t) ->
  ('w, 'q, 'd) t
(** [once_value ~by xs k] is {!once}[ ~by xs k], whose value [y] tells which
    memory is still held: where [by] consumed the handles, their memory that
    [y] does not hold and no other value shares is freed when [k] returns, as
    {!Exec} frees what an operation does not reuse ({!Donate.consumed}). A
    view [k] makes of it shares it, as does an interpretation's read: that
    memory waits for a collection. *)

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
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  acc:Nx_array.Dtype.any ->
  ('v, 's) dtype ->
  ('a, 'b, 'd) t ->
  ('c, 'e, 'd) t ->
  ('v, 's, 'd) t option ->
  ('v, 's, 'd) t
(** [contract ~by ~batch ~contracting ~acc dt a b init] is {!eval} of the
    contraction [Contract] of [a] and [b] from [init] into [dt], in the order
    of summation {!Prim.contract_order} gives. With no live [Extent], it is
    {!Exec.contract} where that answers. *)

val place :
  by:string -> 'e Devices.placement -> ('v, 's, 'd) t -> ('v, 's, 'e) t
(** [place ~by p x] is {!eval} of [Place (p, x)]. With no live [Extent], a
    value on one device already at [p] answers at once, over
    its array. *)

val expand :
  interpretation ->
  by:string ->
  repeat:
    ('d.
    int ->
    ('d index -> 'd any list -> 'd any list) ->
    'd any list ->
    'd any list) ->
  'r prim ->
  'r option
(** [expand i ~by ~repeat op] is {!Expand.run} over {!eval} under [i]'s
    handler again: [op] as simpler operations, which reach [i] again, its
    loops run by [repeat].

    Raises [Invalid_argument] naming [by] and [i] if [i] returned. *)
