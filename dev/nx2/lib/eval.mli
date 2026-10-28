(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dispatch: an operation's one entry, from nx's functions to its meaning.

    An operation's {e eager meaning} is {!run}'s. Its operands must be alive,
    and none may be traced. Its rule is checked and routed once
    ({!Prim.val-route}) before anything is allocated, claimed or computed. Each
    operand is read where the route reads it, by views where it lies
    ({!Exec.placer}), and the operation computes by partials ({!Partial}) or
    by windows ({!Exec.windows}), as the route says. An operation of no
    operands computes where {!Prim.val-route} puts it, each device its window.
    The operation consumes the donated handles among its operands as {!Donate}
    states: it claims them after the route ({!Donate.val-claim}), its first
    result takes the memory of a handle that may hold it and lies where the
    route puts the results ({!Donate.reuse}), and the claim ends once the
    operation returns or raises ({!Donate.consumed}). One that raises frees
    the memory its first result took. A movement of a handle claims nothing.
    It passes on a handle of the same donor over the moved arrays. A [Place]
    onto the host from another set first waits on that set ({!Check.wait}),
    then places the value ({!Exec.place}). A [Check] is issued
    ({!Check.issue}).

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
    otherwise. An operation no handler answers is {!run}'s, and raises where
    an operand is traced.

    A second effect asks one interpretation's handler one of two questions
    about the caller. The first is whether the handler is around the
    caller. The second is whether it is around the caller with no [Extent]
    interpretation between them, and an [Extent] handler in between answers
    it [false]. Where that handler is absent, either answer is [false].

    Each donated handle reaches a rule as its arrays, lent ({!Donate.lent}),
    and its donor lives on: a rule kills nothing. Other operands reach the
    rule as they are.

    One atomic counts the live [Extent] interpretations of the process, on
    every domain: a continuation may resume on any domain, so no domain's own
    count could tell an extent is absent. With the count at zero, an operation
    without a traced operand reads it and nothing else. The count is a hint
    that is never low. While any [Extent] lives, on any domain, every operation
    of every domain performs the dispatch effect, which costs it a perform and
    changes no result; a continuation dropped inside an extent keeps the count
    high for the life of the process.

    This module lies above the rest of the engine: {!Exec}, {!Partial} and
    {!Check} compute what it dispatches, and nx's functions call it. *)

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

val twinned :
  interpretation -> ('v, 's, 'd) t -> ('v, 's, 'd) payload -> ('v, 's, 'd) t
(** [twinned i c p] is a value of [c]'s form that [i] owns, keeping [p], whose
    twin is [c]. *)

val read_through : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [read_through x] is the twin of a traced [x] whose owner is live and
    around the caller with no [Extent] interpretation between them, read
    through in turn, and [x] otherwise. It asks each owner once
    ({!interpret}'s handler). *)

val payload : interpretation -> ('v, 's, 'd) t -> ('v, 's, 'd) payload option
(** [payload i x] is [Some p] iff [i] owns [x], [p] what it keeps. *)

val owner : ('v, 's, 'd) t -> interpretation option
(** [owner x] is the interpretation that owns [x], if [x] is traced. *)

val quiet : unit -> bool
(** [quiet ()] is [true] iff the count of live [Extent] interpretations is
    zero: none is live, on any domain. *)

val run : by:string -> 'r prim -> 'r
(** [run ~by op] is [op]'s eager meaning, as stated above. No interpretation
    receives it.

    Raises [Invalid_argument] naming [by] for a dead or traced operand and
    where [op]'s operands break its rule, before any handle is claimed, and
    as {!Donate.val-claim}, {!Exec} and {!Check.issue} state. *)

val eval : by:string -> 'r prim -> 'r
(** [eval ~by op] is [op]'s meaning, as stated above: {!run}'s where no
    [Extent] is live and no operand is traced, and otherwise what the
    handlers around the caller make of it. Errors name [by]. *)

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
    {!Prim.op3}), with result dtype [dt] ([apply3]'s: its second operand's).
    With no live [Extent] and no traced operand, live arrays at physically
    one placement whose set has kernels, where [k] takes their dtypes,
    compute by one {!Exec.map} of the program {!Prim.node1} to {!Prim.node3}
    keep, which builds no operation and may share the layout of [x]
    ([apply3]'s second operand). Other operands compute by {!run} of the
    operation, whose rule raises on operands it does not take. *)

val apply :
  by:string -> Nx_kernel.Prog.t -> ('w, 'r) dtype -> ('v, 's, 'd) t ->
  ('w, 'r, 'd) t
(** [apply ~by p dt x] is {!eval} of the map of the program [p] over [x], of
    [x]'s shape, with result dtype [dt]; [p] must read one operand of [x]'s
    dtype and give [dt], as {!Prim.literal2}'s programs do. With no live
    [Extent], a live array whose set has kernels computes by one
    {!Exec.map}, which builds no operation, as {!apply1} does. *)

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
    contraction [Contract] of [a] and [b] from [init] into [dt]. With no
    live [Extent], it is {!Exec.contract} where that answers. *)

val place :
  by:string -> 'e Devices.placement -> ('v, 's, 'd) t -> ('v, 's, 'e) t
(** [place ~by p x] is {!eval} of [Place (p, x)]. With no live [Extent], a
    live value on one device already at [p] answers at once, over its array,
    whose memory it marks shared ({!Rig.Claim.share}). *)
