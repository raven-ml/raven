(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Interpretations: scopes that give operations another meaning, and the rule
    that delivers each operation to one.

    An operation goes to the innermost live interpretation that reaches it.
    Every interpretation reaches the operations on its traced values. An
    [Extent] interpretation also reaches every operation its starting fiber
    applies inside its extent, on its domain. Innermost is by start order, which
    counts per domain. An interpretation does not reach the operations its own
    running rule applies, and that rule applying an operation to its own traced
    values raises.

    Each domain counts its live [Extent] interpretations in an atomic of its
    domain-local state, and one atomic counts them on every domain. With that
    one at zero an operation reads it and nothing else; with the domain's count
    at zero, the two. Otherwise it performs one effect, which the innermost [Extent]
    handler on the calling fiber answers if its rule is not running and its
    domain is the caller's; other handlers pass it on. The handler is the truth
    and the count a hint that is never low. A continuation dropped inside an
    extent leaves the count high, which costs every later operation on the
    domain a perform and changes no result. *)

open Value

val interpret : name:string -> reach -> rule -> (interpretation -> 'a) -> 'a
(** [interpret ~name reach r f] is [f i], [i] a new interpretation that gives
    the operations it reaches the meaning [r]. [i] is live until [f] returns or
    raises. *)

val traced :
  interpretation -> ('v, 's, 'd) form -> ('v, 's, 'd) payload -> ('v, 's, 'd) t
(** [traced i f p] is a value of form [f] that [i] owns, keeping [p]. *)

val payload : interpretation -> ('v, 's, 'd) t -> ('v, 's, 'd) payload option
(** [payload i x] is [Some p] iff [i] owns [x], [p] what it keeps. *)

val owner : ('v, 's, 'd) t -> interpretation option
(** [owner x] is the interpretation that owns [x], if [x] is traced. *)

val later : interpretation -> interpretation -> bool
(** [later a b] is [true] iff [a] started after [b]. Raises [Invalid_argument]
    if they started on two domains. *)

val quiet : unit -> bool
(** [quiet ()] is [true] iff the calling domain has no live [Extent]
    interpretation. It allocates nothing. *)

val receiver : by:string -> 'r prim -> interpretation option
(** [receiver ~by op] is the interpretation the rule delivers [op] to, [None] if
    none reaches it.

    Raises [Invalid_argument] naming [by] and the interpretation for a traced
    operand whose interpretation has returned, started on another domain, or is
    running. Each traced operand is checked before the innermost is chosen. *)

val apply : interpretation -> by:string -> 'r prim -> 'r
(** [apply i ~by op] is [i]'s rule on [op], with [i] running until it returns or
    raises. *)

val expanding : interpretation -> (unit -> 'a) -> 'a
(** [expanding i f] is [f ()] with [i] not running, so that the operations of an
    expansion reach [i] again. *)
