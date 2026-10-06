(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Records of a body's run, and their replays.

    A reverse derivative runs a body that its backward pass needs again, a
    loop's step or a remat's function, once at its call, and keeps a {e record}
    of that run: a program over the run's inputs. Each {e entry} is an operation
    of nx or a construct that left the run, every operand named as an input, an
    earlier entry's result, or a value from outside the run:
    - an operation, performed again over the replayed operands;
    - a collective of a map or a detach ([Lanes], [Lane_index], [Detach]), which
      an installation around the run answered, performed again;
    - a construct that carries a function, which holds the records of its
      functions, replayed by performing the construct again over replayed
      operands, so the installations around the replay transform it as those
      around the run did: a loop's step and stop, a remat's function, a custom
      rule's rule and tangent map, a root's solve and residual, and its
      [linear_solve], whose applications of the operator it receives are entries
      the replay applies its own operator to. A loop that took no step runs it
      once at its final carry, its additions dropped, for its record.

    A compiled call runs inline; additions to totals are not recorded. A replay
    runs no code of the body but a [custom_vjp] pullback, a function of
    cotangents, which reads the replayed values: the installations around the
    replay interpret its operations and constructs, so an outer differentiation
    differentiates it and a map batches it. *)

type t
(** The type for records. *)

val results : 'r Nx.Op.t -> 'r -> Nx.packed list
(** [results op r] is the tensors of [op]'s result [r]. *)

val run : Nx.packed list -> (unit -> 'r) -> 'r * t
(** [run inputs f] is [f ()], every operation and construct of its extent
    recorded, and the record over [inputs]. *)

val rename : t -> Nx.Op.mapper
(** [rename r] maps a value [r]'s run made, or an input, to its name in [r], and
    any other value to itself: the coefficients of a tape the run recorded name
    the record's entries ({!Linear.rename}), so that the record holds no value
    of its run. *)

val replay : t -> Nx.packed list -> (unit -> 'a) -> 'a
(** [replay r inputs f] performs [r]'s entries at [inputs], in the caller's
    interpretation, and is [f ()] with every name of [r] and every value of
    [r]'s run that its extent reads replaced by its replayed value: the
    coefficients of a renamed tape, and the values a pullback reads through its
    closure. A value of the run is known weakly: one nothing keeps alive any
    more, no pullback reads.

    Raises [Invalid_argument] if [inputs] differ from the run's in number, and
    if the replay applies a function of a construct that no transformation ran
    when it was recorded. *)
