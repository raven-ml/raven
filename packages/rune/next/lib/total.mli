(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scopes of totals.

    A scope answers the additions to the totals it covers. Code that a scan runs
    away from where it is written, a staged scan's step, runs under a nested
    scope whose sum leaves as one more carry leaf, which the scope adds once the
    scan returns; a scan that nothing stages folds where it is written, inside
    the scope. *)

val collect :
  ('a, 'b) Construct.total ->
  zero:('a, 'b) Nx.t ->
  (unit -> 'r) ->
  'r * ('a, 'b) Nx.t
(** [collect t ~zero f] is [(f (), total)], where [total] is [zero] plus every
    addition [f] made to [t]. The scope marks its extent as a transformation's
    ({!Nx.Op.intercepted}), so a compiled function called inside it runs its
    function, whose additions reach the scope.

    Raises [Invalid_argument] at an addition whose shape is not [zero]'s. *)

val discarding : (unit -> 'r) -> 'r
(** [discarding f] is [f ()] with every addition [f] makes dropped, a staged
    scan's step's included: the additions of code run a second time, which
    counted the first. *)
