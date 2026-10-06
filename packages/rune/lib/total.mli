(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scopes of totals.

    A scope answers the additions to the totals it covers. A function a
    construct carries runs under the scope again ({!Construct.install}), so the
    scope meets its additions at its own level, inside every installation that
    answers the construct further out. A staged loop's step runs under a nested
    scope whose sum leaves as one more carry leaf, which the scope adds once the
    loop returns; a loop that nothing stages folds where it is written, inside
    the scope. A root's [solve] runs under the scope, and its [residual] and
    [linear_solve], which only derivatives run, under {!discarding}. A compiled
    call and a remat run the function that also returns the sum of its
    additions, which a derivative tracks as one more result. *)

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
(** [discarding f] is [f ()] with every addition [f] makes dropped, those of the
    functions its constructs carry and of a compiled call's function included:
    the additions of code run a second time, which counted the first. *)
