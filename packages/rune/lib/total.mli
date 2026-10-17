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
    call runs the function that also takes the total so far and returns it with
    its additions, and a remat the function that also returns the sum of its
    additions; a derivative tracks either as one more result. *)

val collect :
  ('a, 'b) Construct.total ->
  zero:('a, 'b) Nx.t ->
  (unit -> 'r) ->
  'r * ('a, 'b) Nx.t
(** [collect t ~zero f] is [(f (), total)], where [total] is [zero] plus every
    addition [f] made to [t]. A compiled function called inside the scope adds
    to the total within its program, so the scope computes nothing outside it
    and collects on a device without eager kernels. An addition made outside a
    compiled function adds eagerly, where the total and the value lie.

    Raises [Invalid_argument] at an addition whose shape is not [zero]'s. *)

val discarding : (unit -> 'r) -> 'r
(** [discarding f] is [f ()] with every addition [f] makes dropped, those of the
    functions its constructs carry and of a compiled call's function included:
    the additions of code run a second time, which counted the first. *)
