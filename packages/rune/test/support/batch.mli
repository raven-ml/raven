(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Independent compiled calls run as one program.

    tolk compiles a program's kernels together, in one compiler run per worker,
    and a run costs tens of milliseconds however few kernels it holds. Calls
    that read only their own arguments and make their own results share no
    value, so a suite checks many of them at the cost of one program. The
    scheduler may join their elementwise work into one kernel, which one run
    compiles: a batch saves the runs' fixed cost, not the compilation of what
    the calls compute. *)

type ('r, 'd) call
(** The type for calls on placements ['d] whose results read as ['r]. *)

val call :
  ('a, 'd) Nx.Ptree.t ->
  'a ->
  ('b, 'd) Nx.Ptree.t ->
  ('a -> 'b) ->
  ('b -> 'r) ->
  ('r, 'd) call
(** [call s x s' f read] is the call [f x] read by [read], [s] describing [x]
    and [s'] [f x]. *)

val run : ('r, 'd) call list -> 'r list
(** [run calls] is each call's result read, in order, the calls compiled as one
    program: {!Rune.jit} of the function that applies each call's function to
    its own argument. It is [[]] for no call and raises what the trace or the
    program raises.

    Fails, with {!Windtrap.fail}, if the program is evaluated rather than
    compiled ({!Rune.evaluations}). Where rig links no host code, such a program
    skips the running test instead ({!Compiled.here}). *)

val results : ('r, 'd) call list -> (unit -> 'r) list
(** [results calls] is {!run}'s results as thunks, each raising what the trace
    or the program raised, if either did. It fails as {!run} does. *)

val table :
  per:int -> ('a -> ('r, 'd) call) -> 'a list -> ('a * (unit -> 'r)) list
(** [table ~per call rows] pairs each row with its result, the rows in programs
    of [per] calls. The first result forced of a program's rows makes their
    calls and {!run}s them; later ones read the results kept, or raise what that
    run raised. tolk joins a program's independent kernels into one source,
    whose compilation slows past a few dozen kernels, so [per] keeps a program
    to about that.

    Raises [Invalid_argument] if [per < 1]. *)

val law :
  per:int ->
  ('a -> ('r, 'd) call) ->
  'a list ->
  ('a -> (unit -> 'r) -> unit) ->
  'a ->
  unit
(** [law ~per call examples check] is the law [fun c -> check c r], [r ()] being
    [c]'s call's result read. For a case of [examples], the very value and not
    an equal one, [r] reads a result of [table ~per call examples]; for another,
    it runs [c]'s call alone. A property given [examples] as its examples thus
    compiles them [per] to a program. *)

(** {1:checks Checks} *)

type 'd check
(** The type for a call on placements ['d] with the check of its result. *)

val check : ?alone:bool -> ('r, 'd) call -> ((unit -> 'r) -> unit) -> 'd check
(** [check call k] checks [call]'s result read with [k r], [r ()] being the
    result or raising what the call's trace or program raised. A call that is to
    raise is [alone] (defaults to [false]): it compiles by itself, so that the
    others' programs do not raise with it. *)

val checks : 'd check list -> unit
(** [checks cs] compiles the calls of [cs] that are not alone as one program,
    each alone one as its own, and runs each check on its call's result.

    Fails and skips as {!run} does. *)

val each :
  ?alone:('a -> bool) ->
  ('a -> ('r, 'd) call) ->
  ('a -> (unit -> 'r) -> unit) ->
  'a list ->
  unit
(** [each ~alone call k cases] is {!checks} of each case's
    [check ~alone:(alone c) (call c) (k c)]. [alone] defaults to
    [fun _ -> false]. *)

val drawn : int -> 'a Windtrap.Gen.t -> 'a list Windtrap.Gen.t
(** [drawn n gen] draws [n] values of [gen], for a law that checks them in one
    program: a property of [count] cases checks [n * count] values. Shrinking
    keeps the length. *)

val chunks : int -> 'a list -> 'a list list
(** [chunks n xs] is [xs] in order, cut into lists of [n], the last of the rest.
    It gives a property's examples to a law over lists. *)
