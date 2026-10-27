(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Transformations of functions over nx values: totals and compilation.

    A transformation takes the structure ({!Nx.Ptree.t}) of each value whose
    arrays it enumerates; everything else the function uses is captured and is a
    constant of the transformation.

    {b Traced functions.} The function {!jit} receives is {e traced} at the
    call: its OCaml code runs once per call over stand-ins of its arguments'
    arrays, values with a dtype, a shape and a placement but no elements, and
    what it computes is recorded as a program, which the call then evaluates.
    Inside it {!Nx.item} of a stand-in, or of a value computed in it, raises
    ([Nx.item: the value is traced by Rune.jit]), and a draw without [~key]
    raises naming {!jit}.

    {b Extents.} A transformation, a map or a {!Total.collect} is around the
    code its function runs, on whatever domain that code resumes, and the fibers
    a scheduler installed in the function runs on the domain the function runs
    on. It is not around a systhread or a domain the function starts, nor a
    fiber such a scheduler resumes on another domain: there {!lanes} and
    {!lane_index} raise, and an addition to a total is dropped.

    {b Errors.} Every message starts with the entry point the caller applied, as
    ["Rune.jit"] for {!jit}. *)

(** {1:control Controlling differentiation} *)

val detach : ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
(** [detach x] is [x] with a zero derivative under every differentiation around
    the call. Outside every transformation it is [x] itself. *)

(** {1:maps Maps}

    A map's lanes are named by an axis, so that code inside nested maps reads
    the lanes of the map it names. *)

type axis
(** The type for names of maps. *)

val axis : unit -> axis
(** [axis ()] is a fresh name, distinct from every other. *)

val lanes : axis -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
(** [lanes a x], inside the map named [a] of [n] lanes, is every lane's [x]
    stacked on a new leading axis of length [n].

    Raises [Invalid_argument] with no map named [a] around the call
    (["Rune.lanes: no map named by the axis is around the call"]). *)

val lane_index : axis -> unit -> 'd Nx.int32_t
(** [lane_index a ()], inside the map named [a], is the calling lane's index in
    it, a 0-d [int32] from [0] to the map's number of lanes minus one.

    Raises [Invalid_argument] with no map named [a] around the call. *)

(** {1:totals Totals} *)

(** Write-only sums.

    A total is a sum that code anywhere inside a function adds to and that the
    caller reads when the function returns: {!Total.collect}[ t ~zero f] is
    [f ()] with [zero] plus the additions to [t] that reach it. Nothing reads a
    total before its [collect] returns, so an addition never changes a value the
    function computes, and with no [collect] of [t] open it does nothing.

    An addition reaches the innermost open [collect] of its total around the
    call, once per execution of the code that makes it. The additions a traced
    function makes reach it when the construct that traced the function returns,
    so a construct that raises adds none: a {!jit} adds its function's additions
    once per call, as the call returns. *)
module Total : sig
  type ('v, 's, 'd) t
  (** The type for totals of [('v, 's, 'd) Nx.t] values. *)

  val make : unit -> ('v, 's, 'd) t
  (** [make ()] is a fresh total, distinct from every other. *)

  val add : ('v, 's, 'd) t -> ('v, 's, 'd) Nx.t -> unit
  (** [add t v] adds [v] to the innermost open {!collect} of [t], and does
      nothing if none is open.

      Raises [Invalid_argument] if [v]'s shape is not the [zero]'s of the
      [collect] it reaches. *)

  val collect :
    ('v, 's, 'd) t ->
    zero:('v, 's, 'd) Nx.t ->
    (unit -> 'r) ->
    'r * ('v, 's, 'd) Nx.t
  (** [collect t ~zero f] is [(f (), total)], [total] being [zero] plus each
      addition to [t] that reaches it, in the order they reach it. If [f]
      raises, [collect] raises the same exception. *)
end

(** {1:jit Compilation} *)

val jit : ('a -> 'b, 'd) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
(** [jit s f] is [f] traced into a program, [s] its signature, one structure per
    argument and one for the result. Each call traces [f] over stand-ins of its
    arguments' arrays and evaluates the program. Every operation [f] applies is
    staged, those over values [f] captures included.

    Raises [Invalid_argument] when applied to [s] if [s] has no argument, and at
    a call if [f] captures a value that another transformation traces. *)

(** {1:one Functions of one array} *)

val jit' :
  (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('w, 'r, 'd) Nx.t
(** [jit' f] is {!jit} of a function of one array. *)
