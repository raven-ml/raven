(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Transformations of functions over nx values: loops, totals and compilation.

    A transformation takes the structure ({!Nx.Ptree.t}) of each value whose
    arrays it enumerates; everything else the function uses is captured and is a
    constant of the transformation.

    {b Functions of a construct are traced.} The functions that {!scan},
    {!iterate} and {!jit} receive are {e traced} at the call: their OCaml code
    runs over stand-ins of their arguments' arrays, values with a dtype, a
    shape and a placement but no elements, and what it computes from them is
    recorded as a program, which the construct then evaluates. A loop's step
    is traced once, and again where it places a carry array it received of
    every set ({!scan}). So a step's code runs once or a few times whatever the
    number of steps, as do its side effects, and inside it {!Nx.item} of a
    stand-in, or of a value computed from one, raises
    ([Nx.item: the value is traced by Rune.scan]); a branch on a value is
    {!Nx.where} or a nested {!iterate}. An operation that reads no
    stand-in computes at the call, once. A draw without [~key] inside a traced
    function raises, naming the construct, so a step draws from a key in its
    carry or its rows. Outside every transformation a construct is its program
    evaluated: a {!scan} is the fold of its step, bit for bit but for a NaN
    result's sign and payload, which nx leaves unspecified.

    {b Extents.} A transformation, a map or a {!Total.collect} is around the
    code its function runs, on whatever domain that code resumes, and the fibers
    a scheduler installed in the function runs on the domain the function runs
    on. It is not around a systhread or a domain the function starts, nor a
    fiber such a scheduler resumes on another domain: there {!lanes} and
    {!lane_index} raise, and an addition to a total is dropped.

    {b Errors.} Every message starts with the entry point the caller applied, as
    ["Rune.scan"] for {!scan}. *)

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
    once per call, as the call returns. A step of {!scan} or {!iterate}, and an
    [until] once per test, makes its additions once per step: a loop chains each
    total's additions in the order they are made, from an identity that changes
    no sum (a negative zero), and that chain reaches the [collect] as one
    addition, as the loop returns. So the total is the one the steps' additions
    give made in turn where the [zero] is [0.] or [-0.] and the loop's are the
    first additions the [collect] receives. A loop that takes no step adds that
    identity. *)
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

(** {1:loops Loops} *)

val scan :
  ('c, 'd) Nx.Ptree.t ->
  ('x, 'd) Nx.Ptree.t ->
  ('y, 'd) Nx.Ptree.t ->
  f:('c -> 'x -> 'c * 'y) ->
  init:'c ->
  'x ->
  'c * 'y
(** [scan c x y ~f ~init xs] folds the step [f] over the rows of [xs], a value
    of structure [x]: every array of [xs] has the same leading length [n], and
    step [i] passes [f] row [i] of every array. [f carry row] returns the next
    carry, of structure [c], and the step's outputs, of structure [y]; the
    result is the final carry and the outputs, each array stacked along a new
    axis 0. With [n = 0] the result is [init] and outputs of leading length [0].

    [f] is traced at the call over stand-ins of [init]'s arrays and of [xs]'s
    arrays without axis 0. It is traced again while it returns placed an
    array of the carry that it received of every set, such as a constant
    [init] it adds a placed value to, each array at most once.

    Raises [Invalid_argument] if [xs] has no array
    (["Rune.scan: xs has no leaf"]), a 0-d array, or arrays of different leading
    lengths, naming their paths (["Rune.scan: xs: 1: 3 rows, 0: 2"]), and, at
    the trace, if the carry [f] returns differs from the one it received in its
    {!Nx.Ptree.val-visits}, dtypes or shapes, naming the first path where they
    differ, as in
    ["Rune.scan: 0: shape [3] in the carry the step returned, [2] in the carry
     it received"]. *)

val iterate :
  ('c, 'd) Nx.Ptree.t ->
  max:int ->
  until:('c -> 'd Nx.bool_t) ->
  f:('c -> 'c) ->
  'c ->
  'c
(** [iterate c ~max ~until ~f init] applies the step [f] to a carry of structure
    [c], from [init], until [until carry] holds, and returns that carry. [until]
    returns one boolean and is tested before each step, so an [init] that
    satisfies it is returned unchanged. [f] and [until] are traced at the
    call, [f] as {!scan}'s step is.

    Raises [Invalid_argument] if [max < 0]
    (["Rune.iterate: max = -1 is negative"]), if [until] returns other than one
    element, at the trace if [f]'s carry differs from the one it received as
    {!scan} says, and if [until] is still false after [max] steps
    (["Rune.iterate: until is still false after max = 50 steps"]). *)

(** {1:jit Compilation} *)

val jit : ('a -> 'b, 'd) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
(** [jit s f] is [f] traced into a program, [s] its signature, one structure per
    argument and one for the result. Each call traces [f] over stand-ins of its
    arguments' arrays and evaluates the program. Every operation [f] applies is
    staged, those over values [f] captures included.

    Raises [Invalid_argument] when applied to [s] if [s] has no argument, and at
    a call if [f] captures a value that another transformation traces. *)

(** {1:one Functions of one array} *)

val scan' :
  f:
    (('a, 'b, 'd) Nx.t ->
    ('c, 'e, 'd) Nx.t ->
    ('a, 'b, 'd) Nx.t * ('g, 'h, 'd) Nx.t) ->
  init:('a, 'b, 'd) Nx.t ->
  ('c, 'e, 'd) Nx.t ->
  ('a, 'b, 'd) Nx.t * ('g, 'h, 'd) Nx.t
(** [scan' ~f ~init xs] is {!scan} of one array each for the carry, the rows and
    the outputs. *)

val iterate' :
  max:int ->
  until:(('v, 's, 'd) Nx.t -> 'd Nx.bool_t) ->
  f:(('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t
(** [iterate' ~max ~until ~f x] is {!iterate} of one array. *)

val jit' :
  (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('w, 'r, 'd) Nx.t
(** [jit' f] is {!jit} of a function of one array. *)
