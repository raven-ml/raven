(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the transformations' suites compare against: values computed outside
    every transformation, from host floats or eager nx, and witnesses of tensors
    and structures. *)

(** {1:witnesses Witnesses} *)

val tensor : ?rel:float -> ?abs:float -> unit -> ('a, 'b) Nx.t Windtrap.testable
(** [tensor ?rel ?abs ()] compares dtype, shape and elements. Without a
    tolerance, floats and complex components compare bit for bit, so [-0.]
    differs from [0.] and NaNs equal NaNs of the same payload; with one, within
    [rel] of the larger magnitude or within [abs], every NaN equal to every NaN.
*)

val structure :
  ?rel:float -> ?abs:float -> 's Nx.Ptree.t -> 's Windtrap.testable
(** [structure ?rel ?abs s] compares two values of [s]: equal visits, then each
    tensor as {!tensor} does. *)

(** {1:inner Inner products} *)

val dot : ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> float
(** [dot u v] is [Re (Σ conj u · v)] over real or complex tensors of one shape,
    computed in OCaml floats.

    Raises [Invalid_argument] for other dtypes or two shapes. *)

(** {1:differences Finite differences} *)

val central :
  eps:float ->
  ((float, 'b) Nx.t -> (float, 'c) Nx.t) ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'c) Nx.t
(** [central ~eps f x v] is [(f (x + eps·v) − f (x − eps·v)) / 2·eps], the
    directional derivative of [f] at [x] along [v] to second order in [eps]. *)

(** {1:term The composition function}

    [t x = tanh (x²) · eˣ] and its first two derivatives, in OCaml floats. *)

val term : float -> float
val term' : float -> float
val term'' : float -> float

val map : (float -> float) -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [map f x] is [f] of each element of [x], computed in OCaml floats. *)

(** {1:errors Errors} *)

val message : (unit -> 'a) -> string
(** [message f] is the text of the [Invalid_argument] that [f ()] raises.

    Raises [Failure] if [f ()] returns or raises another exception. *)
