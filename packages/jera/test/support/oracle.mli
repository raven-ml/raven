(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What jera's suites compare against: witnesses of tensors and structures,
    finite differences, and values computed in OCaml floats. *)

(** {1:witnesses Witnesses} *)

val tensor : ?rel:float -> ?abs:float -> unit -> ('a, 'b) Nx.t Windtrap.testable
(** [tensor ?rel ?abs ()] compares dtype, shape and elements. Without a
    tolerance, floats compare bit for bit; with one, within [rel] of the larger
    magnitude or within [abs], every NaN equal to every NaN. *)

val structure :
  ?rel:float -> ?abs:float -> 's Nx.Ptree.t -> 's Windtrap.testable
(** [structure ?rel ?abs s] compares two values of [s]: equal visits, then each
    tensor as {!tensor} does. *)

(** {1:host Host values} *)

val floats : ('a, 'b) Nx.t -> float array
(** [floats t] is [t]'s elements as float64, read on the host. *)

val dot : (float, 'b) Nx.t -> (float, 'b) Nx.t -> float
(** [dot u v] is [Σ u v] over two real tensors of one shape, in OCaml floats. *)

(** {1:differences Finite differences} *)

val central :
  eps:float ->
  ((float, 'b) Nx.t -> (float, 'c) Nx.t) ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'c) Nx.t
(** [central ~eps f x v] is [(f (x + eps·v) − f (x − eps·v)) / 2·eps]. *)

val slope : float -> float -> float
(** [slope e1 e2] is [log2 (e1 /. e2)], the order an error shows when its step
    halves from [e1]'s to [e2]'s. *)

(** {1:errors Errors} *)

val message : (unit -> 'a) -> string
(** [message f] is the text of the [Invalid_argument] that [f ()] raises.

    Raises [Failure] if [f ()] returns or raises another exception. *)
