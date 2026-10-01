(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the rule suites compare against: values computed outside every
    transformation, in OCaml floats or eager nx, and the witnesses that compare
    them. *)

(** {1:elements Elements} *)

val complexes : ('a, 'b) Nx.t -> Complex.t array
(** [complexes x] is the elements of the real or complex tensor [x] as complex
    numbers, in C order, read on the host wherever [x] lies.

    Raises [Invalid_argument] for another dtype. *)

val norm : Complex.t array -> float
(** [norm a] is the largest modulus of [a]'s elements, [0.] for none. *)

(** {1:witnesses Witnesses} *)

val exact : unit -> ('a, 'b) Nx.t Windtrap.testable
(** [exact ()] compares dtype, shape and elements bit for bit: [-0.] differs
    from [0.], and every NaN equals every NaN. *)

val close :
  rel:float -> ?floor:float -> unit -> Complex.t array list Windtrap.testable
(** [close ~rel ?floor ()] compares lists of element arrays: [a] and [b] are
    equal when they have the same lengths and, for each pair of arrays,
    [norm (a - b) <= rel * max (norm a) (norm b) + floor] ([floor] defaults to
    [0.]). Positions where both are NaN, or both the same infinity, count as
    equal; a NaN or infinity on one side alone does not. *)

(** {1:structures Structures} *)

val leaves : 's Nx.Ptree.t -> 's -> Complex.t array list
(** [leaves s x] is {!complexes} of each tensor of [x], in walk order. *)

val direction : Random.State.t -> 's Nx.Ptree.t -> 's -> 's
(** [direction r s x] is a value of [x]'s structure, dtypes and shapes, with
    elements drawn from [r] in [[-2, 2]]: both components of a complex element
    are drawn, each at least [0.1] from zero, so that a conjugation shows. *)

val step : 's Nx.Ptree.t -> 's -> float -> 's -> 's
(** [step s x h v] is [x + h v], leaf by leaf. *)

val dot : 's Nx.Ptree.t -> 's -> 's -> float
(** [dot s u v] is [Re (Σ conj u · v)] over every element of every leaf. *)

val magnitude : 's Nx.Ptree.t -> 's -> 's -> float
(** [magnitude s u v] is [Σ |u| · |v|], the scale against which rounding in
    {!dot} is measured. *)

val central :
  's Nx.Ptree.t ->
  'r Nx.Ptree.t ->
  eps:float ->
  ('s -> 'r) ->
  's ->
  's ->
  Complex.t array list
(** [central s r ~eps f x v] is [(f (x + eps v) − f (x − eps v)) / 2 eps], leaf
    by leaf: the derivative of [f] at [x] along [v] to second order in [eps]. *)

(** {1:cumulative The cumulative operations' definitions} *)

val cumprod_derivative : float array -> int list -> float
(** [cumprod_derivative xs s] is the derivative of [Σ_k Π_{j ≤ k} xs_j] with
    respect to the positions [s]: [Σ_{k ≥ max s} Π_{j ≤ k, j ∉ s} xs_j] for
    distinct positions, and [0.] when one repeats, since the sum is multilinear.
    [s = []] is the sum itself. *)

val running_arg : (float -> float -> bool) -> float array -> int array
(** [running_arg better xs] is, at each position [k], the position of the
    element the running extremum up to [k] is: the first of equal elements,
    where [better a b] says [a] replaces [b]. *)
