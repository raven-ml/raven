(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Draws of {!Nx.Rng} computed another way against the host's eager draws,
    which nx's suites and rune's share.

    nx.mli's Random numbers section states every draw the same bits on every
    kernel library and at every placement, but {!Nx.Rng.dirichlet}'s, whose
    row sum is a float {!Nx.sum}, and bounds two dirichlet draws ({!bound}).
    A {!runner} says how a draw is computed: eagerly on a set ({!placed}), or
    compiled, as rune's suites do. Each case draws from a drawn key, a block
    of 2048 elements whose parameters open with their bounds and then draw
    densely where a rejection sampler rejects, and those parameters' first
    element or none. *)

(** {1:samplers Samplers} *)

type sampler
(** The type for a sampler of {!Nx.Rng}. *)

val name : sampler -> string
(** [name s] is [s]'s name, as nx's function. *)

val samplers : sampler list
(** [samplers] is every sampler of {!Nx.Rng}, [bits], [uniform], [normal],
    [exponential], [gumbel], [randint], [bernoulli], [gamma], [beta],
    [dirichlet], [von_mises], [truncated_normal], [poisson], [binomial],
    [categorical], [permutation] and [permute], and the keys of
    [split_batch] and [fold_in_tensor]. *)

type case
(** The type for a case: a key's seed and a sampler's parameters. *)

val gen : sampler -> case Windtrap.Gen.t
(** [gen s] generates cases of [s]'s parameters. *)

(** {1:runners Runners} *)

type ('w, 'r) f0 = { f0 : 'd. 'd Nx.Rng.key -> ('w, 'r, 'd) Nx.t }
(** The type for draws from a key alone. *)

type ('v, 's, 'w, 'r) f1 = {
  f1 : 'd. 'd Nx.Rng.key -> ('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t;
}
(** The type for draws from a key and one argument. *)

type ('v, 's, 'u, 't, 'w, 'r) f2 = {
  f2 :
    'd.
    'd Nx.Rng.key -> ('v, 's, 'd) Nx.t -> ('u, 't, 'd) Nx.t -> ('w, 'r, 'd) Nx.t;
}
(** The type for draws from a key and two arguments. *)

type runner = {
  run0 : 'w 'r. ('w, 'r) f0 -> Nx.host Nx.Rng.key -> ('w, 'r, Nx.host) Nx.t;
  run1 :
    'v 's 'w 'r.
    ('v, 's, 'w, 'r) f1 ->
    Nx.host Nx.Rng.key ->
    ('v, 's, Nx.host) Nx.t ->
    ('w, 'r, Nx.host) Nx.t;
  run2 :
    'v 's 'u 't 'w 'r.
    ('v, 's, 'u, 't, 'w, 'r) f2 ->
    Nx.host Nx.Rng.key ->
    ('v, 's, Nx.host) Nx.t ->
    ('u, 't, Nx.host) Nx.t ->
    ('w, 'r, Nx.host) Nx.t;
}
(** The type for ways to compute a draw from a host key and host arguments,
    its result read back on the host. *)

val placed : (module Nx.Devices) -> runner
(** [placed set] draws eagerly on [set], its key and arguments placed whole
    on each of [set]'s devices. *)

(** {1:laws Laws} *)

type dtype =
  | Float16
  | Bfloat16
  | Float32
  | Float64
      (** The type for a float sampler's dtypes. An integer sampler draws its
          32-bit dtype with [Float32] and its 64-bit one with [Float64]. *)

type shape =
  | Zero  (** No element. *)
  | One  (** One element, or a row sampler's one row of one. *)
  | Block  (** The block, or a row sampler's rows of 8. *)
  | Row  (** A row sampler's block as one row; a flat sampler skips it. *)

type compare =
  | Bits  (** A dirichlet draw has the host's bits. *)
  | Bound  (** A dirichlet draw is within {!bound} of the host's. *)

val bound : (float, 's) Nx.dtype -> n:int -> float -> float -> float
(** [bound dt ~n x y] is the most two {!Nx.Rng.dirichlet} draws [x] and [y]
    at [dt], of rows of [n], may differ by as nx.mli states it: within
    [(2γ(n - 1) + 2u) m] at the dtype they compute at, [m] the larger, and
    that dtype's least subnormal more where [m] is subnormal, then for a
    dtype narrower than [float32] the rounding of each to [dt]. *)

val law :
  runner ->
  rows:compare ->
  dtypes:dtype list ->
  shapes:shape list ->
  sampler ->
  case ->
  unit
(** [law r ~rows ~dtypes ~shapes s c] asserts that each draw of [c] at
    [dtypes] and [shapes], computed by [r], has the bits of the draw computed
    eagerly on the host from the same key and arguments, any two NaNs equal,
    but a dirichlet draw as [rows] says.

    It covers each dtype and shape it draws, each parameter's bounds and, for
    a rejection sampler, enough parameters where a round rejects that some
    element of the block takes a second round with probability above
    [1 - 10{^-9}]. *)
