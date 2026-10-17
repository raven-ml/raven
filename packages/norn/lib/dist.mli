(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Dist}. *)

type ('x, 'f) t

type ('x, 'f) kind =
  | Continuous : ((float, 'f) Nx.t, 'f) kind
  | Counts : (Nx.int32_t, 'f) kind
  | Categories : (Nx.int64_t, 'f) kind
  | Booleans : (Nx.bool_t, 'f) kind

(** {1:continuous Continuous families} *)

val normal :
  loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t

val half_normal : scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t

val lognormal :
  loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t

val student_t :
  df:(float, 'f) Nx.t ->
  loc:(float, 'f) Nx.t ->
  scale:(float, 'f) Nx.t ->
  ((float, 'f) Nx.t, 'f) t

val cauchy :
  loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t

val half_cauchy : scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t

val laplace :
  loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t

val logistic :
  loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t

val exponential : rate:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t

val gamma :
  concentration:(float, 'f) Nx.t ->
  rate:(float, 'f) Nx.t ->
  ((float, 'f) Nx.t, 'f) t

val inverse_gamma :
  concentration:(float, 'f) Nx.t ->
  scale:(float, 'f) Nx.t ->
  ((float, 'f) Nx.t, 'f) t

val beta :
  a:(float, 'f) Nx.t -> b:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t

val uniform :
  low:(float, 'f) Nx.t -> high:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t

val dirichlet : concentration:(float, 'f) Nx.t -> ((float, 'f) Nx.t, 'f) t

val mvn :
  loc:(float, 'f) Nx.t ->
  scale_tril:(float, 'f) Nx.t ->
  ((float, 'f) Nx.t, 'f) t

(** {1:discrete Discrete families} *)

val bernoulli : logits:(float, 'f) Nx.t -> (Nx.bool_t, 'f) t

val poisson : rate:(float, 'f) Nx.t -> (Nx.int32_t, 'f) t

val neg_binomial :
  mean:(float, 'f) Nx.t -> dispersion:(float, 'f) Nx.t -> (Nx.int32_t, 'f) t

val categorical : logits:(float, 'f) Nx.t -> (Nx.int64_t, 'f) t

(** {1:combinators Combinators} *)

val iid : int array -> ('x, 'f) t -> ('x, 'f) t

val sorted : int -> ((float, 'f) Nx.t, 'f) t -> ((float, 'f) Nx.t, 'f) t

val transform :
  'f Bij.t -> ((float, 'f) Nx.t, 'f) t -> ((float, 'f) Nx.t, 'f) t

val mixture : logits:(float, 'f) Nx.t -> ('x, 'f) t -> ('x, 'f) t

(** {1:eliminators Eliminators} *)

val log_density : ('x, 'f) t -> 'x -> (float, 'f) Nx.t

val factors : ('x, 'f) t -> 'x -> (float, 'f) Nx.t

val sample : Nx.Rng.t -> ('x, 'f) t -> 'x

val quantile :
  ((float, 'f) Nx.t, 'f) t -> (float, 'f) Nx.t -> (float, 'f) Nx.t

val coords : ((float, 'f) Nx.t, 'f) t -> 'f Bij.t

val standardize : ((float, 'f) Nx.t, 'f) t -> 'f Bij.t

val mixture_membership : ('x, 'f) t -> 'x -> (float, 'f) Nx.t

(** {1:observers Observers} *)

val family : ('x, 'f) t -> string

val kind : ('x, 'f) t -> ('x, 'f) kind

val dtype : ('x, 'f) t -> (float, 'f) Nx.dtype

val shape : ('x, 'f) t -> int array

val support : ('x, 'f) t -> Support.t

val bounds : ('x, 'f) t -> (float, 'f) Nx.t * (float, 'f) Nx.t

val pp : Format.formatter -> ('x, 'f) t -> unit

(** {1:checks Checks} *)

val check : ?unless:Nx.bool_t -> string -> ('x, 'f) t -> ('x, 'f) t

val valid : ('x, 'f) t -> Nx.bool_t
