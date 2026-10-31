(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Dist}. *)

type ('x, 'f, 'd) t

type ('x, 'f, 'd) kind =
  | Continuous : ((float, 'f, 'd) Nx.t, 'f, 'd) kind
  | Counts : ('d Nx.int32_t, 'f, 'd) kind
  | Categories : ('d Nx.int64_t, 'f, 'd) kind
  | Booleans : ('d Nx.bool_t, 'f, 'd) kind

(** {1:continuous Continuous families} *)

val normal :
  loc:(float, 'f, 'd) Nx.t -> scale:(float, 'f, 'd) Nx.t -> ((float, 'f, 'd) Nx.t, 'f, 'd) t

val half_normal : scale:(float, 'f, 'd) Nx.t -> ((float, 'f, 'd) Nx.t, 'f, 'd) t

val lognormal :
  loc:(float, 'f, 'd) Nx.t -> scale:(float, 'f, 'd) Nx.t -> ((float, 'f, 'd) Nx.t, 'f, 'd) t

val student_t :
  df:(float, 'f, 'd) Nx.t ->
  loc:(float, 'f, 'd) Nx.t ->
  scale:(float, 'f, 'd) Nx.t ->
  ((float, 'f, 'd) Nx.t, 'f, 'd) t

val cauchy :
  loc:(float, 'f, 'd) Nx.t -> scale:(float, 'f, 'd) Nx.t -> ((float, 'f, 'd) Nx.t, 'f, 'd) t

val half_cauchy : scale:(float, 'f, 'd) Nx.t -> ((float, 'f, 'd) Nx.t, 'f, 'd) t

val laplace :
  loc:(float, 'f, 'd) Nx.t -> scale:(float, 'f, 'd) Nx.t -> ((float, 'f, 'd) Nx.t, 'f, 'd) t

val logistic :
  loc:(float, 'f, 'd) Nx.t -> scale:(float, 'f, 'd) Nx.t -> ((float, 'f, 'd) Nx.t, 'f, 'd) t

val exponential : rate:(float, 'f, 'd) Nx.t -> ((float, 'f, 'd) Nx.t, 'f, 'd) t

val gamma :
  concentration:(float, 'f, 'd) Nx.t ->
  rate:(float, 'f, 'd) Nx.t ->
  ((float, 'f, 'd) Nx.t, 'f, 'd) t

val inverse_gamma :
  concentration:(float, 'f, 'd) Nx.t ->
  scale:(float, 'f, 'd) Nx.t ->
  ((float, 'f, 'd) Nx.t, 'f, 'd) t

val beta :
  a:(float, 'f, 'd) Nx.t -> b:(float, 'f, 'd) Nx.t -> ((float, 'f, 'd) Nx.t, 'f, 'd) t

val uniform :
  low:(float, 'f, 'd) Nx.t -> high:(float, 'f, 'd) Nx.t -> ((float, 'f, 'd) Nx.t, 'f, 'd) t

val dirichlet : concentration:(float, 'f, 'd) Nx.t -> ((float, 'f, 'd) Nx.t, 'f, 'd) t

val mvn :
  loc:(float, 'f, 'd) Nx.t ->
  scale_tril:(float, 'f, 'd) Nx.t ->
  ((float, 'f, 'd) Nx.t, 'f, 'd) t

(** {1:discrete Discrete families} *)

val bernoulli : logits:(float, 'f, 'd) Nx.t -> ('d Nx.bool_t, 'f, 'd) t

val poisson : rate:(float, 'f, 'd) Nx.t -> ('d Nx.int32_t, 'f, 'd) t

val neg_binomial :
  mean:(float, 'f, 'd) Nx.t -> dispersion:(float, 'f, 'd) Nx.t -> ('d Nx.int32_t, 'f, 'd) t

val binomial :
  trials:'d Nx.int32_t -> logits:(float, 'f, 'd) Nx.t -> ('d Nx.int32_t, 'f, 'd) t

val categorical : logits:(float, 'f, 'd) Nx.t -> ('d Nx.int64_t, 'f, 'd) t

(** {1:combinators Combinators} *)

val iid : int array -> ('x, 'f, 'd) t -> ('x, 'f, 'd) t

val sorted : int -> ((float, 'f, 'd) Nx.t, 'f, 'd) t -> ((float, 'f, 'd) Nx.t, 'f, 'd) t

val transform :
  ('f, 'd) Bij.t -> ((float, 'f, 'd) Nx.t, 'f, 'd) t -> ((float, 'f, 'd) Nx.t, 'f, 'd) t

val mixture : logits:(float, 'f, 'd) Nx.t -> ('x, 'f, 'd) t -> ('x, 'f, 'd) t

(** {1:eliminators Eliminators} *)

val log_density : ('x, 'f, 'd) t -> 'x -> (float, 'f, 'd) Nx.t

val factors : ('x, 'f, 'd) t -> 'x -> (float, 'f, 'd) Nx.t

val sample : 'd Nx.Rng.key -> ('x, 'f, 'd) t -> 'x

val quantile :
  ((float, 'f, 'd) Nx.t, 'f, 'd) t -> (float, 'f, 'd) Nx.t -> (float, 'f, 'd) Nx.t

val coords : ((float, 'f, 'd) Nx.t, 'f, 'd) t -> ('f, 'd) Bij.t

val standardize : ((float, 'f, 'd) Nx.t, 'f, 'd) t -> ('f, 'd) Bij.t

val mixture_membership : ('x, 'f, 'd) t -> 'x -> (float, 'f, 'd) Nx.t

(** {1:observers Observers} *)

val family : ('x, 'f, 'd) t -> string

val kind : ('x, 'f, 'd) t -> ('x, 'f, 'd) kind

val dtype : ('x, 'f, 'd) t -> (float, 'f) Nx.dtype

val shape : ('x, 'f, 'd) t -> int array
val placement : ('x, 'f, 'd) t -> 'd Nx.Placement.t

val support : ('x, 'f, 'd) t -> Support.t

val bounds : ('x, 'f, 'd) t -> (float, 'f, 'd) Nx.t * (float, 'f, 'd) Nx.t

val pp : Format.formatter -> ('x, 'f, 'd) t -> unit

(** {1:checks Checks} *)

val check : ?unless:'d Nx.bool_t -> string -> ('x, 'f, 'd) t -> ('x, 'f, 'd) t

val valid : ('x, 'f, 'd) t -> 'd Nx.bool_t
