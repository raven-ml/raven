(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Bij}, with the support and bounds of a
    bijector's values that {!Dist} reads. *)

type 'f t

(** {1:constructors Constructors} *)

val identity : 'f t

val exp : 'f t

val greater : low:(float, 'f) Nx.t -> 'f t

val interval : low:(float, 'f) Nx.t -> high:(float, 'f) Nx.t -> 'f t

val affine : loc:(float, 'f) Nx.t -> scale:(float, 'f) Nx.t -> 'f t

val affine_tril : loc:(float, 'f) Nx.t -> scale_tril:(float, 'f) Nx.t -> 'f t

val simplex : 'f t

val ordered : 'f t

val cholesky_corr : 'f t

val sum_to_zero : 'f t

val compose : 'f t -> 'f t -> 'f t

(** {1:maps Maps} *)

val forward : 'f t -> (float, 'f) Nx.t -> (float, 'f) Nx.t * (float, 'f) Nx.t

val inverse : 'f t -> (float, 'f) Nx.t -> (float, 'f) Nx.t

val shape : 'f t -> int array -> int array

val pp : Format.formatter -> 'f t -> unit

(** {1:images Images} *)

val image : 'f t -> Support.t -> int array -> Support.t
(** [image b s shape] is a support holding [b]'s values, of shape [shape], at
    coordinates in [s]. It reads tensor parameters' bounds on the host. *)

val bounds :
  'f t ->
  (float, 'f) Nx.t * (float, 'f) Nx.t ->
  (float, 'f) Nx.t * (float, 'f) Nx.t
(** [bounds b (l, h)] is a lower and an upper bound, elementwise, on the
    values that [b] maps coordinates between [l] and [h] to. It reads nothing
    on the host. *)
