(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Bij}, with the support and bounds of a
    bijector's values that {!Dist} reads. *)

type ('f, 'd) t

(** {1:constructors Constructors} *)

val identity : ('f, 'd) t

val exp : ('f, 'd) t

val greater : low:(float, 'f, 'd) Nx.t -> ('f, 'd) t

val interval : low:(float, 'f, 'd) Nx.t -> high:(float, 'f, 'd) Nx.t -> ('f, 'd) t

val affine : loc:(float, 'f, 'd) Nx.t -> scale:(float, 'f, 'd) Nx.t -> ('f, 'd) t

val affine_tril : loc:(float, 'f, 'd) Nx.t -> scale_tril:(float, 'f, 'd) Nx.t -> ('f, 'd) t

val simplex : ('f, 'd) t

val ordered : ('f, 'd) t

val cholesky_corr : ('f, 'd) t

val sum_to_zero : ('f, 'd) t

val compose : ('f, 'd) t -> ('f, 'd) t -> ('f, 'd) t

(** {1:maps Maps} *)

val forward : ('f, 'd) t -> (float, 'f, 'd) Nx.t -> (float, 'f, 'd) Nx.t * (float, 'f, 'd) Nx.t

val inverse : ('f, 'd) t -> (float, 'f, 'd) Nx.t -> (float, 'f, 'd) Nx.t

val shape : ('f, 'd) t -> int array -> int array

val pp : Format.formatter -> ('f, 'd) t -> unit

(** {1:images Images} *)

val image : ('f, 'd) t -> Support.t -> int array -> Support.t
(** [image b s shape] is a support holding [b]'s values, of shape [shape], at
    coordinates in [s]. It reads tensor parameters' bounds on the host. *)

val bounds :
  ('f, 'd) t ->
  (float, 'f, 'd) Nx.t * (float, 'f, 'd) Nx.t ->
  (float, 'f, 'd) Nx.t * (float, 'f, 'd) Nx.t
(** [bounds b (l, h)] is a lower and an upper bound, elementwise, on the
    values that [b] maps coordinates between [l] and [h] to. It reads nothing
    on the host. *)
