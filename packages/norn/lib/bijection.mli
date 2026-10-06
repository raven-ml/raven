(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Bijectors, with what the library reads of them beside {!Bij}: their image of
    a support. {!Bij} documents every value. *)

type 'f t = {
  name : string;
  image : Support.t -> int array -> Support.t;
      (** [image s shape] is the image of coordinates in [s], for values of
          [shape]. Bounds held in tensors are read on the host. *)
  bounds :
    (float, 'f) Nx.t * (float, 'f) Nx.t -> (float, 'f) Nx.t * (float, 'f) Nx.t;
      (** [bounds (lo, hi)] is the least and greatest values, elementwise, of
          coordinates between [lo] and [hi]. *)
  forward : (float, 'f) Nx.t -> (float, 'f) Nx.t * (float, 'f) Nx.t;
  inverse : (float, 'f) Nx.t -> (float, 'f) Nx.t;
  shape : int array -> int array;
}

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
val forward : 'f t -> (float, 'f) Nx.t -> (float, 'f) Nx.t * (float, 'f) Nx.t
val inverse : 'f t -> (float, 'f) Nx.t -> (float, 'f) Nx.t
val shape : 'f t -> int array -> int array
val image : 'f t -> Support.t -> int array -> Support.t

val bounds :
  'f t ->
  (float, 'f) Nx.t * (float, 'f) Nx.t ->
  (float, 'f) Nx.t * (float, 'f) Nx.t

val pp : Format.formatter -> 'f t -> unit
