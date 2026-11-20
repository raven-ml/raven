(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels and sampling: the implementation of {!Mynd.Sample} but its resizing,
    which [Resize] implements over the kernels here. *)

type kernel = [ `Box | `Triangle | `Cubic of float | `Lanczos of int ]
type edge = [ Image.extension | `Renormalise ]

val half_width : kernel -> float
(** [half_width k] is [k]'s half-width [h]. *)

val check_kernel : string -> kernel -> unit
(** [check_kernel fn k] raises [Invalid_argument] where [k] is invalid
    ({!Mynd.Sample.type-kernel}). *)

val at :
  edge:[< edge ] ->
  kernel ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t

val nearest :
  edge:[< Image.extension ] ->
  ('v, 'r, 'd) Nx.t ->
  (float, 's, 'd) Nx.t ->
  ('v, 'r, 'd) Nx.t

val centres : (float, 's) Nx.dtype -> int * int -> (float, 's, Nx.host) Nx.t
