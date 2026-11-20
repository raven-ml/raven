(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gradients: the implementation of {!Mynd.Filter.section-gradients}. *)

type stencil = [ `Central | `Sobel | `Scharr ]

val gradient :
  edge:[< Nx.extension ] ->
  [< stencil ] ->
  (float, 's, 'd) Nx.t ->
  (float, 's, 'd) Nx.t * (float, 's, 'd) Nx.t

val gradient_maxima :
  (float, 's, 'd) Nx.t * (float, 's, 'd) Nx.t -> 'd Nx.bool_t
