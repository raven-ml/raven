(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Morphology: the implementation of {!Mynd.Filter.section-morphology}. *)

val erode : Nx.host Nx.bool_t -> ('v, 'r, 'd) Nx.t -> ('v, 'r, 'd) Nx.t
val dilate : Nx.host Nx.bool_t -> ('v, 'r, 'd) Nx.t -> ('v, 'r, 'd) Nx.t
