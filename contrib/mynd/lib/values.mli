(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values: the implementation of {!Mynd}'s values ({!Mynd.section-values}). *)

val convert : ('a, 'b) Nx.dtype -> ('v, 'r, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
val srgb_to_linear : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val linear_to_srgb : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val premultiply : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val unpremultiply : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
