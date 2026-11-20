(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Colours: the implementation of {!Mynd.Colour}. *)

val luma : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val luminance : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val to_hsv : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val of_hsv : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val linear_to_lab : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
val lab_to_linear : (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
