(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Morphological operations, the implementation of {!Sowilo}'s morphology
    functions, whose section states their rules. *)

type kernel_shape = Rect | Cross | Ellipse

val structuring_element : kernel_shape -> int * int -> Nx.host Nx.bool_t
val erode :
  kernel:Nx.host Nx.bool_t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
val dilate :
  kernel:Nx.host Nx.bool_t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
val opening :
  kernel:Nx.host Nx.bool_t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
val closing :
  kernel:Nx.host Nx.bool_t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t

val morphological_gradient :
  kernel:Nx.host Nx.bool_t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
