(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The solves for the inverses of the distortion families and for the
    deprojections of ZPN and AIR. This is the only module that names jera.

    Each answer is stated as a zero of its equation, so its derivative is the
    implicit one and no iteration is differentiated. A lane that does not
    converge keeps its last estimate, and the mask each solve returns with it
    is [false] there. *)

val bracket :
  ((float, 'a, 'd) Nx.t -> (float, 'a, 'd) Nx.t) ->
  lo:(float, 'a, 'd) Nx.t ->
  hi:(float, 'a, 'd) Nx.t ->
  (float, 'a, 'd) Nx.t * 'd Nx.bool_t
(** [bracket f ~lo ~hi] is a zero of [f] between [lo] and [hi] in each
    element ({!Jera.Root.bracket}), within twice float64's epsilon relative
    and four times it absolute, and where it converged. *)

val lanes :
  scale:float ->
  jacobian:((float, 'a, 'd) Nx.t -> (float, 'a, 'd) Nx.t) ->
  ((float, 'a, 'd) Nx.t -> (float, 'a, 'd) Nx.t) ->
  (float, 'a, 'd) Nx.t ->
  (float, 'a, 'd) Nx.t * 'd Nx.bool_t
(** [lanes ~scale ~jacobian f guess] is a zero of [f] in each lane of
    [guess], [[...; k]], by Newton's method from [guess]
    ({!Jera.System.lanes}), and where it converged. Its tolerance is four
    times float64's epsilon relative, and that times [scale] absolute, for
    [scale] the size of the points' coordinates. A lane takes at most 20
    steps. *)
