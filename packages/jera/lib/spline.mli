(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Piecewise-cubic interpolants as Chebyshev coefficients, along the leading
    axis of samples of shape [[n] @ rest]. *)

type 'b ends =
  [ `Natural | `Not_a_knot | `Clamped of (float, 'b) Nx.t * (float, 'b) Nx.t ]
(** The type for a spline's end conditions; a clamped slope has shape [rest]. *)

val linear : (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [linear x y] is the coefficients, of shape [[n − 1; 2] @ rest], of the
    broken line through the knots [x] and samples [y]. *)

val cubic : 'b ends -> (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [cubic ends x y] is the coefficients, of shape [[n − 1; 4] @ rest], of the
    cubic spline through [x] and [y] with end conditions [ends]. *)

val steffen : (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [steffen x y] is the coefficients of Steffen's monotone interpolant. *)

val hermite :
  (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [hermite x y m] is the coefficients of the piecewise cubic with values [y]
    and slopes [m] at the knots [x]. *)
