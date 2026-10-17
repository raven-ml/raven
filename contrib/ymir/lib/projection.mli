(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Projections between the native sphere and the plane.

    Every map works on native unit vectors
    [u = (cos θ cos φ, cos θ sin φ, sin θ)], as component triples, and on
    plane points [(x, y)] in radians, so no projection takes a longitude at a
    pole. A zenithal projection puts the native pole at the origin, with
    [x = R sin φ] and [y = -R cos φ] at distance [R] from it. A cylindrical
    one puts [(φ, θ) = (0, 0)] at the origin, [x] along [φ].

    A projection's parameters [pv] are float64 [[...; count code]], angles in
    degrees, as {!Ymir.Transform.code} lists them. Each map returns where its
    point is in the projection's domain, the connected region about the
    reference point where the map is a bijection. Outside it, the map returns
    finite values that no caller keeps. The formulas follow Calabretta and
    Greisen (2002), "Representations of celestial coordinates in FITS", A&A
    395, 1077. *)

(** {!Ymir.Transform.code}. *)
type code =
  | Azp
  | Szp
  | Tan
  | Stg
  | Sin
  | Arc
  | Zpn
  | Zea
  | Air
  | Cyp
  | Cea
  | Car
  | Mer

val name : code -> string
(** [name c] is [c]'s FITS code, as ["TAN"]. *)

val of_name : string -> code option
(** [of_name s] is the code whose {!name} is [s], if any. *)

val count : code -> int
(** [count c] is the number of [c]'s parameters. *)

val first : code -> int
(** [first c] is the index [m] of the FITS keyword [PVi_m] that holds [c]'s
    first parameter: 0 for ZPN, 1 for every other code. *)

val defaults : code -> float array
(** [defaults c] is a fresh array of the [count c] values FITS gives [c]'s
    parameters when a file leaves them out, angles in degrees. *)

val theta0 : code -> float
(** [theta0 c] is the native latitude θ₀, in degrees, that FITS assumes for
    [c]'s reference point: 0 for a cylindrical projection, 90 for a zenithal
    one. *)

val deproject :
  code ->
  (float, Nx.float64_elt) Nx.t ->
  (float, Nx.float64_elt) Nx.t ->
  (float, Nx.float64_elt) Nx.t ->
  ((float, Nx.float64_elt) Nx.t
  * (float, Nx.float64_elt) Nx.t
  * (float, Nx.float64_elt) Nx.t)
  * Nx.bool_t
(** [deproject code pv x y] is the native unit vector of the plane point
    [(x, y)], and where the point is in [code]'s domain. The mask is a scalar
    [true] for a projection whose deprojection covers the plane. ZPN and AIR
    solve for the angle from the pole with {!Solve.bracket}, and their mask
    is [false] where it does not converge. *)

val project :
  code ->
  (float, Nx.float64_elt) Nx.t ->
  (float, Nx.float64_elt) Nx.t
  * (float, Nx.float64_elt) Nx.t
  * (float, Nx.float64_elt) Nx.t ->
  ((float, Nx.float64_elt) Nx.t * (float, Nx.float64_elt) Nx.t) * Nx.bool_t
(** [project code pv u] is the plane point of the native unit vector [u], and
    where [u] is in [code]'s domain. The mask is a scalar [true] for a
    projection defined on the whole sphere. *)

val valid : code -> (float, 'a) Nx.t -> Nx.bool_t * string
(** [valid code pv] is where [pv] defines the projection [code], and the
    condition as text for a message, as ["P1 > 0"]. For a code that every
    [pv] defines, it is a scalar [true] and [""]. *)
