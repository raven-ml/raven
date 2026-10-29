(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The distortion polynomials, SIP's and TPV's, with their Jacobians and
    inverses, for {!Transform.sip} and {!Transform.tpv}.

    SIP (Shupe et al. 2005) adds [f (u, v) = Σ A_pq uᵖ v^q] and
    [g (u, v) = Σ B_pq uᵖ v^q] to pixel offsets [(u, v)] from CRPIX. TPV
    (the FITS registry's TPV convention) replaces intermediate coordinates
    [(ξ, η)], in degrees, by [ξ' = Σ PV1_k t_k (ξ, η)] and
    [η' = Σ PV2_k t_k (η, ξ)] over forty terms [t_k] of degree up to 7, four
    of them odd powers of [r = √(ξ² + η²)].

    Points are [[...; 2]], and Jacobians [[...; 2; 2]]. Both maps are
    bijections where their Jacobian keeps the sign it has at the origin. An
    inverse solves each point's system of two unknowns with {!Solve.lanes},
    so its derivative is the implicit one, and returns with it where the
    solve converged. *)

val det2 : ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [det2 m] is the determinant of [m]'s [2 × 2] matrices on its last two
    axes. *)

(** {1:sip SIP} *)

val sip_forward :
  ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [sip_forward a b x] is [x] plus SIP's polynomials with coefficients [a]
    and [b], [[...; n; n]], at [x]. *)

val sip_jacobian :
  (float, 'a, 'd) Nx.t -> (float, 'a, 'd) Nx.t -> (float, 'a, 'd) Nx.t -> (float, 'a, 'd) Nx.t
(** [sip_jacobian a b x] is the Jacobian of [sip_forward a b] at [x]. *)

val sip_inverse :
  ?seed:(float, 'a, 'd) Nx.t * (float, 'a, 'd) Nx.t ->
  (float, 'a, 'd) Nx.t ->
  (float, 'a, 'd) Nx.t ->
  (float, 'a, 'd) Nx.t ->
  (float, 'a, 'd) Nx.t * 'd Nx.bool_t
(** [sip_inverse ~seed a b x] is the point that [sip_forward a b] maps to
    [x], solved from [sip_forward ap bp x] for [seed] [(ap, bp)], or from
    [x]. {!Solve.lanes}'s [scale] is 10{^ 4} pixels, the offsets from CRPIX
    in the largest mosaics. *)

(** {1:tpv TPV} *)

val tpv_count : int
(** [tpv_count] is 40, the number of TPV's terms on each axis. *)

val tpv_forward : (float, 'a, 'd) Nx.t -> (float, 'a, 'd) Nx.t -> (float, 'a, 'd) Nx.t
(** [tpv_forward pv x] is TPV's map of [x] with the terms [pv],
    [[...; 2; tpv_count]], the first row for [ξ'] and the second for
    [η']. *)

val tpv_jacobian : (float, 'a, 'd) Nx.t -> (float, 'a, 'd) Nx.t -> (float, 'a, 'd) Nx.t
(** [tpv_jacobian pv x] is the Jacobian of [tpv_forward pv] at [x]. Where
    [r = 0] the derivatives of [r]'s odd powers are taken as 0. *)

val tpv_inverse :
  (float, 'a, 'd) Nx.t -> (float, 'a, 'd) Nx.t -> (float, 'a, 'd) Nx.t * 'd Nx.bool_t
(** [tpv_inverse pv x] is the point that [tpv_forward pv] maps to [x],
    solved from [x]. {!Solve.lanes}'s [scale] is 10 degrees, a wide
    field. *)
