(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Angular power spectra and survey science.

    The central type is {!tracer}: one tracer per tomographic bin. {!angular_cl}
    cross-correlates a list of tracers and returns a structured {!cls} value
    with typed accessors.

    {!angular_cl} and {!inverse_growth_bias} are differentiable through Rune.
    {!Cls.noise} and {!Cls.gaussian_covariance} are not (they use in-place
    mutation); compute them once at a fiducial cosmology.

    {[
      let nz1 = Survey.smail ~a:2.0 ~b:1.5 ~z0:0.3 () in
      let nz2 = Survey.smail ~a:2.0 ~b:1.5 ~z0:0.7 () in
      let wl1 = Survey.weak_lensing ~n_gal:26.0 nz1 in
      let wl2 = Survey.weak_lensing ~n_gal:26.0 nz2 in
      let ell = Nx.logspace Nx.float64 1.0 3.0 50 in
      let cls = Survey.angular_cl ~p:Cosmo.planck18 ~ell [ wl1; wl2 ] in
      let cl_auto = Survey.Cls.get cls ~i:0 ~j:0 in
      let cl_cross = Survey.Cls.get cls ~i:0 ~j:1
    ]} *)

(** {1:nz Redshift distributions} *)

type nz
(** A normalized redshift probability density n(z) with a maximum redshift. *)

val smail : ?zmax:float -> a:float -> b:float -> z0:float -> unit -> nz
(** [smail ~a ~b ~z0 ()] is n(z) {e ∝} z{^ a} exp(-(z/z0){^ b}). Auto-normalized
    via Simpson's rule. [zmax] defaults to [10.0]. *)

val tabulated : z:Nx.float64_t -> pz:Nx.float64_t -> unit -> nz
(** [tabulated ~z ~pz ()] is n(z) linearly interpolated from sampled points.
    Auto-normalized. [zmax] is inferred from the last element of [z]. *)

val custom_nz : ?zmax:float -> (Nx.float64_t -> Nx.float64_t) -> nz
(** [custom_nz f] is a redshift distribution with evaluation function [f]. [f z]
    maps a scalar tensor [z] to n(z). For differentiable survey optimization,
    [f] should use tensor operations so gradients flow through Rune. [zmax]
    defaults to [10.0]. *)

val eval_nz : nz -> Nx.float64_t -> Nx.float64_t
(** [eval_nz nz z] evaluates the normalized n(z) at [z]. *)

val nz_zmax : nz -> float
(** [nz_zmax nz] is the maximum redshift of the distribution. *)

(** {1:bias Galaxy bias} *)

type bias = Cosmo.params -> Nx.float64_t -> Nx.float64_t
(** A galaxy bias function. [bias p z] is b(z) under cosmology [p]. *)

val constant_bias : float -> bias
(** [constant_bias b] is a redshift-independent linear bias. Not differentiable
    (constant value). *)

val inverse_growth_bias : float -> bias
(** [inverse_growth_bias b0] is [b0 / D(z)], where D is the linear growth
    factor. Differentiable through Rune. *)

(** {1:power Power spectrum backends} *)

type power = Cosmo.params -> Nx.float64_t -> Nx.float64_t -> Nx.float64_t
(** [power p k z] is the matter power spectrum P(k, z) in (Mpc/h){^ 3}. [k] is a
    1-D tensor of wavenumbers in h/Mpc, [z] is a scalar tensor. *)

val linear : power
(** [linear] is the linear matter power spectrum via Eisenstein & Hu (1998).
    Differentiable through Rune. *)

val nonlinear : power
(** [nonlinear] is the nonlinear power spectrum via Halofit (Takahashi et al.
    2012). Differentiable through Rune (except the nonlinear scale k{_ nl} which
    is found by float-level root-finding). *)

val baryonic_feedback :
  ?a_bary:float -> ?log10_k_star:float -> ?sigma:float -> power -> power
(** [baryonic_feedback base_power] wraps [base_power] with a Gaussian
    suppression in log{_ 10}(k) that models baryonic feedback on the matter
    power spectrum:

    P{_ bary}(k, z) = P(k, z) {e ×} (1 - a{_ bary} {e ×} exp(-(log{_ 10}(k) -
    log{_ 10}(k{_ star})){^ 2} / {e σ}{^ 2})).

    [a_bary] is the suppression amplitude (default [0.0] = no effect).
    [log10_k_star] is the log{_ 10} of the peak suppression wavenumber in h/Mpc
    (default [1.0], i.e. k{_ star} = 10 h/Mpc). [sigma] is the Gaussian width in
    log{_ 10}(k) (default [0.55]).

    Differentiable through Rune. *)

(** {1:tracers Tracers} *)

type tracer
(** The type for a single tomographic tracer. One tracer = one redshift bin with
    its physics (lensing kernel, galaxy bias, etc.) and noise properties.
    {!angular_cl} cross-correlates a list of tracers. *)

val weak_lensing :
  ?ia_bias:bias ->
  ?sigma_e:float ->
  ?m_bias:float ->
  ?n_gal:float ->
  nz ->
  tracer
(** [weak_lensing nz] is a weak gravitational lensing tracer with redshift
    distribution [nz]. [sigma_e] is the intrinsic ellipticity dispersion
    (default [0.26]). [n_gal] is the galaxy number density in
    galaxies/arcmin{^ 2} (default [1.0]). [ia_bias], if provided, adds NLA
    intrinsic alignment.

    [m_bias] is the shear multiplicative bias (default [0.0]). The lensing
    kernel is scaled by [(1 + m_bias)], so auto-spectra scale as [(1 + m){^ 2}]
    and cross-spectra as [(1 + m{_ i})(1 + m{_ j})]. Differentiable through Rune
    when used with {!angular_cl}. *)

val number_counts : bias:bias -> ?n_gal:float -> nz -> tracer
(** [number_counts ~bias nz] is a galaxy number counts tracer with redshift
    distribution [nz] and galaxy bias model [bias]. [n_gal] is the galaxy number
    density in galaxies/arcmin{^ 2} (default [1.0]). *)

val tracer :
  ?noise:float ->
  ?zmax:float ->
  (p:Cosmo.params -> z:Nx.float64_t -> chi:Nx.float64_t -> Nx.float64_t) ->
  tracer
(** [tracer kernel] is a custom tracer with kernel function [kernel].

    [kernel ~p ~z ~chi] returns the full projection kernel W(z) at scalar
    redshift [z] and comoving distance [chi] (Mpc/h) under cosmology [p].

    [noise] is the constant noise power N{_ ℓ} for auto-correlations (default
    [0.0]). [zmax] defaults to [3.0]. *)

(** {1:cls Angular power spectra} *)

type cls
(** The type for a set of angular power spectra. Stores all auto- and
    cross-correlations for a list of tracers, along with the ell values and
    tracer metadata needed for noise and covariance computation. *)

val angular_cl :
  ?p:Cosmo.params -> ?power:power -> ell:Nx.float64_t -> tracer list -> cls
(** [angular_cl ~ell tracers] computes angular power spectra C{_ ℓ} for all
    auto- and cross-correlations via the Limber approximation. Differentiable
    through Rune.

    [power] defaults to {!nonlinear}. [p] defaults to {!Cosmo.planck18}.

    Raises [Invalid_argument] if [omega_b], [n_s], or [sigma8] are not set in
    [p]. *)

(** {2:cls_access Structured access} *)

module Cls : sig
  val get : cls -> i:int -> j:int -> Nx.float64_t
  (** [get cls ~i ~j] is the angular power spectrum C{_ ℓ}{^ ij} between tracers
      [i] and [j]. Returns a 1-D tensor of shape [[n_ell]]. [get cls ~i ~j] and
      [get cls ~j ~i] return the same spectrum.

      Raises [Invalid_argument] if [i] or [j] is out of range. *)

  val ell : cls -> Nx.float64_t
  (** [ell cls] is the multipole values, shape [[n_ell]]. *)

  val n_tracers : cls -> int
  (** [n_tracers cls] is the number of tracers. *)

  val to_tensor : cls -> Nx.float64_t
  (** [to_tensor cls] is all spectra packed as a tensor of shape
      [[n_cls; n_ell]] where [n_cls = n * (n + 1) / 2], ordered as (0,0), (0,1),
      ..., (1,1), .... *)

  val noise : cls -> Nx.float64_t
  (** [noise cls] is the shot noise power spectra. Weak lensing:
      {e σ}{_ e}{^ 2}/n{_ gal}. Number counts: 1/n{_ gal}. Custom: the [noise]
      value. Cross-spectra are zero. Shape [[n_cls; n_ell]].

      Not differentiable. *)

  val gaussian_covariance : ?f_sky:float -> cls -> Nx.float64_t
  (** [gaussian_covariance cls] is the Gaussian covariance matrix. [f_sky]
      defaults to [0.25]. Returns dense matrix of shape [[n; n]] where
      [n = n_cls * n_ell].

      Not differentiable. *)
end
