(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Cosmology for {e Λ}CDM, wCDM, and w0waCDM universes.

    Computes distances, growth factors, and matter power spectra. Supports flat
    and non-flat {e Λ}CDM, wCDM, and w0waCDM cosmologies through a single
    parameter type. All functions are differentiable through Rune.

    {[
      let z = Nx.scalar Nx.float64 0.5 in
      let dl = Cosmo.luminosity_distance z in
      let dl_mpc = Unit.Length.in_mpc dl
    ]}

    Power spectrum functions require [omega_b], [n_s], and [sigma8] to be set
    via {!set} or by using a preset like {!planck18}. *)

(** {1:params Parameters} *)

type params
(** The type for cosmological parameters. Subsumes flat {e Λ}CDM, non-flat
    {e Λ}CDM, wCDM, and w0waCDM. *)

(** {2:float_constructors Float constructors}

    Create parameters from plain floats. *)

val flat_lcdm : h0:float -> omega_m:float -> params
(** [flat_lcdm ~h0 ~omega_m] is flat {e Λ}CDM with {e Ω}{_ L}[ = 1 - omega_m].

    Raises [Invalid_argument] if [h0 <= 0] or [omega_m < 0]. *)

val lcdm : h0:float -> omega_m:float -> omega_l:float -> params
(** [lcdm ~h0 ~omega_m ~omega_l] is {e Λ}CDM with curvature
    {e Ω}{_ k}[ = 1 - omega_m - omega_l].

    Raises [Invalid_argument] if [h0 <= 0]. *)

val wcdm :
  h0:float -> omega_m:float -> ?omega_l:float -> w0:float -> unit -> params
(** [wcdm ~h0 ~omega_m ~w0 ()] is wCDM with constant dark energy equation of
    state [w0]. [omega_l] defaults to [1 - omega_m] (flat). *)

val w0wacdm :
  h0:float ->
  omega_m:float ->
  ?omega_l:float ->
  w0:float ->
  wa:float ->
  unit ->
  params
(** [w0wacdm ~h0 ~omega_m ~w0 ~wa ()] is the CPL parameterization
    [w(z) = w0 + wa * z/(1+z)]. [omega_l] defaults to [1 - omega_m] (flat). *)

(** {2:tensor_constructors Tensor constructors}

    Create parameters from Nx scalar tensors for differentiable construction. *)

val create_flat_lcdm : h0:Nx.float64_t -> omega_m:Nx.float64_t -> params

val create_lcdm :
  h0:Nx.float64_t -> omega_m:Nx.float64_t -> omega_l:Nx.float64_t -> params

val create_wcdm :
  h0:Nx.float64_t ->
  omega_m:Nx.float64_t ->
  ?omega_l:Nx.float64_t ->
  w0:Nx.float64_t ->
  unit ->
  params

val create_w0wacdm :
  h0:Nx.float64_t ->
  omega_m:Nx.float64_t ->
  ?omega_l:Nx.float64_t ->
  w0:Nx.float64_t ->
  wa:Nx.float64_t ->
  unit ->
  params

(** {2:accessors Accessors} *)

val h0 : params -> Nx.float64_t
(** [h0 p] is the Hubble constant H{_ 0} in km s{^ -1} Mpc{^ -1}. *)

val omega_m : params -> Nx.float64_t
(** [omega_m p] is the matter density parameter {e Ω}{_ m}. *)

val omega_l : params -> Nx.float64_t
(** [omega_l p] is the dark energy density parameter {e Ω}{_ Λ}. *)

val omega_k : params -> Nx.float64_t
(** [omega_k p] is the curvature density parameter {e Ω}{_ k}[ = 1 - Ω_m - Ω_Λ].
*)

val w0 : params -> Nx.float64_t
(** [w0 p] is the dark energy equation of state parameter w{_ 0}. *)

val wa : params -> Nx.float64_t
(** [wa p] is the CPL time-varying dark energy parameter w{_ a}. *)

val omega_b : params -> Nx.float64_t
(** [omega_b p] is the baryon density parameter {e Ω}{_ b}.

    Raises [Invalid_argument] if not set. *)

val n_s : params -> Nx.float64_t
(** [n_s p] is the primordial spectral index n{_ s}.

    Raises [Invalid_argument] if not set. *)

val sigma8 : params -> Nx.float64_t
(** [sigma8 p] is the amplitude of matter fluctuations {e σ}{_ 8}.

    Raises [Invalid_argument] if not set. *)

(** {2:set Setting power spectrum parameters} *)

val set : ?omega_b:float -> ?n_s:float -> ?sigma8:float -> params -> params
(** [set ~omega_b ~n_s ~sigma8 p] is [p] with the given power spectrum
    parameters set. Unspecified parameters retain their previous value. *)

val set_t :
  ?h0:Nx.float64_t ->
  ?omega_m:Nx.float64_t ->
  ?omega_l:Nx.float64_t ->
  ?omega_b:Nx.float64_t ->
  ?n_s:Nx.float64_t ->
  ?sigma8:Nx.float64_t ->
  params ->
  params
(** [set_t] is like {!set} but takes Nx scalar tensors for differentiable
    construction. Recomputes {e Ω}{_ k} when [omega_m] or [omega_l] changes. *)

(** {2:presets Presets} *)

val default : params
(** [default] is flat {e Λ}CDM with [h0 = 70], [omega_m = 0.3]. *)

val planck18 : params
(** [planck18] is Planck 2018 flat {e Λ}CDM: [h0 = 67.66], [omega_m = 0.3111],
    [omega_b = 0.0490], [n_s = 0.9665], [sigma8 = 0.8102]. *)

val planck15 : params
(** [planck15] is Planck 2015 flat {e Λ}CDM: [h0 = 67.74], [omega_m = 0.3075],
    [omega_b = 0.0486], [n_s = 0.9667], [sigma8 = 0.8159]. *)

val wmap9 : params
(** [wmap9] is WMAP9 flat {e Λ}CDM: [h0 = 69.32], [omega_m = 0.2865],
    [omega_b = 0.0463], [n_s = 0.9608], [sigma8 = 0.820]. *)

(** {1:e_z Hubble parameter} *)

val e_of : params -> Nx.float64_t -> Nx.float64_t
(** [e_of p z] is E(z) = H(z)/H{_ 0} at redshift [z]. Fully differentiable
    through Rune. *)

val hubble : ?p:params -> Nx.float64_t -> Nx.float64_t
(** [hubble z] is H(z) in km s{^ -1} Mpc{^ -1}. [p] defaults to {!default}. *)

val critical_density : ?p:params -> Nx.float64_t -> Nx.float64_t
(** [critical_density z] is the critical density {e rho}{_ c}(z) in kg m{^ -3}.
    [p] defaults to {!default}. *)

(** {1:distances Distances} *)

val comoving_distance : ?p:params -> Nx.float64_t -> Unit.length Unit.t
(** [comoving_distance z] is the line-of-sight comoving distance at redshift
    [z]. [p] defaults to {!default}. *)

val luminosity_distance : ?p:params -> Nx.float64_t -> Unit.length Unit.t
(** [luminosity_distance z] is the luminosity distance at redshift [z]. For
    non-flat models, applies the curvature correction via the transverse
    comoving distance. [p] defaults to {!default}. *)

val angular_diameter_distance : ?p:params -> Nx.float64_t -> Unit.length Unit.t
(** [angular_diameter_distance z] is the angular diameter distance at redshift
    [z]. [p] defaults to {!default}. *)

val distance_modulus : ?p:params -> Nx.float64_t -> Nx.float64_t
(** [distance_modulus z] is the distance modulus
    {e mu}[ = 5 log10(d_L / Mpc) + 25]. [p] defaults to {!default}. *)

(** {1:angular Angular scale} *)

val angular_size :
  ?p:params -> z:Nx.float64_t -> Unit.length Unit.t -> Unit.angle Unit.t
(** [angular_size ~z length] is the angular size of [length] at redshift [z]
    under the small-angle approximation [{e theta} = l / d_A]. [p] defaults to
    {!default}. *)

val physical_size :
  ?p:params -> z:Nx.float64_t -> Unit.angle Unit.t -> Unit.length Unit.t
(** [physical_size ~z angle] is the physical size subtended by [angle] at
    redshift [z] under the small-angle approximation [l = {e theta} * d_A]. [p]
    defaults to {!default}. *)

(** {1:times Cosmic times} *)

val lookback_time : ?p:params -> Nx.float64_t -> Unit.time Unit.t
(** [lookback_time z] is the lookback time to redshift [z]. [p] defaults to
    {!default}. *)

val age : ?p:params -> Nx.float64_t -> Unit.time Unit.t
(** [age z] is the age of the universe at redshift [z].

    Integrates from [z] to [z = 1000]. This approximation is accurate to ~0.1%
    for late-time cosmology ([z < 10]) but omits the radiation era and is not
    suitable for CMB-epoch calculations. [p] defaults to {!default}. *)

(** {1:inverse Inverse lookup} *)

val z_at_value :
  ?p:params ->
  ?zmin:float ->
  ?zmax:float ->
  ?xtol:float ->
  (p:params -> Nx.float64_t -> Nx.float64_t) ->
  Nx.float64_t ->
  Nx.float64_t
(** [z_at_value f target] finds the redshift [z] where [f ~p z = target] using
    Brent's method. [f] must be a monotonic function of redshift.

    For distance functions, unwrap the unit first:
    {[
    z_at_value
      (fun ~p z -> Unit.Length.in_mpc (Cosmo.comoving_distance ~p z))
      target
    ]}

    [zmin] defaults to [1e-8]. [zmax] defaults to [1000.0]. [xtol] defaults to
    [1e-8].

    {b Warning.} Not differentiable (iterative root-finding).

    Raises [Invalid_argument] if [target] is outside [[f(zmin), f(zmax)]]. *)

(** {1:bao BAO distance measures} *)

val dh : ?p:params -> Nx.float64_t -> Unit.length Unit.t
(** [dh z] is the Hubble distance D{_ H}(z) = c / H(z). [p] defaults to
    {!default}. *)

val dm : ?p:params -> Nx.float64_t -> Unit.length Unit.t
(** [dm z] is the comoving transverse distance D{_ M}(z). Equal to
    {!comoving_distance} for flat cosmologies; includes curvature correction
    otherwise. [p] defaults to {!default}. *)

val dv : ?p:params -> Nx.float64_t -> Unit.length Unit.t
(** [dv z] is the volume-averaged BAO distance D{_ V}(z) = (z D{_ H}(z)
    D{_ M}{^ 2}(z)){^ 1/3}. [p] defaults to {!default}. *)

val sound_horizon : ?p:params -> unit -> Unit.length Unit.t
(** [sound_horizon ()] is the comoving sound horizon at the drag epoch
    r{_ s}(z{_ drag}), using the Eisenstein & Hu (1998) fitting formulae for
    z{_ drag} and the sound horizon integral.

    Raises [Invalid_argument] if [omega_b] is not set in [p]. [p] defaults to
    {!default}. *)

(** {1:growth Structure growth} *)

val growth_factor : ?p:params -> Nx.float64_t -> Nx.float64_t
(** [growth_factor z] is the linear growth factor D(z), normalized to D(0) = 1.
    Computed via the integral form D(a) {e ∝} E(a) {e ∫}{_ 0}{^ a} da' /
    (a'{^ 3} E{^ 3}(a')).

    Does not require [omega_b], [n_s], or [sigma8]. [p] defaults to {!default}.
*)

val growth_rate : ?p:params -> Nx.float64_t -> Nx.float64_t
(** [growth_rate z] is the linear growth rate f(z) = d ln D / d ln a, computed
    from the exact derivative of the integral-form growth factor.

    [p] defaults to {!default}. *)

(** {1:power Matter power spectrum}

    All power spectrum functions require [omega_b], [n_s], and [sigma8] to be
    set in the parameters. Use {!set} or a preset like {!planck18}.

    Wavenumbers [k] are in h/Mpc. Power spectra are in (Mpc/h){^ 3}. *)

val linear_power : ?p:params -> Nx.float64_t -> Nx.float64_t -> Nx.float64_t
(** [linear_power ~p k z] is the linear matter power spectrum P(k, z). Uses the
    Eisenstein & Hu (1998) transfer function with baryon oscillations and
    {e σ}{_ 8} normalization.

    Raises [Invalid_argument] if [omega_b], [n_s], or [sigma8] are not set. *)

val nonlinear_power : ?p:params -> Nx.float64_t -> Nx.float64_t -> Nx.float64_t
(** [nonlinear_power ~p k z] is the nonlinear matter power spectrum via the
    Halofit fitting formula (Takahashi et al. 2012).

    {b Warning.} The nonlinear scale k{_ nl} is found by float-level
    root-finding; gradients do not flow through it. The mapping from k{_ nl} to
    P{_ nl}(k) is differentiable.

    Raises [Invalid_argument] if [omega_b], [n_s], or [sigma8] are not set. *)
