(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Sampled spectral quantities on a wavelength grid.

    A {!'a t} pairs a wavelength grid with spectral values parameterised by a
    phantom {e kind} that tracks the physical meaning of the values:

    - {!flux_density}: spectral flux density f{_ lambda} (W m{^ -2} m{^ -1}).
    - {!radiance}: spectral radiance B{_ lambda} (W m{^ -2} m{^ -1} sr{^ -1}).
    - {!sampled}: arbitrary values with no physical assumption.

    Operations that depend on the physical interpretation of the values (e.g.,
    {!redshift}, {!val-Photometry.ab_mag}) require a specific kind, preventing
    accidental misuse at compile time. Use {!as_flux_density} to explicitly
    reinterpret values when the physical meaning is known to the caller.

    {[
      let wave = Unit.Length.of_m (Nx.linspace Nx.float64 1e-7 1e-5 1000) in
      let sed =
        Spectrum.blackbody
          ~temperature:(Unit.Temperature.of_kelvin (Nx.scalar Nx.float64 5800.0))
          ~wavelength:wave
        |> Spectrum.as_flux_density
      in
      let reddened =
        Extinction.apply (Extinction.ccm89 ~rv) ~av sed
    ]}

    {2:batch Batched spectra}

    Values may have leading batch dimensions: a spectrum with wavelength
    [[n_lambda]] and values [[batch; n_lambda]] represents [batch] spectra
    sharing a wavelength grid. All operations ({!resample}, {!scale}, {!add},
    {!val-Photometry.ab_mag}, {!val-Extinction.apply}, etc.) broadcast over
    leading dimensions via Nx:

    {[
      let values = Nx.stack (List.map Spectrum.values templates) in
      let batch =
        Spectrum.create ~wavelength ~values |> Spectrum.as_flux_density
      in
      let mags = Photometry.ab_mag bp batch  (* shape [batch] *)
    ]}

    {b Note.} {!redshift} with a per-spectrum [z] does not broadcast — it
    changes the wavelength grid, breaking the shared-grid invariant. Use
    [List.map] or [Rune.vmap] for per-spectrum redshifts. *)

(** {1:kinds Spectral kinds} *)

type flux_density
(** Phantom type for spectral flux density f{_ lambda} (W m{^ -2} m{^ -1}). *)

type radiance
(** Phantom type for spectral radiance B{_ lambda} (W m{^ -2} m{^ -1} sr{^ -1}).
*)

type sampled
(** Phantom type for arbitrary sampled spectral values. *)

(** {1:types Types} *)

type 'a t
(** The type for spectra parameterised by spectral kind ['a]. *)

(** {1:constructors Constructors} *)

val create : wavelength:Unit.length Unit.t -> values:Nx.float64_t -> sampled t
(** [create ~wavelength ~values] is a tabulated spectrum. [wavelength] must be
    1-D. [values] must be at least 1-D with its last dimension matching
    [wavelength]; leading dimensions are preserved as batch dimensions.

    Raises [Invalid_argument] if [wavelength] is not 1-D, the last dimension of
    [values] does not match, or [wavelength] is not strictly increasing. *)

(** {1:accessors Accessors} *)

val wavelength : 'a t -> Unit.length Unit.t
(** [wavelength s] is the wavelength grid. *)

val values : 'a t -> Nx.float64_t
(** [values s] is the spectral values. *)

(** {1:casts Kind casts} *)

val as_flux_density : _ t -> flux_density t
(** [as_flux_density s] reinterprets [s] as spectral flux density. The caller is
    responsible for ensuring the values represent f{_ lambda}. Use this when
    working with external data or when only relative values matter (e.g.,
    fitting colours from a blackbody model). *)

val as_sampled : _ t -> sampled t
(** [as_sampled s] forgets the spectral kind. *)

(** {1:models Parametric models} *)

val blackbody :
  temperature:Unit.temperature Unit.t ->
  wavelength:Unit.length Unit.t ->
  radiance t
(** [blackbody ~temperature ~wavelength] is the Planck spectral radiance
    B{_ lambda}(T) in W m{^ -2} m{^ -1} sr{^ -1} at the given wavelengths. This
    is a per-steradian quantity; multiply by a solid angle to obtain spectral
    irradiance. Differentiable through Rune. *)

val power_law :
  amplitude:Nx.float64_t ->
  index:Nx.float64_t ->
  pivot:Unit.length Unit.t ->
  wavelength:Unit.length Unit.t ->
  sampled t
(** [power_law ~amplitude ~index ~pivot ~wavelength] is the spectrum
    [amplitude * (wavelength / pivot){^index}]. Differentiable through Rune. *)

(** {1:operations Operations} *)

val redshift : z:Nx.float64_t -> flux_density t -> flux_density t
(** [redshift ~z s] shifts [s] to redshift [z]. Wavelengths are multiplied by
    [(1+z)] and values are divided by [(1+z)].

    Restricted to {!flux_density} spectra because the [(1+z){^ -1}] dimming
    factor is specific to spectral flux density. Differentiable through Rune. *)

val scale : Nx.float64_t -> 'a t -> 'a t
(** [scale factor s] is [s] with values multiplied element-wise by [factor].
    [factor] may be a scalar or a tensor that broadcasts with the values.
    Differentiable through Rune. *)

val mul : 'a t -> sampled t -> 'a t
(** [mul a b] multiplies values element-wise. [a]'s spectral kind is preserved;
    [b] is treated as a dimensionless modifier (transmission curve, efficiency
    function, etc.). Both must share the same wavelength grid. Differentiable
    through Rune.

    Raises [Invalid_argument] if wavelength grids have different lengths. *)

val div : 'a t -> sampled t -> 'a t
(** [div a b] divides values element-wise. [a]'s spectral kind is preserved; [b]
    is treated as a dimensionless modifier. Both must share the same wavelength
    grid. Differentiable through Rune.

    Raises [Invalid_argument] if wavelength grids have different lengths. *)

val add : 'a t -> 'a t -> 'a t
(** [add a b] is the element-wise sum of two spectra. Both must share the same
    wavelength grid. Differentiable through Rune.

    Raises [Invalid_argument] if wavelength grids have different lengths. *)

val resample : wavelength:Unit.length Unit.t -> 'a t -> 'a t
(** [resample ~wavelength s] resamples [s] onto a new wavelength grid using
    linear interpolation. Leading batch dimensions are preserved. Differentiable
    through Rune with respect to the spectrum values (index computation is not
    differentiable, but the interpolation weights and gather operations are).

    Raises [Invalid_argument] if [wavelength] is not 1-D or not strictly
    increasing. *)

(** {1:lines Line profiles} *)

val gaussian :
  amplitude:Nx.float64_t ->
  center:Unit.length Unit.t ->
  stddev:Unit.length Unit.t ->
  wavelength:Unit.length Unit.t ->
  sampled t
(** [gaussian ~amplitude ~center ~stddev ~wavelength] is the Gaussian profile
    [amplitude * exp(-0.5 * ((lambda - center) / stddev){^2})].

    [amplitude], [center], and [stddev] may be scalar tensors; they broadcast
    against [wavelength]. Differentiable through Rune. *)

val lorentzian :
  amplitude:Nx.float64_t ->
  center:Unit.length Unit.t ->
  fwhm:Unit.length Unit.t ->
  wavelength:Unit.length Unit.t ->
  sampled t
(** [lorentzian ~amplitude ~center ~fwhm ~wavelength] is the Lorentzian profile
    [amplitude * (gamma/2){^2} / ((lambda - center){^2} + (gamma/2){^2})] where
    [gamma = fwhm]. Unit height at [center]. Differentiable through Rune. *)

val voigt :
  amplitude:Nx.float64_t ->
  center:Unit.length Unit.t ->
  sigma:Unit.length Unit.t ->
  gamma:Unit.length Unit.t ->
  wavelength:Unit.length Unit.t ->
  sampled t
(** [voigt ~amplitude ~center ~sigma ~gamma ~wavelength] is the pseudo-Voigt
    approximation of the Voigt profile (Thompson, Cox & Hastings 1987). [sigma]
    is the Gaussian standard deviation and [gamma] is the Lorentzian half-width
    at half-maximum. Accurate to <1% of the exact Faddeeva-based Voigt.
    Differentiable through Rune. *)
