(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Synthetic photometry.

    Computes broadband fluxes and magnitudes by integrating spectra through
    filter bandpasses using trapezoidal quadrature.

    {[
      let bp = Photometry.tophat
        ~lo:(Unit.Length.nm 400.0)
        ~hi:(Unit.Length.nm 500.0) ~n:100 in
      let mag = Photometry.ab_mag bp sed
    ]}

    All photometry functions accept batched spectra (values with leading batch
    dimensions). When a spectrum has shape [[batch; n_lambda]], the result has
    shape [[batch]]. *)

(** {1:types Types} *)

type bandpass
(** The type for filter transmission curves. *)

type detector =
  | Energy
  | Photon
      (** The detector convention.

          - {!Energy}: counts incident energy (default). The bandpass-weighted
            mean is [<f_nu> = integral f_nu T d lambda / integral T d lambda].
          - {!Photon}: counts photons. Weights both numerator and denominator by
            [lambda]:
            [<f_nu> = integral f_nu T lambda d lambda / integral T lambda d
             lambda]. *)

(** {1:constructors Constructors} *)

val bandpass :
  wavelength:Unit.length Unit.t -> throughput:Nx.float64_t -> bandpass
(** [bandpass ~wavelength ~throughput] is a filter from 1-D arrays. [throughput]
    is dimensionless (typically in \[0, 1\]).

    Raises [Invalid_argument] if tensors are not 1-D or have different lengths.
*)

val tophat : lo:Unit.length Unit.t -> hi:Unit.length Unit.t -> n:int -> bandpass
(** [tophat ~lo ~hi ~n] is a rectangular bandpass from [lo] to [hi] with [n]
    wavelength points and unit throughput. *)

(** {1:accessors Accessors} *)

val wavelength : bandpass -> Unit.length Unit.t
(** [wavelength bp] is the wavelength grid. *)

val throughput : bandpass -> Nx.float64_t
(** [throughput bp] is the throughput curve. *)

val pivot_wavelength : bandpass -> Unit.length Unit.t
(** [pivot_wavelength bp] is the pivot wavelength
    {e lambda}{_ p}[ = sqrt(integral T lambda d lambda / integral T/lambda d
                    lambda)]. *)

(** {1:photometry Synthetic photometry} *)

val flux_density :
  ?detector:detector ->
  bandpass ->
  Spectrum.flux_density Spectrum.t ->
  Nx.float64_t
(** [flux_density ?detector bp spectrum] is the bandpass-weighted mean flux
    density [<f> = integral f T w d lambda / integral T w d lambda] where [w] is
    [1] for {!Energy} and [lambda] for {!Photon}. [detector] defaults to
    {!Energy}.

    The spectrum is resampled to the bandpass wavelength grid via linear
    interpolation if they differ. Differentiable through Rune. *)

val ab_mag :
  ?detector:detector ->
  bandpass ->
  Spectrum.flux_density Spectrum.t ->
  Nx.float64_t
(** [ab_mag ?detector bp spectrum] is the AB magnitude of [spectrum] through
    [bp].

    Computes the mean spectral flux density in f{_ nu}:
    [<f_nu> = integral (f_lambda lambda{^2}/c) T w d lambda / integral T w d
     lambda], where [w] is [1] for {!Energy} and [lambda] for {!Photon}, then
    [m_AB = -2.5 log10(<f_nu> / 3631 Jy)]. [detector] defaults to {!Energy}.

    The spectrum is resampled to the bandpass wavelength grid via linear
    interpolation if they differ. Differentiable through Rune. *)

val st_mag :
  ?detector:detector ->
  bandpass ->
  Spectrum.flux_density Spectrum.t ->
  Nx.float64_t
(** [st_mag ?detector bp spectrum] is the ST magnitude of [spectrum] through
    [bp].

    Computes the bandpass-weighted mean f{_ lambda}, then
    [m_ST = -2.5 log10(<f_lambda> / 3.63e-9 erg s{^-1} cm{^-2} A{^-1})].
    [detector] defaults to {!Energy}.

    The spectrum is resampled to the bandpass wavelength grid via linear
    interpolation if they differ. Differentiable through Rune. *)

val vega_mag :
  ?detector:detector ->
  bandpass ->
  Spectrum.flux_density Spectrum.t ->
  Nx.float64_t
(** [vega_mag ?detector bp spectrum] is the Vega magnitude of [spectrum] through
    [bp].

    Computes [-2.5 log10(<f_lambda> / <f_lambda,Vega>)] where the Vega reference
    spectrum is from CALSPEC alpha_lyr_stis_011.fits (Bohlin 2014). [detector]
    defaults to {!Energy}.

    The spectrum is resampled to the bandpass wavelength grid via linear
    interpolation if they differ. Differentiable through Rune. *)

val color :
  ?detector:detector ->
  bandpass ->
  bandpass ->
  Spectrum.flux_density Spectrum.t ->
  Nx.float64_t
(** [color ?detector bp1 bp2 spectrum] is
    [ab_mag ?detector bp1 spectrum - ab_mag ?detector bp2 spectrum].

    Differentiable through Rune. *)

val effective_wavelength :
  ?detector:detector ->
  bandpass ->
  Spectrum.flux_density Spectrum.t ->
  Unit.length Unit.t
(** [effective_wavelength ?detector bp spectrum] is the source-dependent
    effective wavelength
    {e lambda}{_ eff}[ = integral f T w lambda{^2} d lambda / integral f T w
                      lambda d lambda].

    Unlike {!pivot_wavelength}, this depends on the source spectrum. The
    spectrum is resampled if grids differ. Differentiable through Rune. *)
