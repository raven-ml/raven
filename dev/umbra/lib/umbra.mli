(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Computational astronomy for OCaml.

    Umbra provides dimensionally-typed physical quantities, astronomical
    constants, cosmological distance calculations, spectral energy
    distributions, dust extinction, synthetic photometry, coordinate transforms,
    time scales, and catalog cross-matching.

    All computations operate on {!Nx} tensors and are differentiable through
    {!Rune} by default.

    {[
      open Umbra

      let z = Nx.scalar Nx.float64 0.5 in
      let dl = Cosmo.luminosity_distance z in
      let dl_mpc = Unit.Length.in_mpc dl in

      let rv = Nx.scalar Nx.float64 3.1 in
      let av = Nx.scalar Nx.float64 0.5 in
      let wave = Unit.Length.of_m (Nx.linspace Nx.float64 3e-7 1e-6 1000) in
      let bp = Photometry.tophat
        ~lo:(Unit.Length.nm 400.0) ~hi:(Unit.Length.nm 700.0) ~n:1000 in
      let sed =
        Spectrum.blackbody
          ~temperature:(Unit.Temperature.of_kelvin (Nx.scalar Nx.float64 5800.0))
          ~wavelength:wave
        |> Extinction.apply (Extinction.ccm89 ~rv) ~av
        |> Spectrum.as_flux_density
      in
      let mag = Photometry.ab_mag bp sed
    ]} *)

(** {1:units Units and constants} *)

module Unit = Unit
(** Physical quantities with compile-time dimensional safety. *)

module Const = Const
(** Physical and astronomical constants (CODATA 2022, IAU 2015). *)

(** {1:astro Astronomy} *)

module Time = Time
(** Astronomical time with phantom-typed time scales. *)

module Coord = Coord
(** Celestial coordinates with frame transforms and catalog cross-matching. *)

module Altaz = Altaz
(** Altitude-azimuth (horizontal) coordinates. *)

module Galactocentric = Galactocentric
(** Galactocentric Cartesian coordinates. *)

module Cosmo = Cosmo
(** Cosmological distances for {e Λ}CDM, wCDM, and w0waCDM universes. *)

module Spectrum = Spectrum
(** Sampled spectral values on a wavelength grid. *)

module Extinction = Extinction
(** Dust extinction laws. *)

module Photometry = Photometry
(** Synthetic photometry over filter bandpasses. *)

module Filters = Filters
(** Standard astronomical filter bandpasses (SDSS, Johnson-Cousins, 2MASS, Gaia
    DR3). *)

(** {1:survey Survey science} *)

module Survey = Survey
(** Angular power spectra, probes, and survey likelihood. *)
