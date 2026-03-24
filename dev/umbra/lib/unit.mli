(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Physical quantities with compile-time dimensional safety.

    A {e quantity} is an {!Nx.float64_t} tensor of arbitrary shape tagged with a
    phantom dimension type. Arithmetic requires matching dimensions:
    [length t + length t] typechecks, [length t + mass t] does not. Values are
    stored in SI base units internally.

    Each dimension module provides three families of functions:
    - {e Scalar constructors} ([Length.km], [Mass.kg], ...) create a 0-d
      quantity from a [float].
    - {e Tensor constructors} ([Length.of_km], [Mass.of_kg], ...) wrap an
      arbitrary-shape {!Nx.float64_t}.
    - {e Extractors} ([Length.in_km], [Mass.in_kg], ...) return the numeric
      value in a given unit as an {!Nx.float64_t}.

    {[
    open Unit

    let d = Length.(kpc 10.0 + pc 500.0)
    let d_mpc = Length.in_mpc d
    ]} *)

(** {1:types Types} *)

type 'a t
(** The type for a physical quantity with dimension ['a]. Internally an
    {!Nx.float64_t} in SI base units. *)

type length
(** Phantom type for length (SI: metres). *)

type mass
(** Phantom type for mass (SI: kilograms). *)

type time
(** Phantom type for time duration (SI: seconds). *)

type angle
(** Phantom type for angles (SI: radians). *)

type velocity
(** Phantom type for velocity (SI: m/s). *)

type power
(** Phantom type for power / luminosity (SI: watts). *)

type temperature
(** Phantom type for temperature (SI: kelvin). *)

type energy
(** Phantom type for energy (SI: joules). *)

type frequency
(** Phantom type for frequency (SI: hertz). *)

type dimensionless
(** Phantom type for dimensionless quantities. *)

(** {1:arithmetic Arithmetic}

    All operations require matching dimensions. *)

val ( + ) : 'a t -> 'a t -> 'a t
(** [a + b] is the element-wise sum of [a] and [b]. *)

val ( - ) : 'a t -> 'a t -> 'a t
(** [a - b] is the element-wise difference of [a] and [b]. *)

val neg : 'a t -> 'a t
(** [neg x] is the element-wise negation of [x]. *)

val abs : 'a t -> 'a t
(** [abs x] is the element-wise absolute value of [x]. *)

val scale : float -> 'a t -> 'a t
(** [scale s x] multiplies every element of [x] by [s]. *)

val scale_t : Nx.float64_t -> 'a t -> 'a t
(** [scale_t s x] multiplies every element of [x] by the tensor [s]. Keeps the
    result in the typed world when the scale factor is a fitted parameter. *)

val ratio : 'a t -> 'a t -> dimensionless t
(** [ratio a b] is the element-wise division [a / b], yielding a dimensionless
    quantity. *)

val zero : 'a t
(** [zero] is the scalar quantity [0.0]. *)

(** {1:predicates Predicates, comparisons, and converting}

    These functions extract scalar values and are intended for 0-d tensors. *)

val compare : 'a t -> 'a t -> int
(** [compare a b] orders [a] and [b] by their scalar SI values. *)

val equal : 'a t -> 'a t -> bool
(** [equal a b] is [true] iff [a] and [b] have the same scalar SI value. *)

val pp : Format.formatter -> 'a t -> unit
(** [pp] formats the scalar SI value of a quantity. *)

val to_float : 'a t -> float
(** [to_float x] is the scalar value of [x] in SI base units. *)

(** {1:cross Cross-dimension combinators}

    Functions that relate quantities of different dimensions. *)

val length_per_time : length t -> time t -> velocity t
(** [length_per_time d t] is [d / t] as a velocity. *)

val velocity_times_time : velocity t -> time t -> length t
(** [velocity_times_time v t] is [v * t] as a length. *)

val length_per_velocity : length t -> velocity t -> time t
(** [length_per_velocity d v] is [d / v] as a time. *)

val wavelength_to_frequency : length t -> frequency t
(** [wavelength_to_frequency lam] is [c / lam]. *)

val frequency_to_wavelength : frequency t -> length t
(** [frequency_to_wavelength nu] is [c / nu]. *)

val frequency_to_energy : frequency t -> energy t
(** [frequency_to_energy nu] is [h * nu]. *)

val energy_to_frequency : energy t -> frequency t
(** [energy_to_frequency e] is [e / h]. *)

val energy_to_wavelength : energy t -> length t
(** [energy_to_wavelength e] is [h * c / e]. *)

val parallax_to_distance : angle t -> length t
(** [parallax_to_distance p] is the distance corresponding to parallax [p]. Uses
    [d = 1 AU / p]. One arcsecond of parallax gives one parsec. *)

val distance_to_parallax : length t -> angle t
(** [distance_to_parallax d] is the parallax corresponding to distance [d]. Uses
    [p = 1 AU / d]. *)

val flam_to_fnu : wavelength:length t -> Nx.float64_t -> Nx.float64_t
(** [flam_to_fnu ~wavelength flam] converts spectral flux density from
    per-wavelength (F{_ {e lambda}}, W m{^ -2} m{^ -1}) to per-frequency
    (F{_ {e nu}}, W m{^ -2} Hz{^ -1}) at the given wavelengths. Uses
    [f_nu = f_lambda * lambda{^ 2} / c]. *)

val fnu_to_flam : wavelength:length t -> Nx.float64_t -> Nx.float64_t
(** [fnu_to_flam ~wavelength fnu] converts spectral flux density from
    per-frequency (F{_ {e nu}}) to per-wavelength (F{_ {e lambda}}) at the given
    wavelengths. Uses [f_lambda = f_nu * c / lambda{^ 2}]. *)

(** {2:doppler Doppler conventions}

    Three conventions for converting between radial velocity and observed
    wavelength, given a rest wavelength. All agree at [v << c]. *)

val doppler_optical : rest:length t -> velocity t -> length t
(** [doppler_optical ~rest v] is the observed wavelength under the optical (cz)
    convention: [lambda_obs = lambda_rest * (1 + v/c)]. *)

val doppler_optical_inv : rest:length t -> length t -> velocity t
(** [doppler_optical_inv ~rest obs] is the velocity under the optical
    convention: [v = c * (lambda_obs/lambda_rest - 1)]. *)

val doppler_radio : rest:length t -> velocity t -> length t
(** [doppler_radio ~rest v] is the observed wavelength under the radio
    convention: [lambda_obs = lambda_rest / (1 - v/c)]. *)

val doppler_radio_inv : rest:length t -> length t -> velocity t
(** [doppler_radio_inv ~rest obs] is the velocity under the radio convention:
    [v = c * (1 - lambda_rest/lambda_obs)]. *)

val doppler_relativistic : rest:length t -> velocity t -> length t
(** [doppler_relativistic ~rest v] is the observed wavelength under the full
    relativistic Doppler formula:
    [lambda_obs = lambda_rest * sqrt((1 + v/c) / (1 - v/c))]. *)

val doppler_relativistic_inv : rest:length t -> length t -> velocity t
(** [doppler_relativistic_inv ~rest obs] is the velocity under the relativistic
    formula: [v = c * (r{^ 2} - 1) / (r{^ 2} + 1)] where
    [r = lambda_obs/lambda_rest]. *)

(** {1:length Length} *)

module Length : sig
  val of_tensor : Nx.float64_t -> length t
  (** [of_tensor x] wraps [x] as a length. [x] must be in metres. *)

  val to_tensor : length t -> Nx.float64_t
  (** [to_tensor x] is the underlying tensor in metres. *)

  (** {2:scalar Scalar constructors}

      Each function creates a 0-d length quantity from a [float] value in the
      named unit. *)

  val m : float -> length t
  (** [m x] is [x] metres. *)

  val km : float -> length t
  (** [km x] is [x] kilometres. *)

  val cm : float -> length t
  (** [cm x] is [x] centimetres. *)

  val mm : float -> length t
  (** [mm x] is [x] millimetres. *)

  val um : float -> length t
  (** [um x] is [x] micrometres. *)

  val nm : float -> length t
  (** [nm x] is [x] nanometres. *)

  val angstrom : float -> length t
  (** [angstrom x] is [x] angstroms. *)

  val au : float -> length t
  (** [au x] is [x] astronomical units. *)

  val pc : float -> length t
  (** [pc x] is [x] parsecs. *)

  val kpc : float -> length t
  (** [kpc x] is [x] kiloparsecs. *)

  val mpc : float -> length t
  (** [mpc x] is [x] megaparsecs. *)

  val gpc : float -> length t
  (** [gpc x] is [x] gigaparsecs. *)

  val ly : float -> length t
  (** [ly x] is [x] light-years. *)

  val solar_radius : float -> length t
  (** [solar_radius x] is [x] solar radii. *)

  val earth_radius : float -> length t
  (** [earth_radius x] is [x] Earth equatorial radii. *)

  val jupiter_radius : float -> length t
  (** [jupiter_radius x] is [x] Jupiter equatorial radii. *)

  (** {2:tensor Tensor constructors}

      Each function wraps an arbitrary-shape {!Nx.float64_t} (in the named unit)
      as a length quantity. *)

  val of_m : Nx.float64_t -> length t
  val of_km : Nx.float64_t -> length t
  val of_cm : Nx.float64_t -> length t
  val of_mm : Nx.float64_t -> length t
  val of_um : Nx.float64_t -> length t
  val of_nm : Nx.float64_t -> length t
  val of_angstrom : Nx.float64_t -> length t
  val of_au : Nx.float64_t -> length t
  val of_pc : Nx.float64_t -> length t
  val of_kpc : Nx.float64_t -> length t
  val of_mpc : Nx.float64_t -> length t
  val of_gpc : Nx.float64_t -> length t
  val of_ly : Nx.float64_t -> length t
  val of_solar_radius : Nx.float64_t -> length t
  val of_earth_radius : Nx.float64_t -> length t
  val of_jupiter_radius : Nx.float64_t -> length t

  (** {2:extract Extracting}

      Each function returns the numeric value in the named unit as an
      {!Nx.float64_t}. *)

  val in_m : length t -> Nx.float64_t
  val in_km : length t -> Nx.float64_t
  val in_cm : length t -> Nx.float64_t
  val in_mm : length t -> Nx.float64_t
  val in_um : length t -> Nx.float64_t
  val in_nm : length t -> Nx.float64_t
  val in_angstrom : length t -> Nx.float64_t
  val in_au : length t -> Nx.float64_t
  val in_pc : length t -> Nx.float64_t
  val in_kpc : length t -> Nx.float64_t
  val in_mpc : length t -> Nx.float64_t
  val in_gpc : length t -> Nx.float64_t
  val in_ly : length t -> Nx.float64_t
  val in_solar_radius : length t -> Nx.float64_t
  val in_earth_radius : length t -> Nx.float64_t
  val in_jupiter_radius : length t -> Nx.float64_t
end

(** {1:mass Mass} *)

module Mass : sig
  val of_tensor : Nx.float64_t -> mass t
  (** [of_tensor x] wraps [x] as a mass. [x] must be in kilograms. *)

  val to_tensor : mass t -> Nx.float64_t
  (** [to_tensor x] is the underlying tensor in kilograms. *)

  (** {2:scalar Scalar constructors} *)

  val kg : float -> mass t
  (** [kg x] is [x] kilograms. *)

  val g : float -> mass t
  (** [g x] is [x] grams. *)

  val mg : float -> mass t
  (** [mg x] is [x] milligrams. *)

  val solar_mass : float -> mass t
  (** [solar_mass x] is [x] solar masses. *)

  val earth_mass : float -> mass t
  (** [earth_mass x] is [x] Earth masses. *)

  val jupiter_mass : float -> mass t
  (** [jupiter_mass x] is [x] Jupiter masses. *)

  (** {2:tensor Tensor constructors} *)

  val of_kg : Nx.float64_t -> mass t
  val of_g : Nx.float64_t -> mass t
  val of_mg : Nx.float64_t -> mass t
  val of_solar_mass : Nx.float64_t -> mass t
  val of_earth_mass : Nx.float64_t -> mass t
  val of_jupiter_mass : Nx.float64_t -> mass t

  (** {2:extract Extracting} *)

  val in_kg : mass t -> Nx.float64_t
  val in_g : mass t -> Nx.float64_t
  val in_mg : mass t -> Nx.float64_t
  val in_solar_mass : mass t -> Nx.float64_t
  val in_earth_mass : mass t -> Nx.float64_t
  val in_jupiter_mass : mass t -> Nx.float64_t
end

(** {1:time Time duration} *)

module Time : sig
  val of_tensor : Nx.float64_t -> time t
  (** [of_tensor x] wraps [x] as a time duration. [x] must be in seconds. *)

  val to_tensor : time t -> Nx.float64_t
  (** [to_tensor x] is the underlying tensor in seconds. *)

  (** {2:scalar Scalar constructors} *)

  val s : float -> time t
  (** [s x] is [x] seconds. *)

  val ms : float -> time t
  (** [ms x] is [x] milliseconds. *)

  val us : float -> time t
  (** [us x] is [x] microseconds. *)

  val min : float -> time t
  (** [min x] is [x] minutes. *)

  val hr : float -> time t
  (** [hr x] is [x] hours. *)

  val day : float -> time t
  (** [day x] is [x] days (86 400 s). *)

  val yr : float -> time t
  (** [yr x] is [x] Julian years (365.25 days). *)

  val myr : float -> time t
  (** [myr x] is [x] megayears. *)

  val gyr : float -> time t
  (** [gyr x] is [x] gigayears. *)

  (** {2:tensor Tensor constructors} *)

  val of_s : Nx.float64_t -> time t
  val of_ms : Nx.float64_t -> time t
  val of_us : Nx.float64_t -> time t
  val of_min : Nx.float64_t -> time t
  val of_hr : Nx.float64_t -> time t
  val of_day : Nx.float64_t -> time t
  val of_yr : Nx.float64_t -> time t
  val of_myr : Nx.float64_t -> time t
  val of_gyr : Nx.float64_t -> time t

  (** {2:extract Extracting} *)

  val in_s : time t -> Nx.float64_t
  val in_ms : time t -> Nx.float64_t
  val in_us : time t -> Nx.float64_t
  val in_min : time t -> Nx.float64_t
  val in_hr : time t -> Nx.float64_t
  val in_day : time t -> Nx.float64_t
  val in_yr : time t -> Nx.float64_t
  val in_myr : time t -> Nx.float64_t
  val in_gyr : time t -> Nx.float64_t
end

(** {1:angle Angle} *)

module Angle : sig
  val of_tensor : Nx.float64_t -> angle t
  (** [of_tensor x] wraps [x] as an angle. [x] must be in radians. *)

  val to_tensor : angle t -> Nx.float64_t
  (** [to_tensor x] is the underlying tensor in radians. *)

  (** {2:scalar Scalar constructors} *)

  val rad : float -> angle t
  (** [rad x] is [x] radians. *)

  val deg : float -> angle t
  (** [deg x] is [x] degrees. *)

  val arcmin : float -> angle t
  (** [arcmin x] is [x] arcminutes. *)

  val arcsec : float -> angle t
  (** [arcsec x] is [x] arcseconds. *)

  val mas : float -> angle t
  (** [mas x] is [x] milliarcseconds. *)

  val hour_angle : float -> angle t
  (** [hour_angle x] is [x] hour angles (1 h = 15 deg). *)

  (** {2:tensor Tensor constructors} *)

  val of_rad : Nx.float64_t -> angle t
  val of_deg : Nx.float64_t -> angle t
  val of_arcmin : Nx.float64_t -> angle t
  val of_arcsec : Nx.float64_t -> angle t
  val of_mas : Nx.float64_t -> angle t
  val of_hour_angle : Nx.float64_t -> angle t

  (** {2:extract Extracting} *)

  val in_rad : angle t -> Nx.float64_t
  val in_deg : angle t -> Nx.float64_t
  val in_arcmin : angle t -> Nx.float64_t
  val in_arcsec : angle t -> Nx.float64_t
  val in_mas : angle t -> Nx.float64_t
  val in_hour_angle : angle t -> Nx.float64_t

  (** {2:trig Trigonometric functions} *)

  val sin : angle t -> Nx.float64_t
  (** [sin a] is the sine of [a]. *)

  val cos : angle t -> Nx.float64_t
  (** [cos a] is the cosine of [a]. *)

  val tan : angle t -> Nx.float64_t
  (** [tan a] is the tangent of [a]. *)

  val asin : Nx.float64_t -> angle t
  (** [asin x] is the arc sine of [x]. *)

  val acos : Nx.float64_t -> angle t
  (** [acos x] is the arc cosine of [x]. *)

  val atan2 : y:Nx.float64_t -> x:Nx.float64_t -> angle t
  (** [atan2 ~y ~x] is the two-argument arc tangent of [y] and [x]. *)

  (** {2:wrap Wrapping} *)

  val wrap_360 : angle t -> angle t
  (** [wrap_360 a] normalizes [a] into \[0, 360) degrees. *)

  val wrap_180 : angle t -> angle t
  (** [wrap_180 a] normalizes [a] into \[-180, 180) degrees. *)
end

(** {1:velocity Velocity} *)

module Velocity : sig
  val of_tensor : Nx.float64_t -> velocity t
  (** [of_tensor x] wraps [x] as a velocity. [x] must be in m/s. *)

  val to_tensor : velocity t -> Nx.float64_t
  (** [to_tensor x] is the underlying tensor in m/s. *)

  (** {2:scalar Scalar constructors} *)

  val m_s : float -> velocity t
  (** [m_s x] is [x] m/s. *)

  val km_s : float -> velocity t
  (** [km_s x] is [x] km/s. *)

  val km_hr : float -> velocity t
  (** [km_hr x] is [x] km/h. *)

  (** {2:tensor Tensor constructors} *)

  val of_m_s : Nx.float64_t -> velocity t
  val of_km_s : Nx.float64_t -> velocity t
  val of_km_hr : Nx.float64_t -> velocity t

  (** {2:extract Extracting} *)

  val in_m_s : velocity t -> Nx.float64_t
  val in_km_s : velocity t -> Nx.float64_t
  val in_km_hr : velocity t -> Nx.float64_t
end

(** {1:power Power / Luminosity} *)

module Power : sig
  val of_tensor : Nx.float64_t -> power t
  (** [of_tensor x] wraps [x] as a power. [x] must be in watts. *)

  val to_tensor : power t -> Nx.float64_t
  (** [to_tensor x] is the underlying tensor in watts. *)

  (** {2:scalar Scalar constructors} *)

  val w : float -> power t
  (** [w x] is [x] watts. *)

  val kw : float -> power t
  (** [kw x] is [x] kilowatts. *)

  val solar_luminosity : float -> power t
  (** [solar_luminosity x] is [x] solar luminosities. *)

  val erg_s : float -> power t
  (** [erg_s x] is [x] erg/s. *)

  (** {2:tensor Tensor constructors} *)

  val of_w : Nx.float64_t -> power t
  val of_kw : Nx.float64_t -> power t
  val of_solar_luminosity : Nx.float64_t -> power t
  val of_erg_s : Nx.float64_t -> power t

  (** {2:extract Extracting} *)

  val in_w : power t -> Nx.float64_t
  val in_kw : power t -> Nx.float64_t
  val in_solar_luminosity : power t -> Nx.float64_t
  val in_erg_s : power t -> Nx.float64_t
end

(** {1:temperature Temperature} *)

module Temperature : sig
  val of_tensor : Nx.float64_t -> temperature t
  (** [of_tensor x] wraps [x] as a temperature. [x] must be in kelvin. *)

  val to_tensor : temperature t -> Nx.float64_t
  (** [to_tensor x] is the underlying tensor in kelvin. *)

  (** {2:scalar Scalar constructors} *)

  val kelvin : float -> temperature t
  (** [kelvin x] is [x] kelvin. *)

  (** {2:tensor Tensor constructors} *)

  val of_kelvin : Nx.float64_t -> temperature t

  (** {2:extract Extracting} *)

  val in_kelvin : temperature t -> Nx.float64_t
end

(** {1:energy Energy} *)

module Energy : sig
  val of_tensor : Nx.float64_t -> energy t
  (** [of_tensor x] wraps [x] as an energy. [x] must be in joules. *)

  val to_tensor : energy t -> Nx.float64_t
  (** [to_tensor x] is the underlying tensor in joules. *)

  (** {2:scalar Scalar constructors} *)

  val j : float -> energy t
  (** [j x] is [x] joules. *)

  val erg : float -> energy t
  (** [erg x] is [x] ergs. *)

  val ev : float -> energy t
  (** [ev x] is [x] electronvolts. *)

  val kev : float -> energy t
  (** [kev x] is [x] kiloelectronvolts. *)

  val mev : float -> energy t
  (** [mev x] is [x] megaelectronvolts. *)

  (** {2:tensor Tensor constructors} *)

  val of_j : Nx.float64_t -> energy t
  val of_erg : Nx.float64_t -> energy t
  val of_ev : Nx.float64_t -> energy t
  val of_kev : Nx.float64_t -> energy t
  val of_mev : Nx.float64_t -> energy t

  (** {2:extract Extracting} *)

  val in_j : energy t -> Nx.float64_t
  val in_erg : energy t -> Nx.float64_t
  val in_ev : energy t -> Nx.float64_t
  val in_kev : energy t -> Nx.float64_t
  val in_mev : energy t -> Nx.float64_t
end

(** {1:frequency Frequency} *)

module Frequency : sig
  val of_tensor : Nx.float64_t -> frequency t
  (** [of_tensor x] wraps [x] as a frequency. [x] must be in hertz. *)

  val to_tensor : frequency t -> Nx.float64_t
  (** [to_tensor x] is the underlying tensor in hertz. *)

  (** {2:scalar Scalar constructors} *)

  val hz : float -> frequency t
  (** [hz x] is [x] hertz. *)

  val khz : float -> frequency t
  (** [khz x] is [x] kilohertz. *)

  val mhz : float -> frequency t
  (** [mhz x] is [x] megahertz. *)

  val ghz : float -> frequency t
  (** [ghz x] is [x] gigahertz. *)

  (** {2:tensor Tensor constructors} *)

  val of_hz : Nx.float64_t -> frequency t
  val of_khz : Nx.float64_t -> frequency t
  val of_mhz : Nx.float64_t -> frequency t
  val of_ghz : Nx.float64_t -> frequency t

  (** {2:extract Extracting} *)

  val in_hz : frequency t -> Nx.float64_t
  val in_khz : frequency t -> Nx.float64_t
  val in_mhz : frequency t -> Nx.float64_t
  val in_ghz : frequency t -> Nx.float64_t
end

(** {1:dimensionless Dimensionless} *)

module Dimensionless : sig
  val of_tensor : Nx.float64_t -> dimensionless t
  (** [of_tensor x] wraps [x] as a dimensionless quantity. *)

  val to_tensor : dimensionless t -> Nx.float64_t
  (** [to_tensor x] is the underlying tensor. *)

  val v : float -> dimensionless t
  (** [v x] is the scalar dimensionless quantity [x]. *)

  val to_float : dimensionless t -> float
  (** [to_float x] is the scalar value of [x]. Intended for 0-d tensors. *)
end
