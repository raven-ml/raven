(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Physical and astronomical constants.

    Typed constants use {!Unit.t} with the appropriate phantom dimension. Raw SI
    floats are provided for compound dimensions that do not map to a single
    {!Unit} dimension type.

    Fundamental constants follow
    {{:https://physics.nist.gov/cuu/Constants/}CODATA 2022}. Astronomical
    constants follow IAU 2015. *)

(** {1:fundamental Fundamental constants} *)

val c : Unit.velocity Unit.t
(** [c] is the speed of light in vacuum (299 792 458 m/s, exact). *)

(** {1:particle Particle masses} *)

val m_e : Unit.mass Unit.t
(** [m_e] is the electron mass (9.109 383 7139e-31 kg). *)

val m_p : Unit.mass Unit.t
(** [m_p] is the proton mass (1.672 621 923 69e-27 kg). *)

val m_n : Unit.mass Unit.t
(** [m_n] is the neutron mass (1.674 927 498 04e-27 kg). *)

val u : Unit.mass Unit.t
(** [u] is the atomic mass unit (1.660 539 066 60e-27 kg). *)

(** {1:astro Astronomical constants} *)

val au : Unit.length Unit.t
(** [au] is one astronomical unit. *)

val pc : Unit.length Unit.t
(** [pc] is one parsec. *)

val solar_mass : Unit.mass Unit.t
(** [solar_mass] is one solar mass. *)

val solar_radius : Unit.length Unit.t
(** [solar_radius] is one solar radius. *)

val solar_luminosity : Unit.power Unit.t
(** [solar_luminosity] is one solar luminosity. *)

val earth_mass : Unit.mass Unit.t
(** [earth_mass] is one Earth mass. *)

val earth_radius : Unit.length Unit.t
(** [earth_radius] is one Earth radius. *)

val jupiter_mass : Unit.mass Unit.t
(** [jupiter_mass] is one Jupiter mass. *)

val jupiter_radius : Unit.length Unit.t
(** [jupiter_radius] is one Jupiter radius. *)

(** {1:si Raw SI constants}

    Constants with compound dimensions that do not map to a single {!Unit}
    dimension type. CODATA 2022 values. *)

val h_si : float
(** [h_si] is the Planck constant (6.626 070 15e-34 J s, exact). *)

val hbar_si : float
(** [hbar_si] is the reduced Planck constant (1.054 571 817e-34 J s). *)

val g_si : float
(** [g_si] is the gravitational constant (6.674 30e-11 m{^ 3} kg{^ -1} s{^ -2}).
*)

val k_b_si : float
(** [k_b_si] is the Boltzmann constant (1.380 649e-23 J K{^ -1}, exact). *)

val sigma_sb_si : float
(** [sigma_sb_si] is the Stefan-Boltzmann constant (5.670 374 419e-8 W m{^ -2}
    K{^ -4}). *)

val n_a : float
(** [n_a] is the Avogadro constant (6.022 140 76e23 mol{^ -1}, exact). *)

val sigma_t_si : float
(** [sigma_t_si] is the Thomson scattering cross-section (6.652 458 705 1e-29
    m{^ 2}). *)

val b_wien_si : float
(** [b_wien_si] is the Wien displacement law constant (2.897 771 955e-3 m K). *)

val alpha : float
(** [alpha] is the fine-structure constant (7.297 352 5643e-3). *)

val a_0 : Unit.length Unit.t
(** [a_0] is the Bohr radius (5.291 772 105 44e-11 m). *)

val gm_sun_si : float
(** [gm_sun_si] is the solar mass parameter (1.327 124 4e20 m{^ 3} s{^ -2}).
    More precise than [g_si * solar_mass] for orbital mechanics. *)

val gm_earth_si : float
(** [gm_earth_si] is the Earth mass parameter (3.986 004e14 m{^ 3} s{^ -2}). *)

val gm_jup_si : float
(** [gm_jup_si] is the Jupiter mass parameter (1.266 865 3e17 m{^ 3} s{^ -2}).
*)

val l_bol0 : Unit.power Unit.t
(** [l_bol0] is the IAU 2015 zero-point bolometric luminosity (3.0128e28 W). *)
