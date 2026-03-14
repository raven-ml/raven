(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Fundamental constants (CODATA 2022) *)

let c = Unit.Velocity.m_s 299_792_458.0
let m_e = Unit.Mass.kg 9.109_383_713_9e-31
let m_p = Unit.Mass.kg 1.672_621_923_69e-27
let m_n = Unit.Mass.kg 1.674_927_498_04e-27
let u = Unit.Mass.kg 1.660_539_066_60e-27

(* Astronomical constants (IAU 2015) *)

let au = Unit.Length.au 1.0
let pc = Unit.Length.pc 1.0
let solar_mass = Unit.Mass.solar_mass 1.0
let solar_radius = Unit.Length.solar_radius 1.0
let solar_luminosity = Unit.Power.solar_luminosity 1.0
let earth_mass = Unit.Mass.earth_mass 1.0
let earth_radius = Unit.Length.earth_radius 1.0
let jupiter_mass = Unit.Mass.jupiter_mass 1.0
let jupiter_radius = Unit.Length.jupiter_radius 1.0

(* Raw SI floats for compound dimensions (CODATA 2022) *)

let h_si = 6.626_070_15e-34
let hbar_si = 1.054_571_817e-34
let g_si = 6.674_30e-11
let k_b_si = 1.380_649e-23
let sigma_sb_si = 5.670_374_419e-8
let n_a = 6.022_140_76e23
let sigma_t_si = 6.652_458_705_1e-29
let b_wien_si = 2.897_771_955e-3
let alpha = 7.297_352_5643e-3
let a_0 = Unit.Length.m 5.291_772_105_44e-11
let gm_sun_si = 1.327_124_4e20
let gm_earth_si = 3.986_004e14
let gm_jup_si = 1.266_865_3e17
let l_bol0 = Unit.Power.w 3.0128e28
