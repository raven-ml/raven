(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let f64 = Nx.float64

type 'a t = Nx.float64_t
type length
type mass
type time
type angle
type velocity
type power
type temperature
type energy
type frequency
type dimensionless

(* Arithmetic — all Nx ops, fully traced by rune *)

let ( + ) a b = Nx.add a b
let ( - ) a b = Nx.sub a b
let neg x = Nx.neg x
let abs x = Nx.abs x
let scale s x = Nx.mul_s x s
let scale_t s x = Nx.mul s x
let ratio a b = Nx.div a b
let zero = Nx.scalar f64 0.0
let compare a b = Float.compare (Nx.item [] a) (Nx.item [] b)
let equal a b = Float.equal (Nx.item [] a) (Nx.item [] b)
let pp fmt x = Format.fprintf fmt "%g" (Nx.item [] x)
let to_float x = Nx.item [] x

(* Physical constants used in cross-dimension combinators *)

let c_m_s = Nx.scalar f64 299_792_458.0
let h_si = Nx.scalar f64 6.626_070_15e-34
let hc_si = Nx.scalar f64 (6.626_070_15e-34 *. 299_792_458.0)
let one = Nx.scalar f64 1.0
let au_m_t = Nx.scalar f64 1.495_978_707e11

(* Cross-dimension combinators — all Nx ops *)

let length_per_time d t = Nx.div d t
let velocity_times_time v t = Nx.mul v t
let length_per_velocity d v = Nx.div d v
let wavelength_to_frequency lam = Nx.div c_m_s lam
let frequency_to_wavelength nu = Nx.div c_m_s nu
let frequency_to_energy nu = Nx.mul h_si nu
let energy_to_frequency e = Nx.div e h_si
let energy_to_wavelength e = Nx.div hc_si e

(* Parallax: 1 arcsec ↔ 1 parsec. parallax(rad) = 1 AU / distance(m), so
   distance(m) = 1 AU / parallax(rad). Uses the scale factors defined below. *)

let parallax_to_distance p = Nx.div au_m_t p
let distance_to_parallax d = Nx.div au_m_t d

(* Spectral density: f_ν = f_λ · λ²/c, where f_λ is per-metre and f_ν is
   per-hertz. *)
let flam_to_fnu ~wavelength flam =
  Nx.div (Nx.mul flam (Nx.square wavelength)) c_m_s

let fnu_to_flam ~wavelength fnu =
  Nx.div (Nx.mul fnu c_m_s) (Nx.square wavelength)

(* Doppler conventions: velocity ↔ observed wavelength given rest wavelength.
   All three conventions agree at v << c. *)

(* Optical: z = v/c, λ_obs = λ_rest * (1 + v/c) *)
let doppler_optical ~rest v = Nx.mul rest (Nx.add_s (Nx.div v c_m_s) 1.0)

let doppler_optical_inv ~rest obs =
  Nx.mul c_m_s (Nx.sub_s (Nx.div obs rest) 1.0)

(* Radio: v = c*(1 - λ_rest/λ_obs), λ_obs = λ_rest / (1 - v/c) *)
let doppler_radio ~rest v = Nx.div rest (Nx.sub one (Nx.div v c_m_s))
let doppler_radio_inv ~rest obs = Nx.mul c_m_s (Nx.sub one (Nx.div rest obs))

(* Relativistic: λ_obs = λ_rest * sqrt((1+β)/(1-β)), β = v/c *)
let doppler_relativistic ~rest v =
  let beta = Nx.div v c_m_s in
  Nx.mul rest (Nx.sqrt (Nx.div (Nx.add_s beta 1.0) (Nx.sub one beta)))

let doppler_relativistic_inv ~rest obs =
  let r2 = Nx.square (Nx.div obs rest) in
  Nx.mul c_m_s (Nx.div (Nx.sub_s r2 1.0) (Nx.add_s r2 1.0))

(* Scale factors to SI base unit *)

let pc_m = 3.085_677_581_491_367_3e16
let au_m = 1.495_978_707e11
let ly_m = 9.460_730_472_580_8e15
let solar_radius_m = 6.957e8
let earth_radius_m = 6.371e6
let jupiter_radius_m = 7.1492e7
let solar_mass_kg = 1.988_4e30
let earth_mass_kg = 5.972_2e24
let jupiter_mass_kg = 1.898_2e27
let solar_luminosity_w = 3.828e26
let julian_year_s = 365.25 *. 86_400.0
let ev_j = 1.602_176_634e-19

module Length = struct
  let of_tensor x = x
  let to_tensor x = x
  let m x = Nx.scalar f64 x
  let km x = Nx.scalar f64 (x *. 1e3)
  let cm x = Nx.scalar f64 (x *. 1e-2)
  let mm x = Nx.scalar f64 (x *. 1e-3)
  let um x = Nx.scalar f64 (x *. 1e-6)
  let nm x = Nx.scalar f64 (x *. 1e-9)
  let angstrom x = Nx.scalar f64 (x *. 1e-10)
  let au x = Nx.scalar f64 (x *. au_m)
  let pc x = Nx.scalar f64 (x *. pc_m)
  let kpc x = Nx.scalar f64 (x *. pc_m *. 1e3)
  let mpc x = Nx.scalar f64 (x *. pc_m *. 1e6)
  let gpc x = Nx.scalar f64 (x *. pc_m *. 1e9)
  let ly x = Nx.scalar f64 (x *. ly_m)
  let solar_radius x = Nx.scalar f64 (x *. solar_radius_m)
  let earth_radius x = Nx.scalar f64 (x *. earth_radius_m)
  let jupiter_radius x = Nx.scalar f64 (x *. jupiter_radius_m)
  let of_m x = x
  let of_km x = Nx.mul_s x 1e3
  let of_cm x = Nx.mul_s x 1e-2
  let of_mm x = Nx.mul_s x 1e-3
  let of_um x = Nx.mul_s x 1e-6
  let of_nm x = Nx.mul_s x 1e-9
  let of_angstrom x = Nx.mul_s x 1e-10
  let of_au x = Nx.mul_s x au_m
  let of_pc x = Nx.mul_s x pc_m
  let of_kpc x = Nx.mul_s x (pc_m *. 1e3)
  let of_mpc x = Nx.mul_s x (pc_m *. 1e6)
  let of_gpc x = Nx.mul_s x (pc_m *. 1e9)
  let of_ly x = Nx.mul_s x ly_m
  let of_solar_radius x = Nx.mul_s x solar_radius_m
  let of_earth_radius x = Nx.mul_s x earth_radius_m
  let of_jupiter_radius x = Nx.mul_s x jupiter_radius_m
  let in_m x = x
  let in_km x = Nx.div_s x 1e3
  let in_cm x = Nx.mul_s x (1.0 /. 1e-2)
  let in_mm x = Nx.mul_s x (1.0 /. 1e-3)
  let in_um x = Nx.mul_s x (1.0 /. 1e-6)
  let in_nm x = Nx.mul_s x (1.0 /. 1e-9)
  let in_angstrom x = Nx.mul_s x (1.0 /. 1e-10)
  let in_au x = Nx.div_s x au_m
  let in_pc x = Nx.div_s x pc_m
  let in_kpc x = Nx.div_s x (pc_m *. 1e3)
  let in_mpc x = Nx.div_s x (pc_m *. 1e6)
  let in_gpc x = Nx.div_s x (pc_m *. 1e9)
  let in_ly x = Nx.div_s x ly_m
  let in_solar_radius x = Nx.div_s x solar_radius_m
  let in_earth_radius x = Nx.div_s x earth_radius_m
  let in_jupiter_radius x = Nx.div_s x jupiter_radius_m
end

module Mass = struct
  let of_tensor x = x
  let to_tensor x = x
  let kg x = Nx.scalar f64 x
  let g x = Nx.scalar f64 (x *. 1e-3)
  let mg x = Nx.scalar f64 (x *. 1e-6)
  let solar_mass x = Nx.scalar f64 (x *. solar_mass_kg)
  let earth_mass x = Nx.scalar f64 (x *. earth_mass_kg)
  let jupiter_mass x = Nx.scalar f64 (x *. jupiter_mass_kg)
  let of_kg x = x
  let of_g x = Nx.mul_s x 1e-3
  let of_mg x = Nx.mul_s x 1e-6
  let of_solar_mass x = Nx.mul_s x solar_mass_kg
  let of_earth_mass x = Nx.mul_s x earth_mass_kg
  let of_jupiter_mass x = Nx.mul_s x jupiter_mass_kg
  let in_kg x = x
  let in_g x = Nx.mul_s x (1.0 /. 1e-3)
  let in_mg x = Nx.mul_s x (1.0 /. 1e-6)
  let in_solar_mass x = Nx.div_s x solar_mass_kg
  let in_earth_mass x = Nx.div_s x earth_mass_kg
  let in_jupiter_mass x = Nx.div_s x jupiter_mass_kg
end

module Time = struct
  let of_tensor x = x
  let to_tensor x = x
  let s x = Nx.scalar f64 x
  let ms x = Nx.scalar f64 (x *. 1e-3)
  let us x = Nx.scalar f64 (x *. 1e-6)
  let min x = Nx.scalar f64 (x *. 60.0)
  let hr x = Nx.scalar f64 (x *. 3600.0)
  let day x = Nx.scalar f64 (x *. 86_400.0)
  let yr x = Nx.scalar f64 (x *. julian_year_s)
  let myr x = Nx.scalar f64 (x *. julian_year_s *. 1e6)
  let gyr x = Nx.scalar f64 (x *. julian_year_s *. 1e9)
  let of_s x = x
  let of_ms x = Nx.mul_s x 1e-3
  let of_us x = Nx.mul_s x 1e-6
  let of_min x = Nx.mul_s x 60.0
  let of_hr x = Nx.mul_s x 3600.0
  let of_day x = Nx.mul_s x 86_400.0
  let of_yr x = Nx.mul_s x julian_year_s
  let of_myr x = Nx.mul_s x (julian_year_s *. 1e6)
  let of_gyr x = Nx.mul_s x (julian_year_s *. 1e9)
  let in_s x = x
  let in_ms x = Nx.mul_s x (1.0 /. 1e-3)
  let in_us x = Nx.mul_s x (1.0 /. 1e-6)
  let in_min x = Nx.div_s x 60.0
  let in_hr x = Nx.div_s x 3600.0
  let in_day x = Nx.div_s x 86_400.0
  let in_yr x = Nx.div_s x julian_year_s
  let in_myr x = Nx.div_s x (julian_year_s *. 1e6)
  let in_gyr x = Nx.div_s x (julian_year_s *. 1e9)
end

module Angle = struct
  let deg_rad = Float.pi /. 180.0
  let of_tensor x = x
  let to_tensor x = x
  let rad x = Nx.scalar f64 x
  let deg x = Nx.scalar f64 (x *. deg_rad)
  let arcmin x = Nx.scalar f64 (x *. deg_rad /. 60.0)
  let arcsec x = Nx.scalar f64 (x *. deg_rad /. 3600.0)
  let mas x = Nx.scalar f64 (x *. deg_rad /. 3_600_000.0)
  let hour_angle x = Nx.scalar f64 (x *. Float.pi /. 12.0)
  let of_rad x = x
  let of_deg x = Nx.mul_s x deg_rad
  let of_arcmin x = Nx.mul_s x (deg_rad /. 60.0)
  let of_arcsec x = Nx.mul_s x (deg_rad /. 3600.0)
  let of_mas x = Nx.mul_s x (deg_rad /. 3_600_000.0)
  let of_hour_angle x = Nx.mul_s x (Float.pi /. 12.0)
  let in_rad x = x
  let in_deg x = Nx.div_s x deg_rad
  let in_arcmin x = Nx.mul_s (Nx.div_s x deg_rad) 60.0
  let in_arcsec x = Nx.mul_s (Nx.div_s x deg_rad) 3600.0
  let in_mas x = Nx.mul_s (Nx.div_s x deg_rad) 3_600_000.0
  let in_hour_angle x = Nx.mul_s x (12.0 /. Float.pi)
  let sin x = Nx.sin x
  let cos x = Nx.cos x
  let tan x = Nx.tan x
  let asin x = Nx.asin x
  let acos x = Nx.acos x
  let atan2 ~y ~x = Nx.atan2 y x

  let wrap_360 x =
    let d = in_deg x in
    let d = Nx.sub d (Nx.mul_s (Nx.floor (Nx.div_s d 360.0)) 360.0) in
    of_deg d

  let wrap_180 x =
    let d = Nx.add_s (in_deg x) 180.0 in
    let d = Nx.sub d (Nx.mul_s (Nx.floor (Nx.div_s d 360.0)) 360.0) in
    of_deg (Nx.sub_s d 180.0)
end

module Velocity = struct
  let of_tensor x = x
  let to_tensor x = x
  let m_s x = Nx.scalar f64 x
  let km_s x = Nx.scalar f64 (x *. 1e3)
  let km_hr x = Nx.scalar f64 (x *. (1e3 /. 3600.0))
  let of_m_s x = x
  let of_km_s x = Nx.mul_s x 1e3
  let of_km_hr x = Nx.mul_s x (1e3 /. 3600.0)
  let in_m_s x = x
  let in_km_s x = Nx.div_s x 1e3
  let in_km_hr x = Nx.div_s x (1e3 /. 3600.0)
end

module Power = struct
  let of_tensor x = x
  let to_tensor x = x
  let w x = Nx.scalar f64 x
  let kw x = Nx.scalar f64 (x *. 1e3)
  let solar_luminosity x = Nx.scalar f64 (x *. solar_luminosity_w)
  let erg_s x = Nx.scalar f64 (x *. 1e-7)
  let of_w x = x
  let of_kw x = Nx.mul_s x 1e3
  let of_solar_luminosity x = Nx.mul_s x solar_luminosity_w
  let of_erg_s x = Nx.mul_s x 1e-7
  let in_w x = x
  let in_kw x = Nx.div_s x 1e3
  let in_solar_luminosity x = Nx.div_s x solar_luminosity_w
  let in_erg_s x = Nx.mul_s x (1.0 /. 1e-7)
end

module Temperature = struct
  let of_tensor x = x
  let to_tensor x = x
  let kelvin x = Nx.scalar f64 x
  let of_kelvin x = x
  let in_kelvin x = x
end

module Energy = struct
  let of_tensor x = x
  let to_tensor x = x
  let j x = Nx.scalar f64 x
  let erg x = Nx.scalar f64 (x *. 1e-7)
  let ev x = Nx.scalar f64 (x *. ev_j)
  let kev x = Nx.scalar f64 (x *. ev_j *. 1e3)
  let mev x = Nx.scalar f64 (x *. ev_j *. 1e6)
  let of_j x = x
  let of_erg x = Nx.mul_s x 1e-7
  let of_ev x = Nx.mul_s x ev_j
  let of_kev x = Nx.mul_s x (ev_j *. 1e3)
  let of_mev x = Nx.mul_s x (ev_j *. 1e6)
  let in_j x = x
  let in_erg x = Nx.mul_s x (1.0 /. 1e-7)
  let in_ev x = Nx.div_s x ev_j
  let in_kev x = Nx.div_s x (ev_j *. 1e3)
  let in_mev x = Nx.div_s x (ev_j *. 1e6)
end

module Frequency = struct
  let of_tensor x = x
  let to_tensor x = x
  let hz x = Nx.scalar f64 x
  let khz x = Nx.scalar f64 (x *. 1e3)
  let mhz x = Nx.scalar f64 (x *. 1e6)
  let ghz x = Nx.scalar f64 (x *. 1e9)
  let of_hz x = x
  let of_khz x = Nx.mul_s x 1e3
  let of_mhz x = Nx.mul_s x 1e6
  let of_ghz x = Nx.mul_s x 1e9
  let in_hz x = x
  let in_khz x = Nx.div_s x 1e3
  let in_mhz x = Nx.div_s x 1e6
  let in_ghz x = Nx.div_s x 1e9
end

module Dimensionless = struct
  let of_tensor x = x
  let to_tensor x = x
  let v x = Nx.scalar f64 x
  let to_float x = Nx.item [] x
end
