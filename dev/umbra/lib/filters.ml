(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let f64 = Nx.float64
let angstrom_to_m = 1e-10

let make wave_a thru_a =
  let n = Array.length wave_a in
  let w = Nx.create f64 [| n |] wave_a in
  let w = Nx.mul_s w angstrom_to_m in
  let t = Nx.create f64 [| n |] thru_a in
  Photometry.bandpass ~wavelength:(Unit.Length.of_tensor w) ~throughput:t

(* SDSS *)

let sdss_u = make Filter_data.sdss_u_wave Filter_data.sdss_u_thru
let sdss_g = make Filter_data.sdss_g_wave Filter_data.sdss_g_thru
let sdss_r = make Filter_data.sdss_r_wave Filter_data.sdss_r_thru
let sdss_i = make Filter_data.sdss_i_wave Filter_data.sdss_i_thru
let sdss_z = make Filter_data.sdss_z_wave Filter_data.sdss_z_thru

(* Johnson-Cousins *)

let johnson_u = make Filter_data.johnson_u_wave Filter_data.johnson_u_thru
let johnson_b = make Filter_data.johnson_b_wave Filter_data.johnson_b_thru
let johnson_v = make Filter_data.johnson_v_wave Filter_data.johnson_v_thru
let cousins_r = make Filter_data.cousins_r_wave Filter_data.cousins_r_thru
let cousins_i = make Filter_data.cousins_i_wave Filter_data.cousins_i_thru

(* 2MASS *)

let twomass_j = make Filter_data.twomass_j_wave Filter_data.twomass_j_thru
let twomass_h = make Filter_data.twomass_h_wave Filter_data.twomass_h_thru
let twomass_ks = make Filter_data.twomass_ks_wave Filter_data.twomass_ks_thru

(* Gaia DR3 *)

let gaia_g = make Filter_data.gaia_g_wave Filter_data.gaia_g_thru
let gaia_bp = make Filter_data.gaia_bp_wave Filter_data.gaia_bp_thru
let gaia_rp = make Filter_data.gaia_rp_wave Filter_data.gaia_rp_thru

(* Rubin/LSST *)

let rubin_u = make Filter_data.rubin_u_wave Filter_data.rubin_u_thru
let rubin_g = make Filter_data.rubin_g_wave Filter_data.rubin_g_thru
let rubin_r = make Filter_data.rubin_r_wave Filter_data.rubin_r_thru
let rubin_i = make Filter_data.rubin_i_wave Filter_data.rubin_i_thru
let rubin_z = make Filter_data.rubin_z_wave Filter_data.rubin_z_thru
let rubin_y = make Filter_data.rubin_y_wave Filter_data.rubin_y_thru

(* Euclid *)

let euclid_vis = make Filter_data.euclid_vis_wave Filter_data.euclid_vis_thru
let euclid_y = make Filter_data.euclid_y_wave Filter_data.euclid_y_thru
let euclid_j = make Filter_data.euclid_j_wave Filter_data.euclid_j_thru
let euclid_h = make Filter_data.euclid_h_wave Filter_data.euclid_h_thru
