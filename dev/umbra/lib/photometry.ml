(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let f64 = Nx.float64

(* Speed of light for f_lambda to f_nu conversion *)
let _c = 299_792_458.0

(* AB magnitude zero-point: 3631 Jy = 3631e-26 W/m²/Hz *)
let _ab_zp = 3631.0e-26

(* Wavelength stored internally in metres (SI base unit) *)
type bandpass = { wavelength : Nx.float64_t; throughput : Nx.float64_t }
type detector = Energy | Photon

let bandpass ~wavelength ~throughput =
  let wavelength = Unit.Length.to_tensor wavelength in
  if Nx.ndim wavelength <> 1 then
    invalid_arg "Photometry.bandpass: wavelength must be a 1-D tensor";
  if Nx.ndim throughput <> 1 then
    invalid_arg "Photometry.bandpass: throughput must be a 1-D tensor";
  if Nx.numel wavelength <> Nx.numel throughput then
    invalid_arg
      "Photometry.bandpass: wavelength and throughput must have the same length";
  { wavelength; throughput }

let tophat ~lo ~hi ~n =
  let lo_m = Nx.item [] (Unit.Length.to_tensor lo) in
  let hi_m = Nx.item [] (Unit.Length.to_tensor hi) in
  let wavelength = Nx.linspace f64 lo_m hi_m n in
  let throughput = Nx.ones f64 [| n |] in
  { wavelength; throughput }

let wavelength bp = Unit.Length.of_tensor bp.wavelength
let throughput bp = bp.throughput

(* Differentiable trapezoidal integration along the last axis of y. x is always
   1-D (the wavelength grid). When y has leading batch dimensions the result
   preserves them. All Nx ops — fully differentiable through Rune. *)
let trapz y x =
  let m = Nx.numel x in
  let x0 = Nx.slice [ R (0, m - 1) ] x in
  let x1 = Nx.slice [ R (1, m) ] x in
  let dx = Nx.sub x1 x0 in
  let y_shape = Nx.shape y in
  let ndim = Array.length y_shape in
  if ndim <= 1 then begin
    let y0 = Nx.slice [ R (0, m - 1) ] y in
    let y1 = Nx.slice [ R (1, m) ] y in
    let y_avg = Nx.div_s (Nx.add y0 y1) 2.0 in
    Nx.sum (Nx.mul y_avg dx)
  end
  else begin
    let y2d = Nx.reshape [| -1; m |] y in
    let y0 = Nx.slice [ A; R (0, m - 1) ] y2d in
    let y1 = Nx.slice [ A; R (1, m) ] y2d in
    let y_avg = Nx.div_s (Nx.add y0 y1) 2.0 in
    let result = Nx.sum ~axes:[ 1 ] (Nx.mul y_avg dx) in
    let batch_shape = Array.sub y_shape 0 (ndim - 1) in
    Nx.reshape batch_shape result
  end

let pivot_wavelength bp =
  let lam = bp.wavelength in
  let t = bp.throughput in
  (* lambda_p = sqrt(integral T lambda d lambda / integral T/lambda d lambda) *)
  let num = trapz (Nx.mul t lam) lam in
  let den = trapz (Nx.div t lam) lam in
  Unit.Length.of_tensor (Nx.sqrt (Nx.div num den))

(* Detector weight: 1 for energy-counting, lambda for photon-counting *)
let detector_weight detector lam throughput =
  match detector with Energy -> throughput | Photon -> Nx.mul throughput lam

(* ST magnitude zero-point: -2.5 log10(f_lambda / 3.63e-9 erg/s/cm²/Å) In SI:
   3.63e-9 erg/s/cm²/Å = 3.63e-9 * 1e-7 * 1e4 * 1e10 W/m²/m = 3.63e-2 W/m²/m *)
let _st_zp = 3.63e-2

let align_spectrum bp spectrum =
  let lam_bp = bp.wavelength in
  let lam_sp = Unit.Length.to_tensor (Spectrum.wavelength spectrum) in
  let same =
    Nx.numel lam_bp = Nx.numel lam_sp
    && Nx.item [] (Nx.max (Nx.abs (Nx.sub lam_bp lam_sp))) = 0.0
  in
  if same then spectrum
  else Spectrum.resample ~wavelength:(Unit.Length.of_tensor lam_bp) spectrum

let flux_density ?(detector = Energy) bp spectrum =
  let spectrum = align_spectrum bp spectrum in
  let lam = bp.wavelength in
  let f = Spectrum.values spectrum in
  let w = detector_weight detector lam bp.throughput in
  Nx.div (trapz (Nx.mul f w) lam) (trapz w lam)

let ab_mag ?(detector = Energy) bp spectrum =
  let spectrum = align_spectrum bp spectrum in
  let lam = bp.wavelength in
  let f_lambda = Spectrum.values spectrum in
  let f_nu = Nx.div (Nx.mul f_lambda (Nx.square lam)) (Nx.scalar f64 _c) in
  let w = detector_weight detector lam bp.throughput in
  let mean_fnu = Nx.div (trapz (Nx.mul f_nu w) lam) (trapz w lam) in
  Nx.mul_s
    (Nx.log (Nx.div mean_fnu (Nx.scalar f64 _ab_zp)))
    (-2.5 /. Float.log 10.0)

let st_mag ?(detector = Energy) bp spectrum =
  let spectrum = align_spectrum bp spectrum in
  let lam = bp.wavelength in
  let f_lambda = Spectrum.values spectrum in
  let w = detector_weight detector lam bp.throughput in
  let mean_flam = Nx.div (trapz (Nx.mul f_lambda w) lam) (trapz w lam) in
  Nx.mul_s
    (Nx.log (Nx.div mean_flam (Nx.scalar f64 _st_zp)))
    (-2.5 /. Float.log 10.0)

let _vega_spectrum =
  let n = Array.length Vega_data.wave in
  let w = Nx.create f64 [| n |] Vega_data.wave in
  let w = Nx.mul_s w 1e-10 in
  let f = Nx.create f64 [| n |] Vega_data.flux in
  Spectrum.create ~wavelength:(Unit.Length.of_tensor w) ~values:f
  |> Spectrum.as_flux_density

let vega_mag ?(detector = Energy) bp spectrum =
  let f_src = flux_density ~detector bp spectrum in
  let f_vega = flux_density ~detector bp _vega_spectrum in
  Nx.mul_s (Nx.log (Nx.div f_src f_vega)) (-2.5 /. Float.log 10.0)

let color ?detector bp1 bp2 spectrum =
  Nx.sub (ab_mag ?detector bp1 spectrum) (ab_mag ?detector bp2 spectrum)

let effective_wavelength ?(detector = Energy) bp spectrum =
  let spectrum = align_spectrum bp spectrum in
  let lam = bp.wavelength in
  let f = Spectrum.values spectrum in
  let w = detector_weight detector lam bp.throughput in
  let fw = Nx.mul f w in
  let num = trapz (Nx.mul fw (Nx.square lam)) lam in
  let den = trapz (Nx.mul fw lam) lam in
  Unit.Length.of_tensor (Nx.div num den)
