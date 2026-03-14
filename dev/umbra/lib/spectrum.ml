(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let f64 = Nx.float64

(* Physical constants (SI) *)
let _h = 6.626_070_15e-34
let _c = 299_792_458.0
let _k_b = 1.380_649e-23
let _two_hc2 = 2.0 *. _h *. _c *. _c
let _hc_over_k = _h *. _c /. _k_b

(* Spectral kinds — phantom, no runtime representation *)
type flux_density
type radiance
type sampled

(* Wavelength stored internally in metres (SI base unit) *)
type 'a t = { wavelength : Nx.float64_t; values : Nx.float64_t }

let validate_increasing name wl =
  let n = Nx.numel wl in
  if n > 1 then
    for i = 1 to n - 1 do
      if Nx.item [ i ] wl <= Nx.item [ i - 1 ] wl then
        invalid_arg (name ^ ": wavelength must be strictly increasing")
    done

let create ~wavelength ~values =
  let wavelength = Unit.Length.to_tensor wavelength in
  if Nx.ndim wavelength <> 1 then
    invalid_arg "Spectrum.create: wavelength must be a 1-D tensor";
  let v_shape = Nx.shape values in
  let v_ndim = Array.length v_shape in
  if v_ndim = 0 then invalid_arg "Spectrum.create: values must be at least 1-D";
  if v_shape.(v_ndim - 1) <> Nx.numel wavelength then
    invalid_arg
      "Spectrum.create: last dimension of values must match wavelength length";
  validate_increasing "Spectrum.create" wavelength;
  { wavelength; values }

let wavelength t = Unit.Length.of_tensor t.wavelength
let values t = t.values
let as_flux_density t = { wavelength = t.wavelength; values = t.values }
let as_sampled t = { wavelength = t.wavelength; values = t.values }

let blackbody ~temperature ~wavelength =
  let wavelength = Unit.Length.to_tensor wavelength in
  let temp = Unit.Temperature.to_tensor temperature in
  let two_hc2 = Nx.scalar f64 _two_hc2 in
  let hc_k = Nx.scalar f64 _hc_over_k in
  let lam5 = Nx.pow_s wavelength 5.0 in
  let exponent = Nx.div hc_k (Nx.mul wavelength temp) in
  let values =
    Nx.div (Nx.div two_hc2 lam5) (Nx.sub (Nx.exp exponent) (Nx.scalar f64 1.0))
  in
  { wavelength; values }

let power_law ~amplitude ~index ~pivot ~wavelength =
  let wavelength = Unit.Length.to_tensor wavelength in
  let pivot = Unit.Length.to_tensor pivot in
  let ratio = Nx.div wavelength pivot in
  let values = Nx.mul amplitude (Nx.pow ratio index) in
  { wavelength; values }

let redshift ~z t =
  let one_plus_z = Nx.add_s z 1.0 in
  {
    wavelength = Nx.mul t.wavelength one_plus_z;
    values = Nx.div t.values one_plus_z;
  }

let scale factor t = { t with values = Nx.mul factor t.values }

let mul a b =
  if Nx.numel a.wavelength <> Nx.numel b.wavelength then
    invalid_arg "Spectrum.mul: spectra must have the same wavelength grid";
  let max_diff =
    Nx.item [] (Nx.max (Nx.abs (Nx.sub a.wavelength b.wavelength)))
  in
  if max_diff > 0.0 then
    invalid_arg "Spectrum.mul: spectra must have the same wavelength grid";
  { wavelength = a.wavelength; values = Nx.mul a.values b.values }

let div a b =
  if Nx.numel a.wavelength <> Nx.numel b.wavelength then
    invalid_arg "Spectrum.div: spectra must have the same wavelength grid";
  let max_diff =
    Nx.item [] (Nx.max (Nx.abs (Nx.sub a.wavelength b.wavelength)))
  in
  if max_diff > 0.0 then
    invalid_arg "Spectrum.div: spectra must have the same wavelength grid";
  { wavelength = a.wavelength; values = Nx.div a.values b.values }

let add a b =
  if Nx.numel a.wavelength <> Nx.numel b.wavelength then
    invalid_arg "Spectrum.add: spectra must have the same wavelength grid";
  let max_diff =
    Nx.item [] (Nx.max (Nx.abs (Nx.sub a.wavelength b.wavelength)))
  in
  if max_diff > 0.0 then
    invalid_arg "Spectrum.add: spectra must have the same wavelength grid";
  { wavelength = a.wavelength; values = Nx.add a.values b.values }

let resample ~wavelength t =
  let new_wave = Unit.Length.to_tensor wavelength in
  if Nx.ndim new_wave <> 1 then
    invalid_arg "Spectrum.resample: wavelength must be a 1-D tensor";
  validate_increasing "Spectrum.resample" new_wave;
  let old_wave = t.wavelength in
  let old_values = t.values in
  let n_old = Nx.numel old_wave and n_new = Nx.numel new_wave in
  (* Find lower bracket index for each target wavelength (non-differentiable) *)
  let lo_arr =
    Array.init n_new (fun j ->
        let x = Nx.item [ j ] new_wave in
        let lo = ref 0 and hi = ref (n_old - 1) in
        while !hi - !lo > 1 do
          let mid = (!lo + !hi) / 2 in
          if Nx.item [ mid ] old_wave <= x then lo := mid else hi := mid
        done;
        !lo)
  in
  let hi_arr =
    Array.init n_new (fun j -> Int32.of_int (min (lo_arr.(j) + 1) (n_old - 1)))
  in
  let lo_arr = Array.map Int32.of_int lo_arr in
  let lo_t = Nx.create Nx.int32 [| n_new |] lo_arr in
  let hi_t = Nx.create Nx.int32 [| n_new |] hi_arr in
  (* Gather source wavelengths and values at bracket endpoints. Nx.take uses
     B.gather, which Rune differentiates through. *)
  let x0 = Nx.take lo_t old_wave in
  let x1 = Nx.take hi_t old_wave in
  let y0 = Nx.take ~axis:(-1) lo_t old_values in
  let y1 = Nx.take ~axis:(-1) hi_t old_values in
  (* Linear interpolation — differentiable through Rune *)
  let dx = Nx.clamp ~min:1e-30 (Nx.sub x1 x0) in
  let frac = Nx.div (Nx.sub new_wave x0) dx in
  let values = Nx.add y0 (Nx.mul frac (Nx.sub y1 y0)) in
  { wavelength = new_wave; values }

let gaussian ~amplitude ~center ~stddev ~wavelength =
  let wavelength = Unit.Length.to_tensor wavelength in
  let center = Unit.Length.to_tensor center in
  let stddev = Unit.Length.to_tensor stddev in
  let x = Nx.sub wavelength center in
  let z = Nx.div x stddev in
  let values = Nx.mul amplitude (Nx.exp (Nx.mul_s (Nx.mul z z) (-0.5))) in
  { wavelength; values }

let lorentzian ~amplitude ~center ~fwhm ~wavelength =
  let wavelength = Unit.Length.to_tensor wavelength in
  let center = Unit.Length.to_tensor center in
  let half_gamma = Nx.div_s (Unit.Length.to_tensor fwhm) 2.0 in
  let x = Nx.sub wavelength center in
  let hg2 = Nx.mul half_gamma half_gamma in
  let values = Nx.mul amplitude (Nx.div hg2 (Nx.add (Nx.mul x x) hg2)) in
  { wavelength; values }

let voigt ~amplitude ~center ~sigma ~gamma ~wavelength =
  let wavelength = Unit.Length.to_tensor wavelength in
  let center = Unit.Length.to_tensor center in
  let sigma = Unit.Length.to_tensor sigma in
  let gamma = Unit.Length.to_tensor gamma in
  (* Pseudo-Voigt mixing via Thompson, Cox & Hastings (1987). *)
  let sqrt_2ln2 = Float.sqrt (2.0 *. Float.log 2.0) in
  let fg = Nx.mul_s sigma (2.0 *. sqrt_2ln2) in
  let fl = Nx.mul_s gamma 2.0 in
  let fg2 = Nx.mul fg fg in
  let fg3 = Nx.mul fg2 fg in
  let fg4 = Nx.mul fg3 fg in
  let fg5 = Nx.mul fg4 fg in
  let fl2 = Nx.mul fl fl in
  let fl3 = Nx.mul fl2 fl in
  let fl4 = Nx.mul fl3 fl in
  let fl5 = Nx.mul fl4 fl in
  let f =
    Nx.pow_s
      (Nx.add fg5
         (Nx.add
            (Nx.mul_s (Nx.mul fg4 fl) 2.69269)
            (Nx.add
               (Nx.mul_s (Nx.mul fg3 fl2) 2.42843)
               (Nx.add
                  (Nx.mul_s (Nx.mul fg2 fl3) 4.47163)
                  (Nx.add (Nx.mul_s (Nx.mul fg fl4) 0.07842) fl5)))))
      0.2
  in
  let ratio = Nx.div fl f in
  let ratio2 = Nx.mul ratio ratio in
  let ratio3 = Nx.mul ratio2 ratio in
  let eta =
    Nx.add (Nx.mul_s ratio 1.36603)
      (Nx.add (Nx.mul_s ratio2 (-0.47719)) (Nx.mul_s ratio3 0.11116))
  in
  (* Gaussian component (unit height at center) *)
  let x = Nx.sub wavelength center in
  let sig_eff = Nx.div_s f (2.0 *. sqrt_2ln2) in
  let z_g = Nx.div x sig_eff in
  let gauss = Nx.exp (Nx.mul_s (Nx.mul z_g z_g) (-0.5)) in
  (* Lorentzian component (unit height at center) *)
  let hf = Nx.div_s f 2.0 in
  let hf2 = Nx.mul hf hf in
  let lorentz = Nx.div hf2 (Nx.add (Nx.mul x x) hf2) in
  let values =
    Nx.mul amplitude
      (Nx.add (Nx.mul eta lorentz)
         (Nx.mul (Nx.sub (Nx.scalar f64 1.0) eta) gauss))
  in
  { wavelength; values }
