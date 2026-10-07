(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A JWST i2d image as an observation, composed from ymir.fits reads: the SCI
   HDU's data in BUNIT, the ERR HDU's in the same unit as variance, valid where
   both are finite, with JWST's PIXAR_SR as each pixel's area. *)

open Ymir
open Ymir_fits

let ( let* ) = Result.bind

let observation (type e) ?ver ?window (dtype : (float, e) Nx.dtype) hdus =
  let* sci = Fits.get ?ver "SCI" hdus in
  let* err = Fits.get ?ver "ERR" hdus in
  let h = Fits.header sci in
  let keywords =
    {
      Wcs.float = (fun k -> Fits.Header.find Fits.Value.float k h);
      int = (fun k -> Fits.Header.find Fits.Value.int k h);
      text = (fun k -> Fits.Header.find Fits.Value.string k h);
    }
  in
  let* wcs = Wcs.read Frame.icrs keywords in
  let* bunit = Fits.Header.get Fits.Value.string "BUNIT" h in
  let* unit = Fits.Unit.parse bunit in
  let* pixar = Fits.Header.get Fits.Value.float "PIXAR_SR" h in
  let* data = Fits.Image.values ?window dtype sci in
  let* sigma = Fits.Image.values ?window dtype err in
  let* image = Fits.Image.of_hdu sci in
  let grid = Grid.pixels ~shape:(Fits.Image.shape image) dtype wcs in
  let grid =
    match window with
    | None -> grid
    | Some w ->
        let start = Array.map (fun (a, _) -> Int64.of_int a) w in
        Grid.window
          ~start:(Nx.create Nx.int64 [| 2 |] start)
          ~shape:(Array.map (fun (a, b) -> b - a) w)
          grid
  in
  Ok
    (Observation.v
       ~variance:(Quantity.v (Unit.( ** ) unit 2) (Nx.square sigma))
       ~valid:Nx.(cast bit (logical_and (isfinite data) (isfinite sigma)))
       ~area:(Quantity.v Unit.steradian (Nx.scalar dtype pixar))
       grid (Quantity.v unit data))
