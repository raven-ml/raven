(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The code *)

(* On the host with the C library's [log10], which nx's float64 [log10] does not
   match (a gap of nx's: float64 log10), and as numpy computes it, the product
   rounded before the sum: an arm64 build would fuse [a *. b +. c] into one
   multiply-add. *)
let code g =
  let n = Nx.numel g in
  let out = Flat.create Bigarray.int16_signed n 0 in
  Flat.read g (fun v ->
      for i = 0 to n - 1 do
        let x = v.{i} in
        if Float.is_finite x && x > 0. then
          let db = Sys.opaque_identity (20. *. Float.log10 x) in
          let scaled = (db +. 50.) *. 200. in
          out.{i} <- Float.to_int (Float.min (Float.max scaled 0.) 32767.)
      done);
  Flat.wrap Nx.int16 (Nx.shape g) out

let min_coverage = 0.01
let count m = Int64.to_int (Nx.item [] (Nx.count m))

let covered ~roi g =
  let cells = count roi in
  let observed =
    Nx.logical_and roi (Nx.logical_and (Nx.isfinite g) (Nx.greater_s g 0.))
  in
  cells > 0
  && Float.of_int (100 * count observed) /. Float.of_int cells >= min_coverage

let merge = function
  | [] -> invalid_arg "Tessera.Sentinel1.merge: no code"
  | c :: cs ->
      List.fold_left
        (fun acc c' ->
          if Nx.shape c' <> Nx.shape c then
            invalid_arg "Tessera.Sentinel1.merge: codes of different shapes";
          Nx.where (Nx.equal_s acc 0) c' acc)
        c cs

(* Frames *)

let orbit (it : Jord_stac.item) =
  match it.orbit_state with
  | Some ((Ascending | Descending) as o) -> o
  | Some (Geostationary | Crossing) | None ->
      failwith (it.id ^ ": no polar orbit state")

let no_band =
  {
    Jord.Band.name = None;
    common_name = None;
    center_wavelength = None;
    full_width_half_max = None;
    unit = None;
    kind = Measurement;
    scale = 1.;
    offset = 0.;
    nodata = None;
    saturated = None;
  }

(* γ⁰ of polarisation [key] at the cells, NaN where it has none. *)
let gamma s ~centres it key =
  let asset, w = Item.window s ~centres Nearest it key in
  let band = match asset.bands with b :: _ -> b.band | [] -> no_band in
  let x = Nx.unpack Nx.float32 w.samples in
  let _, valid = Jord.Band.decode band Nx.float32 x in
  let x, ok =
    Jord.Warp.nearest ~row:w.row ~col:w.col ~valid w.grid w.points x
  in
  let g = Nx.cast Nx.float64 (Nx.squeeze ~axes:[ 2 ] x) in
  Nx.where (Nx.squeeze ~axes:[ 2 ] ok) g (Nx.scalar_like g Float.nan)

let day s ~roi ~centres items =
  let kept it =
    let vv = gamma s ~centres it "vv" and vh = gamma s ~centres it "vh" in
    let code_of g =
      let c = code g in
      if covered ~roi g then Nx.where roi c (Nx.scalar_like c 0)
      else Nx.zeros_like c
    in
    if covered ~roi vv || covered ~roi vh then
      Some (Nx.stack ~axis:2 [ code_of vv; code_of vh ])
    else None
  in
  match List.filter_map kept items with [] -> None | cs -> Some (merge cs)

let days items =
  let complete, partial = List.partition (Item.has [ "vv"; "vh" ]) items in
  let key it = (Jord.Time.date (Stac.time it), orbit it) in
  (Item.group key (List.sort Item.compare complete), partial)
