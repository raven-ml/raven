(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module S2 = Jord.Sentinel2

let fail = Item.fail

(* Rules *)

let bands = S2.[ B04; B02; B03; B08; B8A; B05; B06; B07; B11; B12 ]

let invalid =
  S2.Scl.
    [
      No_data;
      Saturated_or_defective;
      Cast_shadows;
      Cloud_shadows;
      Cloud_medium_probability;
      Cloud_high_probability;
    ]

let min_coverage = 0.01

(* A tile holds reflectance times 10⁴, the products' quantification value. *)
let quantification = 10000.

let key it =
  let tile = Item.string "s2:mgrs_tile" it in
  match S2.key ~tile (Item.string "s2:datastrip_id" it) with
  | Ok k -> k
  | Error e -> fail it "%s" e

(* Of two processings of one acquisition, the higher baseline, then the greater
   id. *)
let rank it =
  match S2.baseline (Item.string "s2:processing_baseline" it) with
  | Ok pb -> (pb, it.Jord_stac.id)
  | Error e -> fail it "%s" e

let distinct items =
  let best = Hashtbl.create 64 in
  List.iter
    (fun it ->
      let k = key it and r = rank it in
      match Hashtbl.find_opt best k with
      | Some (r', _) when compare r' r >= 0 -> ()
      | _ -> Hashtbl.replace best k (r, it))
    items;
  List.sort Item.compare
    (List.map snd (List.of_seq (Hashtbl.to_seq_values best)))

let first_valid ~roi scls =
  let none = Nx.full Nx.int32 (Nx.shape roi) (-1l) in
  let choose (k, choice) scl =
    let valid = Nx.logical_not (S2.Scl.mem invalid scl) in
    let free = Nx.logical_and roi (Nx.equal_s choice (-1l)) in
    let take = Nx.logical_and free valid in
    (k + 1, Nx.where take (Nx.scalar_like choice (Int32.of_int k)) choice)
  in
  snd (List.fold_left choose (0, none) scls)

let count m = Int64.to_int (Nx.item [] (Nx.count m))

let covered ~roi choice =
  let cells = count roi in
  cells > 0
  && 100.
     *. Float.of_int (count (Nx.greater_equal_s choice 0l))
     /. Float.of_int cells
     >= min_coverage

(* Frames *)

type frame = { bands : Nx.host Nx.int16_t; mask : Nx.host Nx.uint8_t }

let metadata s it =
  let file = List.assoc "product-metadata" it.Jord_stac.assets in
  match Jord_safe.l2a (Stac.contents s file) with
  | Error e -> fail it "product-metadata: %s" e
  | Ok m when m.quantification <> quantification ->
      fail it "a quantification value of %g" m.quantification
  | Ok m -> m

(* The scene classes at the cells, 0 where the product has none. *)
let scl s ~centres it =
  let _, w = Item.window s ~centres Nearest it "SCL" in
  let x, _ =
    Jord.Warp.nearest ~row:w.row ~col:w.col w.grid w.points
      (Nx.unpack Nx.uint8 w.samples)
  in
  Nx.squeeze ~axes:[ 2 ] x

let int16_range = (-32768l, 32767l)

(* Band [b] at the cells, harmonised, and where it has data. *)
let band s ~centres it (m : S2.metadata) b =
  let name = S2.name b in
  let decoder =
    match S2.reflectance m b with Ok d -> d | Error e -> fail it "%s" e
  in
  let _, w = Item.window s ~centres Bilinear it name in
  let dn = Nx.unpack Nx.uint16 w.samples in
  let _, valid = Jord.Band.decode decoder Nx.float32 dn in
  let x, ok =
    Jord.Warp.bilinear ~row:w.row ~col:w.col ~valid w.grid w.points dn
  in
  let x = Nx.squeeze ~axes:[ 2 ] x and ok = Nx.squeeze ~axes:[ 2 ] ok in
  let offset =
    match S2.offset m b with Ok o -> o | Error e -> fail it "%s" e
  in
  let v = Nx.add_s (Nx.cast Nx.int32 x) (Int32.of_int offset) in
  let v = Nx.where ok v (Nx.scalar_like v 0l) in
  let lo, hi = int16_range in
  if Nx.numel v > 0 && (Nx.item [] (Nx.min v) < lo || Nx.item [] (Nx.max v) > hi)
  then fail it "%s: a harmonised value outside int16" name;
  (Nx.cast Nx.int16 v, ok)

let mosaic choice products =
  let h = Nx.dim 0 choice and w = Nx.dim 1 choice in
  let add (k, values, mask) p =
    match p with
    | None -> (k + 1, values, mask)
    | Some (v, ok) ->
        let ok = Nx.logical_and ok (Nx.equal_s choice (Int32.of_int k)) in
        let values = Nx.where (Nx.reshape [| h; w; 1 |] ok) v values in
        (k + 1, values, Nx.logical_or mask ok)
  in
  let zeros = Nx.zeros Nx.int16 [| h; w; List.length bands |] in
  let _, values, mask =
    List.fold_left add (0, zeros, Nx.zeros Nx.bool [| h; w |]) products
  in
  { bands = values; mask = Nx.cast Nx.uint8 mask }

let day s ~roi ~centres items =
  let choice = first_valid ~roi (List.map (scl s ~centres) items) in
  if not (covered ~roi choice) then None
  else
    let product k it =
      if not (Nx.item [] (Nx.any (Nx.equal_s choice (Int32.of_int k)))) then
        None
      else
        let m = metadata s it in
        let vs, oks = List.split (List.map (band s ~centres it m) bands) in
        Some
          (Nx.stack ~axis:2 vs, List.fold_left Nx.logical_and (List.hd oks) oks)
    in
    Some (mosaic choice (List.mapi product items))

(* Days *)

let assets = "product-metadata" :: "SCL" :: List.map S2.name bands

let days items =
  let complete, partial = List.partition (Item.has assets) items in
  ( Item.group (fun it -> Jord.Time.date (Stac.time it)) (distinct complete),
    partial )
