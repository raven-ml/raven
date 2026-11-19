(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type place = { grid : Jord.Grid.t; roi : Nx.host Nx.uint8_t; source : string }

type t = {
  place : place option;
  bands : Nx.host Nx.int16_t;
  masks : Nx.host Nx.uint8_t;
  doys : Nx.host Nx.uint16_t;
  ascending : Nx.host Nx.int16_t;
  ascending_doys : Nx.host Nx.int16_t;
  descending : Nx.host Nx.int16_t;
  descending_doys : Nx.host Nx.int16_t;
}

(* A file's fault, raised by [fail] naming the function [fn]: [failwith] from
   a load, which reads what it is given, and [invalid_arg] from another
   function, whose caller gave it. *)
let error fail fn file fmt =
  Format.kasprintf
    (fun m -> fail ("Tessera.Tile." ^ fn ^ ": " ^ file ^ ": " ^ m))
    fmt

let s2_bands = 10
let s1_bands = 2
let image_description = 270

let shape t = (Nx.dim 1 t.bands, Nx.dim 2 t.bands)

let day_of_year (y, m, d) =
  match Jord.Time.of_date (y, m, d) with
  | t -> Jord.Time.day_of_year t
  | exception Invalid_argument _ ->
      invalid_arg (Printf.sprintf "Tessera.Tile.day_of_year: %d-%d-%d" y m d)

(* Files *)

let bands_file = "bands.npy"
let masks_file = "masks.npy"
let doys_file = "doys.npy"
let roi_file = "roi.tiff"
let orbit_file o = "sar_" ^ o ^ ".npy"
let orbit_doys_file o = "sar_" ^ o ^ "_doy.npy"

let expect fail fn file x want =
  if Nx.shape x <> want then
    error fail fn file "shape %a, %a expected" Nx.pp_shape (Nx.shape x)
      Nx.pp_shape want

(* The frames of [x], of shape [k] followed by [cell]. *)
let frames fail fn file x cell =
  let s = Nx.shape x in
  let k = if Array.length s > 0 then s.(0) else 0 in
  expect fail fn file x (Array.append [| k |] cell);
  k

let check fail fn t =
  if Nx.ndim t.bands <> 4 then
    error fail fn bands_file "shape %a, [t; h; w; %d] expected" Nx.pp_shape
      (Nx.shape t.bands) s2_bands;
  let h, w = shape t in
  let n = frames fail fn bands_file t.bands [| h; w; s2_bands |] in
  expect fail fn masks_file t.masks [| n; h; w |];
  expect fail fn doys_file t.doys [| n |];
  let orbit o x days =
    let k = frames fail fn (orbit_file o) x [| h; w; s1_bands |] in
    expect fail fn (orbit_doys_file o) days [| k |]
  in
  orbit "ascending" t.ascending t.ascending_doys;
  orbit "descending" t.descending t.descending_doys;
  match t.place with
  | Some p when p.grid.rows <> h || p.grid.cols <> w ->
      error fail fn roi_file "a grid of %d × %d, and the tile is %d × %d"
        p.grid.rows p.grid.cols h w
  | Some p -> expect fail fn roi_file p.roi [| h; w |]
  | None -> ()

let empty ?place (h, w) =
  if h < 0 || w < 0 then invalid_arg "Tessera.Tile.empty: negative cells";
  let s1 = Nx.zeros Nx.int16 [| 0; h; w; s1_bands |] in
  let days = Nx.zeros Nx.int16 [| 0 |] in
  let t =
    {
      place;
      bands = Nx.zeros Nx.int16 [| 0; h; w; s2_bands |];
      masks = Nx.zeros Nx.uint8 [| 0; h; w |];
      doys = Nx.zeros Nx.uint16 [| 0 |];
      ascending = s1;
      ascending_doys = days;
      descending = s1;
      descending_doys = days;
    }
  in
  check invalid_arg "empty" t;
  t

let shard ?place (h, w) date ~s2 ~ascending ~descending =
  let t = empty ?place (h, w) in
  let doy = day_of_year date in
  let frame name cell x =
    if Nx.shape x <> Array.append [| h; w |] cell then
      invalid_arg
        (Format.asprintf "Tessera.Tile.shard: %s of shape %a, %a expected" name
           Nx.pp_shape (Nx.shape x) Nx.pp_shape
           (Array.append [| h; w |] cell));
    Nx.reshape (Array.append [| 1; h; w |] cell) x
  in
  let t =
    match s2 with
    | None -> t
    | Some (bands, mask) ->
        {
          t with
          bands = frame "the bands" [| s2_bands |] bands;
          masks = frame "the mask" [||] mask;
          doys = Nx.create Nx.uint16 [| 1 |] [| doy |];
        }
  in
  let orbit name frames doys = function
    | None -> (frames, doys)
    | Some c ->
        (frame name [| s1_bands |] c, Nx.create Nx.int16 [| 1 |] [| doy |])
  in
  let ascending, ascending_doys =
    orbit "the ascending frame" t.ascending t.ascending_doys ascending
  in
  let descending, descending_doys =
    orbit "the descending frame" t.descending t.descending_doys descending
  in
  { t with ascending; ascending_doys; descending; descending_doys }

(* Places *)

let save_place file p =
  let tags =
    (image_description, Nx_io.Tiff.Ascii p.source)
    :: Jord_geotiff.tags ~nodata:0. p.grid
  in
  Nx_io.save_tiff ~tile:(256, 256) ~compression:Deflate ~tags file
    (Nx.reshape [| p.grid.rows; p.grid.cols; 1 |] p.roi)

let load_place file =
  let d = List.hd (Nx_io.load_tiff file) in
  let tags = Nx_io.Tiff.tags d in
  let grid =
    match Jord_geotiff.grid tags with
    | Ok g -> g
    | Error e -> error failwith "load" file "%s" e
  in
  let source =
    match List.assoc_opt image_description tags with
    | Some (Nx_io.Tiff.Ascii s) -> s
    | _ -> ""
  in
  let roi =
    match Nx.unpack Nx.uint8 (Nx_io.Tiff.samples d) with
    | x -> x
    | exception Invalid_argument m -> error failwith "load" file "%s" m
  in
  expect failwith "load" file roi [| grid.rows; grid.cols; 1 |];
  { grid; roi = Nx.reshape [| grid.rows; grid.cols |] roi; source }

let equal_place a b =
  Jord.Grid.equal a.grid b.grid
  && String.equal a.source b.source
  && Nx.item [] (Nx.all (Nx.equal a.roi b.roi))

(* Saving and loading *)

let rec mkdir_p d =
  if not (Sys.file_exists d) then begin
    mkdir_p (Filename.dirname d);
    Sys.mkdir d 0o755
  end

let save dir t =
  check invalid_arg "save" t;
  mkdir_p dir;
  let path f = Filename.concat dir f in
  Nx_io.save_npy (path bands_file) t.bands;
  Nx_io.save_npy (path masks_file) t.masks;
  Nx_io.save_npy (path doys_file) t.doys;
  Nx_io.save_npy (path (orbit_file "ascending")) t.ascending;
  Nx_io.save_npy (path (orbit_doys_file "ascending")) t.ascending_doys;
  Nx_io.save_npy (path (orbit_file "descending")) t.descending;
  Nx_io.save_npy (path (orbit_doys_file "descending")) t.descending_doys;
  Option.iter (save_place (path roi_file)) t.place

let unpack path dt x =
  match Nx.unpack dt x with
  | x -> x
  | exception Invalid_argument m -> error failwith "load" path "%s" m

let load_npy dir file dt =
  let path = Filename.concat dir file in
  if Sys.file_exists path then Some (unpack path dt (Nx_io.load_npy path))
  else None

(* TESSERA's stacker writes its bands as uint16, which hold the same bits as
   int16 below 2¹⁵, and whether a value lies above, read as negative: a
   saturated DN. *)
let load_bands dir =
  let path = Filename.concat dir bands_file in
  if not (Sys.file_exists path) then error failwith "load" path "missing";
  let x = Nx_io.load_npy path in
  match Nx.unpack Nx.int16 x with
  | x -> (x, false)
  | exception Invalid_argument _ ->
      let x = Nx.bitcast Nx.int16 (unpack path Nx.uint16 x) in
      (x, Nx.numel x > 0 && Nx.item [] (Nx.min x) < 0)

let load dir =
  let bands, saturated = load_bands dir in
  let path f = Filename.concat dir f in
  if Nx.ndim bands <> 4 then
    error failwith "load" (path bands_file) "shape %a, [t; h; w; %d] expected"
      Nx.pp_shape (Nx.shape bands) s2_bands;
  let t = Nx.dim 0 bands and h = Nx.dim 1 bands and w = Nx.dim 2 bands in
  let or_zeros dt shape = function Some x -> x | None -> Nx.zeros dt shape in
  let orbit o =
    match load_npy dir (orbit_file o) Nx.int16 with
    | None ->
        (Nx.zeros Nx.int16 [| 0; h; w; s1_bands |], Nx.zeros Nx.int16 [| 0 |])
    | Some x -> (
        match load_npy dir (orbit_doys_file o) Nx.int16 with
        | Some days -> (x, days)
        | None -> error failwith "load" (path (orbit_file o)) "no days")
  in
  let ascending, ascending_doys = orbit "ascending" in
  let descending, descending_doys = orbit "descending" in
  let masks =
    or_zeros Nx.uint8 [| t; h; w |] (load_npy dir masks_file Nx.uint8)
  in
  (* A saturated DN's observation is invalid. *)
  let bands, masks =
    if not saturated then (bands, masks)
    else begin
      expect failwith "load" (path masks_file) masks [| t; h; w |];
      let bad = Nx.any ~axes:[ 3 ] ~keepdims:true (Nx.less_s bands 0) in
      ( Nx.where bad (Nx.scalar_like bands 0) bands,
        Nx.where (Nx.squeeze ~axes:[ 3 ] bad) (Nx.scalar_like masks 0) masks )
    end
  in
  let tile =
    {
      place =
        (if Sys.file_exists (path roi_file) then
           Some (load_place (path roi_file))
         else None);
      bands;
      masks;
      doys = or_zeros Nx.uint16 [| t |] (load_npy dir doys_file Nx.uint16);
      ascending;
      ascending_doys;
      descending;
      descending_doys;
    }
  in
  check failwith "load" tile;
  tile

(* Joining *)

let concat ~place dir ts =
  let first =
    match ts with
    | t :: _ -> t
    | [] -> invalid_arg "Tessera.Tile.concat: no tile"
  in
  List.iter (check invalid_arg "concat") ts;
  let ((h, w) as cells) = shape first in
  let same t =
    shape t = cells
    &&
    match (t.place, place) with
    | None, None -> true
    | Some p, Some q -> equal_place p q
    | _ -> false
  in
  if not (List.for_all same ts) then
    invalid_arg "Tessera.Tile.concat: a tile of other cells or another place";
  mkdir_p dir;
  let join file dt cell field =
    Nx_io.save_npy_concat (Filename.concat dir file) dt cell
      (List.to_seq (List.map field ts))
  in
  join bands_file Nx.int16 [| h; w; s2_bands |] (fun t -> t.bands);
  join masks_file Nx.uint8 [| h; w |] (fun t -> t.masks);
  join doys_file Nx.uint16 [||] (fun t -> t.doys);
  let s1 = [| h; w; s1_bands |] in
  join (orbit_file "ascending") Nx.int16 s1 (fun t -> t.ascending);
  join (orbit_doys_file "ascending") Nx.int16 [||] (fun t -> t.ascending_doys);
  join (orbit_file "descending") Nx.int16 s1 (fun t -> t.descending);
  join (orbit_doys_file "descending") Nx.int16 [||] (fun t ->
      t.descending_doys);
  Option.iter (save_place (Filename.concat dir roi_file)) place

(* Places on the 10 m lattice *)

let lattice = 10.

(* A coordinate moved down or up to the lattice, in float64, as stackstac
   snaps a region's bounds (geom_utils.py, snapped_bounds). *)
let down v = Float.floor (v /. lattice) *. lattice
let up v = Float.ceil (v /. lattice) *. lattice

let on_lattice crs ~rows ~cols (x0, y0) =
  Jord.Grid.v crs ~rows ~cols ~origin:(x0, y0) ~row_step:(0., -.lattice)
    ~col_step:(lattice, 0.)

(* TESSERA's template of [g] (load_roi): cells of 10 m over [g]'s bounds,
   ⌈extent / 10⌉ of them, the bounds' extent computed as rasterio computes
   them; each cell takes the mask's cell that holds its centre, the one after
   a border, and 0 beyond the mask. The index of the cell holding the centre
   of the template's cell [k], along an axis of [n] cells of [step], is
   ⌊(k + ½) · 10 / step⌋: exact for a centre on a border, where the quotient
   is an integer. *)
let template (g : Jord.Grid.t) mask =
  let dx = fst g.col_step and dy = -.snd g.row_step in
  let x0, y0 = g.origin in
  let right = x0 +. (dx *. Float.of_int g.cols)
  and bottom = y0 -. (dy *. Float.of_int g.rows) in
  let cols = Float.to_int (Float.ceil ((right -. x0) /. lattice))
  and rows = Float.to_int (Float.ceil ((y0 -. bottom) /. lattice)) in
  let source n step k =
    let i = Float.floor ((Float.of_int k +. 0.5) *. lattice /. step) in
    if i < Float.of_int n then Float.to_int i else -1
  in
  let r = Array.init rows (source g.rows dy)
  and c = Array.init cols (source g.cols dx) in
  let m = Nx.to_array (Nx.greater mask (Nx.zeros_like mask)) in
  let roi =
    Array.init (rows * cols) (fun k ->
        let i = r.(k / cols) and j = c.(k mod cols) in
        if i >= 0 && j >= 0 && m.((i * g.cols) + j) then 1 else 0)
  in
  (rows, cols, Nx.create Nx.uint8 [| rows; cols |] roi)

let of_landmask ~source (g : Jord.Grid.t) mask =
  let fail fmt =
    Format.kasprintf (fun m -> invalid_arg ("Tessera.Tile.of_landmask: " ^ m))
      fmt
  in
  let north_up =
    fst g.col_step > 0.
    && snd g.col_step = 0.
    && fst g.row_step = 0.
    && snd g.row_step < 0.
  in
  if not north_up then fail "a grid that is not north up";
  if Nx.shape mask <> [| g.rows; g.cols |] then
    fail "a mask of shape %a, [%d; %d] expected" Nx.pp_shape (Nx.shape mask)
      g.rows g.cols;
  let rows, cols, roi = template g mask in
  let x0, y0 = g.origin in
  { grid = on_lattice g.crs ~rows ~cols (down x0, up y0); roi; source }

(* Places of polygons *)

let bounds (p : Jord.Polygons.t) =
  let column c = Nx.slice [ Nx.A; Nx.I c ] p.points in
  let lo c = Nx.item [] (Nx.min (column c))
  and hi c = Nx.item [] (Nx.max (column c)) in
  (lo 0, lo 1, hi 0, hi 1)

let load_landmask ~source file =
  let d = List.hd (Nx_io.load_tiff file) in
  let g =
    match Jord_geotiff.grid (Nx_io.Tiff.tags d) with
    | Ok g -> g
    | Error e -> error failwith "load_landmask" file "%s" e
  in
  (* Band 1, as TESSERA reads a landmask. *)
  match Nx_io.Tiff.samples d with
  | Nx.P x -> (
      match of_landmask ~source g (Nx.slice [ Nx.A; Nx.A; Nx.I 0 ] x) with
      | p -> p
      | exception Invalid_argument m ->
          error failwith "load_landmask" file "%s" m)

let of_polygons ~source (p : Jord.Polygons.t) =
  if Nx.dim 0 p.points = 0 then
    invalid_arg "Tessera.Tile.of_polygons: no point";
  let w, s, e, n = bounds (Jord.Polygons.to_crs Jord.Crs.lonlat p) in
  let crs = Jord.Crs.utm_grid_at ~lon:((w +. e) /. 2.) ~lat:((s +. n) /. 2.) in
  let p = Jord.Polygons.to_crs crs p in
  let x0, y0, x1, y1 = bounds p in
  let x0 = down x0 and y1 = up y1 in
  let cols = Float.to_int ((up x1 -. x0) /. lattice)
  and rows = Float.to_int ((y1 -. down y0) /. lattice) in
  let grid = on_lattice crs ~rows ~cols (x0, y1) in
  let _, cell = Jord.Rasterize.centres grid p in
  let roi =
    Nx.scatter ~axis:0 cell
      (Nx.ones Nx.uint8 [| Nx.numel cell |])
      (Nx.zeros Nx.uint8 [| rows * cols |])
  in
  { grid; roi = Nx.reshape [| rows; cols |] roi; source }
