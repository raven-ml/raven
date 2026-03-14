(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let pi = Float.pi
let deg_to_rad = pi /. 180.0
let two_pi = Nx.scalar Nx.float64 (2.0 *. pi)

type frame = ICRS | Galactic | Ecliptic_j2000 | Supergalactic

(* Internally stores lon/lat in radians *)
type t = { frame : frame; lon : Nx.float64_t; lat : Nx.float64_t }

(* IAU rotation matrices *)

let ra_gp = 192.85948 *. deg_to_rad
let dec_gp = 27.12825 *. deg_to_rad
let l_ncp = 122.93192 *. deg_to_rad

let icrs_to_gal =
  let sd = Float.sin dec_gp and cd = Float.cos dec_gp in
  let sa = Float.sin ra_gp and ca = Float.cos ra_gp in
  let sl = Float.sin l_ncp and cl = Float.cos l_ncp in
  [|
    [|
      (~-.sl *. sa) -. (cl *. ca *. sd);
      (sl *. ca) -. (cl *. sa *. sd);
      cl *. cd;
    |];
    [|
      (cl *. sa) -. (sl *. ca *. sd);
      (~-.cl *. ca) -. (sl *. sa *. sd);
      sl *. cd;
    |];
    [| ca *. cd; sa *. cd; sd |];
  |]

let transpose_3x3 m =
  [|
    [| m.(0).(0); m.(1).(0); m.(2).(0) |];
    [| m.(0).(1); m.(1).(1); m.(2).(1) |];
    [| m.(0).(2); m.(1).(2); m.(2).(2) |];
  |]

let gal_to_icrs = transpose_3x3 icrs_to_gal

(* Fixed J2000.0 mean obliquity: 23.4392911 degrees *)
let obliquity = 23.4392911 *. deg_to_rad

let icrs_to_ecl =
  let se = Float.sin obliquity and ce = Float.cos obliquity in
  [| [| 1.0; 0.0; 0.0 |]; [| 0.0; ce; se |]; [| 0.0; ~-.se; ce |] |]

let ecl_to_icrs = transpose_3x3 icrs_to_ecl

(* Supergalactic: defined relative to Galactic. SGP at (l=47.37, b=6.32), SGL
   origin at l=137.37 *)
let sgl_l0 = 137.37 *. deg_to_rad
let sgp_l = 47.37 *. deg_to_rad
let sgp_b = 6.32 *. deg_to_rad

let gal_to_sgal =
  let sb = Float.sin sgp_b and cb = Float.cos sgp_b in
  let sl = Float.sin sgp_l and cl = Float.cos sgp_l in
  let sl0 = Float.sin sgl_l0 and cl0 = Float.cos sgl_l0 in
  let r00 = (~-.sl0 *. sl) -. (cl0 *. cl *. sb) in
  let r01 = (sl0 *. cl) -. (cl0 *. sl *. sb) in
  let r02 = cl0 *. cb in
  let r10 = (cl0 *. sl) -. (sl0 *. cl *. sb) in
  let r11 = (~-.cl0 *. cl) -. (sl0 *. sl *. sb) in
  let r12 = sl0 *. cb in
  let r20 = cl *. cb in
  let r21 = sl *. cb in
  let r22 = sb in
  [| [| r00; r01; r02 |]; [| r10; r11; r12 |]; [| r20; r21; r22 |] |]

let sgal_to_gal = transpose_3x3 gal_to_sgal

let rotate mat lon_rad lat_rad =
  let cl = Nx.cos lat_rad and sl = Nx.sin lat_rad in
  let ca = Nx.cos lon_rad and sa = Nx.sin lon_rad in
  let x = Nx.mul cl ca and y = Nx.mul cl sa in
  let x' =
    Nx.add
      (Nx.add (Nx.mul_s x mat.(0).(0)) (Nx.mul_s y mat.(0).(1)))
      (Nx.mul_s sl mat.(0).(2))
  in
  let y' =
    Nx.add
      (Nx.add (Nx.mul_s x mat.(1).(0)) (Nx.mul_s y mat.(1).(1)))
      (Nx.mul_s sl mat.(1).(2))
  in
  let z' =
    Nx.add
      (Nx.add (Nx.mul_s x mat.(2).(0)) (Nx.mul_s y mat.(2).(1)))
      (Nx.mul_s sl mat.(2).(2))
  in
  let z_clamped = Nx.clamp ~min:(-1.0) ~max:1.0 z' in
  let lat' = Nx.asin z_clamped in
  let lon' = Nx.atan2 y' x' in
  let mask = Nx.less_s lon' 0.0 in
  let lon' = Nx.where mask (Nx.add lon' two_pi) lon' in
  (lon', lat')

let ensure_1d t = if Nx.ndim t = 0 then Nx.reshape [| 1 |] t else t

let make frame ~lon ~lat =
  let lon_rad = ensure_1d (Unit.Angle.to_tensor lon) in
  let lat_rad = ensure_1d (Unit.Angle.to_tensor lat) in
  if Nx.ndim lon_rad <> 1 || Nx.ndim lat_rad <> 1 then
    invalid_arg "Coord: lon and lat must be scalar or 1-D tensors";
  if Nx.numel lon_rad <> Nx.numel lat_rad then
    invalid_arg "Coord: lon and lat must have the same length";
  { frame; lon = lon_rad; lat = lat_rad }

let of_radec ~ra ~dec = make ICRS ~lon:ra ~lat:dec
let of_galactic ~l ~b = make Galactic ~lon:l ~lat:b
let of_ecliptic_j2000 ~lon ~lat = make Ecliptic_j2000 ~lon ~lat
let of_supergalactic ~sgl ~sgb = make Supergalactic ~lon:sgl ~lat:sgb
let frame c = c.frame
let size c = Nx.numel c.lon
let lon c = Unit.Angle.of_tensor c.lon
let lat c = Unit.Angle.of_tensor c.lat

let to_icrs c =
  match c.frame with
  | ICRS -> c
  | Galactic ->
      let lon', lat' = rotate gal_to_icrs c.lon c.lat in
      { frame = ICRS; lon = lon'; lat = lat' }
  | Ecliptic_j2000 ->
      let lon', lat' = rotate ecl_to_icrs c.lon c.lat in
      { frame = ICRS; lon = lon'; lat = lat' }
  | Supergalactic ->
      let gal_lon, gal_lat = rotate sgal_to_gal c.lon c.lat in
      let icrs_lon, icrs_lat = rotate gal_to_icrs gal_lon gal_lat in
      { frame = ICRS; lon = icrs_lon; lat = icrs_lat }

let ra c = lon (to_icrs c)
let dec c = lat (to_icrs c)

let to_frame target c =
  if c.frame = target then c
  else
    let icrs = to_icrs c in
    match target with
    | ICRS -> icrs
    | Galactic ->
        let lon', lat' = rotate icrs_to_gal icrs.lon icrs.lat in
        { frame = Galactic; lon = lon'; lat = lat' }
    | Ecliptic_j2000 ->
        let lon', lat' = rotate icrs_to_ecl icrs.lon icrs.lat in
        { frame = Ecliptic_j2000; lon = lon'; lat = lat' }
    | Supergalactic ->
        let gal_lon, gal_lat = rotate icrs_to_gal icrs.lon icrs.lat in
        let sg_lon, sg_lat = rotate gal_to_sgal gal_lon gal_lat in
        { frame = Supergalactic; lon = sg_lon; lat = sg_lat }

let icrs c = to_frame ICRS c
let galactic c = to_frame Galactic c
let ecliptic_j2000 c = to_frame Ecliptic_j2000 c
let supergalactic c = to_frame Supergalactic c

let trig_of a b =
  let a = to_icrs a and b = to_icrs b in
  let dlon = Nx.sub b.lon a.lon in
  let cos_lat1 = Nx.cos a.lat and sin_lat1 = Nx.sin a.lat in
  let cos_lat2 = Nx.cos b.lat and sin_lat2 = Nx.sin b.lat in
  let cos_dlon = Nx.cos dlon and sin_dlon = Nx.sin dlon in
  (dlon, cos_lat1, sin_lat1, cos_lat2, sin_lat2, cos_dlon, sin_dlon)

let separation a b =
  if size a <> size b then
    invalid_arg "Coord.separation: arrays must have the same length";
  let _, cos_lat1, sin_lat1, cos_lat2, sin_lat2, cos_dlon, sin_dlon =
    trig_of a b
  in
  (* Vincenty formula *)
  let a1 = Nx.mul cos_lat2 sin_dlon in
  let a2 =
    Nx.sub (Nx.mul cos_lat1 sin_lat2)
      (Nx.mul (Nx.mul sin_lat1 cos_lat2) cos_dlon)
  in
  let num = Nx.sqrt (Nx.add (Nx.square a1) (Nx.square a2)) in
  let den =
    Nx.add (Nx.mul sin_lat1 sin_lat2)
      (Nx.mul (Nx.mul cos_lat1 cos_lat2) cos_dlon)
  in
  let sep = Nx.atan2 num den in
  Unit.Angle.of_tensor (Nx.abs sep)

let position_angle a b =
  if size a <> size b then
    invalid_arg "Coord.position_angle: arrays must have the same length";
  let _, cos_lat1, sin_lat1, cos_lat2, sin_lat2, cos_dlon, sin_dlon =
    trig_of a b
  in
  let num = Nx.mul cos_lat2 sin_dlon in
  let den =
    Nx.sub (Nx.mul cos_lat1 sin_lat2)
      (Nx.mul (Nx.mul sin_lat1 cos_lat2) cos_dlon)
  in
  let pa = Nx.atan2 num den in
  let mask = Nx.less_s pa 0.0 in
  Unit.Angle.of_tensor (Nx.where mask (Nx.add pa two_pi) pa)

(* --- Offset operations --- *)

let offset_by ~position_angle ~separation c =
  let pa = Unit.Angle.to_tensor position_angle in
  let sep = Unit.Angle.to_tensor separation in
  let cos_sep = Nx.cos sep and sin_sep = Nx.sin sep in
  let cos_pa = Nx.cos pa and sin_pa = Nx.sin pa in
  let sin_lat = Nx.sin c.lat and cos_lat = Nx.cos c.lat in
  (* lat2 = asin(sin(lat1)*cos(sep) + cos(lat1)*sin(sep)*cos(pa)) *)
  let sin_lat2 =
    Nx.add (Nx.mul sin_lat cos_sep) (Nx.mul (Nx.mul cos_lat sin_sep) cos_pa)
  in
  let lat2 = Nx.asin (Nx.clamp ~min:(-1.0) ~max:1.0 sin_lat2) in
  (* lon2 = lon1 + atan2(sin(pa)*sin(sep), cos(lat1)*cos(sep) -
     sin(lat1)*sin(sep)*cos(pa)) *)
  let num = Nx.mul sin_pa sin_sep in
  let den =
    Nx.sub (Nx.mul cos_lat cos_sep) (Nx.mul (Nx.mul sin_lat sin_sep) cos_pa)
  in
  let dlon = Nx.atan2 num den in
  let lon2 = Nx.add c.lon dlon in
  let lon2 = Nx.where (Nx.less_s lon2 0.0) (Nx.add lon2 two_pi) lon2 in
  let lon2 =
    Nx.where (Nx.greater_equal lon2 two_pi) (Nx.sub lon2 two_pi) lon2
  in
  { frame = c.frame; lon = lon2; lat = lat2 }

let spherical_offsets_to a b =
  if size a <> size b then
    invalid_arg "Coord.spherical_offsets_to: arrays must have the same length";
  if a.frame <> b.frame then
    invalid_arg
      "Coord.spherical_offsets_to: coordinates must be in the same frame";
  (* Δlon = (lon_b - lon_a) * cos(lat_a), Δlat = lat_b - lat_a *)
  let dlon = Nx.mul (Nx.sub b.lon a.lon) (Nx.cos a.lat) in
  let dlat = Nx.sub b.lat a.lat in
  (Unit.Angle.of_tensor dlon, Unit.Angle.of_tensor dlat)

(* --- Catalog cross-matching --- *)

type coord = t
type result = { indices : Nx.int32_t; separations : Unit.angle Unit.t }

type within_result = {
  indices_a : Nx.int32_t;
  indices_b : Nx.int32_t;
  separations : Unit.angle Unit.t;
}

let to_xyz c =
  let icrs = to_icrs c in
  let n = size c in
  let xs = Array.make n 0.0 in
  let ys = Array.make n 0.0 in
  let zs = Array.make n 0.0 in
  for i = 0 to n - 1 do
    let r = Nx.item [ i ] icrs.lon in
    let d = Nx.item [ i ] icrs.lat in
    let cd = Float.cos d in
    xs.(i) <- cd *. Float.cos r;
    ys.(i) <- cd *. Float.sin r;
    zs.(i) <- Float.sin d
  done;
  (xs, ys, zs)

let chord_to_rad chord_sq =
  let chord = Float.sqrt (Float.max 0.0 chord_sq) in
  let half_chord = Float.min 1.0 (chord /. 2.0) in
  2.0 *. Float.asin half_chord

module Index = struct
  type t = { tree : Kdtree.t }

  let of_coord c =
    let xs, ys, zs = to_xyz c in
    let tree = Kdtree.build xs ys zs in
    { tree }

  let nearest idx query =
    let qx, qy, qz = to_xyz query in
    let n = Array.length qx in
    let indices = Nx.zeros Nx.int32 [| n |] in
    let seps = Nx.zeros Nx.float64 [| n |] in
    for i = 0 to n - 1 do
      let j, dist_sq = Kdtree.nearest idx.tree qx.(i) qy.(i) qz.(i) in
      Nx.set_item [ i ] (Int32.of_int j) indices;
      Nx.set_item [ i ] (chord_to_rad dist_sq) seps
    done;
    { indices; separations = Unit.Angle.of_tensor seps }

  let within idx query ~max_sep =
    let max_sep_rad = Nx.item [] (Unit.Angle.to_tensor max_sep) in
    let half_angle = max_sep_rad /. 2.0 in
    let chord = 2.0 *. Float.sin half_angle in
    let max_dist_sq = chord *. chord in
    let qx, qy, qz = to_xyz query in
    let na = Array.length qx in
    let acc = ref [] and count = ref 0 in
    for i = 0 to na - 1 do
      let matches = Kdtree.within idx.tree qx.(i) qy.(i) qz.(i) max_dist_sq in
      List.iter
        (fun (j, dist_sq) ->
          acc := (i, j, chord_to_rad dist_sq) :: !acc;
          incr count)
        matches
    done;
    let n = !count in
    let out_a = Nx.zeros Nx.int32 [| n |] in
    let out_b = Nx.zeros Nx.int32 [| n |] in
    let out_s = Nx.zeros Nx.float64 [| n |] in
    let k = ref (n - 1) in
    List.iter
      (fun (i, j, sep) ->
        let k' = !k in
        Nx.set_item [ k' ] (Int32.of_int i) out_a;
        Nx.set_item [ k' ] (Int32.of_int j) out_b;
        Nx.set_item [ k' ] sep out_s;
        decr k)
      !acc;
    {
      indices_a = out_a;
      indices_b = out_b;
      separations = Unit.Angle.of_tensor out_s;
    }
end

let nearest query catalog =
  if size catalog = 0 then invalid_arg "Coord.nearest: catalog is empty";
  let idx = Index.of_coord catalog in
  Index.nearest idx query

let within a b ~max_sep =
  let idx = Index.of_coord b in
  Index.within idx a ~max_sep
