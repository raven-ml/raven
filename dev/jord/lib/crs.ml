(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = Lonlat | Utm of { zone : int; north : bool }

let strf = Printf.sprintf
let lonlat = Lonlat

let utm ~zone ~north =
  if zone < 1 || zone > 60 then
    invalid_arg (strf "Jord.Crs.utm: zone %d is not in 1..60" zone);
  Utm { zone; north }

let utm_at ~lon ~lat =
  if not (Float.is_finite lon) then
    invalid_arg (strf "Jord.Crs.utm_at: longitude %g is not finite" lon);
  if not (lat >= -90. && lat <= 90.) then
    invalid_arg (strf "Jord.Crs.utm_at: latitude %g is not in [-90, 90]" lat);
  let k = int_of_float (Float.floor ((lon +. 180.) /. 6.)) in
  Utm { zone = (((k mod 60) + 60) mod 60) + 1; north = lat >= 0. }

let epsg = function
  | Lonlat -> 4326
  | Utm { zone; north } -> (if north then 32600 else 32700) + zone

let of_epsg code =
  if code = 4326 then Ok Lonlat
  else if code > 32600 && code <= 32660 then
    Ok (Utm { zone = code - 32600; north = true })
  else if code > 32700 && code <= 32760 then
    Ok (Utm { zone = code - 32700; north = false })
  else
    Error
      (strf
         "EPSG:%d is not WGS 84 (4326) or WGS 84 / UTM (32601-32660, \
          32701-32760)"
         code)

let equal a b =
  match (a, b) with
  | Lonlat, Lonlat -> true
  | Utm a, Utm b -> a.zone = b.zone && a.north = b.north
  | (Lonlat | Utm _), _ -> false

let hemisphere north = if north then "N" else "S"

let pp ppf = function
  | Lonlat -> Format.pp_print_string ppf "WGS 84"
  | Utm { zone; north } ->
      Format.fprintf ppf "WGS 84 / UTM zone %d%s" zone (hemisphere north)

(* WKT *)

(* Esri's Projection Engine WKT for EPSG:4326 and for EPSG:326zz/327zz, from
   github.com/Esri/projection-engine-db-doc (json/pe_list_geogcs.json and
   pe_list_projcs.json, EPSG 13.102). Every UTM zone's string has this form. *)

let esri_geogcs =
  {|GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]|}

let esri_projcs zone north =
  strf
    {|PROJCS["WGS_1984_UTM_Zone_%d%s",%s,PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",%s],PARAMETER["Central_Meridian",%d.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]|}
    zone (hemisphere north) esri_geogcs
    (if north then "0.0" else "10000000.0")
    ((6 * zone) - 183)

(* PROJ's WKT1_GDAL, OGC's WKT 1 with EPSG's names and identifiers, for
   EPSG:4326 and for EPSG:326zz/327zz (PROJ 9.8.1, pyproj's
   [CRS.from_epsg(code).to_wkt(WktVersion.WKT1_GDAL)]). Every UTM zone's string
   has this form. *)

let ogc_geogcs =
  {|GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]|}

let ogc_projcs zone north =
  strf
    {|PROJCS["WGS 84 / UTM zone %d%s",%s,PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",%d],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",%s],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","%d"]]|}
    zone (hemisphere north) ogc_geogcs
    ((6 * zone) - 183)
    (if north then "0" else "10000000")
    ((if north then 32600 else 32700) + zone)

let tree s = match Wkt.parse s with Ok t -> t | Error e -> failwith e
let is_digit c = '0' <= c && c <= '9'

(* The zone and hemisphere a PROJCS name states after [prefix], as
   "WGS_1984_UTM_Zone_33N" after "wgs_1984_utm_zone_". *)
let zone_of_name prefix name =
  let name = String.lowercase_ascii name in
  let p = String.length prefix and n = String.length name in
  if n < p + 2 || String.sub name 0 p <> prefix then None
  else
    let digits = String.sub name p (n - p - 1) in
    let north =
      match name.[n - 1] with 'n' -> Some true | 's' -> Some false | _ -> None
    in
    match (int_of_string_opt digits, north) with
    | Some zone, Some north
      when zone >= 1 && zone <= 60 && String.for_all is_digit digits ->
        Some (zone, north)
    | _ -> None

(* Each producer's prefix of a UTM zone's name, and its PROJCS. *)
let producers =
  [ ("wgs_1984_utm_zone_", esri_projcs); ("wgs 84 / utm zone ", ogc_projcs) ]

let of_wkt s =
  let refuse kind name =
    Error
      (strf "%s[%S] is not the WKT of WGS 84 or of a WGS 84 / UTM zone" kind
         name)
  in
  match Wkt.parse s with
  | Error _ as e -> e
  | Ok t -> (
      match t with
      | Wkt.Object (key, Wkt.Text name :: _) -> (
          match String.uppercase_ascii key with
          | "GEOGCS" ->
              let same g = Wkt.equal t (tree g) in
              if same esri_geogcs || same ogc_geogcs then Ok Lonlat
              else refuse "GEOGCS" name
          | "PROJCS" ->
              let named (prefix, projcs) =
                match zone_of_name prefix name with
                | Some (zone, north) when Wkt.equal t (tree (projcs zone north))
                  ->
                    Some (Utm { zone; north })
                | _ -> None
              in
              Option.fold ~none:(refuse "PROJCS" name) ~some:Result.ok
                (List.find_map named producers)
          | key -> Error (strf "WKT %s is not a GEOGCS or a PROJCS" key))
      | _ -> Error "WKT text is not a GEOGCS or a PROJCS with a name")

(* Transverse Mercator *)

(* WGS 84's flattening and UTM's parameters. *)
let flattening = 1. /. 298.257223563
let false_easting = 500000.
let false_northing_south = 10000000.
let semi_major = 6378137.
let max_offset = 40.

(* How far along its parallel a point may lie past [max_offset]: twice the
   transforms' stated error, the most a round trip moves a point, so that a
   point of the boundary, or a pole, where every longitude is one point, passes
   both ways. *)
let boundary_slack = 12e-9
let deg = Float.pi /. 180.
let n = flattening /. (2. -. flattening)
let e2 = flattening *. (2. -. flattening)
let e = Float.sqrt e2

(* [series j cs] is n^j (c_j + c_(j+1) n + ... ), cs from the n^j term up, in
   Horner form as Karney advises. Each step is one fused multiply-add and the
   power a product, so every system computes the same bits (OCaml's arm64
   backend would fuse [c +. n *. acc] on its own, and libm's [**] varies). *)
let series j cs =
  let h = List.fold_right (fun c acc -> Float.fma n acc c) cs 0. in
  let rec pow k = if k = 0 then 1. else n *. pow (k - 1) in
  h *. pow j

(* k0 A, metres per radian of ξ and η: A of Karney eq. (29) to n¹⁰ for these
   parameters, in exact rational arithmetic, rounded once. Evaluating eq. (29)
   in float64 lands 2 ulps away, which moves a northing near the pole by 2
   nm. *)
let k0_a = 0x1.847b98aa272e2p+22

(* Karney eq. (35), α1..α6. *)
let alpha =
  [
    series 1
      [
        1. /. 2.;
        -2. /. 3.;
        5. /. 16.;
        41. /. 180.;
        -127. /. 288.;
        7891. /. 37800.;
      ];
    series 2
      [
        13. /. 48.;
        -3. /. 5.;
        557. /. 1440.;
        281. /. 630.;
        -1983433. /. 1935360.;
      ];
    series 3
      [ 61. /. 240.; -103. /. 140.; 15061. /. 26880.; 167603. /. 181440. ];
    series 4 [ 49561. /. 161280.; -179. /. 168.; 6601661. /. 7257600. ];
    series 5 [ 34729. /. 80640.; -3418889. /. 1995840. ];
    series 6 [ 212378941. /. 319334400. ];
  ]

(* Karney eq. (36), β1..β6. *)
let beta =
  [
    series 1
      [
        1. /. 2.;
        -2. /. 3.;
        37. /. 96.;
        -1. /. 360.;
        -81. /. 512.;
        96199. /. 604800.;
      ];
    series 2
      [
        1. /. 48.; 1. /. 15.; -437. /. 1440.; 46. /. 105.; -1118711. /. 3870720.;
      ];
    series 3 [ 17. /. 480.; -37. /. 840.; -209. /. 4480.; 5569. /. 90720. ];
    series 4 [ 4397. /. 161280.; -11. /. 504.; -830251. /. 7257600. ];
    series 5 [ 4583. /. 161280.; -108847. /. 3991680. ];
    series 6 [ 20648693. /. 638668800. ];
  ]

let hypot1 x = Nx.sqrt (Nx.add_s (Nx.square x) 1.)

(* τ' = tan φ' from τ = tan φ, Karney eqs. (7)-(9). *)
let taup tau =
  let s = Nx.mul_s (Nx.atanh (Nx.mul_s (Nx.div tau (hypot1 tau)) e)) e in
  let sigma = Nx.sinh s in
  Nx.sub (Nx.mul tau (hypot1 sigma)) (Nx.mul sigma (hypot1 tau))

(* The real and imaginary parts of Σ c_j sin 2jζ, ζ = ξ + iη, by Clenshaw's
   recurrence y_j = 2 cos 2ζ · y_(j+1) - y_(j+2) + c_j and the sum sin 2ζ · y_1
   (Karney §5). *)
let clenshaw cs xi eta =
  let s2 = Nx.sin (Nx.mul_s xi 2.) and c2 = Nx.cos (Nx.mul_s xi 2.) in
  let sh = Nx.sinh (Nx.mul_s eta 2.) and ch = Nx.cosh (Nx.mul_s eta 2.) in
  let ar = Nx.mul_s (Nx.mul c2 ch) 2. and ai = Nx.mul_s (Nx.mul s2 sh) (-2.) in
  let zero = Nx.zeros_like xi in
  let step c ((yr, yi), (zr, zi)) =
    let r = Nx.add_s (Nx.sub (Nx.sub (Nx.mul ar yr) (Nx.mul ai yi)) zr) c in
    let i = Nx.sub (Nx.add (Nx.mul ar yi) (Nx.mul ai yr)) zi in
    ((r, i), (yr, yi))
  in
  let (yr, yi), _ = List.fold_right step cs ((zero, zero), (zero, zero)) in
  let sr = Nx.mul s2 ch and si = Nx.mul c2 sh in
  (Nx.sub (Nx.mul sr yr) (Nx.mul si yi), Nx.add (Nx.mul sr yi) (Nx.mul si yr))

let central_meridian zone = float_of_int ((6 * zone) - 183)
let false_northing north = if north then 0. else false_northing_south

(* [d] reduced to [-180, 180). *)
let wrap d =
  Nx.sub d (Nx.mul_s (Nx.floor (Nx.div_s (Nx.add_s d 180.) 360.)) 360.)

(* The easting, northing and the longitude from the central meridian of each
   point (lon, lat), Karney eqs. (7)-(13). *)
let forward ~zone ~north lon lat =
  let dlon = wrap (Nx.sub_s lon (central_meridian zone)) in
  let lam = Nx.mul_s dlon deg in
  let tp = taup (Nx.tan (Nx.mul_s lat deg)) in
  let cl = Nx.cos lam in
  let xi' = Nx.atan2 tp cl in
  let eta' = Nx.asinh (Nx.div (Nx.sin lam) (Nx.hypot tp cl)) in
  let sr, si = clenshaw alpha xi' eta' in
  let x = Nx.add_s (Nx.mul_s (Nx.add eta' si) k0_a) false_easting in
  let y = Nx.add_s (Nx.mul_s (Nx.add xi' sr) k0_a) (false_northing north) in
  (x, y, dlon)

(* The longitude, latitude and longitude from the central meridian of each point
   (x, y), and its ξ, Karney eqs. (15)-(22) with two Newton steps. The series
   are odd in ξ and η, so the point is taken to the first quadrant and a point
   past the pole (ξ > π/2) folded back, as GeographicLib's Reverse does. At the
   pole cos ξ' can round below 0, and is clamped. *)
let inverse ~zone ~north x y =
  let xi0 = Nx.div_s (Nx.sub_s y (false_northing north)) k0_a in
  let eta0 = Nx.div_s (Nx.sub_s x false_easting) k0_a in
  let back = Nx.greater_s (Nx.abs xi0) (Float.pi /. 2.) in
  let xi = Nx.where back (Nx.rsub_s Float.pi (Nx.abs xi0)) (Nx.abs xi0) in
  let eta = Nx.abs eta0 in
  let sr, si = clenshaw beta xi eta in
  let xi' = Nx.sub xi sr and eta' = Nx.sub eta si in
  let s = Nx.sinh eta' and c = Nx.maximum_s (Nx.cos xi') 0. in
  let r = Nx.hypot s c in
  let pole = Nx.equal_s r 0. in
  let tp = Nx.div (Nx.sin xi') r in
  let newton tau =
    let tpi = taup tau in
    let num = Nx.add_s (Nx.mul_s (Nx.square tau) (1. -. e2)) 1. in
    let den = Nx.mul_s (hypot1 tau) (1. -. e2) in
    Nx.add tau (Nx.mul (Nx.div (Nx.sub tp tpi) (hypot1 tpi)) (Nx.div num den))
  in
  (* Where r is 0 the point is the pole, latitude 90, longitude 0. *)
  let lat = Nx.div_s (Nx.atan (newton (newton tp))) deg in
  let lat = Nx.where pole (Nx.full_like lat 90.) lat in
  let lam = Nx.where pole (Nx.zeros_like s) (Nx.div_s (Nx.atan2 s c) deg) in
  let lam = Nx.where back (Nx.rsub_s 180. lam) lam in
  let lat = Nx.where (Nx.less_s xi0 0.) (Nx.neg lat) lat in
  let dlon = Nx.where (Nx.less_s eta0 0.) (Nx.neg lam) lam in
  let lon = Nx.add_s dlon (central_meridian zone) in
  let lon =
    Nx.where (Nx.greater_s lon 180.) (Nx.sub_s lon 360.)
      (Nx.where (Nx.less_s lon (-180.)) (Nx.add_s lon 360.) lon)
  in
  (lon, lat, dlon, xi0)

(* Transforms *)

(* Raises through [Nx.check] where a point (a, b) of [c] is neither NaN nor
   [inside] its domain. *)
let require ~batch c a b inside =
  let nan = Nx.logical_or (Nx.isnan a) (Nx.isnan b) in
  let ok = Nx.reshape batch (Nx.logical_or nan inside) in
  let a = Nx.reshape batch a and b = Nx.reshape batch b in
  Nx.check
    Nx.Ptree.(pair tensor tensor)
    ok (a, b)
    (fun i (a, b) ->
      let at = String.concat ", " (List.map string_of_int (Array.to_list i)) in
      Invalid_argument
        (Format.asprintf
           "Jord.Crs.transform: point [%s] (%.17g, %.17g) is outside %a's \
            domain"
           at (Nx.item [] a) (Nx.item [] b) pp c))

(* Whether (offset, lat) lies within [boundary_slack] of the domain, along the
   parallel. *)
let near offset lat =
  let past = Nx.sub_s (Nx.abs offset) max_offset in
  let metres =
    Nx.mul (Nx.mul_s past (deg *. semi_major)) (Nx.cos (Nx.mul_s lat deg))
  in
  Nx.less_equal_s metres boundary_slack

let transform (type d) ~src ~dst (p : d Nx.float64_t) : d Nx.float64_t =
  let batch, a, b = Points.split "Jord.Crs.transform" p in
  let to_lonlat c a b =
    match c with
    | Lonlat -> (a, b)
    | Utm { zone; north } ->
        let lon, lat, offset, xi = inverse ~zone ~north a b in
        (* Beyond half a meridian from the equator, a point is no point's
           image. *)
        let image = Nx.less_equal_s (Nx.abs xi) Float.pi in
        require ~batch c a b (Nx.logical_and (near offset lat) image);
        (lon, lat)
  in
  let of_lonlat c lon lat =
    match c with
    | Lonlat -> (lon, lat)
    | Utm { zone; north } ->
        let x, y, offset = forward ~zone ~north lon lat in
        let valid = Nx.less_equal_s (Nx.abs lat) 90. in
        require ~batch c lon lat (Nx.logical_and (near offset lat) valid);
        (x, y)
  in
  if equal src dst then p
  else
    let lon, lat = to_lonlat src a b in
    let x, y = of_lonlat dst lon lat in
    Points.join batch x y
