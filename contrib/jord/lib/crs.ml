(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The systems' codes and parameters, and the datums' steps to WGS 84, are a
   subset of the EPSG Dataset, which IOGP owns: https://epsg.org, under its
   terms of use, https://epsg.org/terms-of-use.html. *)

type datum = Wgs84 | Nad83 | Euref_fin | Nzgd2000 | Gda94

type projection =
  | Transverse_mercator of {
      lat0 : float;
      lon0 : float;
      k0 : float;
      false_easting : float;
      false_northing : float;
    }
  | Albers of {
      lat0 : float;
      lon0 : float;
      lat1 : float;
      lat2 : float;
      false_easting : float;
      false_northing : float;
    }
  | Pseudo_mercator of {
      lon0 : float;
      false_easting : float;
      false_northing : float;
    }
  | Sinusoidal of {
      lon0 : float;
      radius : float;
      false_easting : float;
      false_northing : float;
    }

type t = { datum : datum; projection : projection option }

let strf = Printf.sprintf

(* Both ellipsoids' semi-major axis. *)
let semi_major = 6378137.

(* [x] with -0 as 0, so that equal systems print alike. *)
let positive_zero x = x +. 0.

(* Raises unless each parameter is finite, each latitude in [-90, 90] and
   [lon0] in [-180, 180]; [lon0] with 180 as -180. *)
let check fn ~lats ~lon0 others =
  List.iter
    (fun (name, x) ->
      if not (Float.is_finite x) then
        invalid_arg (strf "%s: %s %g is not finite" fn name x))
    ((("lon0", lon0) :: lats) @ others);
  List.iter
    (fun (name, x) ->
      if Float.abs x > 90. then
        invalid_arg (strf "%s: %s %g is not in [-90, 90]" fn name x))
    lats;
  if Float.abs lon0 > 180. then
    invalid_arg (strf "%s: lon0 %g is not in [-180, 180]" fn lon0);
  if lon0 = 180. then -180. else positive_zero lon0

let transverse_mercator ~lat0 ~lon0 ~k0 ~false_easting ~false_northing =
  let fn = "Jord.Crs.transverse_mercator" in
  let lon0 =
    check fn ~lats:[ ("lat0", lat0) ] ~lon0
      [
        ("k0", k0);
        ("false_easting", false_easting);
        ("false_northing", false_northing);
      ]
  in
  if not (k0 > 0.) then invalid_arg (strf "%s: k0 %g is not positive" fn k0);
  Transverse_mercator
    {
      lat0 = positive_zero lat0;
      lon0;
      k0;
      false_easting = positive_zero false_easting;
      false_northing = positive_zero false_northing;
    }

let albers ~lat0 ~lon0 ~lat1 ~lat2 ~false_easting ~false_northing =
  let fn = "Jord.Crs.albers" in
  let lon0 =
    check fn
      ~lats:[ ("lat0", lat0); ("lat1", lat1); ("lat2", lat2) ]
      ~lon0
      [ ("false_easting", false_easting); ("false_northing", false_northing) ]
  in
  Albers
    {
      lat0 = positive_zero lat0;
      lon0;
      lat1 = positive_zero (Float.max lat1 lat2);
      lat2 = positive_zero (Float.min lat1 lat2);
      false_easting = positive_zero false_easting;
      false_northing = positive_zero false_northing;
    }

let pseudo_mercator ~lon0 ~false_easting ~false_northing =
  let fn = "Jord.Crs.pseudo_mercator" in
  let lon0 =
    check fn ~lats:[] ~lon0
      [ ("false_easting", false_easting); ("false_northing", false_northing) ]
  in
  Pseudo_mercator
    {
      lon0;
      false_easting = positive_zero false_easting;
      false_northing = positive_zero false_northing;
    }

let sinusoidal ~lon0 ~radius ~false_easting ~false_northing =
  let fn = "Jord.Crs.sinusoidal" in
  let lon0 =
    check fn ~lats:[] ~lon0
      [
        ("radius", radius);
        ("false_easting", false_easting);
        ("false_northing", false_northing);
      ]
  in
  if not (radius > 0.) then
    invalid_arg (strf "%s: radius %g is not positive" fn radius);
  Sinusoidal
    {
      lon0;
      radius;
      false_easting = positive_zero false_easting;
      false_northing = positive_zero false_northing;
    }

let lonlat = { datum = Wgs84; projection = None }

(* MODLAND's sphere: GCTP's radius for MODIS's sinusoidal grids. *)
let modis_radius = 6371007.181

let modis =
  {
    datum = Wgs84;
    projection =
      Some
        (sinusoidal ~lon0:0. ~radius:modis_radius ~false_easting:0.
           ~false_northing:0.);
  }

(* UTM's parameters, which TM35FIN shares. *)
let utm_scale = 0.9996
let utm_false_easting = 500000.
let utm_false_northing_south = 10000000.

let zone_projection zone north =
  transverse_mercator ~lat0:0.
    ~lon0:(float_of_int ((6 * zone) - 183))
    ~k0:utm_scale ~false_easting:utm_false_easting
    ~false_northing:(if north then 0. else utm_false_northing_south)

(* Whether a projection is a zone of UTM's. *)
let is_zone p =
  let is z north = p = zone_projection z north in
  List.exists (fun z -> is z true || is z false) (List.init 60 succ)

let utm ~zone ~north =
  if zone < 1 || zone > 60 then
    invalid_arg (strf "Jord.Crs.utm: zone %d is not in 1..60" zone);
  { datum = Wgs84; projection = Some (zone_projection zone north) }

let utm_at ~lon ~lat =
  if not (Float.is_finite lon) then
    invalid_arg (strf "Jord.Crs.utm_at: longitude %g is not finite" lon);
  if not (lat >= -90. && lat <= 90.) then
    invalid_arg (strf "Jord.Crs.utm_at: latitude %g is not in [-90, 90]" lat);
  let k = int_of_float (Float.floor ((lon +. 180.) /. 6.)) in
  utm ~zone:((((k mod 60) + 60) mod 60) + 1) ~north:(lat >= 0.)

(* The UTM grid's exceptions in bands V and X: 32V widened west, 31X to 37X
   at odd zones. *)
let utm_grid_at ~lon ~lat =
  let c = utm_at ~lon ~lat in
  let lon = lon -. (360. *. Float.floor ((lon +. 180.) /. 360.)) in
  let zone =
    if lat >= 56. && lat < 64. && lon >= 3. && lon < 12. then Some 32
    else if lat >= 72. && lat < 84. && lon >= 0. && lon < 42. then
      Some
        (if lon < 9. then 31
         else if lon < 21. then 33
         else if lon < 33. then 35
         else 37)
    else None
  in
  match zone with Some zone -> utm ~zone ~north:true | None -> c

(* The constructors normalise every parameter and refuse NaN, so structural
   equality is equality of the numbers. *)
let equal (a : t) b = a = b

(* The registry *)

let hemisphere north = if north then "N" else "S"
let geographic datum = { datum; projection = None }

(* Each code jord knows, with its EPSG name and its system, the names of
   EPSG's dataset 12.029. No two systems are equal, so [epsg] reads the table
   backwards. *)
let registry =
  let zones datum name range first =
    List.concat_map
      (fun zone ->
        List.map
          (fun north ->
            ( (if north then first else first + 100) + zone,
              strf "%s / UTM zone %d%s" name zone (hemisphere north),
              { (utm ~zone ~north) with datum } ))
          range)
      (List.init 60 succ)
  in
  (* EPSG numbers NAD83's 1N to 23N as 26901 to 26923, and 24N, 59N and
     60N apart. *)
  let nad83_zones =
    let zone code z =
      ( code,
        strf "NAD83 / UTM zone %dN" z,
        { (utm ~zone:z ~north:true) with datum = Nad83 } )
    in
    List.filter
      (fun (code, _, _) -> code <= 26923)
      (zones Nad83 "NAD83" [ true ] 26900)
    @ [ zone 9712 24; zone 3372 59; zone 3373 60 ]
  in
  [
    (4326, "WGS 84", lonlat);
    (4269, "NAD83", geographic Nad83);
    (10690, "ETRS89-FIN [EUREF-FIN]", geographic Euref_fin);
    ( 3067,
      "ETRS89-FIN [EUREF-FIN] / TM35FIN(E,N)",
      { datum = Euref_fin; projection = Some (zone_projection 35 true) } );
    (4167, "NZGD2000", geographic Nzgd2000);
    (4283, "GDA94", geographic Gda94);
    ( 2193,
      "NZGD2000 / New Zealand Transverse Mercator 2000",
      {
        datum = Nzgd2000;
        projection =
          Some
            (transverse_mercator ~lat0:0. ~lon0:173. ~k0:utm_scale
               ~false_easting:1600000. ~false_northing:10000000.);
      } );
  ]
  @ [
      ( 5070,
        "NAD83 / Conus Albers",
        {
          datum = Nad83;
          projection =
            Some
              (albers ~lat0:23. ~lon0:(-96.) ~lat1:29.5 ~lat2:45.5
                 ~false_easting:0. ~false_northing:0.);
        } );
      ( 3577,
        "GDA94 / Australian Albers",
        {
          datum = Gda94;
          projection =
            Some
              (albers ~lat0:0. ~lon0:132. ~lat1:(-18.) ~lat2:(-36.)
                 ~false_easting:0. ~false_northing:0.);
        } );
    ]
  @ [
      ( 3857,
        "WGS 84 / Pseudo-Mercator",
        {
          datum = Wgs84;
          projection =
            Some
              (pseudo_mercator ~lon0:0. ~false_easting:0. ~false_northing:0.);
        } );
    ]
  @ zones Wgs84 "WGS 84" [ true; false ] 32600
  @ nad83_zones

let row c = List.find_opt (fun (_, _, c') -> equal c c') registry
let epsg c = Option.map (fun (code, _, _) -> code) (row c)

let of_epsg code =
  match List.find_opt (fun (code', _, _) -> code' = code) registry with
  | Some (_, _, c) -> Ok c
  | None ->
      Error
        (strf "EPSG:%d is not one of the systems Jord.Crs.of_epsg reads" code)

let datum_name = function
  | Wgs84 -> "WGS 84"
  | Nad83 -> "NAD83"
  | Euref_fin -> "ETRS89-FIN [EUREF-FIN]"
  | Nzgd2000 -> "NZGD2000"
  | Gda94 -> "GDA94"

let pp ppf c =
  match (row c, c.projection) with
  | Some (_, name, _), _ -> Format.pp_print_string ppf name
  | None, None -> Format.pp_print_string ppf (datum_name c.datum)
  | None, Some (Transverse_mercator p) ->
      Format.fprintf ppf
        "%s / Transverse Mercator (lat0 %.17g, lon0 %.17g, k0 %.17g, \
         false_easting %.17g, false_northing %.17g)"
        (datum_name c.datum) p.lat0 p.lon0 p.k0 p.false_easting
        p.false_northing
  | None, Some (Pseudo_mercator p) ->
      Format.fprintf ppf
        "%s / Pseudo-Mercator (lon0 %.17g, false_easting %.17g, false_northing \
         %.17g)"
        (datum_name c.datum) p.lon0 p.false_easting p.false_northing
  | None, Some (Sinusoidal p) ->
      Format.fprintf ppf
        "%s / Sinusoidal (lon0 %.17g, radius %.17g, false_easting %.17g, \
         false_northing %.17g)"
        (datum_name c.datum) p.lon0 p.radius p.false_easting p.false_northing
  | None, Some (Albers p) ->
      Format.fprintf ppf
        "%s / Albers Equal Area (lat0 %.17g, lon0 %.17g, lat1 %.17g, lat2 \
         %.17g, false_easting %.17g, false_northing %.17g)"
        (datum_name c.datum) p.lat0 p.lon0 p.lat1 p.lat2 p.false_easting
        p.false_northing

(* WKT *)

(* Esri's Projection Engine WKT for EPSG:4326 and 4269 and for
   EPSG:326zz/327zz and 26901 to 26923, from
   github.com/Esri/projection-engine-db-doc (json/pe_list_geogcs.json and
   pe_list_projcs.json, EPSG 13.102). Every UTM zone's string has this form,
   with its datum's GEOGCS and name. *)

let esri_geogcs =
  {|GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]|}

let esri_nad83_geogcs =
  {|GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]|}

let esri_projcs ~name ~geogcs zone north =
  strf
    {|PROJCS["%s_UTM_Zone_%d%s",%s,PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",%s],PARAMETER["Central_Meridian",%d.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]|}
    name zone (hemisphere north) geogcs
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

(* Each producer's prefix of a UTM zone's name, its PROJCS and its datum. *)
let producers =
  [
    ( "wgs_1984_utm_zone_",
      esri_projcs ~name:"WGS_1984" ~geogcs:esri_geogcs,
      Wgs84 );
    ( "nad_1983_utm_zone_",
      esri_projcs ~name:"NAD_1983" ~geogcs:esri_nad83_geogcs,
      Nad83 );
    ("wgs 84 / utm zone ", ogc_projcs, Wgs84);
  ]

(* GDAL's WKT 1 for MODIS's sinusoid, as Microsoft's Planetary Computer
   states it in MODIS items' proj:wkt2, given the values of its three
   parameters. *)
let gdal_modis lon0 fe fn =
  strf
    {|PROJCS["unnamed",GEOGCS["Unknown datum based upon the custom spheroid",DATUM["Not specified (based on custom spheroid)",SPHEROID["Custom spheroid",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",%.17g],PARAMETER["false_easting",%.17g],PARAMETER["false_northing",%.17g],UNIT["Meter",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]|}
    lon0 fe fn

(* The system of a WKT 1 tree: one of the producers' trees. *)
let wkt1 t key name =
  let refuse kind =
    Error
      (strf
         "%s[%S] is not the WKT of WGS 84 or NAD83, of a UTM zone on them \
          that Crs.of_epsg reads, or of MODIS's sinusoid"
         kind name)
  in
  match key with
  | "GEOGCS" ->
      let same g = Wkt.equal t (tree g) in
      if same esri_geogcs || same ogc_geogcs then Ok lonlat
      else if same esri_nad83_geogcs then Ok (geographic Nad83)
      else refuse "GEOGCS"
  | _ -> (
      (* A zone reads only as a system that has a code. Esri writes NAD83's
         58N in the same form, which EPSG does not number. *)
      let named (prefix, projcs, datum) =
        match zone_of_name prefix name with
        | Some (zone, north) when Wkt.equal t (tree (projcs zone north)) ->
            let c = { (utm ~zone ~north) with datum } in
            if Option.is_some (epsg c) then Some c else None
        | _ -> None
      in
      let parameter n =
        List.find_map
          (function
            | Wkt.Object (k, [ Wkt.Text n'; Wkt.Number x ])
              when String.uppercase_ascii k = "PARAMETER"
                   && String.lowercase_ascii n' = n ->
                Some x
            | _ -> None)
          (match t with Wkt.Object (_, a) -> a | _ -> [])
      in
      let modis () =
        match
          ( parameter "longitude_of_center",
            parameter "false_easting",
            parameter "false_northing" )
        with
        | Some lon0, Some fe, Some fn
          when Float.abs lon0 <= 180.
               && Wkt.equal t (tree (gdal_modis lon0 fe fn)) ->
            Some
              {
                modis with
                projection =
                  Some
                    (sinusoidal ~lon0 ~radius:modis_radius ~false_easting:fe
                       ~false_northing:fn);
              }
        | _ -> None
      in
      match List.find_map named producers with
      | Some c -> Ok c
      | None -> (
          match modis () with Some c -> Ok c | None -> refuse "PROJCS"))

(* WKT2 *)

let keyword = function
  | Wkt.Object (k, _) -> String.uppercase_ascii k
  | _ -> ""

let attributes = function Wkt.Object (_, a) -> a | _ -> []

(* The attributes of [t] whose keyword is one of [keys]. *)
let all keys t = List.filter (fun a -> List.mem (keyword a) keys) (attributes t)

exception Refused of string

let refuse fmt = Printf.ksprintf (fun s -> raise (Refused s)) fmt

let only keys t =
  match all keys t with
  | [ a ] -> a
  | [] -> refuse "%s has no %s" (keyword t) (List.hd keys)
  | _ -> refuse "%s has more than one %s" (keyword t) (List.hd keys)

let optional keys t =
  match all keys t with
  | [] -> None
  | [ a ] -> Some a
  | _ -> refuse "%s has more than one %s" (keyword t) (List.hd keys)

let numbers t =
  List.filter_map (function Wkt.Number x -> Some x | _ -> None) (attributes t)

(* The code of [t]'s identifier in EPSG's namespace, a number or a text. *)
let epsg_id t =
  List.find_map
    (fun i ->
      match attributes i with
      | Wkt.Text ns :: code :: _ when String.uppercase_ascii ns = "EPSG" -> (
          match code with
          | Wkt.Number x when Float.is_integer x -> Some (int_of_float x)
          | Wkt.Text c -> int_of_string_opt c
          | _ -> refuse "%s's EPSG identifier is not a code" (keyword t))
      | _ -> None)
    (all [ "ID" ] t)

(* EPSG's codes of each datum, and of its geographic system. *)
let datum_codes =
  [
    (6326, 4326, Wgs84);
    (6269, 4269, Nad83);
    (1391, 10690, Euref_fin);
    (6167, 4167, Nzgd2000);
    (6283, 4283, Gda94);
  ]

let flattening_of = function
  | Wgs84 -> 298.257223563
  | Nad83 | Euref_fin | Nzgd2000 | Gda94 -> 298.257222101

(* A unit's factor, from a [unit] object: [None] where it is [base], by its
   identifier or its printed factor. *)
let unit ~ids ~base u =
  match numbers u with
  | x :: _ when x = base -> None
  | x :: _ -> (
      match epsg_id u with Some c when List.mem c ids -> None | _ -> Some x)
  | [] -> refuse "%s has no factor" (keyword u)

let degree = Float.pi /. 180.
let printed_degree = 0.0174532925199433

let angle_units = [ "ANGLEUNIT"; "UNIT" ]
let length_units = [ "LENGTHUNIT"; "UNIT" ]

(* [x] in [u], in degrees: exactly [x] where [u] is the degree. *)
let in_degrees u x =
  match u with
  | None -> x
  | Some u -> (
      match unit ~ids:[ 9102; 9122 ] ~base:printed_degree u with
      | None -> x
      | Some f -> x *. f /. degree)

let in_units ~ids u x =
  match u with
  | None -> x
  | Some u -> (
      match unit ~ids ~base:1. u with None -> x | Some f -> x *. f)

let in_metres = in_units ~ids:[ 9001 ]

(* The datum a geographic system [g] states, and whether it is MODIS's
   sphere, unnamed: by the identifier of its DATUM or ENSEMBLE, else of [g],
   which must agree, and whose ellipsoid must be the datum's. *)
let wkt2_datum g =
  let d = only [ "DATUM"; "GEODETICDATUM"; "TRF"; "ENSEMBLE" ] g in
  let e = only [ "ELLIPSOID"; "SPHEROID" ] d in
  let a, rf =
    match numbers e with
    | [ a; rf ] -> (in_metres (optional length_units e) a, rf)
    | _ -> refuse "%s is not a semi-major axis and an inverse flattening"
             (keyword e)
  in
  (match optional [ "PRIMEM"; "PRIMEMERIDIAN" ] g with
  | Some p when numbers p <> [ 0. ] ->
      refuse "the prime meridian is not Greenwich's"
  | _ -> ());
  let by f code = List.find_opt (fun r -> f r = code) datum_codes in
  let datum =
    match
      ( Option.map (by (fun (d, _, _) -> d)) (epsg_id d),
        Option.map (by (fun (_, g, _) -> g)) (epsg_id g) )
    with
    | Some (Some (_, _, a)), Some (Some (_, _, b)) when a <> b ->
        refuse "the datum's identifier and its system's disagree"
    | Some (Some (_, _, x)), _ | None, Some (Some (_, _, x)) -> Some x
    | Some None, _ | None, Some None ->
        refuse "the datum is not WGS 84, NAD83, EUREF-FIN, NZGD2000 or GDA94"
    | None, None -> None
  in
  match datum with
  | Some x when a = semi_major && rf = flattening_of x -> `Datum x
  | Some _ -> refuse "the ellipsoid is not the datum's"
  | None when a = modis_radius && rf = 0. -> `Modis
  | None -> refuse "the datum has no EPSG identifier"

(* The parameters of a conversion, by EPSG identifier: each a code and its
   value in degrees, metres or unity. *)
let wkt2_parameters conv =
  List.map
    (fun p ->
      let code =
        match epsg_id p with
        | Some c -> c
        | None -> refuse "a parameter has no EPSG identifier"
      in
      let x =
        match numbers p with
        | x :: _ -> x
        | [] -> refuse "parameter %d has no value" code
      in
      let x =
        if List.mem code [ 8801; 8802; 8821; 8822; 8823; 8824 ] then
          in_degrees (optional angle_units p) x
        else if code = 8805 then
          in_units ~ids:[ 9201 ] (optional [ "SCALEUNIT"; "UNIT" ] p) x
        else in_metres (optional length_units p) x
      in
      (code, x))
    (all [ "PARAMETER" ] conv)

let wkt2_projection ~modis:sphere conv =
  let m = only [ "METHOD"; "PROJECTION" ] conv in
  let ps = wkt2_parameters conv in
  let take codes =
    let codes' = List.sort compare (List.map fst ps) in
    if codes' <> List.sort compare codes then
      refuse "the method's parameters are not EPSG's %s"
        (String.concat ", " (List.map string_of_int codes));
    fun c -> List.assoc c ps
  in
  let named =
    match attributes m with
    | Wkt.Text n :: _ -> String.lowercase_ascii n
    | _ -> ""
  in
  match (epsg_id m, sphere) with
  | Some 9807, false ->
      let p = take [ 8801; 8802; 8805; 8806; 8807 ] in
      transverse_mercator ~lat0:(p 8801) ~lon0:(p 8802) ~k0:(p 8805)
        ~false_easting:(p 8806) ~false_northing:(p 8807)
  | Some 9822, false ->
      let p = take [ 8821; 8822; 8823; 8824; 8826; 8827 ] in
      albers ~lat0:(p 8821) ~lon0:(p 8822) ~lat1:(p 8823) ~lat2:(p 8824)
        ~false_easting:(p 8826) ~false_northing:(p 8827)
  | Some 1024, false ->
      let p = take [ 8801; 8802; 8806; 8807 ] in
      if p 8801 <> 0. then refuse "Pseudo-Mercator's parameter 8801 is not 0";
      pseudo_mercator ~lon0:(p 8802) ~false_easting:(p 8806)
        ~false_northing:(p 8807)
  | None, true when named = "sinusoidal" ->
      let p = take [ 8802; 8806; 8807 ] in
      sinusoidal ~lon0:(p 8802) ~radius:modis_radius ~false_easting:(p 8806)
        ~false_northing:(p 8807)
  | Some c, _ -> refuse "method %d is not 9807, 9822 or 1024" c
  | None, _ -> refuse "the method %S has no EPSG identifier" named

(* Whether each unit of [t]'s coordinate system, and of the system, is
   [ok], the CS being of [kind] and two axes. *)
let wkt2_cs t kind units ok =
  let cs = only [ "CS" ] t in
  (match attributes cs with
  | Wkt.Word k :: Wkt.Number 2. :: _ when String.lowercase_ascii k = kind -> ()
  | _ -> refuse "the coordinate system is not %s and 2D" kind);
  let axes = all [ "AXIS" ] t in
  if List.length axes <> 2 then refuse "the system has not two axes";
  List.iter
    (fun u ->
      match unit ~ids:(fst ok) ~base:(snd ok) u with
      | None -> ()
      | Some _ -> refuse "an axis's unit is not the %s" (List.hd units))
    (List.concat_map (all units) (t :: axes))

let wkt2 t =
  match keyword t with
  | "GEOGCRS" | "GEOGRAPHICCRS" | "GEODCRS" | "GEODETICCRS" -> (
      wkt2_cs t "ellipsoidal" angle_units ([ 9102; 9122 ], printed_degree);
      match wkt2_datum t with
      | `Datum datum -> { datum; projection = None }
      | `Modis -> refuse "MODIS's sphere is not a geographic system")
  | "PROJCRS" | "PROJECTEDCRS" ->
      let base = only [ "BASEGEOGCRS"; "BASEGEODCRS" ] t in
      wkt2_cs t "cartesian" length_units ([ 9001 ], 1.);
      let datum, sphere =
        match wkt2_datum base with
        | `Datum d -> (d, false)
        | `Modis -> (Wgs84, true)
      in
      let conv = only [ "CONVERSION" ] t in
      { datum; projection = Some (wkt2_projection ~modis:sphere conv) }
  | k -> refuse "WKT %s is not a GEOGCRS or a PROJCRS" k

let of_wkt s =
  match Wkt.parse s with
  | Error _ as e -> e
  | Ok t -> (
      match t with
      | Wkt.Object (key, Wkt.Text name :: _) -> (
          match String.uppercase_ascii key with
          | ("GEOGCS" | "PROJCS") as key -> wkt1 t key name
          | _ -> (
              match wkt2 t with
              | c -> Ok c
              | exception Refused e -> Error (strf "%s[%S]: %s" key name e)
              | exception Invalid_argument e ->
                  Error (strf "%s[%S]: %s" key name e)))
      | Wkt.Object (key, _) ->
          Error
            (strf
               "WKT %s is not a GEOGCS, PROJCS, GEOGCRS or PROJCRS with a \
                name"
               (String.uppercase_ascii key))
      | _ -> Error "WKT text is not a coordinate reference system")

(* Transverse Mercator *)

(* The domain's reach from the central meridian. *)
let max_offset = 40.

(* How far along its parallel a point may lie past [max_offset]: twice the
   transforms' stated error, the most a round trip moves a point, so that a
   point of the boundary, or a pole, where every longitude is one point, passes
   both ways. *)
let boundary_slack = 12e-9
let deg = Float.pi /. 180.

(* [series n j cs] is n^j (c_j + c_(j+1) n + ... ), cs from the n^j term up, in
   Horner form as Karney advises. Each step is one fused multiply-add and the
   power a product, so every system computes the same bits (OCaml's arm64
   backend would fuse [c +. n *. acc] on its own, and libm's [**] varies). *)
let series n j cs =
  let h = List.fold_right (fun c acc -> Float.fma n acc c) cs 0. in
  let rec pow k = if k = 0 then 1. else n *. pow (k - 1) in
  h *. pow j

(* Karney eq. (35), α1..α6. *)
let alpha n =
  [
    series n 1
      [
        1. /. 2.;
        -2. /. 3.;
        5. /. 16.;
        41. /. 180.;
        -127. /. 288.;
        7891. /. 37800.;
      ];
    series n 2
      [
        13. /. 48.;
        -3. /. 5.;
        557. /. 1440.;
        281. /. 630.;
        -1983433. /. 1935360.;
      ];
    series n 3
      [ 61. /. 240.; -103. /. 140.; 15061. /. 26880.; 167603. /. 181440. ];
    series n 4 [ 49561. /. 161280.; -179. /. 168.; 6601661. /. 7257600. ];
    series n 5 [ 34729. /. 80640.; -3418889. /. 1995840. ];
    series n 6 [ 212378941. /. 319334400. ];
  ]

(* Karney eq. (36), β1..β6. *)
let beta n =
  [
    series n 1
      [
        1. /. 2.;
        -2. /. 3.;
        37. /. 96.;
        -1. /. 360.;
        -81. /. 512.;
        96199. /. 604800.;
      ];
    series n 2
      [
        1. /. 48.; 1. /. 15.; -437. /. 1440.; 46. /. 105.; -1118711. /. 3870720.;
      ];
    series n 3 [ 17. /. 480.; -37. /. 840.; -209. /. 4480.; 5569. /. 90720. ];
    series n 4 [ 4397. /. 161280.; -11. /. 504.; -830251. /. 7257600. ];
    series n 5 [ 4583. /. 161280.; -108847. /. 3991680. ];
    series n 6 [ 20648693. /. 638668800. ];
  ]

(* An ellipsoid's numbers: its flattening f, third flattening n, e and e²,
   and for Krüger's series A (metres per radian of ξ and η) as the
   unevaluated sum a_hi + a_lo, and α and β. *)
type ellipsoid = {
  f : float;
  n : float;
  e : float;
  e2 : float;
  a_hi : float;
  a_lo : float;
  alpha : float list;
  beta : float list;
}

let ellipsoid ~flattening:f ~a_hi ~a_lo =
  let n = f /. (2. -. f) in
  let e2 = f *. (2. -. f) in
  { f; n; e = Float.sqrt e2; e2; a_hi; a_lo; alpha = alpha n; beta = beta n }

(* A is Karney eq. (29) to n¹⁰ for each ellipsoid, in exact rational
   arithmetic from the double flattening, held as a double-double: its
   rounding, then the rounding of the rest. Evaluating eq. (29) in float64
   lands 2 ulps away, which moves a northing near the pole by 2 nm. GRS80's
   flattening is EPSG's (ellipsoid 7019). *)
let wgs84 =
  ellipsoid ~flattening:(1. /. 298.257223563) ~a_hi:0x1.84a3649552bbcp+22
    ~a_lo:(-0x1.0282d4366c880p-34)

let grs80 =
  ellipsoid ~flattening:(1. /. 298.257222101) ~a_hi:0x1.84a3649545016p+22
    ~a_lo:0x1.d56b7a30a1d15p-32

let ellipsoid_of = function
  | Wgs84 -> wgs84
  | Nad83 | Euref_fin | Nzgd2000 | Gda94 -> grs80

let hypot1 x = Nx.sqrt (Nx.add_s (Nx.square x) 1.)

(* τ' = tan φ' from τ = tan φ, Karney eqs. (7)-(9). *)
let taup ell tau =
  let e = ell.e in
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

(* A projection's transverse Mercator: its ellipsoid, origin, k0 A, and false
   easting and northing. *)
type tm = {
  ell : ellipsoid;
  lat0 : float;
  lon0 : float;
  k0_a : float;
  false_easting : float;
  false_northing : float;
}

(* k0 A is the product of the k0 held and A rounded once: the fused
   k0 a_hi + (k0 a_lo), whose second product and a_lo err by under 2⁻¹⁰⁰ of
   the whole, so it is the correctly rounded k0 A unless that lies so near a
   midpoint between floats. It is for 0.9996 and 10⁵ random k0 in
   [0.5, 1.5] on both ellipsoids. *)
let tm ell ~lat0 ~lon0 ~k0 ~false_easting ~false_northing =
  let k0_a = Float.fma k0 ell.a_hi (k0 *. ell.a_lo) in
  { ell; lat0; lon0; k0_a; false_easting; false_northing }

(* [d] reduced to [-180, 180). *)
let wrap d =
  Nx.sub d (Nx.mul_s (Nx.floor (Nx.div_s (Nx.add_s d 180.) 360.)) 360.)

(* The easting and northing of each point (lon, lat) from the central
   meridian's crossing of the equator, k0 A η and k0 A ξ, Karney eqs.
   (7)-(13). *)
let karney tm lon lat =
  let lam = Nx.mul_s (wrap (Nx.sub_s lon tm.lon0)) deg in
  let tp = taup tm.ell (Nx.tan (Nx.mul_s lat deg)) in
  let cl = Nx.cos lam in
  let xi' = Nx.atan2 tp cl in
  let eta' = Nx.asinh (Nx.div (Nx.sin lam) (Nx.hypot tp cl)) in
  let sr, si = clenshaw tm.ell.alpha xi' eta' in
  (Nx.mul_s (Nx.add eta' si) tm.k0_a, Nx.mul_s (Nx.add xi' sr) tm.k0_a)

(* The northing of the equator's image, beside [like]: the false northing,
   less the natural origin's own northing, which the forward map computes. *)
let northing tm like =
  let fn = Nx.scalar_like like tm.false_northing in
  if tm.lat0 = 0. then fn
  else
    let at v = Nx.scalar_like like v in
    Nx.sub fn (snd (karney tm (at tm.lon0) (at tm.lat0)))

let tm_forward tm lon lat =
  let x, y = karney tm lon lat in
  (Nx.add_s x tm.false_easting, Nx.add y (northing tm lat))

(* The longitude, latitude and longitude from the central meridian of each point
   (x, y), and its ξ, Karney eqs. (15)-(22) with two Newton steps. The series
   are odd in ξ and η, so the point is taken to the first quadrant and a point
   past the pole (ξ > π/2) folded back, as GeographicLib's Reverse does. At the
   pole cos ξ' can round below 0, and is clamped. *)
let tm_lonlat tm x y =
  let ell = tm.ell in
  let xi0 = Nx.div_s (Nx.sub y (northing tm y)) tm.k0_a in
  let eta0 = Nx.div_s (Nx.sub_s x tm.false_easting) tm.k0_a in
  let back = Nx.greater_s (Nx.abs xi0) (Float.pi /. 2.) in
  let xi = Nx.where back (Nx.rsub_s Float.pi (Nx.abs xi0)) (Nx.abs xi0) in
  let eta = Nx.abs eta0 in
  let sr, si = clenshaw ell.beta xi eta in
  let xi' = Nx.sub xi sr and eta' = Nx.sub eta si in
  let s = Nx.sinh eta' and c = Nx.maximum_s (Nx.cos xi') 0. in
  let r = Nx.hypot s c in
  let pole = Nx.equal_s r 0. in
  let tp = Nx.div (Nx.sin xi') r in
  let newton tau =
    let tpi = taup ell tau in
    let num = Nx.add_s (Nx.mul_s (Nx.square tau) (1. -. ell.e2)) 1. in
    let den = Nx.mul_s (hypot1 tau) (1. -. ell.e2) in
    Nx.add tau (Nx.mul (Nx.div (Nx.sub tp tpi) (hypot1 tpi)) (Nx.div num den))
  in
  (* Where r is 0 the point is the pole, latitude 90, longitude 0. *)
  let lat = Nx.div_s (Nx.atan (newton (newton tp))) deg in
  let lat = Nx.where pole (Nx.full_like lat 90.) lat in
  let lam = Nx.where pole (Nx.zeros_like s) (Nx.div_s (Nx.atan2 s c) deg) in
  let lam = Nx.where back (Nx.rsub_s 180. lam) lam in
  let lat = Nx.where (Nx.less_s xi0 0.) (Nx.neg lat) lat in
  let dlon = Nx.where (Nx.less_s eta0 0.) (Nx.neg lam) lam in
  let lon = Nx.add_s dlon tm.lon0 in
  let lon =
    Nx.where (Nx.greater_s lon 180.) (Nx.sub_s lon 360.)
      (Nx.where (Nx.less_s lon (-180.)) (Nx.add_s lon 360.) lon)
  in
  (lon, lat, dlon, xi0)

(* Albers equal-area conic *)

(* Albers' stated error, 144 u D (crs.mli): 118 of it is the map's rounding
   bounded from nx's kernel bounds, and 26 is measured, the effect of the
   cone's constants computed in float64 with the C library's [atanh], which
   states no bound: at most 16.5 u D over 137 cones with glibc's. *)

(* GeographicLib's AlbersEqualArea (Karney, src/AlbersEqualArea.cpp at
   github.com/geographiclib/geographiclib 48959dfc, Init and Forward, lines
   123-282 and 496-519): the two standard parallels become one tangent
   latitude φ₀ and a scale k₀ on it, by divided differences (Kahan and
   Fateman, "Symbolic computation of divided differences", 1999), so that n is
   never a difference of near numbers. A southern cone is the northern one
   reflected in the equator: [sign] is -1 for it, and multiplies each latitude
   once on the way in (the fetched Forward multiplies it twice, line 501,
   which puts a southern cone's points 8190 km from PROJ's) and each northing
   on the way out. Init's polar branch keys on the first parallel given and
   drops the other, so that (90, 60) comes out azimuthal, 4500 km from PROJ;
   after the sort a pole is always the second parallel, which the general
   branch takes, so the port has no polar branch. The constants are computed
   in floats at each call, as Init computes them; the map in nx. *)

let sq x = x *. x
let hyp x = Float.hypot 1. x
let epsx = sq epsilon_float
let epsx2 = sq epsx
let tol0 = Float.sqrt epsilon_float *. Float.sqrt (Float.sqrt epsilon_float)
let lg2eps = 53.

(* sin and cos of [d] degrees, exact at multiples of 90°: Math::sincosd. *)
let sincosd d =
  let r = Float.rem d 360. in
  let q = Float.round (r /. 90.) in
  let r = (r -. (90. *. q)) *. deg in
  let s = Float.sin r and c = Float.cos r in
  match int_of_float q land 3 with
  | 0 -> (s, c)
  | 1 -> (c, -.s)
  | 2 -> (-.s, -.c)
  | _ -> (-.c, s)

(* An ellipsoid's numbers for the cone: e, e², 1 - e², 1 - f and q at the
   pole, q_Z = 1 + (1 - e²) atanh(e)/e. *)
type shape = { e : float; e2 : float; e2m : float; fm : float; qz : float }

let shape (ell : ellipsoid) =
  let e2m = 1. -. ell.e2 in
  {
    e = ell.e;
    e2 = ell.e2;
    e2m;
    fm = 1. -. ell.f;
    qz = 1. +. (e2m *. (Float.atanh ell.e /. ell.e));
  }

let atanhee s x = Float.atanh (s.e *. x) /. s.e

(* (sn x - sn y) / (x - y), sn x = x / √(1 + x²), given sx = sn x. *)
let dsn x y sx sy =
  let t = x *. y in
  if t > 0. then (x +. y) *. sq (sx *. sy /. t) /. (sx +. sy)
  else if x -. y <> 0. then (sx -. sy) /. (x -. y)
  else 1.

(* (atanhee x - atanhee y) / (x - y). *)
let datanhee s x y =
  let t = x -. y and d = 1. -. (s.e2 *. x *. y) in
  if t = 0. then 1. /. d
  else
    (if x *. y < 0. then atanhee s x -. atanhee s y else atanhee s (t /. d))
    /. t

(* atanh √x / √x - 1, accurate for small x. *)
let atanhxm1 x =
  if Float.abs x < 0.5 then begin
    let _, e = Float.frexp x in
    let n =
      if x = 0. then 1 else int_of_float (Float.ceil (lg2eps /. float (-e))) + 1
    in
    let s = ref 0. in
    for k = n - 1 downto 0 do
      s := (x *. !s) +. ((if k > 0 then 1. else 0.) /. float ((2 * k) + 1))
    done;
    !s
  end
  else
    let xs = Float.sqrt (Float.abs x) in
    ((if x > 0. then Float.atanh xs else Float.atan xs) /. xs) -. 1.

(* (datanhee 1 y - datanhee 1 x) / (y - x), by one of three forms. *)
let ddatanhee s x y =
  let x, y = if y < x then (y, x) else (x, y) in
  let q1 = Float.abs s.e2 in
  let q2 = Float.abs (2. *. s.e /. s.e2m *. (1. -. x)) in
  if x <= 0. || not (Float.min q1 q2 < 0.75) then
    (datanhee s 1. y -. datanhee s x y) /. (1. -. x)
  else if q1 < q2 then begin
    (* The series in e². *)
    let s' = ref 0. and z = ref 1. and k = ref 1. and t = ref 0. in
    let c = ref 0. and en = ref 1. and continue = ref true in
    while !continue do
      for _ = 1 to 2 do
        t := (y *. !t) +. !z;
        c := !c +. !t;
        z := !z *. x
      done;
      k := !k +. 2.;
      en := !en *. s.e2;
      let ds = !en *. !c /. !k in
      s' := !s' +. ds;
      continue := Float.abs ds > Float.abs !s' *. epsilon_float /. 2.
    done;
    !s'
  end
  else begin
    (* The series in 1 - x and 1 - y. *)
    let dx = 1. -. x and dy = 1. -. y in
    let ee = ref (s.e2 /. sq s.e2m) in
    let s' = ref !ee and xy = ref 1. and yy = ref 1. and m = ref 1 in
    let continue = ref true in
    while !continue do
      let m' = !m in
      let c = ref (float (m' + 2)) in
      let t = ref !c in
      yy := !yy *. dy;
      xy := (dx *. !xy) +. !yy;
      ee := !ee /. -.s.e2m;
      if m' mod 2 = 0 then ee := !ee *. s.e2;
      let kmax = (m' + 1) / 2 in
      for k = kmax - 1 downto 0 do
        c := !c *. float ((k + 1) * ((2 * (k + m' - (2 * kmax))) + 3));
        c := !c /. float ((kmax - k) * ((2 * (kmax - k)) + 1));
        t := (s.e2 *. !t) +. !c
      done;
      let ds = !t *. !ee *. !xy /. float (m' + 2) in
      s' := !s' +. ds;
      continue := Float.abs ds > Float.abs !s' *. epsilon_float /. 2.;
      incr m
    done;
    !s'
  end

(* tan ξ, ξ the authalic latitude, of tan φ. *)
let txif s tphi =
  let cphi = 1. /. Float.sqrt (1. +. sq tphi) in
  let sphi = tphi *. cphi in
  let es1 = s.e2 *. sphi in
  let es2m1 = 1. -. (es1 *. sphi) in
  let es2m1a = s.e2m *. es2m1 in
  ((tphi /. es2m1) +. (atanhee s sphi /. cphi))
  /. Float.sqrt
       (((1. +. es1) /. es2m1a +. datanhee s 1. sphi)
       *. ((1. -. es1) /. es2m1a +. datanhee s 1. (-.sphi)))

(* A cone: its shape, the tangent latitude's sin n0, cos² of its parametric
   latitude m02, a √m02 ([nrho0]), tan, sin and sec of its
   authalic latitude, the scale k0 on it and k0², and [sign]. *)
type cone = {
  s : shape;
  sign : float;
  n0 : float;
  m02 : float;
  nrho0 : float;
  txi0 : float;
  sxi0 : float;
  scxi0 : float;
  k0 : float;
  k2 : float;
}

(* GeographicLib's Init, with k1 = 1. *)
let cone ell lat1 lat2 =
  let s = shape ell in
  let { e2; e2m; fm; qz; _ } = s in
  let qx = qz /. (2. *. e2m) in
  let unit (s, c) =
    let r = Float.hypot s c in
    (s /. r, c /. r)
  in
  let sphi1, cphi1 = unit (sincosd lat1) in
  let sphi2, cphi2 = unit (sincosd lat2) in
  let cphi1 = Float.max epsx cphi1 and cphi2 = Float.max epsx cphi2 in
  let sign = if sphi1 +. sphi2 >= 0. then 1. else -1. in
  let sphi1 = sphi1 *. sign and sphi2 = sphi2 *. sign in
  let sphi1, cphi1, sphi2, cphi2 =
    if sphi1 > sphi2 then (sphi2, cphi2, sphi1, cphi1)
    else (sphi1, cphi1, sphi2, cphi2)
  in
  let tphi1 = sphi1 /. cphi1 and tphi2 = sphi2 /. cphi2 in
  let tphi0, c =
    if tphi1 = tphi2 then (tphi2, 1.)
    else
      let tbet1 = fm *. tphi1 and tbet2 = fm *. tphi2 in
      let scbet12 = 1. +. sq tbet1 and scbet22 = 1. +. sq tbet2 in
      let txi1 = txif s tphi1 and txi2 = txif s tphi2 in
      let cxi1 = 1. /. hyp txi1 and cxi2 = 1. /. hyp txi2 in
      let sxi1 = txi1 *. cxi1 and sxi2 = txi2 *. cxi2 in
      let dtbet2 = fm *. (tbet1 +. tbet2) in
      let es1 = 1. -. (e2 *. sq sphi1) and es2 = 1. -. (e2 *. sq sphi2) in
      let dsn12 = dsn tphi2 tphi1 sphi2 sphi1 in
      let dsxi =
        ((1. +. (e2 *. sphi1 *. sphi2)) /. (es2 *. es1)
        +. datanhee s sphi2 sphi1)
        *. dsn12 /. (2. *. qx)
      in
      let den = ((sxi2 +. sxi1) *. dtbet2) +. ((scbet22 +. scbet12) *. dsxi) in
      let sn = 2. *. dtbet2 /. den in
      (* (1 - sxi) / (1 - sphi) and 1 - sphi, without cancellation. *)
      let ratio sphi cphi sxi cxi =
        if sphi <= 0. then (1. -. sxi) /. (1. -. sphi)
        else sq (cxi /. cphi) *. (1. +. sphi) /. (1. +. sxi)
      in
      let less sphi cphi =
        if sphi <= 0. then 1. -. sphi else sq cphi /. (1. +. sphi)
      in
      let sum = sphi1 +. sphi2 +. (sphi1 *. sphi2) in
      let sm1 =
        -.dsn12
        *. ((-.(ratio sphi2 cphi2 sxi2 cxi2 +. ratio sphi1 cphi1 sxi1 cxi1)
            *. (1. +. (e2 *. sum))
            /. (1. +. sum))
           +. (((scbet22 *. less sphi2 cphi2) +. (scbet12 *. less sphi1 cphi1))
              *. ((e2 *. (1. +. sphi1 +. sphi2 +. (e2 *. sphi1 *. sphi2))
                  /. (es1 *. es2))
                 +. (e2m *. ddatanhee s sphi1 sphi2))
              /. qz))
        /. den
      in
      let c = den /. (2. *. scbet12 *. scbet22 *. dsxi) in
      let stol = tol0 *. Float.max 1. (Float.abs ((tphi2 +. tphi1) /. 2.)) in
      (* Newton's method on w(tan φ₀) = 0, at most 40 steps: s / (1 - s)
         as GeographicLib writes it. *)
      let rec newton i tphi0 =
        let scphi02 = 1. +. sq tphi0 in
        let scphi0 = Float.sqrt scphi02 in
        let sphi0 = tphi0 /. scphi0 in
        let sphi0m = 1. /. (scphi0 *. (tphi0 +. scphi0)) in
        let g = (1. +. sq (fm *. tphi0)) *. sphi0 in
        let dg = (e2m *. scphi02 *. (1. +. (2. *. sq tphi0))) +. e2 in
        let d =
          sphi0m
          *. (1. -. (e2 *. (1. +. (2. *. sphi0 *. (1. +. sphi0)))))
          /. (e2m *. (1. +. sphi0))
        in
        let dd =
          -2.
          *. (1. -. (e2 *. sq sphi0 *. ((2. *. sphi0) +. 3.)))
          /. (e2m *. sq (1. +. sphi0))
        in
        let a =
          -.e2 *. sq sphi0m
          *. (2. +. ((1. +. e2) *. sphi0))
          /. (e2m *. (1. -. (e2 *. sq sphi0)))
        in
        let b =
          sphi0m *. e2m
          /. (1. -. (e2 *. sphi0))
          *. (atanhxm1 (e2 *. sq (sphi0m /. (1. -. (e2 *. sphi0))))
             -. (e2 *. sphi0m /. e2m))
        in
        let dab =
          2. *. e2
          *. (2. -. (e2 *. (1. +. sq sphi0)))
          /. (e2m *. sq (1. -. (e2 *. sq sphi0)) *. scphi02)
        in
        let u = (sm1 *. g) -. (sn /. qz *. (d -. (g *. (a +. b)))) in
        let du =
          (sm1 *. dg) -. (sn /. qz *. (dd -. (dg *. (a +. b)) -. (g *. dab)))
        in
        let dtu = -.u /. du *. (scphi0 *. scphi02) in
        let tphi0 = tphi0 +. dtu in
        if Float.abs dtu >= stol && i < 39 then newton (i + 1) tphi0 else tphi0
      in
      (newton 0 ((tphi2 +. tphi1) /. 2.), c)
  in
  let txi0 = txif s tphi0 in
  let scxi0 = hyp txi0 in
  let n0 = tphi0 /. hyp tphi0 in
  let m02 = 1. /. (1. +. sq (fm *. tphi0)) in
  let sxi0 = txi0 /. scxi0 in
  let k0 =
    Float.sqrt (if tphi1 = tphi2 then 1. else c /. (m02 +. (n0 *. qz *. sxi0)))
  in
  {
    s;
    sign;
    n0;
    m02;
    nrho0 = semi_major *. Float.sqrt m02;
    txi0;
    sxi0;
    scxi0;
    k0;
    k2 = sq k0;
  }

(* ρ - ρ₀ for the point of sin and cos [sphi], [cphi] on the internal cone,
   in floats: GeographicLib's Forward. √(m02 - n0 dq) is ρ n0 / a, 0 at a
   polar apex, where it is the square root of a rounding residue that may be
   negative, so it is clamped there, as on tensors. *)
let drho_at c sphi cphi =
  let cphi = Float.max epsx cphi in
  let txi = txif c.s (sphi /. cphi) in
  let sxi = txi /. hyp txi in
  let dq = c.s.qz *. dsn txi c.txi0 sxi c.sxi0 *. (txi -. c.txi0) in
  -.semi_major *. dq
  /. (Float.sqrt (Float.max 0. (c.m02 -. (c.n0 *. dq)))
     +. (c.nrho0 /. semi_major))

(* Karney, "On auxiliary latitudes" (Survey Review, 2024; arXiv 2212.05818),
   eq. (A20): φ from the authalic latitude ξ, φ = ξ + Σ c_l sin 2lξ, each c_l
   a polynomial in n to n⁶. The extracted text drops the sign of row 5's
   n⁶ term; the Fourier coefficients of φ(ξ) - ξ show it positive. *)
let phi_of_xi n =
  [
    series n 1
      [
        4. /. 3.;
        4. /. 45.;
        -16. /. 35.;
        -2582. /. 14175.;
        60136. /. 467775.;
        28112932. /. 212837625.;
      ];
    series n 2
      [
        46. /. 45.;
        152. /. 945.;
        -11966. /. 14175.;
        -21016. /. 51975.;
        251310128. /. 638512875.;
      ];
    series n 3
      [
        3044. /. 2835.;
        3802. /. 14175.;
        -94388. /. 66825.;
        -8797648. /. 10945935.;
      ];
    series n 4 [ 6059. /. 4725.; 41072. /. 93555.; -1472637812. /. 638512875. ];
    series n 5 [ 768272. /. 467775.; 455935736. /. 638512875. ];
    series n 6 [ 4210684958. /. 1915538625. ];
  ]

(* Σ c_l sin 2lξ by Clenshaw's recurrence y_l = 2 cos 2ξ · y_(l+1) - y_(l+2) +
   c_l and the sum sin 2ξ · y_1. *)
let sines cs xi =
  let c2 = Nx.mul_s (Nx.cos (Nx.mul_s xi 2.)) 2. in
  let zero = Nx.zeros_like xi in
  let y1, _ =
    List.fold_right
      (fun c (y1, y2) -> (Nx.add_s (Nx.sub (Nx.mul c2 y1) y2) c, y1))
      cs (zero, zero)
  in
  Nx.mul (Nx.sin (Nx.mul_s xi 2.)) y1

(* The cone on tensors *)

let atanhee_t c x = Nx.div_s (Nx.atanh (Nx.mul_s x c.s.e)) c.s.e

(* [datanhee 1 y] of each element. *)
let datanhee1 c y =
  let t = Nx.rsub_s 1. y and d = Nx.rsub_s 1. (Nx.mul_s y c.s.e2) in
  let zero = Nx.equal_s t 0. in
  let t = Nx.where zero (Nx.ones_like t) t in
  let num =
    Nx.where (Nx.less_s y 0.)
      (Nx.rsub_s (atanhee c.s 1.) (atanhee_t c y))
      (atanhee_t c (Nx.div t d))
  in
  Nx.where zero (Nx.rdiv_s 1. d) (Nx.div num t)

(* [txif] of each element. *)
let txif_t c tphi =
  let s = c.s in
  let cphi = Nx.rdiv_s 1. (hypot1 tphi) in
  let sphi = Nx.mul tphi cphi in
  let es1 = Nx.mul_s sphi s.e2 in
  let es2m1 = Nx.rsub_s 1. (Nx.mul es1 sphi) in
  let es2m1a = Nx.mul_s es2m1 s.e2m in
  let side es d = Nx.add (Nx.div es es2m1a) d in
  Nx.div
    (Nx.add (Nx.div tphi es2m1) (Nx.div (atanhee_t c sphi) cphi))
    (Nx.sqrt
       (Nx.mul
          (side (Nx.add_s es1 1.) (datanhee1 c sphi))
          (side (Nx.rsub_s 1. es1) (datanhee1 c (Nx.neg sphi)))))

(* [dsn x y sx sy] of each element of [x] and [sx], [y] and [sy] numbers. *)
let dsn_t x y sx sy =
  let t = Nx.mul_s x y in
  let pos = Nx.greater_s t 0. in
  let t = Nx.where pos t (Nx.ones_like t) in
  let same =
    Nx.div
      (Nx.mul (Nx.add_s x y) (Nx.square (Nx.div (Nx.mul_s sx sy) t)))
      (Nx.add_s sx sy)
  in
  let dx = Nx.sub_s x y in
  let zero = Nx.equal_s dx 0. in
  let dx = Nx.where zero (Nx.ones_like dx) dx in
  Nx.where pos same
    (Nx.where zero (Nx.ones_like dx) (Nx.div (Nx.sub_s sx sy) dx))

(* The easting and northing of each point (λ - λ₀, φ), in degrees, about the
   tangent latitude's crossing of the central meridian: GeographicLib's
   Forward, and ρ - ρ₀ of the internal cone. cos φ is 0 at a pole, as
   GeographicLib's sincosd gives it, so a pole takes Init's path. Where n₀
   is 0 the cone is the cylinder, its limit. *)
let albers_xy c dlon lat =
  let lam = Nx.mul_s dlon deg and phi = Nx.mul_s lat (c.sign *. deg) in
  let pole = Nx.equal_s (Nx.abs lat) 90. in
  let cphi = Nx.where pole (Nx.zeros_like phi) (Nx.cos phi) in
  let sphi = Nx.sin phi and cphi = Nx.maximum_s cphi epsx in
  let txi = txif_t c (Nx.div sphi cphi) in
  let sxi = Nx.div txi (hypot1 txi) in
  let dq =
    Nx.mul (Nx.mul_s (dsn_t txi c.txi0 sxi c.sxi0) c.s.qz) (Nx.sub_s txi c.txi0)
  in
  let drho =
    Nx.div
      (Nx.mul_s dq (-.semi_major))
      (Nx.add_s
         (Nx.sqrt (Nx.maximum_s (Nx.rsub_s c.m02 (Nx.mul_s dq c.n0)) 0.))
         (c.nrho0 /. semi_major))
  in
  let theta = Nx.mul_s lam (c.k2 *. c.n0) in
  let st = Nx.sin theta and ct = Nx.cos theta in
  let t = Nx.add_s (Nx.mul_s drho c.n0) c.nrho0 in
  let cylinder = c.n0 = 0. in
  let arc = if cylinder then Nx.mul_s lam c.k2 else Nx.div_s st c.n0 in
  let x = Nx.div_s (Nx.mul t arc) c.k0 in
  (* 1 - cos θ, without cancellation. *)
  let vers =
    Nx.where (Nx.less_s ct 0.) (Nx.rsub_s 1. ct)
      (Nx.div (Nx.square st) (Nx.add_s ct 1.))
  in
  let bend =
    if cylinder then Nx.zeros_like vers
    else Nx.mul_s (Nx.div_s vers c.n0) c.nrho0
  in
  let y = Nx.div_s (Nx.sub bend (Nx.mul drho ct)) c.k0 in
  (x, Nx.mul_s y c.sign)

(* The longitude from λ₀ and the latitude of each point (x, y) about the
   tangent origin, its ρ - ρ₀, and its radius times n₀ k₀ (the tangent
   parallel's a √m02 on the cylinder): GeographicLib's Reverse to tan ξ, then
   {!phi_of_xi}. *)
let albers_lonlat c phis x y =
  let y = Nx.mul_s y c.sign in
  let nk = c.k0 *. c.n0 in
  let nx = Nx.mul_s x nk and ny = Nx.mul_s y nk in
  let y1 = Nx.rsub_s c.nrho0 ny in
  let r = Nx.hypot nx y1 in
  let den = Nx.add_s r c.nrho0 in
  (* A zero [den] is a polar origin's apex. *)
  let apex = Nx.equal_s den 0. in
  let den = Nx.where apex (Nx.ones_like den) den in
  let num =
    Nx.add
      (Nx.sub
         (Nx.mul (Nx.mul_s x c.k0) nx)
         (Nx.mul_s (Nx.mul_s y (2. *. c.k0)) c.nrho0))
      (Nx.mul (Nx.mul_s y c.k0) ny)
  in
  let drho = Nx.where apex (Nx.zeros_like num) (Nx.div num den) in
  let dsxia =
    Nx.div_s
      (Nx.mul
         (Nx.mul_s (Nx.add_s (Nx.mul_s drho c.n0) (2. *. c.nrho0)) (-.c.scxi0))
         drho)
      (sq semi_major *. c.s.qz)
  in
  let w = Nx.rsub_s 1. (Nx.mul dsxia (Nx.add_s dsxia (2. *. c.txi0))) in
  let txi = Nx.div (Nx.add_s dsxia c.txi0) (Nx.sqrt (Nx.maximum_s w epsx2)) in
  let xi = Nx.atan txi in
  let phi = Nx.add xi (sines phis xi) in
  let lat = Nx.mul_s phi (c.sign /. deg) in
  let dlon =
    if c.n0 = 0. then Nx.div_s (Nx.div x y1) (c.k0 *. deg)
    else Nx.div_s (Nx.atan2 nx y1) (c.k2 *. c.n0 *. deg)
  in
  (dlon, lat, drho, r)

(* Projections *)

(* An Albers projection's cone, φ(ξ)'s coefficients, central meridian, false
   easting, the northing of the tangent origin (the false northing less the
   false origin's own), ρ - ρ₀ at each pole of the internal cone, and how far
   a plane point may lie outside the image: twice the stated error. *)
type albers = {
  cone : cone;
  phis : float list;
  lon0 : float;
  easting : float;
  northing : float;
  pole_north : float;
  pole_south : float;
  slack : float;
}

(* Albers' stated error is [albers_c] u D, D the extent of the image. *)
let albers_c = 144.

(* D, the largest distance of an image point from the tangent origin's image:
   at an end of a pole's arc, (ρ - ρ₀)² + 4 ρ ρ₀ sin²(θ/2) from the law of
   cosines, written without cancellation since ρ₀ grows without bound as the
   parallels near opposite latitudes. *)
let extent c ~pole_north ~pole_south =
  (* 2 ρ₀ sin(θ/2) at the seam, π k₀ a √m02 on the cylinder. *)
  let chord =
    if c.n0 = 0. then Float.pi *. c.k0 *. c.nrho0
    else
      let rho0 = c.nrho0 /. (c.n0 *. c.k0) in
      let theta = Float.min Float.pi (Float.pi *. c.k2 *. c.n0) in
      2. *. rho0 *. Float.sin (theta /. 2.)
  in
  let far drho =
    let d = drho /. c.k0 in
    Float.sqrt (sq d +. ((1. +. (d *. c.n0 *. c.k0 /. c.nrho0)) *. sq chord))
  in
  Float.max (far pole_north) (far pole_south)

let albers_of ell ~lat0 ~lon0 ~lat1 ~lat2 ~false_easting ~false_northing =
  let c = cone ell lat1 lat2 in
  let s0, c0 = sincosd (lat0 *. c.sign) in
  let origin = c.sign *. (-.drho_at c s0 c0 /. c.k0) in
  let pole_north = drho_at c 1. 0. and pole_south = drho_at c (-1.) 0. in
  let d = extent c ~pole_north ~pole_south in
  {
    cone = c;
    phis = phi_of_xi ell.n;
    lon0;
    easting = false_easting;
    northing = false_northing -. origin;
    pole_north;
    pole_south;
    slack = 2. *. albers_c *. (epsilon_float /. 2.) *. d;
  }

let albers_forward m lon lat =
  let x, y = albers_xy m.cone (wrap (Nx.sub_s lon m.lon0)) lat in
  (Nx.add_s x m.easting, Nx.add_s y m.northing)

(* [d] in [-180, 180]: a longitude past either by a turn at most. *)
let turn d =
  Nx.where (Nx.greater_s d 180.) (Nx.sub_s d 360.)
    (Nx.where (Nx.less_s d (-180.)) (Nx.add_s d 360.) d)

(* The inverse, and where a point is an image within the slack: ρ - ρ₀
   between its values at the poles, and θ within the seam's, each along its
   own direction in the plane. A point past the seam within the slack takes
   the seam's longitude. *)
let albers_inverse m x y =
  let c = m.cone in
  let dlon, lat, drho, r =
    albers_lonlat c m.phis (Nx.sub_s x m.easting) (Nx.sub_s y m.northing)
  in
  let slack = m.slack *. c.k0 in
  let radial =
    Nx.logical_and
      (Nx.greater_equal_s drho (m.pole_north -. slack))
      (Nx.less_equal_s drho (m.pole_south +. slack))
  in
  (* The plane's metres past the seam along the parallel, k₀ r (|λ| - π). *)
  let past =
    Nx.mul (Nx.sub_s (Nx.abs (Nx.mul_s dlon deg)) Float.pi) (Nx.mul_s r c.k0)
  in
  let ok = Nx.logical_and radial (Nx.less_equal_s past m.slack) in
  let dlon = Nx.clamp ~min:(-180.) ~max:180. dlon in
  (turn (Nx.add_s dlon m.lon0), lat, ok)

(* Pseudo-Mercator and Sinusoidal, GN 7-2 §3.2.1.2 and Snyder's eqs. (30-1),
   (30-2), (30-6) and (30-7): each a sphere of radius [r] (the datum's
   semi-major axis for Pseudo-Mercator, as method 1024 states), its central
   meridian and false easting and northing. *)
type sphere = {
  r : float;
  lon0 : float;
  easting : float;
  northing : float;
}

(* Pseudo-Mercator's latitude limit, GN 7-2 §3.2.1.2: "should not be used
   poleward of 88º". *)
let mercator_limit = 88.

(* Their stated errors, and how far a point may lie past a limit: twice
   those. *)
let mercator_eps = 10e-9
let sinusoidal_eps = 8e-9

let mercator_forward m lon lat =
  let x = Nx.mul_s (wrap (Nx.sub_s lon m.lon0)) (deg *. m.r) in
  let y = Nx.mul_s (Nx.asinh (Nx.tan (Nx.mul_s lat deg))) m.r in
  (Nx.add_s x m.easting, Nx.add_s y m.northing)

(* The longitude from [lon0] of a plane offset [x] along a parallel of
   radius [r], within 180° when [x] lies past the seam, ±[r] π, by under
   [slack], and where it does. *)
let along ~slack r x =
  let ok = Nx.less_equal_s (Nx.sub_s (Nx.abs x) (Float.pi *. r)) slack in
  (Nx.clamp ~min:(-180.) ~max:180. (Nx.div_s x (r *. deg)), ok)

let mercator_inverse m x y =
  let slack = 2. *. mercator_eps in
  let dlon, seam = along ~slack m.r (Nx.sub_s x m.easting) in
  let lat =
    Nx.div_s (Nx.atan (Nx.sinh (Nx.div_s (Nx.sub_s y m.northing) m.r))) deg
  in
  let past = Nx.sub_s (Nx.abs lat) mercator_limit in
  let ok = Nx.less_equal_s (Nx.mul_s past (deg *. m.r)) slack in
  (turn (Nx.add_s dlon m.lon0), lat, Nx.logical_and seam ok)

let sinusoidal_forward m lon lat =
  let phi = Nx.mul_s lat deg in
  let lam = Nx.mul_s (wrap (Nx.sub_s lon m.lon0)) (deg *. m.r) in
  let x = Nx.mul lam (Nx.cos phi) in
  (Nx.add_s x m.easting, Nx.add_s (Nx.mul_s phi m.r) m.northing)

(* A point on a parallel whose image is shorter than the slack is the pole,
   and takes [lon0] (Snyder, p. 248). *)
let sinusoidal_inverse m x y =
  let slack = 2. *. sinusoidal_eps in
  let phi = Nx.div_s (Nx.sub_s y m.northing) m.r in
  let past = Nx.mul_s (Nx.sub_s (Nx.abs phi) (Float.pi /. 2.)) m.r in
  let ok = Nx.less_equal_s past slack in
  let phi = Nx.clamp ~min:(-.Float.pi /. 2.) ~max:(Float.pi /. 2.) phi in
  let rc = Nx.mul_s (Nx.cos phi) m.r in
  let pole = Nx.less_equal_s (Nx.mul_s rc Float.pi) slack in
  let rc = Nx.where pole (Nx.ones_like rc) rc in
  let dx = Nx.sub_s x m.easting in
  let seam =
    Nx.less_equal_s (Nx.sub (Nx.abs dx) (Nx.mul_s rc Float.pi)) slack
  in
  let dlon = Nx.clamp ~min:(-180.) ~max:180. (Nx.div_s (Nx.div dx rc) deg) in
  let dlon = Nx.where pole (Nx.zeros_like dlon) dlon in
  let ok = Nx.logical_and ok (Nx.logical_or pole seam) in
  (turn (Nx.add_s dlon m.lon0), Nx.div_s phi deg, ok)

(* A projection's map, on its datum's ellipsoid. *)
type map =
  | Tm of tm
  | Albers_map of albers
  | Mercator of sphere
  | Sinusoid of sphere

let map c =
  let ell = ellipsoid_of c.datum in
  match c.projection with
  | None -> None
  | Some (Transverse_mercator p) ->
      Some
        (Tm
           (tm ell ~lat0:p.lat0 ~lon0:p.lon0 ~k0:p.k0
              ~false_easting:p.false_easting ~false_northing:p.false_northing))
  | Some (Albers p) ->
      Some
        (Albers_map
           (albers_of ell ~lat0:p.lat0 ~lon0:p.lon0 ~lat1:p.lat1 ~lat2:p.lat2
              ~false_easting:p.false_easting ~false_northing:p.false_northing))
  | Some (Pseudo_mercator p) ->
      Some
        (Mercator
           {
             r = semi_major;
             lon0 = p.lon0;
             easting = p.false_easting;
             northing = p.false_northing;
           })
  | Some (Sinusoidal p) ->
      Some
        (Sinusoid
           {
             r = p.radius;
             lon0 = p.lon0;
             easting = p.false_easting;
             northing = p.false_northing;
           })

let forward = function
  | Tm t -> tm_forward t
  | Albers_map a -> albers_forward a
  | Mercator m -> mercator_forward m
  | Sinusoid m -> sinusoidal_forward m

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
      (* An unbatched point has no index, which [] would print as one. *)
      let at =
        if i = [||] then ""
        else
          strf " [%s]"
            (String.concat ", " (List.map string_of_int (Array.to_list i)))
      in
      let zone =
        match c.projection with
        | Some (Transverse_mercator _ as p) when is_zone p ->
            ", such as the zone Crs.utm_at gives"
        | _ -> ""
      in
      Invalid_argument
        (Format.asprintf
           "Jord.Crs.transform: point%s (%.17g, %.17g) is outside %a's \
            domain; mask such points with Crs.in_domain, or take them to a \
            system whose domain holds them%s"
           at (Nx.item [] a) (Nx.item [] b) pp c zone))

(* Whether (offset, lat) lies within [boundary_slack] of the domain, along the
   parallel. *)
let near offset lat =
  let past = Nx.sub_s (Nx.abs offset) max_offset in
  let metres =
    Nx.mul (Nx.mul_s past (deg *. semi_major)) (Nx.cos (Nx.mul_s lat deg))
  in
  Nx.less_equal_s metres boundary_slack

(* Whether each point (lon, lat) is in the domain of a projection's map [m]:
   TM's 40° along the parallel, any finite longitude otherwise. *)
let inside m lon lat =
  let valid = Nx.less_equal_s (Nx.abs lat) 90. in
  let finite = Nx.logical_and valid (Nx.isfinite lon) in
  match m with
  | Some (Tm tm) ->
      Nx.logical_and valid (near (wrap (Nx.sub_s lon tm.lon0)) lat)
  | Some (Mercator m) ->
      let past = Nx.sub_s (Nx.abs lat) mercator_limit in
      Nx.logical_and finite
        (Nx.less_equal_s (Nx.mul_s past (deg *. m.r)) (2. *. mercator_eps))
  | None | Some (Albers_map _ | Sinusoid _) -> finite

(* The longitude and latitude of each point (x, y), and where it is the image
   of a point of the domain. *)
let inverse m x y =
  match m with
  | Tm tm ->
      let lon, lat, offset, xi = tm_lonlat tm x y in
      (* Beyond half a meridian from the equator, a point is no point's
         image. *)
      let image = Nx.less_equal_s (Nx.abs xi) Float.pi in
      (lon, lat, Nx.logical_and (near offset lat) image)
  | Albers_map a -> albers_inverse a x y
  | Mercator m -> mercator_inverse m x y
  | Sinusoid m -> sinusoidal_inverse m x y

(* Each point (a, b) of [src] in longitude and latitude on its datum, the
   checks a transform to [dst] makes, in order, each a system it names and
   where the point lies in its domain, and [dst]'s map. Each map is made once
   a call, Albers' cone with it. Each datum's step to WGS 84 changes no
   coordinate. *)
let checked ~src ~dst a b =
  let lon, lat, own =
    match map src with
    | None -> (a, b, [ (src, (a, b), inside None a b) ])
    | Some m ->
        let lon, lat, ok = inverse m a b in
        (lon, lat, [ (src, (a, b), ok) ])
  in
  let into = map dst in
  let checks =
    match into with
    | None -> own
    | Some _ -> own @ [ (dst, (lon, lat), inside into lon lat) ]
  in
  (lon, lat, checks, into)

let in_domain (type d) ~src ~dst (p : d Nx.float64_t) : d Nx.bool_t =
  let batch, a, b = Points.split "Jord.Crs.in_domain" p in
  let _, _, checks, _ = checked ~src ~dst a b in
  let valid = Nx.logical_not (Nx.logical_or (Nx.isnan a) (Nx.isnan b)) in
  let ok =
    List.fold_left (fun ok (_, _, c) -> Nx.logical_and ok c) valid checks
  in
  Nx.reshape batch ok

let transform (type d) ~src ~dst (p : d Nx.float64_t) : d Nx.float64_t =
  let batch, a, b = Points.split "Jord.Crs.transform" p in
  if equal src dst then p
  else
    let lon, lat, checks, into = checked ~src ~dst a b in
    List.iter (fun (c, (a, b), ok) -> require ~batch c a b ok) checks;
    let x, y =
      match into with None -> (lon, lat) | Some m -> forward m lon lat
    in
    Points.join batch x y
