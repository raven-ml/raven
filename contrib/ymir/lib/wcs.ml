(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir_units.Quantity
module U = Ymir_units.Unit

let strf = Printf.sprintf
let ( let* ) = Result.bind

type keywords = {
  float : string -> (float option, string) result;
  int : string -> (int option, string) result;
  text : string -> (string option, string) result;
}

type edit =
  | Float of string * float
  | Int of string * int
  | Text of string * string
  | Remove of string

(* Reading through the view *)

(* [place kw] starts a message about [kw] with its EXTNAME, as in
   ["SCI: "]. *)
let place kw =
  match kw.text "EXTNAME" with Ok (Some n) when n <> "" -> n ^ ": " | _ -> ""

let fail kw fmt = Printf.ksprintf (fun s -> Error (place kw ^ s)) fmt

(* [has find k] is [true] unless [find] answers that [k] is absent: a
   keyword that does not decode, or whose cards disagree, is present, and
   reading it is the view's error. *)
let has find k = match find k with Ok None -> false | _ -> true

let get_float kw k =
  let* x = kw.float k in
  match x with Some x -> Ok x | None -> fail kw "%s is absent" k

let float_or kw k default =
  let* v = kw.float k in
  Ok (Option.value v ~default)

(* Keywords *)

let suffix fn = function
  | None | Some ' ' -> ""
  | Some c when c >= 'A' && c <= 'Z' -> String.make 1 c
  | Some c ->
      invalid_arg (strf "Wcs.%s: alternate %C is not ' ' or A-Z" fn c)

(* A celestial axis pair: the CTYPE prefixes of its longitude and latitude. *)
type pair = { lon : string; lat : string }

let pairs =
  [
    { lon = "RA"; lat = "DEC" };
    { lon = "GLON"; lat = "GLAT" };
    { lon = "ELON"; lat = "ELAT" };
    { lon = "SLON"; lat = "SLAT" };
  ]

(* FITS's projection codes that have no family yet. *)
let unread_codes =
  [
    "SFL"; "PAR"; "MOL"; "AIT"; "COP"; "COE"; "COD"; "COO"; "BON"; "PCO";
    "TSC"; "CSC"; "QSC"; "HPX"; "XPH"; "GLS"; "NCP"; "TNX"; "ZPX"; "TPU";
  ]

let tpv_code = "TPV"

let ctype_text pair_prefix code ~tpv ~sip =
  let p = pair_prefix ^ String.make (4 - String.length pair_prefix) '-' in
  p ^ "-"
  ^ (if tpv then tpv_code else Transform.code_name code)
  ^ if sip then "-SIP" else ""

(* [split_ctype s] is [s]'s coordinate type, its projection code and what
   follows: ["RA---TAN-SIP"] is [("RA", "TAN", "-SIP")]. *)
let split_ctype s =
  if String.length s < 8 || s.[4] <> '-' then None
  else
    let rec strip t =
      let n = String.length t in
      if n > 0 && t.[n - 1] = '-' then strip (String.sub t 0 (n - 1)) else t
    in
    Some
      ( strip (String.sub s 0 4),
        String.sub s 5 3,
        String.sub s 8 (String.length s - 8) )

(* Axes

   A description reads and writes the keywords of its own FITS axes, [ax],
   ascending: a celestial pair's are those of its two axes wherever they lie
   among the header's, and an image's are 1 and 2. *)

let axis_key p i sfx = strf "%s%d%s" p i sfx
let matrix_key p sfx (i, j) = strf "%s%d_%d%s" p i j sfx

(* [elements ax] is each element of the matrix of [ax], row by row. *)
let elements ax =
  List.concat_map
    (fun i -> List.map (fun j -> (i, j)) (Array.to_list ax))
    (Array.to_list ax)

let rec map_ok f = function
  | [] -> Ok []
  | x :: xs ->
      let* y = f x in
      let* ys = map_ok f xs in
      Ok (y :: ys)

type axes = {
  pair : pair;
  code : Transform.code;
  tpv : bool;  (** The CTYPE says TPV. *)
  sip : bool;  (** The CTYPE says -SIP. *)
  lng : int;  (** The longitude's FITS axis. *)
  lat : int;
}

let read_code kw key s =
  match String.trim s with
  | c when c = tpv_code -> Ok (Transform.Tan, true)
  | c -> (
      match Projection.of_name c with
      | Some code -> Ok (code, false)
      | None when List.mem c unread_codes ->
          fail kw "%s: the %s projection is not read yet" key c
      | None -> fail kw "%s: %S is not a celestial projection code" key c)

let read_axes kw sfx ax =
  let ctype i =
    let key = axis_key "CTYPE" i sfx in
    let* s = kw.text key in
    match Option.map split_ctype s with
    | Some (Some (prefix, code, rest)) -> Ok (key, s, prefix, code, rest)
    | Some None | None ->
        fail kw "%s = %s: not a celestial axis" key
          (match s with Some s -> strf "'%s'" s | None -> "absent")
  in
  let* k1, s1, p1, c1, r1 = ctype ax.(0) in
  let* k2, s2, p2, c2, r2 = ctype ax.(1) in
  let* pair, lng, lat =
    match
      List.find_map
        (fun q ->
          if q.lon = p1 && q.lat = p2 then Some (q, ax.(0), ax.(1))
          else if q.lat = p1 && q.lon = p2 then Some (q, ax.(1), ax.(0))
          else None)
        pairs
    with
    | Some x -> Ok x
    | None ->
        fail kw "%s = '%s' and %s = '%s' are not a celestial pair ymir reads"
          k1 (Option.get s1) k2 (Option.get s2)
  in
  let* code, tpv = read_code kw k1 c1 in
  let* code', tpv' = read_code kw k2 c2 in
  let* () =
    if code = code' && tpv = tpv' then Ok ()
    else fail kw "%s and %s name different projections" k1 k2
  in
  let algorithm k s r =
    match r with
    | "" -> Ok false
    | "-SIP" -> Ok true
    | r -> fail kw "%s = '%s': the axis algorithm %S is not read" k s r
  in
  let* sip1 = algorithm k1 (Option.get s1) r1 in
  let* sip2 = algorithm k2 (Option.get s2) r2 in
  let* () =
    if sip1 = sip2 then Ok () else fail kw "%s and %s disagree on -SIP" k1 k2
  in
  Ok { pair; code; tpv; sip = sip1; lng; lat }

(* Parameters *)

(* The FITS limit on [m] in [PVi_m]. *)
let pv_terms = 100

(* The primary description's PV keys, named once: a read probes each. *)
let primary_pv =
  Array.init 2 (fun i -> Array.init pv_terms (strf "PV%d_%d" (i + 1)))

let pv_key i m sfx =
  if sfx = "" && (i = 1 || i = 2) && m >= 0 && m < pv_terms then
    primary_pv.(i - 1).(m)
  else strf "PV%d_%d%s" i m sfx

(* [stated_pv kw sfx i] is each [m] with [PVi_m] in [kw], ascending. *)
let stated_pv kw sfx i =
  let rec go m acc =
    if m < 0 then acc
    else go (m - 1) (if has kw.float (pv_key i m sfx) then m :: acc else acc)
  in
  go (pv_terms - 1) []

(* How the header's PV terms read: the projection's parameters with the
   native reference point, or TPV's terms. *)
type terms =
  | Projection of {
      pv : float array;
      stated : int array;
      phi0 : float;
      theta0 : float;
    }
  | Tpv of { pv : float array; stated : int array }

let tpv_default k = if k mod Distortion.tpv_count = 1 then 1. else 0.

let read_tpv kw sfx axes =
  let count = Distortion.tpv_count in
  let row i =
    let ms = stated_pv kw sfx i in
    match List.find_opt (fun m -> m >= count) ms with
    | Some m ->
        fail kw "%s: TPV's terms are PV%d_0 to PV%d_%d" (pv_key i m sfx) i i
          (count - 1)
    | None -> Ok ms
  in
  let* lng_ms = row axes.lng in
  let* lat_ms = row axes.lat in
  let stated =
    List.map (fun m -> m) lng_ms @ List.map (fun m -> count + m) lat_ms
  in
  let pv = Array.init (2 * count) tpv_default in
  let* () =
    List.fold_left
      (fun acc k ->
        let* () = acc in
        let i = if k < count then axes.lng else axes.lat in
        let* v = get_float kw (pv_key i (k mod count) sfx) in
        pv.(k) <- v;
        Ok ())
      (Ok ()) stated
  in
  Ok (Tpv { pv; stated = Array.of_list stated })

let read_projection kw sfx axes =
  let code = axes.code in
  let first = Projection.first code and count = Projection.count code in
  let lat_ms = stated_pv kw sfx axes.lat in
  let* () =
    match
      List.find_opt (fun m -> m < first || m >= first + count) lat_ms
    with
    | None -> Ok ()
    | Some m when count = 0 ->
        fail kw "%s: the %s projection takes no parameter"
          (pv_key axes.lat m sfx) (Transform.code_name code)
    | Some m ->
        fail kw "%s: the %s projection's parameters are PV%d_%d to PV%d_%d"
          (pv_key axes.lat m sfx) (Transform.code_name code) axes.lat first
          axes.lat
          (first + count - 1)
  in
  let* () =
    match
      List.find_opt (fun m -> m < 1 || m > 4) (stated_pv kw sfx axes.lng)
    with
    | None -> Ok ()
    | Some 0 ->
        fail kw "%s: the fiducial offset is not read" (pv_key axes.lng 0 sfx)
    | Some m ->
        fail kw
          "%s: only PV%d_1 to PV%d_4 (the native reference point, LONPOLE \
           and LATPOLE) are read on the longitude axis"
          (pv_key axes.lng m sfx) axes.lng axes.lng
  in
  let pv = Projection.defaults code in
  let* () =
    List.fold_left
      (fun acc m ->
        let* () = acc in
        let* v = get_float kw (pv_key axes.lat m sfx) in
        pv.(m - first) <- v;
        Ok ())
      (Ok ()) lat_ms
  in
  let* phi0 = float_or kw (pv_key axes.lng 1 sfx) 0. in
  let* theta0 =
    float_or kw (pv_key axes.lng 2 sfx) (Projection.theta0 code)
  in
  Ok
    (Projection
       {
         pv;
         stated = Array.of_list (List.map (fun m -> m - first) lat_ms);
         phi0;
         theta0;
       })

(* TAN with PV terms on its latitude axis is TPV, as SCAMP writes it. *)
let read_terms kw sfx axes =
  if axes.tpv || (axes.code = Transform.Tan && stated_pv kw sfx axes.lat <> [])
  then read_tpv kw sfx axes
  else read_projection kw sfx axes

(* SIP

   A_ORDER and B_ORDER give the forward polynomials' orders, AP_ORDER and
   BP_ORDER the inverse's, each [p_i_j] for [i + j <= order], 0 when
   absent. *)

let sip_max_order = 9

let sip_matrix at kw p order =
  let n = order + 1 in
  let m = Array.make (n * n) 0. in
  let rec fill i j =
    if i > order then Ok m
    else if i + j > order then fill (i + 1) 0
    else
      let* v = float_or kw (strf "%s_%d_%d" p i j) 0. in
      m.((i * n) + j) <- v;
      fill i (j + 1)
  in
  let* m = fill 0 0 in
  Ok (Nx.place at (Nx.create Nx.float64 [| n; n |] m))

let sip_order kw key =
  let* o = kw.int key in
  match o with
  | Some o when o < 0 || o > sip_max_order ->
      fail kw "%s = %d: SIP's orders run from 0 to %d" key o sip_max_order
  | o -> Ok o

let read_sip at kw =
  let pair a b =
    let* oa = sip_order kw (a ^ "_ORDER") in
    let* ob = sip_order kw (b ^ "_ORDER") in
    match (oa, ob) with
    | None, None -> Ok None
    | Some oa, Some ob ->
        let* ma = sip_matrix at kw a oa in
        let* mb = sip_matrix at kw b ob in
        Ok (Some (ma, mb))
    | Some _, None -> fail kw "%s_ORDER is absent beside %s_ORDER" b a
    | None, Some _ -> fail kw "%s_ORDER is absent beside %s_ORDER" a b
  in
  let* forward = pair "A" "B" in
  let* seed = pair "AP" "BP" in
  match forward with
  | Some ab -> Ok (Some (Transform.sip ?seed ab))
  | None -> Ok None

(* Distortions this reader does not build. *)
let unread_distortion kw sfx ax =
  if sfx <> "" then Ok ()
  else
    let keys p = List.map (fun i -> axis_key p i "") (Array.to_list ax) in
    match List.find_opt (has kw.text) (keys "CPDIS" @ keys "D2IMDIS") with
    | Some k -> fail kw "%s: this distortion is not read yet" k
    | None -> Ok ()

(* Frames

   The CTYPE prefix chooses the system; RADESYS and EQUINOX qualify an
   equatorial or ecliptic one after FITS 4.0 §8.3's defaults. *)

type frame = Frame : 'f Frame.t -> frame

let radesys_key kw sfx =
  if sfx = "" && (not (has kw.text "RADESYS")) && has kw.text "RADECSYS" then
    "RADECSYS"
  else "RADESYS" ^ sfx

let equinox kw sfx =
  let* e = kw.float ("EQUINOX" ^ sfx) in
  match e with
  | Some _ -> Ok e
  | None when sfx = "" -> kw.float "EPOCH"
  | None -> Ok None

let time_stage kw what = fail kw "%s needs the time stage, not read yet" what

let read_frame kw sfx axes =
  let* sys = kw.text (radesys_key kw sfx) in
  let* eq = equinox kw sfx in
  let sys, stated =
    match (sys, eq) with
    | Some s, _ -> (String.trim s, strf "RADESYS = '%s'" (String.trim s))
    | None, None -> ("ICRS", "")
    | None, Some e ->
        let s = if e < 1984. then "FK4" else "FK5" in
        (s, strf "%s (EQUINOX %s without RADESYS)" s (Transform.num e))
  in
  match (axes.pair.lon, sys) with
  | "GLON", _ -> Ok (Frame Frame.galactic)
  | "SLON", _ -> Ok (Frame Frame.supergalactic)
  | "RA", "ICRS" -> Ok (Frame Frame.icrs)
  | "RA", "FK5" -> (
      match eq with
      | None | Some 2000. -> Ok (Frame Frame.fk5_j2000)
      | Some e -> time_stage kw (strf "FK5 at equinox %s" (Transform.num e)))
  | "ELON", "ICRS" -> (
      match eq with
      | None | Some 2000. -> Ok (Frame Frame.ecliptic_j2000)
      | Some e ->
          time_stage kw (strf "the ecliptic at equinox %s" (Transform.num e)))
  | "ELON", "FK5" -> time_stage kw "the IAU 1976 ecliptic (ELON under FK5)"
  | _ -> time_stage kw stated

let frame_text (Frame f) =
  match f with
  | Frame.Icrs -> "ICRS"
  | Frame.Fk5_j2000 -> "FK5 at equinox 2000.0"
  | Frame.Galactic -> "Galactic"
  | Frame.Ecliptic_j2000 -> "the J2000 ecliptic"
  | Frame.Supergalactic -> "supergalactic"
  | Frame.Gcrs -> "GCRS"
  | Frame.Cirs -> "CIRS"
  | Frame.True_of_date -> "the true equator and equinox of date"
  | Frame.Tirs -> "TIRS"

(* [same f g] is [g] at [f]'s type when they are one frame. *)
let same : type f g. f Frame.t -> g Frame.t -> f Frame.t option =
 fun f g ->
  match (f, g) with
  | Frame.Icrs, Frame.Icrs -> Some f
  | Frame.Fk5_j2000, Frame.Fk5_j2000 -> Some f
  | Frame.Galactic, Frame.Galactic -> Some f
  | Frame.Ecliptic_j2000, Frame.Ecliptic_j2000 -> Some f
  | Frame.Supergalactic, Frame.Supergalactic -> Some f
  | Frame.Gcrs, Frame.Gcrs -> Some f
  | Frame.Cirs, Frame.Cirs -> Some f
  | Frame.True_of_date, Frame.True_of_date -> Some f
  | Frame.Tirs, Frame.Tirs -> Some f
  | _ -> None

(* Reading *)

(* A celestial CUNIT is an angle in FITS's unit syntax. *)
let unit_of kw sfx axes =
  let cunit i =
    let key = strf "CUNIT%d%s" i sfx in
    let* s = kw.text key in
    match s with
    | None | Some "" -> Ok (U.degree, key)
    | Some s -> (
        match Ymir_units.Fits_unit.parse s with
        | Ok u when U.convertible u U.radian -> Ok (u, key)
        | _ -> fail kw "%s = '%s' is not a celestial unit" key s)
  in
  let* u1, k1 = cunit axes.lng in
  let* u2, k2 = cunit axes.lat in
  if U.equal u1 u2 then Ok u1
  else fail kw "%s and %s differ; one unit for both axes is read" k1 k2


(* The linear step's keywords: PC with CDELT, CD, or CROTA with CDELT. *)
type family = Pc | Cd | Crota

(* [family kw sfx ax] is the family of the matrix of [ax]: CD where a CD
   element is present, CROTA where only CROTA is, PC otherwise. *)
let family kw sfx ax =
  let has_matrix p =
    List.exists (fun e -> has kw.float (matrix_key p sfx e)) (elements ax)
  in
  let crota =
    if sfx <> "" then None
    else
      List.find_opt (has kw.float)
        (List.map (fun i -> axis_key "CROTA" i "") (Array.to_list ax))
  in
  match (has_matrix "CD", has_matrix "PC", crota) with
  | true, true, _ -> fail kw "a header with both CD and PC is ambiguous"
  | true, _, Some k | _, true, Some k ->
      fail kw "%s beside %s is ambiguous" k
        (if has_matrix "CD" then "CD" else "PC")
  | true, false, None -> Ok Cd
  | false, false, Some _ -> Ok Crota
  | false, _, None -> Ok Pc

(* [entries kw sfx family ax] is the matrix of [ax], row by row: a missing CD
   element is 0, and a missing PC element the identity's. CDi_j is PCi_j
   with CDELTi 1 (FITS WCS Paper I, §2.1.2). *)
let entries kw sfx family ax =
  map_ok
    (fun (i, j) ->
      match family with
      | Cd -> float_or kw (matrix_key "CD" sfx (i, j)) 0.
      | Pc | Crota ->
          float_or kw (matrix_key "PC" sfx (i, j)) (if i = j then 1. else 0.))
    (elements ax)

(* PC from CROTA, FITS WCS Paper II eq. 189: [ρ] is the latitude axis's
   CROTA, and the ratio of CDELTs keeps the matrix a rotation of the scaled
   axes. *)
let pc_of_crota kw axes (d1, d2) =
  let* rho = float_or kw (strf "CROTA%d" axes.lat) 0. in
  let r = rho *. Float.pi /. 180. in
  let c = Float.cos r and s = Float.sin r in
  Ok [ c; -.s *. d2 /. d1; s *. d1 /. d2; c ]


(* LONPOLE's default (FITS 4.0 §8.3): 180° when the reference point is below
   the native reference's latitude, else 0°, plus φ₀. *)
let default_lonpole ~delta0 ~theta0 ~phi0 =
  (if delta0 < theta0 then 180. else 0.) +. phi0

(* LATPOLE's default (FITS 4.0 §8.3): 90°, and none where LATPOLE alone fixes
   the celestial pole's latitude, with θ₀ = 0, δ₀ = 0 and LONPOLE 90° from φ₀
   (Paper II §2.4, rule 6). WCSLIB takes 90° there too. *)
let default_latpole ~delta0 ~theta0 ~phi0 ~lonpole =
  let quarter = Float.rem (Float.abs (lonpole -. phi0)) 180. = 90. in
  if theta0 = 0. && delta0 = 0. && quarter then None else Some 90.

(* [pole kw key alt default] reads LONPOLE or LATPOLE, which FITS also spells
   PVi_3 and PVi_4 on the longitude axis [i] (Paper II §2.5); under TPV
   those are TPV's terms, and [alt] is [None]. Absent with no [default], it
   is an [Error]. *)
let pole kw key alt default =
  let* a = kw.float key in
  let* b =
    match alt with None -> Ok None | Some alt -> kw.float alt
  in
  match (a, b, alt) with
  | Some x, Some y, Some alt when not (Float.equal x y) ->
      fail kw "%s = %s and %s = %s spell one value and disagree" key
        (Transform.num x) alt (Transform.num y)
  | Some x, _, _ | None, Some x, _ -> Ok x
  | None, None, _ -> (
      match default with
      | Some d -> Ok d
      | None ->
          fail kw
            "%s is absent, and FITS gives it no default where θ₀ = 0, CRVAL's \
             latitude is 0 and LONPOLE is 90° from φ₀: it alone fixes the \
             celestial pole"
            key)

(* Descriptions

   A description of its axes [ax] reads as a [head], from pixel indices
   through CRPIX, SIP and the matrix, and a [tail], from there to the world:
   CDELT then [world], or [world] alone under CD. Where PC or CD couples two
   descriptions, the header's head holds every axis and each description
   contributes its tail, CDELT 1 in [unit] under CD. *)
type ('w, 'd) leaf = {
  head : ('d Transform.plane, 'd Transform.plane, 'd) Transform.t;
  tail : ('d Transform.plane, 'w, 'd) Transform.t;
  world : ('d Transform.plane, 'w, 'd) Transform.t;
  unit : U.t;
}

let ones ~at u n = Quantity.v u (Nx.place at (Nx.full Nx.float64 [| n |] 1.))

(* [sky ~at ~family frame kw sfx ax] is the celestial pair [ax] of [kw]. *)
let sky ~at ~family (frame : 'f Frame.t) kw sfx ax =
  let f64 = Nx.float64 in
  let create s xs = Nx.place at (Nx.create f64 s xs) in
  let* axes = read_axes kw sfx ax in
  let* () = unread_distortion kw sfx ax in
  let* terms = read_terms kw sfx axes in
  let* sip =
    if axes.sip || (sfx = "" && has kw.int "A_ORDER") then read_sip at kw
    else Ok None
  in
  let* () =
    if Option.is_some sip && ax <> [| 1; 2 |] then
      fail kw "A_ORDER: SIP describes FITS axes 1 and 2, not %d and %d" ax.(0)
        ax.(1)
    else Ok ()
  in
  let* (Frame found as fr) = read_frame kw sfx axes in
  let* frame =
    match same frame found with
    | Some f -> Ok f
    | None ->
        fail kw "%s; the caller expects %s. Read with Frame.%s."
          (frame_text fr) (Frame.name frame) (Frame.name found)
  in
  let* u = unit_of kw sfx axes in
  let* crpix = map_ok (fun i -> float_or kw (axis_key "CRPIX" i sfx) 0.) [ ax.(0); ax.(1) ] in
  let* crval_lng = float_or kw (axis_key "CRVAL" axes.lng sfx) 0. in
  let* crval_lat = float_or kw (axis_key "CRVAL" axes.lat sfx) 0. in
  let* () =
    if family = Cd && not (List.exists (fun e -> has kw.float (matrix_key "CD" sfx e)) (elements ax))
    then
      fail kw "%s is absent: a CD header's celestial axes need CD"
        (matrix_key "CD" sfx (ax.(0), ax.(0)))
    else Ok ()
  in
  let plane x = Quantity.v u x in
  let bare x = Quantity.v U.one x in
  let* d =
    match family with
    | Cd -> Ok [ 1.; 1. ]
    | Pc | Crota -> map_ok (fun i -> float_or kw (axis_key "CDELT" i sfx) 1.) [ ax.(0); ax.(1) ]
  in
  let cdelt = Transform.scale (plane (create [| 2 |] (Array.of_list d))) in
  let* matrix =
    match family with
    | Cd ->
        let* cd = entries kw sfx Cd ax in
        Ok (Transform.linear (plane (create [| 2; 2 |] (Array.of_list cd))))
    | Pc ->
        let* pc = entries kw sfx Pc ax in
        Ok (Transform.linear (bare (create [| 2; 2 |] (Array.of_list pc))))
    | Crota ->
        let* pc = pc_of_crota kw axes (List.hd d, List.nth d 1) in
        Ok (Transform.linear (bare (create [| 2; 2 |] (Array.of_list pc))))
  in
  let pv, stated, phi0, theta0, tpv_stage =
    match terms with
    | Projection { pv; stated; phi0; theta0 } ->
        (pv, stated, phi0, theta0, Transform.id)
    | Tpv { pv; stated } ->
        ( [||],
          [||],
          0.,
          Projection.theta0 axes.code,
          Transform.tpv ~stated (create [| 2; Distortion.tpv_count |] pv) )
  in
  let delta0 = U.ratio Nx.float64 u U.degree *. crval_lat in
  let alt m =
    match terms with
    | Projection _ -> Some (pv_key axes.lng m sfx)
    | Tpv _ -> None
  in
  let* lonpole =
    pole kw ("LONPOLE" ^ sfx) (alt 3)
      (Some (default_lonpole ~delta0 ~theta0 ~phi0))
  in
  let* latpole =
    pole kw ("LATPOLE" ^ sfx) (alt 4)
      (default_latpole ~delta0 ~theta0 ~phi0 ~lonpole)
  in
  let deg x = Quantity.v U.degree x in
  let swap =
    if axes.lng = ax.(0) then Transform.id
    else Transform.axes [| 1; 0 |] ~origin:0
  in
  let projection =
    Transform.celestial ~stated axes.code frame
      ~pv:(create [| Array.length pv |] pv)
      ~native:(deg (create [| 2 |] [| phi0; theta0 |]))
      ~crval:(Quantity.v u (create [| 2 |] [| crval_lng; crval_lat |]))
      ~lonpole:(deg (Nx.scalar_at at f64 lonpole))
      ~latpole:(deg (Nx.scalar_at at f64 latpole))
  in
  let world = Transform.(swap >> tpv_stage >> projection) in
  let sip_stage = Option.value sip ~default:Transform.id in
  Ok
    {
      head =
        Transform.(
          axes [| 1; 0 |] ~origin:1
          >> shift (bare (create [| 2 |] (Array.of_list crpix)))
          >> sip_stage >> matrix);
      tail = (if family = Cd then world else Transform.(cdelt >> world));
      world;
      unit = u;
    }

(* Linear axes

   A linear axis's world is [CRVAL + CDELT · w] for its intermediate pixel
   offset [w] (FITS WCS Paper I, eq. 1), in CUNIT, FITS's unit syntax
   ({!Ymir_units.Fits_unit}). Absent CUNIT, a spectral type's unit is the one
   FITS WCS Paper III's Table 1 gives it, and any other axis is
   dimensionless. *)

let spectral_types =
  U.
    [
      ("FREQ", hertz);
      ("ENER", joule);
      ("WAVN", one / metre);
      ("VRAD", metre / second);
      ("WAVE", metre);
      ("VOPT", metre / second);
      ("ZOPT", one);
      ("AWAV", metre);
      ("VELO", metre / second);
      ("BETA", one);
    ]

(* Paper III's algorithm codes (Table 2, -LOG, -TAB and the grisms), and the
   time paper's scales as axis types. *)
let spectral_codes =
  [
    "F2W"; "F2V"; "F2A"; "W2F"; "W2V"; "W2A"; "V2F"; "V2W"; "V2A"; "A2F";
    "A2W"; "A2V"; "LOG"; "TAB"; "GRI"; "GRA";
  ]

let time_types =
  [
    "TIME"; "TAI"; "TT"; "TDT"; "ET"; "IAT"; "UT1"; "UTC"; "GMT"; "GPS";
    "TCG"; "TCB"; "TDB"; "LOCAL";
  ]

(* [type_unit ctype] is the unit Paper III's Table 1 gives the spectral type
   [ctype] names, if it names one. *)
let type_unit ctype =
  let t = if String.length ctype >= 4 then String.sub ctype 0 4 else ctype in
  List.assoc_opt t spectral_types

let axis_unit kw key ctype =
  let* s = kw.text key in
  let typed = type_unit ctype in
  match s with
  | None | Some "" -> Ok (Option.value typed ~default:U.one)
  | Some s -> (
      match (Ymir_units.Fits_unit.parse s, typed) with
      | Error e, _ -> fail kw "%s: %s" key e
      | Ok u, Some t when not (U.convertible u t) ->
          fail kw "%s = '%s' is not a unit of '%s'" key s ctype
      | Ok u, _ -> Ok u)

(* [linear ~at ~family ~coupled kw sfx a] is the linear axis [a] of [kw],
   whose PC element is one of a matrix that couples it with others where
   [coupled]. *)
let linear ~at ~family ~coupled kw sfx a =
  let one v = Nx.place at (Nx.create Nx.float64 [| 1 |] [| v |]) in
  let key p = axis_key p a sfx in
  let* ctype = kw.text (key "CTYPE") in
  let ctype = String.trim (Option.value ctype ~default:"") in
  let* u = axis_unit kw (key "CUNIT") ctype in
  let* () =
    match stated_pv kw sfx a with
    | [] -> Ok ()
    | m :: _ -> fail kw "%s: a linear axis takes no parameter" (pv_key a m sfx)
  in
  let* crpix = float_or kw (key "CRPIX") 0. in
  let* crval = float_or kw (key "CRVAL") 0. in
  (* A zero scale maps every pixel to CRVAL. *)
  let* matrix, d =
    match family with
    | Cd ->
        let k = matrix_key "CD" sfx (a, a) in
        let* cd = kw.float k in
        (match cd with
        | None -> fail kw "%s is absent: a CD header's linear axis needs it" k
        | Some 0. -> fail kw "%s = 0 maps every pixel to CRVAL%d%s" k a sfx
        | Some cd -> Ok (Transform.linear (Quantity.v u (Nx.reshape [| 1; 1 |] (one cd))), 1.))
    | Pc | Crota ->
        let pc_key = matrix_key "PC" sfx (a, a) in
        let* pc = float_or kw pc_key 1. in
        let* d = float_or kw (key "CDELT") 1. in
        if d = 0. then fail kw "%s = 0 maps every pixel to CRVAL%d%s" (key "CDELT") a sfx
        else if pc = 0. && not coupled then
          fail kw "%s = 0 maps every pixel to CRVAL%d%s" pc_key a sfx
        else Ok (Transform.linear (Quantity.v U.one (Nx.reshape [| 1; 1 |] (one pc))), d)
  in
  let world = Transform.(inverse (shift (Quantity.v u (one crval)))) in
  Ok
    {
      head = Transform.(axes [| 0 |] ~origin:1 >> shift (Quantity.v U.one (one crpix)) >> matrix);
      tail =
        (if family = Cd then world
         else Transform.(scale (Quantity.v u (one d)) >> world));
      world;
      unit = u;
    }

(* Reading N axes *)

let axis_count kw sfx =
  let* w = kw.int ("WCSAXES" ^ sfx) in
  match w with
  | Some n -> Ok n
  | None -> Result.map (Option.value ~default:2) (kw.int "NAXIS")

(* An axis's kind, from its CTYPE. *)
type kind = Celestial | Linear

let classify kw sfx i =
  let key = axis_key "CTYPE" i sfx in
  let* s = kw.text key in
  let s = Option.value s ~default:"" in
  let projection c =
    c = tpv_code
    || Option.is_some (Projection.of_name c)
    || List.mem c unread_codes
  in
  match split_ctype s with
  | Some (_, code, _) when projection code -> Ok Celestial
  | Some (_, code, _) when List.mem code spectral_codes ->
      fail kw "%s = '%s': the spectral algorithm %s is not read yet" key s code
  | Some (_, code, _) ->
      fail kw "%s = '%s': the axis algorithm %S is not read" key s code
  | None when List.mem (String.trim s) time_types ->
      fail kw "%s = '%s': a time axis is not read yet" key s
  | None -> Ok Linear

(* [count world] is the celestial pairs and linear axes [world] reads. *)
let rec count : type w d. (w, d) Transform.endpoint -> int * int = function
  | Transform.Planar -> (0, 1)
  | Sky _ -> (1, 0)
  | Both (a, b) ->
      let p, l = count a and p', l' = count b in
      (p + p', l + l')

let describe ~celestial ~linear =
  let count n one many = strf "%d %s" n (if n = 1 then one else many) in
  let pairs =
    if celestial mod 2 = 0 then
      count (celestial / 2) "celestial pair" "celestial pairs"
    else count celestial "celestial axis" "celestial axes"
  in
  match
    (if celestial = 0 then [] else [ pairs ])
    @ if linear = 0 then [] else [ count linear "linear axis" "linear axes" ]
  with
  | [] -> "no axis"
  | parts -> String.concat " and " parts

let read (type w d) ?alt ~(at : d Nx.Placement.t)
    (world : (w, d) Transform.endpoint) kw :
    ((d Transform.plane, w, d) Transform.t, string) result =
  let sfx = suffix "read" alt in
  let* n = axis_count kw sfx in
  let all = Array.init n succ in
  let* kinds = map_ok (classify kw sfx) (Array.to_list all) in
  let numbered k =
    List.filter_map
      (fun (i, k') -> if k = k' then Some i else None)
      (List.combine (Array.to_list all) kinds)
  in
  let cel = Array.of_list (numbered Celestial) and lin = numbered Linear in
  let pairs_wanted, linear_wanted = count world in
  let* () =
    if pairs_wanted > 1 then
      fail kw "the caller expects %d celestial pairs; Wcs reads one" pairs_wanted
    else if Array.length cel = 2 * pairs_wanted && List.length lin = linear_wanted
    then Ok ()
    else
      fail kw "%s, %s; the caller expects %s"
        (if n = 1 then "1 axis" else strf "%d axes" n)
        (describe ~celestial:(Array.length cel) ~linear:(List.length lin))
        (describe ~celestial:(2 * pairs_wanted) ~linear:linear_wanted)
  in
  let* family = family kw sfx all in
  (* Each axis's description: 0 for the celestial pair, [k + 1] for linear
     axis [k]. *)
  let side = Array.make (n + 1) 0 in
  List.iteri (fun k i -> side.(i) <- k + 1) lin;
  let* coupled =
    if family = Crota then Ok false
    else
      let* m = entries kw sfx family all in
      Ok (List.exists2 (fun (i, j) v -> side.(i) <> side.(j) && v <> 0.) (elements all) m)
  in
  let* () =
    if coupled && sfx = "" && has kw.int "A_ORDER" then
      fail kw "A_ORDER: SIP is not read beside a matrix that couples axes"
    else Ok ()
  in
  let queue = ref lin in
  (* [build world] is the axes [world] reads, ascending, and their stages:
     whole, or each tail after the coupled head. *)
  let rec build : type w.
      (w, d) Transform.endpoint ->
      (int array * (d Transform.plane, w, d) Transform.t, string) result =
   fun world ->
    let pick ax (l : (w, d) leaf) =
      if not coupled then Ok (ax, Transform.(l.head >> l.tail))
      else if family = Cd then
        Ok (ax, Transform.(scale (ones ~at l.unit (Array.length ax)) >> l.world))
      else Ok (ax, l.tail)
    in
    match world with
    | Transform.Sky frame ->
        let* l = sky ~at ~family frame kw sfx cel in
        pick cel l
    | Planar ->
        let a = List.hd !queue in
        queue := List.tl !queue;
        let* l = linear ~at ~family ~coupled kw sfx a in
        pick [| a |] l
    | Both (a, b) ->
        let* sa, ta = build a in
        let* sb, tb = build b in
        let s = Array.append sa sb in
        Array.sort compare s;
        let m = Array.length s in
        (* After the matrix the components are in FITS order; before it,
           pixel indices are in the tensor's, FITS's reversed. *)
        let index p = if coupled then p else m - 1 - p in
        let first =
          List.filter_map
            (fun p -> if Array.mem s.(p) sa then Some (index p) else None)
            (List.init m Fun.id)
        in
        let first = Array.of_list (List.sort compare first) in
        Ok (s, Transform.(split first >> both ta tb))
  in
  let* _, t = build world in
  if not coupled then Ok t
  else
    let create s xs = Nx.place at (Nx.create Nx.float64 s (Array.of_list xs)) in
    let bare s xs = Quantity.v U.one (create s xs) in
    let* crpix =
      map_ok (fun i -> float_or kw (axis_key "CRPIX" i sfx) 0.) (Array.to_list all)
    in
    let* m = entries kw sfx family all in
    Ok
      Transform.(
        axes (Array.init n (fun k -> n - 1 - k)) ~origin:1
        >> shift (bare [| n |] crpix)
        >> linear (bare [| n; n |] m)
        >> t)

(* Writing

   The writer reads the stage list back into keyword values. A keyword that
   reads back equal gets no edit, one that differs is set, and one the header
   lacks is set unless its value is the one FITS assumes for an absent
   keyword. *)

type value = F of float | S of string | I of int
type card = { key : string; value : value; optional : bool }

let card ?(optional = false) key value = { key; value; optional }
(* Values compare bit for bit, so a written -0. reads back as -0. *)
let same x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
let default x d key = card ~optional:(same x d) key (F x)

type 'd celestial = {
  code : Transform.code;
  frame : frame;
  crval : (float, Nx.float64_elt, 'd) Nx.t Quantity.t;
  pv : float array;
  stated : int array;
  native : float array;  (** (φ₀, θ₀) in degrees. *)
  lonpole : float;  (** Degrees. *)
  latpole : float;  (** Degrees. *)
}

type sip = {
  a : float array array;
  b : float array array;
  seed : (float array array * float array array) option;
}

(* The stage list, read as plain data. *)
type 'd step =
  | Axes of int array * int
  | Shift of float array
  | Linear of float array * U.t
  | Scale of float array * U.t
  | Sip of sip
  | Tpv of float array * int array
  | Celestial of 'd celestial
  | At of float array * U.t  (** An inverse shift: CRVAL. *)
  | Split_at of int array
  | Sides of 'd step list * 'd step list
  | Unspelled of string

(* [host q] is [q]'s shape, float64 values and unit. *)
let host q =
  let u = Quantity.unit q in
  let v = Quantity.value u q in
  (Nx.shape v, Nx.to_array (Nx.cast Nx.float64 v), u)

let degrees q =
  let _, x, _ = host (Quantity.convert U.degree q) in
  x

let rows m =
  let n = Nx.dim (-1) m in
  let a = Nx.to_array m in
  Array.init (Nx.dim (-2) m) (fun i -> Array.sub a (i * n) n)

let rec step : type a b d. (a, b, d) Transform.stage -> d step = function
  | Transform.Plane (Transform.Axes { perm; origin }, Transform.Forward) ->
      Axes (perm, origin)
  | Plane (Transform.Shift r, Forward) ->
      let _, r, _ = host r in
      Shift r
  | Plane (Transform.Linear m, Forward) ->
      let _, m, u = host m in
      Linear (m, u)
  | Plane (Transform.Scale d, Forward) ->
      let _, d, u = host d in
      Scale (d, u)
  | Plane (Transform.Sip { a; b; seed }, Forward) ->
      Sip
        {
          a = rows a;
          b = rows b;
          seed = Option.map (fun (p, q) -> (rows p, rows q)) seed;
        }
  | Plane (Transform.Tpv { pv; stated }, Forward) ->
      Tpv (Nx.to_array pv, stated)
  | Deproject c ->
      Celestial
        {
          code = c.code;
          frame = Frame c.frame;
          crval = c.crval;
          pv = Nx.to_array c.pv;
          stated = c.stated;
          native = degrees c.native;
          lonpole = (degrees c.lonpole).(0);
          latpole = (degrees c.latpole).(0);
        }
  | Plane (Transform.Shift r, Inverse) ->
      let _, r, u = host r in
      At (r, u)
  | Split first -> Split_at first
  | Parallel (t, u) -> Sides (steps t, steps u)
  | s -> Unspelled (Transform.stage_name s)

and steps : type a b d. (a, b, d) Transform.t -> d step list = function
  | Transform.Id -> []
  | Stage (s, rest) -> step s :: steps rest

let rank q = Nx.ndim (Quantity.value (Quantity.unit q) q)

let scalar_celestial (c : _ Transform.celestial) =
  rank c.crval = 1
  && rank c.native = 1
  && rank c.lonpole = 0
  && rank c.latpole = 0
  && Nx.ndim c.pv = 1

(* Every leaf of a stage FITS spells holds one value per keyword. *)
let rec unbatched : type a b d. (a, b, d) Transform.t -> bool = function
  | Transform.Id -> true
  | Stage (s, rest) ->
      let ok =
        match s with
        | Transform.Plane (Transform.Shift r, _) -> rank r = 1
        | Plane (Transform.Linear m, _) -> rank m = 2
        | Plane (Transform.Scale d, _) -> rank d = 1
        | Plane (Transform.Axes _, _) -> true
        | Plane (Transform.Sip { a; b; seed }, _) ->
            Nx.ndim a = 2
            && Nx.ndim b = 2
            && Option.fold ~none:true
                 ~some:(fun (p, q) -> Nx.ndim p = 2 && Nx.ndim q = 2)
                 seed
        | Plane (Transform.Tpv { pv; _ }, _) -> Nx.ndim pv = 2
        | Deproject c -> scalar_celestial c
        | Project c -> scalar_celestial c
        | Parallel (t, u) -> unbatched t && unbatched u
        | Rotate _ | Split _ | Join _ -> true
      in
      ok && unbatched rest

let unspellable s =
  let name =
    match s with
    | Unspelled name -> name
    | Axes _ -> "axes"
    | Shift _ -> "shift"
    | Linear _ -> "linear"
    | Scale _ -> "scale"
    | Sip _ -> "sip"
    | Tpv _ -> "tpv"
    | Celestial c -> "celestial " ^ Transform.code_name c.code
    | At _ -> "inverse shift"
    | Split_at _ -> "split"
    | Sides _ -> "both"
  in
  Error (strf "Wcs.write: FITS cannot spell the %s stage there" name)

let ends () =
  Error "Wcs.write: the transform ends before its celestial stage"

(* A geocentric frame's orientation depends on an instant. FITS WCS
   Paper II (Calabretta and Greisen 2002) spells the true equator and equinox of
   date as RADESYS GAPPT with MJD-OBS, but ymir writes no instant into a
   header, so it refuses every geocentric frame. *)
let pair_of (Frame f) =
  let prefix lon = Ok (List.find (fun p -> p.lon = lon) pairs) in
  match f with
  | Frame.Icrs | Frame.Fk5_j2000 -> prefix "RA"
  | Frame.Galactic -> prefix "GLON"
  | Frame.Ecliptic_j2000 -> prefix "ELON"
  | Frame.Supergalactic -> prefix "SLON"
  | Frame.Gcrs | Frame.Cirs | Frame.True_of_date | Frame.Tirs ->
      Error
        (strf
           "Wcs.write: ymir writes no header for a celestial stage in %s, \
            whose orientation depends on an instant"
           (Frame.name f))

let has_equinox kw sfx =
  has kw.float ("EQUINOX" ^ sfx) || (sfx = "" && has kw.float "EPOCH")

let frame_cards kw sfx (Frame f) =
  let radesys = radesys_key kw sfx in
  let icrs = card ~optional:(not (has_equinox kw sfx)) radesys (S "ICRS") in
  match f with
  | Frame.Icrs -> [ icrs ]
  | Frame.Fk5_j2000 ->
      [ card radesys (S "FK5"); card ("EQUINOX" ^ sfx) (F 2000.) ]
  | Frame.Ecliptic_j2000 ->
      [ icrs; card ~optional:true ("EQUINOX" ^ sfx) (F 2000.) ]
  | Frame.Galactic | Frame.Supergalactic -> []
  | Frame.Gcrs | Frame.Cirs | Frame.True_of_date | Frame.Tirs -> []


(* A pole is written with the spelling the header uses: PVi_3 or PVi_4 when
   it has one, and then LONPOLE or LATPOLE only where it has that too.
   Under TPV, [alt] is [None]: those are TPV's terms. *)
let pole_cards kw key alt x d =
  match alt with
  | Some alt when has kw.float alt ->
      [ card ~optional:true key (F x); card ~optional:true alt (F x) ]
  | _ -> (
      match d with
      | Some d -> [ default x d key ]
      | None -> [ card key (F x) ])

(* [term_cards ~key ~default values stated] spells [values]: a stated
   term or one away from its default is always written. *)
let term_cards ~key ~default values stated =
  List.init (Array.length values) (fun k ->
      let v = values.(k) in
      card
        ~optional:((not (Array.mem k stated)) && same v (default k))
        (key k) (F v))

(* [sip_cards p m] spells SIP's [m] under the prefix [p], or [Error] where
   it has a term beyond its order. *)
let sip_cards p m =
  let n = Array.length m in
  let order = n - 1 in
  let beyond = ref false in
  let cards = ref [] in
  Array.iteri
    (fun i row ->
      Array.iteri
        (fun j v ->
          if i + j > order then (if v <> 0. then beyond := true)
          else
            cards :=
              card ~optional:(same v 0.) (strf "%s_%d_%d" p i j) (F v)
              :: !cards)
        row)
    m;
  if !beyond then
    Error
      (strf
         "Wcs.write: the sip stage's %s has a term of degree above %d, \
          which FITS cannot spell"
         p order)
  else Ok (card (p ^ "_ORDER") (I order) :: List.rev !cards)


(* Writing descriptions

   A description is written with its FITS axes [ax], as it was read: its
   head spells CRPIX, SIP and the matrix, its tail CDELT and the keywords of
   its world. *)

let reversed perm =
  let n = Array.length perm in
  Array.for_all Fun.id (Array.mapi (fun k p -> p = n - 1 - k) perm)

let identity n = Array.init (n * n) (fun k -> if k mod (n + 1) = 0 then 1. else 0.)

let matrix_cards kw sfx ax = function
  | `Pc pc ->
      List.map2
        (fun (i, j) v ->
          default v (if i = j then 1. else 0.) (matrix_key "PC" sfx (i, j)))
        (elements ax) (Array.to_list pc)
  | `Cd cd ->
      (* A CD header needs one CD card to read as one. *)
      let keys = List.map (matrix_key "CD" sfx) (elements ax) in
      let some_cd =
        Array.exists (fun x -> x <> 0.) cd || List.exists (has kw.float) keys
      in
      List.map2
        (fun k v ->
          let c = default v 0. k in
          { c with optional = c.optional && some_cd })
        keys (Array.to_list cd)

(* [head kw sfx ~cd ax steps] is a head's cards, its matrix's family, the CD
   unit when that is CD, whether it has SIP, and the steps after it. A matrix
   in 1 before CDELT is PC, and before a coupled form's split PC, or CD where
   [cd]. *)
let head kw sfx ~cd ax steps =
  let n = Array.length ax in
  let* crpix, rest =
    match steps with
    | Axes (perm, 1) :: Shift crpix :: rest
      when reversed perm && Array.length perm = n ->
        Ok (crpix, rest)
    | Axes (_, 1) :: s :: _ | s :: _ -> unspellable s
    | [] -> ends ()
  in
  let sip, rest =
    match rest with Sip s :: rest -> (Some s, rest) | rest -> (None, rest)
  in
  let* () =
    if Option.is_none sip then Ok ()
    else if sfx <> "" then
      Error "Wcs.write: SIP's keywords belong to the primary description"
    else if ax <> [| 1; 2 |] then
      Error "Wcs.write: SIP's keywords describe FITS axes 1 and 2"
    else Ok ()
  in
  let* matrix, unit, rest =
    match rest with
    | Linear (m, u) :: (Scale _ :: _ as rest) when U.equal u U.one ->
        Ok (`Pc m, None, rest)
    | Linear (m, u) :: (Split_at _ :: _ as rest) when U.equal u U.one ->
        Ok ((if cd then `Cd m else `Pc m), None, rest)
    | Linear (m, u) :: rest -> Ok (`Cd m, Some u, rest)
    | Scale _ :: _ -> Ok (`Pc (identity n), None, rest)
    | s :: _ -> unspellable s
    | [] -> ends ()
  in
  let* sip_cards =
    match sip with
    | None -> Ok []
    | Some s ->
        let* a = sip_cards "A" s.a in
        let* b = sip_cards "B" s.b in
        let* seed =
          match s.seed with
          | None -> Ok []
          | Some (ap, bp) ->
              let* ap = sip_cards "AP" ap in
              let* bp = sip_cards "BP" bp in
              Ok (ap @ bp)
        in
        Ok (a @ b @ seed)
  in
  let crpix =
    List.mapi (fun k x -> default x 0. (axis_key "CRPIX" ax.(k) sfx)) (Array.to_list crpix)
  in
  let family = match matrix with `Pc _ -> `Pc | `Cd _ -> `Cd in
  Ok
    ( crpix @ matrix_cards kw sfx ax matrix @ sip_cards,
      family,
      unit,
      Option.is_some sip,
      rest )

(* [cunit_text kw key u] is [kw]'s CUNIT [key] where it reads as [u], and
   [u]'s FITS spelling otherwise. *)
let cunit_text kw key u =
  let* stated = kw.text key in
  let reads_as_u t =
    match Ymir_units.Fits_unit.parse t with
    | Ok w -> U.equal w u
    | Error _ -> false
  in
  match stated with
  | Some t when reads_as_u t -> Ok t
  | _ -> (
      match Ymir_units.Fits_unit.print u with
      | Ok t -> Ok t
      | Error e -> Error ("Wcs.write: " ^ e))

let cdelt_cards sfx ax = function
  | None -> []
  | Some d -> List.mapi (fun k x -> default x 1. (axis_key "CDELT" ax.(k) sfx)) (Array.to_list d)

(* [sky_tail kw sfx ax ~sip ~cdelt u steps] spells a celestial pair's tail,
   its intermediate plane in [u]. *)
let sky_tail kw sfx ax ~sip ~cdelt u steps =
  let lat_first, rest =
    match steps with
    | Axes ([| 1; 0 |], 0) :: rest -> (true, rest)
    | rest -> (false, rest)
  in
  let tpv, rest =
    match rest with
    | Tpv (pv, stated) :: rest -> (Some (pv, stated), rest)
    | rest -> (None, rest)
  in
  let* c =
    match rest with
    | [ Celestial c ] -> Ok c
    | Celestial _ :: s :: _ | s :: _ -> unspellable s
    | [] -> ends ()
  in
  let theta0_default = Projection.theta0 c.code in
  let* () =
    match tpv with
    | Some _ when c.code <> Transform.Tan ->
        Error
          (strf "Wcs.write: FITS spells TPV only before TAN, not before %s"
             (Transform.code_name c.code))
    | Some _ when c.native.(0) <> 0. || c.native.(1) <> theta0_default ->
        Error
          "Wcs.write: FITS spells TPV only with the projection's own native \
           reference point"
    | _ -> Ok ()
  in
  let lng, lat = if lat_first then (ax.(1), ax.(0)) else (ax.(0), ax.(1)) in
  let* cunit_lng, cunit_lat =
    let spelled k = cunit_text kw (axis_key "CUNIT" k sfx) u in
    if not (U.convertible u U.radian) then
      Error
        (Format.asprintf
           "Wcs.write: the intermediate plane is in %a, which no CUNIT spells"
           U.pp u)
    else
      let* a = spelled lng in
      let* b = spelled lat in
      Ok (a, b)
  in
  let* pair = pair_of c.frame in
  let ctype prefix =
    S (ctype_text prefix c.code ~tpv:(Option.is_some tpv) ~sip)
  in
  let crval = Nx.to_array (Quantity.value u c.crval) in
  let phi0 = c.native.(0) and theta0 = c.native.(1) in
  let delta0 = U.ratio Nx.float64 u U.degree *. crval.(1) in
  let pv_cards =
    match tpv with
    | Some (pv, stated) ->
        let count = Distortion.tpv_count in
        term_cards
          ~key:(fun k ->
            pv_key (if k < count then lng else lat) (k mod count) sfx)
          ~default:tpv_default pv stated
    | None ->
        let first = Projection.first c.code in
        let defaults = Projection.defaults c.code in
        term_cards
          ~key:(fun k -> pv_key lat (first + k) sfx)
          ~default:(fun k -> defaults.(k))
          c.pv c.stated
        @ [
            default phi0 0. (pv_key lng 1 sfx);
            default theta0 theta0_default (pv_key lng 2 sfx);
          ]
  in
  let pole_alt m = if Option.is_some tpv then None else Some (pv_key lng m sfx) in
  let deg_unit = U.equal u U.degree in
  Ok
    (cdelt_cards sfx ax cdelt
    @ [
        card (axis_key "CTYPE" lng sfx) (ctype pair.lon);
        card (axis_key "CTYPE" lat sfx) (ctype pair.lat);
        card ~optional:deg_unit (axis_key "CUNIT" lng sfx) (S cunit_lng);
        card ~optional:deg_unit (axis_key "CUNIT" lat sfx) (S cunit_lat);
        default crval.(0) 0. (axis_key "CRVAL" lng sfx);
        default crval.(1) 0. (axis_key "CRVAL" lat sfx);
      ]
    @ pv_cards
    @ pole_cards kw ("LONPOLE" ^ sfx) (pole_alt 3) c.lonpole
        (Some (default_lonpole ~delta0 ~theta0 ~phi0))
    @ pole_cards kw ("LATPOLE" ^ sfx) (pole_alt 4) c.latpole
        (default_latpole ~delta0 ~theta0 ~phi0 ~lonpole:c.lonpole)
    @ frame_cards kw sfx c.frame)

(* [linear_tail kw sfx a ~cdelt u steps] spells a linear axis's tail. Its
   CTYPE stays [kw]'s, and its CUNIT [kw]'s where that reads as [u]. *)
let linear_tail kw sfx a ~cdelt u steps =
  let* crval =
    match steps with
    | [ At (r, ru) ] when U.equal ru u -> Ok r.(0)
    | s :: _ -> unspellable s
    | [] -> ends ()
  in
  let* ctype = kw.text (axis_key "CTYPE" a sfx) in
  let ctype = String.trim (Option.value ctype ~default:"") in
  let typed = type_unit ctype in
  let key = axis_key "CUNIT" a sfx in
  let* cunit = cunit_text kw key u in
  let* () =
    match typed with
    | Some t when not (U.convertible u t) ->
        Error
          (Format.asprintf "Wcs.write: a linear axis of type '%s' is not in %a"
             ctype U.pp u)
    | _ -> Ok ()
  in
  Ok
    (cdelt_cards sfx [| a |] cdelt
    @ [
        card ~optional:true (axis_key "CTYPE" a sfx) (S ctype);
        card
          ~optional:(U.equal u (Option.value typed ~default:U.one))
          key (S cunit);
        default crval 0. (axis_key "CRVAL" a sfx);
      ])

let tail kw sfx ax ~sip ~cdelt u steps =
  if Array.length ax = 2 then sky_tail kw sfx ax ~sip ~cdelt u steps
  else linear_tail kw sfx ax.(0) ~cdelt u steps

(* Writing N axes

   [size steps] is the number of axes [steps] describe. A product's split
   indexes the tensor's axes, FITS's reversed, and a coupled form's split the
   matrix's components, in FITS order. *)

let rec size = function
  | [ Split_at _; Sides (a, b) ] ->
      let* a = size a in
      let* b = size b in
      Ok (a + b)
  | Axes (perm, 1) :: _ -> Ok (Array.length perm)
  | s :: _ -> unspellable s
  | [] -> ends ()

(* [sides ~fits ax first] is the axes of [ax] that [split first] sends to
   each side, ascending. *)
let sides ~fits ax first =
  let m = Array.length ax in
  let pick ks =
    let a = Array.map (fun k -> if fits then ax.(k) else ax.(m - 1 - k)) ks in
    Array.sort compare a;
    a
  in
  (pick first, pick (Transform.others first m))

(* [ones steps] is whether every tail of a coupled form scales by 1. *)
let rec ones = function
  | [ Split_at _; Sides (a, b) ] -> ones a && ones b
  | Scale (d, _) :: _ -> Array.for_all (fun x -> x = 1.) d
  | _ -> false

let rec spell kw sfx ax steps =
  match steps with
  | [ Split_at first; Sides (a, b) ] ->
      let ax_a, ax_b = sides ~fits:false ax first in
      let* fa, ca = spell kw sfx ax_a a in
      let* fb, cb = spell kw sfx ax_b b in
      let* () =
        if fa = fb then Ok ()
        else
          Error
            "Wcs.write: one side spells CD and the other PC, and FITS reads \
             one"
      in
      let p = match fa with `Cd -> "CD" | `Pc -> "PC" in
      (* The elements between the sides are 0. *)
      let cross =
        List.filter_map
          (fun (i, j) ->
            if Array.mem i ax_a = Array.mem j ax_a then None
            else Some (default 0. 0. (matrix_key p sfx (i, j))))
          (elements ax)
      in
      Ok (fa, ca @ cb @ cross)
  | Axes (_, 1) :: Shift _ :: Linear (_, u) :: (Split_at _ :: _ as rest)
    when U.equal u U.one ->
      (* Under CD each CDELT is 1, so a coupled form whose sides scale by 1
         spells PC with CDELT 1 or CD: [kw]'s matrix chooses. *)
      let cd =
        ones rest
        && List.exists (fun e -> has kw.float (matrix_key "CD" sfx e)) (elements ax)
      in
      let* head, family, _, _, rest' = head kw sfx ~cd ax steps in
      let* tails = tails kw sfx ~cd ax rest' in
      Ok (family, head @ tails)
  | _ -> (
      let* head, family, cd_unit, sip, rest = head kw sfx ~cd:false ax steps in
      let* tail =
        match (cd_unit, rest) with
        | Some u, rest -> tail kw sfx ax ~sip ~cdelt:None u rest
        | None, Scale (d, u) :: rest -> tail kw sfx ax ~sip ~cdelt:(Some d) u rest
        | None, s :: _ -> unspellable s
        | None, [] -> ends ()
      in
      Ok (family, head @ tail))

(* [tails kw sfx ~cd ax steps] spells a coupled form's tails, CDELT among
   them unless the head is CD. *)
and tails kw sfx ~cd ax steps =
  match steps with
  | [ Split_at first; Sides (a, b) ] ->
      let ax_a, ax_b = sides ~fits:true ax first in
      let* ca = tails kw sfx ~cd ax_a a in
      let* cb = tails kw sfx ~cd ax_b b in
      Ok (ca @ cb)
  | Scale (d, u) :: rest ->
      tail kw sfx ax ~sip:false ~cdelt:(if cd then None else Some d) u rest
  | s :: _ -> unspellable s
  | [] -> ends ()

(* [present kw v k] is [true] if [k] is present, looked up as [v]'s kind. *)
let present kw v k =
  match v with
  | F _ -> has kw.float k
  | I _ -> has kw.int k
  | S _ -> has kw.text k

(* The keywords of an alternate's description of [n] axes the writer owns,
   each with whether [kw] has it. *)
let owned kw sfx n =
  let ax = Array.init n succ in
  let per_axis p = List.map (fun i -> axis_key p i sfx) (Array.to_list ax) in
  let matrix p = List.map (matrix_key p sfx) (elements ax) in
  let pv i = List.init pv_terms (fun m -> pv_key i m sfx) in
  let sip p =
    List.concat
      (List.init (sip_max_order + 1) (fun i ->
           List.init (sip_max_order + 1 - i) (fun j -> strf "%s_%d_%d" p i j)))
  in
  let looked find keys = List.map (fun k -> (k, has find)) keys in
  let primary =
    if sfx <> "" then []
    else
      looked kw.float
        (List.map (fun i -> axis_key "CROTA" i "") (Array.to_list ax)
        @ List.concat_map sip [ "A"; "B"; "AP"; "BP" ])
      @ looked kw.int [ "A_ORDER"; "B_ORDER"; "AP_ORDER"; "BP_ORDER" ]
  in
  looked kw.text (List.concat_map per_axis [ "CTYPE"; "CUNIT" ])
  @ looked kw.float
      (List.concat_map per_axis [ "CRPIX"; "CRVAL"; "CDELT" ]
      @ matrix "CD" @ matrix "PC"
      @ List.concat_map pv (Array.to_list ax)
      @ [ "LONPOLE" ^ sfx; "LATPOLE" ^ sfx ])
  @ primary

let reads_equal kw { key; value; _ } =
  match value with
  | F x -> ( match kw.float key with Ok (Some y) -> same x y | _ -> false)
  | S s -> (
      match kw.text key with Ok (Some y) -> String.equal s y | _ -> false)
  | I n -> ( match kw.int key with Ok (Some m) -> n = m | _ -> false)

let edit { key; value; _ } =
  match value with
  | F x -> Float (key, x)
  | I n -> Int (key, n)
  | S s -> Text (key, s)

let write ?alt ?window t kw =
  let sfx = suffix "write" alt in
  if not (unbatched t) then
    Error "Wcs.write: a batch of transforms has no single header"
  else
    let steps = steps t in
    let* n = size steps in
    let* _, cards = spell kw sfx (Array.init n succ) steps in
    (* The axes' types and units come first, as a header states them. *)
    let cards =
      let named c =
        String.starts_with ~prefix:"CTYPE" c.key
        || String.starts_with ~prefix:"CUNIT" c.key
      in
      let first, rest = List.partition named cards in
      first @ rest
    in
    let start =
      match window with
      | None -> fun _ -> 0.
      | Some w ->
          if Array.length w <> n then
            invalid_arg "Wcs.write: ~window needs a range for each axis";
          (* FITS axis [i] is tensor axis [n - i]. *)
          fun i -> float_of_int (fst w.(n - i))
    in
    let crpix = List.init n (fun k -> (axis_key "CRPIX" (k + 1) sfx, k + 1)) in
    let cards =
      List.map
        (fun c ->
          match (c.value, List.assoc_opt c.key crpix) with
          | F x, Some i when window <> None -> default (x -. start i) 0. c.key
          | _ -> c)
        cards
    in
    (* A celestial pair alone needs no WCSAXES; other descriptions state
       theirs unless NAXIS does. *)
    let* cards =
      match steps with
      | Axes ([| 1; 0 |], 1) :: rest
        when not (List.exists (function Split_at _ -> true | _ -> false) rest)
        ->
          Ok cards
      | _ ->
          let* naxis = kw.int "NAXIS" in
          let given = Option.value naxis ~default:2 in
          Ok (card ~optional:(given = n) ("WCSAXES" ^ sfx) (I n) :: cards)
    in
    let unspelled =
      List.find_map
        (function
          | { key; value = F x; _ } when not (Float.is_finite x) ->
              Some (key, x)
          | _ -> None)
        cards
    in
    match unspelled with
    | Some (key, x) ->
        Error
          (strf "Wcs.write: FITS cannot spell %s = %s" key (Transform.num x))
    | None ->
        let set =
          List.filter_map
            (fun c ->
              let needed =
                if present kw c.value c.key then not (reads_equal kw c)
                else not c.optional
              in
              if needed then Some (edit c) else None)
            cards
        in
        let wanted = List.map (fun c -> c.key) cards in
        let removed =
          List.filter_map
            (fun (k, has) ->
              if has k && not (List.mem k wanted) then Some (Remove k)
              else None)
            (owned kw sfx n)
        in
        Ok (set @ removed)
