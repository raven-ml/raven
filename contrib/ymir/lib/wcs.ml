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

(* Axes *)

type axes = {
  pair : pair;
  code : Transform.code;
  tpv : bool;  (** The CTYPE says TPV. *)
  sip : bool;  (** The CTYPE says -SIP. *)
  lng : int;  (** The longitude's FITS axis, 1 or 2. *)
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

let read_axes kw sfx =
  let* n =
    let* w = kw.int ("WCSAXES" ^ sfx) in
    match w with
    | Some n -> Ok n
    | None ->
        Result.map (Option.value ~default:2) (kw.int "NAXIS")
  in
  if n <> 2 then
    fail kw
      "%s axes: a celestial pair is read, and a spectral, time or other \
       third axis is not"
      (string_of_int n)
  else
    let ctype i =
      let key = strf "CTYPE%d%s" i sfx in
      let* s = kw.text key in
      match Option.map split_ctype s with
      | Some (Some (prefix, code, rest)) -> Ok (key, s, prefix, code, rest)
      | Some None | None ->
          fail kw "%s = %s: not a celestial axis" key
            (match s with Some s -> strf "'%s'" s | None -> "absent")
    in
    let* k1, s1, p1, c1, r1 = ctype 1 in
    let* k2, s2, p2, c2, r2 = ctype 2 in
    let* pair, lng, lat =
      match
        List.find_map
          (fun q ->
            if q.lon = p1 && q.lat = p2 then Some (q, 1, 2)
            else if q.lat = p1 && q.lon = p2 then Some (q, 2, 1)
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
      if sip1 = sip2 then Ok ()
      else fail kw "%s and %s disagree on -SIP" k1 k2
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

let sip_matrix kw p order =
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
  Ok (Nx.create Nx.float64 [| n; n |] m)

let sip_order kw key =
  let* o = kw.int key in
  match o with
  | Some o when o < 0 || o > sip_max_order ->
      fail kw "%s = %d: SIP's orders run from 0 to %d" key o sip_max_order
  | o -> Ok o

let read_sip kw =
  let pair a b =
    let* oa = sip_order kw (a ^ "_ORDER") in
    let* ob = sip_order kw (b ^ "_ORDER") in
    match (oa, ob) with
    | None, None -> Ok None
    | Some oa, Some ob ->
        let* ma = sip_matrix kw a oa in
        let* mb = sip_matrix kw b ob in
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
let unread_distortion kw sfx =
  if sfx <> "" then Ok ()
  else
    match
      List.find_opt (has kw.text) [ "CPDIS1"; "CPDIS2"; "D2IMDIS1"; "D2IMDIS2" ]
    with
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
        (s, strf "%s (EQUINOX %g without RADESYS)" s e)
  in
  match (axes.pair.lon, sys) with
  | "GLON", _ -> Ok (Frame Frame.galactic)
  | "SLON", _ -> Ok (Frame Frame.supergalactic)
  | "RA", "ICRS" -> Ok (Frame Frame.icrs)
  | "RA", "FK5" -> (
      match eq with
      | None | Some 2000. -> Ok (Frame Frame.fk5_j2000)
      | Some e -> time_stage kw (strf "FK5 at equinox %g" e))
  | "ELON", "ICRS" -> (
      match eq with
      | None | Some 2000. -> Ok (Frame Frame.ecliptic_j2000)
      | Some e -> time_stage kw (strf "the ecliptic at equinox %g" e))
  | "ELON", "FK5" -> time_stage kw "the IAU 1976 ecliptic (ELON under FK5)"
  | _ -> time_stage kw stated

let frame_text (Frame f) =
  match f with
  | Frame.Icrs -> "ICRS"
  | Frame.Fk5_j2000 -> "FK5 at equinox 2000.0"
  | Frame.Galactic -> "Galactic"
  | Frame.Ecliptic_j2000 -> "the J2000 ecliptic"
  | Frame.Supergalactic -> "supergalactic"

(* [same f g] is [g] at [f]'s type when they are one frame. *)
let same : type f g. f Frame.t -> g Frame.t -> f Frame.t option =
 fun f g ->
  match (f, g) with
  | Frame.Icrs, Frame.Icrs -> Some f
  | Frame.Fk5_j2000, Frame.Fk5_j2000 -> Some f
  | Frame.Galactic, Frame.Galactic -> Some f
  | Frame.Ecliptic_j2000, Frame.Ecliptic_j2000 -> Some f
  | Frame.Supergalactic, Frame.Supergalactic -> Some f
  | _ -> None

(* Reading *)

(* The angles a celestial CUNIT names. *)
let angles =
  Ymir_units.Vocabulary.(
    v
      [
        ("deg", Bare, U.degree);
        ("arcmin", Bare, U.arcminute);
        ("arcsec", Bare, U.arcsecond);
        ("mas", Bare, U.milli U.arcsecond);
        ("rad", Bare, U.radian);
      ])

let unit_of kw sfx axes =
  let cunit i =
    let key = strf "CUNIT%d%s" i sfx in
    let* s = kw.text key in
    match s with
    | None | Some "" -> Ok (U.degree, key)
    | Some s -> (
        match Ymir_units.Vocabulary.lookup angles s with
        | Some u -> Ok (u, key)
        | None -> fail kw "%s = '%s' is not a celestial unit" key s)
  in
  let* u1, k1 = cunit axes.lng in
  let* u2, k2 = cunit axes.lat in
  if U.equal u1 u2 then Ok u1
  else fail kw "%s and %s differ; one unit for both axes is read" k1 k2

let read_matrix kw sfx =
  let key p i j = strf "%s%d_%d%s" p i j sfx in
  let keys p = [ key p 1 1; key p 1 2; key p 2 1; key p 2 2 ] in
  let has_matrix p = List.exists (has kw.float) (keys p) in
  let crota = List.find_opt (has kw.float) [ "CROTA1"; "CROTA2" ] in
  let crota = if sfx = "" then crota else None in
  let entries p default =
    List.fold_right
      (fun (i, j) acc ->
        let* acc = acc in
        let* v = float_or kw (key p i j) (default i j) in
        Ok (v :: acc))
      [ (1, 1); (1, 2); (2, 1); (2, 2) ]
      (Ok [])
  in
  let delta i j = if i = j then 1. else 0. in
  match (has_matrix "CD", has_matrix "PC", crota) with
  | true, true, _ -> fail kw "a header with both CD and PC is ambiguous"
  | true, _, Some k | _, true, Some k ->
      fail kw "%s beside %s is ambiguous" k
        (if has_matrix "CD" then "CD" else "PC")
  | true, false, None ->
      let* cd = entries "CD" (fun _ _ -> 0.) in
      Ok (`Cd cd)
  | false, true, None ->
      let* pc = entries "PC" delta in
      Ok (`Pc pc)
  | false, false, None -> Ok (`Pc [ 1.; 0.; 0.; 1. ])
  | false, false, Some _ -> Ok `Crota

(* PC from CROTA, FITS WCS Paper II eq. 189: [ρ] is the latitude axis's
   CROTA, and the ratio of CDELTs keeps the matrix a rotation of the scaled
   axes. *)
let pc_of_crota kw axes (d1, d2) =
  let* rho = float_or kw (strf "CROTA%d" axes.lat) 0. in
  let r = rho *. Float.pi /. 180. in
  let c = Float.cos r and s = Float.sin r in
  Ok [ c; -.s *. d2 /. d1; s *. d1 /. d2; c ]

(* LONPOLE's default (Paper II §2.4): 180° when the reference point is below
   the native reference's latitude, else 0°, plus φ₀. *)
let default_lonpole ~delta0 ~theta0 ~phi0 =
  (if delta0 < theta0 then 180. else 0.) +. phi0

let default_latpole = 90.

(* [pole kw key alt default] reads LONPOLE or LATPOLE, which FITS also spells
   PVi_3 and PVi_4 on the longitude axis [i] (Paper II §2.5); under TPV
   those are TPV's terms, and [alt] is [None]. *)
let pole kw key alt default =
  let* a = kw.float key in
  let* b =
    match alt with None -> Ok None | Some alt -> kw.float alt
  in
  match (a, b, alt) with
  | Some x, Some y, Some alt when not (Float.equal x y) ->
      fail kw "%s = %g and %s = %g spell one value and disagree" key x alt y
  | Some x, _, _ | None, Some x, _ -> Ok x
  | None, None, _ -> Ok default

let read ?alt (frame : 'f Frame.t) kw =
  let f64 = Nx.float64 in
  let sfx = suffix "read" alt in
  let* axes = read_axes kw sfx in
  let* () = unread_distortion kw sfx in
  let* terms = read_terms kw sfx axes in
  let* sip =
    if axes.sip || (sfx = "" && has kw.int "A_ORDER") then read_sip kw
    else Ok None
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
  let* crpix1 = float_or kw ("CRPIX1" ^ sfx) 0. in
  let* crpix2 = float_or kw ("CRPIX2" ^ sfx) 0. in
  let* crval_lng = float_or kw (strf "CRVAL%d%s" axes.lng sfx) 0. in
  let* crval_lat = float_or kw (strf "CRVAL%d%s" axes.lat sfx) 0. in
  let* matrix = read_matrix kw sfx in
  let plane x = Quantity.v u x in
  let bare x = Quantity.v U.one x in
  let* matrix_stages =
    let cdelt () =
      let* d1 = float_or kw ("CDELT1" ^ sfx) 1. in
      let* d2 = float_or kw ("CDELT2" ^ sfx) 1. in
      Ok (d1, d2)
    in
    let pc_stage pc (d1, d2) =
      Transform.(
        linear (bare (Nx.create f64 [| 2; 2 |] (Array.of_list pc)))
        >> scale (plane (Nx.create f64 [| 2 |] [| d1; d2 |])))
    in
    match matrix with
    | `Cd cd ->
        Ok
          (Transform.linear
             (plane (Nx.create f64 [| 2; 2 |] (Array.of_list cd))))
    | `Pc pc ->
        let* d = cdelt () in
        Ok (pc_stage pc d)
    | `Crota ->
        let* d = cdelt () in
        let* pc = pc_of_crota kw axes d in
        Ok (pc_stage pc d)
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
          Transform.tpv ~stated
            (Nx.create f64 [| 2; Distortion.tpv_count |] pv) )
  in
  let delta0 = U.ratio Nx.float64 u U.degree *. crval_lat in
  let alt m =
    match terms with
    | Projection _ -> Some (pv_key axes.lng m sfx)
    | Tpv _ -> None
  in
  let* lonpole =
    pole kw ("LONPOLE" ^ sfx) (alt 3) (default_lonpole ~delta0 ~theta0 ~phi0)
  in
  let* latpole = pole kw ("LATPOLE" ^ sfx) (alt 4) default_latpole in
  let deg x = Quantity.v U.degree x in
  let swap =
    if axes.lng = 1 then Transform.id else Transform.axes [| 1; 0 |] ~origin:0
  in
  let projection =
    Transform.celestial ~stated axes.code frame
      ~pv:(Nx.create f64 [| Array.length pv |] pv)
      ~native:(deg (Nx.create f64 [| 2 |] [| phi0; theta0 |]))
      ~crval:(Quantity.v u (Nx.create f64 [| 2 |] [| crval_lng; crval_lat |]))
      ~lonpole:(deg (Nx.scalar f64 lonpole))
      ~latpole:(deg (Nx.scalar f64 latpole))
  in
  let sip_stage = Option.value sip ~default:Transform.id in
  Ok
    Transform.(
      axes [| 1; 0 |] ~origin:1
      >> shift (bare (Nx.create f64 [| 2 |] [| crpix1; crpix2 |]))
      >> sip_stage >> matrix_stages >> swap >> tpv_stage >> projection)

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

type celestial = {
  code : Transform.code;
  frame : frame;
  crval : (float, Nx.float64_elt) Nx.t Quantity.t;
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
type step =
  | Axes of int array * int
  | Shift of float array
  | Linear of float array * U.t
  | Scale of float array * U.t
  | Sip of sip
  | Tpv of float array * int array
  | Celestial of celestial
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

let step : type a b. (a, b) Transform.stage -> step = function
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
  | s -> Unspelled (Transform.stage_name s)

let rec steps : type a b. (a, b) Transform.t -> step list = function
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
let rec unbatched : type a b. (a, b) Transform.t -> bool = function
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
        | Rotate _ -> true
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
  in
  Error (strf "Wcs.write: FITS cannot spell the %s stage there" name)

let ends () =
  Error "Wcs.write: the transform ends before its celestial stage"

let pair_of (Frame f) =
  let prefix lon = List.find (fun p -> p.lon = lon) pairs in
  match f with
  | Frame.Icrs | Frame.Fk5_j2000 -> prefix "RA"
  | Frame.Galactic -> prefix "GLON"
  | Frame.Ecliptic_j2000 -> prefix "ELON"
  | Frame.Supergalactic -> prefix "SLON"

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

let index_pairs = [ (1, 1); (1, 2); (2, 1); (2, 2) ]

(* A pole is written with the spelling the header uses: PVi_3 or PVi_4 when
   it has one, and then LONPOLE or LATPOLE only where it has that too.
   Under TPV, [alt] is [None]: those are TPV's terms. *)
let pole_cards kw key alt x d =
  match alt with
  | Some alt when has kw.float alt ->
      [ card ~optional:true key (F x); card ~optional:true alt (F x) ]
  | _ -> [ default x d key ]

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

(* [cards kw sfx window steps] is the cards [steps] spell, or [Error] naming a
   stage FITS cannot spell there. *)
let cards kw sfx window steps =
  let* crpix, rest =
    match steps with
    | Axes ([| 1; 0 |], 1) :: Shift crpix :: rest -> Ok (crpix, rest)
    | Axes ([| 1; 0 |], 1) :: s :: _ | s :: _ -> unspellable s
    | [] -> ends ()
  in
  let sip, rest =
    match rest with Sip s :: rest -> (Some s, rest) | rest -> (None, rest)
  in
  let* () =
    if Option.is_some sip && sfx <> "" then
      Error
        "Wcs.write: SIP's keywords belong to the primary description"
    else Ok ()
  in
  let* linear, rest =
    match rest with
    | Linear (pc, u) :: Scale (d, du) :: rest when U.equal u U.one ->
        Ok (`Pc (pc, d, du), rest)
    | Scale (d, du) :: rest -> Ok (`Pc ([| 1.; 0.; 0.; 1. |], d, du), rest)
    | Linear (cd, u) :: rest -> Ok (`Cd (cd, u), rest)
    | s :: _ -> unspellable s
    | [] -> ends ()
  in
  let lat_first, rest =
    match rest with
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
          (strf
             "Wcs.write: FITS spells TPV only before TAN, not before \
              %s"
             (Transform.code_name c.code))
    | Some _ when c.native.(0) <> 0. || c.native.(1) <> theta0_default ->
        Error
          "Wcs.write: FITS spells TPV only with the projection's own \
           native reference point"
    | _ -> Ok ()
  in
  let u = match linear with `Pc (_, _, u) | `Cd (_, u) -> u in
  let* cunit =
    match Ymir_units.Vocabulary.spell angles u with
    | Some { decade = 0; words = [ { prefix = 0; symbol; num = 1; den = 1 } ] }
      ->
        Ok symbol
    | _ ->
        Error
          (Format.asprintf
             "Wcs.write: the intermediate plane is in %a, which no CUNIT \
              spells"
             U.pp u)
  in
  let crpix =
    match window with
    | None -> crpix
    | Some w ->
        if Array.length w <> 2 then
          invalid_arg "Wcs.write: ~window needs a range for each axis";
        (* FITS axis [k + 1] is tensor axis [1 - k]. *)
        Array.mapi (fun k r -> r -. float_of_int (fst w.(1 - k))) crpix
  in
  let lng, lat = if lat_first then (2, 1) else (1, 2) in
  let pair = pair_of c.frame in
  let tpv_present = Option.is_some tpv and sip_present = Option.is_some sip in
  let ctype prefix =
    S (ctype_text prefix c.code ~tpv:tpv_present ~sip:sip_present)
  in
  let crval = Nx.to_array (Quantity.value u c.crval) in
  let phi0 = c.native.(0) and theta0 = c.native.(1) in
  let delta0 = U.ratio Nx.float64 u U.degree *. crval.(1) in
  let at p (i, j) = strf "%s%d_%d%s" p i j sfx in
  let matrix p m d =
    List.map
      (fun (i, j) ->
        default m.((2 * (i - 1)) + (j - 1)) (d i j) (at p (i, j)))
      index_pairs
  in
  let linear_cards =
    match linear with
    | `Pc (pc, d, _) ->
        matrix "PC" pc (fun i j -> if i = j then 1. else 0.)
        @ [
            default d.(0) 1. ("CDELT1" ^ sfx);
            default d.(1) 1. ("CDELT2" ^ sfx);
          ]
    | `Cd (cd, _) ->
        (* A CD header needs one CD card to read as one. *)
        let some_cd =
          Array.exists (fun x -> x <> 0.) cd
          || List.exists (fun ij -> has kw.float (at "CD" ij)) index_pairs
        in
        List.map
          (fun c -> { c with optional = c.optional && some_cd })
          (matrix "CD" cd (fun _ _ -> 0.))
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
  let pole_alt m = if tpv_present then None else Some (pv_key lng m sfx) in
  let deg_unit = cunit = "deg" in
  Ok
    ([
       card (strf "CTYPE%d%s" lng sfx) (ctype pair.lon);
       card (strf "CTYPE%d%s" lat sfx) (ctype pair.lat);
       card ~optional:deg_unit (strf "CUNIT%d%s" lng sfx) (S cunit);
       card ~optional:deg_unit (strf "CUNIT%d%s" lat sfx) (S cunit);
       default crpix.(0) 0. ("CRPIX1" ^ sfx);
       default crpix.(1) 0. ("CRPIX2" ^ sfx);
     ]
    @ linear_cards
    @ [
        default crval.(0) 0. (strf "CRVAL%d%s" lng sfx);
        default crval.(1) 0. (strf "CRVAL%d%s" lat sfx);
      ]
    @ pv_cards
    @ pole_cards kw ("LONPOLE" ^ sfx) (pole_alt 3) c.lonpole
        (default_lonpole ~delta0 ~theta0 ~phi0)
    @ pole_cards kw ("LATPOLE" ^ sfx) (pole_alt 4) c.latpole default_latpole
    @ frame_cards kw sfx c.frame
    @ sip_cards)

(* [present kw v k] is [true] if [k] is present, looked up as [v]'s kind. *)
let present kw v k =
  match v with
  | F _ -> has kw.float k
  | I _ -> has kw.int k
  | S _ -> has kw.text k

(* The keywords of an alternate's description the writer owns, each with
   whether [kw] has it. *)
let owned kw sfx =
  let per_axis p = List.map (fun i -> strf "%s%d%s" p i sfx) [ 1; 2 ] in
  let matrix p =
    List.map (fun (i, j) -> strf "%s%d_%d%s" p i j sfx) index_pairs
  in
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
        ([ "CROTA1"; "CROTA2" ] @ List.concat_map sip [ "A"; "B"; "AP"; "BP" ])
      @ looked kw.int [ "A_ORDER"; "B_ORDER"; "AP_ORDER"; "BP_ORDER" ]
  in
  looked kw.text (List.concat_map per_axis [ "CTYPE"; "CUNIT" ])
  @ looked kw.float
      (List.concat_map per_axis [ "CRPIX"; "CRVAL"; "CDELT" ]
      @ matrix "CD" @ matrix "PC" @ pv 1 @ pv 2
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
    let* cards = cards kw sfx window (steps t) in
    let unspelled =
      List.find_map
        (function
          | { key; value = F x; _ } when not (Float.is_finite x) ->
              Some (key, x)
          | _ -> None)
        cards
    in
    match unspelled with
    | Some (key, x) -> Error (strf "Wcs.write: FITS cannot spell %s = %g" key x)
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
            (owned kw sfx)
        in
        Ok (set @ removed)
