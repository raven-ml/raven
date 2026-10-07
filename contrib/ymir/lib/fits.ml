(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir_units.Quantity
module U = Ymir_units.Unit
include Ymir_fits.Fits

let strf = Printf.sprintf
let ( let* ) = Result.bind

(* [place h] starts a message about [h] with its EXTNAME, as in ["SCI: "]. *)
let place h =
  match Header.find Value.string "EXTNAME" h with
  | Ok (Some n) when n <> "" -> n ^ ": "
  | _ -> ""

let fail h fmt = Printf.ksprintf (fun s -> Error (place h ^ s)) fmt

module Wcs = struct
  (* Keywords *)

  let suffix fn = function
    | None | Some ' ' -> ""
    | Some c when c >= 'A' && c <= 'Z' -> String.make 1 c
    | Some c ->
        invalid_arg (strf "Fits.Wcs.%s: alternate %C is not ' ' or A-Z" fn c)

  (* A celestial axis pair: the CTYPE prefixes of its longitude and latitude. *)
  type pair = { lon : string; lat : string }

  let pairs =
    [
      { lon = "RA"; lat = "DEC" };
      { lon = "GLON"; lat = "GLAT" };
      { lon = "ELON"; lat = "ELAT" };
      { lon = "SLON"; lat = "SLAT" };
    ]

  (* The zenithal and cylindrical codes beyond TAN and ARC. *)
  let later_codes =
    [
      "AZP";
      "SZP";
      "STG";
      "SIN";
      "ZPN";
      "ZEA";
      "AIR";
      "CYP";
      "CEA";
      "CAR";
      "MER";
    ]

  let ctype_text pair_prefix code =
    let p = pair_prefix ^ String.make (4 - String.length pair_prefix) '-' in
    p ^ "-" ^ Transform.code_name code

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

  let find_float h k = Header.find Value.float k h
  let find_string h k = Header.find Value.string k h

  let float_or h k default =
    let* v = find_float h k in
    Ok (Option.value v ~default)

  let present h k =
    match Header.find Value.text k h with Ok (Some _) -> true | _ -> false

  (* Axes *)

  type axes = {
    pair : pair;
    code : Transform.code;
    lng : int;  (** The longitude's FITS axis, 1 or 2. *)
    lat : int;
  }

  let read_code h key s =
    match String.trim s with
    | "TAN" -> Ok Transform.Tan
    | "ARC" -> Ok Transform.Arc
    | "TPV" -> fail h "%s = '%s': the TPV distortion is not read yet" key s
    | c when List.mem c later_codes ->
        fail h "%s: the %s projection is not read yet" key c
    | c -> fail h "%s: %S is not a celestial projection code" key c

  let read_axes h sfx =
    let* n =
      let* w = Header.find Value.int ("WCSAXES" ^ sfx) h in
      match w with
      | Some n -> Ok n
      | None ->
          Result.map (Option.value ~default:2) (Header.find Value.int "NAXIS" h)
    in
    if n <> 2 then
      fail h
        "%s axes: a celestial pair is read, and a spectral, time or other \
         third axis is not"
        (string_of_int n)
    else
      let ctype i =
        let key = strf "CTYPE%d%s" i sfx in
        let* s = find_string h key in
        match Option.map split_ctype s with
        | Some (Some (prefix, code, rest)) -> Ok (key, s, prefix, code, rest)
        | Some None | None ->
            fail h "%s = %s: not a celestial axis" key
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
            fail h "%s = '%s' and %s = '%s' are not a celestial pair ymir reads"
              k1 (Option.get s1) k2 (Option.get s2)
      in
      let* code = read_code h k1 c1 in
      let* code' = read_code h k2 c2 in
      let* () =
        if code = code' then Ok ()
        else fail h "%s and %s name different projections" k1 k2
      in
      let suffix_error k s r =
        match r with
        | "" -> Ok ()
        | "-SIP" -> fail h "%s = '%s': SIP distortion is not read yet" k s
        | r -> fail h "%s = '%s': the axis algorithm %S is not read" k s r
      in
      let* () = suffix_error k1 (Option.get s1) r1 in
      let* () = suffix_error k2 (Option.get s2) r2 in
      Ok { pair; code; lng; lat }

  (* Keywords that need a stage this reader does not build. *)
  let unread h sfx axes =
    let lat_pv =
      List.init 100 (fun m -> strf "PV%d_%d%s" axes.lat m sfx)
      |> List.find_opt (present h)
    in
    let lng_pv =
      List.init 100 (fun m -> strf "PV%d_%d%s" axes.lng m sfx)
      |> List.find_opt (fun k ->
          present h k
          && not
               (List.mem k
                  (List.init 4 (fun m -> strf "PV%d_%d%s" axes.lng (m + 1) sfx))))
    in
    let distortion =
      if sfx <> "" then None
      else
        List.find_opt (present h)
          [ "A_ORDER"; "B_ORDER"; "CPDIS1"; "CPDIS2"; "D2IMDIS1"; "D2IMDIS2" ]
    in
    match (lat_pv, lng_pv, distortion) with
    | Some k, _, _ when axes.code = Transform.Tan ->
        fail h "%s: TAN with PV terms is the TPV distortion, not read yet" k
    | Some k, _, _ ->
        fail h "%s: the %s projection takes no parameter" k
          (Transform.code_name axes.code)
    | None, Some k, _ ->
        fail h
          "%s: only PV%d_1 to PV%d_4 (the native reference point, LONPOLE and \
           LATPOLE) are read on the longitude axis"
          k axes.lng axes.lng
    | None, None, Some k -> fail h "%s: this distortion is not read yet" k
    | None, None, None -> Ok ()

  (* Frames

     The CTYPE prefix chooses the system; RADESYS and EQUINOX qualify an
     equatorial or ecliptic one after FITS 4.0 §8.3's defaults. *)

  type frame = Frame : 'f Frame.t -> frame

  let radesys_key h sfx =
    if sfx = "" && (not (present h "RADESYS")) && present h "RADECSYS" then
      "RADECSYS"
    else "RADESYS" ^ sfx

  let equinox h sfx =
    let* e = find_float h ("EQUINOX" ^ sfx) in
    match e with
    | Some _ -> Ok e
    | None when sfx = "" -> find_float h "EPOCH"
    | None -> Ok None

  let time_stage h what = fail h "%s needs the time stage, not read yet" what

  let read_frame h sfx axes =
    let* sys = find_string h (radesys_key h sfx) in
    let* eq = equinox h sfx in
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
        | Some e -> time_stage h (strf "FK5 at equinox %g" e))
    | "ELON", "ICRS" -> (
        match eq with
        | None | Some 2000. -> Ok (Frame Frame.ecliptic_j2000)
        | Some e -> time_stage h (strf "the ecliptic at equinox %g" e))
    | "ELON", "FK5" -> time_stage h "the IAU 1976 ecliptic (ELON under FK5)"
    | _ -> time_stage h stated

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

  let unit_of h sfx axes =
    let cunit i =
      let key = strf "CUNIT%d%s" i sfx in
      let* s = find_string h key in
      match s with
      | None | Some "" -> Ok (U.degree, key)
      | Some s -> (
          match Unit.parse s with
          | Ok u when U.convertible u U.radian -> Ok (u, key)
          | Ok _ -> fail h "%s = '%s' is not an angle" key s
          | Error e -> fail h "%s: %s" key e)
    in
    let* u1, k1 = cunit axes.lng in
    let* u2, k2 = cunit axes.lat in
    if U.equal u1 u2 then Ok u1
    else fail h "%s and %s differ; one unit for both axes is read" k1 k2

  let read_matrix h sfx =
    let key p i j = strf "%s%d_%d%s" p i j sfx in
    let keys p = [ key p 1 1; key p 1 2; key p 2 1; key p 2 2 ] in
    let has p = List.exists (present h) (keys p) in
    let crota = List.find_opt (present h) [ "CROTA1"; "CROTA2" ] in
    let crota = if sfx = "" then crota else None in
    let entries p default =
      List.fold_right
        (fun (i, j) acc ->
          let* acc = acc in
          let* v = float_or h (key p i j) (default i j) in
          Ok (v :: acc))
        [ (1, 1); (1, 2); (2, 1); (2, 2) ]
        (Ok [])
    in
    let delta i j = if i = j then 1. else 0. in
    match (has "CD", has "PC", crota) with
    | true, true, _ -> fail h "a header with both CD and PC is ambiguous"
    | true, _, Some k | _, true, Some k ->
        fail h "%s beside %s is ambiguous" k (if has "CD" then "CD" else "PC")
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
  let pc_of_crota h axes (d1, d2) =
    let* rho = float_or h (strf "CROTA%d" axes.lat) 0. in
    let r = rho *. Float.pi /. 180. in
    let c = Float.cos r and s = Float.sin r in
    Ok [ c; -.s *. d2 /. d1; s *. d1 /. d2; c ]

  let read_native h sfx axes =
    let* phi0 = float_or h (strf "PV%d_1%s" axes.lng sfx) 0. in
    let* theta0 = float_or h (strf "PV%d_2%s" axes.lng sfx) 90. in
    Ok (phi0, theta0)

  (* LONPOLE's default (Paper II §2.4): 180° when the reference point is below
     the native reference's latitude, else 0°, plus φ₀. *)
  let default_lonpole ~delta0 ~theta0 ~phi0 =
    (if delta0 < theta0 then 180. else 0.) +. phi0

  let default_latpole = 90.

  (* [pole h key alt default] reads LONPOLE or LATPOLE, which FITS also spells
     PVi_3 and PVi_4 on the longitude axis [i] (Paper II §2.5). *)
  let pole h key alt default =
    let* a = find_float h key in
    let* b = find_float h alt in
    match (a, b) with
    | Some x, Some y when not (Float.equal x y) ->
        fail h "%s = %g and %s = %g spell one value and disagree" key x alt y
    | Some x, _ | None, Some x -> Ok x
    | None, None -> Ok default

  let read ?alt (frame : 'f Frame.t) h =
    let f64 = Nx.float64 in
    let sfx = suffix "read" alt in
    let* axes = read_axes h sfx in
    let* () = unread h sfx axes in
    let* (Frame found as fr) = read_frame h sfx axes in
    let* frame =
      match same frame found with
      | Some f -> Ok f
      | None ->
          fail h "%s; the caller expects %s. Read with Frame.%s."
            (frame_text fr) (Frame.name frame) (Frame.name found)
    in
    let* u = unit_of h sfx axes in
    let* crpix1 = float_or h ("CRPIX1" ^ sfx) 0. in
    let* crpix2 = float_or h ("CRPIX2" ^ sfx) 0. in
    let* crval_lng = float_or h (strf "CRVAL%d%s" axes.lng sfx) 0. in
    let* crval_lat = float_or h (strf "CRVAL%d%s" axes.lat sfx) 0. in
    let* matrix = read_matrix h sfx in
    let plane x = Quantity.v u x in
    let bare x = Quantity.v U.one x in
    let* matrix_stages =
      let cdelt () =
        let* d1 = float_or h ("CDELT1" ^ sfx) 1. in
        let* d2 = float_or h ("CDELT2" ^ sfx) 1. in
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
          let* pc = pc_of_crota h axes d in
          Ok (pc_stage pc d)
    in
    let* phi0, theta0 = read_native h sfx axes in
    let delta0 = U.ratio Nx.float64 u U.degree *. crval_lat in
    let* lonpole =
      pole h ("LONPOLE" ^ sfx)
        (strf "PV%d_3%s" axes.lng sfx)
        (default_lonpole ~delta0 ~theta0 ~phi0)
    in
    let* latpole =
      pole h ("LATPOLE" ^ sfx) (strf "PV%d_4%s" axes.lng sfx) default_latpole
    in
    let deg x = Quantity.v U.degree x in
    let swap =
      if axes.lng = 1 then Transform.id else Transform.axes [| 1; 0 |] ~origin:0
    in
    let projection =
      Transform.celestial axes.code frame ~pv:(Nx.zeros f64 [| 0 |])
        ~native:(deg (Nx.create f64 [| 2 |] [| phi0; theta0 |]))
        ~crval:(Quantity.v u (Nx.create f64 [| 2 |] [| crval_lng; crval_lat |]))
        ~lonpole:(deg (Nx.scalar f64 lonpole))
        ~latpole:(deg (Nx.scalar f64 latpole))
    in
    Ok
      Transform.(
        axes [| 1; 0 |] ~origin:1
        >> shift (bare (Nx.create f64 [| 2 |] [| crpix1; crpix2 |]))
        >> matrix_stages >> swap >> projection)

  (* Writing

     The writer reads the stage list back into keyword values. A card that
     reads back equal keeps its record, a card that differs is set in place,
     and a card the header lacks is added unless its value is the one FITS
     assumes for an absent keyword. *)

  type value = F of float | S of string
  type card = { key : string; value : value; optional : bool }

  let card ?(optional = false) key value = { key; value; optional }
  let default x d key = card ~optional:(Float.equal x d) key (F x)

  type celestial = {
    code : Transform.code;
    frame : frame;
    crval : (float, Nx.float64_elt) Nx.t Quantity.t;
    native : float array;  (** (φ₀, θ₀) in degrees. *)
    lonpole : float;  (** Degrees. *)
    latpole : float;  (** Degrees. *)
  }

  (* The stage list, read as plain data. *)
  type step =
    | Axes of int array * int
    | Shift of float array
    | Linear of float array * U.t
    | Scale of float array * U.t
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
    | Deproject c ->
        Celestial
          {
            code = c.code;
            frame = Frame c.frame;
            crval = c.crval;
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
          | Deproject c -> scalar_celestial c
          | Project c -> scalar_celestial c
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
      | Celestial c -> "celestial " ^ Transform.code_name c.code
    in
    Error (strf "Fits.Wcs.write: FITS cannot spell the %s stage there" name)

  let ends () =
    Error "Fits.Wcs.write: the transform ends before its celestial stage"

  let pair_of (Frame f) =
    let prefix lon = List.find (fun p -> p.lon = lon) pairs in
    match f with
    | Frame.Icrs | Frame.Fk5_j2000 -> prefix "RA"
    | Frame.Galactic -> prefix "GLON"
    | Frame.Ecliptic_j2000 -> prefix "ELON"
    | Frame.Supergalactic -> prefix "SLON"

  let has_equinox h sfx =
    present h ("EQUINOX" ^ sfx) || (sfx = "" && present h "EPOCH")

  let frame_cards h sfx (Frame f) =
    let radesys = radesys_key h sfx in
    let icrs = card ~optional:(not (has_equinox h sfx)) radesys (S "ICRS") in
    match f with
    | Frame.Icrs -> [ icrs ]
    | Frame.Fk5_j2000 ->
        [ card radesys (S "FK5"); card ("EQUINOX" ^ sfx) (F 2000.) ]
    | Frame.Ecliptic_j2000 ->
        [ icrs; card ~optional:true ("EQUINOX" ^ sfx) (F 2000.) ]
    | Frame.Galactic | Frame.Supergalactic -> []

  let index_pairs = [ (1, 1); (1, 2); (2, 1); (2, 2) ]

  (* A pole is written with the spelling the header uses: PVi_3 or PVi_4 when
     it has one, and then LONPOLE or LATPOLE only where it has that too. *)
  let pole_cards h key alt x d =
    if present h alt then
      [ card ~optional:true key (F x); card ~optional:true alt (F x) ]
    else [ default x d key ]

  (* [cards h sfx window steps] is the cards [steps] spell, or [Error] naming a
     stage FITS cannot spell there. *)
  let cards h sfx window steps =
    let* crpix, rest =
      match steps with
      | Axes ([| 1; 0 |], 1) :: Shift crpix :: rest -> Ok (crpix, rest)
      | Axes ([| 1; 0 |], 1) :: s :: _ | s :: _ -> unspellable s
      | [] -> ends ()
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
    let* c =
      match rest with
      | [ Celestial c ] -> Ok c
      | Celestial _ :: s :: _ | s :: _ -> unspellable s
      | [] -> ends ()
    in
    let u = match linear with `Pc (_, _, u) | `Cd (_, u) -> u in
    let* cunit =
      if U.convertible u U.radian then Unit.print u
      else
        Error
          (Format.asprintf
             "Fits.Wcs.write: the intermediate plane is in %a, not an angle"
             U.pp u)
    in
    let crpix =
      match window with
      | None -> crpix
      | Some w ->
          if Array.length w <> 2 then
            invalid_arg "Fits.Wcs.write: ~window needs a range for each axis";
          (* FITS axis [k + 1] is tensor axis [1 - k]. *)
          Array.mapi (fun k r -> r -. float_of_int (fst w.(1 - k))) crpix
    in
    let lng, lat = if lat_first then (2, 1) else (1, 2) in
    let pair = pair_of c.frame in
    let ctype prefix = S (ctype_text prefix c.code) in
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
            || List.exists (fun ij -> present h (at "CD" ij)) index_pairs
          in
          List.map
            (fun c -> { c with optional = c.optional && some_cd })
            (matrix "CD" cd (fun _ _ -> 0.))
    in
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
          default phi0 0. (strf "PV%d_1%s" lng sfx);
          default theta0 90. (strf "PV%d_2%s" lng sfx);
        ]
      @ pole_cards h ("LONPOLE" ^ sfx) (strf "PV%d_3%s" lng sfx) c.lonpole
          (default_lonpole ~delta0 ~theta0 ~phi0)
      @ pole_cards h ("LATPOLE" ^ sfx) (strf "PV%d_4%s" lng sfx) c.latpole
          default_latpole
      @ frame_cards h sfx c.frame)

  (* The keywords of an alternate's description the writer owns. *)
  let owned sfx =
    let per_axis p = List.map (fun i -> strf "%s%d%s" p i sfx) [ 1; 2 ] in
    let matrix p =
      List.map (fun (i, j) -> strf "%s%d_%d%s" p i j sfx) index_pairs
    in
    let pv i = List.init 100 (fun m -> strf "PV%d_%d%s" i m sfx) in
    let sip p =
      List.concat
        (List.init 10 (fun i ->
             List.init (10 - i) (fun j -> strf "%s_%d_%d" p i j)))
    in
    let primary =
      if sfx <> "" then []
      else
        [ "CROTA1"; "CROTA2"; "A_ORDER"; "B_ORDER"; "AP_ORDER"; "BP_ORDER" ]
        @ List.concat_map sip [ "A"; "B"; "AP"; "BP" ]
    in
    List.concat_map per_axis [ "CTYPE"; "CUNIT"; "CRPIX"; "CRVAL"; "CDELT" ]
    @ matrix "CD" @ matrix "PC" @ pv 1 @ pv 2
    @ [ "LONPOLE" ^ sfx; "LATPOLE" ^ sfx ]
    @ primary

  let reads_equal h { key; value; _ } =
    match value with
    | F x -> (
        match find_float h key with
        | Ok (Some y) -> Float.equal x y
        | _ -> false)
    | S s -> (
        match find_string h key with
        | Ok (Some y) -> String.equal s y
        | _ -> false)

  let set h { key; value; _ } =
    match value with
    | F x -> Header.set Value.float key x h
    | S s -> Header.set Value.string key s h

  let record_key r =
    if String.length r >= 10 && r.[8] = '=' && r.[9] = ' ' then
      String.trim (String.sub r 0 8)
    else ""

  let end_record = "END" ^ String.make 77 ' '

  let write ?alt ?window t h =
    let sfx = suffix "write" alt in
    if not (unbatched t) then
      Error "Fits.Wcs.write: a batch of transforms has no single header"
    else
      let* cards = cards h sfx window (steps t) in
      let h =
        List.fold_left
          (fun h c ->
            if present h c.key && not (reads_equal h c) then set h c else h)
          h cards
      in
      let wanted = List.map (fun c -> c.key) cards in
      let removed =
        List.filter
          (fun k -> present h k && not (List.mem k wanted))
          (owned sfx)
      in
      let first =
        List.find_index
          (fun r -> List.mem (record_key r) removed)
          (Header.records h)
      in
      let h = List.fold_left (fun h k -> Header.remove k h) h removed in
      match
        List.filter (fun c -> not (c.optional || present h c.key)) cards
      with
      | [] -> Ok h
      | fresh ->
          let added = Header.records (List.fold_left set Header.empty fresh) in
          let records = Header.records h in
          let at = Option.value first ~default:(List.length records) in
          let before = List.filteri (fun i _ -> i < at) records in
          let after = List.filteri (fun i _ -> i >= at) records in
          Header.of_string
            (String.concat "" (before @ added @ after @ [ end_record ]))
end

(* Observations *)

(* [per_cell u] is [u] with a file's pixel⁻¹ read as {!Grid.cell}⁻¹. *)
let per_cell u =
  List.fold_left
    (fun acc (t, num, den) ->
      match t with
      | U.Symbol { name = "pix"; scope = Some scope } when num = -1 && den = 1
        ->
          U.(acc * scoped ~scope "pix" / Grid.cell)
      | _ -> acc)
    u (U.terms u)

let data_unit hdu =
  let* u = unit hdu in
  match u with
  | Some u -> Ok (per_cell u)
  | None ->
      fail (header hdu)
        "no BUNIT gives the data's unit; build the observation with \
         Observation.v"

let observation (type e) ~(dtype : (float, e) Nx.dtype) ~frame ~data ?error ?ver
    ?window hdus =
  let* hdu = get ?ver data hdus in
  let h = header hdu in
  let* image = Image.of_hdu hdu in
  let shape = Image.shape image in
  let* () =
    if Array.length shape = 2 then Ok ()
    else
      fail h "a %d-axis image has no celestial grid; ymir reads two axes"
        (Array.length shape)
  in
  let* t = Wcs.read frame h in
  let* unit = data_unit hdu in
  let* values = Image.values ?window dtype hdu in
  let* variance, finite =
    match error with
    | None -> Ok (None, Nx.isfinite values)
    | Some name ->
        let* e = get ?ver name hdus in
        let* ei = Image.of_hdu e in
        let* () =
          if Image.shape ei = shape then Ok ()
          else fail (header e) "an error of another shape than %s's" data
        in
        let* eu = data_unit e in
        let* () =
          if U.convertible eu unit then Ok ()
          else
            fail (header e) "an error in %s does not convert to %s's %s"
              (U.to_string eu) data (U.to_string unit)
        in
        let* sigma = Image.values ?window dtype e in
        let sigma = Quantity.value unit (Quantity.v eu sigma) in
        Ok
          ( Some (Quantity.v U.(unit ** 2) (Nx.square sigma)),
            Nx.logical_and (Nx.isfinite values) (Nx.isfinite sigma) )
  in
  let* area = Header.find Value.float "PIXAR_SR" h in
  let area =
    Option.map (fun a -> Quantity.v U.steradian (Nx.scalar dtype a)) area
  in
  let grid = Grid.pixels ~shape dtype t in
  let grid =
    match window with
    | None -> grid
    | Some w ->
        let start =
          Nx.create Nx.int64 [| 2 |]
            (Array.map (fun (a, _) -> Int64.of_int a) w)
        in
        Grid.window ~start ~shape:(Array.map (fun (a, b) -> b - a) w) grid
  in
  Ok
    (Observation.v ?variance ~valid:(Nx.cast Nx.bit finite) ?area grid
       (Quantity.v unit values))
