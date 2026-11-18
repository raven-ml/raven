(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Jord

let strf = Printf.sprintf
let ( let* ) = Result.bind

(* XML *)

(* An element by its local name, with its attributes by theirs: ESA's files
   bind each prefix to one schema, so local names are unambiguous. *)
type tree = El of string * (string * string) list * tree list | Data of string

let parse s =
  let el ((_, name), atts) kids =
    El (name, List.map (fun ((_, a), v) -> (a, v)) atts, kids)
  in
  let data d = Data d in
  let i = Xmlm.make_input ~strip:true (`String (0, s)) in
  match Xmlm.input_doc_tree ~el ~data i with
  | _, root when Xmlm.eoi i -> Ok root
  | _ -> Error "content follows the root element"
  | exception Xmlm.Error ((line, col), e) ->
      Error (strf "line %d, column %d: %s" line col (Xmlm.error_message e))

let name = function El (n, _, _) -> n | Data _ -> "text"

let children n = function
  | El (_, _, kids) ->
      List.filter (function El (m, _, _) -> m = n | Data _ -> false) kids
  | Data _ -> []

let child n t =
  match children n t with
  | [ c ] -> Ok c
  | [] -> Error (strf "%s has no %s" (name t) n)
  | _ -> Error (strf "%s has more than one %s" (name t) n)

let path ns t = List.fold_left (fun t n -> Result.bind t (child n)) (Ok t) ns

let text = function
  | El (_, _, [ Data d ]) -> Ok d
  | t -> Error (strf "%s holds no text" (name t))

let attr a = function
  | El (n, atts, _) ->
      Option.to_result (List.assoc_opt a atts)
        ~none:(strf "%s has no attribute %s" n a)
  | Data _ -> Error (strf "text has no attribute %s" a)

(* An xs:int (A_SPECIAL_VALUES, RADIO_OFFSET_TYPE): an optional minus sign
   and decimal digits. *)
let int t =
  let* s = text t in
  let n = String.length s in
  let digits = if n > 1 && s.[0] = '-' then String.sub s 1 (n - 1) else s in
  match int_of_string_opt s with
  | Some i when String.for_all (fun c -> c >= '0' && c <= '9') digits -> Ok i
  | _ -> Error (strf "%s: %S is not an integer" (name t) s)

let rec all f = function
  | [] -> Ok []
  | x :: xs ->
      let* y = f x in
      let* ys = all f xs in
      Ok (y :: ys)

(* A number in plain decimal form, as 10000 or 1.0E4: digits, an optional
   fraction and an optional exponent. OCaml's float_of_string also reads
   underscores and hexadecimal, which no xs:double holds. *)
let decimal s =
  let n = String.length s in
  let digit i = i < n && s.[i] >= '0' && s.[i] <= '9' in
  let rec digits i = if digit i then digits (i + 1) else i in
  let i = digits 0 in
  let i = if i > 0 && i < n && s.[i] = '.' then digits (i + 1) else i in
  if i = 0 then false
  else if i < n && (s.[i] = 'e' || s.[i] = 'E') then
    let e = i + 1 in
    let e = if e < n && (s.[e] = '+' || s.[e] = '-') then e + 1 else e in
    digit e && digits e = n
  else i = n

(* A signed decimal, as ESA writes wavelengths, a tile's corner and its
   angles. *)
let signed s =
  let n = String.length s in
  let d = if n > 1 && s.[0] = '-' then String.sub s 1 (n - 1) else s in
  if decimal d then Some (float_of_string s) else None

let number t =
  let* s = text t in
  match signed s with
  | Some v when Float.is_finite v -> Ok v
  | _ -> Error (strf "%s: %S is not a number" (name t) s)

(* Products *)

(* The names a level gives the values that decode its DNs: its root element,
   its quantification value's path in Product_Image_Characteristics, and its
   list of offsets and their elements. The rules are the same at both levels. *)
type level = {
  root : string;
  quantification : string list;
  offsets : string;
  offset : string;
}

(* A band by the name ESA's spectral list gives it, which writes B1 for B01. *)
let band = function
  | "B1" -> Ok Sentinel2.B01
  | "B2" -> Ok Sentinel2.B02
  | "B3" -> Ok Sentinel2.B03
  | "B4" -> Ok Sentinel2.B04
  | "B5" -> Ok Sentinel2.B05
  | "B6" -> Ok Sentinel2.B06
  | "B7" -> Ok Sentinel2.B07
  | "B8" -> Ok Sentinel2.B08
  | "B8A" -> Ok Sentinel2.B8A
  | "B9" -> Ok Sentinel2.B09
  | "B10" -> Ok Sentinel2.B10
  | "B11" -> Ok Sentinel2.B11
  | "B12" -> Ok Sentinel2.B12
  | b -> Error (strf "physicalBand %S is not a Sentinel-2 band" b)

(* A band's Wavelength, MIN, MAX and CENTRAL, each positive nanometres. *)
let wavelength w =
  let nm n =
    let* e = child n w in
    let* v = number e in
    match attr "unit" e with
    | Ok "nm" when v > 0. -> Ok v
    | _ -> Error (strf "%s is not a positive number of nanometres" n)
  in
  let* min = nm "MIN" in
  let* max = nm "MAX" in
  let* central = nm "CENTRAL" in
  if min <= central && central <= max then Ok { Sentinel2.central; min; max }
  else Error (strf "CENTRAL %g is not from MIN %g to MAX %g" central min max)

(* Each band index's band and wavelengths, by the spectral list, which must
   number each band as ESA's specification does. *)
let spectral chars =
  let info s =
    let* id = attr "bandId" s in
    let* b = Result.bind (attr "physicalBand" s) band in
    let* () =
      if id = string_of_int (Sentinel2.index b) then Ok ()
      else Error (strf "bandId %S is not %s's" id (Sentinel2.name b))
    in
    match children "Wavelength" s with
    | [] -> Ok (id, (b, None))
    | [ w ] ->
        let* w = wavelength w in
        Ok (id, (b, Some w))
    | _ -> Error (strf "%s's Wavelength is stated twice" (Sentinel2.name b))
  in
  match children "Spectral_Information_List" chars with
  | [] -> Ok []
  | [ list ] -> all info (children "Spectral_Information" list)
  | _ -> Error "Spectral_Information_List is stated twice"

let offset level spectral o =
  let* id = attr "band_id" o in
  match List.assoc_opt id spectral with
  | None -> Error (strf "%s's band_id %S is not a band's" level.offset id)
  | Some (b, _) -> (
      match text o with
      | Ok "Not applicable" -> Ok None
      | _ ->
          let* v = int o in
          Ok (Some (b, v)))

let rec unique what = function
  | [] -> Ok ()
  | (b, _) :: rest when List.mem_assoc b rest ->
      Error (strf "%s is stated twice for %s" what (Sentinel2.name b))
  | _ :: rest -> unique what rest

(* The offsets by band, in the file's order: none where the product states
   no list, as before baseline 04.00, and none for a band whose value is "Not
   applicable", which RADIO_OFFSET_TYPE allows. *)
let offsets level spectral chars =
  match children level.offsets chars with
  | [] -> Ok []
  | _ :: _ :: _ -> Error (strf "%s is stated twice" level.offsets)
  | [ list ] ->
      let* offsets = all (offset level spectral) (children level.offset list) in
      let offsets = List.filter_map Fun.id offsets in
      let* () = unique level.offset offsets in
      Ok offsets

(* NODATA and SATURATED, the two special values of A_SPECIAL_VALUES. *)
let specials chars =
  let value v =
    let* t = Result.bind (child "SPECIAL_VALUE_TEXT" v) text in
    let* i = Result.bind (child "SPECIAL_VALUE_INDEX" v) int in
    Ok (t, i)
  in
  let* values = all value (children "Special_Values" chars) in
  let one t =
    match List.filter (fun (s, _) -> s = t) values with
    | [ (_, i) ] -> Ok i
    | [] -> Error (strf "no special value %s" t)
    | _ -> Error (strf "special value %s is stated twice" t)
  in
  let* nodata = one "NODATA" in
  let* saturated = one "SATURATED" in
  Ok (nodata, saturated)

(* The quantification value, an xs:double: its inverse is the band's
   scale, so both must be positive and finite. *)
let quantification level chars =
  let* s = Result.bind (path level.quantification chars) text in
  let q = if decimal s then float_of_string s else Float.nan in
  if Float.is_finite q && q > 0. && Float.is_finite (1. /. q) then Ok q
  else
    let name = List.hd (List.rev level.quantification) in
    Error (strf "%s %S is not a positive decimal" name s)

let product level s =
  let* root = parse s in
  let* () =
    if name root = level.root then Ok ()
    else Error (strf "the root is %s, not %s" (name root) level.root)
  in
  let* info = child "General_Info" root in
  let pb = [ "Product_Info"; "PROCESSING_BASELINE" ] in
  let* baseline = Result.bind (path pb info) text in
  let* chars = child "Product_Image_Characteristics" info in
  let* quantification = quantification level chars in
  let* nodata, saturated = specials chars in
  let* spectral = spectral chars in
  let* offsets = offsets level spectral chars in
  let stated (_, (b, w)) = Option.map (fun w -> (b, w)) w in
  let wavelengths = List.filter_map stated spectral in
  let* () = unique "Wavelength" wavelengths in
  Ok
    {
      Sentinel2.baseline;
      quantification;
      offsets;
      nodata;
      saturated;
      wavelengths;
    }

let l1c =
  product
    {
      root = "Level-1C_User_Product";
      quantification = [ "QUANTIFICATION_VALUE" ];
      offsets = "Radiometric_Offset_List";
      offset = "RADIO_ADD_OFFSET";
    }

let l2a =
  product
    {
      root = "Level-2A_User_Product";
      quantification =
        [ "QUANTIFICATION_VALUES_LIST"; "BOA_QUANTIFICATION_VALUE" ];
      offsets = "BOA_ADD_OFFSET_VALUES_LIST";
      offset = "BOA_ADD_OFFSET";
    }

(* Tiles *)

(* An index of the PSD's numbering, as "3", from [lo] to [hi]. *)
let index a lo hi t =
  let* s = attr a t in
  let digits = String.for_all (fun c -> c >= '0' && c <= '9') s in
  match int_of_string_opt s with
  | Some i when digits && lo <= i && i <= hi -> Ok i
  | _ -> Error (strf "%s %S is not from %d to %d" a s lo hi)

(* One angle of a grid (AN_ANGLE_GRID): its steps, positive metres, and its
   values, a row per VALUES, each a decimal or NaN, ESA's "not computed". *)
let angle t =
  let step n =
    let* e = child n t in
    let* v = number e in
    match attr "unit" e with
    | Ok "m" when v > 0. -> Ok v
    | _ -> Error (strf "%s is not a positive number of metres" n)
  in
  let* row = step "ROW_STEP" in
  let* col = step "COL_STEP" in
  let value s =
    if s = "NaN" then Ok Float.nan
    else Option.to_result (signed s) ~none:(strf "VALUES holds %S" s)
  in
  let values v =
    let* s = text v in
    let words = List.filter (( <> ) "") (String.split_on_char ' ' s) in
    let* row = all value words in
    Ok (Array.of_list row)
  in
  let* list = child "Values_List" t in
  let* rows = all values (children "VALUES" list) in
  Ok ((row, col), Array.of_list rows)

(* A grid's zenith and azimuth, of the steps and size of [like], the Sun's
   zenith, which a grid without [like] is. *)
let grid ?like t =
  let* z = Result.bind (child "Zenith" t) angle in
  let* a = Result.bind (child "Azimuth" t) angle in
  let steps, rows = Option.value like ~default:z in
  let cols = if rows = [||] then 0 else Array.length rows.(0) in
  let fits (s, r) =
    s = steps
    && Array.length r = Array.length rows
    && Array.for_all (fun r -> Array.length r = cols) r
  in
  if cols = 0 then Error (strf "%s states no values" (name t))
  else if fits z && fits a then Ok (z, snd a)
  else Error (strf "%s's steps or size differ from the Sun's" (name t))

type tile = {
  sensing_time : float;
  nodes : Grid.t;
  sun : Nx.host Nx.float64_t * Nx.host Nx.float64_t;
  view : Nx.host Nx.float64_t * Nx.host Nx.float64_t;
}

(* The tile's system, HORIZONTAL_CS_CODE, as "EPSG:32633". *)
let system geocoding =
  let* s = Result.bind (child "HORIZONTAL_CS_CODE" geocoding) text in
  let digits c = c <> "" && String.for_all (fun c -> c >= '0' && c <= '9') c in
  let code =
    match String.split_on_char ':' s with
    | [ "EPSG"; c ] when digits c -> int_of_string_opt c
    | _ -> None
  in
  match code with
  | Some c -> Crs.of_epsg c
  | None -> Error (strf "HORIZONTAL_CS_CODE %S is not an EPSG code" s)

(* The tile's top left corner, which each of its Geopositions states. *)
let corner geocoding =
  let ul g =
    let* x = Result.bind (child "ULX" g) number in
    let* y = Result.bind (child "ULY" g) number in
    Ok (x, y)
  in
  let* corners = all ul (children "Geoposition" geocoding) in
  match corners with
  | [] -> Error "Tile_Geocoding has no Geoposition"
  | c :: cs when List.for_all (( = ) c) cs -> Ok c
  | _ -> Error "the Geopositions state different corners"

let bands = List.length Sentinel2.bands
let detectors = 12

let tile s =
  let* root = parse s in
  let* () =
    match name root with
    | "Level-1C_Tile_ID" | "Level-2A_Tile_ID" -> Ok ()
    | n -> Error (strf "the root is %s, not a tile's" n)
  in
  let* time = Result.bind (path [ "General_Info"; "SENSING_TIME" ] root) text in
  let* sensing_time =
    Result.map_error (strf "SENSING_TIME: %s") (Time.of_rfc3339 time)
  in
  let* info = child "Geometric_Info" root in
  let* geocoding = child "Tile_Geocoding" info in
  let* crs = system geocoding in
  let* ulx, uly = corner geocoding in
  let* angles = child "Tile_Angles" info in
  let* ((((row, col), z) as like), a) =
    Result.bind (child "Sun_Angles_Grid" angles) grid
  in
  let rows = Array.length z and cols = Array.length z.(0) in
  let n = rows * cols * bands * detectors in
  let vz = Array.make n Float.nan and va = Array.make n Float.nan in
  let seen = Hashtbl.create 91 in
  let view g =
    let* b = index "bandId" 0 (bands - 1) g in
    let* d = index "detectorId" 1 detectors g in
    let* (_, gz), ga = grid ~like g in
    if Hashtbl.mem seen (b, d) then
      Error (strf "bandId %d, detectorId %d is stated twice" b d)
    else (
      Hashtbl.add seen (b, d) ();
      for i = 0 to rows - 1 do
        for j = 0 to cols - 1 do
          let k = (((((i * cols) + j) * bands) + b) * detectors) + d - 1 in
          vz.(k) <- gz.(i).(j);
          va.(k) <- ga.(i).(j)
        done
      done;
      Ok ())
  in
  let* _ = all view (children "Viewing_Incidence_Angles_Grids" angles) in
  let flat x =
    Nx.create Nx.float64 [| rows; cols |] (Array.concat (Array.to_list x))
  in
  let dense x = Nx.create Nx.float64 [| rows; cols; bands; detectors |] x in
  let nodes =
    Grid.v crs ~rows ~cols
      ~origin:(ulx -. (col /. 2.), uly +. (row /. 2.))
      ~row_step:(0., -.row) ~col_step:(col, 0.)
  in
  let sun = (flat z, flat a) and view = (dense vz, dense va) in
  Ok { sensing_time; nodes; sun; view }
