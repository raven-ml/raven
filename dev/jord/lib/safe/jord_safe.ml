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

(* Level-2A *)

(* A band by the name ESA's spectral list gives it, which writes B1 for B01.
   B10 has no Level-2A image. *)
let band = function
  | "B1" -> Ok (Some Sentinel2.B01)
  | "B2" -> Ok (Some Sentinel2.B02)
  | "B3" -> Ok (Some Sentinel2.B03)
  | "B4" -> Ok (Some Sentinel2.B04)
  | "B5" -> Ok (Some Sentinel2.B05)
  | "B6" -> Ok (Some Sentinel2.B06)
  | "B7" -> Ok (Some Sentinel2.B07)
  | "B8" -> Ok (Some Sentinel2.B08)
  | "B8A" -> Ok (Some Sentinel2.B8A)
  | "B9" -> Ok (Some Sentinel2.B09)
  | "B10" -> Ok None
  | "B11" -> Ok (Some Sentinel2.B11)
  | "B12" -> Ok (Some Sentinel2.B12)
  | b -> Error (strf "physicalBand %S is not a Sentinel-2 band" b)

(* The band of each band index, by the spectral list. *)
let spectral chars =
  let* list = child "Spectral_Information_List" chars in
  all
    (fun s ->
      let* id = attr "bandId" s in
      let* b = Result.bind (attr "physicalBand" s) band in
      Ok (id, b))
    (children "Spectral_Information" list)

let offset bands o =
  let* id = attr "band_id" o in
  match List.assoc_opt id bands with
  | None -> Error (strf "BOA_ADD_OFFSET's band_id %S is not a band's" id)
  | Some None -> Ok None
  | Some (Some b) -> (
      match text o with
      | Ok "Not applicable" -> Ok None
      | _ ->
          let* v = int o in
          Ok (Some (b, v)))

let rec unique = function
  | [] -> Ok ()
  | (b, _) :: rest when List.mem_assoc b rest ->
      Error (strf "BOA_ADD_OFFSET is stated twice for %s" (Sentinel2.name b))
  | _ :: rest -> unique rest

(* BOA_ADD_OFFSET by band, in the file's order: none where the product
   states no list, as before baseline 04.00, and none for a band whose
   value is "Not applicable", which RADIO_OFFSET_TYPE allows. *)
let offsets chars =
  match children "BOA_ADD_OFFSET_VALUES_LIST" chars with
  | [] -> Ok []
  | _ :: _ :: _ -> Error "BOA_ADD_OFFSET_VALUES_LIST is stated twice"
  | [ list ] ->
      let* bands = spectral chars in
      let* offsets = all (offset bands) (children "BOA_ADD_OFFSET" list) in
      let offsets = List.filter_map Fun.id offsets in
      let* () = unique offsets in
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

(* BOA_QUANTIFICATION_VALUE, an xs:double: its inverse is the band's
   scale, so both must be positive and finite. *)
let quantification chars =
  let q = [ "QUANTIFICATION_VALUES_LIST"; "BOA_QUANTIFICATION_VALUE" ] in
  let* s = Result.bind (path q chars) text in
  let q = if decimal s then float_of_string s else Float.nan in
  if Float.is_finite q && q > 0. && Float.is_finite (1. /. q) then Ok q
  else Error (strf "BOA_QUANTIFICATION_VALUE %S is not a positive decimal" s)

let l2a s =
  let* root = parse s in
  let* () =
    if name root = "Level-2A_User_Product" then Ok ()
    else Error (strf "%s is not a Level-2A product's metadata" (name root))
  in
  let* info = child "General_Info" root in
  let pb = [ "Product_Info"; "PROCESSING_BASELINE" ] in
  let* baseline = Result.bind (path pb info) text in
  let* chars = child "Product_Image_Characteristics" info in
  let* quantification = quantification chars in
  let* nodata, saturated = specials chars in
  let* offsets = offsets chars in
  Ok { Sentinel2.baseline; quantification; offsets; nodata; saturated }
