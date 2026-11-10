(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

exception Refused of string

let refuse fmt = Printf.ksprintf (fun s -> raise (Refused s)) fmt

(* The description's numbers: a main file's code and version, its header's
   length, and its shape types. *)
let file_code = 9994
let version = 1000
let header = 100
let null = 0
let polygon_types = [ 5; 15; 25 ]

(* Main file *)

(* Reads [n] bytes at [off] of [s] by [get], or refuses naming [what]. *)
let bytes s ~off ~n what get =
  if off < 0 || off + n > String.length s then
    refuse "%s: the file ends early" what;
  get s off

let i32_be s off what =
  Int32.to_int (bytes s ~off ~n:4 what String.get_int32_be)

let i32_le s off what =
  Int32.to_int (bytes s ~off ~n:4 what String.get_int32_le)

let f64_le s off what =
  Int64.float_of_bits (bytes s ~off ~n:8 what String.get_int64_le)

type shape =
  | Null
  | Rings of int array * float array (* Part starts, points x, y. *)

let shapes shp =
  let what = "the main file" in
  if i32_be shp 0 what <> file_code then
    refuse "%s: its file code is not 9994" what;
  if i32_le shp 28 what <> version then
    refuse "%s: its version is not 1000" what;
  let kind = i32_le shp 32 what in
  if not (List.mem kind polygon_types) then
    refuse "%s: shape type %d is not Polygon, PolygonZ or PolygonM" what kind;
  let len = 2 * i32_be shp 24 what in
  if len < header || len > String.length shp then
    refuse "%s: its length (%d bytes) is not the file's" what len;
  let rec go off k acc =
    if off >= len then List.rev acc
    else
      let what = Printf.sprintf "shape %d" k in
      let size = 2 * i32_be shp (off + 4) what in
      let c = off + 8 in
      if size < 4 || c + size > len then
        refuse "%s: its length leaves the file" what;
      let t = i32_le shp c what in
      let shape =
        if t = null then Null
        else if t <> kind then
          refuse "%s: shape type %d is not the file's %d" what t kind
        else
          let parts = i32_le shp (c + 36) what
          and n = i32_le shp (c + 40) what in
          if parts < 0 || n < 0 || 44 + (4 * parts) + (16 * n) > size then
            refuse "%s: its %d parts and %d points leave the record" what parts
              n;
          let starts =
            Array.init parts (fun p -> i32_le shp (c + 44 + (4 * p)) what)
          in
          Array.iteri
            (fun p s ->
              let prev = if p = 0 then 0 else starts.(p - 1) in
              if (p = 0 && s <> 0) || s < prev || s > n then
                refuse "%s: part %d starts at point %d" what p s)
            starts;
          let base = c + 44 + (4 * parts) in
          Rings
            ( starts,
              Array.init (2 * n) (fun m -> f64_le shp (base + (8 * m)) what) )
      in
      go (c + size) (k + 1) (shape :: acc)
  in
  go header 0 []

(* dBASE table *)

let deleted = '*'

let table dbf =
  let what = "the table" in
  let u8 off = Char.code (bytes dbf ~off ~n:1 what String.get) in
  let u16 off = bytes dbf ~off ~n:2 what String.get_uint16_le in
  let records = i32_le dbf 4 what land 0xFFFF_FFFF in
  let hlen = u16 8 and rlen = u16 10 in
  (* Field descriptors of 32 bytes from byte 32, ended by 0x0D. *)
  let rec fields off acc =
    if off >= hlen || u8 off = 0x0D then List.rev acc
    else
      let raw = bytes dbf ~off ~n:11 what (fun s o -> String.sub s o 11) in
      let name = List.hd (String.split_on_char '\000' raw) in
      fields (off + 32) ((name, u8 (off + 16)) :: acc)
  in
  let fields = fields 32 [] in
  if 1 + List.fold_left (fun n (_, w) -> n + w) 0 fields <> rlen then
    refuse "%s: its fields' widths do not sum to its record length %d" what rlen;
  if hlen + (records * rlen) > String.length dbf then
    refuse "%s: its %d records leave the file" what records;
  let keep = Array.init records (fun k -> dbf.[hlen + (k * rlen)] <> deleted) in
  let trim s =
    let n = ref (String.length s) in
    while !n > 0 && (s.[!n - 1] = ' ' || s.[!n - 1] = '\000') do
      decr n
    done;
    String.sub s 0 !n
  in
  let _, columns =
    List.fold_left
      (fun (start, acc) (name, w) ->
        let text k = trim (String.sub dbf (hlen + (k * rlen) + start) w) in
        let kept = List.filter (fun k -> keep.(k)) (List.init records Fun.id) in
        (start + w, (name, Array.of_list (List.map text kept)) :: acc))
      (1, []) fields
  in
  (keep, List.rev columns)

let read crs ~shp ~dbf =
  match
    let shapes = shapes shp in
    let keep, columns = table dbf in
    if Array.length keep <> List.length shapes then
      refuse "the main file has %d shapes and the table %d records"
        (List.length shapes) (Array.length keep);
    let kept = List.filteri (fun k _ -> keep.(k)) shapes in
    let rings = ref [ 0 ] and features = ref [ 0 ] and nrings = ref 0 in
    let count = ref 0 and coords = ref [] in
    List.iter
      (fun shape ->
        (match shape with
        | Null -> ()
        | Rings (starts, xy) ->
            let n = Array.length xy / 2 in
            Array.iteri
              (fun p _ ->
                let stop =
                  if p + 1 < Array.length starts then starts.(p + 1) else n
                in
                rings := (!count + stop) :: !rings;
                incr nrings)
              starts;
            coords := xy :: !coords;
            count := !count + n);
        features := !nrings :: !features)
      kept;
    let points = Array.concat (List.rev !coords) in
    if not (Array.for_all Float.is_finite points) then
      refuse "the main file: a point is not finite";
    let points = Nx.create Nx.float64 [| !count; 2 |] points in
    let rings = Array.of_list (List.rev !rings) in
    let features = Array.of_list (List.rev !features) in
    (Polygons.v crs points ~rings ~features, columns)
  with
  | r -> Ok r
  | exception Refused e -> Error e
