(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Font files for the tests: the bundled faces altered, and fonts built table by
   table. *)

(* Surgery on a font's bytes, to make the fonts the bundled faces are not. *)

let get_u16 s i = String.get_uint16_be s i
let get_u32 s i = Int32.to_int (String.get_int32_be s i) land 0xFFFF_FFFF

(* The position of [tag]'s table directory record and of its table. *)
let table data tag =
  let rec find i =
    if i = get_u16 data 4 then invalid_arg (Printf.sprintf "no %S table" tag)
    else
      let r = 12 + (16 * i) in
      if String.sub data r 4 = tag then (r, get_u32 data (r + 8))
      else find (i + 1)
  in
  find 0

let patch data pos f =
  let b = Bytes.of_string data in
  f b pos;
  Bytes.to_string b

let set_u16 data pos v =
  patch data pos (fun b pos -> Bytes.set_uint16_be b pos (v land 0xFFFF))

let set_u32 data pos v =
  patch data pos (fun b pos -> Bytes.set_int32_be b pos (Int32.of_int v))

let set_bytes data pos s =
  patch data pos (fun b pos -> Bytes.blit_string s 0 b pos (String.length s))

let rename data tag tag' = set_bytes data (fst (table data tag)) tag'

(* [data] with the table of [tag] replaced by [bytes], appended to the file
   under the tag [tag']. *)
let replace data tag tag' bytes =
  let r, _ = table data tag in
  let padded =
    data ^ String.make ((4 - (String.length data mod 4)) mod 4) '\000'
  in
  let data = set_bytes (padded ^ bytes) r tag' in
  set_u32
    (set_u32 data (r + 8) (String.length padded))
    (r + 12) (String.length bytes)

let head_pos data = snd (table data "head")
let os2_pos data = snd (table data "OS/2")

(* A version 0 [kern] table of [(coverage, pairs)] subtables of format 0. *)
let kern_table subtables =
  let b = Buffer.create 64 in
  Buffer.add_uint16_be b 0;
  Buffer.add_uint16_be b (List.length subtables);
  List.iter
    (fun (coverage, pairs) ->
      let n = List.length pairs in
      List.iter (Buffer.add_uint16_be b)
        [ 0; 14 + (6 * n); coverage; n; 0; 0; 0 ];
      List.iter
        (fun (l, r, v) ->
          Buffer.add_uint16_be b l;
          Buffer.add_uint16_be b r;
          Buffer.add_uint16_be b (v land 0xFFFF))
        pairs)
    subtables;
  Buffer.contents b

(* Fonts built table by table, for what the bundled faces do not exercise: 1000
   units per em, long glyph offsets, a format 12 character map. *)

let be16 v =
  let b = Bytes.create 2 in
  Bytes.set_uint16_be b 0 (v land 0xFFFF);
  Bytes.to_string b

let be32 v =
  let b = Bytes.create 4 in
  Bytes.set_int32_be b 0 (Int32.of_int v);
  Bytes.to_string b

let cat = String.concat ""

(* [block size parts] is the offset of each of [parts] laid out after a header
   of [size] bytes, and the parts concatenated. *)
let block size parts =
  let _, offsets =
    List.fold_left
      (fun (off, acc) p -> (off + String.length p, off :: acc))
      (size, []) parts
  in
  (List.rev offsets, cat parts)

let file tables =
  let tables = List.sort compare tables in
  let n = List.length tables in
  let pad s = s ^ String.make ((4 - (String.length s mod 4)) mod 4) '\000' in
  let _, records, data =
    List.fold_left
      (fun (off, records, data) (tag, t) ->
        ( off + String.length (pad t),
          records ^ tag ^ be32 0 ^ be32 off ^ be32 (String.length t),
          data ^ pad t ))
      (12 + (16 * n), "", "")
      tables
  in
  cat [ be32 0x00010000; be16 n; be16 0; be16 0; be16 0; records; data ]

type point = { x : int; y : int; on : bool }

let on x y = { x; y; on = true }
let off x y = { x; y; on = false }

(* A simple glyph, coordinates as words. *)
let simple ?(instructions = "") contours =
  let b = Buffer.create 64 in
  let add16 v = Buffer.add_uint16_be b (v land 0xFFFF) in
  let points = List.concat contours in
  add16 (List.length contours);
  List.iter add16 [ 0; 0; 0; 0 ];
  ignore
    (List.fold_left
       (fun n c ->
         add16 (n + List.length c - 1);
         n + List.length c)
       0 contours);
  add16 (String.length instructions);
  Buffer.add_string b instructions;
  List.iter (fun p -> Buffer.add_uint8 b (Bool.to_int p.on)) points;
  let deltas f =
    ignore
      (List.fold_left
         (fun prev p ->
           add16 (f p - prev);
           f p)
         0 points)
  in
  deltas (fun p -> p.x);
  deltas (fun p -> p.y);
  Buffer.contents b

(* A composite glyph: (flags, glyph, arguments, transform) of each component,
   [MORE_COMPONENTS] added. *)
let composite components =
  let last = List.length components - 1 in
  cat
    (be16 0xFFFF :: be16 0 :: be16 0 :: be16 0 :: be16 0
    :: List.mapi
         (fun i (flags, g, args, transform) ->
           cat
             [
               be16 (if i < last then flags lor 0x20 else flags);
               be16 g;
               args;
               cat (List.map be16 transform);
             ])
         components)

(* Character map subtables: format 12 groups (first, last, glyph), and format 4
   segments (first, last, delta, glyph ids), the final U+FFFF segment added. *)
let format_12 groups =
  cat
    [
      be16 12;
      be16 0;
      be32 (16 + (12 * List.length groups));
      be32 0;
      be32 (List.length groups);
      cat (List.map (fun (a, b, g) -> be32 a ^ be32 b ^ be32 g) groups);
    ]

let format_4 segments =
  let segments = segments @ [ (0xFFFF, 0xFFFF, 1, None) ] in
  let n = List.length segments in
  let ids = Buffer.create 16 in
  let field f = cat (List.map (fun s -> be16 (f s)) segments) in
  let range_offsets =
    List.mapi
      (fun i (_, _, _, glyphs) ->
        match glyphs with
        | None -> be16 0
        | Some gs ->
            let at = 16 + (8 * n) + Buffer.length ids in
            List.iter (fun g -> Buffer.add_uint16_be ids g) gs;
            be16 (at - (16 + (6 * n) + (2 * i))))
      segments
  in
  let body =
    cat
      [
        field (fun (_, b, _, _) -> b);
        be16 0;
        field (fun (a, _, _, _) -> a);
        field (fun (_, _, d, _) -> d);
        cat range_offsets;
        Buffer.contents ids;
      ]
  in
  cat
    [
      be16 4;
      be16 (14 + String.length body);
      be16 0;
      be16 (2 * n);
      be16 0;
      be16 0;
      be16 0;
      body;
    ]

let cmap_table subtables =
  let offsets, body =
    block
      (4 + (8 * List.length subtables))
      (List.map (fun (_, _, t) -> t) subtables)
  in
  cat
    ([ be16 0; be16 (List.length subtables) ]
    @ List.map2
        (fun (p, e, _) off -> be16 p ^ be16 e ^ be32 off)
        subtables offsets
    @ [ body ])

let cmap_12 groups = cmap_table [ (3, 10, format_12 groups) ]

let tiny_file ?(tables = []) ?(cmap = cmap_12 [ (0x41, 0x41, 1) ])
    ?(upem = 1000) glyphs =
  let n = List.length glyphs in
  let offsets =
    List.fold_left
      (fun acc g -> (List.hd acc + String.length g) :: acc)
      [ 0 ] glyphs
    |> List.rev
  in
  let base =
    [
      ( "head",
        cat
          [
            be32 0x00010000;
            be32 0;
            be32 0;
            be32 0x5F0F3CF5;
            be16 0;
            be16 upem;
            String.make 16 '\000';
            be16 0;
            be16 0;
            be16 1000;
            be16 1000;
            be16 0;
            be16 0;
            be16 0;
            be16 1;
            be16 0;
          ] );
      ( "hhea",
        cat
          [
            be32 0x00010000;
            be16 800;
            be16 (-200);
            be16 100;
            String.make 24 '\000';
            be16 1;
          ] );
      ("maxp", be32 0x00010000 ^ be16 n);
      ("hmtx", be16 500 ^ be16 0);
      ("loca", cat (List.map be32 offsets));
      ("glyf", cat glyphs);
      ("cmap", cmap);
    ]
  in
  file (tables @ List.filter (fun (t, _) -> not (List.mem_assoc t tables)) base)

let square = [ on 0 0; on 100 0; on 100 100; on 0 100 ]

(* Component flags *)
let words = 0x1
let xy = 0x2
let one_scale = 0x8
let xy_scales = 0x40
let two_by_two = 0x80
let scaled_offset = 0x800
let half = 8192
let one = 16384

(* A [name] table of [(platform, encoding, language, name id, bytes)]
   records. *)
let name_table records =
  let strings = cat (List.map (fun (_, _, _, _, s) -> s) records) in
  let _, entries =
    List.fold_left
      (fun (off, acc) (platform, encoding, language, id, s) ->
        ( off + String.length s,
          acc
          ^ cat
              [
                be16 platform;
                be16 encoding;
                be16 language;
                be16 id;
                be16 (String.length s);
                be16 off;
              ] ))
      (0, "") records
  in
  cat
    [
      be16 0;
      be16 (List.length records);
      be16 (6 + (12 * List.length records));
      entries;
      strings;
    ]

(* [utf16 s] is the UTF-16BE of the ASCII string [s]. *)
let utf16 s =
  cat (List.init (String.length s) (fun i -> be16 (Char.code s.[i])))

(* GPOS tables built from lookups, for the kerning paths the bundled faces do
   not take. *)

let coverage_1 glyphs =
  cat (be16 1 :: be16 (List.length glyphs) :: List.map be16 glyphs)

let coverage_2 ranges =
  cat
    (be16 2
    :: be16 (List.length ranges)
    :: List.map (fun (a, b, i) -> be16 a ^ be16 b ^ be16 i) ranges)

let classes_1 start classes =
  cat
    (be16 1 :: be16 start :: be16 (List.length classes) :: List.map be16 classes)

let classes_2 ranges =
  cat
    (be16 2
    :: be16 (List.length ranges)
    :: List.map (fun (a, b, c) -> be16 a ^ be16 b ^ be16 c) ranges)

(* Pair adjustment format 1: [sets] gives each covered glyph's (second glyph,
   value records) list. *)
let pairs_1 ?coverage ?(format2 = 0) ~format sets =
  let coverage =
    Option.value coverage ~default:(coverage_1 (List.map fst sets))
  in
  let set (_, pairs) =
    cat
      (be16 (List.length pairs)
      :: List.map (fun (g, v) -> be16 g ^ cat (List.map be16 v)) pairs)
  in
  let header = 10 + (2 * List.length sets) in
  let offsets, body = block header (coverage :: List.map set sets) in
  cat
    ([
       be16 1;
       be16 (List.hd offsets);
       be16 format;
       be16 format2;
       be16 (List.length sets);
     ]
    @ List.map be16 (List.tl offsets)
    @ [ body ])

(* Pair adjustment format 2: [records.(c1).(c2)] are the value records. *)
let pairs_2 ?(format2 = 0) ~coverage ~classes1 ~classes2 ~format records =
  let count1 = List.length records in
  let count2 = match records with [] -> 0 | r :: _ -> List.length r in
  let matrix =
    cat (List.concat_map (List.map (fun v -> cat (List.map be16 v))) records)
  in
  let header = 16 + String.length matrix in
  let offsets, body = block header [ coverage; classes1; classes2 ] in
  match offsets with
  | [ cov; c1; c2 ] ->
      cat
        [
          be16 2;
          be16 cov;
          be16 format;
          be16 format2;
          be16 c1;
          be16 c2;
          be16 count1;
          be16 count2;
          matrix;
          body;
        ]
  | _ -> assert false

let lookup kind subtables =
  let header = 6 + (2 * List.length subtables) in
  let offsets, body = block header subtables in
  cat
    ([ be16 kind; be16 0; be16 (List.length subtables) ]
    @ List.map be16 offsets @ [ body ])

let extension kind subtable = cat [ be16 1; be16 kind; be32 8; subtable ]

(* A GPOS table whose features are [(tag, lookup indices)]. *)
let gpos_table features lookups =
  let feature (_, indices) =
    cat (be16 0 :: be16 (List.length indices) :: List.map be16 indices)
  in
  let header = 2 + (6 * List.length features) in
  let offsets, body = block header (List.map feature features) in
  let feature_list =
    cat
      (be16 (List.length features)
       :: List.map2 (fun (tag, _) off -> tag ^ be16 off) features offsets
      @ [ body ])
  in
  let offsets, body = block (2 + (2 * List.length lookups)) lookups in
  let lookup_list =
    cat ((be16 (List.length lookups) :: List.map be16 offsets) @ [ body ])
  in
  let offsets, body = block 10 [ be16 0; feature_list; lookup_list ] in
  match offsets with
  | [ scripts; features; lookups ] ->
      cat [ be16 1; be16 0; be16 scripts; be16 features; be16 lookups; body ]
  | _ -> assert false
