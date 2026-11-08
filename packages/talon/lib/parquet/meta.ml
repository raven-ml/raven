(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type bigbytes = Thrift.bigbytes

exception
  Error of {
    row_group : int option;
    bytes : (int * int) option;
    text : string option;
    msg : string;
  }

let fail ?row_group ?bytes ?text fmt =
  Printf.ksprintf (fun msg -> raise (Error { row_group; bytes; text; msg })) fmt

(* Schema *)

type physical =
  | Boolean
  | Int32
  | Int64
  | Int96
  | Float
  | Double
  | Byte_array
  | Fixed_len_byte_array

type repetition = Required | Optional | Repeated

type logical =
  | String
  | Map
  | List
  | Enum
  | Decimal of { precision : int; scale : int }
  | Date
  | Time of { unit_ : Talon.Type.unit_; utc : bool }
  | Timestamp of { unit_ : Talon.Type.unit_; utc : bool }
  | Integer of { bits : int; signed : bool }
  | Null
  | Json
  | Bson
  | Uuid
  | Float16
  | Variant
  | Geometry
  | Geography
  | Other of int

type element = {
  name : string;
  physical : physical option;
  length : int;
  repetition : repetition option;
  children : int;
  converted : int option;
  precision : int;
  scale : int;
  logical : logical option;
}

(* Footer *)

type codec =
  | Uncompressed
  | Snappy
  | Gzip
  | Lzo
  | Brotli
  | Lz4
  | Zstd
  | Lz4_raw
  | Unknown_codec of int

type stats = {
  nulls : int option;
  nans : int option;
  min : string option;
  max : string option;
  min_exact : bool option;
  max_exact : bool option;
}

type column_meta = {
  physical : physical;
  path : string list;
  codec : codec;
  values : int;
  uncompressed_size : int;
  compressed_size : int;
  data_page : int;
  dictionary_page : int option;
  stats : stats;
}

type chunk = {
  file_path : string option;
  encrypted : bool;
  meta : column_meta option;
}

type row_group = { rows : int; chunks : chunk array }

type file = {
  rows : int;
  schema : element array;
  row_groups : row_group array;
  footer : int;
}

type encoding =
  | Plain
  | Plain_dictionary
  | Rle
  | Bit_packed
  | Delta_binary_packed
  | Delta_length_byte_array
  | Delta_byte_array
  | Rle_dictionary
  | Byte_stream_split
  | Unknown_encoding of int

(* Decoding fields *)

let malformed pos fmt =
  Printf.ksprintf (fun m -> raise (Thrift.Error (pos, m))) fmt

let required start what field = function
  | Some v -> v
  | None -> malformed start "%s has no field %s" what field

(* [count int r ty] reads a size, a count or an offset with [int]. *)
let count int r ty =
  let pos = Thrift.pos r in
  let n = int r ty in
  if n < 0 then malformed pos "a negative size or count, %d" n;
  n

let enum r ty what of_int =
  let pos = Thrift.pos r in
  let n = Thrift.i32 r ty in
  match of_int n with
  | Some v -> v
  | None -> malformed pos "an undefined %s, %d" what n

let physical = function
  | 0 -> Some Boolean
  | 1 -> Some Int32
  | 2 -> Some Int64
  | 3 -> Some Int96
  | 4 -> Some Float
  | 5 -> Some Double
  | 6 -> Some Byte_array
  | 7 -> Some Fixed_len_byte_array
  | _ -> None

let repetition = function
  | 0 -> Some Required
  | 1 -> Some Optional
  | 2 -> Some Repeated
  | _ -> None

let codec r ty =
  match Thrift.i32 r ty with
  | 0 -> Uncompressed
  | 1 -> Snappy
  | 2 -> Gzip
  | 3 -> Lzo
  | 4 -> Brotli
  | 5 -> Lz4
  | 6 -> Zstd
  | 7 -> Lz4_raw
  | n -> Unknown_codec n

let encoding r ty =
  match Thrift.i32 r ty with
  | 0 -> Plain
  | 2 -> Plain_dictionary
  | 3 -> Rle
  | 4 -> Bit_packed
  | 5 -> Delta_binary_packed
  | 6 -> Delta_length_byte_array
  | 7 -> Delta_byte_array
  | 8 -> Rle_dictionary
  | 9 -> Byte_stream_split
  | n -> Unknown_encoding n

(* [unit_ r ty] is the [TimeUnit] union, [None] for a member talon does not
   know. *)
let unit_ r ty =
  let u = ref None in
  Thrift.structure r ty (fun id ty ->
      (u :=
         match id with
         | 1 -> Some Talon.Type.Ms
         | 2 -> Some Us
         | 3 -> Some Ns
         | _ -> None);
      Thrift.skip r ty);
  !u

let temporal r ty id make =
  let utc = ref None and unit_' = ref None in
  Thrift.structure r ty (fun field ty ->
      match field with
      | 1 -> utc := Some (Thrift.bool r ty)
      | 2 -> unit_' := Some (unit_ r ty)
      | _ -> Thrift.skip r ty);
  match (!utc, !unit_') with
  | Some utc, Some (Some unit_) -> make unit_ utc
  | _ -> Other id

let logical r ty =
  let l = ref (Other 0) in
  Thrift.structure r ty (fun id ty ->
      let empty v =
        Thrift.skip r ty;
        v
      in
      l :=
        match id with
        | 1 -> empty String
        | 2 -> empty Map
        | 3 -> empty List
        | 4 -> empty Enum
        | 5 ->
            let start = Thrift.pos r in
            let scale = ref None and precision = ref None in
            Thrift.structure r ty (fun id ty ->
                match id with
                | 1 -> scale := Some (Thrift.i32 r ty)
                | 2 -> precision := Some (Thrift.i32 r ty)
                | _ -> Thrift.skip r ty);
            let scale = required start "DecimalType" "1 (scale)" !scale in
            let precision =
              required start "DecimalType" "2 (precision)" !precision
            in
            Decimal { precision; scale }
        | 6 -> empty Date
        | 7 -> temporal r ty id (fun unit_ utc -> Time { unit_; utc })
        | 8 -> temporal r ty id (fun unit_ utc -> Timestamp { unit_; utc })
        | 10 -> (
            let bits = ref None and signed = ref None in
            Thrift.structure r ty (fun id ty ->
                match id with
                | 1 -> bits := Some (Thrift.i8 r ty)
                | 2 -> signed := Some (Thrift.bool r ty)
                | _ -> Thrift.skip r ty);
            match (!bits, !signed) with
            | Some bits, Some signed -> Integer { bits; signed }
            | _ -> Other id)
        | 11 -> empty Null
        | 12 -> empty Json
        | 13 -> empty Bson
        | 14 -> empty Uuid
        | 15 -> empty Float16
        | 16 -> empty Variant
        | 17 -> empty Geometry
        | 18 -> empty Geography
        | id -> empty (Other id));
  !l

let element r ty =
  let start = Thrift.pos r in
  let name = ref None and physical' = ref None and length = ref 0 in
  let repetition' = ref None and children = ref 0 and converted = ref None in
  let precision = ref 0 and scale = ref 0 and logical' = ref None in
  Thrift.structure r ty (fun id ty ->
      match id with
      | 1 -> physical' := Some (enum r ty "physical type" physical)
      | 2 -> length := count Thrift.i32 r ty
      | 3 -> repetition' := Some (enum r ty "repetition" repetition)
      | 4 -> name := Some (Thrift.binary r ty)
      | 5 -> children := count Thrift.i32 r ty
      | 6 -> converted := Some (Thrift.i32 r ty)
      | 7 -> scale := Thrift.i32 r ty
      | 8 -> precision := Thrift.i32 r ty
      | 10 -> logical' := Some (logical r ty)
      | _ -> Thrift.skip r ty);
  {
    name = required start "SchemaElement" "4 (name)" !name;
    physical = !physical';
    length = !length;
    repetition = !repetition';
    children = !children;
    converted = !converted;
    precision = !precision;
    scale = !scale;
    logical = !logical';
  }

let stats r ty =
  let nulls = ref None and nans = ref None in
  let min = ref None and max = ref None in
  let min_exact = ref None and max_exact = ref None in
  Thrift.structure r ty (fun id ty ->
      match id with
      | 3 -> nulls := Some (Thrift.i64 r ty)
      | 5 -> max := Some (Thrift.binary r ty)
      | 6 -> min := Some (Thrift.binary r ty)
      | 7 -> max_exact := Some (Thrift.bool r ty)
      | 8 -> min_exact := Some (Thrift.bool r ty)
      | 9 -> nans := Some (Thrift.i64 r ty)
      | _ -> Thrift.skip r ty);
  {
    nulls = !nulls;
    nans = !nans;
    min = !min;
    max = !max;
    min_exact = !min_exact;
    max_exact = !max_exact;
  }

let no_stats =
  {
    nulls = None;
    nans = None;
    min = None;
    max = None;
    min_exact = None;
    max_exact = None;
  }

let column_meta r ty =
  let start = Thrift.pos r in
  let physical' = ref None and path = ref None and codec' = ref None in
  let values = ref None and uncompressed = ref None and compressed = ref None in
  let data_page = ref None and dictionary_page = ref None in
  let stats' = ref no_stats in
  Thrift.structure r ty (fun id ty ->
      match id with
      | 1 -> physical' := Some (enum r ty "physical type" physical)
      | 3 -> path := Some (Thrift.list r ty Thrift.binary)
      | 4 -> codec' := Some (codec r ty)
      | 5 -> values := Some (count Thrift.i64 r ty)
      | 6 -> uncompressed := Some (count Thrift.i64 r ty)
      | 7 -> compressed := Some (count Thrift.i64 r ty)
      | 9 -> data_page := Some (count Thrift.i64 r ty)
      | 11 -> dictionary_page := Some (count Thrift.i64 r ty)
      | 12 -> stats' := stats r ty
      | _ -> Thrift.skip r ty);
  let get field v = required start "ColumnMetaData" field v in
  let physical = get "1 (type)" !physical' in
  let path = get "3 (path_in_schema)" !path in
  let codec = get "4 (codec)" !codec' in
  let values = get "5 (num_values)" !values in
  let uncompressed_size = get "6 (total_uncompressed_size)" !uncompressed in
  let compressed_size = get "7 (total_compressed_size)" !compressed in
  let data_page = get "9 (data_page_offset)" !data_page in
  let dictionary_page =
    Option.bind !dictionary_page (fun p -> if p > 0 then Some p else None)
  in
  {
    physical;
    path;
    codec;
    values;
    uncompressed_size;
    compressed_size;
    data_page;
    dictionary_page;
    stats = !stats';
  }

let chunk r ty =
  let file_path = ref None and encrypted = ref false and meta = ref None in
  Thrift.structure r ty (fun id ty ->
      match id with
      | 1 -> file_path := Some (Thrift.binary r ty)
      | 3 -> meta := Some (column_meta r ty)
      | 8 | 9 ->
          encrypted := true;
          Thrift.skip r ty
      | _ -> Thrift.skip r ty);
  { file_path = !file_path; encrypted = !encrypted; meta = !meta }

let row_group r ty =
  let start = Thrift.pos r in
  let chunks = ref None and rows = ref None in
  Thrift.structure r ty (fun id ty ->
      match id with
      | 1 -> chunks := Some (Array.of_list (Thrift.list r ty chunk))
      | 3 -> rows := Some (count Thrift.i64 r ty)
      | _ -> Thrift.skip r ty);
  let chunks = required start "RowGroup" "1 (columns)" !chunks in
  let rows = required start "RowGroup" "3 (num_rows)" !rows in
  { rows; chunks }

let file_meta r ~footer =
  let start = Thrift.pos r in
  let schema = ref None and rows = ref None and row_groups = ref None in
  let encrypted = ref false in
  Thrift.fields r (fun id ty ->
      match id with
      | 2 -> schema := Some (Array.of_list (Thrift.list r ty element))
      | 3 -> rows := Some (count Thrift.i64 r ty)
      | 4 -> row_groups := Some (Array.of_list (Thrift.list r ty row_group))
      | 8 ->
          encrypted := true;
          Thrift.skip r ty
      | _ -> Thrift.skip r ty);
  if !encrypted then fail "the file is encrypted, which talon does not read";
  let schema = required start "FileMetaData" "2 (schema)" !schema in
  let rows = required start "FileMetaData" "3 (num_rows)" !rows in
  let row_groups = required start "FileMetaData" "4 (row_groups)" !row_groups in
  { rows; schema; row_groups; footer }

let magic (b : bigbytes) pos s =
  let ok = ref true in
  String.iteri (fun i c -> if b.{pos + i} <> Char.code c then ok := false) s;
  !ok

let footer b =
  let len = Bigarray.Array1.dim b in
  if len < 12 then fail "not a Parquet file: it has %d bytes, fewer than 12" len;
  if magic b (len - 4) "PARE" then
    fail "the file is encrypted, which talon does not read";
  if not (magic b 0 "PAR1") then
    fail ~bytes:(0, 3) "not a Parquet file: it does not start with \"PAR1\"";
  if not (magic b (len - 4) "PAR1") then
    fail
      ~bytes:(len - 4, len - 1)
      "not a Parquet file: it does not end with \"PAR1\"";
  let n =
    b.{len - 8}
    lor (b.{len - 7} lsl 8)
    lor (b.{len - 6} lsl 16)
    lor (b.{len - 5} lsl 24)
  in
  if n > len - 12 then
    fail
      ~bytes:(len - 8, len - 5)
      "the footer's length, %d bytes, does not fit the file" n;
  let footer = len - 8 - n in
  let m =
    try file_meta (Thrift.make b ~pos:footer ~limit:(len - 8)) ~footer
    with Thrift.Error (pos, msg) ->
      fail ~bytes:(pos, pos) "the footer is malformed: %s" msg
  in
  let left =
    Array.fold_left
      (fun left (g : row_group) ->
        if g.rows > left then
          fail "the row groups hold more rows than the file's %d" m.rows;
        left - g.rows)
      m.rows m.row_groups
  in
  if left <> 0 then
    fail "the row groups hold %d rows, and the file says %d" (m.rows - left)
      m.rows;
  m

(* Pages *)

type page =
  | Data of {
      values : int;
      encoding : encoding;
      levels : encoding;
      repetition : encoding option;
    }
  | Data_v2 of {
      values : int;
      nulls : int;
      rows : int;
      encoding : encoding;
      definition_bytes : int;
      repetition_bytes : int;
      compressed : bool;
    }
  | Dictionary of { values : int; encoding : encoding }
  | Other

type header = {
  page : page;
  uncompressed_size : int;
  compressed_size : int;
  crc : int option;
}

let page_header b ~pos ~limit =
  let r = Thrift.make b ~pos ~limit in
  let page_type = ref None
  and uncompressed = ref None
  and compressed = ref None in
  let crc = ref None
  and data = ref None
  and v2 = ref None
  and dict = ref None in
  let data_page r ty =
    let start = Thrift.pos r in
    let values = ref None and enc = ref None and levels = ref None in
    let repetition = ref None in
    Thrift.structure r ty (fun id ty ->
        match id with
        | 1 -> values := Some (count Thrift.i32 r ty)
        | 2 -> enc := Some (encoding r ty)
        | 3 -> levels := Some (encoding r ty)
        | 4 -> repetition := Some (encoding r ty)
        | _ -> Thrift.skip r ty);
    let get field v = required start "DataPageHeader" field v in
    let values = get "1 (num_values)" !values in
    let encoding = get "2 (encoding)" !enc in
    let levels = get "3 (definition_level_encoding)" !levels in
    Data { values; encoding; levels; repetition = !repetition }
  in
  let data_page_v2 r ty =
    let start = Thrift.pos r in
    let values = ref None and nulls = ref None and rows = ref None in
    let enc = ref None and def = ref None and rep = ref None in
    let compressed = ref true in
    Thrift.structure r ty (fun id ty ->
        match id with
        | 1 -> values := Some (count Thrift.i32 r ty)
        | 2 -> nulls := Some (count Thrift.i32 r ty)
        | 3 -> rows := Some (count Thrift.i32 r ty)
        | 4 -> enc := Some (encoding r ty)
        | 5 -> def := Some (count Thrift.i32 r ty)
        | 6 -> rep := Some (count Thrift.i32 r ty)
        | 7 -> compressed := Thrift.bool r ty
        | _ -> Thrift.skip r ty);
    let get field v = required start "DataPageHeaderV2" field v in
    let values = get "1 (num_values)" !values in
    let nulls = get "2 (num_nulls)" !nulls in
    let rows = get "3 (num_rows)" !rows in
    let encoding = get "4 (encoding)" !enc in
    let definition_bytes = get "5 (definition_levels_byte_length)" !def in
    let repetition_bytes = get "6 (repetition_levels_byte_length)" !rep in
    Data_v2
      {
        values;
        nulls;
        rows;
        encoding;
        definition_bytes;
        repetition_bytes;
        compressed = !compressed;
      }
  in
  let dictionary_page r ty =
    let start = Thrift.pos r in
    let values = ref None and enc = ref None in
    Thrift.structure r ty (fun id ty ->
        match id with
        | 1 -> values := Some (count Thrift.i32 r ty)
        | 2 -> enc := Some (encoding r ty)
        | _ -> Thrift.skip r ty);
    let get field v = required start "DictionaryPageHeader" field v in
    let values = get "1 (num_values)" !values in
    let encoding = get "2 (encoding)" !enc in
    Dictionary { values; encoding }
  in
  Thrift.fields r (fun id ty ->
      match id with
      | 1 -> page_type := Some (Thrift.i32 r ty)
      | 2 -> uncompressed := Some (count Thrift.i32 r ty)
      | 3 -> compressed := Some (count Thrift.i32 r ty)
      | 4 -> crc := Some (Thrift.i32 r ty land 0xFFFF_FFFF)
      | 5 -> data := Some (data_page r ty)
      | 7 -> dict := Some (dictionary_page r ty)
      | 8 -> v2 := Some (data_page_v2 r ty)
      | _ -> Thrift.skip r ty);
  let get field v = required pos "PageHeader" field v in
  let page_type = get "1 (type)" !page_type in
  let uncompressed_size = get "2 (uncompressed_page_size)" !uncompressed in
  let compressed_size = get "3 (compressed_page_size)" !compressed in
  let page =
    match page_type with
    | 0 -> get "5 (data_page_header)" !data
    | 2 -> get "7 (dictionary_page_header)" !dict
    | 3 -> get "8 (data_page_header_v2)" !v2
    | _ -> Other
  in
  ({ page; uncompressed_size; compressed_size; crc = !crc }, Thrift.pos r)

(* Writing *)

type column_out = { column : column_meta; encodings : encoding list }

let physical_code = function
  | Boolean -> 0
  | Int32 -> 1
  | Int64 -> 2
  | Int96 -> 3
  | Float -> 4
  | Double -> 5
  | Byte_array -> 6
  | Fixed_len_byte_array -> 7

let repetition_code = function Required -> 0 | Optional -> 1 | Repeated -> 2

let codec_code = function
  | Uncompressed -> 0
  | Snappy -> 1
  | Gzip -> 2
  | Lzo -> 3
  | Brotli -> 4
  | Lz4 -> 5
  | Zstd -> 6
  | Lz4_raw -> 7
  | Unknown_codec n -> n

let encoding_code = function
  | Plain -> 0
  | Plain_dictionary -> 2
  | Rle -> 3
  | Bit_packed -> 4
  | Delta_binary_packed -> 5
  | Delta_length_byte_array -> 6
  | Delta_byte_array -> 7
  | Rle_dictionary -> 8
  | Byte_stream_split -> 9
  | Unknown_encoding n -> n

let opt id f = function Some v -> [ (id, f v) ] | None -> []

(* [converted] is the deprecated annotation written beside a logical type, as
   LogicalTypes.md asks of writers. *)
let converted : logical -> int option = function
  | String -> Some 0
  | Map -> Some 1
  | List -> Some 3
  | Enum -> Some 4
  | Decimal _ -> Some 5
  | Date -> Some 6
  | Time { unit_ = Ms; _ } -> Some 7
  | Time { unit_ = Us; _ } -> Some 8
  | Timestamp { unit_ = Ms; _ } -> Some 9
  | Timestamp { unit_ = Us; _ } -> Some 10
  | Integer { bits; signed } -> (
      match (bits, signed) with
      | 8, false -> Some 11
      | 16, false -> Some 12
      | 32, false -> Some 13
      | 64, false -> Some 14
      | 8, true -> Some 15
      | 16, true -> Some 16
      | 32, true -> Some 17
      | 64, true -> Some 18
      | _ -> None)
  | Json -> Some 19
  | Bson -> Some 20
  | _ -> None

let unit_value : Talon.Type.unit_ -> Thrift.value = function
  | S -> invalid_arg "Meta: Parquet has no TimeUnit for seconds"
  | Ms -> Struct [ (1, Struct []) ]
  | Us -> Struct [ (2, Struct []) ]
  | Ns -> Struct [ (3, Struct []) ]

let logical_value : logical -> Thrift.value = function
  | String -> Struct [ (1, Struct []) ]
  | List -> Struct [ (3, Struct []) ]
  | Date -> Struct [ (6, Struct []) ]
  | Time { unit_; utc } ->
      Struct [ (7, Struct [ (1, Bool utc); (2, unit_value unit_) ]) ]
  | Timestamp { unit_; utc } ->
      Struct [ (8, Struct [ (1, Bool utc); (2, unit_value unit_) ]) ]
  | Integer { bits; signed } ->
      Struct [ (10, Struct [ (1, I8 bits); (2, Bool signed) ]) ]
  | Float16 -> Struct [ (15, Struct []) ]
  | _ -> invalid_arg "Meta: this logical type is not written"

let element_value (e : element) : Thrift.value =
  let conv = Option.bind e.logical converted in
  Struct
    (opt 1 (fun p -> Thrift.I32 (physical_code p)) e.physical
    @ (if e.length > 0 then [ (2, Thrift.I32 e.length) ] else [])
    @ opt 3 (fun r -> Thrift.I32 (repetition_code r)) e.repetition
    @ [ (4, Thrift.Binary e.name) ]
    @ (if e.children > 0 then [ (5, Thrift.I32 e.children) ] else [])
    @ opt 6 (fun c -> Thrift.I32 c) conv
    @ opt 10 logical_value e.logical)

let stats_value (s : stats) : Thrift.value =
  Struct
    (opt 3 (fun n -> Thrift.I64 n) s.nulls
    @ opt 5 (fun v -> Thrift.Binary v) s.max
    @ opt 6 (fun v -> Thrift.Binary v) s.min
    @ opt 7 (fun v -> Thrift.Bool v) s.max_exact
    @ opt 8 (fun v -> Thrift.Bool v) s.min_exact
    @ opt 9 (fun n -> Thrift.I64 n) s.nans)

let column_value (c : column_out) : Thrift.value =
  let m = c.column in
  Struct
    ([
       (1, Thrift.I32 (physical_code m.physical));
       ( 2,
         Thrift.List
           (List.map (fun e -> Thrift.I32 (encoding_code e)) c.encodings) );
       (3, Thrift.List (List.map (fun p -> Thrift.Binary p) m.path));
       (4, Thrift.I32 (codec_code m.codec));
       (5, Thrift.I64 m.values);
       (6, Thrift.I64 m.uncompressed_size);
       (7, Thrift.I64 m.compressed_size);
       (9, Thrift.I64 m.data_page);
     ]
    @ opt 11 (fun p -> Thrift.I64 p) m.dictionary_page
    @ [ (12, stats_value m.stats) ])

let chunk_value (c : column_out) : Thrift.value =
  Struct [ (2, I64 0); (3, column_value c) ]

let row_group_value (columns, rows) : Thrift.value =
  let size =
    List.fold_left (fun t c -> t + c.column.uncompressed_size) 0 columns
  in
  let compressed =
    List.fold_left (fun t c -> t + c.column.compressed_size) 0 columns
  in
  Struct
    [
      (1, Thrift.List (List.map chunk_value columns));
      (2, Thrift.I64 size);
      (3, Thrift.I64 rows);
      (6, Thrift.I64 compressed);
    ]

(* [footer_bytes ~schema ~row_groups ~rows ~leaves ~created_by] is the
   FileMetaData, version 1, as Parquet's own definition recommends, with a type
   order for each of the [leaves] leaf columns: without one, min_value and
   max_value mean nothing. *)
let footer_bytes ~schema ~row_groups ~rows ~leaves ~created_by =
  let b = Buffer.create 1024 in
  Thrift.encode b
    (Struct
       [
         (1, I32 1);
         (2, List (Array.to_list (Array.map element_value schema)));
         (3, I64 rows);
         (4, List (List.map row_group_value row_groups));
         (6, Binary created_by);
         ( 7,
           List
             (List.init leaves (fun _ -> Thrift.Struct [ (1, Struct []) ])) );
       ]);
  Buffer.contents b

let data_page_bytes ~values ~encoding ~uncompressed ~compressed =
  let b = Buffer.create 64 in
  Thrift.encode b
    (Struct
       [
         (1, I32 0);
         (2, I32 uncompressed);
         (3, I32 compressed);
         ( 5,
           Struct
             [
               (1, I32 values);
               (2, I32 (encoding_code encoding));
               (3, I32 (encoding_code Rle));
               (4, I32 (encoding_code Rle));
             ] );
       ]);
  Buffer.contents b

let dictionary_page_bytes ~values ~uncompressed ~compressed =
  let b = Buffer.create 32 in
  Thrift.encode b
    (Struct
       [
         (1, I32 2);
         (2, I32 uncompressed);
         (3, I32 compressed);
         (7, Struct [ (1, I32 values); (2, I32 (encoding_code Plain)) ]);
       ]);
  Buffer.contents b
