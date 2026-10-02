(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next
open Windtrap
module P = Talon_next_parquet

(* Fixtures *)

let path name = Filename.concat "fixtures" name

let buffer name =
  match Nx_device.Buffer.of_file (path name) with
  | Ok b -> b
  | Error why -> failwith why

let file_bytes name = In_channel.with_open_bin (path name) In_channel.input_all

let of_string s =
  Nx_device.Buffer.of_bigarray
    (Bigarray.Array1.init Bigarray.int8_unsigned Bigarray.c_layout
       (String.length s) (fun i -> Char.code s.[i]))

let sniff name = Error.get_ok (P.sniff (buffer name))
let format_text f = Format.asprintf "%a" P.pp_format f
let error_text e = Format.asprintf "%a" Error.pp e
let outcome = function Ok _ -> "ok" | Error e -> error_text e

(* [read_from f b column] is the column [column] of the file [b], read as [f],
   through a query of its source. *)
let read_from f b column =
  Query.of_source (P.source f b)
  |> Query.select Expr.[ keep (Sel.names [ column ]) ]
  |> Query.run
  |> Result.map (fun t -> Talon_next.column t column)

let read f name column = read_from f (buffer name) column

(* Rendering, as fixtures.py renders pyarrow's values *)

let quote s =
  let b = Buffer.create (String.length s + 2) in
  Buffer.add_char b '"';
  String.iter
    (fun c ->
      if c >= ' ' && c <= '~' && c <> '"' && c <> '\\' then Buffer.add_char b c
      else Printf.bprintf b "\\x%02x" (Char.code c))
    s;
  Buffer.add_char b '"';
  Buffer.contents b

(* The nx dtype that stores a talon type, by the type's printed form. *)
let storage ty =
  let starts p = String.starts_with ~prefix:p ty in
  if starts "clock" || starts "datetime" then "int64"
  else if ty = "date" then "int32"
  else ty

let cells ty (Nx.P x) =
  let map f a = Array.map f (Nx.to_array a) in
  match ty with
  | "bool" ->
      map (fun b -> if b = 1 then "true" else "false") (Nx.cast Nx.uint8 x)
  | "uint64" -> map (Printf.sprintf "%Lu") (Nx.bitcast Nx.int64 x)
  | "float16" -> map (Printf.sprintf "0x%04x") (Nx.bitcast Nx.uint16 x)
  | "float32" -> map (Printf.sprintf "0x%08lx") (Nx.bitcast Nx.int32 x)
  | "float64" -> map (Printf.sprintf "0x%016Lx") (Nx.bitcast Nx.int64 x)
  | _ -> map Int64.to_string (Nx.cast Nx.int64 x)

let render ty c =
  let nulls validity cells =
    match validity with
    | None -> cells
    | Some v ->
        let valid = Nx.to_array (Nx_bits.to_bool v) in
        Array.map2 (fun ok s -> if ok then s else "null") valid cells
  in
  match Column.layout c with
  | Fixed { validity; values = Nx.P x as values } ->
      let dtype = Format.asprintf "%a" Nx.pp_dtype (Nx.dtype x) in
      equal string ~msg:"storage dtype" (storage ty) dtype;
      nulls validity (cells ty values)
  | Varsize { validity; offsets; child } ->
      let o = Nx.to_array offsets in
      let d = Nx.to_array (Column.to_tensor Nx.uint8 child) in
      let cell i =
        let first = Int64.to_int o.(i) and last = Int64.to_int o.(i + 1) in
        quote (String.init (last - first) (fun k -> Char.chr d.(first + k)))
      in
      nulls validity (Array.init (Array.length o - 1) cell)
  | Children _ -> fail "a record column"

let line file g column ty values =
  let values = Array.to_list values in
  let md5 = Digest.to_hex (Digest.string (String.concat " " values)) in
  let first = List.filteri (fun i _ -> i < 8) values in
  String.trim
    (Printf.sprintf "%s %d %s %s %d %s %s" file g (quote column) (quote ty)
       (List.length values) md5 (String.concat " " first))

(* The type [f] reads [column] as, from its printed form. *)
let format_type f column =
  let s =
    Format.asprintf "%a" Schema.pp (Schema.v [ (column, Type.Any Type.bool) ])
  in
  let prefix = "  " ^ String.sub s 0 (String.length s - 4) in
  match
    List.find_opt
      (String.starts_with ~prefix)
      (String.split_on_char '\n' (format_text f))
  with
  | None -> failf "no column %S in the format" column
  | Some l ->
      let rest =
        String.sub l (String.length prefix)
          (String.length l - String.length prefix)
      in
      String.trim (String.sub rest 0 (String.index rest '\xe2'))

(* Lines of values.txt and overrides.txt: [(file, line)], the expected line
   being [file group "column" "type" rows md5 values] or [file group "column"
   error]. *)
let expected name =
  In_channel.with_open_text name In_channel.input_lines
  |> List.map (fun l -> (List.hd (String.split_on_char ' ' l), l))

let files lines = List.sort_uniq String.compare (List.map fst lines)
let values = expected "values.txt"
let overrides = expected "overrides.txt"
let fields l = Scanf.sscanf l "%_s %d %S %[^\n]" (fun g c rest -> (g, c, rest))

let lines_of lines file =
  List.filter_map (fun (name, l) -> if name = file then Some l else None) lines

(* [declared name] is the format of the file [name], whose every column has a
   line in values.txt, with its decimals declared as float64. *)
let declared name =
  let columns =
    List.sort_uniq String.compare
      (List.map
         (fun l ->
           let _, c, _ = fields l in
           c)
         (lines_of values name))
  in
  let declare f c =
    if format_type f c = "undeclared" then
      P.with_type c (Type.Any Type.float64) f
    else f
  in
  List.fold_left declare (sniff name) columns

(* Each column of [file] is read whole, then cut into its row groups' lines. A
   column that fails fails at the first row group whose line is [error]. *)
let agree ?(retype = fun f _ _ -> f) lines file =
  let f = declared file in
  let lines = lines_of lines file in
  let check column =
    let groups =
      List.filter_map
        (fun l ->
          let g, c, rest = fields l in
          if c = column then Some (g, l, rest) else None)
        lines
    in
    match List.find_opt (fun (_, _, rest) -> rest = "error") groups with
    | Some (g, _, _) -> (
        match read f file column with
        | Ok _ -> failf "%s: read a column that fails" column
        | Error e ->
            let place = List.hd (String.split_on_char ':' (error_text e)) in
            equal string ~msg:column (Printf.sprintf "row group %d" g) place)
    | None -> (
        let _, _, rest = List.hd groups in
        let ty = Scanf.sscanf rest "%S" Fun.id in
        let f = retype f column ty in
        equal string ~msg:column ty (format_type f column);
        match read f file column with
        | Error e -> failf "%s: %s" column (error_text e)
        | Ok c ->
            let cells = render ty c in
            let cut first (g, l, rest) =
              let rows = Scanf.sscanf rest "%_S %d" Fun.id in
              equal string l
                (line file g column ty (Array.sub cells first rows));
              first + rows
            in
            let rows = List.fold_left cut 0 groups in
            equal ~msg:column int (Array.length cells) rows)
  in
  List.iter check
    (List.sort_uniq String.compare
       (List.map
          (fun l ->
            let _, c, _ = fields l in
            c)
          lines))

(* Agreement *)

let override f column = function
  | "string" -> P.with_type column (Type.Any Type.string) f
  | "datetime[us]" -> P.with_type column (Type.Any (Type.datetime Us)) f
  | "int64" -> P.with_type column (Type.Any Type.int64) f
  | ty -> failf "no override for %s" ty

let agreement =
  group "Columns read as pyarrow reads them"
    [
      cases ~name:Fun.id "defaults" (files values) (agree values);
      cases ~name:Fun.id "with_type" (files overrides)
        (agree ~retype:override overrides);
    ]

(* [footer s] is the position of the footer of the Parquet file [s]. *)
let footer s =
  let n = String.length s in
  n - 8 - Int32.to_int (String.get_int32_le s (n - 8))

(* [cut_footer s k] is the Parquet file [s] with the last [k] bytes of its
   footer cut, and the footer's length rewritten. *)
let cut_footer s k =
  let first = footer s in
  let len = String.length s - 8 - first - k in
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.of_int len);
  String.sub s 0 (first + len) ^ Bytes.to_string b ^ "PAR1"

(* Sniffing *)

let sniffing =
  group "sniff"
    [
      cases ~name:Fun.id "reads a file and its bytes alike"
        [
          "alltypes_plain.parquet";
          "types.parquet";
          "names.parquet";
          "empty.parquet";
        ] (fun name ->
          let bytes = Error.get_ok (P.sniff (of_string (file_bytes name))) in
          equal text (format_text (sniff name)) (format_text bytes));
      test "refuses what talon cannot read in full" (fun () ->
          let refused =
            [
              "hadoop_lz4_compressed.parquet";
              "non_hadoop_lz4_compressed.parquet";
              "brotli.parquet";
              "large_string_map.brotli.parquet";
              "map_no_value.parquet";
              "incorrect_map_schema.parquet";
              "datapage_v2.snappy.parquet";
              "list_columns.parquet";
              "nulls.snappy.parquet";
              "repeated_primitive_no_list.parquet";
              "uniform_encryption.parquet.encrypted";
              "encrypt_columns_plaintext_footer.parquet.encrypted";
              "bad_data/PARQUET-1481.parquet";
              "bad_data/ARROW-RS-GH-6229-DICTHEADER.parquet";
            ]
          in
          List.iter
            (fun name ->
              Printf.printf "%s: %s\n" name (outcome (P.sniff (buffer name))))
            refused;
          expect (output ())
          @@ __POS_OF__
               {|
            hadoop_lz4_compressed.parquet: row group 0: column "c0" is compressed with LZ4 in Hadoop's framing, which talon does not read; LZ4_RAW is read
            non_hadoop_lz4_compressed.parquet: row group 0: column "c0" is compressed with LZ4 in Hadoop's framing, which talon does not read; LZ4_RAW is read
            brotli.parquet: row group 0: column "i32" is compressed with Brotli, which talon does not read
            large_string_map.brotli.parquet: column "arr" is a map: talon reads flat Parquet files only
            map_no_value.parquet: column "my_map" is a map: talon reads flat Parquet files only
            incorrect_map_schema.parquet: column "my_map" is a map: talon reads flat Parquet files only
            datapage_v2.snappy.parquet: column "e" is a list: talon reads flat Parquet files only
            list_columns.parquet: column "int64_list" is a list: talon reads flat Parquet files only
            nulls.snappy.parquet: column "b_struct" is a record: talon reads flat Parquet files only
            repeated_primitive_no_list.parquet: column "Int32_list" is a repeated field: talon reads flat Parquet files only
            uniform_encryption.parquet.encrypted: the file is encrypted, which talon does not read
            encrypt_columns_plaintext_footer.parquet.encrypted: the file is encrypted, which talon does not read
            bad_data/PARQUET-1481.parquet: byte 307: the footer is malformed: an undefined physical type, -7
            bad_data/ARROW-RS-GH-6229-DICTHEADER.parquet: row group 0: bytes 129-450: column "name": its pages overrun the file, whose footer starts at byte 291
            |});
      test "refuses what is not a Parquet file" (fun () ->
          let file = file_bytes "alltypes_plain.parquet" in
          let n = String.length file in
          let cases =
            [
              ("no bytes", "");
              ("11 bytes", "PAR1PAR1PAR");
              ("PAR1 at the start only", String.sub file 0 (n - 1) ^ "X");
              ("PAR1 at the end only", "X" ^ String.sub file 1 (n - 1));
              ("a footer longer than the file", "PAR1" ^ "\xff\xff\x00\x00PAR1");
              ( "a footer of 5 bytes in 4",
                "PAR1" ^ "\x00\x00\x00\x00" ^ "\x05\x00\x00\x00PAR1" );
              ( "a footer of 4 bytes in 4",
                "PAR1" ^ "\x00\x00\x00\x00" ^ "\x04\x00\x00\x00PAR1" );
              ("a footer cut short", cut_footer file 1);
            ]
          in
          List.iter
            (fun (what, s) ->
              Printf.printf "%s: %s\n" what (outcome (P.sniff (of_string s))))
            cases;
          expect (output ())
          @@ __POS_OF__
               {|
            no bytes: not a Parquet file: it has 0 bytes, fewer than 12
            11 bytes: not a Parquet file: it has 11 bytes, fewer than 12
            PAR1 at the start only: bytes 1847-1850: not a Parquet file: it does not end with "PAR1"
            PAR1 at the end only: bytes 0-3: not a Parquet file: it does not start with "PAR1"
            a footer longer than the file: bytes 4-7: the footer's length, 65535 bytes, does not fit the file
            a footer of 5 bytes in 4: bytes 8-11: the footer's length, 5 bytes, does not fit the file
            a footer of 4 bytes in 4: byte 4: the footer is malformed: FileMetaData has no field 2 (schema)
            a footer cut short: byte 1842: the footer is malformed: unexpected end of the data
            |});
    ]

(* Formats *)

let any t = Type.Any t

(* fallback.parquet's string column, read as a categorical of its sorted values
   but one, which then fails, and of all of them, whose codes then read back as
   the strings. *)
let categorical () =
  let name = "fallback.parquet" in
  let f = sniff name in
  let strings =
    match read f name "string" with
    | Ok c -> render "string" c
    | Error e -> failf "%s" (error_text e)
  in
  let dict =
    Array.to_list strings
    |> List.filter (( <> ) "null")
    |> List.sort_uniq compare
    |> List.map (fun s -> Scanf.sscanf s "%S" Fun.id)
    |> Array.of_list
  in
  let read_as d =
    read (P.with_type "string" (any (Type.categorical d)) f) name "string"
  in
  (match read_as (Array.sub dict 1 (Array.length dict - 1)) with
  | Ok _ -> fail "read a value outside the dictionary"
  | Error e ->
      expect (error_text e)
      @@ __POS_OF__
           {| row group 0: "v0é\x0a\"": the value is not in the categorical's dictionary |});
  match read_as dict with
  | Error e -> failf "%s" (error_text e)
  | Ok c ->
      let codes = render "int32" c in
      let back =
        Array.map
          (fun s -> if s = "null" then s else quote dict.(int_of_string s))
          codes
      in
      equal (array string) strings back

let retype f column t =
  match P.with_type column t f with
  | f -> Printf.sprintf "%s reads as %s" column (format_type f column)
  | exception Invalid_argument msg -> msg

let refusal f =
  match f () with _ -> "no refusal" | exception Invalid_argument msg -> msg

let formats =
  group "Formats"
    [
      test "with_type refuses a type the column does not read as" (fun () ->
          let types = sniff "types.parquet"
          and plain = sniff "alltypes_plain.parquet" in
          List.iter print_endline
            [
              retype types "carier" (any Type.string);
              retype types "i32" (any Type.int64);
              retype types "f64" (any Type.float32);
              retype types "string" (any Type.int64);
              retype types "d9" (any Type.binary);
              retype plain "timestamp_col" (any Type.date);
              retype plain "timestamp_col" (any (Type.datetime ~zone:"UTC" Us));
              retype types "ts_us_utc" (any (Type.datetime Us));
              retype types "ts_us_utc"
                (any (Type.datetime ~zone:"Europe/Paris" Us));
              retype types "ts_ms" (any (Type.datetime ~zone:"UTC" Ms));
            ];
          expect (output ())
          @@ __POS_OF__
               {|
            Talon_next_parquet.with_type: no column "carier" in the format.
            Talon_next_parquet.with_type: column "i32" (optional int32) reads as int32, not int64. Cast it once read (Expr.cast).
            Talon_next_parquet.with_type: column "f64" (optional double) reads as float64, not float32. Cast it once read (Expr.cast).
            Talon_next_parquet.with_type: column "string" (optional binary (STRING)) reads as string, binary or a categorical, not int64. Cast it once read (Expr.cast).
            Talon_next_parquet.with_type: column "d9" (optional fixed_len_byte_array(4) (DECIMAL(9,2))) reads as float64 or int64, not binary. Cast it once read (Expr.cast).
            Talon_next_parquet.with_type: column "timestamp_col" (optional int96) reads as datetime of any unit, without a zone, not date. Cast it once read (Expr.cast).
            Talon_next_parquet.with_type: column "timestamp_col" (optional int96) reads as datetime of any unit, without a zone, not datetime[us, UTC]. Cast it once read (Expr.cast).
            Talon_next_parquet.with_type: column "ts_us_utc" (optional int64 (TIMESTAMP(MICROS,true))) reads as datetime[us, UTC], not datetime[us]. Cast it once read (Expr.cast).
            Talon_next_parquet.with_type: column "ts_us_utc" (optional int64 (TIMESTAMP(MICROS,true))) reads as datetime[us, UTC], not datetime[us, Europe/Paris]. Cast it once read (Expr.cast).
            Talon_next_parquet.with_type: column "ts_ms" (optional int64 (TIMESTAMP(MILLIS,false))) reads as datetime[ms], not datetime[ms, UTC]. Cast it once read (Expr.cast).
            |});
      test "with_type of a column's own type is the identity" (fun () ->
          let f = sniff "types.parquet" in
          let same column t =
            equal text (format_text f) (format_text (P.with_type column t f))
          in
          same "i8" (any Type.int8);
          same "u64" (any Type.uint64);
          same "clock_ms" (any (Type.clock Ms));
          same "ts_us_utc" (any (Type.datetime ~zone:"UTC" Us));
          same "string" (any Type.string);
          same "fsb" (any Type.binary));
      test "with_type accepts what the bytes allow" (fun () ->
          let f = sniff "types.parquet"
          and plain = sniff "alltypes_plain.parquet" in
          List.iter print_endline
            [
              retype f "string" (any Type.binary);
              retype f "binary" (any Type.string);
              retype f "fsb" (any (Type.categorical [| "a" |]));
              retype f "uuid" (any Type.string);
              retype plain "timestamp_col" (any (Type.datetime S));
            ];
          expect (output ())
          @@ __POS_OF__
               {|
            string reads as binary
            binary reads as string
            fsb reads as categorical["a"]
            uuid reads as string
            timestamp_col reads as datetime[s]
            |});
      test "a categorical reads codes into its dictionary" categorical;
      test "with_type declares a decimal as float64, or int64 to 18 digits"
        (fun () ->
          let f = sniff "decimals_int.parquet" in
          List.iter print_endline
            [
              retype f "d9_2" (any Type.float64);
              retype f "d18_6" (any Type.int64);
              retype f "d38_10" (any Type.float64);
              retype f "d38_0" (any Type.int64);
              retype f "d9_0" (any Type.int32);
            ];
          expect (output ())
          @@ __POS_OF__
               {|
            d9_2 reads as float64
            d18_6 reads as int64
            d38_10 reads as float64
            Talon_next_parquet.with_type: column "d38_0" (optional fixed_len_byte_array(16) (DECIMAL(38,0))) reads as float64, not int64. Cast it once read (Expr.cast).
            Talon_next_parquet.with_type: column "d9_0" (optional int32 (DECIMAL(9,0))) reads as float64 or int64, not int32. Cast it once read (Expr.cast).
            |});
      test "source and file refuse a decimal left undeclared" (fun () ->
          let name = "decimals_bytes.parquet" in
          let f = P.with_type "d9_0" (any Type.float64) (sniff name) in
          print_endline (refusal (fun () -> P.source f (buffer name)));
          print_endline (refusal (fun () -> P.file (path name)));
          expect (output ())
          @@ __POS_OF__
               {|
            Talon_next_parquet.source: column "d9_2" (optional fixed_len_byte_array(4) (DECIMAL(9,2))) is a decimal, which reads as float64, or as its unscaled int64 integers up to 18 digits. Declare one with with_type.
            Talon_next_parquet.file: column "d9_0" (optional fixed_len_byte_array(4) (DECIMAL(9,0))) is a decimal, which reads as float64, or as its unscaled int64 integers up to 18 digits. Declare one with with_type.
            |});
      cases ~name:fst "pp_format"
        [
          ( "alltypes_plain.parquet",
            __POS_OF__
              {|
            parquet (11 columns)
              id int32                   ← optional int32
              bool_col bool              ← optional boolean
              tinyint_col int32          ← optional int32
              smallint_col int32         ← optional int32
              int_col int32              ← optional int32
              bigint_col int64           ← optional int64
              float_col float32          ← optional float
              double_col float64         ← optional double
              date_string_col binary     ← optional binary
              string_col binary          ← optional binary
              timestamp_col datetime[ns] ← optional int96
            |}
          );
          ( "byte_stream_split_extended.gzip.parquet",
            __POS_OF__
              {|
              parquet (14 columns)
                float16_plain float16                ← optional fixed_len_byte_array(2) (FLOAT16)
                float16_byte_stream_split float16    ← optional fixed_len_byte_array(2) (FLOAT16)
                float_plain float32                  ← optional float
                float_byte_stream_split float32      ← optional float
                double_plain float64                 ← optional double
                double_byte_stream_split float64     ← optional double
                int32_plain int32                    ← optional int32
                int32_byte_stream_split int32        ← optional int32
                int64_plain int64                    ← optional int64
                int64_byte_stream_split int64        ← optional int64
                flba5_plain binary                   ← optional fixed_len_byte_array(5)
                flba5_byte_stream_split binary       ← optional fixed_len_byte_array(5)
                decimal_plain undeclared             ← optional fixed_len_byte_array(4) (DECIMAL(7,3))
                decimal_byte_stream_split undeclared ← optional fixed_len_byte_array(4) (DECIMAL(7,3))
              |}
          );
          ( "unknown-logical-type.parquet",
            __POS_OF__
              {|
            parquet (2 columns)
              "column with known type" string   ← optional binary (STRING)
              "column with unknown type" binary ← optional binary
            |}
          );
          ( "int32_with_uuid_logical_type.parquet",
            __POS_OF__
              {|
            parquet (1 column)
              int32_uuid int32 ← required int32
            |}
          );
          ( "flba12_timestamp.parquet",
            __POS_OF__
              {|
            parquet (3 columns)
              timestamp_millis binary ← optional fixed_len_byte_array(12)
              timestamp_micros binary ← optional fixed_len_byte_array(12)
              timestamp_nanos binary  ← optional fixed_len_byte_array(12)
            |}
          );
          ( "types.parquet",
            __POS_OF__
              {|
              parquet (30 columns)
                bool bool                     ← optional boolean
                i8 int8                       ← optional int32 (INTEGER(8,true))
                i16 int16                     ← optional int32 (INTEGER(16,true))
                i32 int32                     ← optional int32
                i64 int64                     ← optional int64
                u8 uint8                      ← optional int32 (INTEGER(8,false))
                u16 uint16                    ← optional int32 (INTEGER(16,false))
                u32 uint32                    ← optional int32 (INTEGER(32,false))
                u64 uint64                    ← optional int64 (INTEGER(64,false))
                f16 float16                   ← optional fixed_len_byte_array(2) (FLOAT16)
                f32 float32                   ← optional float
                f64 float64                   ← optional double
                d9 undeclared                 ← optional fixed_len_byte_array(4) (DECIMAL(9,2))
                d18 undeclared                ← optional fixed_len_byte_array(8) (DECIMAL(18,6))
                date date                     ← optional int32 (DATE)
                clock_ms clock[ms]            ← optional int32 (TIME(MILLIS,false))
                clock_us clock[us]            ← optional int64 (TIME(MICROS,false))
                clock_ns clock[ns]            ← optional int64 (TIME(NANOS,false))
                ts_ms datetime[ms]            ← optional int64 (TIMESTAMP(MILLIS,false))
                ts_us_utc datetime[us, UTC]   ← optional int64 (TIMESTAMP(MICROS,true))
                ts_us_paris datetime[us, UTC] ← optional int64 (TIMESTAMP(MICROS,true))
                ts_ns datetime[ns]            ← optional int64 (TIMESTAMP(NANOS,false))
                string string                 ← optional binary (STRING)
                binary binary                 ← optional binary
                fsb binary                    ← optional fixed_len_byte_array(5)
                uuid binary                   ← optional fixed_len_byte_array(16) (UUID)
                all_null int32                ← optional int32
                no_null int64                 ← optional int64
                req_i32 int32                 ← required int32
                req_string string             ← required binary (STRING)
              |}
          );
          ( "decimals_int.parquet",
            __POS_OF__
              {|
            parquet (6 columns)
              d9_0 undeclared   ← optional int32 (DECIMAL(9,0))
              d9_2 undeclared   ← optional int32 (DECIMAL(9,2))
              d18_0 undeclared  ← optional int64 (DECIMAL(18,0))
              d18_6 undeclared  ← optional int64 (DECIMAL(18,6))
              d38_0 undeclared  ← optional fixed_len_byte_array(16) (DECIMAL(38,0))
              d38_10 undeclared ← optional fixed_len_byte_array(16) (DECIMAL(38,10))
            |}
          );
          ( "decimals_bytes.parquet",
            __POS_OF__
              {|
            parquet (6 columns)
              d9_0 undeclared   ← optional fixed_len_byte_array(4) (DECIMAL(9,0))
              d9_2 undeclared   ← optional fixed_len_byte_array(4) (DECIMAL(9,2))
              d18_0 undeclared  ← optional fixed_len_byte_array(8) (DECIMAL(18,0))
              d18_6 undeclared  ← optional fixed_len_byte_array(8) (DECIMAL(18,6))
              d38_0 undeclared  ← optional fixed_len_byte_array(16) (DECIMAL(38,0))
              d38_10 undeclared ← optional fixed_len_byte_array(16) (DECIMAL(38,10))
            |}
          );
          ( "names.parquet",
            __POS_OF__
              {|
            parquet (4 columns)
              "a b" int64   ← optional int64
              "\"q\"" int64 ← optional int64
              é int64       ← optional int64
              "" int64      ← optional int64
            |}
          );
          ( "empty.parquet",
            __POS_OF__
              {|
              parquet (30 columns)
                bool bool                     ← optional boolean
                i8 int8                       ← optional int32 (INTEGER(8,true))
                i16 int16                     ← optional int32 (INTEGER(16,true))
                i32 int32                     ← optional int32
                i64 int64                     ← optional int64
                u8 uint8                      ← optional int32 (INTEGER(8,false))
                u16 uint16                    ← optional int32 (INTEGER(16,false))
                u32 uint32                    ← optional int32 (INTEGER(32,false))
                u64 uint64                    ← optional int64 (INTEGER(64,false))
                f16 float16                   ← optional fixed_len_byte_array(2) (FLOAT16)
                f32 float32                   ← optional float
                f64 float64                   ← optional double
                d9 undeclared                 ← optional fixed_len_byte_array(4) (DECIMAL(9,2))
                d18 undeclared                ← optional fixed_len_byte_array(8) (DECIMAL(18,6))
                date date                     ← optional int32 (DATE)
                clock_ms clock[ms]            ← optional int32 (TIME(MILLIS,false))
                clock_us clock[us]            ← optional int64 (TIME(MICROS,false))
                clock_ns clock[ns]            ← optional int64 (TIME(NANOS,false))
                ts_ms datetime[ms]            ← optional int64 (TIMESTAMP(MILLIS,false))
                ts_us_utc datetime[us, UTC]   ← optional int64 (TIMESTAMP(MICROS,true))
                ts_us_paris datetime[us, UTC] ← optional int64 (TIMESTAMP(MICROS,true))
                ts_ns datetime[ns]            ← optional int64 (TIMESTAMP(NANOS,false))
                string string                 ← optional binary (STRING)
                binary binary                 ← optional binary
                fsb binary                    ← optional fixed_len_byte_array(5)
                uuid binary                   ← optional fixed_len_byte_array(16) (UUID)
                all_null int32                ← optional int32
                no_null int64                 ← optional int64
                req_i32 int32                 ← required int32
                req_string string             ← required binary (STRING)
              |}
          );
        ]
        (fun (name, baseline) -> expect (format_text (sniff name)) baseline);
    ]

(* Synthesized files

   Files of one column "x" and one row group, written with a Thrift compact
   encoder, so that their fields can hold any value, including those no writer
   produces: sizes at the limits of their types, and integers of another wire
   type than their field's. *)

type thrift =
  | I32 of int
  | I64 of int
  | Bin of string
  | List of thrift list
  | Struct of (int * thrift) list

let wire = function
  | I32 _ -> 5
  | I64 _ -> 6
  | Bin _ -> 8
  | List _ -> 9
  | Struct _ -> 12

let rec varint b n =
  if Int64.logand n (-128L) = 0L then Buffer.add_uint8 b (Int64.to_int n)
  else begin
    Buffer.add_uint8 b (Int64.to_int (Int64.logand n 0x7FL) lor 0x80);
    varint b (Int64.shift_right_logical n 7)
  end

let zigzag n =
  let n = Int64.of_int n in
  Int64.logxor (Int64.shift_left n 1) (Int64.shift_right n 63)

let rec encode b = function
  | I32 n | I64 n -> varint b (zigzag n)
  | Bin s ->
      varint b (Int64.of_int (String.length s));
      Buffer.add_string b s
  | List vs ->
      let n = List.length vs and ty = wire (List.hd vs) in
      if n < 15 then Buffer.add_uint8 b ((n lsl 4) lor ty)
      else begin
        Buffer.add_uint8 b (0xF0 lor ty);
        varint b (Int64.of_int n)
      end;
      List.iter (encode b) vs
  | Struct fields ->
      let field last (id, v) =
        Buffer.add_uint8 b (((id - last) lsl 4) lor wire v);
        encode b v;
        id
      in
      ignore (List.fold_left field 0 (List.sort compare fields));
      Buffer.add_uint8 b 0

let thrift v =
  let b = Buffer.create 64 in
  encode b v;
  Buffer.contents b

(* [set fields changes] is [fields] with the values of [changes]. *)
let set fields changes =
  List.map
    (fun (id, v) -> (id, Option.value ~default:v (List.assoc_opt id changes)))
    fields
  @ List.filter (fun (id, _) -> not (List.mem_assoc id fields)) changes

let int32s vs =
  let b = Bytes.create (4 * List.length vs) in
  List.iteri (fun i v -> Bytes.set_int32_le b (4 * i) (Int32.of_int v)) vs;
  Bytes.to_string b

(* [page ?header ?data_page values data] is a [DATA_PAGE] of [values] values
   whose bytes are [data], with the changes [header] and [data_page] to its
   header's fields. *)
let page ?(header = []) ?(data_page = []) values data =
  let size = I32 (String.length data) in
  let data_page = set [ (1, I32 values); (2, I32 0); (3, I32 3) ] data_page in
  thrift
    (Struct
       (set [ (1, I32 0); (2, size); (3, size); (5, Struct data_page) ] header))
  ^ data

(* [row_groups ?column groups] is a file of a required int32 column with a row
   group [(rows, chunk, pages)] of [rows] rows in [pages] for each of [groups],
   with the changes [column] to its schema element and [chunk] to the group's
   column metadata. *)
let row_groups ?(column = []) groups =
  let column = set [ (1, I32 1); (3, I32 0); (4, Bin "x") ] column in
  let group (pos, gs) (rows, chunk, pages) =
    let size = I64 (String.length pages) in
    let chunk =
      set
        [
          (1, List.assoc 1 column);
          (3, List [ Bin "x" ]);
          (4, I32 0);
          (5, I64 rows);
          (6, size);
          (7, size);
          (9, I64 pos);
        ]
        chunk
    in
    let g =
      Struct [ (1, List [ Struct [ (3, Struct chunk) ] ]); (3, I64 rows) ]
    in
    (pos + String.length pages, g :: gs)
  in
  let _, gs = List.fold_left group (4, []) groups in
  let rows = List.fold_left (fun n (r, _, _) -> n + r) 0 groups in
  let meta =
    thrift
      (Struct
         [
           (2, List [ Struct [ (4, Bin "schema"); (5, I32 1) ]; Struct column ]);
           (3, I64 rows);
           (4, List (List.rev gs));
         ])
  in
  let len = Bytes.create 4 in
  Bytes.set_int32_le len 0 (Int32.of_int (String.length meta));
  let pages = String.concat "" (List.map (fun (_, _, p) -> p) groups) in
  of_string ("PAR1" ^ pages ^ meta ^ Bytes.to_string len ^ "PAR1")

let parquet ?column ?(chunk = []) ~rows pages =
  row_groups ?column [ (rows, chunk, pages) ]

let read_x b =
  match P.sniff b with Error e -> Error e | Ok f -> read_from f b "x"

let i32_max = 0x7FFF_FFFF

let synthesized =
  group "Synthesized files"
    [
      test "a file of two int32s reads them" (fun () ->
          match read_x (parquet ~rows:2 (page 2 (int32s [ 7; -8 ]))) with
          | Ok c -> equal (array string) [| "7"; "-8" |] (render "int32" c)
          | Error e -> failf "%s" (error_text e));
      test "a decimal annotation of a negative scale is ignored" (fun () ->
          let b =
            parquet ~rows:2
              ~column:[ (6, I32 5); (7, I32 (-2)); (8, I32 5) ]
              (page 2 (int32s [ 7; -8 ]))
          in
          equal text "parquet (1 column)\n  x int32 ← required int32"
            (format_text (Error.get_ok (P.sniff b))));
      test "an int8 that does not fit names its row" (fun () ->
          (* Rows 0, 2 and 3 hold 1, 2 and 300: the levels are a bit-packed run
             of 1, 0, 1, 1. *)
          let levels = "\x02\x00\x00\x00\x03\x0d" in
          let file =
            parquet ~rows:4
              ~column:[ (3, I32 1); (6, I32 15) ]
              (page 4 (levels ^ int32s [ 1; 2; 300 ]))
          in
          expect (outcome (read_x file))
          @@ __POS_OF__
               {| row group 0: the value 300 of row 3 does not fit int8 |});
      test "sizes at the limits of their types are errors" (fun () ->
          let show what b =
            Printf.printf "%s: %s\n" what (outcome (read_x b))
          in
          let two = int32s [ 7; 8 ] in
          let header changes = parquet ~rows:2 (page ~header:changes 2 two) in
          show "a page of i32_max bytes" (header [ (3, I32 i32_max) ]);
          show "a page of 2^31 bytes" (header [ (3, I32 (i32_max + 1)) ]);
          show "a page size of wire type i64" (header [ (3, I64 8) ]);
          show "levels of twice i32_max bytes"
            (header
               [
                 (1, I32 3);
                 ( 8,
                   Struct
                     [
                       (1, I32 2);
                       (2, I32 0);
                       (3, I32 2);
                       (4, I32 0);
                       (5, I32 i32_max);
                       (6, I32 i32_max);
                     ] );
               ]);
          show "a chunk of max_int bytes"
            (parquet ~rows:2 ~chunk:[ (7, I64 max_int) ] (page 2 two));
          show "a chunk at max_int"
            (parquet ~rows:2
               ~chunk:[ (7, I64 max_int); (9, I64 max_int) ]
               (page 2 two));
          show "2^40 rows of i32_max bytes"
            (parquet ~rows:(1 lsl 40)
               ~column:[ (1, I32 7); (2, I32 i32_max) ]
               (page 2 two));
          expect (output ())
          @@ __POS_OF__
               {|
            a page of i32_max bytes: row group 0: bytes 4-30: the page overruns its column chunk
            a page of 2^31 bytes: row group 0: byte 9: a page header is malformed: an i32 outside its range, 2147483648
            a page size of wire type i64: row group 0: byte 9: a page header is malformed: expected an i32, found wire type 6
            levels of twice i32_max bytes: row group 0: bytes 4-48: the levels overrun the page
            a chunk of max_int bytes: row group 0: bytes 4-4611686018427387902: column "x": its pages overrun the file, whose footer starts at byte 27
            a chunk at max_int: row group 0: byte 4611686018427387903: column "x": its pages overrun the file, whose footer starts at byte 27
            2^40 rows of i32_max bytes: row group 0: bytes 4-26: the chunk's 1099511627776 values do not fit in memory
            |});
    ]

(* Sources *)

let float32s vs =
  let b = Bytes.create (4 * List.length vs) in
  List.iteri
    (fun i v -> Bytes.set_int32_le b (4 * i) (Int32.bits_of_float v))
    vs;
  Bytes.to_string b

(* [stats ?nulls ?nans bytes] is a chunk's [Statistics] of minimum and maximum
   [bytes] ([min ^ max]), split in two halves. *)
let stats ?nulls ?nans bytes =
  let half = String.length bytes / 2 in
  let opt id = Option.map (fun n -> (id, I64 n)) in
  ( 12,
    Struct
      (List.filter_map Fun.id
         [
           opt 3 nulls;
           Some (5, Bin (String.sub bytes half half));
           Some (6, Bin (String.sub bytes 0 half));
           opt 9 nans;
         ]) )

(* A file of two row groups: the first holds 1 and 2 with their statistics, and
   the second has statistics but a page that overruns its chunk, so the query
   reads it only if they do not exclude it. [~column] makes [x] optional and
   changes its type, [values] are the first group's values, [ints] the encoding
   of a pair of statistics. *)
let pruned ?(column = []) ~values ~bytes second =
  let levels = "\x02\x00\x00\x00\x04\x01" in
  let first = (2, [ stats (bytes 1 2) ], page 2 (levels ^ values)) in
  let bad = page ~header:[ (3, I32 1000) ] 2 (levels ^ values) in
  row_groups ~column:([ (3, I32 1) ] @ column) [ first; (2, [ second ], bad) ]

let outcome_of to_string = function
  | Ok vs -> String.concat " " (Array.to_list (Array.map to_string vs))
  | Error e -> error_text e

let filtered show b x ps =
  let f = Error.get_ok (P.sniff b) in
  List.iter
    (fun (name, p) ->
      let q = Query.filter p (Query.of_source (P.source f b)) in
      Printf.printf "%s: %s\n" name (outcome_of show (Query.values x q)))
    ps

let sources =
  group "Sources"
    [
      test "statistics skip the row groups no row of which passes" (fun () ->
          let ints a b = int32s [ a; b ] in
          let b =
            pruned ~values:(ints 1 2) ~bytes:ints (stats ~nulls:0 (ints 10 20))
          in
          let x = Col.int "x" in
          filtered string_of_int b x
            Expr.
              [
                ("x < 10", x < int 10);
                ("x <= 10", x <= int 10);
                ("x > 20", x > int 20);
                ("x >= 20", x >= int 20);
                ("x = 5", x = int 5);
                ("x = 15", x = int 15);
                ("x <> 10", x <> int 10);
                ("x in [3; 25]", is_in [ 3; 25 ] x);
                ("x in []", is_in [] x);
                ("not (x >= 10)", not (x >= int 10));
                ("x < 10 || x > 20", x < int 10 || x > int 20);
                ("x > 1 && x < 10", x > int 1 && x < int 10);
                ("x > 1 && not (x >= 10)", x > int 1 && not (x >= int 10));
                ("x is null", is_null x);
                ("x is not null", not (is_null x));
              ];
          expect (output ())
          @@ __POS_OF__
               {|
            x < 10: 1 2
            x <= 10: row group 1: bytes 33-62: the page overruns its column chunk
            x > 20:
            x >= 20: row group 1: bytes 33-62: the page overruns its column chunk
            x = 5:
            x = 15: row group 1: bytes 33-62: the page overruns its column chunk
            x <> 10: row group 1: bytes 33-62: the page overruns its column chunk
            x in [3; 25]:
            x in []:
            not (x >= 10): row group 1: bytes 33-62: the page overruns its column chunk
            x < 10 || x > 20: 1 2
            x > 1 && x < 10: 2
            x > 1 && not (x >= 10): row group 1: bytes 33-62: the page overruns its column chunk
            x is null:
            x is not null: row group 1: bytes 33-62: the page overruns its column chunk
            |});
      test "a row group of nulls passes no comparison" (fun () ->
          let ints a b = int32s [ a; b ] in
          let nulls = (12, Struct [ (3, I64 2) ]) in
          let b = pruned ~values:(ints 1 2) ~bytes:ints nulls in
          let x = Col.int "x" in
          filtered string_of_int b x
            Expr.
              [
                ("x > 0", x > int 0);
                ("x <> 0", x <> int 0);
                ("x is not null", not (is_null x));
                ("x is null", is_null x);
              ];
          expect (output ())
          @@ __POS_OF__
               {|
            x > 0: 1 2
            x <> 0: 1 2
            x is not null: 1 2
            x is null: row group 1: bytes 33-62: the page overruns its column chunk
            |});
      test "float statistics skip a row group only without NaN" (fun () ->
          let floats a b = float32s [ float_of_int a; float_of_int b ] in
          let x = Col.float "x" in
          List.iter
            (fun (name, nans) ->
              let second = stats ~nulls:0 ?nans (floats 10 20) in
              let b =
                pruned
                  ~column:[ (1, I32 4) ]
                  ~values:(floats 1 2) ~bytes:floats second
              in
              print_endline name;
              filtered string_of_float b x
                Expr.[ ("x < 5.", x < float 5.); ("x > 15.", x > float 15.) ])
            [ ("no NaN count", None); ("no NaN", Some 0); ("a NaN", Some 1) ];
          expect (output ())
          @@ __POS_OF__
               {|
            no NaN count
            x < 5.: row group 1: bytes 33-62: the page overruns its column chunk
            x > 15.: row group 1: bytes 33-62: the page overruns its column chunk
            no NaN
            x < 5.: 1. 2.
            x > 15.: row group 1: bytes 33-62: the page overruns its column chunk
            a NaN
            x < 5.: row group 1: bytes 33-62: the page overruns its column chunk
            x > 15.: row group 1: bytes 33-62: the page overruns its column chunk
            |});
      test "a read decodes only the request's columns" (fun () ->
          let name = "nation.dict-malformed.parquet" in
          let f = sniff name in
          List.iter
            (fun c ->
              Printf.printf "%s: %s\n" c
                (outcome (Result.map Column.length (read f name c))))
            [ "nation_key"; "name" ];
          expect (output ())
          @@ __POS_OF__
               {|
            nation_key: ok
            name: row group 0: bytes 421-450: the page overruns its column chunk
            |});
      test "a read of no column has the rows of the file" (fun () ->
          let name = "names.parquet" in
          let q =
            Query.select []
              (Query.of_source (P.source (sniff name) (buffer name)))
          in
          equal int 2 (Talon_next.rows (Error.get_ok (Query.run q))));
      test "file names its source and counts its rows" (fun () ->
          let plan s = Format.asprintf "%a" Query.pp (Query.of_source s) in
          print_endline (plan (Error.get_ok (P.file (path "names.parquet"))));
          print_endline
            (plan (P.source (sniff "names.parquet") (buffer "names.parquet")));
          expect (output ())
          @@ __POS_OF__
               {|
            query → "a b" int64, "\"q\"" int64, é int64, "" int64
            parquet "fixtures/names.parquet" (4 columns, 2 rows)
            query → "a b" int64, "\"q\"" int64, é int64, "" int64
            parquet (4 columns)
            |});
      test "file refuses what sniff refuses, naming the file" (fun () ->
          List.iter
            (fun name -> print_endline (outcome (P.file (path name))))
            [ "brotli.parquet"; "map_no_value.parquet"; "missing.parquet" ];
          expect (output ())
          @@ __POS_OF__
               {|
            fixtures/brotli.parquet: row group 0: column "i32" is compressed with Brotli, which talon does not read
            fixtures/map_no_value.parquet: column "my_map" is a map: talon reads flat Parquet files only
            fixtures/missing.parquet: No such file or directory
            |});
    ]

(* Decoding errors *)

let decoding =
  test "Columns that fail to read" (fun () ->
      let show name ?(f = sniff name) column =
        Printf.printf "%s %s: %s\n" name column (outcome (read f name column))
      in
      show "datapage_v1-corrupt-checksum.parquet" "a";
      show "rle-dict-uncompressed-corrupt-checksum.parquet" "binary_field";
      show "bad_data/ARROW-GH-47662.parquet" "flba_field";
      show "nation.dict-malformed.parquet" "nation_key";
      show "nation.dict-malformed.parquet" "name";
      show "int96_from_spark.parquet" "a";
      show "int96_from_spark.parquet" "a"
        ~f:
          (P.with_type "a"
             (any (Type.datetime Us))
             (sniff "int96_from_spark.parquet"));
      show "int96.parquet" "ts_ns"
        ~f:
          (P.with_type "ts_ns" (any (Type.datetime Us)) (sniff "int96.parquet"));
      show "bad_encoding.parquet" "i32";
      show "types.parquet" "a b" ~f:(sniff "names.parquet");
      expect (output ())
      @@ __POS_OF__
           {|
        datapage_v1-corrupt-checksum.parquet a: row group 0: bytes 4-10271: the page fails its CRC-32 checksum
        rle-dict-uncompressed-corrupt-checksum.parquet binary_field: row group 0: bytes 57-115: the page fails its CRC-32 checksum
        bad_data/ARROW-GH-47662.parquet flba_field: row group 0: bytes 4-393: the PLAIN values take 400 bytes of the page's 364
        nation.dict-malformed.parquet nation_key: ok
        nation.dict-malformed.parquet name: row group 0: bytes 421-450: the page overruns its column chunk
        int96_from_spark.parquet a: row group 0: an int96 timestamp is outside the range of datetime[ns]. Read the column as a datetime of a coarser unit (with_type).
        int96_from_spark.parquet a: row group 0: an int96 timestamp is outside the range of datetime[us]. Read the column as a datetime of a coarser unit (with_type).
        int96.parquet ts_ns: row group 0: an int96 timestamp is not a whole number of microseconds. Read the column as a datetime of a finer unit (with_type).
        bad_encoding.parquet i32: row group 0: bytes 4-360: the values are in encoding 15, which Parquet does not define
        types.parquet a b: the file has no column "a b"
        |})

let () =
  exit
    (run "talon.next.parquet"
       [ agreement; sniffing; formats; synthesized; sources; decoding ])
