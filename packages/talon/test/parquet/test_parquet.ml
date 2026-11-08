(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module P = Talon_parquet

(* Fixtures *)

(* [path name] is the fixture [name], written with [/], which every system
   opens, so that messages name it alike. *)
let path name = "support/" ^ name

let buffer name =
  match Rig_disk.of_file (path name) with
  | Ok b -> b
  | Error why -> failwith why

let file_bytes name = In_channel.with_open_bin (path name) In_channel.input_all
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
  |> Result.map (fun t -> Talon.column t column)

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
  else if ty = "bool" then "bit"
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
        let valid = Nx.to_array v in
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
let values = expected "golden/values.txt"
let overrides = expected "golden/overrides.txt"
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
  | "datetime[us]" -> P.with_type column (Type.Any (Type.wall Us)) f
  | "int64" -> P.with_type column (Type.Any Type.int64) f
  | ty -> failf "no override for %s" ty

let agreement =
  group "Columns read as pyarrow reads them"
    [
      cases "defaults" ~id:Fun.id (files values) (agree values);
      cases "with_type" ~id:Fun.id (files overrides)
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

(* Parquet stores a timestamp's clock in one bit, isAdjustedToUTC: pyarrow's
   zoned timestamps read as instants and its plain ones as wall clocks, and an
   Arrow zone name is not kept. *)
let clocks =
  group "clocks"
    [
      test "each clock binds its handle and round-trips through a zone"
        (fun () ->
          let name = "types.parquet" in
          let q = Query.of_source (P.source (declared name) (buffer name)) in
          let out e =
            Talon.column
              (Error.get_ok (Query.run (Query.select Expr.[ "out" := e ] q)))
              "out"
          in
          let u = Col.instant "ts_us_utc" and w = Col.wall "ts_ms" in
          let localize a =
            Expr.Temporal.localize Tz.utc ~ambiguous:`Fail ~gap:`Fail a
          in
          let opts k c = Column.options k c in
          let t_us = option int64 in
          let ticks a = Array.map (Option.map Time.to_us) a in
          equal (array t_us)
            (ticks (opts Kind.instant (out u)))
            (ticks
               (opts Kind.instant
                  (out (localize (Expr.Temporal.wall Tz.utc u)))));
          equal (array t_us)
            (ticks (opts Kind.wall (out w)))
            (ticks
               (opts Kind.wall (out (Expr.Temporal.wall Tz.utc (localize w)))));
          let problem e =
            match Query.select Expr.[ "out" := e ] q with
            | exception Invalid_argument _ -> ()
            | _ -> Windtrap.fail "expected a problem at binding"
          in
          problem (Col.wall "ts_us_utc");
          problem (Col.instant "ts_ms");
          problem (Expr.Temporal.field `Hour (Col.instant "ts_us_paris")));
    ]

let sniffing =
  group "sniff"
    [
      cases "reads a file and its bytes alike" ~id:Fun.id
        [
          "alltypes_plain.parquet";
          "types.parquet";
          "names.parquet";
          "empty.parquet";
        ] (fun name ->
          let bytes =
            Error.get_ok (P.sniff (Rig.Buffer.of_string (file_bytes name)))
          in
          equal text (format_text (sniff name)) (format_text bytes));
      test "refuses what talon cannot read in full" (fun () ->
          let refused =
            [
              "hadoop_lz4_compressed.parquet";
              "non_hadoop_lz4_compressed.parquet";
              "brotli.parquet";
              "large_string_map.brotli.parquet";
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
            large_string_map.brotli.parquet: row group 0: column "arr.key_value.key" is compressed with Brotli, which talon does not read
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
              Printf.printf "%s: %s\n" what
                (outcome (P.sniff (Rig.Buffer.of_string s))))
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
              retype plain "timestamp_col" (any (Type.instant Us));
              retype types "ts_us_utc" (any (Type.wall Us));
              retype types "ts_ms" (any (Type.instant Ms));
            ];
          expect (output ())
          @@ __POS_OF__
               {|
            Talon_parquet.with_type: no column "carier" in the format.
            Talon_parquet.with_type: column "i32" (optional int32) reads as int32, not int64. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "f64" (optional double) reads as float64, not float32. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "string" (optional binary (STRING)) reads as string, binary or a categorical, not int64. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "d9" (optional fixed_len_byte_array(4) (DECIMAL(9,2))) reads as float64 or int64, not binary. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "timestamp_col" (optional int96) reads as wall-clock datetime of any unit, not date. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "timestamp_col" (optional int96) reads as wall-clock datetime of any unit, not datetime[us, UTC]. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "ts_us_utc" (optional int64 (TIMESTAMP(MICROS,true))) reads as datetime[us, UTC], not datetime[us]. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "ts_ms" (optional int64 (TIMESTAMP(MILLIS,false))) reads as datetime[ms], not datetime[ms, UTC]. Cast it once read (Expr.cast).
            |});
      test "with_type of a column's own type is the identity" (fun () ->
          let f = sniff "types.parquet" in
          let same column t =
            equal text (format_text f) (format_text (P.with_type column t f))
          in
          same "i8" (any Type.int8);
          same "u64" (any Type.uint64);
          same "clock_ms" (any (Type.clock Ms));
          same "ts_us_utc" (any (Type.instant Us));
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
              retype plain "timestamp_col" (any (Type.wall S));
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
            Talon_parquet.with_type: column "d38_0" (optional fixed_len_byte_array(16) (DECIMAL(38,0))) reads as float64, not int64. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "d9_0" (optional int32 (DECIMAL(9,0))) reads as float64 or int64, not int32. Cast it once read (Expr.cast).
            |});
      test "source and file refuse a decimal left undeclared" (fun () ->
          let name = "decimals_bytes.parquet" in
          let f = P.with_type "d9_0" (any Type.float64) (sniff name) in
          print_endline (refusal (fun () -> P.source f (buffer name)));
          print_endline (refusal (fun () -> P.file (path name)));
          expect (output ())
          @@ __POS_OF__
               {|
            Talon_parquet.source: column "d9_2" (optional fixed_len_byte_array(4) (DECIMAL(9,2))) is a decimal, which reads as float64, or as its unscaled int64 integers up to 18 digits. Declare one with with_type.
            Talon_parquet.file: column "d9_0" (optional fixed_len_byte_array(4) (DECIMAL(9,0))) is a decimal, which reads as float64, or as its unscaled int64 integers up to 18 digits. Declare one with with_type.
            |});
      cases "pp_format" ~id:fst
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
  Rig.Buffer.of_string ("PAR1" ^ pages ^ meta ^ Bytes.to_string len ^ "PAR1")

let parquet ?column ?(chunk = []) ~rows pages =
  row_groups ?column [ (rows, chunk, pages) ]

let read_x b =
  match P.sniff b with Error e -> Error e | Ok f -> read_from f b "x"

let i32_max = 0x7FFF_FFFF

(* Nested files: leaves under a schema's [elements], and pages of levels
   written by hand. *)

let required = 0
let optional = 1
let repeated = 2

let schema_group ?converted repetition name children =
  Struct
    ((3, I32 repetition) :: (4, Bin name) :: (5, I32 children)
    :: Option.fold ~none:[] ~some:(fun c -> [ (6, I32 c) ]) converted)

let int32_field repetition name =
  Struct [ (1, I32 1); (3, I32 repetition); (4, Bin name) ]

let list_group = schema_group ~converted:3

(* [nested_parquet ?rows ~elements chunks] is a file of one row group of [rows]
   rows, with the schema [elements] and a chunk [(path, values, pages)] per
   leaf. *)
let nested_parquet ?(rows = 2) ?(stats = []) ~elements chunks =
  let chunk (pos, acc) (path, values, pages) =
    let size = I64 (String.length pages) in
    let meta =
      Struct
        [
          (1, I32 1);
          (3, List (List.map (fun s -> Bin s) path));
          (4, I32 0);
          (5, I64 values);
          (6, size);
          (7, size);
          (9, I64 pos);
        ]
    in
    let meta = match meta with Struct f -> Struct (f @ stats) | m -> m in
    (pos + String.length pages, Struct [ (3, meta) ] :: acc)
  in
  let _, cs = List.fold_left chunk (4, []) chunks in
  let root = Struct [ (4, Bin "schema"); (5, I32 1) ] in
  let footer =
    thrift
      (Struct
         [
           (2, List (root :: elements));
           (3, I64 rows);
           (4, List [ Struct [ (1, List (List.rev cs)); (3, I64 rows) ] ]);
         ])
  in
  let pages = String.concat "" (List.map (fun (_, _, p) -> p) chunks) in
  Rig.Buffer.of_string
    ("PAR1" ^ pages ^ footer ^ int32s [ String.length footer ] ^ "PAR1")

(* [runs rs] is the RLE runs [rs], each [(count, value)] of a byte. *)
let runs rs =
  String.concat ""
    (List.map
       (fun (n, v) -> Printf.sprintf "%c%c" (Char.chr (n lsl 1)) (Char.chr v))
       rs)

let prefixed s = int32s [ String.length s ] ^ s

let level_page ?(data_page = [ (4, I32 3) ]) ~rep ~def values data =
  page ~data_page values (prefixed (runs rep) ^ prefixed (runs def) ^ data)

let nested_text b =
  match P.sniff b with
  | Error e -> error_text e
  | Ok f -> (
      match Query.run (Query.of_source (P.source f b)) with
      | Ok t -> Format.asprintf "%a" Talon.pp t
      | Error e -> error_text e)

let x_elements =
  [
    list_group optional "x" 1;
    schema_group repeated "list" 1;
    int32_field optional "element";
  ]

let x_path = [ "x"; "list"; "element" ]

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
      test "a text dictionary index past the dictionary is an error"
        (fun () ->
          (* A dictionary page of the texts "a" and "b", then a data page of
             two indices at bit width 3: an RLE run of 2 of index 5. *)
          let dict =
            thrift
              (Struct
                 [
                   (1, I32 2);
                   (2, I32 10);
                   (3, I32 10);
                   (7, Struct [ (1, I32 2); (2, I32 0) ]);
                 ])
            ^ "\x01\x00\x00\x00a\x01\x00\x00\x00b"
          in
          let data = page ~data_page:[ (2, I32 8) ] 2 "\x03\x04\x05" in
          let file =
            parquet ~rows:2
              ~column:[ (1, I32 6); (6, I32 0) ]
              ~chunk:[ (9, I64 (4 + String.length dict)); (11, I64 4) ]
              (dict ^ data)
          in
          expect (outcome (read_x file))
          @@ __POS_OF__
               {| row group 0: bytes 27-44: the dictionary indices: a value is out of its bounds |});
      test "an RLE boolean run of value 2 is an error" (fun () ->
          (* Four booleans: a 4-byte length, then an RLE run of 4 of value 2. *)
          let file =
            parquet ~rows:4
              ~column:[ (1, I32 0) ]
              (page ~data_page:[ (2, I32 3) ] 4 "\x02\x00\x00\x00\x08\x02")
          in
          expect (outcome (read_x file))
          @@ __POS_OF__
               {| row group 0: bytes 4-24: the RLE booleans: a value is out of its bounds |});
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
      test "compressed pages decode, and one that does not says where"
        (fun () ->
          (* Eight int32s, compressed with the codec [codec] (Parquet's
             CompressionCodec) into [data], in a page whose header states [size]
             bytes once decompressed. *)
          let raw = int32s [ 1; 2; 3; 4; 5; 6; 7; 8 ] in
          let show ?(size = String.length raw) what codec data =
            let b =
              parquet ~rows:8
                ~chunk:[ (4, I32 codec); (6, I64 1000) ]
                (page ~header:[ (2, I32 size) ] 8 data)
            in
            Printf.printf "%s (%d bytes): %s\n" what (String.length data)
              (match read_x b with
              | Ok c -> String.concat " " (Array.to_list (render "int32" c))
              | Error e -> error_text e)
          in
          let snappy = Lz.Snappy.encode_string raw
          and zstd = Lz.Zstd.encode_string raw
          and gzip = Lz.Gzip.encode_string raw
          and lz4 = Lz.Lz4.Block.encode_string raw in
          let cut s = String.sub s 0 (String.length s - 1) in
          let flip_last s =
            let b = Bytes.of_string s in
            let i = Bytes.length b - 1 in
            Bytes.set_uint8 b i (Bytes.get_uint8 b i lxor 1);
            Bytes.to_string b
          in
          show "snappy" 1 snappy;
          show "gzip of two members" 2
            (Lz.Gzip.encode_string (String.sub raw 0 12)
            ^ Lz.Gzip.encode_string (String.sub raw 12 20));
          show "zstd" 6 zstd;
          show "lz4_raw" 7 lz4;
          show "zstd with a wrong checksum" 6 (flip_last zstd);
          show "gzip cut short" 2 (cut gzip);
          show "lz4_raw cut short" 7 (cut lz4);
          show "snappy of fewer bytes than its header" 1
            (Lz.Snappy.encode_string (String.sub raw 0 16));
          show "snappy of more bytes than its header" ~size:16 1 snappy;
          show "snappy followed by a byte" 1 (snappy ^ "\x00");
          show "lz4_raw followed by a byte" 7 (lz4 ^ "\x00");
          expect (output ())
          @@ __POS_OF__
               {|
            snappy (34 bytes): 1 2 3 4 5 6 7 8
            gzip of two members (69 bytes): 1 2 3 4 5 6 7 8
            zstd (30 bytes): 1 2 3 4 5 6 7 8
            lz4_raw (34 bytes): 1 2 3 4 5 6 7 8
            zstd with a wrong checksum (30 bytes): row group 0: bytes 4-48: the page does not decompress: malformed at byte 26: Content_Checksum mismatch
            gzip cut short (41 bytes): row group 0: bytes 4-59: the page does not decompress: truncated at byte 41
            lz4_raw cut short (33 bytes): row group 0: bytes 4-51: the page does not decompress: truncated at byte 33
            snappy of fewer bytes than its header (18 bytes): row group 0: bytes 4-36: the page decompresses to 16 bytes, not its header's 32
            snappy of more bytes than its header (34 bytes): row group 0: bytes 4-52: the page does not decompress: output too long at byte 0
            snappy followed by a byte (35 bytes): row group 0: bytes 4-53: the page does not decompress: malformed at byte 34: output longer than the preamble
            lz4_raw followed by a byte (35 bytes): row group 0: bytes 4-53: the page does not decompress: truncated at byte 35
            |});
    ]

(* Nested columns *)

let scalars (Nx.P x) =
  let map f a = Array.map f (Nx.to_array a) in
  match Format.asprintf "%a" Nx.pp_dtype (Nx.dtype x) with
  | "bit" ->
      map (fun b -> if b = 1 then "true" else "false") (Nx.cast Nx.uint8 x)
  | "float64" -> map (Printf.sprintf "0x%016Lx") (Nx.bitcast Nx.int64 x)
  | _ -> map Int64.to_string (Nx.cast Nx.int64 x)

let is_bytes c =
  match Column.type_ c with Type.Any (String | Binary) -> true | _ -> false

(* [nested_cells c] is the rows of [c], as gen/nested.py renders pyarrow's. *)
let rec nested_cells c =
  let masked validity cs =
    match validity with
    | None -> cs
    | Some v ->
        Array.map2 (fun ok s -> if ok then s else "null") (Nx.to_array v) cs
  in
  match Column.layout c with
  | Fixed { validity; values } -> masked validity (scalars values)
  | Varsize { validity; offsets; child } when is_bytes c ->
      let o = Nx.to_array offsets in
      let d = Nx.to_array (Column.to_tensor Nx.uint8 child) in
      let cell i =
        let first = Int64.to_int o.(i) and last = Int64.to_int o.(i + 1) in
        quote (String.init (last - first) (fun k -> Char.chr d.(first + k)))
      in
      masked validity (Array.init (Array.length o - 1) cell)
  | Varsize { validity; offsets; child } ->
      let o = Array.map Int64.to_int (Nx.to_array offsets) in
      let items = nested_cells child in
      let row i =
        let its = Array.sub items o.(i) (o.(i + 1) - o.(i)) in
        "[" ^ String.concat ", " (Array.to_list its) ^ "]"
      in
      masked validity (Array.init (Array.length o - 1) row)
  | Children { validity; length; fields } ->
      let fs = List.map (fun (n, c) -> (n, nested_cells c)) fields in
      let row i =
        let field (n, cs) = n ^ ": " ^ cs.(i) in
        "{" ^ String.concat ", " (List.map field fs) ^ "}"
      in
      masked validity (Array.init length row)

let read_nested file c =
  match read (sniff file) file c with
  | Ok c -> c
  | Error e -> failf "%s %s: %s" file c (error_text e)

let split l = Scanf.sscanf l "%s %S %[^\n]" (fun f c rest -> (f, c, rest))

(* The law: rows of nullable lists of nullable lists of nullable integers,
   flattened by hand to Dremel's stripes, read back as themselves. Definition
   levels: 0 null row, 1 empty, 2 null inner list, 3 empty, 4 null, 5 a value. *)

let stripe_elements =
  [
    list_group optional "x" 1;
    schema_group repeated "list" 1;
    list_group optional "element" 1;
    schema_group repeated "list" 1;
    int32_field optional "element";
  ]

let shred rows =
  let entries = ref [] and values = ref [] in
  let entry r d = entries := (r, d) :: !entries in
  let inner r = function
    | None -> entry r 2
    | Some [||] -> entry r 3
    | Some xs ->
        Array.iteri
          (fun i x ->
            let r = if i = 0 then r else 2 in
            match x with
            | None -> entry r 4
            | Some v ->
                entry r 5;
                values := v :: !values)
          xs
  in
  let row = function
    | None -> entry 0 0
    | Some [||] -> entry 0 1
    | Some ls -> Array.iteri (fun i l -> inner (min i 1) l) ls
  in
  Array.iter row rows;
  (List.rev !entries, List.rev !values)

let rendered rows =
  let bracket f xs =
    "[" ^ String.concat ", " (Array.to_list (Array.map f xs)) ^ "]"
  in
  let opt f = function None -> "null" | Some x -> f x in
  Array.map (opt (bracket (opt (bracket (opt string_of_int))))) rows

let stripes =
  let open Gen in
  let small g = array ~size:(int_range 0 3) g in
  array ~size:(int_range 1 12)
    (option (small (option (small (option (int_range (-9) 9))))))

(* [striped rows k] is the file of [rows], its pages cut at the entry [k]. *)
let striped rows k =
  let entries, values = shred rows in
  let n = List.length entries in
  let k = k mod (n + 1) in
  let sub l a b = List.filteri (fun i _ -> a <= i && i < b) l in
  let present a b =
    List.length (List.filter (fun (_, d) -> d = 5) (sub entries a b))
  in
  let page a b =
    if a = b then ""
    else
      let es = sub entries a b in
      level_page
        ~rep:(List.map (fun (r, _) -> (1, r)) es)
        ~def:(List.map (fun (_, d) -> (1, d)) es)
        (b - a)
        (int32s (sub values (present 0 a) (present 0 b)))
  in
  nested_parquet ~rows:(Array.length rows) ~elements:stripe_elements
    [ ([ "x"; "list"; "element"; "list"; "element" ], n, page 0 k ^ page k n) ]

let nested_files =
  group "Nested files"
    [
      prop "a table read from its stripes equals the table they flatten"
        Gen.(pair stripes nat)
        (fun (rows, k) ->
          let has f = Array.exists f rows in
          let inner f = has (function Some ls -> Array.exists f ls | _ -> false) in
          cover "a null row" (has Option.is_none);
          cover "an empty row" (has (( = ) (Some [||])));
          cover "a null inner list" (inner Option.is_none);
          cover "an empty inner list" (inner (( = ) (Some [||])));
          cover "a null integer"
            (inner (function Some xs -> Array.mem None xs | _ -> false));
          let b = striped rows k in
          match read_from (Error.get_ok (P.sniff b)) b "x" with
          | Error e -> failf "%s" (error_text e)
          | Ok c -> equal (array string) (rendered rows) (nested_cells c));
      cases "columns read as pyarrow reads them"
        ~id:(fun l -> let f, c, _ = split l in f ^ " " ^ c)
        (In_channel.with_open_text "golden/nested.txt" In_channel.input_lines)
        (fun l ->
          let f, c, want = split l in
          equal string want
            (String.concat " " (Array.to_list (nested_cells (read_nested f c)))));
      test "with_type retypes a nested column's leaves, not its shape" (fun () ->
          let name = "nested_v1_plain.parquet" in
          let f =
            P.with_type "texts" (Type.Any (Type.list Type.binary)) (sniff name)
          in
          (match read f name "texts" with
          | Ok c ->
              let (Type.Any t) = Column.type_ c in
              equal string "list[binary]" (Format.asprintf "%a" Type.pp t)
          | Error e -> failf "%s" (error_text e));
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> P.with_type "lists" (Type.Any Type.int64) (sniff name)));
      test "a conjunct on a nested column prunes no row group" (fun () ->
          (* The leaf's statistics say no null, and the first row is null. *)
          let b =
            nested_parquet
              ~stats:[ (12, Struct [ (3, I64 0) ]) ]
              ~elements:x_elements
              [ ( x_path,
                  2,
                  level_page ~rep:[ (2, 0) ] ~def:[ (1, 0); (1, 3) ] 2
                    (int32s [ 7 ]) ) ]
          in
          Query.of_source (P.source (Error.get_ok (P.sniff b)) b)
          |> Query.filter Expr.(is_null (Col.v (Kind.list Kind.int) "x"))
          |> Query.run |> Error.get_ok |> Talon.rows |> equal int 1);
      test "a filter on a flat column takes the rows of a nested one" (fun () ->
          let name = "nested_v1_dict.parquet" in
          let t =
            Query.of_source (P.source (sniff name) (buffer name))
            |> Query.filter Expr.(Col.string "tag" = string "t1")
            |> Query.select Expr.[ keep (Sel.names [ "people" ]) ]
            |> Query.run |> Error.get_ok
          in
          let all = nested_cells (read_nested name "people") in
          equal (list string)
            (List.filteri (fun i _ -> i mod 3 = 1) (Array.to_list all))
            (Array.to_list (nested_cells (Talon.column t "people"))));
      test "levels written by hand" (fun () ->
          let leaf ?rows ?claim ~rep ~def values data =
            let claim = Option.value claim ~default:values in
            nested_parquet ?rows ~elements:x_elements
              [ (x_path, claim, level_page ~rep ~def values data) ]
          in
          let show name b = Printf.printf "%s: %s\n" name (nested_text b) in
          show "a level past the maximum"
            (leaf ~rep:[ (2, 2) ] ~def:[ (2, 3) ] 2 (int32s [ 1; 2 ]));
          show "a first entry that does not start a row"
            (leaf ~rep:[ (2, 1) ] ~def:[ (2, 3) ] 2 (int32s [ 1; 2 ]));
          show "an entry repeating an undefined list"
            (leaf ~rep:[ (1, 0); (1, 1) ] ~def:[ (1, 3); (1, 1) ] 2
               (int32s [ 1 ]));
          show "pages that do not add up to the chunk"
            (leaf ~claim:5 ~rep:[ (2, 0) ] ~def:[ (2, 3) ] 2 (int32s [ 1; 2 ]));
          show "fewer rows than the row group"
            (leaf ~rows:3 ~rep:[ (1, 0); (2, 1) ] ~def:[ (3, 3) ] 3
               (int32s [ 1; 2; 3 ]));
          show "two fields of a record that disagree"
            (let field name rep =
               ( [ "x"; name ],
                 2,
                 level_page ~rep ~def:[ (2, 1) ] 2 (int32s [ 1; 2 ]) )
             in
             let f = int32_field repeated in
             nested_parquet
               ~elements:[ schema_group required "x" 2; f "a"; f "b" ]
               [ field "a" [ (2, 0) ]; field "b" [ (1, 0); (1, 1) ] ]);
          expect (output ()) @@ __POS_OF__ {|
            a level past the maximum: row group 0: bytes 4-40: column "x.list.element": the repetition levels: a value is out of its bounds
            a first entry that does not start a row: row group 0: column "x.list.element": the levels: the first entry does not start a row
            an entry repeating an undefined list: row group 0: column "x": the levels: an entry repeats a field that is not defined
            pages that do not add up to the chunk: row group 0: bytes 4-40: column "x.list.element": the pages hold 2 values, and the column chunk 5
            fewer rows than the row group: row group 0: column "x" holds 1 rows in a row group of 3
            two fields of a record that disagree: row group 0: column "x": field "b" holds 1 values for the record's 2
            |});
      test "the shapes the list rules read" (fun () ->
          let shape want elements paths def =
            let chunk path =
              let p = level_page ~rep:[ (2, 0) ] ~def:[ (2, def) ] 2 "" in
              (path, 2, p)
            in
            let b = nested_parquet ~elements (List.map chunk paths) in
            let text = format_text (Error.get_ok (P.sniff b)) in
            let line = List.nth (String.split_on_char '\n' text) 1 in
            equal string want (List.hd (String.split_on_char '\xe2' line))
          in
          let l = list_group optional "x" 1 in
          let a = int32_field required "a" and b = int32_field required "b" in
          shape "  x list[record[a int32, b int32]] "
            [ l; schema_group repeated "pair" 2; a; b ]
            [ [ "x"; "pair"; "a" ]; [ "x"; "pair"; "b" ] ]
            2;
          shape "  x list[record[a int32]] "
            [ l; schema_group repeated "array" 1; a ]
            [ [ "x"; "array"; "a" ] ] 2;
          shape "  x list[record[a int32, b int32]] "
            [ schema_group repeated "x" 2; a; b ]
            [ [ "x"; "a" ]; [ "x"; "b" ] ] 1);
      test "a schema talon refuses names the column" (fun () ->
          let refused name elements =
            Printf.printf "%s: %s\n" name
              (nested_text (nested_parquet ~elements [ (x_path, 2, "") ]))
          in
          let a = int32_field repeated "a" and r = int32_field required "a" in
          refused "a LIST of two fields" [ list_group optional "x" 2; a; a ];
          refused "a repeated LIST outside a list"
            [ list_group repeated "x" 1; a ];
          refused "a MAP whose entries are not a group"
            [ schema_group ~converted:1 optional "x" 1; a ];
          refused "a group of no fields" [ schema_group optional "x" 0 ];
          refused "two fields of one name"
            [ schema_group optional "x" 2; r; r ];
          expect (output ()) @@ __POS_OF__ {|
            a LIST of two fields: column "x" is a list of 2 fields, not of one repeated field
            a repeated LIST outside a list: column "x" is a repeated list or map, which Parquet allows only as the element of another list
            a MAP whose entries are not a group: column "x" is a map whose entries are not a group
            a group of no fields: the schema is malformed: column "x" is a group of no fields
            two fields of one name: column "x" has two fields named "a"
            |});
      test "a level past 255 reads" (fun () ->
          (* Definition levels 300 and 0, as RLE runs of 2-byte values. *)
          let depth = 300 in
          let elements =
            List.init depth (fun i ->
                schema_group optional (if i = 0 then "x" else "g") 1)
            @ [ int32_field required "v" ]
          in
          let path = "x" :: List.init (depth - 1) (fun _ -> "g") @ [ "v" ] in
          let def = "\x02\x2c\x01\x02\x00\x00" in
          let p = page ~data_page:[] 2 (prefixed def ^ int32s [ 42 ]) in
          let b = nested_parquet ~elements [ (path, 2, p) ] in
          match read_from (Error.get_ok (P.sniff b)) b "x" with
          | Error e -> failf "%s" (error_text e)
          | Ok c -> equal int 1 (Column.null_count c));
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

type bytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(* The spread of [talon_parquet_stubs.c], which reading runs into a fresh
   buffer: [spread valid src bits rows dst] puts the values of [src], each of
   [bits] bits, at the rows the bit array [valid] marks in [dst], zero at the
   others, and is the number of values it took. *)
external spread : bytes -> bytes -> int -> int -> bytes -> int
  = "talon_parquet_spread_fixed"
[@@noalloc]

let filled n v =
  let a = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n in
  Bigarray.Array1.fill a v;
  a

(* A fresh buffer holds any bits: here, ones. 731 rows end at bit 3 of byte
   91. *)
let spread_tail =
  cases "spread writes the bits past a bit destination's last row as 0"
    ~id:(fun (name, _, _) -> name)
    [ ("every row valid", 0xff, 731); ("every other row valid", 0x55, 366) ]
    (fun (_, valid, taken) ->
      let rows = 731 in
      let bytes = (rows + 7) / 8 in
      let dst = filled bytes 0xff in
      equal int taken
        (spread (filled bytes valid) (filled bytes 0xff) 1 rows dst);
      equal int 0 (dst.{rows / 8} lsr (rows mod 8)))

(* The RLE and bit-packing hybrid of [talon_parquet_stubs.c]: [hybrid src pos
   len width n dst at bound] decodes [n] values of [width] bits from the [len]
   bytes at [pos] into [dst] from index [at], each below [bound], and is the
   number of bytes it consumed, or a negative error: [-2] for a value not below
   [bound]. *)
external hybrid_int64 :
  bytes ->
  int ->
  int ->
  int ->
  int ->
  (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  int ->
  int ->
  int = "talon_parquet_hybrid_byte" "talon_parquet_hybrid"
[@@noalloc]

external hybrid_bytes :
  bytes -> int -> int -> int -> int -> bytes -> int -> int -> int
  = "talon_parquet_hybrid_byte" "talon_parquet_hybrid"
[@@noalloc]

(* A run of the hybrid, as the Parquet format specifies it: [Repeated (k, v)]
   is [v] [k] times, [Packed vs] the values [vs], bit-packed in groups of
   eight. *)
type run = Repeated of int * int | Packed of int list

(* [encode w runs] is the bytes of [runs] of values of [w] bits: a run's header
   is a ULEB128 varint, [k lsl 1] before a repeated value in its [⌈w / 8⌉]
   little-endian bytes, [g lsl 1 lor 1] before [g] groups of eight values
   packed from each byte's least significant bit, the last group padded with
   zeros. *)
let encode w runs =
  let b = Buffer.create 64 in
  let rec varint v =
    if v < 0x80 then Buffer.add_char b (Char.chr v)
    else (
      Buffer.add_char b (Char.chr (v land 0x7f lor 0x80));
      varint (v lsr 7))
  in
  let run = function
    | Repeated (k, v) ->
        varint (k lsl 1);
        for i = 0 to ((w + 7) / 8) - 1 do
          Buffer.add_char b (Char.chr ((v lsr (8 * i)) land 0xff))
        done
    | Packed vs ->
        let groups = (List.length vs + 7) / 8 in
        varint ((groups lsl 1) lor 1);
        let bits = Bytes.make (groups * w) '\000' in
        List.iteri
          (fun i v ->
            for j = 0 to w - 1 do
              if (v lsr j) land 1 = 1 then begin
                let at = (i * w) + j in
                let c = Char.code (Bytes.get bits (at / 8)) in
                Bytes.set bits (at / 8) (Char.chr (c lor (1 lsl (at mod 8))))
              end
            done)
          vs;
        Buffer.add_bytes b bits
  in
  List.iter run runs;
  Buffer.contents b

let values_of = function
  | Repeated (k, v) -> List.init k (Fun.const v)
  | Packed vs -> vs

(* Runs of values of [w] bits; only the last packed run may end inside a
   group. *)
let hybrid_case =
  let open Gen in
  let* w = int_range 0 32 in
  let value = int_range 0 ((1 lsl w) - 1) in
  let run =
    one_of
      [
        map (fun (k, v) -> Repeated (k, v)) (pair (int_range 0 20) value);
        map
          (fun vs -> Packed vs)
          (list ~size:(map (( * ) 8) (int_range 0 5)) value);
      ]
  in
  let* runs = list ~size:(int_range 0 4) run in
  let* last = list ~size:(int_range 0 17) value in
  let* pos = int_range 0 7 in
  let+ slack = int_range 0 9 in
  (w, runs @ [ Packed last ], pos, slack)

(* The runs that decoding [n] values reads: it stops at the [n]th, before
   runs of no values. *)
let rec needed runs =
  let empty r = values_of r = [] in
  match runs with
  | r :: rs when empty r && List.for_all empty rs -> []
  | r :: rs -> r :: needed rs
  | [] -> []

let int64s n = Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout n

let hybrid_runs =
  let pp ppf (w, runs, pos, slack) =
    Format.fprintf ppf "width %d, %d runs, at %d, %d bytes after" w
      (List.length runs) pos slack
  in
  prop "the hybrid decodes the runs the format specifies"
    (Gen.with_pp pp hybrid_case)
    (fun (w, runs, pos, slack) ->
      let data = encode w runs in
      let len = String.length data in
      let used = String.length (encode w (needed runs)) in
      let src = filled (pos + len + slack) 0xff in
      String.iteri (fun i c -> src.{pos + i} <- Char.code c) data;
      let vs = Array.of_list (List.concat_map values_of runs) in
      let n = Array.length vs in
      let bound = 1 + Array.fold_left max (-1) vs in
      let bits = function Packed vs -> List.length vs * w | Repeated _ -> 0 in
      cover "a packed run of more than eight bytes"
        (List.exists (fun r -> bits r > 64) runs);
      cover "a packed run of under eight bytes"
        (List.exists (fun r -> bits r > 0 && bits r < 64) runs);
      (* Indices unpack eight at a time while eight bytes from the eighth's
         first byte lie in the span. *)
      let rec whole before = function
        | [] -> false
        | r :: rest ->
            let upto = before @ [ r ] in
            let packed =
              match r with
              | Packed vs when List.compare_length_with vs 8 >= 0 ->
                  let payload = (List.length vs + 7) / 8 * w in
                  String.length (encode w upto) - payload + (7 * w / 8) + 8
                  <= len
              | _ -> false
            in
            packed || whole upto rest
      in
      cover "eight values unpacked at once" (w > 0 && whole [] runs);
      let dst = int64s (n + 2) in
      Bigarray.Array1.fill dst (-1L);
      equal ~msg:"bytes" int used (hybrid_int64 src pos len w n dst 1 bound);
      equal ~msg:"values" (array int64) (Array.map Int64.of_int vs)
        (Array.init n (fun i -> dst.{i + 1}));
      equal ~msg:"around" (pair int64 int64) (-1L, -1L) (dst.{0}, dst.{n + 1});
      if w <= 8 then begin
        let b = filled (n + 1) 0xff in
        equal ~msg:"bytes, into bytes" int used
          (hybrid_bytes src pos len w n b 0 bound);
        equal ~msg:"values, into bytes" (array int) vs
          (Array.init n (fun i -> b.{i}))
      end;
      if bound > 0 then
        equal ~msg:"a value at the bound" int (-2)
          (hybrid_int64 src pos len w n dst 0 (bound - 1)))

(* [hybrid_bits src pos len width n dst at] decodes [n] values of [width] bits,
   at most 1, into the bit array [dst] from bit [at], as definition levels and
   RLE booleans decode, and writes no other bit of [dst]: a page's rows may
   start inside a byte. *)
external hybrid_bits : bytes -> int -> int -> int -> int -> bytes -> int -> int
  = "talon_parquet_hybrid_bits_byte" "talon_parquet_hybrid_bits"
[@@noalloc]

let bit_case =
  let open Gen in
  let* w = int_range 0 1 in
  let value = int_range 0 w in
  let run =
    one_of
      [
        map (fun (k, v) -> Repeated (k, v)) (pair (int_range 0 150) value);
        map
          (fun vs -> Packed vs)
          (list ~size:(map (( * ) 8) (int_range 0 20)) value);
      ]
  in
  let* runs = list ~size:(int_range 1 4) run in
  let* at = int_range 0 130 in
  let+ ones = bool in
  (w, runs, at, ones)

let hybrid_bit_runs =
  let pp ppf (w, runs, at, ones) =
    Format.fprintf ppf "width %d, %d runs, from bit %d, over %s" w
      (List.length runs) at
      (if ones then "ones" else "zeros")
  in
  prop "the hybrid decodes bits at any bit, and writes no other"
    (Gen.with_pp pp bit_case)
    (fun (w, runs, at, ones) ->
      let data = encode w runs in
      let len = String.length data in
      let src = filled len 0 in
      String.iteri (fun i c -> src.{i} <- Char.code c) data;
      let vs = Array.of_list (List.concat_map values_of runs) in
      let n = Array.length vs in
      let size = (at + n + 7) / 8 + 2 in
      let around = if ones then 1 else 0 in
      let dst = filled size (if ones then 0xff else 0) in
      cover "64 bits or more from inside a byte" (n >= 64 && at mod 8 <> 0);
      cover "64 bits or more from a byte" (n >= 64 && at mod 8 = 0);
      let bit i = (dst.{i / 8} lsr (i mod 8)) land 1 in
      let consumed = hybrid_bits src 0 len w n dst at in
      equal ~msg:"bytes" int (String.length (encode w (needed runs))) consumed;
      equal ~msg:"values" (array int) vs (Array.init n (fun i -> bit (at + i)));
      equal ~msg:"before" (array int) (Array.make at around)
        (Array.init at bit);
      let past = (8 * size) - at - n in
      equal ~msg:"after" (array int) (Array.make past around)
        (Array.init past (fun i -> bit (at + n + i))))

(* Dictionary-encoded text: a dictionary page of PLAIN byte arrays, then a data
   page of the indices, bit-packed at the width of the largest. The words run
   from empty to past 32 bytes through 4, 8 and 16, where the gather changes
   how it copies a row, in ASCII, two- and four-byte UTF-8 and U+0000. *)
let word =
  let open Gen in
  let scalar = of_list [ "a"; "b"; "\xc3\xa9"; "\000"; "\xf0\x9d\x84\x9e" ] in
  (* [fill n] is scalars of [n] bytes in all, padded with ["a"]. *)
  let rec fill n =
    if n = 0 then constant ""
    else
      let* s = scalar in
      let s = if String.length s <= n then s else "a" in
      let+ rest = fill (n - String.length s) in
      s ^ rest
  in
  bind
    (one_of
       [
         int_range 0 3; int_range 4 7; int_range 8 15; int_range 16 32;
         int_range 33 40;
       ])
    fill

let dictionary_case =
  let open Gen in
  let* words = array ~size:(int_range 1 12) word in
  let+ indices =
    list ~size:(int_range 0 40) (int_range 0 (Array.length words - 1))
  in
  (words, indices)

(* [plain_values ws] is the PLAIN encoding of the byte arrays [ws]. *)
let plain_values ws =
  let b = Buffer.create 64 in
  List.iter
    (fun w ->
      Buffer.add_int32_le b (Int32.of_int (String.length w));
      Buffer.add_string b w)
    ws;
  Buffer.contents b

(* [dictionary_pages words indices] is a dictionary page of [words], then a
   data page of [indices], and the dictionary page's length. *)
let dictionary_pages words indices =
  let plain = plain_values (Array.to_list words) in
  let dict =
    thrift
      (Struct
         [
           (1, I32 2);
           (2, I32 (String.length plain));
           (3, I32 (String.length plain));
           (7, Struct [ (1, I32 (Array.length words)); (2, I32 0) ]);
         ])
    ^ plain
  in
  let rec width w =
    if 1 lsl w >= Array.length words then w else width (w + 1)
  in
  let w = width 0 in
  let data =
    page ~data_page:[ (2, I32 8) ] (List.length indices)
      (String.make 1 (Char.chr w) ^ encode w [ Packed indices ])
  in
  (dict ^ data, String.length dict)

let dictionary_file words indices =
  let pages, dict = dictionary_pages words indices in
  parquet ~rows:(List.length indices)
    ~column:[ (1, I32 6); (6, I32 0) ]
    ~chunk:[ (9, I64 (4 + dict)); (11, I64 4) ]
    pages

(* [text_file rows columns] is a file of one row group of [rows] rows of
   required byte-array columns [(name, chunk)], each chunk [`Dict (words,
   indices)] or [`Plain values]. *)
let text_file rows columns =
  let chunk (pos, acc) (name, c) =
    let pages, offsets =
      match c with
      | `Dict (words, indices) ->
          let pages, dict = dictionary_pages words indices in
          (pages, [ (9, I64 (pos + dict)); (11, I64 pos) ])
      | `Plain vs ->
          (page (List.length vs) (plain_values vs), [ (9, I64 pos) ])
    in
    let size = I64 (String.length pages) in
    let meta =
      [ (1, I32 6); (3, List [ Bin name ]); (4, I32 0); (5, I64 rows) ]
      @ [ (6, size); (7, size) ] @ offsets
    in
    let element = Struct [ (1, I32 6); (3, I32 0); (4, Bin name) ] in
    let c = (pages, Struct [ (3, Struct meta) ], element) in
    (pos + String.length pages, c :: acc)
  in
  let _, cs = List.fold_left chunk (4, []) columns in
  let cs = List.rev cs in
  let root = Struct [ (4, Bin "schema"); (5, I32 (List.length cs)) ] in
  let meta =
    thrift
      (Struct
         [
           (2, List (root :: List.map (fun (_, _, e) -> e) cs));
           (3, I64 rows);
           ( 4,
             List
               [
                 Struct
                   [
                     (1, List (List.map (fun (_, m, _) -> m) cs));
                     (3, I64 rows);
                   ];
               ] );
         ])
  in
  let len = Bytes.create 4 in
  Bytes.set_int32_le len 0 (Int32.of_int (String.length meta));
  let pages = String.concat "" (List.map (fun (p, _, _) -> p) cs) in
  Rig.Buffer.of_string ("PAR1" ^ pages ^ meta ^ Bytes.to_string len ^ "PAR1")

let dictionary_text =
  prop "dictionary-encoded text reads as its words at the indices"
    dictionary_case (fun (words, indices) ->
      let used lo hi =
        List.exists
          (fun i ->
            let l = String.length words.(i) in
            lo <= l && l <= hi)
          indices
      in
      cover "a row of 1 to 3 bytes" (used 1 3);
      cover "a row of 4 to 7 bytes" (used 4 7);
      cover "a row of 8 to 15 bytes" (used 8 15);
      cover "a row of 16 to 32 bytes" (used 16 32);
      cover "a row past 32 bytes" (used 33 max_int);
      let col = Error.get_ok (read_x (dictionary_file words indices)) in
      equal
        (array (option string))
        (Array.of_list (List.map (fun i -> Some words.(i)) indices))
        (Column.options Kind.string col))

(* In-scan filters: predicates on one text column, drawn over its values and
   texts it does not hold. *)

type op = [ `Eq | `Ne | `Lt | `Le | `Gt | `Ge ]

type pred =
  | Cmp of op * string
  | In of string list
  | Null
  | Not of pred
  | And of pred * pred
  | Or of pred * pred

let op_name : op -> string = function
  | `Eq -> "="
  | `Ne -> "<>"
  | `Lt -> "<"
  | `Le -> "<="
  | `Gt -> ">"
  | `Ge -> ">="

let rec pp_pred ppf = function
  | Cmp (op, v) -> Format.fprintf ppf "x %s %S" (op_name op) v
  | In vs ->
      Format.fprintf ppf "x in [%a]"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
           (fun ppf -> Format.fprintf ppf "%S"))
        vs
  | Null -> Format.pp_print_string ppf "x is null"
  | Not p -> Format.fprintf ppf "not (%a)" pp_pred p
  | And (p, q) -> Format.fprintf ppf "(%a && %a)" pp_pred p pp_pred q
  | Or (p, q) -> Format.fprintf ppf "(%a || %a)" pp_pred p pp_pred q

(* [filter_of x v p] is [p] on the column [x], whose values [v] makes of
   texts. *)
let rec filter_of x v = function
  | Cmp (op, s) -> (
      let s = Expr.const (v s) in
      match op with
      | `Eq -> Expr.(x = s)
      | `Ne -> Expr.(x <> s)
      | `Lt -> Expr.(x < s)
      | `Le -> Expr.(x <= s)
      | `Gt -> Expr.(x > s)
      | `Ge -> Expr.(x >= s))
  | In ss -> Expr.is_in (List.map v ss) x
  | Null -> Expr.is_null x
  | Not p -> Expr.not (filter_of x v p)
  | And (p, q) -> Expr.(filter_of x v p && filter_of x v q)
  | Or (p, q) -> Expr.(filter_of x v p || filter_of x v q)

(* The files whose text columns the source filters in its scan: dictionary
   pages, dictionary pages that fall back to PLAIN ones, PLAIN pages only,
   and no rows; each column with nulls but [req_string]; [fsb] of fixed
   length. *)
let scanned =
  [
    ("types.parquet", [ "string"; "binary"; "req_string"; "fsb" ]);
    ("fallback.parquet", [ "string"; "binary" ]);
    ("types_plain.parquet", [ "string"; "binary"; "req_string"; "fsb" ]);
    ("empty.parquet", [ "string"; "binary" ]);
  ]

let is_binary c = c = "binary" || c = "fsb"

(* [scan_format file] is [file]'s format, its decimals read as floats. *)
let scan_format file =
  let float f c =
    match P.with_type c (Type.Any Type.float64) f with
    | f -> f
    | exception Invalid_argument _ -> f
  in
  List.fold_left float (sniff file) [ "d9"; "d18" ]

let scan_source file = P.source (scan_format file) (buffer file)

(* [texts file c] is the texts of the column [c] of [file], as read. *)
let texts =
  let memo = Hashtbl.create 8 in
  fun file c ->
    match Hashtbl.find_opt memo (file, c) with
    | Some ts -> ts
    | None ->
        let col = Error.get_ok (read (scan_format file) file c) in
        let ts =
          if is_binary c then
            Array.map (Option.map (fun b -> (b : Binary.t :> string)))
              (Column.options Kind.binary col)
          else Column.options Kind.string col
        in
        let ts = Array.of_list (List.filter_map Fun.id (Array.to_list ts)) in
        Hashtbl.replace memo (file, c) ts;
        ts

let in_scan_case =
  let open Gen in
  let* file, columns = of_list scanned in
  let* c = of_list columns in
  let have = texts file c in
  let absent = [ ""; "v"; "zz"; "\x00"; "v1\xc3\xa9" ] in
  let text =
    if Array.length have = 0 then of_list absent
    else frequency [ (3, of_list (Array.to_list have)); (1, of_list absent) ]
  in
  let op = of_list [ `Eq; `Ne; `Lt; `Le; `Gt; `Ge ] in
  let atom =
    frequency
      [
        (4, map (fun (o, t) -> Cmp (o, t)) (pair op text));
        (2, map (fun ts -> In ts) (list ~size:(int_range 0 9) text));
        (3, constant Null);
      ]
  in
  let rec pred depth =
    if depth = 0 then atom
    else
      let sub = pred (depth - 1) in
      frequency
        [
          (3, atom);
          (1, map (fun p -> Not p) sub);
          (1, map (fun (p, q) -> And (p, q)) (pair sub sub));
          (1, map (fun (p, q) -> Or (p, q)) (pair sub sub));
        ]
  in
  let+ p = pred 2 and+ positive = bool and+ shown = bool in
  (file, c, p, positive, shown)

(* [kept ~shown c f q] is the rows that [q] keeps, filtered by [f], read back
   as the floats of [f64] and, if [shown], the texts of [c]. *)
let kept ~shown c f q =
  let names = if shown then [ c; "f64" ] else [ "f64" ] in
  let t =
    Query.filter f q
    |> Query.select Expr.[ keep (Sel.names names) ]
    |> Query.run |> Error.get_ok
  in
  let text =
    if not shown then Array.make (Talon.rows t) "-"
    else if is_binary c then
      Array.map
        (Option.fold ~none:"null" ~some:(fun b ->
             quote (b : Binary.t :> string)))
        (Column.options Kind.binary (Talon.column t c))
    else
      Array.map
        (Option.fold ~none:"null" ~some:quote)
        (Column.options Kind.string (Talon.column t c))
  in
  let floats = Column.options Kind.float (Talon.column t "f64") in
  Array.map2
    (fun s x ->
      s ^ " " ^ Option.fold ~none:"null" ~some:(Printf.sprintf "%h") x)
    text floats

let in_scan_filters =
  let pp ppf (file, c, p, positive, shown) =
    Format.fprintf ppf "%s, %s: %a%s%s" file c pp_pred p
      (if positive then " && f64 > 0" else "")
      (if shown then "" else ", f64 alone")
  in
  prop "a source keeps the rows a filter on one text column keeps"
    (Gen.with_pp pp in_scan_case) (fun (file, c, p, positive, shown) ->
      let f =
        if is_binary c then filter_of (Col.binary c) Binary.of_string p
        else filter_of (Col.string c) Fun.id p
      in
      let f = if positive then Expr.(f && Col.float "f64" > float 0.) else f in
      let src = scan_source file in
      let whole = Error.get_ok (Query.run (Query.of_source src)) in
      let expected = kept ~shown:true c f (Query.of_table whole) in
      let rows = Talon.rows whole in
      cover "dictionary pages" (file = "types.parquet");
      cover "dictionary pages falling back to PLAIN"
        (file = "fallback.parquet");
      cover "PLAIN pages" (file = "types_plain.parquet");
      cover "no rows" (rows = 0);
      cover "some rows pass, others not"
        (Array.length expected > 0 && Array.length expected < rows);
      cover "a null row passes"
        (Array.exists (String.starts_with ~prefix:"null ") expected);
      cover "the filtered column not read back" (not shown);
      let expected = kept ~shown c f (Query.of_table whole) in
      equal (array string) expected (kept ~shown c f (Query.of_source src)))

(* Columns of a valid and an invalid UTF-8 word, read as text: a filter in
   the scan fails where a read fails, at the row, and not on a word no row
   reads, on the column it filters or another, dictionary-encoded or PLAIN. *)
let invalid_words =
  test "a filtered read fails only at a row of invalid UTF-8" (fun () ->
      let rows q =
        match Query.run q with
        | Ok t -> string_of_int (Talon.rows t) ^ " rows"
        | Error e -> error_text e
      in
      let as_text b names =
        let f = Error.get_ok (P.sniff b) in
        List.fold_left (fun f c -> P.with_type c (any Type.string) f) f names
      in
      List.iter
        (fun indices ->
          let b = dictionary_file [| "ok"; "\xff" |] indices in
          let q = Query.of_source (P.source (as_text b [ "x" ]) b) in
          Printf.printf "x at %s: read %s, x = \"ok\" %s\n"
            (String.concat " " (List.map string_of_int indices))
            (rows q)
            (rows (Query.filter Expr.(Col.string "x" = string "ok") q)))
        [ [ 0; 0; 0 ]; [ 0; 1; 0 ] ];
      let s = `Dict ([| "a"; "b" |], [ 0; 0; 1; 0 ]) in
      List.iter
        (fun (name, y) ->
          let b = text_file 4 [ ("s", s); ("y", y) ] in
          let q = Query.of_source (P.source (as_text b [ "s"; "y" ]) b) in
          let s_is v = Query.filter Expr.(Col.string "s" = string v) q in
          Printf.printf "%s y: read %s, s = \"a\" %s, s = \"b\" %s\n"
            name (rows q) (rows (s_is "a")) (rows (s_is "b")))
        [
          ("dictionary", `Dict ([| "ok"; "\xff" |], [ 0; 0; 1; 0 ]));
          ("PLAIN", `Plain [ "ok"; "ok"; "\xff"; "ok" ]);
        ];
      expect (output ())
      @@ __POS_OF__
           {|
        x at 0 0 0: read 3 rows, x = "ok" 3 rows
        x at 0 1 0: read row group 0: column "x": row 1: invalid UTF-8 at byte 0, x = "ok" row group 0: column "x": row 1: invalid UTF-8 at byte 0
        dictionary y: read row group 0: column "y": row 2: invalid UTF-8 at byte 0, s = "a" row group 0: column "y": row 2: invalid UTF-8 at byte 0, s = "b" row group 0: column "y": row 2: invalid UTF-8 at byte 0
        PLAIN y: read row group 0: column "y": row 2: invalid UTF-8 at byte 0, s = "a" row group 0: column "y": row 2: invalid UTF-8 at byte 0, s = "b" row group 0: column "y": row 2: invalid UTF-8 at byte 0
        |})

(* [first_batch name] is the first batch a query of the file [name]'s source
   folds over: its first row group, as decoded. *)
let first_batch name =
  let b = buffer name in
  let q = Query.of_source (P.source (sniff name) b) in
  let first acc t = match acc with None -> Some t | some -> some in
  Option.get (Error.get_ok (Query.fold q ~init:None first))

let sources =
  group "Sources"
    [
      spread_tail;
      hybrid_runs;
      hybrid_bit_runs;
      dictionary_text;
      in_scan_filters;
      invalid_words;
      test
        "a decoded bool column of nulls in 731 rows is canonical, the bits \
         past its rows clear" (fun () ->
          let t = first_batch "bits_v1.parquet" in
          equal int 731 (rows t);
          satisfies ~claim:"the decoded column itself" pass
            (fun c -> c == Talon.column t "b")
            (Talon.column t "b"));
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
          equal int 2 (Talon.rows (Error.get_ok (Query.run q))));
      test "file names its source and counts its rows" (fun () ->
          let plan s = Format.asprintf "%a" Query.pp (Query.of_source s) in
          print_endline (plan (Error.get_ok (P.file (path "names.parquet"))));
          print_endline
            (plan (P.source (sniff "names.parquet") (buffer "names.parquet")));
          expect (output ())
          @@ __POS_OF__
               {|
            query → "a b" int64, "\"q\"" int64, é int64, "" int64
            parquet "support/names.parquet" (4 columns, 2 rows)
            query → "a b" int64, "\"q\"" int64, é int64, "" int64
            parquet (4 columns)
            |});
      test "file refuses what sniff refuses, naming the file" (fun () ->
          List.iter
            (fun name -> print_endline (outcome (P.file (path name))))
            [ "brotli.parquet"; "missing.parquet" ];
          expect (output ())
          @@ __POS_OF__
               {|
            support/brotli.parquet: row group 0: column "i32" is compressed with Brotli, which talon does not read
            support/missing.parquet: No such file or directory
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
             (any (Type.wall Us))
             (sniff "int96_from_spark.parquet"));
      show "int96.parquet" "ts_ns"
        ~f:
          (P.with_type "ts_ns" (any (Type.wall Us)) (sniff "int96.parquet"));
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

(* Memory *)

(* nulls_dict.parquet's column [x]: 2^16 int64s in one row group, a tenth of
   them null, in a dictionary of 97 values. *)
let nulls_rows = 1 lsl 16

(* Reads of one source, which keeps the buffer its pages decompress into. *)
let read_nulls =
  let s = P.source (sniff "nulls_dict.parquet") (buffer "nulls_dict.parquet") in
  fun () ->
    Query.of_source s
    |> Query.run
    |> Result.map (fun t -> Talon.column t "x")
    |> Error.get_ok

(* [with_ratio r f] is [f ()] with the collector's [custom_major_ratio] at
   [r]. *)
let with_ratio r f =
  let was = (Gc.get ()).custom_major_ratio in
  Gc.set { (Gc.get ()) with custom_major_ratio = r };
  Fun.protect f ~finally:(fun () ->
      Gc.set { (Gc.get ()) with custom_major_ratio = was })

(* The decoder's shapes: every column of the types fixture in one row group,
   with dictionary pages and with plain pages, its decimals read both ways,
   its text read as categoricals, int96 timestamps, and a file of no row. *)
let shapes =
  let columns =
    [
      "bool"; "i8"; "i16"; "i32"; "i64"; "u8"; "u16"; "u32"; "u64"; "f16";
      "f32"; "f64"; "date"; "clock_ms"; "clock_us"; "clock_ns"; "ts_ms";
      "ts_us_utc"; "ts_us_paris"; "ts_ns"; "string"; "binary"; "fsb"; "uuid";
      "all_null"; "no_null"; "req_i32"; "req_string";
    ]
  in
  let texts = Array.init 40 (fun i -> Printf.sprintf "v%d\xc3\xa9\n\"" i) in
  let typed =
    [
      ("d9", Some (any Type.float64)); ("d9", Some (any Type.int64));
      ("d18", Some (any Type.float64)); ("d18", Some (any Type.int64));
      ("string", Some (any (Type.categorical texts)));
      ( "req_string",
        Some (any (Type.categorical (Array.init 7 (Printf.sprintf "r%d")))) );
    ]
  in
  let file name =
    List.map (fun c -> (name, c, None)) columns
    @ List.map (fun (c, ty) -> (name, c, ty)) typed
  in
  file "one_group.parquet" @ file "one_group_plain.parquet"
  @ [
      ("one_group_int96.parquet", "ts_ns", None);
      ("empty.parquet", "i64", None);
      ("empty.parquet", "string", None);
    ]

let memory =
  group "Memory"
    [
      (* A column is host memory: the host holds its arrays' bytes while it
         is live, and has them back once it is collected. Each case is a
         shape of the decoder, in one row group so that the column is that
         chunk's arrays: validity bits and the spread of values and offsets
         under nulls, dictionary indices gathered as values and as byte
         strings, plain byte strings, booleans, int96 timestamps, decimals as
         unscaled integers and as floats, categoricals, and no row. *)
      cases "a column is the host's memory while it is live" shapes
        ~id:(fun (file, column, ty) ->
          Printf.sprintf "%s %s%s" file column
            (match ty with
            | None -> ""
            | Some (Type.Any t) -> Format.asprintf " as %a" Type.pp t))
        (fun (file, column, ty) ->
          let f = sniff file in
          let f =
            if file = "one_group_int96.parquet" then f
            else
              f
              |> P.with_type "d9" (any Type.float64)
              |> P.with_type "d18" (any Type.float64)
          in
          let f =
            match ty with Some ty -> P.with_type column ty f | None -> f
          in
          let b = buffer file in
          Talon_held.check (fun () ->
              Error.get_ok (read_from f b column)));
      (* Dropped columns pace the collector as rig's host buffers do: a
         major cycle at most once per 4 MiB of buffers of 64 KiB or more,
         whatever [custom_major_ratio] says of the heap. A read of the column
         of nulls and a dictionary makes four arrays of its 2^16 rows of 8
         bytes in such buffers, the table of values, the indices, the
         gathered values and the spread ones, and its validity, of 8 KiB,
         dies young; it decompresses into a buffer its source keeps. So at a
         ratio of 1, which paces a bigarray of Bigarray's own at a cycle
         each, 64 reads bring at most a cycle per 4 MiB of those arrays, and
         two under way at either end. *)
      test "dropped columns pace the collector by the program's memory"
        (fun () ->
          let reads = 64 in
          let c =
            with_ratio 1 @@ fun () ->
            ignore (read_nulls ());
            Talon_held.cycles (fun () ->
                for _ = 1 to reads do
                  ignore (Sys.opaque_identity (read_nulls ()))
                done)
          in
          let made = reads * 4 * 8 * nulls_rows in
          at_most
            ~msg:("major cycles at a ratio of 1: " ^ Talon_held.pp_cycles c)
            int
            ~than:((made / (4 lsl 20)) + 2)
            c.majors);
    ]

let () =
  exit
    (run "talon.parquet"
       [
         agreement; clocks; sniffing; formats; synthesized; nested_files;
         sources; decoding; memory;
       ])
