(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next
open Windtrap
module Csv = Talon_next_csv
module Reader = Bytesrw.Bytes.Reader

(* Helpers *)

let any t = Type.Any t
let error_text e = Format.asprintf "%a" Error.pp e
let format_text f = Format.asprintf "%a" Csv.pp_format f
let sniff ?rows ?nulls s = Csv.sniff ?rows ?nulls (Reader.of_string s)
let sniffed ?rows ?nulls s = Error.get_ok (sniff ?rows ?nulls s)

(* [columns ty] is a format without a header whose columns are named [c1], [c2],
   … and read as [ty]. *)
let columns ?nulls types =
  Csv.format ~header:false ?nulls
    (List.mapi (fun i t -> (Printf.sprintf "c%d" (i + 1), t)) types)

let read ?slice_length f s =
  Csv.Private.read f (Reader.of_string ?slice_length s)

(* The shortest of [%.15g], [%.16g] and [%.17g] that reads back as [x]. *)
let float_text x =
  let fits p = float_of_string (Printf.sprintf "%.*g" p x) = x in
  Printf.sprintf "%.*g" (if fits 15 then 15 else if fits 16 then 16 else 17) x

(* [strings c] is the rows of the varsize column [c]. *)
let strings (c : Csv.Private.column) =
  match c with
  | Fixed _ -> fail "a fixed column"
  | Varsize { valid; offsets; data } ->
      let o = Array.map Int64.to_int (Nx.to_array offsets) in
      let d = Nx.to_array data in
      let valid = Option.map Nx.to_array valid in
      List.init
        (Array.length o - 1)
        (fun i ->
          match valid with
          | Some v when not v.(i) -> None
          | _ ->
              Some
                (String.init
                   (o.(i + 1) - o.(i))
                   (fun k -> Char.chr d.(o.(i) + k))))

let read_float s =
  match read (columns [ any Type.float64 ]) s with
  | Ok [ [| Fixed { values = Nx.P v; _ } |] ] ->
      (Nx.to_array (Nx.cast Nx.float64 v)).(0)
  | _ -> fail "not one float"

(* [cells c] is the dtype of [c] and its rows printed, [∅] at a null. *)
let cells (c : Csv.Private.column) =
  let nulls valid n f =
    let valid = Option.map Nx.to_array valid in
    List.init n (fun i ->
        match valid with Some v when not v.(i) -> "∅" | _ -> f i)
  in
  match c with
  | Fixed { valid; values = Nx.P x } ->
      let dtype = Format.asprintf "%a" Nx.pp_dtype (Nx.dtype x) in
      let row =
        match dtype with
        | "bool" ->
            let a = Nx.to_array (Nx.cast Nx.uint8 x) in
            fun i -> if a.(i) = 1 then "true" else "false"
        | "uint64" ->
            let a = Nx.to_array (Nx.bitcast Nx.int64 x) in
            fun i -> Printf.sprintf "%Lu" a.(i)
        | "float16" | "float32" | "float64" ->
            let a = Nx.to_array (Nx.cast Nx.float64 x) in
            fun i -> float_text a.(i)
        | _ ->
            let a = Nx.to_array (Nx.cast Nx.int64 x) in
            fun i -> Int64.to_string a.(i)
      in
      (dtype, nulls valid (Nx.shape x).(0) row)
  | Varsize { valid; offsets; data } ->
      let o = Array.map Int64.to_int (Nx.to_array offsets) in
      let d = Nx.to_array data in
      let row i =
        Printf.sprintf "%S"
          (String.init (o.(i + 1) - o.(i)) (fun k -> Char.chr d.(o.(i) + k)))
      in
      ("bytes", nulls valid (Array.length o - 1) row)

(* [decode f s] is each column of [f] read from [s], printed as its dtype and
   its rows across batches: [int8 [1; ∅]]. *)
let decode ?slice_length f s =
  match read ?slice_length f s with
  | Error e -> "error: " ^ error_text e
  | Ok [] -> "no rows"
  | Ok (b :: _ as batches) ->
      String.concat "\n"
        (List.init (Array.length b) (fun j ->
             let dtype = fst (cells b.(j)) in
             let rows = List.concat_map (fun b -> snd (cells b.(j))) batches in
             Printf.sprintf "%s [%s]" dtype (String.concat "; " rows)))

let failure f s = match read f s with Ok _ -> "ok" | Error e -> error_text e

(* Formats *)

let formats =
  let raises_invalid name f =
    test name (fun () -> raises_match (Exn.invalid_arg ?substring:None) f)
  in
  let two = [ ("a", any Type.int64); ("b", any Type.string) ] in
  group "format"
    [
      test "defaults to a comma, a double quote, a header and no null token"
        (fun () ->
          expect (format_text (Csv.format two))
          @@ __POS_OF__
               {|
            csv (2 columns, separator ',', quote '"', header)
              a int64
              b string
            |});
      test "prints its null tokens quoted, and a tab as an escape" (fun () ->
          expect
            (format_text
               (Csv.format ~sep:'\t' ~quote:'\'' ~header:false
                  ~nulls:[ "NA"; "\"" ] two))
          @@ __POS_OF__
               {|
            csv (2 columns, separator '\t', quote '\'', no header, nulls ["NA"; "\""])
              a int64
              b string
            |});
      raises_invalid "refuses no column" (fun () -> Csv.format []);
      raises_invalid "refuses two columns of one name" (fun () ->
          Csv.format [ ("a", any Type.int64); ("a", any Type.string) ]);
      test "names a control byte in a refused name as hexadecimal" (fun () ->
          raises_match (Exn.invalid_arg ~substring:{|"a\x0a"|}) (fun () ->
              Csv.format [ ("a\n", any Type.int64); ("a\n", any Type.int64) ]));
      raises_invalid "refuses a name that is not UTF-8" (fun () ->
          Csv.format [ ("\xff", any Type.int64) ]);
      raises_invalid "refuses a clock, which CSV does not read" (fun () ->
          Csv.format [ ("a", any (Type.clock Us)) ]);
      raises_invalid "refuses a list type" (fun () ->
          Csv.format [ ("a", any (Type.list Type.int64)) ]);
      raises_invalid "refuses a line feed as the separator" (fun () ->
          Csv.format ~sep:'\n' two);
      raises_invalid "refuses a carriage return as the quote" (fun () ->
          Csv.format ~quote:'\r' two);
      raises_invalid "refuses a separator outside ASCII" (fun () ->
          Csv.format ~sep:'\xc3' two);
      raises_invalid "refuses a separator equal to the quote" (fun () ->
          Csv.format ~sep:'"' two);
      cases
        ~name:(fun t -> Printf.sprintf "refuses the null token %S" t)
        "null tokens"
        [ "a,b"; "\"x"; "a\nb"; "\r" ]
        (fun t ->
          raises_match (Exn.invalid_arg ?substring:None) (fun () ->
              Csv.format ~nulls:[ t ] two));
      test "accepts an empty name and a name in UTF-8" (fun () ->
          ignore
            (Csv.format [ ("", any Type.int64); ("café", any Type.string) ]));
    ]

let with_types =
  let f = Csv.format [ ("a", any Type.int64); ("b", any Type.string) ] in
  group "with_type"
    [
      test "changes the type of the column it names" (fun () ->
          expect (format_text (Csv.with_type "a" (any Type.float32) f))
          @@ __POS_OF__
               {|
            csv (2 columns, separator ',', quote '"', header)
              a float32
              b string
            |});
      test "marks the type declared in a sniffed format" (fun () ->
          let s = sniffed "a,b\n1,x\n" in
          expect (format_text (Csv.with_type "a" (any Type.int8) s))
          @@ __POS_OF__
               {|
            csv (2 columns, separator ',', quote '"', header), types sniffed from 1 row
              a int8   declared
              b string
            |});
      test "raises on a column the format does not have" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"\"c\"") (fun () ->
              Csv.with_type "c" (any Type.int64) f));
      test "raises on a type CSV does not read" (fun () ->
          raises_match (Exn.invalid_arg ?substring:None) (fun () ->
              Csv.with_type "a" (any (Type.duration Ms)) f));
    ]

(* Sniffing *)

let dialect s = List.hd (String.split_on_char '\n' (format_text (sniffed s)))

let column_types s =
  List.tl (String.split_on_char '\n' (format_text (sniffed s)))
  |> List.map String.trim

let sniff_errors () =
  let limited s n = Reader.limit n (Reader.of_string ~slice_length:2 s) in
  let outcome r =
    match Csv.sniff r with Ok _ -> "ok" | Error e -> error_text e
  in
  List.iter
    (fun (name, r) -> Printf.printf "%s: %s\n" name (outcome r))
    [
      ("empty", Reader.of_string "");
      ("a byte order mark", Reader.of_string "\xEF\xBB\xBF");
      ("ragged", Reader.of_string "a,b\n1,2\n3\n");
      ("ragged with a tab", Reader.of_string "a\tb\n1\t2\t3\n");
      ("unterminated quote", Reader.of_string "a,b\n1,\"2\n3,4\n");
      ("duplicate names", Reader.of_string "a,b,a\n1,2,3\n");
      ("name not UTF-8", Reader.of_string "a,\xff\n1,2\n");
      ("stream error", limited "a,b\n1,2\n" 5);
    ];
  expect (output ())
  @@ __POS_OF__
       {|
    empty: the input holds no record
    a byte order mark: the input holds no record
    ragged: line 3, column 1: no separator splits every record into the same number of fields: with ',', the record has 1 field, and the first one 2
    ragged with a tab: line 2, column 1: no separator splits every record into the same number of fields: with '\t', the record has 3 fields, and the first one 2
    unterminated quote: line 2, column 3: a quoted field does not end
    duplicate names: line 1, column 5: "a": the header names two columns the same
    name not UTF-8: line 1, column 3: "\xff": the column name is not UTF-8
    stream error: reader:5: Limit of 5 bytes exceeded.
    |}

let sniffing =
  group "sniff"
    [
      cases
        ~name:(fun (name, _, _) -> name)
        "separators"
        [
          ( "a comma",
            "a,b\n1,2\n",
            "csv (2 columns, separator ',', quote '\"', header), types sniffed \
             from 1 row" );
          ( "a tab",
            "a\tb\n1\t2\n",
            "csv (2 columns, separator '\\t', quote '\"', header), types \
             sniffed from 1 row" );
          ( "a semicolon",
            "a;b;c\n1;2;3\n",
            "csv (3 columns, separator ';', quote '\"', header), types sniffed \
             from 1 row" );
          ( "a bar",
            "a|b\n1|2\n",
            "csv (2 columns, separator '|', quote '\"', header), types sniffed \
             from 1 row" );
          ( "the one that splits into the most fields",
            "a;b,c;d\n1;2,3;4\n",
            "csv (3 columns, separator ';', quote '\"', header), types sniffed \
             from 1 row" );
          ( "the first on a tie",
            "a;b,c\n1;2,3\n",
            "csv (2 columns, separator ',', quote '\"', header), types sniffed \
             from 1 row" );
          ( "one that splits every record alike, over one that does not",
            "a;b;c,d\n1;2,3\n",
            "csv (2 columns, separator ',', quote '\"', header), types sniffed \
             from 1 row" );
          ( "not a separator between quotes",
            "a;\"b,c,d\"\n1;2\n",
            "csv (2 columns, separator ';', quote '\"', header), types sniffed \
             from 1 row" );
          ( "a comma for one column",
            "a\n1\n",
            "csv (1 column, separator ',', quote '\"', header), types sniffed \
             from 1 row" );
        ]
        (fun (_, s, expected) -> equal string expected (dialect s));
      cases
        ~name:(fun (name, _, _) -> name)
        "types"
        [
          ("bool from true and false", "x\ntrue\nfalse\n", [ "x bool" ]);
          ("string from True and FALSE", "x\nTrue\nFALSE\n", [ "x string" ]);
          ("string from 0 and 1 with true", "x\n1\ntrue\n", [ "x string" ]);
          ("int64 from integers", "x\n1\n-2\n+3\n", [ "x int64" ]);
          ( "int64 at its extremes",
            "x\n9223372036854775807\n-9223372036854775808\n",
            [ "x int64" ] );
          ( "string from an integer int64 does not hold",
            "x\n1\n9223372036854775808\n",
            [ "x string" ] );
          ("float64 from integers and a float", "x\n1\n2.5\n", [ "x float64" ]);
          ("float64 from an exponent", "x\n1e3\n", [ "x float64" ]);
          ("float64 from nan and inf", "x\nnan\n-inf\n1\n", [ "x float64" ]);
          ("string from a leading zero", "x\n007\n1\n", [ "x string" ]);
          ( "string from a leading zero before a point",
            "x\n01.5\n",
            [ "x string" ] );
          ("float64 from a zero before a point", "x\n0.5\n0\n", [ "x float64" ]);
          ("date from dates", "x\n2024-01-31\n0099-12-01\n", [ "x date" ]);
          ("string from an invalid date", "x\n2023-02-29\n", [ "x string" ]);
          ( "datetime[us] from datetimes",
            "x\n2024-01-31T10:00:00\n2024-01-31 10:00:00.123456\n",
            [ "x datetime[us]" ] );
          ( "datetime[ns] from a fraction finer than a microsecond",
            "x\n2024-01-31T10:00:00\n2024-01-31T10:00:00.1234567\n",
            [ "x datetime[ns]" ] );
          ( "datetime[us] from trailing zeros finer than a microsecond",
            "x\n2024-01-31T10:00:00.1234560\n",
            [ "x datetime[us]" ] );
          ( "string when a fine fraction meets a date out of nanoseconds",
            "x\n2024-01-31T10:00:00.1234567\n1500-01-01T00:00:00\n",
            [ "x string" ] );
          ( "datetime[us, UTC] from offsets",
            "x\n2024-01-31T10:00:00Z\n2024-01-31T10:00:00+05:30\n",
            [ "x datetime[us, UTC]" ] );
          ( "string from datetimes with and without offsets",
            "x\n2024-01-31T10:00:00Z\n2024-01-31T10:00:00\n",
            [ "x string" ] );
          ( "string from a date and a datetime",
            "x\n2024-01-31\n2024-01-31T10:00:00\n",
            [ "x string" ] );
          ( "string from a column of nulls",
            "x,y\n,1\n,2\n",
            [ "x string"; "y int64" ] );
          ("int64 past nulls", "x,y\n,a\n1,b\n,c\n", [ "x int64"; "y string" ]);
          ( "int64 past empty lines",
            "x,y\n1,2\n\n3,4\n\n\n",
            [ "x int64"; "y int64" ] );
          ( "string from a negative integer int64 does not hold",
            "x\n-9223372036854775809\n",
            [ "x string" ] );
          ("string from a quoted empty string", "x\n\"\"\n1\n", [ "x string" ]);
          ("never a categorical", "x\na\nb\na\nb\na\n", [ "x string" ]);
          ("int64 from a quoted integer", "x\n\"12\"\n", [ "x int64" ]);
        ]
        (fun (_, s, expected) -> equal (list string) expected (column_types s));
      cases
        ~name:(fun (name, _, _) -> name)
        "header"
        [
          ( "a header over typed columns",
            "a,b\n1,x\n",
            [ "a int64"; "b string" ] );
          ( "no header when every typed column's first field reads as its type",
            "1,x\n2,y\n",
            [ "column_1 int64"; "column_2 string" ] );
          ( "no header when the first field is null",
            ",x\n2,y\n",
            [ "column_1 int64"; "column_2 string" ] );
          ( "a header when one typed column's first field does not read",
            "1,y\n2,3\n",
            [ "1 int64"; "y int64" ] );
          ("a header over string columns", "x\ny\n", [ "x string" ]);
          ("a header over no other record", "a,b\n", [ "a string"; "b string" ]);
        ]
        (fun (_, s, expected) -> equal (list string) expected (column_types s));
      test "reads the header's quoted names undoubled" (fun () ->
          equal (list string) [ "\"a,\\\"b\" int64" ]
            (column_types "\"a,\"\"b\"\n1\n"));
      test "skips a byte order mark" (fun () ->
          equal (list string) [ "a int64" ] (column_types "\xEF\xBB\xBFa\n1\n"));
      test "infers types from the first rows records after the header"
        (fun () ->
          let f = sniffed ~rows:2 "x\n1\n2\nz\n" in
          expect (format_text f)
          @@ __POS_OF__
               {|
            csv (1 column, separator ',', quote '"', header), types sniffed from 2 rows
              x int64
            |});
      test "infers types from the first rows records without a header"
        (fun () ->
          let f = sniffed ~rows:1 "1\n2\nz\n" in
          expect (format_text f)
          @@ __POS_OF__
               {|
            csv (1 column, separator ',', quote '"', no header), types sniffed from 1 row
              column_1 int64
            |});
      test "does not count empty lines as rows" (fun () ->
          let f = sniffed ~rows:2 "x\n\n1\n\n\r\n2\nz\n" in
          expect (format_text f)
          @@ __POS_OF__
               {|
            csv (1 column, separator ',', quote '"', header), types sniffed from 2 rows
              x int64
            |});
      test "counts no rows under a header alone" (fun () ->
          equal string
            "csv (2 columns, separator ',', quote '\"', header), types sniffed \
             from no rows"
            (dialect "a,b\n\n"));
      test "counts its rows with thousands separators" (fun () ->
          equal string
            "csv (1 column, separator ',', quote '\"', header), types sniffed \
             from 1,500 rows"
            (dialect
               ("x\n" ^ String.concat "" (List.init 1500 (fun _ -> "1\n")))));
      test "skips its null tokens and keeps them in the format" (fun () ->
          expect (format_text (sniffed ~nulls:[ "NA" ] "x,y\nNA,1\n1,NA\n"))
          @@ __POS_OF__
               {|
            csv (2 columns, separator ',', quote '"', header, nulls ["NA"]), types sniffed from 2 rows
              x int64
              y int64
            |});
      test "infers from a quoted null token, which is a value" (fun () ->
          equal (list string) [ "x string" ]
            (List.tl
               (String.split_on_char '\n'
                  (format_text (sniffed ~nulls:[ "NA" ] "x\nNA\n\"NA\"\n1\n")))
            |> List.map String.trim));
      test "infers a type from fields that are not null tokens" (fun () ->
          let f = sniffed ~nulls:[ "NA"; "-" ] "x\nNA\n1\n-\n" in
          equal text "int64 [∅; 1; ∅]" (decode f "x\nNA\n1\n-\n"));
      cases
        ~name:(fun t -> Printf.sprintf "raises on the null token %S" t)
        "null tokens"
        [ "a,b"; "a\tb"; "a;b"; "a|b"; "\""; "\n" ]
        (fun t ->
          raises_match (Exn.invalid_arg ?substring:None) (fun () ->
              Csv.sniff ~nulls:[ t ] (Reader.of_string "a\n")));
      test "raises on rows below 1" (fun () ->
          raises_match (Exn.invalid_arg ?substring:None) (fun () ->
              Csv.sniff ~rows:0 (Reader.of_string "a\n")));
      cases
        ~name:(fun (n, rows) ->
          Printf.sprintf "leaves the reader whole, %d-byte slices, rows %d" n
            rows)
        "peeking"
        [ (1, 1); (3, 1); (3, 5); (64, 2); (65536, 16384) ]
        (fun (slice_length, rows) ->
          let s = "a,\"b\nc\"\r\n1,2\n3,\"x\"\"y\"\n4,5" in
          let r = Reader.of_string ~slice_length s in
          ignore (Error.get_ok (Csv.sniff ~rows r));
          equal string s (Reader.to_string r));
      test "fails on errors as baselined" sniff_errors;
    ]

(* Syntax *)

let texts n = columns (List.init n (fun _ -> any Type.string))

let syntax_errors () =
  List.iter
    (fun (name, f, s) -> Printf.printf "%s: %s\n" name (failure f s))
    [
      ("quote in an unquoted field", texts 2, "a,b\"c\n");
      ("byte after a closing quote", texts 2, "a,\"b\"c\n");
      ("unterminated quote", texts 2, "a,b\n\"c\nd\n,e\n");
      ("carriage return alone", texts 2, "a,b\rc\n");
      ("carriage return at the end", texts 1, "a\r");
      ("too many fields", texts 2, "a,b\nc,d,e\n");
      ("too few fields", texts 2, "a,b\nc\n");
      ("lines after a quoted line break", texts 2, "\"a\nb\",c\nd,e,f\n");
      ( "the header names another column",
        Csv.format [ ("a", any Type.string); ("b", any Type.string) ],
        "a,c\nx,y\n" );
      ( "the header has too few fields",
        Csv.format [ ("a", any Type.string); ("b", any Type.string) ],
        "a\n" );
      ("no header", Csv.format [ ("a", any Type.string) ], "");
      ( "a byte order mark and no header",
        Csv.format [ ("a", any Type.string) ],
        "\xEF\xBB\xBF" );
      ("a column after a byte order mark", texts 2, "\xEF\xBB\xBFa,\"b\"c\n");
    ];
  expect (output ())
  @@ __POS_OF__
       {|
    quote in an unquoted field: line 1, column 4: a quote in a field that does not start with one
    byte after a closing quote: line 1, column 6: a quoted field's closing quote is followed by a byte other than a separator or a line break
    unterminated quote: line 2, column 1: a quoted field does not end
    carriage return alone: line 1, column 4: a carriage return that no line feed follows
    carriage return at the end: line 1, column 2: a carriage return that no line feed follows
    too many fields: line 2, column 5: the record has 3 fields, and the format 2 columns
    too few fields: line 2, column 2: the record has 1 field, and the format 2 columns
    lines after a quoted line break: line 3, column 5: the record has 3 fields, and the format 2 columns
    the header names another column: line 1, column 3: "c": the header does not name column 2 "b"
    the header has too few fields: line 1, column 2: the record has 1 field, and the format 2 columns
    no header: the input is empty, and the format has a header
    a byte order mark and no header: the input is empty, and the format has a header
    a column after a byte order mark: line 1, column 6: a quoted field's closing quote is followed by a byte other than a separator or a line break
    |}

let syntax =
  group "syntax"
    [
      cases
        ~name:(fun (name, _, _, _) -> name)
        "records"
        [
          ( "a line feed ends a record",
            2,
            "a,b\nc,d\n",
            "bytes [\"a\"; \"c\"]\nbytes [\"b\"; \"d\"]" );
          ( "a carriage return and a line feed end a record",
            2,
            "a,b\r\nc,d\r\n",
            "bytes [\"a\"; \"c\"]\nbytes [\"b\"; \"d\"]" );
          ( "the last record ends at the end of the input",
            2,
            "a,b\nc,d",
            "bytes [\"a\"; \"c\"]\nbytes [\"b\"; \"d\"]" );
          ( "a quoted field holds the separator",
            1,
            "\"a,b\"\n",
            "bytes [\"a,b\"]" );
          ( "a quoted field holds line breaks",
            1,
            "\"a\nb\r\nc\"\n",
            "bytes [\"a\\nb\\r\\nc\"]" );
          ( "a doubled quote is one quote",
            1,
            "\"a\"\"b\"\"\"\n",
            "bytes [\"a\\\"b\\\"\"]" );
          ( "an unquoted empty field is null",
            2,
            ",x\n",
            "bytes [∅]\nbytes [\"x\"]" );
          ( "a quoted empty field is the empty string",
            2,
            "\"\",x\n",
            "bytes [\"\"]\nbytes [\"x\"]" );
          ( "spaces are part of a field",
            2,
            " a , b \n",
            "bytes [\" a \"]\nbytes [\" b \"]" );
          ( "a separator before a line break ends an empty field",
            2,
            "a,\nb,",
            "bytes [\"a\"; \"b\"]\nbytes [∅; ∅]" );
          ( "an empty line in one column is skipped",
            1,
            "a\n\nb\n",
            "bytes [\"a\"; \"b\"]" );
          ( "an empty line between records is skipped",
            2,
            "a,b\n\nc,d\n",
            "bytes [\"a\"; \"c\"]\nbytes [\"b\"; \"d\"]" );
          ( "empty lines at the end are skipped",
            2,
            "a,b\n\n\n",
            "bytes [\"a\"]\nbytes [\"b\"]" );
          ( "empty lines of carriage returns and line feeds are skipped",
            1,
            "\r\na\r\n\r\n\nb\r\n\r\n",
            "bytes [\"a\"; \"b\"]" );
          ( "a line holding a quoted empty field is a record",
            1,
            "a\n\"\"\n",
            "bytes [\"a\"; \"\"]" );
          ("only empty lines are no record", 1, "\n\r\n\n", "no rows");
          ("a quoted field ends the input", 1, "\"a\"", "bytes [\"a\"]");
          ( "a byte order mark is not part of the first field",
            1,
            "\xEF\xBB\xBFa\n",
            "bytes [\"a\"]" );
          ( "a byte order mark is a byte order mark only first",
            1,
            "a\n\xEF\xBB\xBF\n",
            "bytes [\"a\"; \"\\239\\187\\191\"]" );
          ("no input is no record", 1, "", "no rows");
        ]
        (fun (_, n, s, expected) -> equal text expected (decode (texts n) s));
      test "fails on errors as baselined" syntax_errors;
    ]

(* Values *)

let values =
  cases
    ~name:(fun (ty, s, _) ->
      Printf.sprintf "reads %s as %s" (String.escaped s)
        (Format.asprintf "%a" (fun ppf (Type.Any t) -> Type.pp ppf t) ty))
    "values"
    [
      (any Type.bool, "true\nfalse\n", "bool [true; false]");
      (any Type.int8, "-128\n127\n+5\n-0\n", "int8 [-128; 127; 5; 0]");
      (any Type.int16, "-32768\n32767\n", "int16 [-32768; 32767]");
      ( any Type.int32,
        "-2147483648\n2147483647\n",
        "int32 [-2147483648; 2147483647]" );
      (any Type.uint8, "0\n255\n-0\n", "uint8 [0; 255; 0]");
      (any Type.uint16, "65535\n", "uint16 [65535]");
      (any Type.uint32, "4294967295\n", "uint32 [4294967295]");
      ( any Type.int64,
        "-9223372036854775808\n9223372036854775807\n0009\n",
        "int64 [-9223372036854775808; 9223372036854775807; 9]" );
      ( any Type.uint64,
        "18446744073709551615\n9223372036854775808\n+1\n",
        "uint64 [18446744073709551615; 9223372036854775808; 1]" );
      ( any Type.float64,
        "1.5\n-0.25\n.5\n5.\n1e3\n1E-3\n+2\n0.1\n123456789012345678901\n",
        "float64 [1.5; -0.25; 0.5; 5; 1000; 0.001; 2; 0.1; \
         1.2345678901234568e+20]" );
      ( any Type.float64,
        "inf\n-Infinity\nNaN\n1e400\n-1e-400\n4.9e-324\n",
        "float64 [inf; -inf; nan; inf; -0; 4.94065645841247e-324]" );
      ( any Type.float64,
        "9007199254740993\n0.30000000000000004\n2.2250738585072011e-308\n",
        "float64 [9007199254740992; 0.30000000000000004; \
         2.225073858507201e-308]" );
      ( any Type.float32,
        "0.1\n16777217\n",
        "float32 [0.10000000149011612; 16777216]" );
      ( any Type.float16,
        "0.1\n65504\n1e5\n",
        "float16 [0.0999755859375; 65504; inf]" );
      ( any Type.float32,
        "1.00000005960464477550461637\n\
         1.000000059604644775390625\n\
         1.000000059604644775390624\n\
         -1.00000005960464477550461637\n",
        "float32 [1.0000001192092896; 1; 1; -1.0000001192092896]" );
      ( any Type.float32,
        "340282356779733661637539395458142568447\n\
         340282356779733661637539395458142568448\n\
         1e39\n",
        "float32 [3.4028234663852886e+38; inf; inf]" );
      ( any Type.float16,
        "1.00048828125000000001\n1.00048828125\n65519.99999\n65520\n",
        "float16 [1.0009765625; 1; 65504; inf]" );
      ( any (Type.decimal ~precision:5 ~scale:2),
        "123.45\n-1.5\n1.500\n0\n.5\n-0.00\n",
        "int64 [12345; -150; 150; 0; 50; 0]" );
      ( any (Type.decimal ~precision:18 ~scale:0),
        "999999999999999999\n",
        "int64 [999999999999999999]" );
      ( any Type.date,
        "1970-01-01\n2000-02-29\n0000-01-01\n9999-12-31\n1969-12-31\n",
        "int32 [0; 11016; -719528; 2932896; -1]" );
      ( any (Type.datetime Us),
        "1970-01-01T00:00:01\n\
         1970-01-01 00:00:00.5\n\
         1969-12-31T23:59:59.999999\n",
        "int64 [1000000; 500000; -1]" );
      ( any (Type.datetime Ms),
        "1970-01-01T00:00:00.001\n1970-01-01T00:00:00.100000\n",
        "int64 [1; 100]" );
      (any (Type.datetime S), "1970-01-02T00:00:00.000\n", "int64 [86400]");
      ( any (Type.datetime ~zone:"Europe/Paris" Ns),
        "1970-01-01T01:00:00+01:00\n\
         1970-01-01T00:00:00Z\n\
         1969-12-31T19:00:00-05:00\n",
        "int64 [0; 0; 0]" );
      ( any (Type.datetime ~zone:"UTC" Ns),
        "2262-04-11T23:47:16.854775807Z\n1677-09-21T00:12:43.145224192Z\n",
        "int64 [9223372036854775807; -9223372036854775808]" );
      ( any Type.string,
        "caf\xc3\xa9\n\"a\"\"b\"\n\"\"\n",
        "bytes [\"caf\\195\\169\"; \"a\\\"b\"; \"\"]" );
      (any Type.binary, "\xff\xfe\n", "bytes [\"\\255\\254\"]");
      ( any (Type.categorical [| "a"; "b\"c"; "" |]),
        "\"b\"\"c\"\na\n\"\"\n",
        "int32 [1; 0; 2]" );
    ]
    (fun (ty, s, expected) -> equal text expected (decode (columns [ ty ]) s))

let refusals () =
  List.iter
    (fun (ty, s) ->
      Printf.printf "%s as %s: %s\n" (String.escaped s)
        (Format.asprintf "%a" (fun ppf (Type.Any t) -> Type.pp ppf t) ty)
        (failure (columns [ ty ]) (s ^ "\n")))
    [
      (any Type.bool, "True");
      (any Type.bool, "1");
      (any Type.int8, "128");
      (any Type.int8, "-129");
      (any Type.int8, "1.0");
      (any Type.int8, " 1");
      (any Type.int8, "+");
      (any Type.uint8, "-1");
      (any Type.uint32, "4294967296");
      (any Type.int32, "99999999999999999999999");
      (any Type.int64, "9223372036854775808");
      (any Type.int64, "-9223372036854775809");
      (any Type.uint64, "18446744073709551616");
      (any Type.uint64, "-1");
      (any Type.float64, ".");
      (any Type.float64, "e5");
      (any Type.float64, "1e");
      (any Type.float64, "1.2.3");
      (any Type.float64, "0x10");
      (any Type.float64, "1_000");
      (any Type.float64, "nan1");
      (any Type.float64, "--1");
      (any (Type.decimal ~precision:5 ~scale:2), "1.234");
      (any (Type.decimal ~precision:5 ~scale:2), "1234.5");
      (any (Type.decimal ~precision:5 ~scale:2), "1e2");
      (any (Type.decimal ~precision:5 ~scale:2), ".");
      (any Type.date, "2023-02-29");
      (any Type.date, "2024-1-01");
      (any Type.date, "2024-13-01");
      (any Type.date, "2024-01-01T00:00:00");
      (any (Type.datetime Us), "2024-01-01T00:00:00.1234567");
      (any (Type.datetime Us), "2024-01-01T00:00:00Z");
      (any (Type.datetime Us), "2024-01-01T24:00:00");
      (any (Type.datetime Us), "2024-01-01T00:00:60");
      (any (Type.datetime Us), "2024-01-01T00:00:00.");
      (any (Type.datetime Us), "2024-01-01T00:00:00.1234567890");
      (any (Type.datetime Us), "2024-01-01t00:00:00");
      (any (Type.datetime S), "2024-01-01T00:00:00.5");
      (any (Type.datetime ~zone:"UTC" Us), "2024-01-01T00:00:00");
      (any (Type.datetime ~zone:"UTC" Us), "2024-01-01T00:00:00+0100");
      (any (Type.datetime ~zone:"UTC" Ns), "2262-04-11T23:47:16.854775808Z");
      (any (Type.datetime ~zone:"UTC" Ns), "1677-09-21T00:12:43.145224191Z");
      (any Type.string, "\xff");
      (any Type.string, "\xc3");
      (any (Type.categorical [| "a" |]), "b");
      (any (Type.datetime Ms), "2024-01-01T00:00:00.0015");
      (any (Type.datetime ~zone:"UTC" Us), "2024-01-01T00:00:00+24:00");
      (any (Type.datetime ~zone:"UTC" Us), "2024-01-01T00:00:00+01:60");
    ];
  expect (output ())
  @@ __POS_OF__
       {|
    True as bool: line 1, column 1: "True": column "c1": cannot read as bool: not true or false. Declare the null token (~nulls), or read the column as string (with_type) and map its spellings with an expression.
    1 as bool: line 1, column 1: "1": column "c1": cannot read as bool: not true or false. Declare the null token (~nulls), or read the column as string (with_type) and map its spellings with an expression.
    128 as int8: line 1, column 1: "128": column "c1": cannot read as int8: out of range. Declare the null token (~nulls) or another type (with_type).
    -129 as int8: line 1, column 1: "-129": column "c1": cannot read as int8: out of range. Declare the null token (~nulls) or another type (with_type).
    1.0 as int8: line 1, column 1: "1.0": column "c1": cannot read as int8: not an integer. Declare the null token (~nulls) or another type (with_type).
     1 as int8: line 1, column 1: " 1": column "c1": cannot read as int8: not an integer. Declare the null token (~nulls) or another type (with_type).
    + as int8: line 1, column 1: "+": column "c1": cannot read as int8: not an integer. Declare the null token (~nulls) or another type (with_type).
    -1 as uint8: line 1, column 1: "-1": column "c1": cannot read as uint8: out of range. Declare the null token (~nulls) or another type (with_type).
    4294967296 as uint32: line 1, column 1: "4294967296": column "c1": cannot read as uint32: out of range. Declare the null token (~nulls) or another type (with_type).
    99999999999999999999999 as int32: line 1, column 1: "99999999999999999999999": column "c1": cannot read as int32: out of range. Declare the null token (~nulls) or another type (with_type).
    9223372036854775808 as int64: line 1, column 1: "9223372036854775808": column "c1": cannot read as int64: out of range. Declare the null token (~nulls) or another type (with_type).
    -9223372036854775809 as int64: line 1, column 1: "-9223372036854775809": column "c1": cannot read as int64: out of range. Declare the null token (~nulls) or another type (with_type).
    18446744073709551616 as uint64: line 1, column 1: "18446744073709551616": column "c1": cannot read as uint64: out of range. Declare the null token (~nulls) or another type (with_type).
    -1 as uint64: line 1, column 1: "-1": column "c1": cannot read as uint64: out of range. Declare the null token (~nulls) or another type (with_type).
    . as float64: line 1, column 1: ".": column "c1": cannot read as float64: not a number. Declare the null token (~nulls) or another type (with_type).
    e5 as float64: line 1, column 1: "e5": column "c1": cannot read as float64: not a number. Declare the null token (~nulls) or another type (with_type).
    1e as float64: line 1, column 1: "1e": column "c1": cannot read as float64: not a number. Declare the null token (~nulls) or another type (with_type).
    1.2.3 as float64: line 1, column 1: "1.2.3": column "c1": cannot read as float64: not a number. Declare the null token (~nulls) or another type (with_type).
    0x10 as float64: line 1, column 1: "0x10": column "c1": cannot read as float64: not a number. Declare the null token (~nulls) or another type (with_type).
    1_000 as float64: line 1, column 1: "1_000": column "c1": cannot read as float64: not a number. Declare the null token (~nulls) or another type (with_type).
    nan1 as float64: line 1, column 1: "nan1": column "c1": cannot read as float64: not a number. Declare the null token (~nulls) or another type (with_type).
    --1 as float64: line 1, column 1: "--1": column "c1": cannot read as float64: not a number. Declare the null token (~nulls) or another type (with_type).
    1.234 as decimal[5, 2]: line 1, column 1: "1.234": column "c1": cannot read as decimal[5, 2]: more than 2 digits after the point. Declare the null token (~nulls) or another type (with_type).
    1234.5 as decimal[5, 2]: line 1, column 1: "1234.5": column "c1": cannot read as decimal[5, 2]: more than 5 digits. Declare the null token (~nulls) or another type (with_type).
    1e2 as decimal[5, 2]: line 1, column 1: "1e2": column "c1": cannot read as decimal[5, 2]: not a decimal number. Declare the null token (~nulls) or another type (with_type).
    . as decimal[5, 2]: line 1, column 1: ".": column "c1": cannot read as decimal[5, 2]: not a decimal number. Declare the null token (~nulls) or another type (with_type).
    2023-02-29 as date: line 1, column 1: "2023-02-29": column "c1": cannot read as date: not a day of the calendar. Declare the null token (~nulls) or another type (with_type).
    2024-1-01 as date: line 1, column 1: "2024-1-01": column "c1": cannot read as date: not YYYY-MM-DD. Declare the null token (~nulls) or another type (with_type).
    2024-13-01 as date: line 1, column 1: "2024-13-01": column "c1": cannot read as date: not a day of the calendar. Declare the null token (~nulls) or another type (with_type).
    2024-01-01T00:00:00 as date: line 1, column 1: "2024-01-01T00:00:00": column "c1": cannot read as date: not YYYY-MM-DD. Declare the null token (~nulls) or another type (with_type).
    2024-01-01T00:00:00.1234567 as datetime[us]: line 1, column 1: "2024-01-01T00:00:00.1234567": column "c1": cannot read as datetime[us]: not a whole number of microseconds. Declare the null token (~nulls) or another type (with_type).
    2024-01-01T00:00:00Z as datetime[us]: line 1, column 1: "2024-01-01T00:00:00Z": column "c1": cannot read as datetime[us]: not YYYY-MM-DDThh:mm:ss without an offset. Declare the null token (~nulls) or another type (with_type).
    2024-01-01T24:00:00 as datetime[us]: line 1, column 1: "2024-01-01T24:00:00": column "c1": cannot read as datetime[us]: not a time of day. Declare the null token (~nulls) or another type (with_type).
    2024-01-01T00:00:60 as datetime[us]: line 1, column 1: "2024-01-01T00:00:60": column "c1": cannot read as datetime[us]: not a time of day. Declare the null token (~nulls) or another type (with_type).
    2024-01-01T00:00:00. as datetime[us]: line 1, column 1: "2024-01-01T00:00:00.": column "c1": cannot read as datetime[us]: not YYYY-MM-DDThh:mm:ss without an offset. Declare the null token (~nulls) or another type (with_type).
    2024-01-01T00:00:00.1234567890 as datetime[us]: line 1, column 1: "2024-01-01T00:00:00.1234567890": column "c1": cannot read as datetime[us]: not YYYY-MM-DDThh:mm:ss without an offset. Declare the null token (~nulls) or another type (with_type).
    2024-01-01t00:00:00 as datetime[us]: line 1, column 1: "2024-01-01t00:00:00": column "c1": cannot read as datetime[us]: not YYYY-MM-DDThh:mm:ss without an offset. Declare the null token (~nulls) or another type (with_type).
    2024-01-01T00:00:00.5 as datetime[s]: line 1, column 1: "2024-01-01T00:00:00.5": column "c1": cannot read as datetime[s]: not a whole number of seconds. Declare the null token (~nulls) or another type (with_type).
    2024-01-01T00:00:00 as datetime[us, UTC]: line 1, column 1: "2024-01-01T00:00:00": column "c1": cannot read as datetime[us, UTC]: not YYYY-MM-DDThh:mm:ss with an offset. Declare the null token (~nulls) or another type (with_type).
    2024-01-01T00:00:00+0100 as datetime[us, UTC]: line 1, column 1: "2024-01-01T00:00:00+0100": column "c1": cannot read as datetime[us, UTC]: not YYYY-MM-DDThh:mm:ss with an offset. Declare the null token (~nulls) or another type (with_type).
    2262-04-11T23:47:16.854775808Z as datetime[ns, UTC]: line 1, column 1: "2262-04-11T23:47:16.854775808Z": column "c1": cannot read as datetime[ns, UTC]: out of range. Declare the null token (~nulls) or another type (with_type).
    1677-09-21T00:12:43.145224191Z as datetime[ns, UTC]: line 1, column 1: "1677-09-21T00:12:43.145224191Z": column "c1": cannot read as datetime[ns, UTC]: out of range. Declare the null token (~nulls) or another type (with_type).
    \255 as string: line 1, column 1: "\xff": column "c1": cannot read as string: not valid UTF-8. Read the column as binary (with_type).
    \195 as string: line 1, column 1: "\xc3": column "c1": cannot read as string: not valid UTF-8. Read the column as binary (with_type).
    b as categorical["a"]: line 1, column 1: "b": column "c1": cannot read as categorical["a"]: not in the dictionary. Declare the null token (~nulls) or another type (with_type).
    2024-01-01T00:00:00.0015 as datetime[ms]: line 1, column 1: "2024-01-01T00:00:00.0015": column "c1": cannot read as datetime[ms]: not a whole number of milliseconds. Declare the null token (~nulls) or another type (with_type).
    2024-01-01T00:00:00+24:00 as datetime[us, UTC]: line 1, column 1: "2024-01-01T00:00:00+24:00": column "c1": cannot read as datetime[us, UTC]: not an offset. Declare the null token (~nulls) or another type (with_type).
    2024-01-01T00:00:00+01:60 as datetime[us, UTC]: line 1, column 1: "2024-01-01T00:00:00+01:60": column "c1": cannot read as datetime[us, UTC]: not an offset. Declare the null token (~nulls) or another type (with_type).
    |}

let nulls =
  group "nulls"
    [
      test "a null token is null, quoted or not empty" (fun () ->
          equal text "int64 [1; ∅; ∅]\nbytes [∅; \"NA\"; \"N/A\"]"
            (decode
               (columns ~nulls:[ "NA" ] [ any Type.int64; any Type.string ])
               "1,NA\nNA,\"NA\"\n,N/A\n"));
      test "a null holds zero, or the empty byte string" (fun () ->
          match
            read (columns [ any Type.int64; any Type.string ]) "7,a\n,\n"
          with
          | Ok [ [| Fixed { values = Nx.P x; _ }; Varsize { offsets; _ } |] ] ->
              equal (list int64) [ 7L; 0L ]
                (Array.to_list (Nx.to_array (Nx.cast Nx.int64 x)));
              equal (list int64) [ 0L; 1L; 1L ]
                (Array.to_list (Nx.to_array offsets))
          | _ -> fail "not one batch of a fixed and a varsize column");
      test "every row valid has no validity" (fun () ->
          match read (columns [ any Type.int64 ]) "1\n2\n" with
          | Ok [ [| Fixed { valid = None; _ } |] ] -> ()
          | _ -> fail "a validity mask");
    ]

let data_errors () =
  let sniffed_f = sniffed ~rows:2 "dep_delay,carrier\n1,AA\n2,B6\nNA,F9\n" in
  let s = "dep_delay,carrier\n1,AA\n2,B6\nNA,F9\n" in
  List.iter
    (fun (name, f, s) -> Printf.printf "%s: %s\n" name (failure f s))
    [
      ("sniffed", sniffed_f, s);
      ("declared", Csv.with_type "dep_delay" (any Type.float64) sniffed_f, s);
      ("range", columns [ any Type.int8 ], "1\n\"300\"\n");
      ("utf-8", columns [ any Type.string ], "a\n\"b\nc\xff\"\n");
      ( "earliest row first",
        columns [ any Type.int8; any Type.int8 ],
        "1,1\n2,x\ny,3\n" );
      ( "value before syntax",
        columns [ any Type.int8; any Type.int8 ],
        "1,1\n2,x\n3,\"4\n" );
      ("categorical", columns [ any (Type.categorical [| "a" |]) ], "a\nb\n");
      ( "leftmost column first",
        columns [ any Type.int8; any Type.int8 ],
        "x,y\n" );
      ( "earlier row first",
        columns [ any Type.int8; any Type.int8 ],
        "x,1\n2,y\n" );
      ("sniffed from one row", sniffed ~rows:1 "x\n1\nNA\n", "x\n1\nNA\n");
      ("bool spelled True", columns [ any Type.bool ], "true\nTrue\n");
      ("sniffed string not UTF-8", sniffed "x\na\n", "x\na\n\xff\n");
    ];
  expect (output ())
  @@ __POS_OF__
       {|
    sniffed: line 4, column 1: "NA": column "dep_delay": cannot read as int64: not an integer. The type was sniffed from rows 1 to 2. Declare the null token (~nulls) or the type (with_type).
    declared: line 4, column 1: "NA": column "dep_delay": cannot read as float64: not a number. Declare the null token (~nulls) or another type (with_type).
    range: line 2, column 1: "300": column "c1": cannot read as int8: out of range. Declare the null token (~nulls) or another type (with_type).
    utf-8: line 2, column 1: "b\x0ac\xff": column "c1": cannot read as string: not valid UTF-8. Read the column as binary (with_type).
    earliest row first: line 2, column 3: "x": column "c2": cannot read as int8: not an integer. Declare the null token (~nulls) or another type (with_type).
    value before syntax: line 2, column 3: "x": column "c2": cannot read as int8: not an integer. Declare the null token (~nulls) or another type (with_type).
    categorical: line 2, column 1: "b": column "c1": cannot read as categorical["a"]: not in the dictionary. Declare the null token (~nulls) or another type (with_type).
    leftmost column first: line 1, column 1: "x": column "c1": cannot read as int8: not an integer. Declare the null token (~nulls) or another type (with_type).
    earlier row first: line 1, column 1: "x": column "c1": cannot read as int8: not an integer. Declare the null token (~nulls) or another type (with_type).
    sniffed from one row: line 3, column 1: "NA": column "x": cannot read as int64: not an integer. The type was sniffed from row 1. Declare the null token (~nulls) or the type (with_type).
    bool spelled True: line 2, column 1: "True": column "c1": cannot read as bool: not true or false. Declare the null token (~nulls), or read the column as string (with_type) and map its spellings with an expression.
    sniffed string not UTF-8: line 3, column 1: "\xff": column "x": cannot read as string: not valid UTF-8. Read the column as binary (with_type).
    |}

(* Batches *)

let big_rows = 200_000

(* About 2.4 MiB of records, each a quoted field with a line break and a
   number. *)
let big =
  let b = Buffer.create (big_rows * 13) in
  for i = 0 to big_rows - 1 do
    Printf.bprintf b "\"r\n%d\",%d\n" i i
  done;
  Buffer.contents b

let batches =
  let f = columns [ any Type.string; any Type.int64 ] in
  group "batches"
    [
      test "split a large input, keeping every record in order" (fun () ->
          let bs = Error.get_ok (read f big) in
          greater int ~than:1 (List.length bs);
          let ints =
            List.concat_map
              (fun b ->
                match b.(1) with
                | Csv.Private.Fixed { values = Nx.P x; _ } ->
                    Array.to_list (Nx.to_array (Nx.cast Nx.int64 x))
                | Varsize _ -> fail "a varsize column")
              bs
          in
          equal (list int64) (List.init big_rows Int64.of_int) ints);
      cases ~name:(Printf.sprintf "are the same in slices of %d bytes")
        "slicing" [ 7; 4096; 100_000 ] (fun slice_length ->
          let rows r =
            List.map
              (fun b -> (fst (cells b.(1)), List.length (snd (cells b.(1)))))
              r
          in
          equal
            (list (pair string int))
            (rows (Error.get_ok (read f big)))
            (rows (Error.get_ok (read ~slice_length f big))));
      test "keep a quoted line feed at the 1 MiB edge in its field" (fun () ->
          (* The quoted line feed is the input's byte 2^20 - 1, the first that
             may end a batch. *)
          let s =
            "xx\n"
            ^ String.concat "" (List.init 524285 (fun _ -> "x\n"))
            ^ "\"a\nb\"\nz\n"
          in
          equal int ((1 lsl 20) - 1) (String.index_from s 1048574 '\n');
          let rows =
            List.concat_map
              (fun b -> strings b.(0))
              (Error.get_ok (read (texts 1) s))
          in
          equal int 524288 (List.length rows);
          equal
            (list (option string))
            [ Some "x"; Some "a\nb"; Some "z" ]
            (List.filteri (fun i _ -> i >= 524285) rows));
      test "count lines across batches" (fun () ->
          expect (failure f (big ^ "x,y\n"))
          @@ __POS_OF__
               {| line 400001, column 3: "y": column "c2": cannot read as int64: not an integer. Declare the null token (~nulls) or another type (with_type). |});
    ]

(* Properties *)

(* A table of [cols] binary columns encoded by hand: a null is an unquoted empty
   field, or the null token [NA] when it is a record's only field, a field is
   quoted when it must be or when [quote] says, each record ends in a line feed
   or a carriage return and a line feed, the last one maybe not at all, and an
   empty line may precede a record. *)
type table = {
  cols : int;
  rows : (string option * bool) list list;
  crlf : bool list;
  blank : bool list;
  last : bool;
  slice : int;
}

let encode t =
  let b = Buffer.create 256 in
  let field ~alone (v, quote) =
    match v with
    | None -> if alone then Buffer.add_string b "NA"
    | Some s ->
        let must =
          s = ""
          || String.exists
               (fun c -> c = ',' || c = '"' || c = '\n' || c = '\r')
               s
        in
        if must || quote then begin
          Buffer.add_char b '"';
          String.iter
            (fun c ->
              if c = '"' then Buffer.add_string b "\"\""
              else Buffer.add_char b c)
            s;
          Buffer.add_char b '"'
        end
        else Buffer.add_string b s
  in
  let n = List.length t.rows in
  List.iteri
    (fun i ((row, crlf), blank) ->
      let break = if crlf then "\r\n" else "\n" in
      if blank then Buffer.add_string b break;
      List.iteri
        (fun j f ->
          if j > 0 then Buffer.add_char b ',';
          field ~alone:(t.cols = 1) f)
        row;
      if i < n - 1 || t.last then Buffer.add_string b break)
    (List.combine (List.combine t.rows t.crlf) t.blank);
  Buffer.contents b

let table =
  let open Gen in
  let* cols = int_range 1 4 in
  let* n = int_range 0 12 in
  let byte = of_list [ 'a'; 'b'; ' '; ','; '"'; '\n'; '\r'; '\xff' ] in
  let cell = pair (option (string_of ~size:(int_range 0 5) byte)) bool in
  let* rows = list ~size:(constant n) (list ~size:(constant cols) cell) in
  let* crlf = list ~size:(constant n) bool in
  let* blank = list ~size:(constant n) bool in
  let* last = bool in
  let+ slice = int_range 1 16 in
  { cols; rows; crlf; blank; last; slice }

let pp_table ppf t = Format.fprintf ppf "%S (slices of %d)" (encode t) t.slice

let round_trip =
  prop "hand-encoded records read back as their fields"
    (Gen.with_pp pp_table table) (fun t ->
      let f =
        columns ~nulls:[ "NA" ] (List.init t.cols (fun _ -> any Type.binary))
      in
      let got =
        Error.get_ok (read ~slice_length:t.slice f (encode t))
        |> List.concat_map (fun b ->
            let columns = Array.to_list (Array.map strings b) in
            List.init
              (List.length (List.hd columns))
              (fun i -> List.map (fun c -> List.nth c i) columns))
      in
      equal (list (list (option string))) (List.map (List.map fst) t.rows) got)

let float_texts =
  let open Gen in
  let digits = string_of ~size:(int_range 0 22) (char_range '0' '9') in
  let* int = digits in
  let* frac = digits in
  let* exp = option (int_range (-340) 330) in
  let int = if int = "" && frac = "" then "0" else int in
  constant
    (int
    ^ (if frac = "" then "" else "." ^ frac)
    ^ match exp with None -> "" | Some e -> "e" ^ string_of_int e)

let floats =
  group "float64"
    [
      prop "reads a number as float_of_string does"
        (Gen.with_pp Format.pp_print_string float_texts) (fun s ->
          equal int64
            (Int64.bits_of_float (float_of_string s))
            (Int64.bits_of_float (read_float s)));
      prop "reads a float printed with 17 digits as itself" Gen.float (fun x ->
          equal int64 (Int64.bits_of_float x)
            (Int64.bits_of_float (read_float (Printf.sprintf "%.17g" x))));
    ]

let ints =
  prop "reads an int64 printed in decimal as itself" Gen.int64 (fun x ->
      equal text
        (Printf.sprintf "int64 [%Ld]" x)
        (decode (columns [ any Type.int64 ]) (Int64.to_string x)))

let dates =
  prop "reads a date as Time.Date counts its days"
    Gen.(int_range (-719528) 2932896)
    (fun days ->
      let d = Option.get (Time.Date.of_days days) in
      let y, m, dd = Time.Date.to_civil d in
      equal text
        (Printf.sprintf "int32 [%d]" days)
        (decode
           (columns [ any Type.date ])
           (Printf.sprintf "%04d-%02d-%02d" y m dd)))

let () =
  exit
    (run "talon.next.csv"
       [
         formats;
         with_types;
         sniffing;
         syntax;
         values;
         test "refusals are as baselined" refusals;
         nulls;
         test "data errors are as baselined" data_errors;
         batches;
         round_trip;
         floats;
         ints;
         dates;
       ])
