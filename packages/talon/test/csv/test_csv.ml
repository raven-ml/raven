(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module Csv = Talon_csv
module Reader = Bytesrw.Bytes.Reader
module Writer = Bytesrw.Bytes.Writer
module G = Talon_gen

(* Helpers *)

let any t = Type.Any t
let pp_any ppf (Type.Any t) = Type.pp ppf t
let type_name t = Format.asprintf "%a" pp_any t
let error_text e = Format.asprintf "%a" Error.pp e
let format_text f = Format.asprintf "%a" Csv.pp_format f

let sniffed ?rows ?nulls s =
  Error.get_ok (Csv.sniff ?rows ?nulls (Reader.of_string s))

let read ?slice_length f s = Csv.decode f (Reader.of_string ?slice_length s)
let failure f s = match read f s with Ok _ -> "ok" | Error e -> error_text e

(* [columns types] is a format without a header whose columns [c1], [c2], … read
   as [types]. *)
let columns ?nulls types =
  Csv.format ~header:false ?nulls
    (List.mapi (fun i t -> (Printf.sprintf "c%d" (i + 1), t)) types)

(* [bytes n] is a format of [n] binary columns, which read any field. *)
let bytes ?nulls n = columns ?nulls (List.init n (fun _ -> any Type.binary))

(* [fields t] is the rows of [t]'s binary columns, [None] at a null. *)
let fields t =
  let bytes (name, _) =
    Array.map
      (Option.map (fun (b : Binary.t) -> (b :> string)))
      (Column.options Kind.binary (column t name))
  in
  let cs = List.map bytes (Schema.columns (schema t)) in
  List.init (rows t) (fun i -> List.map (fun c -> c.(i)) cs)

let decoded ?slice_length f s = fields (Error.get_ok (read ?slice_length f s))
let records = list (list (option string))

(* [table] compares tables column by column, floats by their bits, whatever
   their batches. *)
let table =
  let same a b =
    let (Type.Any ty) = Column.type_ a in
    let rows c = Column.options (Type.kind ty) c in
    Testable.equal (array (option (G.witness ty))) (rows a) (rows b)
  in
  let equal a b =
    Schema.equal (schema a) (schema b)
    && rows a = rows b
    && List.for_all
         (fun (n, _) -> same (column a n) (column b n))
         (Schema.columns (schema a))
  in
  Testable.make ~pp:Talon.pp ~equal

(* The types CSV reads ({!Csv.format}), each width and unit. *)
let csv_types =
  Type.
    [
      Any bool;
      Any int8;
      Any int16;
      Any int32;
      Any int64;
      Any uint8;
      Any uint16;
      Any uint32;
      Any uint64;
      Any float16;
      Any float32;
      Any float64;
      Any string;
      Any binary;
      Any (categorical [| "a"; ""; "b\"c"; "x,y"; "l\nm"; "NA"; "café" |]);
      Any date;
      Any (wall S);
      Any (wall Ms);
      Any (wall Us);
      Any (wall Ns);
      Any (instant S);
      Any (instant Ms);
      Any (instant Us);
      Any (instant Ns);
    ]

(* The types CSV does not read. *)
let other_types =
  Type.
    [
      Any (clock Us);
      Any (duration Ms);
      Any (list int64);
      Any (record [ ("a", Any int64) ]);
      Any (tensor Nx.float32 [| 2 |]);
      Any (ext ~name:"units.mass" float64);
    ]

(* Fields written by hand, under the null token NA *)

(* [quoted s] is [s] between double quotes, its quotes doubled. *)
let quoted s = "\"" ^ String.concat "\"\"" (String.split_on_char '"' s) ^ "\""

(* [must_quote s] is [true] iff [s] read unquoted would not give [s] back. *)
let must_quote s =
  s = "" || s = "NA"
  || String.starts_with ~prefix:"\xEF\xBB\xBF" s
  || String.exists (fun c -> c = ',' || c = '"' || c = '\n' || c = '\r') s

(* [field (v, quote)] is the field that holds [v], quoted when it must be or
   when [quote] says, and empty for [None]. *)
let field (v, quote) =
  match v with
  | None -> ""
  | Some s -> if quote || must_quote s then quoted s else s

(* Syntax *)

(* A file of [cols] binary columns written by hand: each record ends in a line
   feed or a carriage return and a line feed, the last one maybe not at all, an
   empty line may precede a record, a byte order mark may start the input, and a
   one-column record writes a null as NA. *)
type file = {
  cols : int;
  rows : (string option * bool) list list;
  crlf : bool list;
  blank : bool list;
  last : bool;
  bom : bool;
  slice : int;
}

let encode t =
  let b = Buffer.create 256 in
  if t.bom then Buffer.add_string b "\xEF\xBB\xBF";
  let n = List.length t.rows in
  List.iteri
    (fun i ((row, crlf), blank) ->
      let break = if crlf then "\r\n" else "\n" in
      if blank then Buffer.add_string b break;
      let fields = List.map field row in
      let fields = if fields = [ "" ] then [ "NA" ] else fields in
      Buffer.add_string b (String.concat "," fields);
      if i < n - 1 || t.last then Buffer.add_string b break)
    (List.combine (List.combine t.rows t.crlf) t.blank);
  Buffer.contents b

let file =
  let open Gen in
  let* cols = int_range 1 4 in
  let* n = int_range 0 12 in
  let byte = of_list [ 'a'; 'b'; ' '; ','; '"'; '\n'; '\r'; '\xff' ] in
  let cell = pair (option (string_of ~size:(int_range 0 5) byte)) bool in
  let* rows = list ~size:(constant n) (list ~size:(constant cols) cell) in
  let* crlf = list ~size:(constant n) bool in
  let* blank = list ~size:(constant n) bool in
  let* last = bool in
  let* bom = frequency [ (3, constant false); (1, constant true) ] in
  let+ slice = int_range 1 16 in
  { cols; rows; crlf; blank; last; bom; slice }

let pp_file ppf t = Format.fprintf ppf "%S (slices of %d)" (encode t) t.slice

(* A byte order mark read in slices of one byte, and a null in one column. *)
let bom_in_slices =
  {
    cols = 1;
    rows = [ [ (Some "a", false) ]; [ (None, false) ] ];
    crlf = [ false; false ];
    blank = [ false; false ];
    last = true;
    bom = true;
    slice = 1;
  }

let reads_its_fields t =
  let cells = List.concat t.rows in
  let holds p =
    List.exists (fun (v, _) -> Option.fold ~none:false ~some:p v) cells
  in
  cover "a byte order mark" t.bom;
  cover "a byte order mark in slices shorter than it" (t.bom && t.slice < 3);
  cover "a record that ends at the end of the input"
    ((not t.last) && t.rows <> []);
  cover "an empty line" (List.mem true t.blank);
  cover "a quoted line break"
    (holds (String.exists (fun c -> c = '\n' || c = '\r')));
  cover "a doubled quote" (holds (fun s -> String.contains s '"'));
  cover "a null token in a one-column file"
    (t.cols = 1 && List.exists (fun (v, _) -> v = None) cells);
  equal records
    (List.map (List.map fst) t.rows)
    (decoded ~slice_length:t.slice (bytes ~nulls:[ "NA" ] t.cols) (encode t))

let wide_record = String.concat "," (List.init 5000 (fun _ -> "y"))
let some_rows = List.map (List.map Option.some)

let syntax =
  group "syntax"
    [
      cases
        "a record ends at a line feed, a carriage return and a line feed, or \
         the end of the input"
        ~id:(fun (name, _, _) -> name)
        [
          ("a line feed", "a,b\nc,d\n", [ [ "a"; "b" ]; [ "c"; "d" ] ]);
          ("a CRLF", "a,b\r\nc,d\r\n", [ [ "a"; "b" ]; [ "c"; "d" ] ]);
          ("the end of the input", "a,b\nc,d", [ [ "a"; "b" ]; [ "c"; "d" ] ]);
          ( "a quoted field at the end of the input",
            "\"a\",\"b\"",
            [ [ "a"; "b" ] ] );
        ]
        (fun (_, s, expected) ->
          equal records (some_rows expected) (decoded (bytes 2) s));
      test "a separator before a line break ends an empty field" (fun () ->
          equal records
            [ [ Some "a"; None ]; [ Some "b"; None ] ]
            (decoded (bytes 2) "a,\nb,"));
      cases "an empty line is no record, wherever it is"
        ~id:(fun (name, _, _, _) -> name)
        [
          ("in one column", 1, "a\n\nb\n", [ [ "a" ]; [ "b" ] ]);
          ("between records", 2, "a,b\n\nc,d\n", [ [ "a"; "b" ]; [ "c"; "d" ] ]);
          ("at the end", 2, "a,b\n\n\n", [ [ "a"; "b" ] ]);
          ("of CRLFs", 1, "\r\na\r\n\r\n\nb\r\n\r\n", [ [ "a" ]; [ "b" ] ]);
          ("on every line", 1, "\n\r\n\n", []);
          ("in no input", 1, "", []);
        ]
        (fun (_, n, s, expected) ->
          equal records (some_rows expected) (decoded (bytes n) s));
      test "an empty line before a header is no record" (fun () ->
          let f =
            Csv.format [ ("a", any Type.binary); ("b", any Type.int64) ]
          in
          (* Two empty lines, the header, then [y,z] on line 5. *)
          expect (failure f "\r\n\na,b\nx,1\ny,z\n")
          @@ __POS_OF__
               {| line 5, column 3: "z": column "b": cannot read as int64: not an integer. Declare the null token (~nulls) or another type (with_type). |});
      cases "a quoted field holds the separator, line breaks and quotes"
        ~id:(fun (name, _, _) -> name)
        [
          ("the separator", "\"a,b\"\n", "a,b");
          ("line breaks", "\"a\nb\r\nc\"\n", "a\nb\r\nc");
          ("a doubled quote as one quote", "\"a\"\"b\"\"\"\n", "a\"b\"");
          ("nothing, the empty string", "\"\"\n", "");
        ]
        (fun (_, s, expected) ->
          equal records [ [ Some expected ] ] (decoded (bytes 1) s));
      test "an unquoted field keeps its spaces" (fun () ->
          equal records
            [ [ Some " a "; Some " b " ] ]
            (decoded (bytes 2) " a , b \n"));
      cases "a byte order mark is skipped at the start of the input only"
        ~id:(fun (name, _, _) -> name)
        [
          ("at the start", "\xEF\xBB\xBFa\n", [ "a" ]);
          ("on a later line", "a\n\xEF\xBB\xBF\n", [ "a"; "\xEF\xBB\xBF" ]);
          ("in part", "\xEF\xBBa\n", [ "\xEF\xBBa" ]);
        ]
        (fun (_, s, expected) ->
          equal records
            (List.map (fun s -> [ Some s ]) expected)
            (decoded (bytes 1) s));
      prop ~count:200 ~examples:[ bom_in_slices ]
        "records read as a writer by hand writes them"
        (Gen.with_pp pp_file file) reads_its_fields;
      cases "decode fails at the first byte that breaks the syntax"
        ~id:(fun (name, _, _, _) -> name)
        [
          ( "a quote in an unquoted field",
            2,
            "a,b\"c\n",
            __POS_OF__
              {| line 1, column 4: a quote in a field that does not start with one |}
          );
          ( "a byte after a closing quote",
            2,
            "a,\"b\"c\n",
            __POS_OF__
              {| line 1, column 6: a quoted field's closing quote is followed by a byte other than a separator or a line break |}
          );
          ( "a quoted field that does not end, at its quote",
            2,
            "a,b\n\"c\nd\n,e\n",
            __POS_OF__ {| line 2, column 1: a quoted field does not end |} );
          ( "a carriage return alone",
            2,
            "a,b\rc\n",
            __POS_OF__
              {| line 1, column 4: a carriage return that no line feed follows |}
          );
          ( "a carriage return at the end",
            1,
            "a\r",
            __POS_OF__
              {| line 1, column 2: a carriage return that no line feed follows |}
          );
          ( "a column after a byte order mark",
            2,
            "\xEF\xBB\xBFa,\"b\"c\n",
            __POS_OF__
              {| line 1, column 6: a quoted field's closing quote is followed by a byte other than a separator or a line break |}
          );
        ]
        (fun (_, n, s, baseline) -> expect (failure (bytes n) s) baseline);
      cases "decode fails at a record with another number of fields"
        ~id:(fun (name, _, _, _) -> name)
        [
          ( "too many",
            2,
            "a,b\nc,d,e\n",
            __POS_OF__
              {| line 2, column 5: the record has 3 fields, and the format 2 columns |}
          );
          ( "too few",
            2,
            "a,b\nc\n",
            __POS_OF__
              {| line 2, column 2: the record has 1 field, and the format 2 columns |}
          );
          ( "after a quoted line break",
            2,
            "\"a\nb\",c\nd,e,f\n",
            __POS_OF__
              {| line 3, column 5: the record has 3 fields, and the format 2 columns |}
          );
          ( "wider than the first",
            1,
            "a\n" ^ wide_record ^ "\n",
            __POS_OF__
              {| line 2, column 3: the record has 5000 fields, and the format 1 column |}
          );
        ]
        (fun (_, n, s, baseline) -> expect (failure (bytes n) s) baseline);
    ]

(* Values *)

(* The texts of a column's rows, [None] at a null, and whether each is quoted
   when it need not be. *)
type texts = Texts : 'a Type.t * string option array * bool list -> texts

(* [print c] is the text of each row of [c], as {!Column.print} writes it. *)
let print c =
  let p = Column.print c in
  match Column.type_ p with
  | Type.Any Binary ->
      Array.map
        (Option.map (fun (b : Binary.t) -> (b :> string)))
        (Column.options Kind.binary p)
  | _ -> Column.options Kind.string p

(* Text that most types do not read: no type but binary reads [\xff], [300] is
   out of [int8]'s range. *)
let junk = [ "x"; "300"; "1e"; "\xff"; "true1" ]

let texts =
  let open Gen in
  let* (Type.Any ty) = of_list ~pp:pp_any csv_types in
  let* vs = G.options ty in
  let texts = print (Column.of_options ty vs) in
  let n = Array.length texts in
  let* bad =
    if n = 0 then constant None
    else
      frequency
        [
          (3, constant None);
          (1, map Option.some (pair (int_range 0 (n - 1)) (of_list junk)));
        ]
  in
  let texts = Array.copy texts in
  Option.iter (fun (i, s) -> texts.(i) <- Some s) bad;
  let+ quotes = list ~size:(constant n) bool in
  Texts (ty, texts, quotes)

let pp_texts ppf (Texts (ty, texts, _)) =
  Format.fprintf ppf "%a [%s]" Type.pp ty
    (String.concat "; "
       (Array.to_list
          (Array.map
             (function None -> "null" | Some s -> Printf.sprintf "%S" s)
             texts)))

(* One column, a null written NA. The line of record [r] is one more than the
   line feeds before it: each record's own, and those its quoted field holds. *)
let reads_as_parse (Texts (ty, texts, quotes)) =
  List.iter
    (fun t -> cover (type_name t) (type_name t = type_name (any ty)))
    csv_types;
  let rows = List.combine (Array.to_list texts) quotes in
  let record (v, q) = if v = None then "NA\n" else field (v, q) ^ "\n" in
  let line_feeds s = List.length (String.split_on_char '\n' s) - 1 in
  let line r =
    List.fold_left
      (fun n row -> n + line_feeds (record row))
      1
      (List.filteri (fun i _ -> i < r) rows)
  in
  let expected =
    Column.parse (any ty)
      (Column.of_options Type.binary
         (Array.map (Option.map Binary.of_string) texts))
  in
  cover "a field that does not read" (Result.is_error expected);
  let decoded =
    read
      (columns ~nulls:[ "NA" ] [ any ty ])
      (String.concat "" (List.map record rows))
  in
  match (expected, decoded) with
  | Ok c, Ok t -> equal table (Talon.v [ ("c1", c) ]) t
  | Error (r, _), Error e ->
      starts_with
        ~prefix:(Printf.sprintf "line %d, column 1:" (line r))
        (error_text e)
  | Ok _, Error e ->
      failf "Column.parse reads the texts, decode fails: %s" (error_text e)
  | Error (r, reason), Ok _ ->
      failf "Column.parse fails at row %d (%s), decode reads the texts" r reason

(* A dictionary's text holding a doubled quote. *)
let quoted_categorical =
  Texts
    ( Type.categorical [| "a"; "b\"c" |],
      [| Some "b\"c"; Some "a" |],
      [ true; false ] )

let delays = "dep_delay,carrier\n1,AA\n2,B6\nNA,F9\n"
let delays_sniffed = lazy (sniffed ~rows:2 delays)

let values =
  group "values"
    [
      prop ~count:600 ~examples:[ quoted_categorical ]
        "a field reads as Column.parse reads its text"
        (Gen.with_pp pp_texts texts)
        reads_as_parse;
      cases
        "an unquoted empty field or null token is null, and a quoted one is \
         text"
        ~id:(fun (name, _, _) -> name)
        [
          ("an unquoted empty field", ",x\n", None);
          ("an unquoted null token", "NA,x\n", None);
          ("a quoted null token", "\"NA\",x\n", Some "NA");
          ("a quoted empty field", "\"\",x\n", Some "");
          ("a field that starts as a null token", "NAN,x\n", Some "NAN");
        ]
        (fun (_, s, expected) ->
          equal records
            [ [ expected; Some "x" ] ]
            (decoded (bytes ~nulls:[ "NA" ] 2) s));
      cases
        "a field that does not read fails with its line, column, text and fix"
        ~id:(fun (name, _, _, _) -> name)
        [
          ( "a sniffed type",
            delays_sniffed,
            delays,
            __POS_OF__
              {| line 4, column 1: "NA": column "dep_delay": cannot read as int64: not an integer. The type was sniffed from rows 1 to 2. Declare the null token (~nulls) or the type (with_type). |}
          );
          ( "a type sniffed from one row",
            lazy (sniffed ~rows:1 "x\n1\nNA\n"),
            "x\n1\nNA\n",
            __POS_OF__
              {| line 3, column 1: "NA": column "x": cannot read as int64: not an integer. The type was sniffed from row 1. Declare the null token (~nulls) or the type (with_type). |}
          );
          ( "a declared type",
            lazy
              (Csv.with_type "dep_delay" (any Type.float64)
                 (Lazy.force delays_sniffed)),
            delays,
            __POS_OF__
              {| line 4, column 1: "NA": column "dep_delay": cannot read as float64: not a number. Declare the null token (~nulls) or another type (with_type). |}
          );
          ( "a quoted value out of range",
            lazy (columns [ any Type.int8 ]),
            "1\n\"300\"\n",
            __POS_OF__
              {| line 2, column 1: "300": column "c1": cannot read as int8: out of range. Declare the null token (~nulls) or another type (with_type). |}
          );
          ( "text that is not UTF-8",
            lazy (columns [ any Type.string ]),
            "a\n\"b\nc\xff\"\n",
            __POS_OF__
              {| line 2, column 1: "b\x0ac\xff": column "c1": cannot read as string: invalid UTF-8 at byte 3. Read the column as binary (with_type). |}
          );
          ( "sniffed text that is not UTF-8",
            lazy (sniffed "x\na\n"),
            "x\na\n\xff\n",
            __POS_OF__
              {| line 3, column 1: "\xff": column "x": cannot read as string: invalid UTF-8 at byte 0. Read the column as binary (with_type). |}
          );
          ( "text outside a categorical's dictionary",
            lazy (columns [ any (Type.categorical [| "a" |]) ]),
            "a\nb\n",
            __POS_OF__
              {| line 2, column 1: "b": column "c1": cannot read as categorical["a"]: not in the dictionary. Declare the null token (~nulls) or another type (with_type). |}
          );
          ( "a bool spelled True",
            lazy (columns [ any Type.bool ]),
            "true\nTrue\n",
            __POS_OF__
              {| line 2, column 1: "True": column "c1": cannot read as bool: not true or false. Declare the null token (~nulls), or read the column as string (with_type) and map its spellings with an expression. |}
          );
        ]
        (fun (_, f, s, baseline) -> expect (failure (Lazy.force f) s) baseline);
      cases "decode fails at the earliest record, then its first field"
        ~id:(fun (name, _, _, _) -> name)
        [
          ( "the earliest record",
            [ any Type.int8; any Type.int8 ],
            "1,1\n2,x\ny,3\n",
            __POS_OF__
              {| line 2, column 3: "x": column "c2": cannot read as int8: not an integer. Declare the null token (~nulls) or another type (with_type). |}
          );
          ( "the first field",
            [ any Type.int8; any Type.int8 ],
            "x,y\n",
            __POS_OF__
              {| line 1, column 1: "x": column "c1": cannot read as int8: not an integer. Declare the null token (~nulls) or another type (with_type). |}
          );
          ( "an earlier record over a later field",
            [ any Type.int8; any Type.int8 ],
            "x,1\n2,y\n",
            __POS_OF__
              {| line 1, column 1: "x": column "c1": cannot read as int8: not an integer. Declare the null token (~nulls) or another type (with_type). |}
          );
          ( "a value before a syntax error",
            [ any Type.int8; any Type.int8 ],
            "1,1\n2,x\n3,\"4\n",
            __POS_OF__
              {| line 2, column 3: "x": column "c2": cannot read as int8: not an integer. Declare the null token (~nulls) or another type (with_type). |}
          );
          ( "a value before a wider record",
            [ any Type.int64 ],
            "x\n" ^ wide_record ^ "\n",
            __POS_OF__
              {| line 1, column 1: "x": column "c1": cannot read as int64: not an integer. Declare the null token (~nulls) or another type (with_type). |}
          );
        ]
        (fun (_, types, s, baseline) ->
          expect (failure (columns types) s) baseline);
    ]

(* Formats *)

let two = [ ("a", any Type.int64); ("b", any Type.string) ]

let formats =
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
      cases "raises on a description CSV cannot read"
        ~id:(fun (name, _, _, _, _) -> name)
        [
          ("no column", ',', '"', [], []);
          ( "two columns of one name",
            ',',
            '"',
            [],
            [ ("a\n", any Type.int64); ("a\n", any Type.string) ] );
          ( "a name that is not UTF-8",
            ',',
            '"',
            [],
            [ ("\xff", any Type.int64) ] );
          ("a line feed as the separator", '\n', '"', [], two);
          ("a carriage return as the separator", '\r', '"', [], two);
          ("a line feed as the quote", ',', '\n', [], two);
          ("a carriage return as the quote", ',', '\r', [], two);
          ("a separator outside ASCII", '\xc3', '"', [], two);
          ("a quote outside ASCII", ',', '\xe9', [], two);
          ("a separator equal to the quote", '"', '"', [], two);
          ("a null token holding the separator", ',', '"', [ "a,b" ], two);
          ("a null token holding the quote", ',', '"', [ "\"x" ], two);
          ("a null token holding a line feed", ',', '"', [ "a\nb" ], two);
          ("a null token holding a carriage return", ',', '"', [ "\r" ], two);
          ( "a null token that starts with a byte order mark",
            ',',
            '"',
            [ "\xEF\xBB\xBFNA" ],
            two );
        ]
        (fun (_, sep, quote, nulls, columns) ->
          raises_match (Exn.invalid_arg ~sub:"Talon_csv.format") (fun () ->
              Csv.format ~sep ~quote ~nulls columns));
      cases "raises on a type CSV does not read" ~id:type_name other_types
        (fun t ->
          raises_match (Exn.invalid_arg ~sub:"Talon_csv.format") (fun () ->
              Csv.format [ ("a", t) ]));
      test
        "has a header name its columns byte for byte, the empty name included"
        (fun () ->
          let f =
            Csv.format [ ("", any Type.binary); ("café", any Type.binary) ]
          in
          equal records
            [ [ Some "x"; Some "y" ] ]
            (decoded f "\"\",café\nx,y\n"));
    ]

let with_types =
  let f = Csv.format two in
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
          raises_match (Exn.invalid_arg ~sub:"Talon_csv.with_type") (fun () ->
              Csv.with_type "c" (any Type.int64) f));
      cases "raises on a type CSV does not read" ~id:type_name other_types
        (fun t ->
          raises_match (Exn.invalid_arg ~sub:"Talon_csv.with_type") (fun () ->
              Csv.with_type "a" t f));
    ]

let dialect s = List.hd (String.split_on_char '\n' (format_text (sniffed s)))

let pp_formats =
  group "pp_format"
    [
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
      test "counts the rows it sniffed from with thousands separators"
        (fun () ->
          equal string
            "csv (1 column, separator ',', quote '\"', header), types sniffed \
             from 1,500 rows"
            (dialect
               ("x\n" ^ String.concat "" (List.init 1500 (fun _ -> "1\n")))));
      test "counts no rows under a header alone" (fun () ->
          equal string
            "csv (2 columns, separator ',', quote '\"', header), types sniffed \
             from no rows"
            (dialect "a,b\n\n"));
    ]

(* Sniffing *)

let column_types f =
  List.tl (String.split_on_char '\n' (format_text f)) |> List.map String.trim

let sniffing =
  group "sniff"
    [
      cases "separators"
        ~id:(fun (name, _, _) -> name)
        [
          ("a comma", "a,b\n1,2\n", "2 columns, separator ','");
          ("a tab", "a\tb\n1\t2\n", "2 columns, separator '\\t'");
          ("a semicolon", "a;b;c\n1;2;3\n", "3 columns, separator ';'");
          ("a bar", "a|b\n1|2\n", "2 columns, separator '|'");
          ( "the one that splits into the most fields",
            "a;b,c;d\n1;2,3;4\n",
            "3 columns, separator ';'" );
          ("the first on a tie", "a;b,c\n1;2,3\n", "2 columns, separator ','");
          ( "one that splits every record alike, over one that does not",
            "a;b;c,d\n1;2,3\n",
            "2 columns, separator ','" );
          ( "not a separator between quotes",
            "a;\"b,c,d\"\n1;2\n",
            "2 columns, separator ';'" );
          ("a comma for one column", "a\n1\n", "1 column, separator ','");
        ]
        (fun (_, s, expected) ->
          equal string
            (Printf.sprintf
               "csv (%s, quote '\"', header), types sniffed from 1 row" expected)
            (dialect s));
      cases "types"
        ~id:(fun (name, _, _) -> name)
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
          ( "string from a negative integer int64 does not hold",
            "x\n-9223372036854775809\n",
            [ "x string" ] );
          ( "string from a float and an integer int64 does not hold",
            "x\n1.5\n9223372036854775808\n",
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
          ( "datetime[ns, UTC] from an offset and a fine fraction",
            "x\n2024-01-31T10:00:00Z\n2024-01-31T10:00:00.1234567+05:30\n",
            [ "x datetime[ns, UTC]" ] );
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
          ("string from a quoted empty string", "x\n\"\"\n1\n", [ "x string" ]);
          ("int64 from a quoted integer", "x\n\"12\"\n", [ "x int64" ]);
          ("never a categorical", "x\na\nb\na\nb\na\n", [ "x string" ]);
        ]
        (fun (_, s, expected) ->
          equal (list string) expected (column_types (sniffed s)));
      cases "header"
        ~id:(fun (name, _, _) -> name)
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
        (fun (_, s, expected) ->
          equal (list string) expected (column_types (sniffed s)));
      test "reads the header's quoted names undoubled" (fun () ->
          equal (list string) [ "\"a,\\\"b\" int64" ]
            (column_types (sniffed "\"a,\"\"b\"\n1\n")));
      test "skips a byte order mark" (fun () ->
          equal (list string) [ "a int64" ]
            (column_types (sniffed "\xEF\xBB\xBFa\n1\n")));
      test "infers types from the first rows records after the header"
        (fun () ->
          expect (format_text (sniffed ~rows:2 "x\n1\n2\nz\n"))
          @@ __POS_OF__
               {|
            csv (1 column, separator ',', quote '"', header), types sniffed from 2 rows
              x int64
            |});
      test "infers types from the first rows records without a header"
        (fun () ->
          expect (format_text (sniffed ~rows:1 "1\n2\nz\n"))
          @@ __POS_OF__
               {|
            csv (1 column, separator ',', quote '"', no header), types sniffed from 1 row
              column_1 int64
            |});
      test "does not count empty lines as rows" (fun () ->
          expect (format_text (sniffed ~rows:2 "x\n\n1\n\n\r\n2\nz\n"))
          @@ __POS_OF__
               {|
            csv (1 column, separator ',', quote '"', header), types sniffed from 2 rows
              x int64
            |});
      test "infers types from 16,384 records by default" (fun () ->
          let s =
            "x\n" ^ String.concat "" (List.init 16384 (fun _ -> "1\n")) ^ "z\n"
          in
          expect (format_text (sniffed s))
          @@ __POS_OF__
               {|
            csv (1 column, separator ',', quote '"', header), types sniffed from 16,384 rows
              x int64
            |});
      test "infers types from records past the first MiB" (fun () ->
          (* 12,000 records of 100 bytes span two of the scanner's batches. *)
          let s =
            "x,y\n"
            ^ String.concat ""
                (List.init 11999 (fun _ -> "1," ^ String.make 97 'a' ^ "\n"))
            ^ "z,b\n"
          in
          equal (list string) [ "x string"; "y string" ]
            (column_types (sniffed s));
          equal (list string) [ "x int64"; "y string" ]
            (column_types (sniffed ~rows:11999 s)));
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
            (column_types (sniffed ~nulls:[ "NA" ] "x\nNA\n\"NA\"\n1\n")));
      test "makes a format that reads its null tokens as null" (fun () ->
          let s = "x\nNA\n1\n-\n" in
          equal table
            (v [ ("x", Column.of_options Type.int64 [| None; Some 1; None |]) ])
            (Error.get_ok (read (sniffed ~nulls:[ "NA"; "-" ] s) s)));
      cases
        "raises on a null token holding a separator, the quote or a line \
         break, or starting with a byte order mark"
        ~id:(Printf.sprintf "%S")
        [ "a,b"; "a\tb"; "a;b"; "a|b"; "\""; "\n"; "\xEF\xBB\xBFNA" ] (fun t ->
          raises_match (Exn.invalid_arg ~sub:"Talon_csv.sniff") (fun () ->
              Csv.sniff ~nulls:[ t ] (Reader.of_string "a\n")));
      test "raises on rows below 1" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Talon_csv.sniff") (fun () ->
              Csv.sniff ~rows:0 (Reader.of_string "a\n")));
      cases "leaves the reader whole"
        ~id:(fun (n, rows) ->
          Printf.sprintf "in %d-byte slices, sniffing %d rows" n rows)
        [ (1, 1); (3, 1); (3, 5); (64, 2); (65536, 16384) ]
        (fun (slice_length, rows) ->
          let s = "a,\"b\nc\"\r\n1,2\n3,\"x\"\"y\"\n4,5" in
          let r = Reader.of_string ~slice_length s in
          ignore (Error.get_ok (Csv.sniff ~rows r));
          equal string s (Reader.to_string r));
      test "leaves the reader where it was" (fun () ->
          let r = Reader.of_string ~slice_length:5 "skip\na,b\n1,2\n" in
          ignore (Reader.read r);
          ignore (Error.get_ok (Csv.sniff r));
          equal string "a,b\n1,2\n" (Reader.to_string r));
      cases "fails with the line and column where the input fails"
        ~id:(fun (name, _, _) -> name)
        [
          ("on no input", "", __POS_OF__ {| the input holds no record |});
          ( "on a byte order mark alone",
            "\xEF\xBB\xBF",
            __POS_OF__ {| the input holds no record |} );
          ( "on empty lines alone",
            "\n\r\n\n",
            __POS_OF__ {| the input holds no record |} );
          ( "on records no separator splits alike",
            "a,b\n1,2\n3\n",
            __POS_OF__
              {| line 3, column 1: no separator splits every record into the same number of fields: with ',', the record has 1 field, and the first one 2 |}
          );
          ( "on records a tab splits unevenly",
            "a\tb\n1\t2\t3\n",
            __POS_OF__
              {| line 2, column 1: no separator splits every record into the same number of fields: with '\t', the record has 3 fields, and the first one 2 |}
          );
          ( "on a quoted field that does not end",
            "a,b\n1,\"2\n3,4\n",
            __POS_OF__ {| line 2, column 3: a quoted field does not end |} );
          ( "on a header that names two columns the same",
            "a,b,a\n1,2,3\n",
            __POS_OF__
              {| line 1, column 5: "a": the header names two columns the same |}
          );
          ( "on a column name that is not UTF-8",
            "a,\xff\n1,2\n",
            __POS_OF__
              {| line 1, column 3: "\xff": the column name is not UTF-8 |} );
        ]
        (fun (_, s, baseline) ->
          expect
            (match Csv.sniff (Reader.of_string s) with
            | Ok _ -> "ok"
            | Error e -> error_text e)
            baseline);
      test "fails when reading its reader raises" (fun () ->
          let outcome r =
            match Csv.sniff r with Ok _ -> "ok" | Error e -> error_text e
          in
          expect
            (String.concat "\n"
               [
                 outcome
                   (Reader.limit 5
                      (Reader.of_string ~slice_length:2 "a,b\n1,2\n"));
                 outcome (Reader.make (fun () -> raise (Sys_error "disk")));
               ])
          @@ __POS_OF__
               {|
            reader:5: Limit of 5 bytes exceeded.
            disk
            |});
    ]

(* Reading *)

(* [quoted_records n] is a header and about [n] bytes of records, each a quoted
   line break and a number. *)
let quoted_records n =
  let b = Buffer.create (n + 16) in
  Buffer.add_string b "a,b\n";
  let i = ref 0 in
  while Buffer.length b < n do
    Printf.bprintf b "\"r\n%d\",%d\n" !i !i;
    incr i
  done;
  Buffer.contents b

let big_rows = 200_000

(* About 2.4 MiB of records, each a quoted field with a line break and a
   number. *)
let big =
  lazy
    (let b = Buffer.create (big_rows * 13) in
     for i = 0 to big_rows - 1 do
       Printf.bprintf b "\"r\n%d\",%d\n" i i
     done;
     Buffer.contents b)

let text_int = columns [ any Type.binary; any Type.int64 ]

(* [with_ratio r f] is [f ()] with the collector's [custom_major_ratio] at
   [r]. *)
let with_ratio r f =
  let was = (Gc.get ()).custom_major_ratio in
  Gc.set { (Gc.get ()) with custom_major_ratio = r };
  Fun.protect f ~finally:(fun () ->
      Gc.set { (Gc.get ()) with custom_major_ratio = was })

(* 2^14 rows of 8 bytes in one slice, one batch: a text column of 128 KiB of
   bytes and 128 KiB of offsets. *)
let text_column () =
  let rows = 1 lsl 14 in
  let s = "s\n" ^ String.concat "" (List.init rows (fun _ -> "abcdefgh\n")) in
  let f = Csv.format [ ("s", any Type.binary) ] in
  column (Error.get_ok (read ~slice_length:(String.length s) f s)) "s"

let text_column_bytes = (8 lsl 14) + (8 * ((1 lsl 14) + 1))

(* Dropped columns pace the collector as rig's host buffers do, whatever
   [custom_major_ratio] says of the heap: the bytes of host buffers that reach
   the major heap bring a cycle's work per bound of at least 4 MiB, and a buffer
   that starts young with more than [custom_minor_max_size] bytes brings an
   eighth of its bytes at birth as well. A decode makes its column's bytes and
   offsets, 128 KiB each, in such buffers, and holds them across a minor
   collection that promotes them. So at a ratio of 1, which paces a bigarray of
   Bigarray's own at a cycle's work each, 64 decodes bring at most a cycle's
   work per 4 MiB of their arrays and of an eighth of them more than at the
   default ratio. The count starts from a compaction, which leaves the heap
   about its live words: a cycle's work, which the runtime sizes as sweeping the
   heap and marking part of it, then completes at most a cycle, and two more are
   under way at either end. The cycles the decodes bring at either ratio are the
   heap's own, paced by its allocations. *)
let paces_collector () =
  let decodes = 64 in
  let cycles r =
    with_ratio r @@ fun () ->
    ignore (text_column ());
    Talon_held.cycles (fun () ->
        for _ = 1 to decodes do
          let held = text_column () in
          Gc.minor ();
          ignore (Sys.opaque_identity held)
        done)
  in
  let usual = cycles (Gc.get ()).custom_major_ratio in
  let at_1 = cycles 1 in
  let made = decodes * (text_column_bytes + (text_column_bytes / 8)) in
  at_most
    ~msg:
      (Printf.sprintf "major cycles at a ratio of 1: %s; at the default: %s"
         (Talon_held.pp_cycles at_1)
         (Talon_held.pp_cycles usual))
    int
    ~than:(usual.majors + (made / (4 lsl 20)) + 2)
    at_1.majors

let decoding =
  let ab = Csv.format [ ("a", any Type.binary); ("b", any Type.binary) ] in
  group "decode"
    [
      test "reads a large input in batches, every record in order" (fun () ->
          let t = Error.get_ok (read text_int (Lazy.force big)) in
          greater int ~than:1 (List.length (batches t));
          equal (array int)
            (Array.init big_rows Fun.id)
            (Column.values Kind.int (column t "c2")));
      cases "cuts the same batches whatever the reader's slices"
        ~id:(Printf.sprintf "in slices of %d bytes") [ 1; 7; 256; 4096 ]
        (fun slice_length ->
          (* About 2 KiB in batches of 256 bytes. *)
          let s = quoted_records 2048 in
          let sizes slice_length =
            List.map rows
              (batches
                 (Error.get_ok
                    (Csv.decode_in_batches ~batch:256 ab
                       (Reader.of_string ?slice_length s))))
          in
          equal (list int) (sizes None) (sizes (Some slice_length)));
      test "keeps a quoted line feed at the 1 MiB edge in its field" (fun () ->
          (* The quoted line feed is the input's byte 2^20 - 1, the first that
             may end a batch. *)
          let s =
            "xx\n"
            ^ String.concat "" (List.init 524285 (fun _ -> "x\n"))
            ^ "\"a\nb\"\nz\n"
          in
          equal int ((1 lsl 20) - 1) (String.index_from s 1048574 '\n');
          let rows = decoded (bytes 1) s in
          equal int 524288 (List.length rows);
          equal records
            [ [ Some "x" ]; [ Some "a\nb" ]; [ Some "z" ] ]
            (List.filteri (fun i _ -> i >= 524285) rows));
      test "counts lines across batches" (fun () ->
          expect (failure text_int (Lazy.force big ^ "x,y\n"))
          @@ __POS_OF__
               {| line 400001, column 3: "y": column "c2": cannot read as int64: not an integer. Declare the null token (~nulls) or another type (with_type). |});
      test "counts empty lines and line breaks of two bytes across batches"
        (fun () ->
          let records = 200_000 in
          let s =
            String.concat "" (List.init records (fun _ -> "a,1\r\n\r\n"))
            ^ "x,y\n"
          in
          greater int ~than:(1 lsl 20) (String.length s);
          (* Each record and its empty line are two lines. *)
          starts_with
            ~prefix:(Printf.sprintf "line %d, column 3:" ((2 * records) + 1))
            (failure text_int s));
      test "reads a last record that no line feed ends" (fun () ->
          let s =
            "a,0\n"
            ^ String.concat ""
                (List.init 1000 (fun i -> Printf.sprintf "a,%d\n" i))
            ^ "b,2"
          in
          let rows = decoded (bytes 2) s in
          equal int 1002 (List.length rows);
          equal records
            [ [ Some "b"; Some "2" ] ]
            (List.filteri (fun i _ -> i = 1001) rows));
      cases "fails on an empty input when the format has a header"
        ~id:(fun (name, _) -> name)
        [
          ("no byte", "");
          ("a byte order mark", "\xEF\xBB\xBF");
          ("empty lines", "\n\r\n\n");
        ]
        (fun (_, s) ->
          expect (failure (Csv.format [ ("a", any Type.string) ]) s)
          @@ __POS_OF__ {| the input is empty, and the format has a header |});
      test "reads an empty input without a header as no rows" (fun () ->
          equal table
            (v [ ("c1", Column.of_options Type.int64 [||]) ])
            (Error.get_ok (read (columns [ any Type.int64 ]) "")));
      cases "fails on a header that does not name the format's columns"
        ~id:(fun (name, _, _) -> name)
        [
          ( "another column",
            "a,c\nx,y\n",
            __POS_OF__
              {| line 1, column 3: "c": the header does not name column 2 "b" |}
          );
          ( "the columns in another order",
            "b,a\nx,y\n",
            __POS_OF__
              {| line 1, column 1: "b": the header does not name column 1 "a" |}
          );
          ( "too few fields",
            "a\n",
            __POS_OF__
              {| line 1, column 2: the record has 1 field, and the format 2 columns |}
          );
        ]
        (fun (_, s, baseline) -> expect (failure ab s) baseline);
      test "fails when reading its reader raises" (fun () ->
          let outcome r =
            match Csv.decode ab r with Ok _ -> "ok" | Error e -> error_text e
          in
          expect
            (String.concat "\n"
               [
                 outcome
                   (Reader.limit 5
                      (Reader.of_string ~slice_length:2 "a,b\n1,2\n"));
                 outcome (Reader.make (fun () -> raise (Sys_error "disk")));
               ])
          @@ __POS_OF__
               {|
            reader:5: Limit of 5 bytes exceeded.
            disk
            |});
      (* A text column is host memory: the host holds its bytes and offsets
         while it is live, and has them back once it is collected. *)
      test "makes a text column of the host's memory" (fun () ->
          Talon_held.check text_column);
      test "makes columns whose drop paces the collector by memory"
        paces_collector;
    ]

(* Sources *)

let run_ok q = Error.get_ok (Query.run q)

let run_error q =
  match Query.run q with Ok _ -> "ok" | Error e -> error_text e

let source ?slice_length f s =
  Query.of_source (Csv.source f (fun () -> Reader.of_string ?slice_length s))

let keep names q =
  Query.select (if names = [] then [] else Expr.[ keep (Sel.names names) ]) q

let ab = Csv.format [ ("a", any Type.int64); ("b", any Type.int8) ]

(* Record 4's [b] does not read. *)
let bad_b = "a,b\n1,1\n2,\n3,x\n"

type request = { t : file; read : bool list; limit : int option }

let request =
  let open Gen in
  let* t = file in
  let* read = list ~size:(constant t.cols) bool in
  let+ limit = option (int_range 0 14) in
  { t; read; limit }

let pp_request ppf r =
  Format.fprintf ppf "%a, reading %s, limit %s" pp_file r.t
    (String.concat "" (List.map (fun b -> if b then "1" else "0") r.read))
    (match r.limit with None -> "none" | Some n -> string_of_int n)

let ask r q =
  let names =
    List.filteri
      (fun j _ -> List.nth r.read j)
      (List.init r.t.cols (fun j -> Printf.sprintf "c%d" (j + 1)))
  in
  let q = keep names q in
  match r.limit with None -> q | Some n -> Query.slice ~offset:0 ~length:n q

(* A limit of 0 over a record. *)
let limit_0 =
  {
    t =
      {
        cols = 2;
        rows = [ [ (Some "a", false); (None, false) ] ];
        crlf = [ false ];
        blank = [ false ];
        last = true;
        bom = false;
        slice = 4;
      };
    read = [ true; false ];
    limit = Some 0;
  }

let reads_as_decode r =
  cover "no column read" (not (List.mem true r.read));
  cover "a limit below the rows"
    (match r.limit with Some n -> n < List.length r.t.rows | None -> false);
  cover "a limit of 0" (r.limit = Some 0);
  let f = bytes ~nulls:[ "NA" ] r.t.cols and s = encode r.t in
  let decoded = Query.of_table (Error.get_ok (read f s)) in
  equal table
    (run_ok (ask r decoded))
    (run_ok (ask r (source ~slice_length:r.t.slice f s)))

let sources =
  group "source"
    [
      test "reads the table decode reads" (fun () ->
          let s = "a,b\n1,1\n2,\n" in
          equal table (Error.get_ok (read ab s)) (run_ok (source ab s)));
      test "is named csv in plans" (fun () ->
          expect (Format.asprintf "%a" Query.pp (source ab bad_b))
          @@ __POS_OF__
               {|
            query → a int64, b int8
            csv (2 columns)
            |});
      test "reads the request's columns only" (fun () ->
          equal table
            (v [ ("a", Column.v Type.int64 [| 1; 2; 3 |]) ])
            (run_ok (keep [ "a" ] (source ab bad_b))));
      test "stops after its limit" (fun () ->
          equal table
            (v
               [
                 ("a", Column.v Type.int64 [| 1; 2 |]);
                 ("b", Column.of_options Type.int8 [| Some 1; None |]);
               ])
            (run_ok (Query.slice ~offset:0 ~length:2 (source ab bad_b))));
      test "counts rows without reading a column" (fun () ->
          equal int 3 (rows (run_ok (keep [] (source ab bad_b)))));
      test "applies no conjunct" (fun () ->
          let q = Query.filter Expr.(Col.int "a" > int 1) (source ab bad_b) in
          expect (Format.asprintf "%a" Query.pp (Query.optimize q))
          @@ __POS_OF__
               {|
            query → a int64, b int8
            filter (a > 1)
            └ csv (2 columns)
            |});
      test "fails where decode fails" (fun () ->
          equal string (failure ab bad_b) (run_error (source ab bad_b)));
      test "fails on a record of another number of fields within its limit"
        (fun () ->
          let s = "a,b\n1,1\n2\n" in
          expect (run_error (Query.slice ~offset:0 ~length:2 (source ab s)))
          @@ __POS_OF__
               {| line 3, column 2: the record has 1 field, and the format 2 columns |};
          equal table
            (v
               [
                 ("a", Column.v Type.int64 [| 1 |]);
                 ("b", Column.v Type.int8 [| 1 |]);
               ])
            (run_ok (Query.slice ~offset:0 ~length:1 (source ab s))));
      test "opens a reader for each read" (fun () ->
          let opened = ref 0 in
          let open_ () =
            incr opened;
            Reader.of_string "a,b\n1,1\n"
          in
          let q = Query.of_source (Csv.source ab open_) in
          ignore (run_ok q);
          ignore (run_ok q);
          equal int 2 !opened);
      prop ~examples:[ limit_0 ]
        "reads what decode reads, whatever its reader's slices, columns and \
         limit"
        (Gen.with_pp pp_request request)
        reads_as_decode;
    ]

(* Files *)

(* [in_temp_dir contents] makes [data.csv] of [contents] in a fresh working
   directory. *)
let in_temp_dir contents =
  chdir (temp_dir ());
  Out_channel.with_open_bin "data.csv" (fun oc -> output_string oc contents)

let x_int = Csv.format [ ("x", any Type.int64) ]

let files =
  group "file"
    [
      test "sniffs its file with the null tokens and reads it" (fun () ->
          in_temp_dir "x,y\n1,NA\n2,b\n";
          let s = Error.get_ok (Csv.file ~nulls:[ "NA" ] "data.csv") in
          equal table
            (v
               [
                 ("x", Column.v Type.int64 [| 1; 2 |]);
                 ("y", Column.of_options Type.string [| None; Some "b" |]);
               ])
            (run_ok (Query.of_source s)));
      test "names its file in its plans and errors" (fun () ->
          in_temp_dir "x\n1\nz\n";
          let q =
            Query.of_source (Error.get_ok (Csv.file ~format:x_int "data.csv"))
          in
          expect (Format.asprintf "%a@.%s" Query.pp q (run_error q))
          @@ __POS_OF__
               {|
            query → x int64
            csv "data.csv" (1 column)
            data.csv:3:1: "z": column "x": cannot read as int64: not an integer. Declare the null token (~nulls) or another type (with_type).
            |});
      test "names its file in a sniffing error" (fun () ->
          in_temp_dir "a,b\n1\n";
          expect
            (match Csv.file "data.csv" with
            | Ok _ -> "ok"
            | Error e -> error_text e)
          @@ __POS_OF__
               {| data.csv:2:1: no separator splits every record into the same number of fields: with ',', the record has 1 field, and the first one 2 |});
      test "fails at the call on a path where no file is, with a format or not"
        (fun () ->
          chdir (temp_dir ());
          let outcome = function Ok _ -> "ok" | Error e -> error_text e in
          expect
            (String.concat "\n"
               [
                 outcome (Csv.file "missing.csv");
                 outcome (Csv.file ~format:x_int "missing.csv");
               ])
          @@ __POS_OF__
               {|
            missing.csv: No such file or directory
            missing.csv: No such file or directory
            |});
      test "reads a named pipe once, with a format" (fun () ->
          chdir (temp_dir ());
          (try Unix.mkfifo "data.csv" 0o600
           with Unix.Unix_error _ | Invalid_argument _ ->
             skip ~reason:"no named pipes" ());
          (* No writer has the pipe open at the call, so a call that opened it
             would wait for ever. *)
          let s = Error.get_ok (Csv.file ~format:x_int "data.csv") in
          let writer =
            Domain.spawn (fun () ->
                Out_channel.with_open_bin "data.csv" (fun oc ->
                    output_string oc "x\n1\n2\n"))
          in
          let t = run_ok (Query.of_source s) in
          Domain.join writer;
          equal table (v [ ("x", Column.v Type.int64 [| 1; 2 |]) ]) t);
      test "fails a read of a file removed since the call" (fun () ->
          in_temp_dir "x\n1\n";
          let s = Error.get_ok (Csv.file ~format:x_int "data.csv") in
          Sys.remove "data.csv";
          expect (run_error (Query.of_source s))
          @@ __POS_OF__ {| data.csv: No such file or directory |});
      test "raises on null tokens given with a format" (fun () ->
          let f = Csv.format ~nulls:[ "NA" ] [ ("x", any Type.int64) ] in
          raises_match (Exn.invalid_arg ~sub:"Talon_csv.file") (fun () ->
              Csv.file ~format:f ~nulls:[ "NA" ] "data.csv"));
      test "opens its file again for each read" (fun () ->
          in_temp_dir "x\n1\n";
          let q =
            Query.of_source (Error.get_ok (Csv.file ~format:x_int "data.csv"))
          in
          equal table (v [ ("x", Column.v Type.int64 [| 1 |]) ]) (run_ok q);
          Out_channel.with_open_bin "data.csv" (fun oc ->
              output_string oc "x\n2\n3\n");
          equal table (v [ ("x", Column.v Type.int64 [| 2; 3 |]) ]) (run_ok q));
    ]

(* Writing *)

let written f q =
  let b = Buffer.create 256 in
  match Csv.encode f q (Writer.of_buffer b) with
  | Ok () -> Buffer.contents b
  | Error e -> Buffer.contents b ^ "error: " ^ error_text e

(* A table to write: up to four columns of the types CSV reads, named to need
   quoting, cut into batches, and a dialect. One column with a null takes a null
   token. *)
type writable = { f : Csv.format; t : Talon.t; slice : int }

let names = [ "a"; ""; "x,y"; "q\"t"; "NA"; "l\nm"; "s;t" ]

let writable =
  let open Gen in
  let* samples =
    list ~size:(int_range 1 4)
      (let* (Type.Any ty) = of_list ~pp:pp_any csv_types in
       let+ vs = G.options ty in
       G.Sample (ty, vs))
  in
  let n =
    List.fold_left
      (fun n (G.Sample (_, vs)) -> Int.min n (Array.length vs))
      max_int samples
  in
  let* k = int_range 0 (List.length names - 1) in
  let names =
    List.filteri (fun i _ -> i >= k) names
    @ List.filteri (fun i _ -> i < k) names
  in
  let columns =
    List.mapi
      (fun i (G.Sample (ty, vs)) ->
        (List.nth names i, Column.of_options ty (Array.sub vs 0 n)))
      samples
  in
  let t = Talon.v ~rows:n columns in
  let* sep = of_list [ ','; ';'; '\t'; '|'; '.'; ' ' ] in
  let* header = bool in
  let* tokens = of_list [ []; [ "NA" ]; [ "NA"; "-" ] ] in
  let nulls =
    match columns with
    | [ (_, c) ] when Column.null_count c > 0 && tokens = [] -> [ "NA" ]
    | _ -> tokens
  in
  let* t = G.split t in
  let+ slice = int_range 1 16 in
  let types = List.map (fun (n, c) -> (n, Column.type_ c)) columns in
  { f = Csv.format ~sep ~header ~nulls types; t; slice }

let pp_writable ppf w =
  Format.fprintf ppf "@[<v>%a@,%a@,%S@]" Csv.pp_format w.f Talon.pp w.t
    (written w.f (Query.of_table w.t))

(* Floats whose text holds the separator [.], and a byte order mark that starts
   the file. *)
let writable_examples =
  [
    {
      f = Csv.format ~sep:'.' [ ("x", any Type.float64) ];
      t = v [ ("x", Column.v Type.float64 [| 1.5; -0.; 1e21 |]) ];
      slice = 3;
    };
    {
      f = Csv.format ~header:false [ ("b", any Type.binary) ];
      t =
        v [ ("b", Column.v Type.binary [| Binary.of_string "\xEF\xBB\xBFa" |]) ];
      slice = 1;
    };
  ]

let reads_back w =
  let types = List.map snd (Schema.columns (schema w.t)) in
  List.iter
    (fun t ->
      cover (type_name t)
        (List.exists (fun u -> type_name u = type_name t) types))
    csv_types;
  cover "one column with a null"
    (match Schema.columns (schema w.t) with
    | [ (n, _) ] -> Column.null_count (column w.t n) > 0
    | _ -> false);
  Law.round_trip table string
    (fun t -> written w.f (Query.of_table t))
    (fun s -> Error.get_ok (read ~slice_length:w.slice w.f s))
    w.t

let writing =
  group "encode"
    [
      prop ~count:450 ~examples:writable_examples
        "decode reads back the rows encode writes"
        (Gen.with_pp pp_writable writable)
        reads_back;
      test "quotes a field only when reading needs it" (fun () ->
          let s =
            [|
              Some "plain";
              Some "";
              None;
              Some "a,b";
              Some "say \"hi\"";
              Some "\"lead";
              Some "two\nlines";
              Some "cr\r";
              Some "semi;colon";
              Some "NA";
              Some " spaced ";
              Some "\xff";
            |]
          in
          let t =
            v
              [
                ( "s",
                  Column.of_options Type.binary
                    (Array.map (Option.map Binary.of_string) s) );
                ("n", Column.v Type.int64 (Array.init 12 Fun.id));
              ]
          in
          let f =
            Csv.format ~nulls:[ "NA" ]
              [ ("s", any Type.binary); ("n", any Type.int64) ]
          in
          equal string
            "s,n\n\
             plain,0\n\
             \"\",1\n\
             ,2\n\
             \"a,b\",3\n\
             \"say \"\"hi\"\"\",4\n\
             \"\"\"lead\",5\n\
             \"two\n\
             lines\",6\n\
             \"cr\r\",7\n\
             semi;colon,8\n\
             \"NA\",9\n\
            \ spaced ,10\n\
             \xff,11\n"
            (written f (Query.of_table t)));
      test "quotes a field that starts with a byte order mark" (fun () ->
          let bom = "\xEF\xBB\xBF" in
          let t =
            v
              [
                ( "y",
                  Column.v Type.binary
                    [| Binary.of_string (bom ^ "a"); Binary.of_string "b" |] );
              ]
          in
          equal string
            ("\"" ^ bom ^ "a\"\nb\n")
            (written
               (Csv.format ~header:false [ ("y", any Type.binary) ])
               (Query.of_table t)));
      test "writes the text Column.print writes" (fun () ->
          let t =
            v
              [
                ("b", Column.v Type.bool [| true; false |]);
                ("i", Column.v Type.int8 [| -128; 127 |]);
                ( "c",
                  Column.v (Type.categorical [| ""; "u,v" |]) [| ""; "u,v" |] );
                ( "t",
                  Column.v (Type.instant Ms)
                    [|
                      Option.get (Time.of_ms 1500L); Option.get (Time.of_ms 0L);
                    |] );
                ( "y",
                  Column.v Type.binary
                    [| Binary.of_string "b"; Binary.of_string "" |] );
              ]
          in
          let f =
            Csv.format ~sep:';' ~header:false (Schema.columns (schema t))
          in
          expect (written f (Query.of_table t))
          @@ __POS_OF__
               {|
            true;-128;"";1970-01-01T00:00:01.5Z;b
            false;127;u,v;1970-01-01T00:00:00Z;""
            |});
      test "writes a null in one column as its first non-empty null token"
        (fun () ->
          let t =
            v [ ("a", Column.of_options Type.int64 [| Some 1; None |]) ]
          in
          equal string "a\n1\nNA\n"
            (written
               (Csv.format ~nulls:[ ""; "NA" ] [ ("a", any Type.int64) ])
               (Query.of_table t)));
      test "fails at a null in one column without a null token, at its line"
        (fun () ->
          let t =
            v [ ("a", Column.of_options Type.string [| Some "x\ny"; None |]) ]
          in
          (* The header is line 1, the quoted field lines 2 and 3. *)
          expect
            (written (Csv.format [ ("a", any Type.string) ]) (Query.of_table t))
          @@ __POS_OF__
               {|
            a
            "x
            y"
            error: line 4, column 1: column "a": a null in the only column is an empty line, which is no record. Declare a null token (~nulls).
            |});
      test "fails as its query fails, after the rows before it" (fun () ->
          let t =
            of_batches
              [
                v [ ("a", Column.v Type.string [| "1" |]) ];
                v [ ("a", Column.v Type.string [| "x" |]) ];
              ]
          in
          let q =
            Query.select
              Expr.[ "a" := Str.parse Type.int64 (Col.string "a") ]
              (Query.of_table t)
          in
          expect (written (Csv.format [ ("a", any Type.int64) ]) q)
          @@ __POS_OF__
               {|
            a
            1
            error: select ["a" := Str.parse int64 a]: row 1: "x": not an integer.
            |});
      test "fails when writing raises" (fun () ->
          let q =
            Query.of_table (v [ ("a", Column.v Type.int64 [| 1; 22; 333 |]) ])
          in
          let f = Csv.format [ ("a", any Type.int64) ] in
          let outcome w =
            match Csv.encode f q w with
            | Ok () -> "ok"
            | Error e -> error_text e
          in
          let limited =
            Writer.limit 8 ~eod:false (Writer.of_buffer (Buffer.create 16))
          in
          expect
            (String.concat "\n"
               [
                 outcome limited;
                 outcome (Writer.make (fun _ -> raise (Sys_error "disk full")));
               ])
          @@ __POS_OF__
               {|
            writer:8: Limit of 8 bytes exceeded.
            disk full
            |});
      test "ends the writer when asked and every record is written" (fun () ->
          let ends ?eod q =
            let eods = ref 0 in
            let w =
              Writer.make (fun s ->
                  if Bytesrw.Bytes.Slice.is_eod s then incr eods)
            in
            ignore (Csv.encode ?eod (Csv.format [ ("a", any Type.int64) ]) q w);
            !eods
          in
          let ok = Query.of_table (v [ ("a", Column.v Type.int64 [| 1 |]) ]) in
          let failing =
            Query.select
              Expr.[ "a" := Str.parse Type.int64 (Col.string "a") ]
              (Query.of_table (v [ ("a", Column.v Type.string [| "x" |]) ]))
          in
          equal (list int) [ 0; 0; 1; 0 ]
            [
              ends ok;
              ends ~eod:false ok;
              ends ~eod:true ok;
              ends ~eod:true failing;
            ]);
      test "raises when the query's columns are not the format's" (fun () ->
          let q =
            Query.of_table
              (v
                 [
                   ("a", Column.v Type.int64 [| 1 |]);
                   ("b", Column.v Type.int64 [| 2 |]);
                 ])
          in
          let refused f =
            match Csv.encode f q (Writer.of_buffer (Buffer.create 1)) with
            | _ -> "no exception"
            | exception Invalid_argument m -> m
          in
          let i64 = any Type.int64 in
          expect
            (String.concat "\n"
               [
                 refused (Csv.format [ ("a", any Type.int32); ("b", i64) ]);
                 refused (Csv.format [ ("a", i64); ("c", i64) ]);
                 refused (Csv.format [ ("b", i64); ("a", i64) ]);
               ])
          @@ __POS_OF__
               {|
            Talon_csv.encode: the query's columns are not the format's: a int64, b int64, against a int32, b int64
            Talon_csv.encode: the query's columns are not the format's: a int64, b int64, against a int64, c int64
            Talon_csv.encode: the query's columns are not the format's: a int64, b int64, against b int64, a int64
            |});
    ]

let () =
  exit
    (run "talon.csv"
       [
         syntax;
         values;
         formats;
         with_types;
         pp_formats;
         sniffing;
         decoding;
         sources;
         files;
         writing;
       ])
