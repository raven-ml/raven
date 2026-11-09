(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module J = Talon_json
module Reader = Bytesrw.Bytes.Reader
module G = Talon_gen

let any t = Type.Any t
let error_text e = Format.asprintf "%a" Error.pp e
let format_text f = Format.asprintf "%a" J.pp_format f
let sniff ?rows s = J.sniff ?rows (Reader.of_string s)
let sniffed ?rows s = Error.get_ok (sniff ?rows s)
let read ?slice_length f s = J.decode f (Reader.of_string ?slice_length s)
let lines ss = String.concat "\n" ss ^ "\n"

(* Rendering, as gen/corpus.py renders Python's values. *)

let escape s =
  let b = Buffer.create (String.length s) in
  String.iter
    (function
      | '\\' -> Buffer.add_string b "\\\\"
      | '\n' -> Buffer.add_string b "\\n"
      | '\t' -> Buffer.add_string b "\\t"
      | '\r' -> Buffer.add_string b "\\r"
      | '\x1f' -> Buffer.add_string b "\\x1f"
      | c -> Buffer.add_char b c)
    s;
  Buffer.contents b

let rec render c =
  let (Type.Any t) = Column.type_ c in
  let valid =
    match Column.validity c with
    | None -> Fun.const true
    | Some v ->
        let v = Nx.to_array v in
        fun i -> v.(i)
  in
  let rows f =
    List.init (Column.length c) (fun i -> if valid i then f i else "null")
  in
  match (t, Column.layout c) with
  | List _, Varsize { offsets; child; _ } ->
      let items = Array.of_list (render child) in
      let o = Array.map Int64.to_int (Nx.to_array offsets) in
      rows (fun i ->
          "["
          ^ String.concat ","
              (Array.to_list (Array.sub items o.(i) (o.(i + 1) - o.(i))))
          ^ "]")
  | Record _, Children { fields; _ } ->
      let parts =
        List.map (fun (f, c) -> (f, Array.of_list (render c))) fields
      in
      rows (fun i ->
          "{"
          ^ String.concat ","
              (List.map (fun (f, r) -> f ^ "=" ^ r.(i)) parts)
          ^ "}")
  | String, _ ->
      let v = Column.options (Type.kind Type.string) c in
      rows (fun i -> "\"" ^ escape (Option.get v.(i)) ^ "\"")
  | Binary, _ ->
      let v =
        Column.options (Type.kind Type.string)
          (Column.print (Result.get_ok (Column.parse (any Type.string) c)))
      in
      rows (fun i -> "\"" ^ escape (Option.get v.(i)) ^ "\"")
  | _ ->
      let v = Column.options (Type.kind Type.string) (Column.print c) in
      rows (fun i -> Option.get v.(i))

let rendered t name = render (column t name)
let names t = List.map fst (Schema.columns (schema t))
let strings = list (option string)

let cell t name =
  Array.to_list (Column.options (Type.kind Type.string) (column t name))

let ints t name =
  Array.to_list
    (Column.options (Type.kind Type.string) (Column.print (column t name)))
let floats = list (option float_exact)

let doubles t name =
  Array.to_list (Column.options (Type.kind Type.float64) (column t name))

let fails ?f s =
  let r =
    match f with
    | Some f -> Result.map ignore (read f s)
    | None -> Result.map ignore (sniff s)
  in
  match r with
  | Ok () -> fail ("read succeeded: " ^ String.escaped s)
  | Error e -> error_text e

(* Sniffing *)

let sniffing =
  group "sniff"
    [
      test "lines: types per column, by first appearance" (fun () ->
          equal string
            "json (5 columns, lines), types sniffed from 2 rows\n\
            \  b bool\n\
            \  i int64\n\
            \  f float64\n\
            \  s string\n\
            \  later int64"
            (format_text
               (sniffed
                  (lines
                     [
                       {|{"b":true,"i":1,"f":1.5,"s":"x"}|};
                       {|{"b":false,"i":2,"f":2,"s":"y","later":7}|};
                     ]))));
      test "array framing is sniffed from the first bracket" (fun () ->
          equal string
            "json (1 column, array), types sniffed from 2 rows\n  a int64"
            (format_text (sniffed {| [ {"a":1}, {"a":2} ] |})));
      test "an integer and a fraction are float64" (fun () ->
          equal string
            "json (1 column, lines), types sniffed from 2 rows\n  a float64"
            (format_text (sniffed (lines [ {|{"a":1}|}; {|{"a":1.5}|} ]))));
      test "integers beyond int64 are strings" (fun () ->
          equal string
            "json (1 column, lines), types sniffed from 1 row\n  a string"
            (format_text (sniffed (lines [ {|{"a":9223372036854775808}|} ]))));
      test "mixed JSON types are strings, reading as JSON text" (fun () ->
          let s = lines [ {|{"a":1.50}|}; {|{"a":"x"}|}; {|{"a":{"b":[1, 2]}}|} ] in
          let f = sniffed s in
          let t = Error.get_ok (read f s) in
          equal strings
            [ Some "1.50"; Some "x"; Some {|{"b":[1, 2]}|} ]
            (cell t "a"));
      test "nested objects and arrays sniff as records and lists" (fun () ->
          equal string
            "json (2 columns, lines), types sniffed from 1 row\n\
            \  u record[name string, tags list[int64]]\n\
            \  x list[list[float64]]"
            (format_text
               (sniffed
                  {|{"u":{"name":"a","tags":[1]},"x":[[1.5],[2.5]]}|})));
      test "only nulls is string, an empty list a list of string" (fun () ->
          equal string
            "json (2 columns, lines), types sniffed from 1 row\n\
            \  n string\n\
            \  e list[string]"
            (format_text (sniffed {|{"n":null,"e":[]}|})));
      test "nesting past the depth limit sniffs as string" (fun () ->
          let deep =
            String.make 200 '[' ^ String.make 200 ']'
          in
          let s = {|{"a":|} ^ deep ^ "}\n" in
          let f = sniffed s in
          let t = Error.get_ok (read f s) in
          equal int 1 (Talon.rows t);
          let depth = ref 0 and ty = ref (Column.type_ (column t "a")) in
          let is_list = ref true in
          while !is_list do
            let (Type.Any u) = !ty in
            match u with
            | List e ->
                incr depth;
                ty := any e
            | String -> is_list := false
            | _ -> fail "expected a list of strings at the end"
          done;
          equal int 63 !depth);
      test "a declared type overrides" (fun () ->
          let f =
            J.with_type "a" (any Type.float32) (sniffed (lines [ {|{"a":1}|} ]))
          in
          equal string
            "json (1 column, lines), types sniffed from 1 row\n\
            \  a float32 declared"
            (format_text f));
      test "sniff stops at its rows, whatever the slices" (fun () ->
          let s = lines (List.init 10_000 (Printf.sprintf {|{"a":%d}|})) in
          let read = ref 0 in
          let inner = Reader.of_string ~slice_length:16 s in
          let r =
            Reader.make (fun () ->
                let slice = Reader.read inner in
                read := !read + Bytesrw.Bytes.Slice.length slice;
                slice)
          in
          let f = Error.get_ok (J.sniff ~rows:10 r) in
          equal string
            "json (1 column, lines), types sniffed from 10 rows\n  a int64"
            (format_text f);
          less ~than:(String.length s / 10) int !read);
      test "a sniffed column's failure cites the rows, a declared one's does \
            not"
        (fun () ->
          let s = lines (List.init 10 (fun _ -> {|{"a":1}|})) ^ {|{"a":1.5}|} in
          let f = sniffed ~rows:4 s in
          let want =
            "line 11, column 6: byte 85: \"1.5\": column \"a\": a number with a \
             fraction or an exponent does not read as int64. The type was \
             sniffed from rows 1 to 4. Declare another (with_type)."
          in
          equal string want (fails ~f s);
          equal string
            "line 11, column 6: byte 85: \"1.5\": column \"a\": a number with a \
             fraction or an exponent does not read as int32. Declare another \
             type (with_type)."
            (fails ~f:(J.with_type "a" (any Type.int32) f) s));
      test "rows limits the sample" (fun () ->
          equal string
            "json (1 column, lines), types sniffed from 1 row\n  a int64"
            (format_text
               (sniffed ~rows:1 (lines [ {|{"a":1}|}; {|{"a":1.5}|} ]))));
      test "sniff leaves the reader as it was" (fun () ->
          let s = lines [ {|{"a":1}|}; {|{"a":2}|} ] in
          let r = Reader.of_string s in
          let f = Error.get_ok (J.sniff r) in
          equal int 2 (Talon.rows (Error.get_ok (J.decode f r))));
      test "errors" (fun () ->
          let e = fails in
          equal string "the input holds no record" (e "");
          equal string "the input holds no record" (e "[]");
          equal string "no sampled row names a column" (e "{}\n");
          equal string
            "line 1, column 1: byte 0: a row is a JSON object, and this is a \
             number"
            (e "1\n");
          equal string
            "line 1, column 8: byte 7: the object names \"a\" twice"
            (e {|{"a":1,"a":2}|}));
      test "Invalid_argument" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"Talon_json.sniff: rows is 0, not positive")
            (fun () -> ignore (sniff ~rows:0 "{}"));
          raises_match
            (Exn.invalid_arg
               ~sub:"Talon_json.format: a format has at least one column")
            (fun () -> ignore (J.format []));
          raises_match
            (Exn.invalid_arg
               ~sub:"Talon_json.format: two columns are named \"a\"")
            (fun () ->
              ignore (J.format [ ("a", any Type.bool); ("a", any Type.bool) ]));
          raises_match
            (Exn.invalid_arg
               ~sub:"Talon_json.format: column \"a\": JSON does not read clock[ms]")
            (fun () -> ignore (J.format [ ("a", any (Type.clock Ms)) ]));
          raises_match
            (Exn.invalid_arg
               ~sub:"Talon_json.with_type: no column \"z\" in the format")
            (fun () ->
              ignore (J.with_type "z" (any Type.bool)
                        (J.format [ ("a", any Type.bool) ]))));
    ]

(* Values *)

let flat = J.format [ ("a", any Type.int64); ("b", any Type.string) ]

let rows_of s =
  Error.get_ok (read (J.format [ ("a", any Type.string) ]) s)

let values =
  group "values"
    [
      test "missing keys and null read as null, extra keys are skipped" (fun () ->
          let t =
            Error.get_ok
              (read flat
                 (lines
                    [
                      {|{"a":1,"b":"x","c":[{"deep":true}]}|};
                      {|{"b":null}|};
                      {|{"a":null}|};
                    ]))
          in
          equal strings [ Some "1"; None; None ] (ints t "a");
          equal strings [ Some "x"; None; None ] (cell t "b"));
      test "names in any order" (fun () ->
          let t = Error.get_ok (read flat {|{"b":"x","a":2}|}) in
          equal strings [ Some "2" ] (ints t "a"));
      test "int64 and float64 by format" (fun () ->
          let f = J.format [ ("i", any Type.int64); ("x", any Type.float64) ] in
          let t =
            Error.get_ok
              (read f
                 {|{"i":9223372036854775807,"x":-9223372036854775808}|})
          in
          equal strings [ Some "9223372036854775807" ] (ints t "i");
          equal floats [ Some (-9.2233720368547758e18) ] (doubles t "x"));
      test "bool, date and categorical" (fun () ->
          let dict = Type.categorical [| "red"; "green" |] in
          let f =
            J.format
              [
                ("b", any Type.bool);
                ("d", any Type.date);
                ("c", any dict);
              ]
          in
          let t =
            Error.get_ok
              (read f {|{"b":true,"d":"2024-02-29","c":"green"}|})
          in
          equal strings [ Some "2024-02-29" ]
            (Array.to_list
               (Column.options (Type.kind Type.string) (Column.print (column t "d"))));
          equal strings [ Some "green" ]
            (Array.to_list
               (Column.options (Type.kind Type.string) (Column.print (column t "c")))));
      test "records and lists, nested nulls and absences" (fun () ->
          let item =
            Type.record
              [ ("x", any Type.int64); ("y", any (Type.list Type.string)) ]
          in
          let f = J.format [ ("r", any item); ("l", any (Type.list Type.int64)) ] in
          let t =
            Error.get_ok
              (read f
                 (lines
                    [
                      {|{"r":{"y":["a",null],"x":1},"l":[1,null,3]}|};
                      {|{"r":{"x":null},"l":[]}|};
                      {|{"r":null,"l":null}|};
                    ]))
          in
          equal (list string)
            [ {|{x=1,y=["a",null]}|}; "{x=null,y=null}"; "null" ]
            (rendered t "r");
          equal (list string) [ "[1,null,3]"; "[]"; "null" ] (rendered t "l"));
      test "RFC 8259 escapes" (fun () ->
          let t =
            rows_of
              {|{"a":"q\" s\\ f\/ b\b f\f n\n r\r t\t é 😀 A"}|}
          in
          equal strings
            [ Some "q\" s\\ f/ b\b f\012 n\n r\r t\t \xc3\xa9 \xf0\x9f\x98\x80 A" ]
            (cell t "a"));
      test "a string column reads other values as their text" (fun () ->
          let t = rows_of (lines [ {|{"a":1.50}|}; {|{"a":true}|}; {|{"a":{"x":[1]}}|} ]) in
          equal strings [ Some "1.50"; Some "true"; Some {|{"x":[1]}|} ] (cell t "a"));
      test "a duplicate of a key no column reads is skipped" (fun () ->
          let t =
            Error.get_ok
              (read (J.format [ ("a", any Type.int64) ])
                 {|{"a":1,"z":2,"z":3,"q":{"z":1,"z":2}}|})
          in
          equal strings [ Some "1" ] (ints t "a"));
      test "whitespace, byte order mark and crlf" (fun () ->
          let t =
            Error.get_ok
              (read (J.format [ ("a", any Type.int64) ])
                 "\xef\xbb\xbf  {  \"a\"  :  1  }  \r\n\r\n\t\n{\"a\":2}\r")
          in
          equal strings [ Some "1"; Some "2" ] (ints t "a"));
      test "array framing, multi-line and a trailing space" (fun () ->
          let f =
            J.format ~framing:J.Array [ ("a", any Type.int64) ]
          in
          let t =
            Error.get_ok
              (read f "[\n  {\"a\": 1},\n  {\"a\":\n    2}\n]  \n ")
          in
          equal strings [ Some "1"; Some "2" ] (ints t "a"));
      test "an empty input and an empty array have no rows" (fun () ->
          equal int 0 (Talon.rows (Error.get_ok (read flat "")));
          let f = J.format ~framing:J.Array [ ("a", any Type.int64) ] in
          equal int 0 (Talon.rows (Error.get_ok (read f "[  ]")));
          equal string
            "line 1, column 1: byte 0: the input ends before the array does"
            (fails ~f ""));
    ]

(* Failures *)

let cases =
  let int_col = J.format [ ("a", any Type.int64) ] in
  let float_col = J.format [ ("a", any Type.float64) ] in
  let str_col = J.format [ ("a", any Type.string) ] in
  let bool_col = J.format [ ("a", any Type.bool) ] in
  let arr = J.format ~framing:J.Array [ ("a", any Type.int64) ] in
  [
    ( "a fraction in an integer column",
      int_col,
      {|{"a":1.5}|},
      "line 1, column 6: byte 5: \"1.5\": column \"a\": a number with a \
       fraction or an exponent does not read as int64. Declare another \
       type (with_type)." );
    ( "integer overflow",
      int_col,
      {|{"a":9223372036854775808}|},
      "line 1, column 6: byte 5: \"9223372036854775808\": column \"a\": cannot \
       read as int64: out of range. Declare another type (with_type)."
    );
    ( "float overflow",
      float_col,
      {|{"a":1e999}|},
      "line 1, column 6: byte 5: column \"a\": the number exceeds float64. \
       Declare another type (with_type)." );
    ( "float32 overflow without an exponent",
      J.format [ ("a", any Type.float32) ],
      {|{"a":1000000000000000000000000000000000000000}|},
      "line 1, column 6: byte 5: column \"a\": the number exceeds float32. \
       Declare another type (with_type)." );
    ( "float16 overflow",
      J.format [ ("a", any Type.float16) ],
      {|{"a":70000}|},
      "line 1, column 6: byte 5: column \"a\": the number exceeds float16. \
       Declare another type (with_type)." );
    ( "a carriage return inside a value",
      int_col,
      "{\"a\":\r1}",
      "line 1, column 6: byte 5: the record ends in the middle of a value" );
    ( "a carriage return between records",
      int_col,
      "{\"a\":1}\r{\"a\":2}",
      "line 1, column 8: byte 7: a carriage return not before a line feed" );
    ( "a string in a number column",
      float_col,
      {|{"a":"1.5"}|},
      "line 1, column 6: byte 5: column \"a\": a string does not read as \
       float64, which holds a number. Declare another type (with_type)." );
    ( "a number in a boolean column",
      bool_col,
      {|{"a":1}|},
      "line 1, column 6: byte 5: column \"a\": a number does not read as bool, \
       which holds a boolean. Declare another type (with_type)." );
    ( "leading zero",
      int_col,
      {|{"a":01}|},
      "line 1, column 7: byte 6: a number has a leading zero" );
    ( "leading plus",
      float_col,
      {|{"a":+1}|},
      "line 1, column 6: byte 5: expected a value, found the byte '+'" );
    ( "bare point",
      float_col,
      {|{"a":.5}|},
      "line 1, column 6: byte 5: expected a value, found the byte '.'" );
    ( "trailing point",
      float_col,
      {|{"a":1.}|},
      "line 1, column 8: byte 7: expected a digit, found the byte '}'" );
    ( "bare exponent",
      float_col,
      {|{"a":1e}|},
      "line 1, column 8: byte 7: expected a digit, found the byte '}'" );
    ( "NaN",
      float_col,
      {|{"a":NaN}|},
      "line 1, column 6: byte 5: expected a value, found the byte 'N'" );
    ( "a raw control character in a string",
      str_col,
      "{\"a\":\"x\ty\"}",
      "line 1, column 8: byte 7: a control character in a string is written \
       as an escape" );
    ( "invalid UTF-8 in a string",
      str_col,
      "{\"a\":\"x\xFFy\"}",
      "line 1, column 8: byte 7: a string is not valid UTF-8" );
    ( "truncated UTF-8",
      str_col,
      "{\"a\":\"x\xC3\"}",
      "line 1, column 8: byte 7: a string is not valid UTF-8" );
    ( "an unknown escape",
      str_col,
      {|{"a":"\x"}|},
      "line 1, column 7: byte 6: the escape \\x is not JSON's" );
    ( "a short unicode escape",
      str_col,
      {|{"a":"\u12"}|},
      "line 1, column 7: byte 6: a \\u escape needs four hexadecimal digits" );
    ( "a lone high surrogate",
      str_col,
      {|{"a":"\ud83d"}|},
      "line 1, column 7: byte 6: an unpaired surrogate escape" );
    ( "a lone low surrogate",
      str_col,
      {|{"a":"\ude00"}|},
      "line 1, column 7: byte 6: an unpaired surrogate escape" );
    ( "a string that does not end",
      str_col,
      {|{"a":"xyz|},
      "line 1, column 6: byte 5: the string does not end on its line" );
    ( "a duplicate key",
      int_col,
      {|{"a":1,"a":2}|},
      "line 1, column 8: byte 7: the object names \"a\" twice" );
    ( "a trailing comma in an object",
      int_col,
      {|{"a":1,}|},
      "line 1, column 8: byte 7: expected a string as a name, found the byte \
       '}'" );
    ( "a missing colon",
      int_col,
      {|{"a" 1}|},
      "line 1, column 6: byte 5: expected ':', found the byte '1'" );
    ( "an unclosed object",
      int_col,
      "{\"a\":1\n{\"a\":2}",
      "line 1, column 7: byte 6: the record ends in the middle of a value" );
    ( "two values on a line",
      int_col,
      {|{"a":1} {"a":2}|},
      "line 1, column 9: byte 8: the line holds more than one JSON value" );
    ( "a row that is no object",
      int_col,
      "[1]\n",
      "line 1, column 1: byte 0: a row is a JSON object, and this is an array"
    );
    ( "a later row names its line",
      int_col,
      "{\"a\":1}\n{\"a\":\"x\"}\n",
      "line 2, column 6: byte 13: column \"a\": a string does not read as \
       int64, which holds a number. Declare another type (with_type)." );
    ( "a trailing comma in an array",
      arr,
      {|[{"a":1},]|},
      "line 1, column 10: byte 9: a comma is followed by the array's end" );
    ( "an array that does not end",
      arr,
      {|[{"a":1}|},
      "line 1, column 9: byte 8: the input ends before the array does" );
    ( "bytes after the array",
      arr,
      {|[{"a":1}] x|},
      "line 1, column 11: byte 10: bytes follow the array" );
    ( "a bracket mismatch",
      arr,
      {|[{"a":1}}|},
      "line 1, column 9: byte 8: expected a comma or the array's end, found \
       the byte '}'" );
  ]

let failures =
  group "failures"
    (List.map
       (fun (name, f, s, want) ->
         test name (fun () -> equal string want (fails ~f s)))
       cases)

(* Depth *)

let depth = 1_000_000

let deep =
  group "deep nesting"
    [
      test "a deep value in a skipped key costs no stack" (fun () ->
          let s =
            {|{"keep":1,"deep":|} ^ String.make depth '[' ^ String.make depth ']'
            ^ {|}|}
          in
          let f = J.format [ ("keep", any Type.int64) ] in
          equal strings [ Some "1" ] (ints (Error.get_ok (read f s)) "keep"));
      test "a deep value reads as a string column" (fun () ->
          let v = String.make depth '[' ^ String.make depth ']' in
          let s = {|{"a":|} ^ v ^ {|}|} in
          let f = J.format [ ("a", any Type.string) ] in
          equal int (String.length v)
            (String.length
               (Option.get (List.hd (cell (Error.get_ok (read f s)) "a")))));
      test "sniffing a deep value gives a shallow string" (fun () ->
          let v = String.make depth '[' ^ String.make depth ']' in
          let s = {|{"a":|} ^ v ^ "}\n" in
          let t = Error.get_ok (read (sniffed s) s) in
          equal int 1 (Talon.rows t));
      test "deep and malformed is an error" (fun () ->
          let s = {|{"a":|} ^ String.make depth '[' ^ "}" in
          let f = J.format [ ("a", any Type.string) ] in
          is_error (read f s));
      test "a value deeper than its type is a type error" (fun () ->
          let f = J.format [ ("a", any (Type.list Type.int64)) ] in
          equal string
            "line 1, column 7: byte 6: column \"a[]\": an array does not read as \
             int64, which holds a number. Declare another type (with_type)."
            (fails ~f {|{"a":[[1]]}|}));
    ]

(* Batches *)

(* Rows of about 40 bytes, so that [n] rows cross many one-MiB batches. *)
let keep names q =
  Query.select (if names = [] then [] else Expr.[ keep (Sel.names names) ]) q

let many n =
  String.concat ""
    (List.init n (fun i -> Printf.sprintf {|{"id":%d,"s":"row %d"}|} i i ^ "\n"))

let many_array n =
  "["
  ^ String.concat ",\n"
      (List.init n (fun i -> Printf.sprintf {|{"id":%d}|} i))
  ^ "]"

let id = J.format [ ("id", any Type.int64) ]

let batches =
  group "batches"
    [
      prop ~count:20 "the reader's slices do not change a table"
        Gen.(pair (int_range 1 5000) (int_range 1 64))
        (fun (n, slice_length) ->
          let s = many n in
          let t = Error.get_ok (read ~slice_length (sniffed s) s) in
          equal int n (Talon.rows t);
          equal (option string) (Some (string_of_int (n - 1)))
            (List.nth (ints t "id") (n - 1)));
      test "lines cross batches" (fun () ->
          let n = 200_000 in
          let s = many n in
          let t = Error.get_ok (read id s) in
          equal int n (Talon.rows t);
          greater ~than:1 int (List.length (Talon.batches t)));
      test "an array crosses batches, elements intact" (fun () ->
          let n = 150_000 in
          let f = J.format ~framing:J.Array [ ("id", any Type.int64) ] in
          let t = Error.get_ok (read f (many_array n)) in
          equal int n (Talon.rows t);
          greater ~than:1 int (List.length (Talon.batches t));
          equal (option string) (Some (string_of_int (n - 1)))
            (List.nth (ints t "id") (n - 1)));
      test "a large top-level array streams: batches come before it ends"
        (fun () ->
          let n = 150_000 in
          let open_ () = Reader.of_string (many_array n ^ " x") in
          let f = J.format ~framing:J.Array [ ("id", any Type.int64) ] in
          let seen = ref 0 in
          let r =
            Query.fold
              (Query.of_source (J.source f open_))
              ~init:() (fun () b -> seen := !seen + Talon.rows b)
          in
          is_error r;
          greater ~than:0 int !seen;
          at_most ~than:n int !seen);
      test "a limit stops before a later error" (fun () ->
          let s = "{\"id\":1}\n{\"id\":\"x\"}\n" in
          let t =
            Error.get_ok
              (Query.run
                 (Query.slice ~offset:0 ~length:1
                    (Query.of_source (J.source id (fun () -> Reader.of_string s)))))
          in
          equal int 1 (Talon.rows t));
      test "a source reads only its columns, but checks every row" (fun () ->
          let f = J.format [ ("a", any Type.int64); ("b", any Type.string) ] in
          let s = {|{"a":1,"b":"x"}|} ^ "\n" in
          let q =
            keep [ "b" ]
              (Query.of_source (J.source f (fun () -> Reader.of_string s)))
          in
          equal strings [ Some "x" ] (cell (Error.get_ok (Query.run q)) "b");
          let broken = "{\"a\":1,\"a\":2,\"b\":\"x\"}\n" in
          is_error
            (Query.run
               (keep [ "b" ]
                  (Query.of_source
                     (J.source f (fun () -> Reader.of_string broken))))));
    ]

(* Files *)

let files =
  group "files"
    [
      test "a file is sniffed, named and read" (fun () ->
          let path = Filename.temp_file "talon_json" ".ndjson" in
          Fun.protect ~finally:(fun () -> Sys.remove path) @@ fun () ->
          Out_channel.with_open_bin path (fun oc ->
              output_string oc "{\"x\":1}\n{\"x\":\"oops\"}\n");
          let source = Error.get_ok (J.file ~format:(J.format [ ("x", any Type.int64) ]) path) in
          let q = Query.of_source source in
          equal string
            (Printf.sprintf
               "%s:2:6: byte 13: column \"x\": a string does not read as int64, \
                which holds a number. Declare another type (with_type)."
               path)
            (match Query.run q with
            | Ok _ -> "read"
            | Error e -> error_text e));
      test "a missing file is an error" (fun () ->
          is_error (J.file "/nonexistent/talon.ndjson"));
    ]

(* The corpus: files written by pinned Python tools and what Python reads. *)

let read_file path = In_channel.with_open_bin path In_channel.input_all

let corpus =
  let golden =
    String.split_on_char '\n' (read_file "golden/values.txt")
    |> List.filter (fun l -> l <> "")
  in
  let files =
    List.fold_left
      (fun acc l ->
        let f = List.hd (String.split_on_char '\t' l) in
        if List.mem f acc then acc else acc @ [ f ])
      [] golden
  in
  let lines_of file =
    let s = read_file (Filename.concat "support" file) in
    let f = sniffed s in
    let t = Error.get_ok (read f s) in
    List.map
      (fun name ->
        let (Type.Any ty) = Column.type_ (column t name) in
        String.concat "\t"
          [
            file;
            name;
            Format.asprintf "%a" Type.pp ty;
            String.concat "\x1f" (rendered t name);
          ])
      (names t)
  in
  test "the corpus reads as Python reads it" (fun () ->
      equal (list string) golden (List.concat_map lines_of files))

(* Round trip *)

let rec writable (Type.Any t) =
  match t with
  | Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64
  | Float16 | Float32 | Float64 | String | Binary | Categorical _ | Date
  | Datetime _ ->
      true
  | List e -> writable (any e)
  | Record fields ->
      fields <> [] && List.for_all (fun (_, t) -> writable t) fields
  | Clock _ | Duration _ | Tensor _ | Ext _ -> false

let table_w =
  let pp ppf t =
    Format.fprintf ppf "%a: %d rows" Schema.pp (schema t) (rows t)
  in
  Testable.make ~pp ~equal:Talon.equal

let encode f t =
  let b = Buffer.create 256 in
  Error.get_ok (J.encode f (Query.of_table t) (Bytesrw.Bytes.Writer.of_buffer b));
  Buffer.contents b

(* A table encode refuses (a NaN, an infinity, a binary value that is not
   UTF-8) has no JSON: [Error] is the contract, stated in its own test. *)
let encoded f t =
  let b = Buffer.create 256 in
  match J.encode f (Query.of_table t) (Bytesrw.Bytes.Writer.of_buffer b) with
  | Ok () -> Some (Buffer.contents b)
  | Error _ -> None

let round_trip =
  prop ~count:300 "written tables read back per type"
    (Gen.pair G.sample Gen.bool)
    (fun (G.Sample (ty, vs), array) ->
      assume (writable (any ty));
      let framing = if array then J.Array else J.Lines in
      let t = Talon.v [ ("a", Column.of_options ty vs) ] in
      let f = J.format ~framing [ ("a", any ty) ] in
      assume (Option.is_some (encoded f t));
      cover "a list" (match ty with List _ -> true | _ -> false);
      cover "a record" (match ty with Record _ -> true | _ -> false);
      cover "no rows" (Talon.rows t = 0);
      Law.round_trip table_w string (encode f)
        (fun s -> Error.get_ok (read f s))
        t)

let record_round_trip =
  let item = Type.record [ ("x", any Type.int64); ("y", any Type.string) ] in
  test "records and lists of records round trip" (fun () ->
      let t =
        Talon.v
          [
            ( "r",
              Column.of_options (Type.list item)
                [|
                  Some
                    [|
                      Record.(
                        add (Type.kind Type.string) "y" (Some "a\"b\n")
                          (add (Type.kind Type.int64) "x" (Some 1) empty))
                    |];
                  Some [||];
                  None;
                |] );
          ]
      in
      let f = J.format [ ("r", any (Type.list item)) ] in
      equal table_w t (Error.get_ok (read f (encode f t))))

let writing =
  group "writing"
    [
      test "the text of each framing" (fun () ->
          let t = Talon.v [ ("a", Column.of_options Type.int64 [| Some 1; None |]) ] in
          equal string "{\"a\":1}\n{\"a\":null}\n"
            (encode (J.format [ ("a", any Type.int64) ]) t);
          equal string "[\n{\"a\":1},\n{\"a\":null}\n]\n"
            (encode (J.format ~framing:J.Array [ ("a", any Type.int64) ]) t);
          equal string "[\n]\n"
            (encode (J.format ~framing:J.Array [ ("a", any Type.int64) ])
               (Talon.v [ ("a", Column.of_options Type.int64 [||]) ])));
      test "strings escape what JSON requires, floats keep their fraction"
        (fun () ->
          let f = J.format [ ("s", any Type.string); ("x", any Type.float64) ] in
          equal string
            "{\"s\":\"a\\\"b\\\\c\\n\\u0001\xc3\xa9\",\"x\":150.0}\n\
             {\"s\":null,\"x\":-0.0}\n"
            (encode f
               (Talon.v
                  [
                    ( "s",
                      Column.of_options Type.string
                        [| Some "a\"b\\c\n\001\xc3\xa9"; None |] );
                    ( "x",
                      Column.of_options Type.float64 [| Some 150.; Some (-0.) |]
                    );
                  ])));
      test "NaN and invalid UTF-8 are errors at their row" (fun () ->
          let f = J.format [ ("x", any Type.float64) ] in
          let err t =
            let b = Buffer.create 16 in
            match J.encode f t (Bytesrw.Bytes.Writer.of_buffer b) with
            | Ok () -> "wrote"
            | Error e -> Buffer.contents b ^ error_text e
          in
          equal string
            "{\"x\":1.0}\nline 2: column \"x\": JSON has no text for an \
             infinity or NaN. Filter them out or replace them with null first."
            (err
               (Query.of_table
                  (Talon.v
                     [
                       ( "x",
                         Column.of_options Type.float64
                           [| Some 1.; Some Float.nan |] );
                     ])));
          let fb = J.format [ ("b", any Type.binary) ] in
          let b = Buffer.create 16 in
          is_error
            (J.encode fb
               (Query.of_table
                  (Talon.v
                     [
                       ( "b",
                         Column.of_options Type.binary
                           [| Some (Binary.of_string "\xff") |] );
                     ]))
               (Bytesrw.Bytes.Writer.of_buffer b)));
      test "a mismatched schema raises" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Talon_json.encode") (fun () ->
              ignore
                (J.encode
                   (J.format [ ("x", any Type.bool) ])
                   (Query.of_table
                      (Talon.v [ ("a", Column.of_options Type.int64 [||]) ]))
                   (Bytesrw.Bytes.Writer.of_buffer (Buffer.create 1)))));
    ]

let () =
  exit
    (run "talon.json"
       [
         sniffing;
         values;
         failures;
         deep;
         batches;
         files;
         corpus;
         round_trip;
         record_round_trip;
         writing;
       ])
