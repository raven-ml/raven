(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module J = Talon_json
module Reader = Bytesrw.Bytes.Reader
module Writer = Bytesrw.Bytes.Writer
module G = Talon_gen

let any t = Type.Any t
let lines ss = String.concat "\n" ss ^ "\n"
let strings = list (option string)
let error_text e = Format.asprintf "%a" Error.pp e
let format_text f = Format.asprintf "%a" J.pp_format f
let plan_text q = Format.asprintf "%a" Query.pp q

(* [one ty] is the format of rows whose one column, [a], reads as [ty]. *)
let one ?framing ty = J.format ?framing [ ("a", any ty) ]
let read ?slice_length f s = J.decode f (Reader.of_string ?slice_length s)
let sniffed ?rows s = Error.get_ok (J.sniff ?rows (Reader.of_string s))

let decode_error f s =
  match read f s with
  | Ok _ -> failf "decode read %S" s
  | Error e -> error_text e

let sniff_error s =
  match J.sniff (Reader.of_string s) with
  | Ok _ -> failf "sniff read %S" s
  | Error e -> error_text e

(* [place line column byte] starts an error found at a byte, as Error.pp writes
   it. *)
let place line column byte =
  Printf.sprintf "line %d, column %d: byte %d: " line column byte

(* [printed t name] is the column [name] of [t] as Column.print writes it. *)
let printed t name =
  Array.to_list (Column.options Kind.string (Column.print (column t name)))

(* [sniffed_type rows] is the line of the column [a] in the format sniffed from
   [rows]. *)
let sniffed_type rows =
  List.nth (String.split_on_char '\n' (format_text (sniffed (lines rows)))) 1

(* [nested n s] is [n] JSON arrays around [s]. *)
let nested n s = String.make n '[' ^ s ^ String.make n ']'

(* [list_type n s] is [n] talon lists around the type text [s]. *)
let list_type n s =
  String.concat "" (List.init n (Fun.const "list[")) ^ s ^ String.make n ']'

let table_w =
  let pp ppf t =
    Format.fprintf ppf "%a: %d rows" Schema.pp (schema t) (rows t)
  in
  Testable.make ~pp ~equal:Talon.equal

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
          ^ String.concat "," (List.map (fun (f, r) -> f ^ "=" ^ r.(i)) parts)
          ^ "}")
  | String, _ ->
      let v = Column.options Kind.string c in
      rows (fun i -> "\"" ^ escape (Option.get v.(i)) ^ "\"")
  | Binary, _ ->
      let v =
        Column.options Kind.string
          (Column.print (Result.get_ok (Column.parse (any Type.string) c)))
      in
      rows (fun i -> "\"" ^ escape (Option.get v.(i)) ^ "\"")
  | _ ->
      let v = Column.options Kind.string (Column.print c) in
      rows (fun i -> Option.get v.(i))

let rendered t name = render (column t name)

(* [in_small_stack f] is [f ()], computed in a forked child whose OCaml stack
   holds 256K words (2 MiB), so that a reader recursing once per level of a
   million levels overflows there. Without fork the test skips. *)
let in_small_stack f =
  let rd, wr = Unix.pipe () in
  match Unix.fork () with
  | exception (Invalid_argument _ | Failure _ | Unix.Unix_error _) ->
      Unix.close rd;
      Unix.close wr;
      skip ~reason:"this platform cannot fork" ()
  | 0 ->
      Unix.close rd;
      Gc.set { (Gc.get ()) with stack_limit = 262_144 };
      let out =
        match f () with
        | s -> s
        | exception e -> "raised " ^ Printexc.to_string e
      in
      let oc = Unix.out_channel_of_descr wr in
      output_string oc out;
      close_out oc;
      Unix._exit 0
  | pid ->
      Unix.close wr;
      let ic = Unix.in_channel_of_descr rd in
      let out = In_channel.input_all ic in
      close_in ic;
      ignore (Unix.waitpid [] pid);
      out

(* A value a million levels deep. *)
let depth = 1_000_000
let deep = lazy (nested depth "")

(* Rows of ids, of about 15 bytes each: 100,000 of them hold about 1.4 MB, past
   a batch of one MiB. *)
let ids = 100_000

let id_lines =
  lazy
    (String.concat ""
       (List.init ids (fun i -> Printf.sprintf {|{"id":%d}|} i ^ "\n")))

let id_array =
  lazy
    ("["
    ^ String.concat ",\n" (List.init ids (Printf.sprintf {|{"id":%d}|}))
    ^ "]")

let id_format framing = J.format ~framing [ ("id", any Type.int64) ]

let id_text framing =
  Lazy.force (match framing with J.Lines -> id_lines | Array -> id_array)

(* The table of each framing's text, read once. *)
let id_tables =
  lazy
    (List.map
       (fun framing ->
         (framing, Error.get_ok (read (id_format framing) (id_text framing))))
       [ J.Lines; J.Array ])

let id_table framing = List.assoc framing (Lazy.force id_tables)

(* Streaming *)

let streaming =
  group "streaming"
    [
      test "the first batches of an array arrive before its end is read"
        (fun () ->
          let open_ () = Reader.of_string (Lazy.force id_array ^ " x") in
          let seen = ref 0 in
          let r =
            Query.fold
              (Query.of_source (J.source (id_format Array) open_))
              ~init:()
              (fun () b -> seen := !seen + rows b)
          in
          is_error r;
          greater ~than:0 int !seen);
      test "a row longer than a batch reads whole" (fun () ->
          let long = String.make 1_200_000 'a' in
          let t =
            Error.get_ok
              (read (one Type.string)
                 (lines [ {|{"a":"|} ^ long ^ {|"}|}; {|{"a":"b"}|} ]))
          in
          equal strings [ Some long; Some "b" ] (printed t "a"));
    ]

(* Syntax *)

let syntax =
  group "syntax"
    [
      cases
        "a value that breaks RFC 8259 fails at the byte of the value or escape \
         that breaks it"
        ~id:fst
        [
          ({|{"a":01}|}, (7, 6));
          ({|{"a":+1}|}, (6, 5));
          ({|{"a":.5}|}, (6, 5));
          ({|{"a":1.}|}, (8, 7));
          ({|{"a":1e}|}, (8, 7));
          ({|{"a":1e+}|}, (9, 8));
          ({|{"a":-}|}, (7, 6));
          ({|{"a":NaN}|}, (6, 5));
          ({|{"a":tru}|}, (6, 5));
          ("{\"a\":\"x\ty\"}", (8, 7));
          ("{\"a\":\"x\xFFy\"}", (8, 7));
          ("{\"a\":\"x\xC3\"}", (8, 7));
          ("{\"a\":\"\xC0\x80\"}", (7, 6));
          ("{\"a\":\"\xED\xA0\x80\"}", (7, 6));
          ({|{"a":"\x"}|}, (7, 6));
          ({|{"a":"\u12"}|}, (7, 6));
          ({|{"a":"\ud83d"}|}, (7, 6));
          ({|{"a":"\ude00"}|}, (7, 6));
          ({|{"a":"xyz|}, (6, 5));
          ({|{"a":1,}|}, (8, 7));
          ({|{"a" 1}|}, (6, 5));
        ]
        (fun (row, (column, byte)) ->
          starts_with ~prefix:(place 1 column byte)
            (decode_error (one Type.string) row));
      cases "a malformed value of a key no column reads fails" ~id:fst
        [
          ({|{"a":1,"z":[1,}|}, (15, 14)); ("{\"a\":1,\"z\":\"\xFF\"}", (13, 12));
        ]
        (fun (row, (column, byte)) ->
          starts_with ~prefix:(place 1 column byte)
            (decode_error (one Type.int64) row));
      test "a string reads any Unicode scalar value, escaped or raw" (fun () ->
          let t =
            Error.get_ok
              (read (one Type.string)
                 {|{"a":"q\" s\\ f\/ b\b f\f n\n r\r t\t é 😀 \u0041 \ud83d\ude00 \u0000"}|})
          in
          equal strings
            [
              Some
                "q\" s\\ f/ b\b f\012 n\n\
                \ r\r t\t \xc3\xa9 \xf0\x9f\x98\x80 A \xf0\x9f\x98\x80 \000";
            ]
            (printed t "a"));
      test "a byte order mark is skipped, and line 1 starts after it" (fun () ->
          let t =
            Error.get_ok (read (one Type.int64) "\xEF\xBB\xBF{\"a\":1}")
          in
          equal strings [ Some "1" ] (printed t "a");
          starts_with ~prefix:(place 1 6 8)
            (decode_error (one Type.int64) "\xEF\xBB\xBF{\"a\":x}"));
      (* Rounded to nearest, a number past 2^1024 is an infinity, which a float
         column refuses. *)
      cases "a number of any length reads as its column's type holds it"
        ~id:(fun (_, s, _) -> s)
        [
          (any Type.float64, "1." ^ String.make 500 '0' ^ "1", Some "1");
          (any Type.float64, "0." ^ String.make 400 '0' ^ "1e401", Some "1");
          (any Type.float64, "1e-99999", Some "0");
          (any Type.float64, "-1e-99999", Some "-0");
          (any Type.float64, "1e99999", None);
          ( any Type.string,
            "1" ^ String.make 400 '0',
            Some ("1" ^ String.make 400 '0') );
        ]
        (fun (Type.Any ty, number, want) ->
          let row = {|{"a":|} ^ number ^ "}" in
          match want with
          | Some v ->
              equal strings [ Some v ]
                (printed (Error.get_ok (read (one ty) row)) "a")
          | None ->
              starts_with ~prefix:(place 1 6 5) (decode_error (one ty) row));
      test "a value a million levels deep in a skipped key is read past"
        (fun () ->
          let row = {|{"keep":1,"deep":|} ^ Lazy.force deep ^ "}" in
          let f = J.format [ ("keep", any Type.int64) ] in
          equal string "1"
            (in_small_stack (fun () ->
                 Option.get
                   (List.hd (printed (Error.get_ok (read f row)) "keep")))));
      test "a string column holds a value a million levels deep as its text"
        (fun () ->
          let row = {|{"a":|} ^ Lazy.force deep ^ "}" in
          equal string
            (string_of_int (2 * depth))
            (in_small_stack (fun () ->
                 let t = Error.get_ok (read (one Type.string) row) in
                 string_of_int
                   (String.length (Option.get (List.hd (printed t "a")))))));
      test "a malformed value a million levels deep fails at its byte"
        (fun () ->
          let row = {|{"a":|} ^ String.make depth '[' ^ "}" in
          let prefix = place 1 (depth + 6) (depth + 5) in
          equal string prefix
            (in_small_stack (fun () ->
                 let e = decode_error (one Type.string) row in
                 String.sub e 0 (min (String.length e) (String.length prefix)))));
      cases "a key the format reads, named twice, fails"
        ~id:(fun (_, row, _) -> row)
        [
          (one Type.int64, {|{"a":1,"a":2}|}, (8, 7));
          (one Type.int64, {|{"a":1,"\u0061":2}|}, (8, 7));
          ( one (Type.record [ ("x", any Type.int64) ]),
            {|{"a":{"x":1,"x":2}}|},
            (13, 12) );
        ]
        (fun (f, row, (column, byte)) ->
          starts_with ~prefix:(place 1 column byte) (decode_error f row));
      test "a key the format does not read may repeat" (fun () ->
          let t =
            Error.get_ok
              (read (one Type.int64) {|{"a":1,"z":2,"z":3,"q":{"z":1,"z":2}}|})
          in
          equal strings [ Some "1" ] (printed t "a"));
      (* NDJSON: "The newline character MAY be preceded by a carriage
         return". *)
      cases "lines may end in CRLF, a CR or the input's end, and be blank"
        ~id:(Printf.sprintf "%S")
        [
          "{\"a\":1}\r\n{\"a\":2}\r\n";
          "{\"a\":1}\n\n \t\n{\"a\":2}";
          "{\"a\":1}\n{\"a\":2}\r";
          "  {  \"a\"  :  1  }  \n{\"a\":2}";
        ] (fun text ->
          equal strings [ Some "1"; Some "2" ]
            (printed (Error.get_ok (read (one Type.int64) text)) "a"));
      cases "a line holds one value, with no line break inside it" ~id:fst
        [
          ("{\"a\":\r1}", (6, 5));
          ("{\"a\":1}\r{\"a\":2}", (8, 7));
          ("{\"a\":1\n{\"a\":2}", (7, 6));
          ("{\"a\":\"x\ny\"}", (6, 5));
          ({|{"a":1} {"a":2}|}, (9, 8));
        ]
        (fun (text, (column, byte)) ->
          starts_with ~prefix:(place 1 column byte)
            (decode_error (one Type.string) text));
      test "an array's rows may span lines" (fun () ->
          let t =
            Error.get_ok
              (read
                 (one ~framing:Array Type.int64)
                 "[\n  {\"a\": 1},\n  {\"a\":\n    2}\n]  \n ")
          in
          equal strings [ Some "1"; Some "2" ] (printed t "a"));
      cases "only whitespace follows an array, and commas separate its rows"
        ~id:fst
        [
          ({|[{"a":1},]|}, (10, 9));
          ({|[{"a":1}|}, (9, 8));
          ({|[{"a":1}] x|}, (11, 10));
          ({|[{"a":1}}|}, (9, 8));
        ]
        (fun (text, (column, byte)) ->
          starts_with ~prefix:(place 1 column byte)
            (decode_error (one ~framing:Array Type.int64) text));
    ]

(* Values *)

let flat = J.format [ ("a", any Type.int64); ("b", any Type.string) ]
let read_one ty value = Error.get_ok (read (one ty) ({|{"a":|} ^ value ^ "}"))
let fails_one ty value = decode_error (one ty) ({|{"a":|} ^ value ^ "}")
let type_text (Type.Any t) = Format.asprintf "%a" Type.pp t

let values =
  group "values"
    [
      test "a key the format does not name is skipped, an absent one is null"
        (fun () ->
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
          equal strings [ Some "1"; None; None ] (printed t "a");
          equal strings [ Some "x"; None; None ] (printed t "b"));
      test "a row's keys read in any order" (fun () ->
          let t = Error.get_ok (read flat {|{"b":"x","a":2}|}) in
          equal strings [ Some "2" ] (printed t "a"));
      (* Error.pp quotes a text, a quote escaped. *)
      cases "a row that is not an object fails with its place and text" ~id:fst
        [
          ("1", {|"1"|});
          ({|"x"|}, {|"\"x\""|});
          ("[1]", {|"[1]"|});
          ("true", {|"true"|});
          ("null", {|"null"|});
        ]
        (fun (row, text) ->
          starts_with
            ~prefix:(place 1 1 0 ^ text ^ ": ")
            (decode_error flat row));
      cases "a row that breaks the syntax fails at the byte that breaks it"
        ~id:fst
        [ ("xyz", (1, 0)); ("nonsense", (1, 0)); ("[1,}", (4, 3)) ]
        (fun (row, (column, byte)) ->
          starts_with ~prefix:(place 1 column byte) (decode_error flat row));
      test "bool reads true and false" (fun () ->
          let t =
            Error.get_ok
              (read (one Type.bool) (lines [ {|{"a":true}|}; {|{"a":false}|} ]))
          in
          equal strings [ Some "true"; Some "false" ] (printed t "a"));
      cases "an integer type reads a whole number at the ends of its range"
        ~id:(fun (t, s) -> type_text t ^ " " ^ s)
        [
          (any Type.int8, "-128");
          (any Type.int8, "127");
          (any Type.int16, "-32768");
          (any Type.int16, "32767");
          (any Type.int32, "-2147483648");
          (any Type.int32, "2147483647");
          (any Type.int64, "-9223372036854775808");
          (any Type.int64, "9223372036854775807");
          (any Type.uint8, "0");
          (any Type.uint8, "255");
          (any Type.uint16, "65535");
          (any Type.uint32, "4294967295");
          (any Type.uint64, "18446744073709551615");
        ]
        (fun (Type.Any ty, number) ->
          equal strings [ Some number ] (printed (read_one ty number) "a"));
      test "minus zero reads as zero in an unsigned type" (fun () ->
          equal strings [ Some "0" ] (printed (read_one Type.uint8 "-0") "a"));
      cases "an integer type refuses a number past its range, or not whole"
        ~id:(fun (t, s) -> type_text t ^ " " ^ s)
        [
          (any Type.int8, "-129");
          (any Type.int8, "128");
          (any Type.int16, "-32769");
          (any Type.int16, "32768");
          (any Type.int32, "-2147483649");
          (any Type.int32, "2147483648");
          (any Type.int64, "-9223372036854775809");
          (any Type.int64, "9223372036854775808");
          (any Type.uint8, "-1");
          (any Type.uint8, "256");
          (any Type.uint16, "65536");
          (any Type.uint32, "4294967296");
          (any Type.uint64, "18446744073709551616");
          (any Type.int64, "1.0");
          (any Type.int64, "1e2");
        ]
        (fun (Type.Any ty, number) ->
          let e = fails_one ty number in
          starts_with ~prefix:(place 1 6 5) e;
          contains ~sub:(Printf.sprintf "%S" number) e);
      (* Each input lies within one float64 ulp of a midpoint of the narrower
         type, so reading through a float64 rounds twice and lands on the even
         neighbour, where the nearest value, rounded once, is the other. *)
      cases "a float type reads the nearest value, rounded once, ties to even"
        ~id:(fun (t, s, _) -> type_text t ^ " " ^ s)
        [
          (any Type.float32, "1.0000000596046448", 1. +. ldexp 1. (-23));
          (any Type.float32, "1.000000059604644775390625", 1.);
          (any Type.float16, "1.00048828125000000001", 1. +. ldexp 1. (-10));
          (any Type.float16, "1.00048828125", 1.);
          (any Type.float64, "9007199254740993", 9007199254740992.);
        ]
        (fun (Type.Any ty, number, want) ->
          equal
            (array (option float_exact))
            [| Some want |]
            (Column.options Kind.float (column (read_one ty number) "a")));
      (* IEEE 754-2019 §7.4: a value at least the largest finite one plus half
         its ulp rounds to an infinity, which a float column refuses. *)
      cases "a float past its type's largest value by half an ulp fails"
        ~id:(fun (t, s, _) -> type_text t ^ " " ^ s)
        [
          (any Type.float64, "1.7976931348623158e308", Some Float.max_float);
          (any Type.float64, "1.7976931348623159e308", None);
          (any Type.float64, "-1e999", None);
          (any Type.float32, "3.4028235e38", Some 0x1.fffffep127);
          (any Type.float32, "3.4028236e38", None);
          (any Type.float32, "1" ^ String.make 39 '0', None);
          (any Type.float16, "65519", Some 65504.);
          (any Type.float16, "65520", None);
        ]
        (fun (Type.Any ty, number, want) ->
          match want with
          | Some x ->
              equal
                (array (option float_exact))
                [| Some x |]
                (Column.options Kind.float (column (read_one ty number) "a"))
          | None -> starts_with ~prefix:(place 1 6 5) (fails_one ty number));
      cases "string reads a string's text, and any other value's JSON text"
        ~id:fst
        [
          ({|"x\n"|}, "x\n");
          ("1.50", "1.50");
          ("true", "true");
          ({|{"x" : [1, 2]}|}, {|{"x" : [1, 2]}|});
        ]
        (fun (value, text) ->
          equal strings [ Some text ] (printed (read_one Type.string value) "a"));
      cases "binary reads a string's bytes, and any other value's JSON text"
        ~id:fst
        [ ({|"a\u0000"|}, "a\000"); ("1.50", "1.50"); ("[ ]", "[ ]") ]
        (fun (value, bytes) ->
          equal
            (array (option string))
            [| Some bytes |]
            (Array.map
               (Option.map (fun b -> (b : Binary.t :> string)))
               (Column.options Kind.binary
                  (column (read_one Type.binary value) "a"))));
      cases "categorical, date and datetime read a string as Column.parse does"
        ~id:(fun (t, s) -> type_text t ^ " " ^ s)
        [
          (any (Type.categorical [| "red"; "green" |]), "green");
          (any (Type.categorical [| "red"; "green" |]), "blue");
          (any Type.date, "2024-02-29");
          (any Type.date, "2024-02-30");
          (any (Type.instant Us), "2024-01-02T03:04:05.5Z");
          (any (Type.wall Ms), "2024-01-02T03:04:05");
          (any (Type.wall Ms), "2024-01-02T03:04:05Z");
        ]
        (fun (Type.Any ty, text) ->
          let value = {|"|} ^ text ^ {|"|} in
          match
            Column.parse (any ty)
              (Column.of_options Type.string [| Some text |])
          with
          | Ok c ->
              equal strings
                (Array.to_list (Column.options Kind.string (Column.print c)))
                (printed (read_one ty value) "a")
          | Error _ -> starts_with ~prefix:(place 1 6 5) (fails_one ty value));
      test "lists and records read element by element, null and absent as null"
        (fun () ->
          let item =
            Type.record
              [ ("x", any Type.int64); ("y", any (Type.list Type.string)) ]
          in
          let f =
            J.format [ ("r", any item); ("l", any (Type.list Type.int64)) ]
          in
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
      (* Error.pp quotes a text, with a quote or backslash escaped, and cuts one
         past 64 bytes with an ellipsis. *)
      cases "a value of another JSON type fails with its place, text and fix"
        ~id:(fun (_, value, _, _, _) -> value)
        [
          (any Type.float64, {|"1.5"|}, 5, {|"\"1.5\""|}, {|"a"|});
          (any Type.bool, "1", 5, {|"1"|}, {|"a"|});
          (any Type.int64, "true", 5, {|"true"|}, {|"a"|});
          ( any (Type.list Type.int64),
            {|{"x": [1]}|},
            5,
            {|"{\"x\": [1]}"|},
            {|"a"|} );
          (any (Type.list Type.int64), "[[1]]", 6, {|"[1]"|}, {|"a[]"|});
          ( any (Type.record [ ("x", any Type.int64) ]),
            "[1]",
            5,
            {|"[1]"|},
            {|"a"|} );
          ( any Type.int64,
            nested 1 (String.concat "," (List.init 40 (Fun.const "1"))),
            5,
            {|"[|} ^ String.concat "," (List.init 32 (Fun.const "1")) ^ {|"…|},
            {|"a"|} );
        ]
        (fun (Type.Any ty, value, byte, text, path) ->
          let e = fails_one ty value in
          starts_with
            ~prefix:(place 1 (byte + 1) byte ^ text ^ ": column " ^ path ^ ": ")
            e;
          contains ~sub:"with_type" e);
    ]

(* Sniffing *)

let sniffing =
  group "sniffing"
    [
      cases "the first value picks the framing" ~id:fst
        [
          ({|{"a":1}|}, "lines");
          ("\n  {\"a\":1}", "lines");
          ({| [ {"a":1}, {"a":2} ] |}, "array");
          ("\xEF\xBB\xBF[{\"a\":1}]", "array");
        ]
        (fun (text, framing) ->
          starts_with
            ~prefix:("json (1 column, " ^ framing ^ ")")
            (format_text (sniffed text)));
      cases "a column's type joins its values"
        ~id:(fun (rows, _) -> String.concat " " rows)
        [
          ([ {|{"a":true}|}; {|{"a":false}|} ], "bool");
          ([ {|{"a":1}|}; {|{"a":-9223372036854775808}|} ], "int64");
          ([ {|{"a":1}|}; {|{"a":null}|} ], "int64");
          ([ {|{"a":-0}|} ], "int64");
          ([ {|{"a":1}|}; {|{"a":1.5}|} ], "float64");
          ([ {|{"a":1e2}|} ], "float64");
          ([ {|{"a":9223372036854775808}|} ], "string");
          ([ {|{"a":-9223372036854775809}|} ], "string");
          ([ {|{"a":1.5}|}; {|{"a":9223372036854775808}|} ], "string");
          ([ {|{"a":"x"}|}; {|{"a":1}|} ], "string");
          ([ {|{"a":null}|} ], "string");
          ([ {|{"a":"2024-01-01"}|} ], "string");
          ([ {|{"a":[]}|} ], "list[string]");
          ([ {|{"a":[1]}|}; {|{"a":[1.5]}|} ], "list[float64]");
          ([ {|{"a":[[1.5]]}|} ], "list[list[float64]]");
          ( [ {|{"a":{"x":1}}|}; {|{"a":{"y":"z"}}|} ],
            "record[x int64, y string]" );
          ([ {|{"a":{}}|} ], "string");
          ([ {|{"a":{"x":1}}|}; {|{"a":1}|} ], "string");
        ]
        (fun (rows, ty) -> equal string ("  a " ^ ty) (sniffed_type rows));
      test "the columns are the keys in the order they first appear" (fun () ->
          equal string
            "json (3 columns, lines), types sniffed from 2 rows\n\
            \  b bool\n\
            \  i int64\n\
            \  later int64"
            (format_text
               (sniffed
                  (lines
                     [ {|{"b":true,"i":1}|}; {|{"i":2,"later":7,"b":false}|} ]))));
      test "a column of several JSON types reads as their JSON text" (fun () ->
          let s =
            lines [ {|{"a":1.50}|}; {|{"a":"x"}|}; {|{"a":{"b":[1, 2]}}|} ]
          in
          let t = Error.get_ok (read (sniffed s) s) in
          equal strings
            [ Some "1.50"; Some "x"; Some {|{"b":[1, 2]}|} ]
            (printed t "a"));
      (* The row's object is level 1, so a key's 63 arrays reach level 64. *)
      cases "a value 64 levels deep keeps its type, one deeper sniffs as string"
        ~id:fst
        [
          ("63 arrays", (nested 63 "1", list_type 63 "int64"));
          ("64 arrays", (nested 64 "1", list_type 63 "string"));
        ]
        (fun (_, (value, ty)) ->
          equal string ("  a " ^ ty) (sniffed_type [ {|{"a":|} ^ value ^ "}" ]));
      (* Rendered as gen/corpus.py renders a list of strings. *)
      test "a value a million levels deep sniffs as string, holding its text"
        (fun () ->
          let row = {|{"a":|} ^ Lazy.force deep ^ "}" in
          equal string
            (list_type 63 "string" ^ "\n"
            ^ nested 63 ("\"" ^ nested (depth - 63) "" ^ "\""))
            (in_small_stack (fun () ->
                 let f = sniffed row in
                 let t = Error.get_ok (read f row) in
                 type_text (Column.type_ (column t "a"))
                 ^ "\n"
                 ^ List.hd (rendered t "a"))));
    ]

(* Formats *)

let format =
  group "format"
    [
      test "framing defaults to Lines" (fun () ->
          starts_with ~prefix:"json (1 column, lines)"
            (format_text (one Type.int64)));
      cases "raises for columns JSON does not read" ~id:fst
        [
          ("none", []);
          ("a repeated name", [ ("a", any Type.bool); ("a", any Type.bool) ]);
          ("a name not UTF-8", [ ("\xff", any Type.bool) ]);
          ("a clock", [ ("a", any (Type.clock Ms)) ]);
          ("a duration", [ ("a", any (Type.duration S)) ]);
          ("a tensor", [ ("a", any (Type.tensor Nx.float32 [| 2 |])) ]);
          ("an extension", [ ("a", any (Type.ext ~name:"m" Type.float64)) ]);
          ( "a record of a clock",
            [ ("a", any (Type.record [ ("c", any (Type.clock S)) ])) ] );
          ("a list of durations", [ ("a", any (Type.list (Type.duration Ns))) ]);
        ]
        (fun (_, columns) ->
          raises_match (Exn.invalid_arg ~sub:"Talon_json.format") (fun () ->
              J.format columns));
    ]

let with_type =
  group "with_type"
    [
      test "a declared column's failures no longer cite sniffing" (fun () ->
          let s = lines (List.init 10 (fun _ -> {|{"a":1}|})) ^ {|{"a":1.5}|} in
          let f = sniffed ~rows:4 s in
          let sniffed_error = decode_error f s in
          starts_with ~prefix:(place 11 6 85) sniffed_error;
          contains ~sub:"sniffed from rows 1 to 4" sniffed_error;
          not_contains ~sub:"sniffed"
            (decode_error (J.with_type "a" (any Type.int32) f) s));
      cases "raises for an unknown column or a type JSON does not read" ~id:fst
        [ ("z", any Type.bool); ("a", any (Type.clock S)) ]
        (fun (name, ty) ->
          raises_match (Exn.invalid_arg ~sub:"Talon_json.with_type") (fun () ->
              J.with_type name ty (one Type.bool)));
    ]

let pp_format =
  group "pp_format"
    [
      test "a stated format prints its framing and its columns" (fun () ->
          equal string "json (2 columns, array)\n  id int64\n  name string"
            (format_text
               (J.format ~framing:Array
                  [ ("id", any Type.int64); ("name", any Type.string) ])));
      test "a sniffed format adds its rows and marks declared columns, aligned"
        (fun () ->
          let f = sniffed {|{"é":1,"name":"x"}|} in
          equal string
            "json (2 columns, lines), types sniffed from 1 row\n\
            \  é float32   declared\n\
            \  name string"
            (format_text (J.with_type "é" (any Type.float32) f)));
    ]

(* Sniff *)

let sniff =
  group "sniff"
    [
      test "rows bounds the sample" (fun () ->
          equal string "  a int64"
            (List.nth
               (String.split_on_char '\n'
                  (format_text
                     (sniffed ~rows:1 (lines [ {|{"a":1}|}; {|{"a":1.5}|} ]))))
               1));
      cases "rows defaults to 16384" ~id:fst
        [
          ("a fraction in row 16,384", (16_383, "float64"));
          ("a fraction in row 16,385", (16_384, "int64"));
        ]
        (fun (_, (ints, ty)) ->
          let s =
            String.concat "" (List.init ints (Fun.const "{\"a\":1}\n"))
            ^ "{\"a\":1.5}\n"
          in
          equal string
            ("json (1 column, lines), types sniffed from 16,384 rows\n  a " ^ ty)
            (format_text (sniffed s)));
      test "sniff reads no further than its rows need" (fun () ->
          let row i = Printf.sprintf {|{"a":%d}|} i ^ "\n" in
          let s = String.concat "" (List.init 10_000 row) in
          let first_ten = String.length (String.concat "" (List.init 10 row)) in
          let inner = Reader.of_string ~slice_length:16 s in
          let read = ref 0 in
          let r =
            Reader.make (fun () ->
                let slice = Reader.read inner in
                read := !read + Bytesrw.Bytes.Slice.length slice;
                slice)
          in
          ignore (Error.get_ok (J.sniff ~rows:10 r));
          at_most ~than:(first_ten + 16) int !read);
      test "sniff leaves the reader as it was" (fun () ->
          let s = lines [ {|{"a":1}|}; {|{"a":2}|} ] in
          let r = Reader.of_string ~slice_length:1 s in
          let f = Error.get_ok (J.sniff r) in
          equal table_w (Error.get_ok (read f s)) (Error.get_ok (J.decode f r)));
      cases "fails on a sample it cannot infer from" ~id:fst
        [
          ("", None);
          ("[]", None);
          ("{}\n{}\n", None);
          ("1\n", Some (1, 1, 0));
          ({|{"a":1,"a":2}|}, Some (1, 8, 7));
          ({|{"a":{"x":1,"x":2}}|}, Some (1, 13, 12));
          ({|{"a":1,}|}, Some (1, 8, 7));
        ]
        (fun (text, at) ->
          let e = sniff_error text in
          Option.iter
            (fun (line, column, byte) ->
              starts_with ~prefix:(place line column byte) e)
            at);
      test "a reader that raises is an Error" (fun () ->
          is_error ~msg:"Bytesrw.Bytes.Stream.Error"
            (J.sniff
               (Reader.limit 5 (Reader.of_string ~slice_length:2 "{\"a\":1}\n")));
          is_error ~msg:"Sys_error"
            (J.sniff (Reader.make (fun () -> raise (Sys_error "disk")))));
      cases "rows below 1 raises" ~id:string_of_int [ 0; -1 ] (fun rows ->
          raises_match (Exn.invalid_arg ~sub:"Talon_json.sniff") (fun () ->
              J.sniff ~rows (Reader.of_string "{\"a\":1}")));
    ]

(* Decode *)

let read_file path = In_channel.with_open_bin path In_channel.input_all

let decode =
  group "decode"
    [
      prop ~count:20
        ~examples:[ (1, false); (1, true) ]
        "batches do not depend on the reader's slices"
        Gen.(pair (int_range 1 64) bool)
        (fun (slice_length, array) ->
          let framing = if array then J.Array else J.Lines in
          let f = id_format framing and s = id_text framing in
          let whole = id_table framing in
          let sliced = Error.get_ok (read ~slice_length f s) in
          equal (list int)
            (List.map rows (batches whole))
            (List.map rows (batches sliced));
          equal table_w whole sliced);
      cases "a text of more than a batch reads every row" ~id:fst
        [ ("lines", J.Lines); ("array", J.Array) ]
        (fun (_, framing) ->
          let t = Error.get_ok (read (id_format framing) (id_text framing)) in
          greater ~than:1 int (List.length (batches t));
          equal strings
            (List.init ids (fun i -> Some (string_of_int i)))
            (printed t "id"));
      test "Lines reads an empty input as no row, and Array fails on it"
        (fun () ->
          equal int 0 (rows (Error.get_ok (read (one Type.int64) "")));
          equal int 0
            (rows (Error.get_ok (read (one ~framing:Array Type.int64) "[ ]")));
          starts_with ~prefix:(place 1 1 0)
            (decode_error (one ~framing:Array Type.int64) ""));
      (* Both values fail once read, column by column, [a] first. *)
      test "the error is the earliest failing value" (fun () ->
          let f =
            J.format [ ("a", any Type.float64); ("b", any Type.float64) ]
          in
          starts_with ~prefix:(place 1 6 5)
            (decode_error f {|{"b":1e400,"a":1e400}|}));
      test "a reader that raises is an Error" (fun () ->
          is_error ~msg:"Bytesrw.Bytes.Stream.Error"
            (J.decode (one Type.int64)
               (Reader.limit 5 (Reader.of_string ~slice_length:2 "{\"a\":1}\n")));
          is_error ~msg:"Sys_error"
            (J.decode (one Type.int64)
               (Reader.make (fun () -> raise (Sys_error "disk")))));
      (* Files written by pinned Python tools, and what Python reads from them
         (gen/corpus.py). *)
      test "files orjson and json write read as Python reads them" (fun () ->
          let golden =
            List.filter
              (fun l -> l <> "")
              (String.split_on_char '\n' (read_file "golden/values.txt"))
          in
          let files =
            List.fold_left
              (fun files l ->
                let file = List.hd (String.split_on_char '\t' l) in
                if List.mem file files then files else files @ [ file ])
              [] golden
          in
          let lines_of file =
            let s = read_file (Filename.concat "support" file) in
            let t = Error.get_ok (read (sniffed s) s) in
            List.map
              (fun (name, ty) ->
                String.concat "\t"
                  [
                    file;
                    name;
                    type_text ty;
                    String.concat "\x1f" (rendered t name);
                  ])
              (Schema.columns (schema t))
          in
          equal (list string) golden (List.concat_map lines_of files));
    ]

(* Source *)

let source =
  group "source"
    [
      test "a source is named json" (fun () ->
          contains ~sub:"json (2 columns)"
            (plan_text
               (Query.of_source (J.source flat (fun () -> Reader.of_string "")))));
      test "each read calls open_ once" (fun () ->
          let opened = ref 0 in
          let open_ () =
            incr opened;
            Reader.of_string {|{"a":1}|}
          in
          let q = Query.of_source (J.source (one Type.int64) open_) in
          ignore (Error.get_ok (Query.run q));
          ignore (Error.get_ok (Query.run q));
          equal int 2 !opened);
      test "a limit stops before a later error" (fun () ->
          let s = "{\"a\":1}\n{\"a\":\"x\"}\n" in
          let q =
            Query.slice ~offset:0 ~length:1
              (Query.of_source
                 (J.source (one Type.int64) (fun () -> Reader.of_string s)))
          in
          equal int 1 (rows (Error.get_ok (Query.run q))));
      test "a request reads its columns and checks every row's syntax and keys"
        (fun () ->
          let keep_b s =
            Query.run
              (Query.select
                 Expr.[ keep (Sel.names [ "b" ]) ]
                 (Query.of_source
                    (J.source flat (fun () -> Reader.of_string s))))
          in
          equal strings [ Some "x" ]
            (printed (Error.get_ok (keep_b {|{"a":1,"b":"x"}|})) "b");
          is_error ~msg:"a malformed a" (keep_b {|{"a":1.,"b":"x"}|});
          is_error ~msg:"a named twice" (keep_b {|{"a":1,"a":2,"b":"x"}|}));
      test
        "a value of a column the request does not name is not read as its type"
        (fun () ->
          let q =
            Query.select
              Expr.[ keep (Sel.names [ "b" ]) ]
              (Query.of_source
                 (J.source flat (fun () ->
                      Reader.of_string {|{"a":"x","b":"y"}|})))
          in
          equal strings [ Some "y" ] (printed (Error.get_ok (Query.run q)) "b"));
    ]

(* File *)

let file =
  group "file"
    [
      test "a file is named json \"path\", and its errors name the path"
        (fun () ->
          let path = temp_file ~suffix:".ndjson" () in
          Out_channel.with_open_bin path (fun oc ->
              output_string oc "{\"x\":1}\n{\"x\":\"oops\"}\n");
          let f = J.format [ ("x", any Type.int64) ] in
          let q = Query.of_source (Error.get_ok (J.file ~format:f path)) in
          contains ~sub:(Printf.sprintf "json %S" path) (plan_text q);
          starts_with ~prefix:(path ^ ":2:6: byte 13: ")
            (match Query.run q with Ok _ -> "read" | Error e -> error_text e));
      test "a missing path is an Error at the call, with a format or not"
        (fun () ->
          let path = Filename.concat (temp_dir ()) "missing.ndjson" in
          is_error ~msg:"sniffed" (J.file path);
          is_error ~msg:"declared" (J.file ~format:(one Type.int64) path));
      test "with a format, a named pipe is read once" (fun () ->
          chdir (temp_dir ());
          (try Unix.mkfifo "data.ndjson" 0o600
           with Unix.Unix_error _ | Invalid_argument _ ->
             skip ~reason:"no named pipes" ());
          (* No writer has the pipe open at the call, so a call that opened it
             would wait for ever. *)
          let s =
            Error.get_ok (J.file ~format:(one Type.int64) "data.ndjson")
          in
          let writer =
            Domain.spawn (fun () ->
                Out_channel.with_open_bin "data.ndjson" (fun oc ->
                    output_string oc "{\"a\":1}\n{\"a\":2}\n"))
          in
          let t = Error.get_ok (Query.run (Query.of_source s)) in
          Domain.join writer;
          equal strings [ Some "1"; Some "2" ] (printed t "a"));
      test "each read opens the file again" (fun () ->
          let path = temp_file ~suffix:".ndjson" () in
          let write text =
            Out_channel.with_open_bin path (fun oc -> output_string oc text)
          in
          write "{\"a\":1}\n";
          let q =
            Query.of_source
              (Error.get_ok (J.file ~format:(one Type.int64) path))
          in
          equal strings [ Some "1" ] (printed (Error.get_ok (Query.run q)) "a");
          write "{\"a\":2}\n{\"a\":3}\n";
          equal strings [ Some "2"; Some "3" ]
            (printed (Error.get_ok (Query.run q)) "a"));
      test
        "with a format, a file gone before the read fails it, naming its path"
        (fun () ->
          let path = temp_file ~suffix:".ndjson" () in
          let source = Error.get_ok (J.file ~format:(one Type.int64) path) in
          Sys.remove path;
          match Query.run (Query.of_source source) with
          | Ok _ -> fail "read a removed file"
          | Error e -> contains ~sub:path (error_text e));
    ]

(* Encode *)

(* The types JSON reads (talon_json.mli, Values). *)
let rec writable (Type.Any t) =
  match t with
  | Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64
  | Float16 | Float32 | Float64 | String | Binary | Categorical _ | Date
  | Datetime _ ->
      true
  | List e -> writable (any e)
  | Record fields -> List.for_all (fun (_, t) -> writable t) fields
  | Clock _ | Duration _ | Tensor _ | Ext _ -> false

(* [no_text ty v] is whether [v] holds a value JSON has no text for: an
   infinity, a NaN, or a binary value that is not UTF-8. *)
let rec no_text : type a. a Type.t -> a -> bool =
 fun ty v ->
  match ty with
  | Float16 -> not (Float.is_finite v)
  | Float32 -> not (Float.is_finite v)
  | Float64 -> not (Float.is_finite v)
  | Binary -> not (String.is_valid_utf_8 (v : Binary.t :> string))
  | List e -> Array.exists (no_text e) v
  | Record fields ->
      List.exists
        (fun (name, Type.Any t) ->
          match Record.field (Type.kind t) name v with
          | Some x -> no_text t x
          | None -> false)
        fields
  | _ -> false

let json_sample =
  Gen.with_pp G.pp_sample
    Gen.(
      let* (Type.Any ty) = such_that writable G.type_ in
      let+ vs = G.options ty in
      G.Sample (ty, vs))

let encode_result ?eod f t =
  let b = Buffer.create 256 in
  let r = J.encode ?eod f (Query.of_table t) (Writer.of_buffer b) in
  (Buffer.contents b, r)

let encode f t =
  let text, r = encode_result f t in
  Error.get_ok r;
  text

let rows_of_x ty vs = Talon.v [ ("x", Column.of_options ty vs) ]
let item = Type.record [ ("x", any Type.int64); ("y", any Type.string) ]

let records =
  G.Sample
    ( Type.list item,
      [|
        Some
          [|
            Record.(
              empty |> add Kind.int "x" (Some 1)
              |> add Kind.string "y" (Some "a\"b\n"));
          |];
        Some [||];
        None;
      |] )

let encoding =
  group "encode"
    [
      prop ~count:300
        ~examples:
          [
            (records, false);
            (G.Sample (Type.float64, [| Some Float.nan |]), true);
          ]
        "decode reads back what encode writes, which fails on a value JSON has \
         no text for"
        (Gen.pair json_sample Gen.bool)
        (fun (G.Sample (ty, vs), array) ->
          let refused =
            Array.exists (function Some v -> no_text ty v | None -> false) vs
          in
          cover "a list" (match ty with List _ -> true | _ -> false);
          cover "a list of records"
            (match ty with List (Record _) -> true | _ -> false);
          cover "a record" (match ty with Record _ -> true | _ -> false);
          cover "no rows" (Array.length vs = 0);
          cover "a value JSON has no text for" refused;
          let f = one ~framing:(if array then Array else Lines) ty in
          let t = Talon.v [ ("a", Column.of_options ty vs) ] in
          if refused then is_error (snd (encode_result f t))
          else
            Law.round_trip table_w string (encode f)
              (fun s -> Error.get_ok (read f s))
              t);
      cases "each framing writes its rows' text"
        ~id:(fun (id, _, _) -> id)
        [
          ( "lines",
            rows_of_x Type.int64 [| Some 1; None |],
            (J.Lines, "{\"x\":1}\n{\"x\":null}\n") );
          ( "array",
            rows_of_x Type.int64 [| Some 1; None |],
            (J.Array, "[\n{\"x\":1},\n{\"x\":null}\n]\n") );
          ("empty array", rows_of_x Type.int64 [||], (J.Array, "[\n]\n"));
        ]
        (fun (_, t, (framing, text)) ->
          equal string text
            (encode (J.format ~framing [ ("x", any Type.int64) ]) t));
      test "text is written with its bytes and the escapes JSON requires"
        (fun () ->
          equal string "{\"x\":\"a\\\"b\\\\c\\n\\u0001\\t\xc3\xa9\"}\n"
            (encode
               (J.format [ ("x", any Type.string) ])
               (rows_of_x Type.string [| Some "a\"b\\c\n\001\t\xc3\xa9" |])));
      (* Column.print writes a float in the fewest digits, without an exponent
         from 1e-7 up to 1e21; JSON adds .0 where they would read as an
         integer. *)
      cases "a float is written so that it reads back as a float"
        ~id:(fun (t, x, _) -> Printf.sprintf "%s %h" (type_text t) x)
        [
          (any Type.float64, 150., "150.0");
          (any Type.float64, -0., "-0.0");
          (any Type.float64, 1.5, "1.5");
          (any Type.float64, 1e-7, "0.0000001");
          (any Type.float64, 9.999999999999999e20, "999999999999999900000.0");
          (any Type.float64, 1e21, "1e+21");
          ( any Type.float32,
            Int32.float_of_bits (Int32.bits_of_float 0.1),
            "0.1" );
        ]
        (fun (Type.Any ty, x, text) ->
          let write (ty : float Type.t) =
            encode (J.format [ ("x", any ty) ]) (rows_of_x ty [| Some x |])
          in
          let written =
            match ty with
            | Float32 -> write Type.float32
            | Float64 -> write Type.float64
            | _ -> fail "a float type"
          in
          equal string ({|{"x":|} ^ text ^ "}\n") written);
      (* An Array's first row is on line 2, after its bracket. *)
      cases
        "a value JSON has no text for fails at its line, after the rows before \
         it"
        ~id:(fun (id, _, _) -> id)
        (List.concat_map
           (fun (id, t, first) ->
             [
               (id ^ ", lines", t, (J.Lines, first ^ "\n", 2));
               (id ^ ", array", t, (J.Array, "[\n" ^ first, 3));
             ])
           [
             ( "NaN",
               rows_of_x Type.float64 [| Some 1.; Some Float.nan |],
               {|{"x":1.0}|} );
             ( "infinity",
               rows_of_x Type.float64 [| Some 1.; Some Float.infinity |],
               {|{"x":1.0}|} );
             ( "-infinity",
               rows_of_x Type.float64 [| Some 1.; Some Float.neg_infinity |],
               {|{"x":1.0}|} );
             ( "binary not UTF-8",
               rows_of_x Type.binary
                 [|
                   Some (Binary.of_string "a"); Some (Binary.of_string "\xff");
                 |],
               {|{"x":"a"}|} );
           ])
        (fun (_, t, (framing, written, line)) ->
          let f = J.format ~framing [ ("x", Column.type_ (column t "x")) ] in
          match encode_result f t with
          | _, Ok () -> fail "encode wrote every row"
          | text, Error e ->
              equal string written text;
              starts_with
                ~prefix:(Printf.sprintf "line %d: " line)
                (error_text e));
      test "a failing run is an Error, after the rows before it" (fun () ->
          let q =
            Query.of_source
              (J.source (one Type.int64) (fun () ->
                   Reader.of_string "{\"a\":1}\n{\"a\":\"x\"}\n"))
          in
          let b = Buffer.create 16 in
          is_error (J.encode (one Type.int64) q (Writer.of_buffer b));
          equal string "{\"a\":1}\n" (Buffer.contents b));
      test "a writer that raises is an Error" (fun () ->
          let q = Query.of_table (rows_of_x Type.int64 [| Some 1 |]) in
          let f = J.format [ ("x", any Type.int64) ] in
          is_error ~msg:"Bytesrw.Bytes.Stream.Error"
            (J.encode f q
               (Writer.limit 2 ~eod:false (Writer.of_buffer (Buffer.create 16))));
          is_error ~msg:"Sys_error"
            (J.encode f q
               (Writer.make (fun _ -> raise (Sys_error "disk full")))));
      test "eod is written only when asked, once every row is" (fun () ->
          let ending ?eod q =
            let seen = ref "no eod" in
            let w =
              Writer.make (fun s ->
                  if Bytesrw.Bytes.Slice.is_eod s then seen := "eod")
            in
            ignore (J.encode ?eod (one Type.int64) q w);
            !seen
          in
          let ok =
            Query.of_table
              (Talon.v [ ("a", Column.of_options Type.int64 [| Some 1 |]) ])
          in
          let failing =
            Query.of_source
              (J.source (one Type.int64) (fun () ->
                   Reader.of_string {|{"a":"x"}|}))
          in
          equal (list string)
            [ "no eod"; "eod"; "no eod" ]
            [ ending ok; ending ~eod:true ok; ending ~eod:true failing ]);
      cases "raises when the query's columns are not the format's" ~id:fst
        [
          ("another name", [ ("a", any Type.int64); ("c", any Type.string) ]);
          ("another type", [ ("a", any Type.int32); ("b", any Type.string) ]);
          ("another order", [ ("b", any Type.string); ("a", any Type.int64) ]);
        ]
        (fun (_, columns) ->
          let t =
            Talon.v
              [
                ("a", Column.of_options Type.int64 [||]);
                ("b", Column.of_options Type.string [||]);
              ]
          in
          raises_match (Exn.invalid_arg ~sub:"Talon_json.encode") (fun () ->
              J.encode (J.format columns) (Query.of_table t)
                (Writer.of_buffer (Buffer.create 1))));
    ]

let () =
  exit
    (run "talon.json"
       [
         streaming;
         syntax;
         values;
         sniffing;
         format;
         with_type;
         pp_format;
         sniff;
         decode;
         source;
         file;
         encoding;
       ])
