(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Error = Talon.Error

let to_string e = Format.asprintf "%a" Error.pp e
let rejects f = raises_match Exn.invalid_arg f

let v =
  group "v"
    [
      test "accepts line 1 and column 1" (fun () ->
          equal string "f:1:1: bad"
            (to_string (Error.v ~file:"f" ~line:1 ~column:1 "bad")));
      test "raises on a line or a column below 1" (fun () ->
          rejects (fun () -> Error.v ~line:0 "bad");
          rejects (fun () -> Error.v ~line:1 ~column:0 "bad"));
      test "raises on a column without a line" (fun () ->
          rejects (fun () -> Error.v ~file:"f" ~column:1 "bad"));
      test "raises on a negative row group" (fun () ->
          rejects (fun () -> Error.v ~row_group:(-1) "bad"));
      cases "raises on bytes that start before 0 or end before they start"
        ~id:(fun (first, last) -> Printf.sprintf "(%d, %d)" first last)
        [ (-1, 0); (-1, -1); (5, 4); (min_int, max_int) ]
        (fun bytes -> rejects (fun () -> Error.v ~bytes "bad"));
    ]

let a n = String.make n 'a'

let pp =
  group "pp"
    [
      cases "writes the places coarsest first, then the message"
        ~id:(fun (_, expected) -> expected)
        [
          (Error.v ~file:"f.csv" ~line:3 "bad", "f.csv:3: bad");
          (Error.v ~file:"f.csv" ~line:3 ~column:7 "bad", "f.csv:3:7: bad");
          (Error.v ~line:3 "bad", "line 3: bad");
          (Error.v ~line:3 ~column:7 "bad", "line 3, column 7: bad");
          (Error.v ~row_group:0 "bad", "row group 0: bad");
          (Error.v ~text:"NA" "bad", "\"NA\": bad");
          ( Error.v ~file:"zoneinfo/Paris" ~bytes:(44, 51) "bad",
            "zoneinfo/Paris: bytes 44-51: bad" );
          ( Error.v ~file:"f.parquet" ~row_group:2 ~bytes:(10, 19) ~text:"x"
              "bad",
            "f.parquet: row group 2: bytes 10-19: \"x\": bad" );
          ( Error.v ~file:"flights.csv" ~line:48213 ~column:12 ~text:"NA"
              "cannot read as float64.",
            "flights.csv:48213:12: \"NA\": cannot read as float64." );
        ]
        (fun (e, expected) -> equal string expected (to_string e));
      test "names a range of one byte as a byte" (fun () ->
          equal string "f: byte 4: bad"
            (to_string (Error.v ~file:"f" ~bytes:(4, 4) "bad")));
      test "omits the places the error does not have" (fun () ->
          equal string "f: bad" (to_string (Error.v ~file:"f" "bad"));
          equal string "byte 0: bad" (to_string (Error.v ~bytes:(0, 0) "bad"));
          equal string "bad" (to_string (Error.v "bad")));
      (* The rows hold each range's bounds and a neighbour outside it: U+001F
         and U+0020, U+007F, U+0080 to U+009F and U+00A0, U+202A to U+202E and
         U+2029, U+202F, U+2066 to U+2069 and U+2065, U+206A. *)
      cases
        "escapes quotes, backslashes, control characters, bidirectional \
         controls and bytes outside UTF-8 in the text"
        ~id:(fun (_, expected) -> expected)
        [
          ("", {|""|});
          ({|say "hi"|}, {|"say \"hi\""|});
          ({|a\b|}, {|"a\\b"|});
          ("a\nb\x7f\x00", {|"a\x0ab\x7f\x00"|});
          ("\x1f\x20", {|"\x1f "|});
          ("caf\xc3\xa9", "\"caf\xc3\xa9\"");
          ("\xff\xc3", {|"\xff\xc3"|});
          ("\xe2\x82a", {|"\xe2\x82a"|});
          ("a\xc2\x80b\xc2\x9b31m", {|"a\xc2\x80b\xc2\x9b31m"|});
          ("\xc2\x9f", {|"\xc2\x9f"|});
          ("a\xc2\xa0b", "\"a\xc2\xa0b\"");
          ("\xe2\x80\xaaa\xe2\x80\xaeb", {|"\xe2\x80\xaaa\xe2\x80\xaeb"|});
          ("\xe2\x81\xa6a\xe2\x81\xa9", {|"\xe2\x81\xa6a\xe2\x81\xa9"|});
          ( "\xe2\x80\xa9\xe2\x80\xaf\xe2\x81\xa5\xe2\x81\xaa",
            "\"\xe2\x80\xa9\xe2\x80\xaf\xe2\x81\xa5\xe2\x81\xaa\"" );
        ]
        (fun (t, expected) ->
          equal string (expected ^ ": m") (to_string (Error.v ~text:t "m")));
      cases "cuts a text longer than 64 bytes"
        ~id:(fun (name, _, _) -> name)
        [
          ("keeps 64 bytes", a 64, Printf.sprintf "%S" (a 64));
          ("cuts the 65th byte", a 65, Printf.sprintf "%S…" (a 64));
          ( "cuts before a character that the 65th byte ends",
            a 63 ^ "\xc3\xa9",
            Printf.sprintf "%S…" (a 63) );
          ( "keeps a character that the 64th byte ends",
            a 62 ^ "\xc3\xa9",
            "\"" ^ a 62 ^ "\xc3\xa9\"" );
          ( "cuts before a control that the 65th byte ends",
            a 62 ^ "\xe2\x80\xae",
            Printf.sprintf "%S…" (a 62) );
          ( "cuts an invalid 65th byte",
            a 64 ^ "\xff",
            Printf.sprintf "%S…" (a 64) );
          ( "keeps an invalid 64th byte",
            a 63 ^ "\xff\xff",
            "\"" ^ a 63 ^ "\\xff\"…" );
        ]
        (fun (_, t, expected) ->
          equal string (expected ^ ": m") (to_string (Error.v ~text:t "m")));
    ]

let get_ok =
  group "get_ok"
    [
      test "is the value of Ok" (fun () -> equal int 3 (Error.get_ok (Ok 3)));
      test "raises Failure with the error as pp writes it" (fun () ->
          raises (Failure "f: byte 2: bad") (fun () ->
              Error.get_ok (Error (Error.v ~file:"f" ~bytes:(2, 2) "bad"))));
    ]

let () = exit (run "error" [ v; pp; get_ok ])
