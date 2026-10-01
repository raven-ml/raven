(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Error = Talon_next.Error

let to_string e = Format.asprintf "%a" Error.pp e

let pp =
  group "pp"
    [
      test "puts the file and the bytes before the message" (fun () ->
          equal string "zoneinfo/Paris: bytes 44-51: bad"
            (to_string (Error.v ~file:"zoneinfo/Paris" ~bytes:(44, 51) "bad")));
      test "names a range of one byte as a byte" (fun () ->
          equal string "f: byte 4: bad"
            (to_string (Error.v ~file:"f" ~bytes:(4, 4) "bad")));
      test "omits what the error does not locate" (fun () ->
          equal string "f: bad" (to_string (Error.v ~file:"f" "bad"));
          equal string "byte 0: bad" (to_string (Error.v ~bytes:(0, 0) "bad"));
          equal string "bad" (to_string (Error.v "bad")));
      cases
        ~name:(fun (_, expected) -> Printf.sprintf "prints %s" expected)
        "places, coarsest first"
        [
          (Error.v ~file:"f.csv" ~line:3 "bad", "f.csv:3: bad");
          (Error.v ~file:"f.csv" ~line:3 ~column:7 "bad", "f.csv:3:7: bad");
          (Error.v ~line:3 "bad", "line 3: bad");
          (Error.v ~line:3 ~column:7 "bad", "line 3, column 7: bad");
          (Error.v ~row_group:0 "bad", "row group 0: bad");
          (Error.v ~text:"NA" "bad", "\"NA\": bad");
          ( Error.v ~file:"f.parquet" ~row_group:2 ~bytes:(10, 19) ~text:"x"
              "bad",
            "f.parquet: row group 2: bytes 10-19: \"x\": bad" );
          ( Error.v ~file:"flights.csv" ~line:48213 ~column:12 ~text:"NA"
              "cannot read as float64.",
            "flights.csv:48213:12: \"NA\": cannot read as float64." );
        ]
        (fun (e, expected) -> equal string expected (to_string e));
    ]

let a n = String.make n 'a'

let text =
  group "pp of the raw text"
    [
      cases
        ~name:(fun (_, expected) -> Printf.sprintf "quotes as %s" expected)
        "escapes"
        [
          ("", {|""|});
          ({|say "hi"|}, {|"say \"hi\""|});
          ({|a\b|}, {|"a\\b"|});
          ("a\nb\x7f\x00", {|"a\x0ab\x7f\x00"|});
          ("caf\xc3\xa9", "\"caf\xc3\xa9\"");
          ("\xff\xc3", {|"\xff\xc3"|});
          ("\xe2\x82a", {|"\xe2\x82a"|});
          ("a\xc2\x80b\xc2\x9b31m", {|"a\xc2\x80b\xc2\x9b31m"|});
          ("a\xc2\xa0b", "\"a\xc2\xa0b\"");
          ("\xe2\x80\xaaa\xe2\x80\xaeb", {|"\xe2\x80\xaaa\xe2\x80\xaeb"|});
          ("\xe2\x81\xa6a\xe2\x81\xa9", {|"\xe2\x81\xa6a\xe2\x81\xa9"|});
          ( "\xe2\x80\xa9\xe2\x80\xaf\xe2\x81\xa5\xe2\x81\xaa",
            "\"\xe2\x80\xa9\xe2\x80\xaf\xe2\x81\xa5\xe2\x81\xaa\"" );
        ]
        (fun (t, expected) ->
          equal string (expected ^ ": m") (to_string (Error.v ~text:t "m")));
      cases
        ~name:(fun (name, _, _) -> name)
        "cut at 64 bytes"
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

let v =
  let rejects f = raises_match (Exn.invalid_arg ~substring:"Error.v") f in
  group "v"
    [
      cases
        ~name:(fun (first, last) ->
          Printf.sprintf "rejects the bytes (%d, %d)" first last)
        "range"
        [ (-1, 0); (-1, -1); (5, 4); (min_int, max_int) ]
        (fun bytes -> rejects (fun () -> Error.v ~bytes "bad"));
      test "rejects a line or a column below 1" (fun () ->
          rejects (fun () -> Error.v ~line:0 "bad");
          rejects (fun () -> Error.v ~line:1 ~column:0 "bad"));
      test "accepts line 1 and column 1" (fun () ->
          equal string "f:1:1: bad"
            (to_string (Error.v ~file:"f" ~line:1 ~column:1 "bad")));
      test "rejects a column without a line" (fun () ->
          rejects (fun () -> Error.v ~file:"f" ~column:1 "bad"));
      test "rejects a negative row group and accepts row group 0" (fun () ->
          rejects (fun () -> Error.v ~row_group:(-1) "bad");
          ignore (Error.v ~row_group:0 "bad"));
    ]

let get_ok =
  group "get_ok"
    [
      test "returns the value of Ok" (fun () ->
          equal int 3 (Error.get_ok (Ok 3)));
      test "raises Failure with the printed error" (fun () ->
          raises (Failure "f: byte 2: bad") (fun () ->
              Error.get_ok (Error (Error.v ~file:"f" ~bytes:(2, 2) "bad"))));
    ]

let () = exit (run "error" [ pp; text; v; get_ok ])
