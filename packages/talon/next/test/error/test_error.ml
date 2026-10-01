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
    ]

let v =
  group "v"
    [
      cases
        ~name:(fun (first, last) ->
          Printf.sprintf "rejects the bytes (%d, %d)" first last)
        "range"
        [ (-1, 0); (-1, -1); (5, 4); (min_int, max_int) ]
        (fun bytes ->
          raises_match (Exn.invalid_arg ~substring:"Error.v") (fun () ->
              Error.v ~bytes "bad"));
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

let () = exit (run "error" [ pp; v; get_ok ])
