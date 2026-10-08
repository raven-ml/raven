(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Files on the disk, written under this test's directory in _build. *)

open Windtrap
module B = Device_core.Buffer

let strf = Printf.sprintf
let timeout = 60.
let dir = "files"

let () =
  if Sys.file_exists dir then
    Array.iter (fun f -> Sys.remove (Filename.concat dir f)) (Sys.readdir dir)
  else Sys.mkdir dir 0o755

let names = ref 0

let new_path () =
  incr names;
  Filename.concat dir (strf "f%d" !names)

let contents path = In_channel.with_open_bin path In_channel.input_all

let host_of_string s =
  let b = B.create Device_core.host (String.length s) in
  String.iteri (Bigarray.Array1.set (B.bigarray Bigarray.char b)) s;
  b

let string_of_host b =
  let a = B.bigarray Bigarray.char b in
  String.init (Bigarray.Array1.dim a) (Bigarray.Array1.get a)

let read b =
  let h = B.create Device_core.host (B.length b) in
  B.copy ~src:b ~dst:h;
  string_of_host h

let files =
  group ~timeout "files"
    [
      test "a file created, written at an offset and opened again reads back"
        (fun () ->
          let path = new_path () in
          let file = require_ok (Device_disk.create_file path 10) in
          B.copy ~src:(host_of_string "abcde")
            ~dst:(B.view file ~first:3 ~length:5);
          Device_disk.flush file;
          equal string "\000\000\000abcde\000\000" (contents path);
          let again = require_ok (Device_disk.of_file path) in
          equal string "abcde" (read (B.view again ~first:3 ~length:5)));
      test "the host borrows a file's pages" (fun () ->
          let path = new_path () in
          Out_channel.with_open_bin path (fun oc -> output_string oc "pages");
          let file = require_ok (Device_disk.of_file path) in
          match B.borrow Device_core.host file with
          | None -> fail "no borrow"
          | Some pages -> equal string "pages" (string_of_host pages));
      test "a copy into a file opened for reading is refused" (fun () ->
          let path = new_path () in
          Out_channel.with_open_bin path (fun oc -> output_string oc "x");
          let file = require_ok (Device_disk.of_file path) in
          raises_match (Exn.invalid_arg ~substring:"Device_core.Buffer.copy: ")
            (fun () -> B.copy ~src:(host_of_string "y") ~dst:file));
    ]

let () = exit (run "device_disk" [ files ])
