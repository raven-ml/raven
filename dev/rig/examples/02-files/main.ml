(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Files as memory.

   The disk is a device whose buffers are files. Its memory is reached in two
   ways: a copy reads or writes the file's bytes, and a borrow by the host maps
   the file's pages in place. Copy to read all of a file; borrow to touch part
   of it where it lies. *)

open Rig

let path = "numbers.bin"
let int32s b = Buffer.bigarray Bigarray.int32 b

let show name b =
  let a = int32s b in
  let xs = List.init (Bigarray.Array1.dim a) (fun i -> Int32.to_string a.{i}) in
  Printf.printf "%-8s [%s]\n" name (String.concat "; " xs)

let host_ints xs =
  let b = Buffer.create host (4 * List.length xs) in
  List.iteri (fun i x -> (int32s b).{i} <- Int32.of_int x) xs;
  b

let () =
  if Sys.file_exists path then Sys.remove path;
  Printf.printf "%s computes: %b\n\n" (name Rig_disk.device)
    (computes Rig_disk.device);

  (* Write: a new file is a buffer of the disk, which a copy fills. [barrier]
     orders those writes before later changes to the file system. *)
  let file = Result.get_ok (Rig_disk.create_file path 16) in
  Buffer.copy ~src:(host_ints [ 10; 20; 30; 40 ]) ~dst:file;
  Rig_disk.barrier file;

  (* Read: a view of a file is a range of its bytes, and a copy reads just
     those. *)
  let opened = Result.get_ok (Rig_disk.of_file path) in
  Printf.printf "%s holds %d bytes\n" path (Buffer.length opened);
  let middle = Buffer.create host 8 in
  Buffer.copy ~src:(Buffer.view opened ~first:4 ~length:8) ~dst:middle;
  show "copied" middle;

  (* Borrow: the host maps the file's pages and reads them in place. A file
     opened for reading is mapped copy-on-write: a write through the borrow
     changes the process's page, never the file. *)
  let pages = Option.get (Buffer.borrow host opened) in
  show "borrowed" pages;
  (int32s pages).{0} <- 7l;
  show "written" pages;
  let again = Buffer.create host 16 in
  Buffer.copy ~src:opened ~dst:again;
  show "file" again;

  (* A file opened for reading takes no copy, and a missing file is an [Error]
     naming it. *)
  (match Buffer.copy ~src:again ~dst:opened with
  | () -> ()
  | exception Invalid_argument _ ->
      print_endline "a copy into a file opened for reading is refused");
  match Rig_disk.of_file "missing.bin" with
  | Ok _ -> ()
  | Error why -> print_endline why
