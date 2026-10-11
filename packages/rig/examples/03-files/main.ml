(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Files as memory.

   The disk is a device whose buffers are files. A program uses its memory
   in two ways. A copy reads or writes the file's bytes, and a borrow by the
   host maps the file's pages in place. Copy to read all of a file, and
   borrow to touch part of it where it lies. *)

open Rig

let path = "numbers.bin"

(* [a]'s elements, joined. *)
let ints a =
  String.concat "; "
    (List.init (Bigarray.Array1.dim a) (fun i -> Int32.to_string a.{i}))

let show name b =
  Buffer.read b Bigarray.int32 (fun a ->
      Printf.printf "%-8s [%s]\n" name (ints a))

(* A bigarray of [l]. *)
let int32s l = Bigarray.(Array1.of_array int32 c_layout (Array.of_list l))

let () =
  if Sys.file_exists path then Sys.remove path;
  Printf.printf "%s computes: %b\n\n" (name Rig_disk.device)
    (computes Rig_disk.device);

  (* Write: a new file is a buffer of the disk, which a copy fills. [barrier]
     orders those writes before later changes to the file system. *)
  let file = Result.get_ok (Rig_disk.create_file path 16) in
  let numbers = Buffer.of_bigarray (int32s [ 10l; 20l; 30l; 40l ]) in
  Buffer.copy ~src:numbers ~dst:file;
  Rig_disk.barrier file;

  (* Read: a view of a file is a range of its bytes, and a copy reads just
     those. *)
  let opened = Result.get_ok (Rig_disk.of_file path) in
  Printf.printf "%s holds %d bytes\n" path (Buffer.length opened);
  let middle = Buffer.create host 8 in
  Buffer.copy ~src:(Buffer.view opened ~first:4 ~length:8) ~dst:middle;
  show "copied" middle;

  (* Borrow: the host maps the file's pages and reads them in place. The memory
     of a file opened for reading admits no write, through the borrow too. *)
  let pages = Option.get (Buffer.borrow host opened) in
  show "borrowed" pages;
  (match Buffer.write pages Bigarray.int32 (fun w -> w.{0} <- 7l) with
  | () -> ()
  | exception Invalid_argument _ ->
      print_endline "a write of a file opened for reading is refused");
  let again = Buffer.create host 16 in
  Buffer.copy ~src:opened ~dst:again;
  show "file" again;

  (* A copy into a file opened for reading is refused, and opening a missing
     file is an [Error] naming it. *)
  (match Buffer.copy ~src:again ~dst:opened with
  | () -> ()
  | exception Invalid_argument _ ->
      print_endline "a copy into a file opened for reading is refused");
  match Rig_disk.of_file "missing.bin" with
  | Ok _ -> ()
  | Error why -> print_endline why
