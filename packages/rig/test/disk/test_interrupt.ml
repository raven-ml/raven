(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks of the disk. A copy runs once for each allocation it makes,
   in a forked child, with every allocation from that one on raising until the
   copy returns, as an asynchronous exception does. After it the disk keeps
   its bound of descriptors, every file reads its bytes, and once the files
   are collected no descriptor of theirs stays open. *)

open Windtrap
module B = Rig.Buffer
module Support = Rig_disk_support

let strf = Printf.sprintf
let timeout = 60.

(* The disk keeps at most this many descriptors open while no copy uses
   them. *)
let max_descriptors = 64

(* Files *)

(* This executable's directory, under _build, holds the files, which the
   suite removes as it ends. *)
let exe =
  let e = Sys.executable_name in
  if Filename.is_relative e then Filename.concat (Sys.getcwd ()) e else e

let dir = Filename.concat (Filename.dirname exe) "interrupt-files"

let clear () =
  if Sys.file_exists dir then
    Array.iter (fun f -> Sys.remove (Filename.concat dir f)) (Sys.readdir dir)
  else Sys.mkdir dir 0o755

(* One file more than the disk keeps open, each holding its index. *)
let count = max_descriptors + 1
let bytes i = strf "file %d" i

let paths =
  lazy
    (List.init count (fun i ->
         let path = Filename.concat dir (strf "f%d" i) in
         Out_channel.with_open_bin path (fun oc -> output_string oc (bytes i));
         path))

let read b =
  let h = B.create Rig.host (B.length b) in
  B.copy ~src:b ~dst:h;
  B.read h Bigarray.char (fun a ->
      String.init (Bigarray.Array1.dim a) (Bigarray.Array1.get a))

(* Collects the dropped buffers, then drains the disk, which closes the
   descriptors of collected files. *)
let collect () =
  for _ = 1 to 10 do
    Gc.full_major ()
  done;
  ignore (B.create Rig_disk.device 0)

let descriptors () = Option.get (Support.descriptors ())

(* In a child *)

(* [in_child what f] is [f ()], run in a forked child, so that what one
   attempt leaves never changes the next one. It fails with the child's
   failure. *)
let in_child what f =
  let r, w = Unix.pipe () in
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let answer =
        match f () with v -> Ok v | exception e -> Error (Printexc.to_string e)
      in
      let oc = Unix.out_channel_of_descr w in
      Marshal.to_channel oc (answer : (bool, string) result) [];
      close_out oc;
      Unix._exit 0
  | pid -> (
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      let answer = In_channel.input_all ic in
      close_in ic;
      match (answer, snd (Unix.waitpid [] pid)) with
      | "", (WEXITED n | WSIGNALED n | WSTOPPED n) ->
          failf "the child ended (%d) without answering%s" n what
      | answer, _ -> (
          match (Marshal.from_string answer 0 : (bool, string) result) with
          | Ok v -> v
          | Error e -> failf "%s%s" e what))

(* Walks *)

(* Opens and reads every file, which leaves the first one's descriptor closed
   for the others, then reads the first file interrupted from its [n]-th
   allocation: the read reopens its descriptor and closes the least recently
   used other one. Is whether the allocation was reached. *)
let interrupt n =
  let at = strf ", interrupted from allocation %d" n in
  in_child at (fun () ->
      let base = descriptors () in
      let files = List.map Rig_disk.of_file (Lazy.force paths) in
      let files = List.map Result.get_ok files in
      List.iter (fun b -> ignore (read b)) files;
      let full = descriptors () in
      equal ~msg:"descriptors kept" int (base + max_descriptors) full;
      let first = List.hd files in
      let h = B.create Rig.host (B.length first) in
      let reached =
        Rig_support.interrupted n (fun () -> B.copy ~src:first ~dst:h)
      in
      List.iteri
        (fun i b -> equal ~msg:(strf "file %d" i) string (bytes i) (read b))
        files;
      equal ~msg:"descriptors kept after" int full (descriptors ());
      ignore (Sys.opaque_identity files);
      collect ();
      equal ~msg:"descriptors once collected" int base (descriptors ());
      reached)

let test_interrupt () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  Rig_support.needs_fork ();
  ignore (Lazy.force paths);
  let rec from n = if interrupt n then from (n + 1) in
  from 1

let () =
  clear ();
  let code =
    run "rig_disk.interrupt"
      [
        group ~timeout "an exception at any allocation"
          [
            test
              "a read that reopens a descriptor and closes another keeps the \
               bound and leaks no descriptor"
              test_interrupt;
          ];
      ]
  in
  clear ();
  Sys.rmdir dir;
  exit code
