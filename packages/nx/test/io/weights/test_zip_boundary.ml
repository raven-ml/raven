(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An entry of 0xFFFFFFFF bytes whose sizes follow its data, at the size where
   writers disagree on their width, loads either way: Java writes 8 bytes each
   from that size on, Go 4 up to it, and neither a ZIP64 field in the local
   header. Each archive is a sparse file of 4 GiB, which a test run cannot
   hold. The dune file says how to run it. *)

open Windtrap

let le n v = String.init n (fun i -> Char.chr ((v lsr (8 * i)) land 0xff))
let size = 0xffffffff
let name = "big.bin"

(* [archive width] is a file holding the stored entry [name] of [size] zeros
   and its descriptor with sizes of [width] bytes each. Its central record
   states the sizes in a ZIP64 field, as both writers do, and a CRC-32 of 0:
   no load reads the entry, since it is no npy file. *)
let archive width =
  let path = temp_file ~suffix:".npz" () in
  let fd = Unix.openfile path [ Unix.O_WRONLY; Unix.O_TRUNC ] 0 in
  let write s =
    let n = Unix.write_substring fd s 0 (String.length s) in
    assert (n = String.length s)
  in
  let local =
    "PK\003\004" ^ le 2 20 ^ le 2 0x0008 ^ le 2 0 ^ le 4 0 ^ le 4 0 ^ le 4 0
    ^ le 4 0
    ^ le 2 (String.length name)
    ^ le 2 0 ^ name
  in
  write local;
  let data_end = String.length local + size in
  ignore (Unix.lseek fd data_end Unix.SEEK_SET);
  let descriptor = "PK\007\008" ^ le 4 0 ^ le width size ^ le width size in
  write descriptor;
  let central_offset = data_end + String.length descriptor in
  let extra = le 2 0x0001 ^ le 2 16 ^ le 8 size ^ le 8 size in
  let central =
    "PK\001\002" ^ le 2 45 ^ le 2 45 ^ le 2 0x0008 ^ le 2 0 ^ le 4 0 ^ le 4 0
    ^ le 4 0xffffffff ^ le 4 0xffffffff
    ^ le 2 (String.length name)
    ^ le 2 (String.length extra)
    ^ le 2 0 ^ le 2 0 ^ le 2 0 ^ le 4 0 ^ le 4 0 ^ name ^ extra
  in
  write central;
  let zip64_end = central_offset + String.length central in
  write
    ("PK\006\006" ^ le 8 44 ^ le 2 45 ^ le 2 45 ^ le 4 0 ^ le 4 0 ^ le 8 1
   ^ le 8 1
    ^ le 8 (String.length central)
    ^ le 8 central_offset);
  write ("PK\006\007" ^ le 4 0 ^ le 8 zip64_end ^ le 4 1);
  write
    ("PK\005\006" ^ le 2 0 ^ le 2 0 ^ le 2 1 ^ le 2 1
    ^ le 4 (String.length central)
    ^ le 4 0xffffffff ^ le 2 0);
  Unix.close fd;
  path

let () =
  exit
    (run "nx.io zip boundary"
       [
         cases "an entry of 0xFFFFFFFF bytes with a data descriptor loads"
           ~id:fst
           [
             ("8-byte sizes, as Java writes", 8);
             ("4-byte sizes, as Go writes", 4);
           ]
           (fun (_, width) ->
             let path = archive width in
             Fun.protect ~finally:(fun () -> Sys.remove path) @@ fun () ->
             equal (list string) []
               (Nx_io.Archive.names (Nx_io.load_npz path)));
       ])
