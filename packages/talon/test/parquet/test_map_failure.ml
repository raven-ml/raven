(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

(* [with_sparse_file f] is [f path] for a sparse file of 2 EiB at [path], larger
   than any process's address space, so mapping it fails with ENOMEM, as a
   mapping does at [vm.max_map_count]. It lives on /dev/shm, a tmpfs, which
   holds files that large; ext4 holds 16 TiB at most. A sparse file holds no
   page, and it is removed on every path. *)
let with_sparse_file f =
  let path =
    Filename.temp_file ~temp_dir:"/dev/shm" "talon-map-failure" ".parquet"
  in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      let fd = Unix.openfile path [ Unix.O_WRONLY ] 0o600 in
      Fun.protect
        ~finally:(fun () -> Unix.close fd)
        (fun () -> Unix.ftruncate fd (1 lsl 61));
      f path)

let names_the_file_once () =
  with_sparse_file @@ fun path ->
  let result = Talon_parquet.file path in
  match result with
  | Ok _ -> fail "a file beyond the address space must not map"
  | Error e ->
      let msg = Format.asprintf "%a" Error.pp e in
      contains ~sub:path msg;
      let rec count i =
        match String.index_from_opt msg i path.[0] with
        | Some j when String.length msg - j >= String.length path
                      && String.sub msg j (String.length path) = path ->
            1 + count (j + 1)
        | Some j -> count (j + 1)
        | None -> 0
      in
      equal ~msg:("the path appears once in: " ^ msg) int 1 (count 0)

let () =
  exit
    (run "parquet map failure"
       [ test "a file that cannot be mapped names the path once" names_the_file_once ])
