(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

(* A file larger than the address space this test runs in, so mapping it fails
   as a mapping does at [vm.max_map_count]. *)
let sparse_file () =
  let path = Filename.temp_file "talon-map-failure" ".parquet" in
  let fd = Unix.openfile path [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC ] 0o600 in
  Unix.ftruncate fd (8 * 1024 * 1024 * 1024);
  Unix.close fd;
  path

let names_the_file_once () =
  let path = sparse_file () in
  let result = Talon_parquet.file path in
  Sys.remove path;
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
