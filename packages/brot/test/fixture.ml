(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Locating the data files the tests read. Paths are relative to
   packages/brot/test: the build directory comes first, so that files declared
   as dune dependencies win, then the source tree, which is where files that are
   downloaded rather than committed live. *)

open Windtrap

let locate relative =
  let candidates =
    relative
    ::
    (match Sys.getenv_opt "DUNE_SOURCEROOT" with
    | Some root ->
        [ Filename.concat root (Filename.concat "packages/brot/test" relative) ]
    | None -> [])
  in
  List.find_opt Sys.file_exists candidates

let read relative =
  match locate relative with
  | None -> failf "missing fixture %s" relative
  | Some path ->
      let ic = open_in_bin path in
      Fun.protect
        ~finally:(fun () -> close_in ic)
        (fun () -> really_input_string ic (in_channel_length ic))

(* A downloaded file is optional, so that a fresh checkout still has a green
   test suite. Setting BROT_REQUIRE_FIXTURES turns its absence into a failure,
   which is what CI wants: there, a skipped parity test is a hole in the
   gate. *)
let with_download relative ~from f =
  match locate relative with
  | Some path -> f path
  | None when Sys.getenv_opt "BROT_REQUIRE_FIXTURES" = None ->
      skip ~reason:(Printf.sprintf "%s is missing; run %s" relative from) ()
  | None ->
      failf "%s is missing and BROT_REQUIRE_FIXTURES is set; run %s" relative
        from
