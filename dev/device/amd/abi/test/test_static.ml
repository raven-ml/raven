(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The library's tables are static data: a program that links it and reads every
   table holds no more live words than one that links the same Stdlib modules
   alone, but for the modules' own few words (live_with.ml, live_without.ml). A
   register table built at initialisation held 47,000. *)

open Windtrap

let timeout = Device_amd_abi_support.timeout
let bound = 64

let live =
  test ~timeout "linking the library and reading its tables keeps no table live"
    (fun () ->
      match In_channel.with_open_text "live.txt" In_channel.input_lines with
      | [ without; with_ ] ->
          let without = int_of_string without and with_ = int_of_string with_ in
          at_most int ~than:(without + bound) with_
      | lines -> failf "live.txt has %d lines" (List.length lines))

let () = exit (run "device_amd_abi.static" [ live ])
