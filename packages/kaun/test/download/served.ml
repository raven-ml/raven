(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A curl that writes "served" to its standard output, whatever it is asked,
   and succeeds. It is a program on every system, Windows included, where a
   script cannot stand in for curl.exe. *)

let () = print_string "served"
