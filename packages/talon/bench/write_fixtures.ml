(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Writes the bench's fixtures, customers.csv and transactions.csv, into the
   directory named on the command line, for bench_talon.py. *)

let () =
  match Sys.argv with
  | [| _; dir |] -> Fixtures.write dir
  | _ ->
      prerr_endline "usage: write_fixtures.exe DIR";
      exit 2
