(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Times talon's answers to the TPC-H queries into tpch.thumper. *)

let () =
  Cases.run "tpch" ~sizes:[ "sf0.1"; "sf1"; "sf10" ] [ ("tpch", Tpch.workload) ]
