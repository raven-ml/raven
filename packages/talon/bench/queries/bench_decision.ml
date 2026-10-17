(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Times talon's answers to the decision-support queries into
   decision.thumper. *)

let () =
  Cases.run "decision"
    ~sizes:[ "sf0.1"; "sf1"; "sf10" ]
    [ ("decision", Decision.workload) ]
  |> exit
