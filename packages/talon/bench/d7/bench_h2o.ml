(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Times talon's answers to the H2O questions into h2o.thumper. *)

let () =
  Cases.run "h2o" ~sizes:[ "1e6"; "1e7"; "1e8" ]
    [ ("groupby", H2o.groupby); ("join", H2o.join) ]
