(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A run with params, one metric and a summary, in the default store. *)

open Munin

let () =
  let id =
    Session.within
      (Session.start ~experiment:"demo" ~name:"baseline"
         ~params:[ ("lr", `Float 0.001); ("epochs", `Int 3) ]
         ~steps:100 ())
    @@ fun s ->
    let loss = Session.metric s ~goal:`Minimize "train/loss" in
    for step = 1 to 100 do
      Session.log loss ~step (1. /. Float.of_int step)
    done;
    Session.set_summary s [ ("test/acc", `Float 0.93) ];
    Session.id s
  in
  Printf.printf "run: %s\n" id
