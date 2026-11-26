(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rig's calls against a reference written from rig.mli, with forked children
   (Rig_model), and what forking leaves the parent. Its own suite: a process
   that ran a domain cannot fork. *)

open Windtrap

let tests =
  [
    group "stops and forks"
      [
        stateful ~count:80 ~steps:60
          "a forked child finds its drivers' devices lost, its io devices live \
           and its host memory's bytes, while its parent reads a file"
          (Rig_model.commands ~two:false ~fork:true);
      ];
    group "reclamation"
      [
        test "forked programs leave the parent no C memory or descriptor"
          (fun () -> Rig_model.leaves_nothing ~fork:true);
      ];
  ]

let () = exit (run "rig.model-fork" tests)
