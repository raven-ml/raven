(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rig's calls against a reference written from rig.mli, with forked children
   (Rig_model), and what forking leaves the parent. Its own suite: a process
   that ran a domain cannot fork. *)

open Windtrap

let timeout = 120.

let tests =
  [
    group ~timeout "model"
      [
        stateful ~count:40 ~steps:60
          "a forked child sees its devices lost and its host's bytes"
          (Rig_model.commands ~two:false ~fork:true);
      ];
    group ~timeout "leaks"
      [
        test "forked programs leave the parent no C memory or descriptor"
          (fun () -> Rig_model.leaves_nothing ~fork:true);
      ];
  ]

let () =
  Rig_model.hold_gpu ();
  exit (run "rig.model-fork" tests)
