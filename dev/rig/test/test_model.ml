(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rig's calls against a reference written from rig.mli, on one domain and on
   two (Rig_model). *)

open Windtrap

let timeout = 120.

let tests =
  [
    group ~timeout "model"
      [
        stateful ~count:400 ~steps:60
          "every call answers as rig.mli says, on every kind of device"
          (Rig_model.commands ~two:false ~fork:false);
        stateful ~count:30 ~steps:30 ~domains:2
          "calls on two domains answer as some order of them"
          (Rig_model.commands ~two:true ~fork:false);
      ];
  ]

let () =
  Rig_model.hold_gpu ();
  exit (run "rig.model" tests)
