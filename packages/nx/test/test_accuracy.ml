(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The accuracy nx.mli states for its transcendental functions, on the host
   (Nx_test.Accuracy), called from the program's first domain and from one it
   spawns. *)

let () =
  let open Nx_test.Accuracy in
  exit
    (Windtrap.run "nx accuracy"
       (groups host
       @ [ Windtrap.group "from a spawned domain" (groups spawned) ]))
