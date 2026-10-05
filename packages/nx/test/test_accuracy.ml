(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The accuracy nx.mli states for its transcendental functions, on the host
   (Nx_test.Accuracy). *)

let () = exit (Windtrap.run "nx accuracy" Nx_test.Accuracy.(groups host))
