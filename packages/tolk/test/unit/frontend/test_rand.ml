(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Random-number generation on the process-wide default device (DEV selects
   the backend; CPU on machines without an accelerator). The cases live in
   [Rand_cases]; [test_rand_cuda] runs the device-independent subset pinned
   to CUDA. *)

let () = Windtrap.run "Tolk_frontend rand" Rand_cases.all_groups
