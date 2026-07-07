(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The device-independent random-number cases pinned to the CUDA backend:
   random values must be identical to the CPU run (and the reference
   implementation). Skips when no CUDA device is available. *)

open Windtrap

let () = Unix.putenv "DEV" "CUDA"

let cuda_available =
  try
    ignore (Tolk_frontend.Run.device ());
    true
  with _ -> false

let () =
  if cuda_available then
    run "Tolk_frontend rand (CUDA)" Rand_cases.exact_groups
  else
    run "Tolk_frontend rand (CUDA)"
      [
        group "cuda"
          [ test "skipped" (fun () -> skip ~reason:"CUDA unavailable" ()) ];
      ]
