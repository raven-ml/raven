(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Widths past what an array holds, for Nx.extend, whose rounds past one fold
   allocate before its result exists: each raises naming the function before
   anything is allocated. The suite runs under an address-space limit (see
   dune), so a missing check fails within it instead of allocating the
   machine's memory. *)

open Windtrap

let x = Nx.create Nx.int32 [| 2 |] [| 1l; 2l |]
let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

let extends =
  group "extend"
    [
      test "widths whose sum wraps an int raise naming them" (fun () ->
          raises
            (Invalid_argument
               (Printf.sprintf
                  "Nx.extend: widths (%d, %d) of int32 [2] count past an int"
                  max_int max_int))
            (fun () -> Nx.extend `Symmetric_half (max_int, max_int) x));
      test "widths past an array's extent raise" (fun () ->
          invalid ~by:"Nx.extend" (fun () ->
              Nx.extend `Periodic (1 lsl 60, 0) x));
    ]

let () = exit (run "nx widths" [ extends ])
