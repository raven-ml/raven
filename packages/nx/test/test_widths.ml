(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Widths past what an array holds, for Nx.extend, whose rounds past one fold
   allocate before its result exists: each raises naming the function before
   anything is allocated. The suite runs under a limit of its memory, so a
   missing check fails within it instead of allocating the machine's memory. *)

open Windtrap

external limit_memory : int -> int = "nx_test_limit_memory"

(* Room for the suite past what the process holds at its start: the memory it
   maps on Linux, the memory it commits on Windows. The limit is relative:
   AddressSanitizer maps terabytes of shadow before the program runs. *)
let room = 2_000_000_000
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

let () =
  (match limit_memory room with
  | 0 -> ()
  | e -> failwith (Printf.sprintf "cannot limit the memory: error %d" e));
  exit (run "nx widths" [ extends ])
