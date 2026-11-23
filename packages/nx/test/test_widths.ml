(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Widths past what an array holds, for Nx.extend, whose rounds past one fold
   allocate before its result exists: each raises naming the function before
   anything is allocated. The suite runs under a limit of its address space, so
   a missing check fails within it instead of allocating the machine's
   memory. *)

open Windtrap

external limit_address_space : int -> int = "nx_test_limit_address_space"

(* The bytes the process maps: its status's VmSize. *)
let mapped () =
  let field l = Scanf.sscanf_opt l "VmSize: %d kB" (fun k -> k * 1024) in
  let lines =
    In_channel.with_open_text "/proc/self/status" In_channel.input_lines
  in
  match List.find_map field lines with
  | Some n -> n
  | None -> failwith "no VmSize in /proc/self/status"

(* Room for the suite past what the process maps at its start. The limit is
   relative: AddressSanitizer maps terabytes of shadow before the program
   runs. *)
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
  (match limit_address_space (mapped () + room) with
  | 0 -> ()
  | e -> failwith (Printf.sprintf "setrlimit: errno %d" e));
  exit (run "nx widths" [ extends ])
