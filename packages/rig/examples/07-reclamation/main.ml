(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reclamation.

   Nothing frees a buffer by hand. Once a buffer is unreachable, the
   collector hands its memory back to its device, which keeps it for reuse.
   A device holds at most its budget. An allocation that the budget refuses
   first frees the device's cache, collects unreachable buffers and tries
   again. It raises [Out_of_memory] when it is larger than the budget, or
   when live buffers fill the budget. *)

open Rig

let kib = 1024
let mib = 1024 * kib

let try_alloc f =
  match f () with
  | _ -> print_endline "allocated"
  | exception Out_of_memory (d, n) ->
      Printf.printf "Out_of_memory: %s cannot allocate %d KiB; %d KiB in use\n"
        (name d) (n / kib)
        (allocated d / kib)

let () =
  let d = Result.get_ok (memory_device "M") in
  Printf.printf "budget of %s: %s\n" (name d)
    (if budget d = max_int then "unbounded" else string_of_int (budget d));
  set_budget d mib;
  Printf.printf "budget of %s: %d KiB\n\n" (name d) (budget d / kib);

  (* An allocation larger than the budget is refused. *)
  try_alloc (fun () -> Buffer.create d (2 * mib));

  (* Buffers the program does not keep come back. 100 buffers of 256 KiB,
     25 MiB in all, are allocated within a budget of 1 MiB, because each
     refused allocation collects the ones before it. *)
  for _ = 1 to 100 do
    ignore (Sys.opaque_identity (Buffer.create d (256 * kib)))
  done;
  print_endline "100 buffers of 256 KiB allocated, none kept";

  (* Live buffers are never freed. Four of them fill the budget, and the
     next allocation raises once reclaiming finds no room. *)
  let live = List.init 4 (fun _ -> Buffer.create d (256 * kib)) in
  try_alloc (fun () -> Buffer.create d (256 * kib));

  (* Once they are unreachable, their memory serves the next allocation. *)
  ignore (Sys.opaque_identity live);
  try_alloc (fun () -> Buffer.create d mib)
