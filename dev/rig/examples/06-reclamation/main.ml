(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reclamation.

   Nothing frees a buffer by hand. Once a buffer is unreachable, the collector
   hands its memory back to its device, which keeps it for reuse. A device holds
   at most its budget: an allocation the budget refuses first releases the
   device's cache, collects unreachable buffers and tries again, and raises
   [Out_of_memory] only when live buffers fill the budget. *)

open Rig

let kib = 1024
let mib = 1024 * kib

let try_alloc f =
  match f () with
  | _ -> print_endline "allocated"
  | exception Out_of_memory (d, n) ->
      Printf.printf "Out_of_memory: %s cannot allocate %d KiB\n" (name d)
        (n / kib)

let () =
  let d = Result.get_ok (memory_device "M") in
  Printf.printf "budget of %s: %s\n" (name d)
    (if budget d = max_int then "unbounded" else string_of_int (budget d));
  set_budget d mib;
  Printf.printf "budget of %s: %d KiB\n\n" (name d) (budget d / kib);

  (* More than the budget is refused at once. *)
  try_alloc (fun () -> Buffer.create d (2 * mib));

  (* Buffers that are dropped come back: 100 buffers of 256 KiB, 25 MiB in all,
     fit a budget of 1 MiB because each refused allocation collects the ones
     before it. *)
  for _ = 1 to 100 do
    ignore (Sys.opaque_identity (Buffer.create d (256 * kib)))
  done;
  print_endline "100 buffers of 256 KiB allocated and dropped";

  (* Live buffers are never released: four of them fill the budget, and the next
     allocation raises once reclaiming finds nothing to take. *)
  let live = List.init 4 (fun _ -> Buffer.create d (256 * kib)) in
  try_alloc (fun () -> Buffer.create d (256 * kib));

  (* Once they are dropped, their memory serves the next allocation. *)
  ignore (Sys.opaque_identity live);
  try_alloc (fun () -> Buffer.create d mib)
