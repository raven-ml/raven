(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPUs as nx devices, on real hardware: each GPU 0 that rig opens through its
   vendor's runtime or kernel driver keeps the laws of every memory
   ([Nx_test.Runtimes.laws]). A GPU has no eager backend, so its operations
   raise naming the remedies. A path without a GPU skips. *)

open Windtrap
open Nx_test

let paths = [ "METAL"; "CUDA"; "NV"; "AMD" ]

let suite (path, opener) =
  let memories = match opener () with Ok m -> [ m ] | Error _ -> [] in
  group path (Runtimes.laws memories)

let () =
  let openers =
    List.filter (fun (path, _) -> List.mem path paths) (Rig_gpus.openers ())
  in
  exit
    (run "nx on GPUs"
       (match openers with
       | [] -> [ test "on no GPU" (fun () -> skip ~reason:"no GPU" ()) ]
       | openers -> List.map suite openers))
