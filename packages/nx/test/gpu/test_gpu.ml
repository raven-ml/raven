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

let suite (g : Rig_gpus.gpu) =
  let memories = match g.open_ () with Ok m -> [ m ] | Error _ -> [] in
  group g.name (Runtimes.laws memories)

let () =
  let gpus =
    List.concat_map
      (fun (p : Rig_gpus.path) ->
        if not (List.mem p.kind paths) then []
        else match p.gpus () with Ok gpus -> gpus | Error _ -> [])
      (Rig_gpus.paths ())
  in
  exit
    (run "nx on GPUs"
       (match gpus with
       | [] -> [ test "on no GPU" (fun () -> skip ~reason:"no GPU" ()) ]
       | gpus -> List.map suite gpus))
