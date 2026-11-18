(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* opens in a process that has spawned a domain, which can no longer fork:
   a string that names no device is refused before any fork, and a name
   raises as Unix.fork does. Its own executable, since the domain stays. *)

open Windtrap

let fork_failure =
  "Unix.fork may not be called after any domain has been spawned"

let after_domain () =
  Domain.join (Domain.spawn ignore);
  equal ~msg:"no name, refused before the fork" bool true
    (Result.is_error (Rig_gpus.opens "CUDA:01"));
  raises ~msg:"the host's name" (Failure fork_failure) (fun () ->
      ignore (Rig_gpus.opens "CPU"));
  raises ~msg:"a GPU's name" (Failure fork_failure) (fun () ->
      ignore (Rig_gpus.opens "NV:4096"));
  equal ~msg:"open_ still opens" bool true
    (Result.is_ok (Rig_gpus.open_ "CPU:1"))

let () =
  if Sys.win32 then exit 0;
  exit
  @@ run "rig_gpus opens after a domain"
       [
         test "opens refuses a non-name, and raises Failure where fork does"
           after_domain;
       ]
