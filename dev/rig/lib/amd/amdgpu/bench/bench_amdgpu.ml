(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Opening GPU 0 through the amdgpu driver and stopping it: the first open of
   each worker also opens KFD, acquires the GPU's address space and makes the
   event page; the others make the device's events, queues, rings and word.
   Without an AMD GPU the suite has no rows. *)

module A = Rig_amd
module P = Rig_amd_amdgpu

let open_stop () =
  match P.open_ 0 with Ok g -> ignore (A.stop g) | Error why -> failwith why

let () =
  if P.count () > 0 then
    exit
    @@ Thumper.run "rig_amd_amdgpu"
         [ Thumper.group "open" [ Thumper.bench "0" open_stop ] ]
