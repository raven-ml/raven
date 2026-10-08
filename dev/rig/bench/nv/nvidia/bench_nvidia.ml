(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Opening GPU 0 through NVIDIA's kernel driver and stopping it: the first open
   of each worker also opens the RM's client, the GPU's file and its objects,
   and registers the GPU with the unified memory driver; the others reuse them
   and make the device's channel group, channels, rings and word. Without an
   NVIDIA GPU the suite has no rows. *)

let open_stop () =
  match Rig_nv_nvidia.open_ 0 with
  | Ok g -> Rig_nv.stop g
  | Error why -> failwith why

let () =
  Rig_nv_nvidia_support.hold_gpu ();
  if Rig_nv_nvidia.count () > 0 then
    exit
    @@ Thumper.run "rig_nv_nvidia"
         [ Thumper.group "open" [ Thumper.bench "0" open_stop ] ]
