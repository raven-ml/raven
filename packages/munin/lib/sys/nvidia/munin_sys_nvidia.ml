(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let read ~window =
  let usage i =
    match Rig_nv_nvidia.usage ~window i with
    | Ok u -> Some (u.memory_total, u.memory_free, u.busy)
    | Error _ -> None
  in
  match Rig_nv_nvidia.buses () with
  | Error _ -> []
  | Ok buses -> List.mapi (fun i bus -> (bus, usage i)) buses
