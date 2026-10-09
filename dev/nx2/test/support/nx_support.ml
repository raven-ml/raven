(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let devices =
  Array.init 4 (fun k ->
      match Rig.memory_device (Printf.sprintf "m%d" k) with
      | Ok d -> d
      | Error e -> failwith e)

let memory k = devices.(k)
