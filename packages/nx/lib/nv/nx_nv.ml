(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let open_ interface i =
  Result.map Nx.Device.make (Nx_nv_device.get ~interface i)

let ok = function Ok d -> d | Error e -> failwith e
let get i = open_ Kernel i
let device i = ok (get i)
let get_pci i = open_ Pci i
let device_pci i = ok (get_pci i)
