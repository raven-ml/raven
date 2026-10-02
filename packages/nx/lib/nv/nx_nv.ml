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
let detach = Nx_nv_device.detach
let attach = Nx_nv_device.attach
let reset = Nx_nv_device.reset
let fetch_firmware = Nx_nv_device.fetch_firmware
