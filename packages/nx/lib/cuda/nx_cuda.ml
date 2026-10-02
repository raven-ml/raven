(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let get i = Result.map Nx.Device.make (Nx_cuda_device.get i)
let device i = match get i with Ok d -> d | Error e -> failwith e
