(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let addresses x =
  let buffers =
    match Nx.Repr.v x with
    | Host a -> [ a.buffer ]
    | Placed r -> Nx.Repr.Storage.buffers (Nx.Repr.Placed.storage r)
    | Traced _ -> invalid_arg "a traced value has no storage"
  in
  List.map Nx_device.Buffer.address buffers
