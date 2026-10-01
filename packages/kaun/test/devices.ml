(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Test devices: each holds its values in host memory the host addresses as it
   is, so nx computes on them and a compiled call runs on them as on devices of
   their own. *)

let device name =
  Nx_device.Driver.device ~name ~arch:"test" ~budget:max_int
    (Host_visible
       { memory = Nx_device.Driver.host_memory; mapping = Some Identity })

let cpu1 = device "CPU:1"
let cpu2 = device "CPU:2"

(* [addresses x] is where [x]'s storage starts on each of its devices. A result
   that took a consumed argument's storage has the addresses the argument had. *)
let addresses x =
  match Nx.Repr.v x with
  | Placed p ->
      List.map Nx_device.Buffer.address
        (Nx.Repr.Storage.buffers (Nx.Repr.Placed.storage p))
  | Host _ | Traced _ -> invalid_arg "Devices.addresses: not a placed value"
