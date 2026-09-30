(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A device whose values live in memory of its own, as on a GPU: a runtime over
   host memory, whose statistics count the bytes read from it. *)

let runtime =
  Nx_device.Driver.device ~name:"TEST" ~arch:"test" ~budget:max_int
    (Host_visible { memory = Nx_device.Driver.host_memory; mapping = None })

let device = Nx.Device.of_runtime runtime
let place x = Nx.place (Nx.Placement.device device) x

(* The bytes read from the device so far. *)
let bytes_read () = Nx_device.Stats.bytes_out (Nx_device.stats runtime)
