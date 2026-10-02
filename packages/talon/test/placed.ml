(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A test device: its values live in memory of their own, whose statistics count
   the bytes read from it. *)

let device = Nx.Device.v (Cpu 1)
let place x = Nx.place (Nx.Placement.on device) x

(* The bytes read from the device so far. *)
let bytes_read () =
  Nx_device.Stats.bytes_out (Nx_device.stats (Nx.Device.memory device))
