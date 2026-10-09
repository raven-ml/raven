(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's per-call costs above the array layer, each row what a user calls:
   reading a value's shape, placing a value where it already lies, and placing a
   host value on a device that maps the host's memory. *)

module A = Nx_array

let memory =
  match Rig.memory_device "bench-m0" with Ok d -> d | Error e -> failwith e

module Mem = (val Nx.devices [ memory ])

let host n =
  Nx.Repr.of_array Nx.host
    (A.of_array A.Dtype.Float32 [| n |] (Array.make n 1.))

let x1 = host 1
let x16 = host 16

let dispatch_rows =
  Thumper.group "dispatch"
    [ Thumper.bench "shape" (fun () -> Nx.shape (Thumper.black_box x1)) ]

let place_rows =
  Thumper.group "place"
    [
      Thumper.bench "equal-1" (fun () ->
          Nx.place Nx.Placement.host (Thumper.black_box x1));
      Thumper.bench "borrow-memory-device-16" (fun () ->
          Nx.place Mem.on (Thumper.black_box x16));
    ]

let () = exit @@ Thumper.run "nx" [ dispatch_rows; place_rows ]
