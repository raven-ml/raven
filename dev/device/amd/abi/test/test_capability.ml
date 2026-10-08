(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The key, as a device's records are found: in a table of bindings, each under
   its own key. *)

open Windtrap
open Device_amd_abi
module S = Device_amd_abi_support

let timeout = S.timeout

type binding = B : 'a Type.Id.t * 'a -> binding

let find : type a. a Type.Id.t -> binding list -> a option =
 fun k bindings ->
  let found (B (k', v)) : a option =
    match Type.Id.provably_equal k k' with
    | Some Type.Equal -> Some v
    | None -> None
  in
  List.find_map found bindings

let gpu = S.gpu ~sdma:(7, 0, 0) ~compute_units:64 (12, 0, 1)

let record =
  {
    Capability.gpu;
    clock_hz = 100_000_000;
    compute = Pm4;
    place = 0x7f00_1000n;
    segment = 0x7f00_2000n;
  }

let key =
  test ~timeout "a record is found under the key" (fun () ->
      let other : int Type.Id.t = Type.Id.make () in
      match
        find Capability.key [ B (other, 1); B (Capability.key, record) ]
      with
      | None -> fail "no record under the key"
      | Some c -> equal int 100_000_000 c.clock_hz)

let () = exit (run "device_amd_abi.capability" [ key ])
