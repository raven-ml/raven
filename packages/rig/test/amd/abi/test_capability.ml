(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rig finds the record an AMD device declares, and only on it. *)

open Windtrap
open Rig_amd_abi
module S = Rig_amd_abi_support

let timeout = S.timeout

module P = Rig_support.Polled

(* A device whose facts declare [capability], open in rig until [f] returns. *)
let with_device capability f =
  let make () = Ok (P.make ~capability ()) in
  match Rig.open_ { name = "amd"; driver = (module P); make } with
  | Error why -> fail why
  | Ok d -> Fun.protect ~finally:(fun () -> Rig.close d) (fun () -> f d)

let gpu = S.gpu ~sdma:(7, 0, 0) ~compute_units:64 (12, 0, 1)

let record =
  {
    Capability.gpu;
    clock_hz = 100_000_000;
    compute = Pm4;
    wgps = [| [| 0xff; 0xff |] |];
    trace = (fun () -> Error "no trace buffers");
  }

let key =
  group ~timeout "key"
    [
      test "finds the record the device declared" (fun () ->
          with_device (Capability.declare record) @@ fun d ->
          match Rig.capability d Capability.key with
          | None -> fail "no record on the device"
          | Some c -> equal int 100_000_000 c.clock_hz);
      test "finds none on another device" (fun () ->
          is_none ~msg:"the host" (Rig.capability Rig.host Capability.key);
          with_device (Rig_edge.Capability (Type.Id.make (), record))
          @@ fun d ->
          is_none ~msg:"the record under another key"
            (Rig.capability d Capability.key));
    ]

let () = exit (run "rig_amd_abi.capability" [ key ])
