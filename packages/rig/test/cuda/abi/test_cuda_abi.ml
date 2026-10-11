(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rig finds the record a CUDA device declares, and only on it. *)

open Windtrap
module Cuda = Rig_cuda_abi
module P = Rig_support.Polled

(* A device whose facts declare [capability], open in rig until [f] returns. *)
let with_device capability f =
  let make () = Ok (P.make ~capability ()) in
  match Rig.open_ { name = "cuda"; driver = (module P); make } with
  | Error why -> fail why
  | Ok d -> Fun.protect ~finally:(fun () -> Rig.close d) (fun () -> f d)

let launch_kernel = 0x7f00_1000n
let symbol = function "cuLaunchKernel" -> Some launch_kernel | _ -> None
let graph _ = Error "no graphs here"
let record = { Cuda.symbol; graph }

let test_found () =
  with_device (Cuda.declare record) @@ fun d ->
  match Rig.capability d Cuda.key with
  | None -> fail "no record on the device"
  | Some cuda ->
      equal ~msg:"its lookup" (option nativeint) (Some launch_kernel)
        (cuda.symbol "cuLaunchKernel")

let test_alone () =
  is_none ~msg:"the host" (Rig.capability Rig.host Cuda.key);
  with_device (Rig_edge.Capability (Type.Id.make (), record)) @@ fun d ->
  is_none ~msg:"the record under another key" (Rig.capability d Cuda.key)

let () =
  exit
  @@ run "rig_cuda_abi"
       [
         group ~timeout:10. "key"
           [
             test "finds the record the device declared" test_found;
             test "finds none on another device" test_alone;
           ];
       ]
