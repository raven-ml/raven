(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rig finds the record a Metal device declares, and only on it. *)

open Windtrap
module Metal = Rig_metal_abi
module P = Rig_support.Polled

(* A device whose facts declare [capability], open in rig until [f] returns. *)
let with_device capability f =
  let make () = Ok (P.make ~capability ()) in
  match Rig.open_ { name = "metal"; driver = (module P); make } with
  | Error why -> fail why
  | Ok d -> Fun.protect ~finally:(fun () -> Rig.close d) (fun () -> f d)

let split = 0x7f00_2000n
let record = { Metal.align = 4; icb = (fun _ _ -> Error "no Metal"); split }

let test_found () =
  with_device (Metal.declare record) @@ fun d ->
  match Rig.capability d Metal.key with
  | None -> fail "no record on the device"
  | Some metal -> equal ~msg:"its split" nativeint split metal.split

let test_alone () =
  is_none ~msg:"the host" (Rig.capability Rig.host Metal.key);
  with_device (Rig_edge.Capability (Type.Id.make (), record)) @@ fun d ->
  is_none ~msg:"the record under another key" (Rig.capability d Metal.key)

let collected () =
  let collections = ref 0 in
  let[@inline never] made () =
    Metal.icb ~handle:17n ~commands:[| 23n; 42n |] ~collect:(fun () ->
        incr collections)
  in
  let b = made () in
  Gc.full_major ();
  equal int ~msg:"the reachable ICB owns its objects" 0 !collections;
  equal nativeint 17n b.handle;
  equal (array nativeint) [| 23n; 42n |] b.commands;
  ignore (Sys.opaque_identity b);
  for round = 1 to 4 do
    Gc.full_major ();
    equal int ~msg:(Printf.sprintf "collection %d" round) 1 !collections
  done

let () =
  exit
  @@ run "rig_metal_abi"
       [
         group ~timeout:10. "key"
           [
             test "finds the record the device declared" test_found;
             test "finds none on another device" test_alone;
           ];
         test "an unreachable ICB collects its objects once" collected;
       ]
