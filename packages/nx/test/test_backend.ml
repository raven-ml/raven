(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Backends run with Nx.Op.kernels: which eager operations a backend computes,
   the innermost covering one, a backend that lacks an operation, devices it
   does not cover, the operations nx answers itself, and domains. *)

open Windtrap
open Nx_test

(* A device the host does not compute on, as a GPU: memory the host addresses,
   and a driver that loads programs. *)
let gpu =
  Nx_device.Driver.device ~name:"GPU" ~arch:"test" ~budget:max_int
    ~load:(fun ~binary:_ -> Error "no programs")
    (Host_visible { memory = Nx_device.Driver.host_memory; mapping = None })

(* [counting label covers] is a backend of nx.cpu's kernels named [label] over
   the devices [covers] accepts, counting its additions. *)
let counting label covers =
  let adds = ref 0 in
  let k =
    (module struct
      include (Nx_cpu : Nx_backend.S)

      let name = label
      let runs_on = covers

      let binary k a b ~dst =
        if k = Nx_backend.Add then incr adds;
        Nx_cpu.binary k a b ~dst
    end : Nx_backend.S)
  in
  (k, adds)

let under k f = Nx.Op.intercept (Nx.Op.kernels k) f
let on_host d = Nx_device.equal d Nx_device.host
let on_gpu d = Nx_device.equal d gpu

module Lacking = struct
  include (Nx_cpu : Nx_backend.S)

  let name = "lacking"

  let matmul _ _ ~dst:_ =
    raise (Nx_backend.Refused "lacking does not implement matmul")
end

let vec a = Nx.create Nx.float32 [| Array.length a |] a
let floats = tensor float_exact
let refused = function Nx_backend.Refused _ -> true | _ -> false

let runs =
  group "run"
    [
      test "a backend computes the eager operations on host values it wraps"
        (fun () ->
          let k, adds = counting "counting" on_host in
          let y =
            under k (fun () -> Nx.add (vec [| 1.; 2. |]) (vec [| 3.; 4. |]))
          in
          equal ~msg:"one addition through the backend" int 1 !adds;
          equal floats (vec [| 4.; 6. |]) y);
      test "outside its run, a backend computes nothing" (fun () ->
          let k, adds = counting "counting" on_host in
          ignore (under k (fun () -> ()));
          ignore (Nx.add (vec [| 1. |]) (vec [| 2. |]));
          equal int 0 !adds);
      test "the innermost backend that covers the operands computes" (fun () ->
          let outer, outer_adds = counting "outer" on_host in
          let inner, inner_adds = counting "inner" on_host in
          ignore
            (under outer (fun () ->
                 under inner (fun () -> Nx.add (vec [| 1. |]) (vec [| 2. |]))));
          equal ~msg:"inner" int 1 !inner_adds;
          equal ~msg:"outer" int 0 !outer_adds);
      test "a backend passes operations on devices it does not cover outward"
        (fun () ->
          let outer, outer_adds = counting "outer" on_host in
          let inner, inner_adds = counting "inner" on_gpu in
          ignore
            (under outer (fun () ->
                 under inner (fun () -> Nx.add (vec [| 1. |]) (vec [| 2. |]))));
          equal ~msg:"inner" int 0 !inner_adds;
          equal ~msg:"outer" int 1 !outer_adds);
      test
        "a backend that lacks an operation raises Refused, and the operation \
         reaches no outer backend" (fun () ->
          let outer, _ = counting "outer" on_host in
          let x = Nx.ones Nx.float32 [| 2; 2 |] in
          raises_match refused (fun () ->
              under outer (fun () ->
                  under (module Lacking) (fun () -> ignore (Nx.matmul x x)))));
      test
        "constants, movements, reads and place work under a backend that \
         computes nothing" (fun () ->
          let x = vec [| 1.; 2.; 3.; 4. |] in
          let y =
            under
              (module Lacking)
              (fun () ->
                let z = Nx.zeros Nx.float32 [| 2 |] in
                let t = Nx.transpose (Nx.reshape [| 2; 2 |] x) in
                let p = Nx.place (Nx.Placement.on [ Nx.Device.cpu 1 ]) t in
                equal (list float_exact) [ 0.; 0. ]
                  (Array.to_list (Nx.to_array z));
                Nx.place Nx.Placement.host p)
          in
          equal floats (Nx.create Nx.float32 [| 2; 2 |] [| 1.; 3.; 2.; 4. |]) y);
      test
        "an eager operation on a device no backend covers raises before any \
         work, naming the remedies" (fun () ->
          let x = Nx.place (Nx.Placement.on [ gpu ]) (vec [| 1.; 2. |]) in
          raises
            (Invalid_argument
               "Nx.add: an operand is on GPU, where nx.cpu does not compute; \
                apply it under Rune.jit, run it under a backend's run that \
                covers GPU, or Nx.place it on Nx.Placement.host") (fun () ->
              ignore (Nx.add x x));
          equal ~msg:"reads work" floats (vec [| 1.; 2. |]) x);
      test
        "a backend that covers a device computes its eager operations, keeping \
         the placement" (fun () ->
          let k, adds = counting "gpu" on_gpu in
          let p = Nx.Placement.on [ gpu ] in
          let x = Nx.place p (vec [| 1.; 2. |]) in
          let y = under k (fun () -> Nx.add x (vec [| 3.; 4. |])) in
          equal ~msg:"one addition through the backend" int 1 !adds;
          equal Devices.placement p (Nx.placement y);
          equal floats (vec [| 4.; 6. |]) y);
      test "a domain spawned inside a run computes with nx.cpu" (fun () ->
          let k, adds = counting "counting" on_host in
          let y =
            under k (fun () ->
                Domain.join
                  (Domain.spawn (fun () -> Nx.add (vec [| 1. |]) (vec [| 2. |]))))
          in
          equal int 0 !adds;
          equal floats (vec [| 3. |]) y);
    ]

let () = exit (run "nx backends" [ runs ])
