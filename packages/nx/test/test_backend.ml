(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Backends carried by placements: backends that include nx.cpu's kernels and
   change one, on the host device and on a runtime device; refusals, mixed
   operands, the devices a backend runs on, moves between backends, and the two
   ways the host device holds values. *)

open Windtrap
open Nx_test

module Counting = struct
  include (Nx_cpu : Nx_backend.S)

  let name = "counting"
  let adds = ref 0

  let binary k a b ~dst =
    if k = Nx_backend.Add then incr adds;
    Nx_cpu.binary k a b ~dst
end

module Refusing = struct
  include (Nx_cpu : Nx_backend.S)

  let name = "refusing"
  let matmul _ _ ~dst:_ = raise (Nx_backend.Refused "refusing: matmul: none")
end

module Hostless = struct
  include (Nx_cpu : Nx_backend.S)

  let name = "hostless"
  let runs_on _ = false
end

let counting = Nx_backend.make (module Counting)
let refusing = Nx_backend.make (module Refusing)
let hostless = Nx_backend.make (module Hostless)

(* A runtime over host memory. *)
let runtime name =
  Nx_device.Driver.device ~name ~arch:"test" ~budget:max_int
    (Host_visible { memory = Nx_device.Driver.host_memory; mapping = None })

let r1 = Nx.Device.of_runtime (runtime "R1")
let vec a = Nx.create Nx.float32 [| Array.length a |] a
let floats = tensor float_exact

let is_placed (type a b) (x : (a, b) Nx.t) =
  match Nx.Repr.v x with Placed _ -> true | Host _ | Traced _ -> false

let backends =
  group "backends"
    [
      test
        "a backend that includes the host's runs its operations and keeps its \
         placement, on the host device and on a runtime device" (fun () ->
          List.iter
            (fun d ->
              let p = Nx.Placement.device ~backend:counting d in
              let x = Nx.place p (vec [| 1.; 2.; 3. |]) in
              let before = !Counting.adds in
              let y = Nx.add x x in
              equal ~msg:"one add through the backend" int (before + 1)
                !Counting.adds;
              is_true ~msg:"the result keeps the placement"
                (Nx.Placement.equal (Nx.placement y) p);
              equal floats (vec [| 2.; 4.; 6. |]) y;
              let z = Nx.mul y x in
              equal ~msg:"an operation it does not change is the host's" floats
                (vec [| 2.; 8.; 18. |])
                z)
            [ Nx.Device.host; r1 ]);
      test "an operation on host values does not reach another backend"
        (fun () ->
          let before = !Counting.adds in
          ignore (Nx.add (vec [| 1. |]) (vec [| 2. |]));
          equal int before !Counting.adds);
      test "a refused operation raises Refused" (fun () ->
          let p = Nx.Placement.device ~backend:refusing Nx.Device.host in
          let x = Nx.place p (Nx.ones Nx.float32 [| 2; 2 |]) in
          raises_match
            (function Nx_backend.Refused _ -> true | _ -> false)
            (fun () -> ignore (Nx.matmul x x));
          equal floats (Nx.full Nx.float32 [| 2; 2 |] 2.) (Nx.add x x));
      test "operands with two backends raise, naming both placements" (fun () ->
          let x =
            Nx.place
              (Nx.Placement.device ~backend:counting Nx.Device.host)
              (vec [| 1. |])
          and y =
            Nx.place
              (Nx.Placement.device ~backend:refusing Nx.Device.host)
              (vec [| 1. |])
          in
          raises_match
            (function
              | Invalid_argument msg ->
                  String.equal msg
                    "Nx.add: operands on CPU with counting and CPU with \
                     refusing; place one of them"
              | _ -> false)
            (fun () -> ignore (Nx.add x y)));
      test "a host operand joins a placement of another backend" (fun () ->
          let p = Nx.Placement.device ~backend:counting Nx.Device.host in
          let y = Nx.add (Nx.place p (vec [| 1. |])) (vec [| 2. |]) in
          is_true (Nx.Placement.equal (Nx.placement y) p);
          equal floats (vec [| 3. |]) y);
      test "a placement refuses a backend that does not run on the host"
        (fun () ->
          raises_invalid_arg (fun () ->
              Nx.Placement.device ~backend:hostless Nx.Device.host);
          raises_invalid_arg (fun () ->
              Nx.Placement.device ~backend:hostless r1);
          raises_invalid_arg (fun () ->
              Nx.Placement.replicated ~backend:hostless [ Nx.Device.host; r1 ]));
      test "placements differ by backend, and print it" (fun () ->
          let p = Nx.Placement.device ~backend:counting Nx.Device.host in
          is_false (Nx.Placement.equal p Nx.Placement.host);
          is_true (Nx_backend.equal (Nx.Placement.backend p) counting);
          is_true
            (Nx_backend.equal
               (Nx.Placement.backend Nx.Placement.host)
               Nx_cpu.backend);
          equal string "counting" (Nx_backend.name counting);
          equal string "CPU with counting"
            (Format.asprintf "%a" Nx.Placement.pp p);
          equal string "CPU"
            (Format.asprintf "%a" Nx.Placement.pp Nx.Placement.host));
      test "a move between backends on the same devices views the storage"
        (fun () ->
          let x =
            Nx.place
              (Nx.Placement.device ~backend:counting r1)
              (vec [| 1.; 2. |])
          in
          let q = Nx.Placement.device r1 in
          let y = Nx.place q x in
          is_true (Nx.Placement.equal (Nx.placement y) q);
          (match (Nx.Repr.v x, Nx.Repr.v y) with
          | Placed a, Placed b ->
              is_true ~msg:"one storage"
                (Nx.Repr.Placed.storage a == Nx.Repr.Placed.storage b)
          | _ -> fail "expected placed values");
          equal floats (vec [| 1.; 2. |]) y);
      test
        "the host device holds a value of the host placement as a host tensor, \
         and one of another placement as a placed value" (fun () ->
          let x = vec [| 1.; 2. |] in
          is_false ~msg:"created on the host" (is_placed x);
          let p = Nx.Placement.device ~backend:counting Nx.Device.host in
          let y = Nx.place p x in
          is_true ~msg:"another backend on the host device" (is_placed y);
          is_true ~msg:"its results" (is_placed (Nx.add y y));
          is_true ~msg:"created there" (is_placed (Nx.zeros_like y));
          let z = Nx.place Nx.Placement.host y in
          is_false ~msg:"moved back to the host placement" (is_placed z);
          equal floats x z);
    ]

let () = exit (run "nx backends" [ backends ])
