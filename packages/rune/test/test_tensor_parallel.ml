(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A tensor-parallel MLP over CPU:1..CPU:4: the first weight split by columns,
   the second by rows, both captured where they live, and the batch entering
   from the host. It equals one device, uploads the batch once per device, moves
   between devices exactly what the allreduce of its result moves, and leaves
   replicas of that result equal bit for bit.

   The CPU opener is replaced before any device opens by one whose allocator
   records every transfer between devices, so the test reads traffic off the
   run. It runs alone in its executable for that reason. *)

open Windtrap
open Rune_test_support.Support

(* Counting devices *)

let transfers : (string * string, int) Hashtbl.t = Hashtbl.create 8

let counting (Tolk.Device.Allocator.Pack a) =
  let transfer ~dest ~src ~dest_device ~src_device n =
    let key = (src_device, dest_device) in
    let sum = Option.value (Hashtbl.find_opt transfers key) ~default:0 in
    Hashtbl.replace transfers key (sum + n);
    let bytes = Bytes.create n in
    a.copyout bytes src;
    a.copyin dest bytes;
    true
  in
  Tolk.Device.Allocator.Pack
    { a with transfer = Some transfer; supports_transfer = true }

let create name =
  let cpu = Tolk_cpu.create ~aligned:false name in
  let renderer = Tolk.Device.renderer cpu in
  let renderer_set =
    Tolk.Device.Renderer_set.make ~device:name
      ~arch:(Tolk.Renderer.target renderer).Tolk_uop.Target.arch
      [ ("CLANG", fun _ -> renderer) ]
  in
  let allocator =
    counting
      (Tolk.Device.Allocator.Pack
         (Tolk.Device.Lru_allocator.wrap
            (Tolk_uop.Storage.Host_allocator.make ~synchronize:ignore)))
  in
  Tolk.Device.make ~name ~allocator ~renderer_set
    ~runtime:(Tolk.Device.runtime cpu)
    ~synchronize:(fun () -> Tolk.Device.synchronize cpu)
    ~bufferize:(Tolk.Device.bufferize cpu)
    ()

let () = Tolk.Device.register "CPU" create

(* Bytes a call moves between two of rune's devices. *)
let peer_bytes f =
  let before = Hashtbl.copy transfers in
  let y = f () in
  let moved =
    Hashtbl.fold
      (fun ((src, dst) as key) n sum ->
        let n = n - Option.value (Hashtbl.find_opt before key) ~default:0 in
        if src <> dst && src <> "CPU" && dst <> "CPU" then sum + n else sum)
      transfers 0
  in
  (y, moved)

let cpus = List.init 4 (fun i -> Rune.device (Printf.sprintf "CPU:%d" (i + 1)))
let batch, width, hidden = (8, 64, 256)

let mat seed r c =
  Nx.create f32 [| r; c |]
    (Array.init (r * c) (fun i -> sin (float_of_int ((seed * 7919) + i)) *. 0.2))

let bits v = Array.map Int32.bits_of_float (to_arr v)

let test_tensor_parallel_mlp () =
  let w1 = mat 1 width hidden and w2 = mat 2 hidden width in
  let x = mat 3 batch width in
  let mlp w1 w2 x = Nx.matmul (Nx.relu (Nx.matmul x w1)) w2 in
  let expect = Rune.jit' ~devices:[ List.hd cpus ] (mlp w1 w2) x in
  let columns = Nx.place (Nx.Placement.sharded ~axis:1 cpus) w1
  and rows = Nx.place (Nx.Placement.sharded ~axis:0 cpus) w2 in
  let tp = Rune.jit' (mlp columns rows) in
  let y = tp x in
  equal ~msg:"the result is a copy on each device"
    (Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal)
    (Nx.Placement.replicated cpus)
    (Nx.placement y);
  check_arr ~eps:1e-5 ~msg:"equals one device" (to_arr expect) y;
  Rune.reset_jit_stats ();
  let y, peer = peer_bytes (fun () -> tp x) in
  equal ~msg:"uploads the batch once per device" int
    (List.length cpus * batch * width * 4)
    (Rune.jit_stats ()).bytes_to_device;
  let sum = Rune.jit' (Nx.sum ~axes:[ 0 ]) in
  let partials =
    Nx.place
      (Nx.Placement.sharded ~axis:0 cpus)
      (Nx.reshape [| 4; batch; width |] (mat 4 (4 * batch) width))
  in
  ignore (sum partials);
  let _, allreduce = peer_bytes (fun () -> sum partials) in
  is_true ~msg:"the allreduce moves bytes" (allreduce > 0);
  equal ~msg:"moves exactly the allreduce's bytes" int allreduce peer;
  let replicas =
    List.map (fun d -> bits (Nx.place (Nx.Placement.device d) y)) cpus
  in
  List.iteri
    (fun i r ->
      equal
        ~msg:(Printf.sprintf "replica %d equals replica 0 bit for bit" i)
        (array int32) (List.hd replicas) r)
    replicas

let tests =
  [
    group "tensor parallelism"
      [ test "a column-then-row MLP" test_tensor_parallel_mlp ];
  ]

let () = run "rune tensor parallelism" tests
