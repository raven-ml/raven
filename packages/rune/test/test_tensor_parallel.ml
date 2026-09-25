(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Parallel programs over CPU:1..CPU:4. A tensor-parallel MLP: the first weight
   split by columns, the second by rows, both captured where they live, and the
   batch entering from the host. It equals one device, uploads the batch once
   per device, moves between devices exactly what the allreduce of its result
   moves, and leaves replicas of that result equal bit for bit. An
   expert-parallel product: each device multiplies the routes to its own
   experts, and only their products and the routes' ids cross devices.

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
    ~synchronize:(fun timeout -> Tolk.Device.synchronize ?timeout cpu)
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

(* Experts *)

let bytes shape f =
  Nx.create Nx.uint8 shape (Array.init (Array.fold_left ( * ) 1 shape) f)

(* Sixteen MXFP4 experts of [8; 64], four per device, and eight tokens routed to
   two experts each, which the last expert of each device never is. Their scales
   are NaN, so a device that read one would poison its product. *)
let experts, per, n, k, tokens, top = (16, 4, 8, 64, 8, 2)
let unchosen e = e mod per = per - 1

let routes =
  let chosen =
    List.filter (fun e -> not (unchosen e)) (List.init experts Fun.id)
  in
  Nx.create Nx.int32 [| tokens; top |]
    (Array.init (tokens * top) (fun i ->
         Int32.of_int (List.nth chosen (i * 5 mod List.length chosen))))

let stack =
  Nx_quant.mxfp4
    ~scales:
      (bytes
         [| experts; n; k / 32 |]
         (fun i -> if unchosen (i / (n * k / 32)) then 255 else 120 + (i mod 9)))
    (bytes [| experts; n; k / 2 |] (fun i -> i * 37 mod 256))

let tokens_x = mat 5 tokens k |> Nx.reshape [| tokens; 1; 1; k |]

(* Each device's lane of four experts, its routes kept and the others' marked
   -1, and the lanes' products summed. *)
let moe experts firsts =
  let lanes =
    Rune.vmap
      Nx.Ptree.(Nx_quant.ptree @-> tensor @-> returns tensor)
      (fun experts first ->
        let local = Nx.sub routes first in
        let mine =
          Nx.logical_and
            (Nx.greater_equal_s local 0l)
            (Nx.less_s local (Int32.of_int per))
        in
        Nx_quant.apply
          ~ids:(Nx.where mine local (Nx.scalar_like local (-1l)))
          experts tokens_x)
      experts firsts
  in
  Nx.sum ~axes:[ 0 ] lanes

let test_expert_parallel_product () =
  let expect =
    Rune.jit'
      ~devices:[ List.hd cpus ]
      (Nx_quant.apply ~ids:routes stack)
      tokens_x
  in
  is_true ~msg:"the unused experts' NaN is not read on one device"
    (Array.for_all Float.is_finite (to_arr expect));
  let (Nx_quant.Mxfp4 { codes; scales }) = stack in
  let lanes x =
    Nx.reshape (Array.append [| 4; per |] (Array.sub (Nx.shape x) 1 2)) x
  in
  let split = Nx.Placement.sharded ~axis:0 cpus in
  let lanes =
    Nx_quant.place split (Nx_quant.mxfp4 ~scales:(lanes scales) (lanes codes))
  in
  let firsts =
    Nx.place split (Nx.create Nx.int32 [| 4 |] [| 0l; 4l; 8l; 12l |])
  in
  let f =
    Rune.jit Nx.Ptree.(Nx_quant.ptree @-> tensor @-> returns tensor) moe
  in
  ignore (f lanes firsts);
  let y, peer = peer_bytes (fun () -> f lanes firsts) in
  equal ~msg:"equals one device" (array float_exact) (to_arr expect) (to_arr y);
  (* Every lane's routes over the experts split by lane are a partial product on
     each device, summed across them: what crosses is the allreduce of every
     lane's products, and the gather of the lanes' ids that brings the routes
     whole. No expert crosses. *)
  let sum = Rune.jit' (Nx.sum ~axes:[ 0 ]) in
  let partials =
    Nx.place split
      (Nx.reshape [| 4; 4 * tokens * top; 1; n |] (mat 6 (16 * tokens * top) n))
  in
  ignore (sum partials);
  let _, allreduce = peer_bytes (fun () -> sum partials) in
  let gather = Rune.jit' (Nx.place (Nx.Placement.replicated cpus)) in
  let ids = Nx.place split (Nx.zeros Nx.int32 [| 4 * tokens * top |]) in
  ignore (gather ids);
  let _, ids_gather = peer_bytes (fun () -> gather ids) in
  equal ~msg:"the lanes' products and ids cross devices" int
    (allreduce + ids_gather) peer

let tests =
  [
    group "tensor parallelism"
      [ test "a column-then-row MLP" test_tensor_parallel_mlp ];
    group "expert parallelism"
      [
        test "each device multiplies its experts' routes"
          test_expert_parallel_product;
      ];
  ]

let () = run "rune tensor parallelism" tests
