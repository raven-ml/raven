(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let storage_view ~src ~offset ~size ~dtype =
  let module U = Tolk_uop.Uop in
  let module D = Tolk_uop.Dtype in
  let offset = U.alu_binary ~op:Tolk_uop.Ops.Mul ~lhs:offset
      ~rhs:(U.const_int (D.itemsize (U.dtype src))) in
  let bytes = U.bitcast ~src ~dtype:D.int8 in
  U.bitcast ~dtype ~src:(U.shrink ~src:bytes ~offset
      ~size:(U.const_int (size * D.itemsize dtype)))

let bufferized_call sink =
  let sink, map = Tolk.Bufferize.run sink in
  Tolk.Callify.transform_to_call sink, map


(* Multi-device execution: sharded and replicated schedules running on
   several CPU device instances (and, when available, a duplicated CUDA
   device tuple), exercising the engine's multi-buffer resolution, per-device
   kernel launches, and cross-device copies. *)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop

let () = Device.register "CPU" Tolk_cpu.create
let () = Device.register "CUDA" Tolk_cuda.create

let cuda_available =
  lazy (match Device.get "CUDA" with _ -> true | exception _ -> false)

(* Data helpers *)

let f32_to_bytes values =
  let bytes = Bytes.create (Array.length values * 4) in
  Array.iteri
    (fun i v -> Bytes.set_int32_le bytes (i * 4) (Int32.bits_of_float v))
    values;
  bytes

let read_f32 buf =
  let bytes = Device.Buffer.as_bytes buf in
  Array.init
    (Bytes.length bytes / 4)
    (fun i -> Int32.float_of_bits (Bytes.get_int32_le bytes (i * 4)))

let f32_buf device data =
  let buf =
    Device.create_buffer ~size:(Array.length data) ~dtype:Dtype.float32 device
  in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf (f32_to_bytes data);
  buf

let shape_node dims =
  match List.map U.const_int dims with [ d ] -> d | ds -> U.stack ds

let f32_buffer_node device_name dims =
  U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.float32
    ~shape:(shape_node dims) ~device:(U.Single device_name) ()

(* Sharding: copy to the device tuple, then take each device's slice of the
   shard axis at an offset along the device range. The frontend shard API
   builds the same graph. *)

let int_ = U.const_int
let emit = function [ d ] -> d | ds -> U.stack ds

let shard_shrink shape ndev src axis =
  let dim = List.nth shape axis in
  let sz = dim / ndev in
  let dnum =
    U.range ~size:(int_ ndev) ~axis:(-1) ~kind:Axis_type.Device ()
  in
  let off = U.alu_binary ~op:Ops.Mul ~lhs:dnum ~rhs:(int_ sz) in
  let before =
    List.mapi (fun i _ -> if i <> axis then int_ 0 else off) shape
  in
  let size =
    List.mapi (fun i s -> if i <> axis then int_ s else int_ sz) shape
  in
  U.shrink ~src ~offset:(emit before) ~size:(emit size)

let sharded x shape devices axis =
  let copied = U.copy ~src:x ~device:(U.Multi devices) () in
  U.unshard ~src:(shard_shrink shape (List.length devices) copied axis) ~axes:[axis] ()

(* Realize a graph through the frontend, with [host] compiling the schedule,
   and return the realized node. *)
let realize ?(host = "CPU") u =
  let t = Tolk_frontend.Tensor.of_uop u in
  Helpers.Context_var.with_context
    [ Helpers.Context_var.B (Helpers.dev, [ Target.of_string host ]) ]
    (fun () -> Tolk_frontend.Run.realize_many [ t ]);
  Tolk_frontend.Tensor.uop t

(* The storage of a realized node, one buffer per device. *)
let device_buffers node =
  match U.Arg.as_param_arg (U.arg (U.buf_uop node)) with
  | Some { buffer = Some bufs; _ } -> bufs
  | _ -> fail "node did not realize to storage"

(* [data] in a fresh buffer on [device], viewed at [dims]. *)
let input device dims data =
  U.reshape ~src:(U.from_buffer (f32_buf (Device.get device) data))
    ~shape:(shape_node dims)

let devs2 = [ "CPU:1"; "CPU:2" ]
let devs4 = [ "CPU:1"; "CPU:2"; "CPU:3"; "CPU:4" ]

(* Shard [data] reshaped to [shape] over the devices, run [op] on the sharded
   value, copy the result back to the host device, and return the realized
   array. [host] names the device holding the input and gathered output; the
   shard devices must share its backend, as one schedule compiles with one
   renderer. *)
let run_sharded ?(host = "CPU") ~devices ~shape ~axis data op =
  let xs = sharded (input host shape data) shape devices axis in
  let out = realize ~host (U.copy ~src:(op xs) ~device:(U.Single host) ()) in
  read_f32 (List.hd (device_buffers out))

let iota n = Array.init n (fun i -> float_of_int (i + 1))

(* The linear schedule of a sum over axis 0 of [shape], sharded on that axis
   over [devices] and gathered to the host device. *)
let schedule_reduction ~devices ~shape =
  let x = f32_buffer_node "CPU" [ List.fold_left ( * ) 1 shape ] in
  let xs = sharded (U.reshape ~src:x ~shape:(shape_node shape)) shape devices 0 in
  let sum = U.reduce_axis ~src:xs ~op:Ops.Add ~axes:[ 0 ] in
  let out = U.contiguous ~src:(U.copy ~src:sum ~device:(U.Single "CPU") ()) () in
  let call, _ = bufferized_call (U.sink [ out ]) in
  fst (Schedule.create_linear_with_vars
         ~get_kernel_graph:Rangeify.get_kernel_graph call)

let forced_strategies =
  Helpers.Context_var.
    [ [ B (Helpers.ring, 2) ];
      [ B (Helpers.all2all, 2) ];
      [ B (Helpers.allreduce_node_ndevs, 2) ];
      [ B (Helpers.allreduce_node_ndevs, 4) ] ]

let local_range size axis = U.range ~size:(int_ size) ~axis ~kind:Axis_type.Local ()
let alu op lhs rhs = Symbolic.simplify (U.alu_binary ~op ~lhs ~rhs)
let rewrite = U.graph_rewrite Tolk.Multi.multi_pm
let fragment shape = U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.float32
    ~shape:(shape_node shape) ~addrspace:Dtype.Reg ()

let () =
  run "Multi_device"
    [
      group "Ownership"
        [
          test "flat sharded params allocate only their local maximum" (fun () ->
              let n = U.variable ~name:"sharded_n" ~min_val:1 ~max_val:5 () in
              let full = alu Ops.Mul n (int_ 2) in
              let p = U.param ~slot:0 ~dtype:Dtype.float32
                  ~shape:(emit [full; int_ 4]) ~axis:0 ~device:(U.Multi devs2) () in
              equal (list int) [10; 4] (U.max_shape p);
              equal (list int) [5; 4] (U.max_shard_shape p);
              let storage = U.storage_base p in
              equal int 0 (Array.length (U.src storage));
              let arg = Option.get (U.Arg.as_param_arg (U.arg storage)) in
              equal (option int) (Some 20) arg.size;
              let lowered = rewrite (U.alu_binary ~op:Ops.Add ~lhs:p ~rhs:p) in
              is_true (U.op lowered = Ops.Unshard);
              is_true (U.equal (Symbolic.simplify (List.hd (U.shape lowered))) full));
          test "axes sort with ranges and close their scope" (fun () ->
              let a = local_range 2 0 and b = local_range 3 1 in
              let value = U.expand ~src:(U.cast ~src:(alu Ops.Add a b) ~dtype:Dtype.float32)
                  ~dims:(shape_node [2; 4]) in
              let u = U.unshard ~src:value ~axes:[1; 0] ~ranges:[b; a] () in
              equal (list int) [0; 1] (List.map fst (U.sharding u));
              is_true (List.map snd (U.sharding u) = [a; b]);
              equal (list int) [4; 12] (U.max_shape u);
              equal int 0 (List.length (U.ranges u));
              raises_match (function Invalid_argument _ -> true | _ -> false)
                (fun () -> ignore (U.axis u)));
          test "reshape divides each axis by its own range count" (fun () ->
              let a = local_range 2 0 and b = local_range 3 1 in
              let u = U.unshard ~src:(fragment [2; 4]) ~axes:[0; 1] ~ranges:[a; b] () in
              let reshaped = rewrite (U.reshape ~src:u ~shape:(shape_node [4; 3; 4])) in
              equal (list int) [4; 3; 4] (U.max_shape reshaped);
              equal (list int) [2; 1; 4] (U.max_shape (U.src reshaped).(0));
              equal (list int) [0; 1] (List.map fst (U.sharding reshaped)));
          test "multi-axis ALU slices whole tiles locally" (fun () ->
              let ranges = [local_range 2 0; local_range 3 1] in
              let u = U.unshard ~src:(fragment [2; 4]) ~axes:[0; 1] ~ranges () in
              let whole = fragment [4; 12] in
              let result = rewrite (U.alu_binary ~op:Ops.Add ~lhs:u ~rhs:whole) in
              equal (list int) [4; 12] (U.max_shape result);
              equal (list int) [2; 4] (U.max_shape (U.src result).(0)));
          test "permutation keeps the owning range with its axis" (fun () ->
              let a = local_range 2 0 and b = local_range 3 1 in
              let u = U.unshard ~src:(fragment [2; 4]) ~axes:[0; 1] ~ranges:[a; b] () in
              let result = rewrite (U.permute ~src:u ~order:[1; 0]) in
              is_true (List.map snd (U.sharding result) = [b; a]);
              equal (list int) [12; 4] (U.max_shape result));
          test "own-shard shrink resolves one axis at a time" (fun () ->
              let a = local_range 2 0 and b = local_range 3 1 in
              let u = U.unshard ~src:(fragment [2; 4]) ~axes:[0; 1] ~ranges:[a; b] () in
              let sliced = rewrite (U.shrink ~src:u ~offset:(emit [alu Ops.Mul a (int_ 2); int_ 0])
                  ~size:(shape_node [2; 12])) in
              equal (list int) [1] (List.map fst (U.sharding sliced));
              equal (list int) [2; 12] (U.max_shape sliced));
          test "thread indices resolve only their owned shard" (fun () ->
              let r = local_range 4 0 and i = local_range 4 1 in
              let value = fragment [4] in
              let u = U.unshard ~src:value ~axes:[0] ~ranges:[r] () in
              List.iter (fun index ->
                  let result = rewrite (U.index ~ptr:u ~idxs:[index] ()) in
                  let view = Option.get (U.as_index result) in
                  is_true (view.ptr == value);
                  is_true (List.for_all2 U.equal [i] view.idxs))
                [alu Ops.Add (alu Ops.Mul r (int_ 4)) i;
                 alu Ops.Add r (alu Ops.Mul i (int_ 4))];
              raises_match (function Invalid_argument _ -> true | _ -> false)
                (fun () -> ignore (rewrite (U.index ~ptr:u ~idxs:[int_ 0] ()))));
          test "unsharded stores select each fragment's destination" (fun () ->
              let r = local_range 2 0 in
              let value = fragment [4] in
              let u = U.unshard ~src:value ~axes:[0] ~ranges:[r] () in
              let destination = fragment [8] in
              let result = rewrite (U.store ~dst:destination ~value:u ()) in
              let store = Option.get (U.as_store result) in
              equal (list int) [4] (U.max_shape store.dst);
              is_true (store.value == value);
              is_true (U.op store.dst = Ops.Shrink));
          test "two-axis device gather preserves every tile" (fun () ->
              let devices = List.init 6 (fun i -> "CPU:" ^ string_of_int (i + 1)) in
              let data = iota 48 in
              let copied = U.copy ~src:(input "CPU" [4; 12] data) ~device:(U.Multi devices) () in
              let r = U.range ~size:(int_ 6) ~axis:(-1) ~kind:Axis_type.Device () in
              let a = alu Ops.Floordiv r (int_ 3) and b = alu Ops.Floormod r (int_ 3) in
              let local = U.shrink ~src:copied
                  ~offset:(emit [alu Ops.Mul a (int_ 2); alu Ops.Mul b (int_ 4)])
                  ~size:(shape_node [2; 4]) in
              let tiled = U.unshard ~src:local ~axes:[0; 1] ~ranges:[a; b] () in
              let result = U.alu_binary ~op:Ops.Add ~lhs:tiled ~rhs:tiled in
              let out = realize (U.copy ~src:result ~device:(U.Single "CPU") ()) in
              equal (array (float 1e-6)) (Array.map (fun x -> x *. 2.) data)
                (read_f32 (List.hd (device_buffers out))));
          test "partial multi-axis allreduce is rejected" (fun () ->
              let ranges = [local_range 2 0; local_range 3 1] in
              let u = U.unshard ~src:(fragment [2; 4]) ~axes:[0; 1] ~ranges () in
              raises_match (function Invalid_argument _ -> true | _ -> false)
                (fun () -> ignore (rewrite (U.reduce_axis ~src:u ~op:Ops.Add ~axes:[0]))));
        ];
      group "Resolution"
        [
          test "mstack joins and mselect indexes seeded shards" (fun () ->
              let data1 = [| 1.; 2.; 3.; 4. |] in
              let data2 = [| 10.; 20.; 30.; 40. |] in
              let a = f32_buffer_node "CPU:1" [ 4 ] in
              let b = f32_buffer_node "CPU:2" [ 4 ] in
              let binding = Realize.Buffers.create () in
              Realize.Buffers.seed binding a
                (f32_buf (Device.get "CPU:1") data1);
              Realize.Buffers.seed binding b
                (f32_buf (Device.get "CPU:2") data2);
              let ctx = Realize.exec_context () in
              let ms = U.mstack [ a; b ] in
              (match Realize.resolve_buffer binding ctx ms with
              | Realize.Multi m ->
                  equal ~msg:"mstack shard devices" (list string)
                    [ "CPU:1"; "CPU:2" ]
                    (List.map Device.Buffer.device
                       (Device.Multi_buffer.bufs m))
              | Realize.Single _ -> fail "MSTACK resolved to a single buffer");
              let second =
                Realize.resolve binding ctx (U.mselect ~src:ms ~index:1)
              in
              equal ~msg:"mselect shard contents" (array (float 1e-6)) data2
                (read_f32 second);
              raises_match
                (function Invalid_argument _ -> true | _ -> false)
                (fun () ->
                  ignore
                    (Realize.resolve binding ctx (U.mselect ~src:a ~index:0))));
          test "multi buffer node allocates one shard per device" (fun () ->
              let node =
                U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.float32
                  ~shape:(shape_node [ 4 ]) ~device:(U.Multi devs2) ()
              in
              let binding = Realize.Buffers.create () in
              let ctx = Realize.exec_context () in
              match Realize.resolve_buffer binding ctx node with
              | Realize.Multi m ->
                  equal ~msg:"shard devices" (list string) devs2
                    (List.map Device.Buffer.device
                       (Device.Multi_buffer.bufs m));
                  (* byte view of a multi buffer views every shard. *)
                  let sliced =
                    storage_view ~src:node ~offset:(U.const_int 1) ~size:2
                      ~dtype:Dtype.float32
                  in
                  (match Realize.resolve_buffer binding ctx sliced with
                  | Realize.Multi v ->
                      List.iter2
                        (fun view base ->
                          is_true ~msg:"view shares its shard's base"
                            (Device.Buffer.base_id view
                            = Device.Buffer.base_id base);
                          equal ~msg:"view size" int 2
                            (Device.Buffer.size view);
                          equal ~msg:"view offset" int 4
                            (Device.Buffer.offset view))
                        (Device.Multi_buffer.bufs v)
                        (Device.Multi_buffer.bufs m)
                  | Realize.Single _ ->
                      fail "byte view of a multi buffer resolved to single")
              | Realize.Single _ ->
                  fail "multi-device BUFFER resolved to a single buffer");
        ];
      group "Execution"
        [
          test "shard and gather round-trips" (fun () ->
              let data = iota 8 in
              let got =
                run_sharded ~devices:devs2 ~shape:[ 8 ] ~axis:0 data Fun.id
              in
              equal (array (float 1e-6)) data got);
          test "copy to device tuple replicates" (fun () ->
              let data = iota 8 in
              let bufs =
                device_buffers
                  (realize
                     (U.copy ~src:(input "CPU" [ 8 ] data)
                        ~device:(U.Multi devs2) ()))
              in
              equal ~msg:"shard devices" (list string) devs2
                (List.map Device.Buffer.device bufs);
              List.iter
                (fun buf ->
                  equal ~msg:"replicated shard contents"
                    (array (float 1e-6)) data (read_f32 buf))
                bufs);
          test "elementwise on sharded tensors" (fun () ->
              let data = iota 8 in
              let got =
                run_sharded ~devices:devs2 ~shape:[ 8 ] ~axis:0 data (fun xs ->
                    U.alu_binary ~op:Ops.Add ~lhs:xs ~rhs:xs)
              in
              equal (array (float 1e-6))
                (Array.map (fun v -> v +. v) data)
                got);
          test "elementwise with a broadcast operand" (fun () ->
              let data = iota 8 in
              let ones =
                U.expand
                  ~src:(U.const (Const.float Dtype.float32 1.0))
                  ~dims:(shape_node [ 8 ])
              in
              let got =
                run_sharded ~devices:devs2 ~shape:[ 8 ] ~axis:0 data (fun xs ->
                    U.alu_binary ~op:Ops.Add ~lhs:xs ~rhs:ones)
              in
              equal (array (float 1e-6)) (Array.map (fun v -> v +. 1.) data)
                got);
          test "reduce over the sharded axis allreduces on 2 devices"
            (fun () ->
              let data = iota 8 in
              let got =
                run_sharded ~devices:devs2 ~shape:[ 8 ] ~axis:0 data (fun xs ->
                    U.reduce_axis ~src:xs ~op:Ops.Add ~axes:[ 0 ])
              in
              equal (array (float 1e-6)) [| 36. |] got);
          test "reduce over the sharded axis allreduces on 4 devices"
            (fun () ->
              let data = iota 8 in
              let got =
                run_sharded ~devices:devs4 ~shape:[ 8 ] ~axis:0 data (fun xs ->
                    U.reduce_axis ~src:xs ~op:Ops.Add ~axes:[ 0 ])
              in
              equal (array (float 1e-6)) [| 36. |] got);
          test "reduce over a non-sharded axis stays sharded" (fun () ->
              let data = iota 8 in
              let got =
                run_sharded ~devices:devs2 ~shape:[ 2; 4 ] ~axis:0 data
                  (fun xs -> U.reduce_axis ~src:xs ~op:Ops.Add ~axes:[ 1 ])
              in
              (* Row sums of [[1..4]; [5..8]]. *)
              equal (array (float 1e-6)) [| 10.; 26. |] got);
        ];
      group "Collectives"
        [
          test "forced ring handles aligned empty chunks on four devices" (fun () ->
              Helpers.Context_var.with_context [Helpers.Context_var.B (Helpers.ring, 2)] (fun () ->
                  let data = iota (32 * 64) in
                  let got = run_sharded ~devices:devs4 ~shape:[32; 64] ~axis:0 data
                      (fun x -> U.reduce_axis ~src:x ~op:Ops.Add ~axes:[0]) in
                  let expected = Array.init 64 (fun i ->
                      let sum = ref 0. in
                      for row = 0 to 31 do sum := !sum +. data.(row * 64 + i) done;
                      !sum) in
                  equal (array (float 1e-6)) expected got));
          test "forced strategies reduce uneven chunks on four devices" (fun () ->
              List.iter (fun bindings ->
                  Helpers.Context_var.with_context bindings (fun () ->
                      let data = iota 28 in
                      let got = run_sharded ~devices:devs4 ~shape:[4; 7] ~axis:0 data
                          (fun x -> U.reduce_axis ~src:x ~op:Ops.Add ~axes:[0]) in
                      let expected = Array.init 7 (fun i ->
                          data.(i) +. data.(7+i) +. data.(14+i) +. data.(21+i)) in
                      equal (array (float 1e-6)) expected got))
                forced_strategies);
          test "each forced strategy schedules its own collective" (fun () ->
              let schedule bindings =
                Helpers.Context_var.with_context bindings (fun () ->
                    let calls = Array.to_list (U.src (schedule_reduction ~devices:devs4 ~shape:[4; 7])) in
                    String.concat "," (List.map (fun call ->
                        U.semantic_key (Option.get (U.as_call call)).body) calls)) in
              let keys = List.map schedule forced_strategies in
              equal ~msg:"same bindings, same schedule" string (List.hd keys)
                (schedule (List.hd forced_strategies));
              equal ~msg:"distinct schedules" int (List.length keys)
                (List.length (List.sort_uniq String.compare keys)));
          test "hierarchical scalar handles empty chunks" (fun () ->
              Helpers.Context_var.with_context [Helpers.Context_var.B (Helpers.allreduce_node_ndevs, 4)] (fun () ->
                  let got = run_sharded ~devices:devs4 ~shape:[4] ~axis:0 (iota 4)
                      (fun x -> U.reduce_axis ~src:x ~op:Ops.Add ~axes:[0]) in
                  equal (array (float 1e-6)) [|10.|] got));
          test "hierarchical maximum handles negative values" (fun () ->
              Helpers.Context_var.with_context [Helpers.Context_var.B (Helpers.allreduce_node_ndevs, 2)] (fun () ->
                  let data = Array.map (~-.) (iota 28) in
                  let got = run_sharded ~devices:devs4 ~shape:[4; 7] ~axis:0 data
                      (fun x -> U.reduce_axis ~src:x ~op:Ops.Max ~axes:[0]) in
                  equal (array (float 1e-6)) (Array.sub data 0 7) got));
          test "symbolic allreduce retains logical sizes under forced ring" (fun () ->
              Helpers.Context_var.with_context [Helpers.Context_var.B (Helpers.ring, 2)] (fun () ->
                  let v = U.variable ~name:"collective_size" ~min_val:1 ~max_val:7 () in
                  let src = U.param ~slot:0 ~dtype:Dtype.float32 ~shape:v ~device:(U.Multi devs4) () in
                  let result = Option.get (Allreduce.handle_allreduce src ~op:Ops.Add ~device:(U.Multi devs4)) in
                  is_true (List.for_all2 U.equal [v] (U.shape result));
                  let call = Option.get (Allreduce.create_allreduce_function src ~op:Ops.Add ~device:(U.Multi devs4) ()) in
                  is_true (List.for_all2 U.equal [v] (U.shape call));
                  equal (list int) [7] (U.max_shape call);
                  List.iter (fun n ->
                      let data = iota 28 in
                      let size = U.bind ~var:v ~value:(int_ n) in
                      let got = run_sharded ~devices:devs4 ~shape:[4; 7] ~axis:0 data (fun x ->
                          let sliced = U.shrink ~src:x ~offset:(shape_node [0; 0]) ~size:(emit [int_ 4; size]) in
                          U.reduce_axis ~src:sliced ~op:Ops.Add ~axes:[0]) in
                      let expected = Array.init n (fun i ->
                          data.(i) +. data.(7+i) +. data.(14+i) +. data.(21+i)) in
                      equal (array (float 1e-6)) expected (Array.sub got 0 n)) [2; 6]));
        ];
      group "Cuda"
        [
          test "duplicated device tuple runs on one CUDA device" (fun () ->
              if not (Lazy.force cuda_available) then
                skip ~reason:"no CUDA device" ();
              let data = iota 8 in
              let got =
                run_sharded ~host:"CUDA" ~devices:[ "CUDA:0"; "CUDA:0" ]
                  ~shape:[ 8 ] ~axis:0 data (fun xs ->
                    U.reduce_axis ~src:xs ~op:Ops.Add ~axes:[ 0 ])
              in
              equal (array (float 1e-6)) [| 36. |] got);
        ];
    ]
