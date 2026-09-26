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
  let copied = U.copy ~src:x ~device:(U.Multi (List.map Option.some devices)) () in
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

(* [data] of [shape] split along [axis] over [devices], in storage of its own:
   what an indexed write lands in. *)
let split_storage devices shape axis data =
  let n = List.length devices in
  let local = List.mapi (fun i d -> if i = axis then d / n else d) shape in
  let dst =
    U.unshard
      ~src:
        (Tolk_frontend.Tensor.uop
           (Tolk_frontend.Creation.empty ~dtype:Dtype.float32
              ~device:(U.Multi (List.map Option.some devices)) local))
      ~axes:[ axis ] ()
  in
  let value = sharded (input "CPU" shape data) shape devices axis in
  Tolk_frontend.Tensor.of_uop
    (U.after ~src:dst ~deps:[ U.store ~dst ~value () ])

let ints shape values =
  let n = Array.length values in
  let buf = Device.create_buffer ~size:n ~dtype:Dtype.int32 (Device.get "CPU") in
  Device.Buffer.ensure_allocated buf;
  let bytes = Bytes.create (n * 4) in
  Array.iteri (fun i v -> Bytes.set_int32_le bytes (i * 4) (Int32.of_int v)) values;
  Device.Buffer.copyin buf bytes;
  U.reshape ~src:(U.from_buffer buf) ~shape:(shape_node shape)

let broadcast u shape =
  Tolk_frontend.Tensor.uop
    (Tolk_frontend.Movement.expand (Tolk_frontend.Tensor.of_uop u) shape)

(* [t] gathered to the host device and read. *)
let gathered t =
  let out =
    realize (U.copy ~src:(Tolk_frontend.Tensor.uop t) ~device:(U.Single "CPU") ())
  in
  read_f32 (List.hd (device_buffers out))

(* Rows [rows] of the [r] by [c] matrix [data] replaced by [values]. *)
let with_rows ~c data rows values =
  let out = Array.copy data in
  List.iteri
    (fun i r -> Array.blit values (i * c) out (r * c) c)
    rows;
  out

let bytes_node shape bytes =
  let buf =
    Device.create_buffer ~size:(Bytes.length bytes) ~dtype:Dtype.uint8
      (Device.get "CPU")
  in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf bytes;
  U.reshape ~src:(U.from_buffer buf) ~shape:(shape_node shape)

(* [u], on the host device, split along [axis] over [devs4], or a copy on each
   of them. *)
let spread ?axis shape u =
  Tolk_frontend.Tensor.of_uop
    (match axis with
    | Some axis -> sharded u shape devs4 axis
    | None -> U.copy ~src:u ~device:(U.Multi (List.map Option.some devs4)) ())

let wave n = Array.init n (fun i -> Float.of_int ((i * 7 mod 11) - 5) /. 8.)

(* Each block of the [nb; m; k] [xs] times the [k; n] matrix of [ws] its id
   names, or zeros. *)
let block_reference ~m ~n ~k xs ws ids =
  Array.init (Array.length ids * m * n) (fun o ->
      let b = o / (m * n) and r = o / n mod m and c = o mod n in
      if ids.(b) < 0 then 0.
      else
        let acc = ref 0. in
        for j = 0 to k - 1 do
          acc := !acc +. (xs.((((b * m) + r) * k) + j) *. ws.((((ids.(b) * k) + j) * n) + c))
        done;
        !acc)

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
                  ~shape:(emit [full; int_ 4]) ~axis:0 ~device:(U.Multi (List.map Option.some devs2)) () in
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
          test "an operand of lower rank keeps its axis where it broadcasts"
            (fun () ->
              let w =
                U.unshard ~src:(fragment [ 8; 2 ]) ~axes:[ 1 ]
                  ~ranges:[ local_range 4 0 ] ()
              in
              let x = fragment [ 3; 1; 8 ] in
              let product = U.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:w in
              equal (list int) [ 3; 8; 8 ] (U.max_shape product);
              equal (option int) (Some 2) (U.axis product));
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
              let copied = U.copy ~src:(input "CPU" [4; 12] data) ~device:(U.Multi (List.map Option.some devices)) () in
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
          test "mstack joins and mselect indexes owned shards" (fun () ->
              let data1 = [| 1.; 2.; 3.; 4. |] in
              let data2 = [| 10.; 20.; 30.; 40. |] in
              let a = U.from_buffer (f32_buf (Device.get "CPU:1") data1) in
              let b = U.from_buffer (f32_buf (Device.get "CPU:2") data2) in
              let ctx = Realize.exec_context () in
              let ms = U.mstack [ a; b ] in
              (match Realize.resolve_buffer ctx ms with
              | Realize.Multi m ->
                  equal ~msg:"mstack shard devices" (list string)
                    [ "CPU:1"; "CPU:2" ]
                    (List.map Device.Buffer.device
                       (Device.Multi_buffer.bufs m))
              | Realize.Single _ -> fail "MSTACK resolved to a single buffer");
              let second =
                Realize.resolve ctx (U.mselect ~src:ms ~index:1)
              in
              equal ~msg:"mselect shard contents" (array (float 1e-6)) data2
                (read_f32 second);
              raises_match
                (function Invalid_argument _ -> true | _ -> false)
                (fun () ->
                  ignore
                    (Realize.resolve ctx (U.mselect ~src:a ~index:0))));
          test "multi buffer node allocates one shard per device" (fun () ->
              let node =
                U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.float32
                  ~shape:(shape_node [ 4 ]) ~device:(U.Multi (List.map Option.some devs2)) ()
              in
              let ctx = Realize.exec_context () in
              match Realize.resolve_buffer ctx node with
              | Realize.Multi m ->
                  equal ~msg:"shard devices" (list string) devs2
                    (List.map Device.Buffer.device
                       (Device.Multi_buffer.bufs m));
                  (* byte view of a multi buffer views every shard. *)
                  let sliced =
                    storage_view ~src:node ~offset:(U.const_int 1) ~size:2
                      ~dtype:Dtype.float32
                  in
                  (match Realize.resolve_buffer ctx sliced with
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
                        ~device:(U.Multi (List.map Option.some devs2)) ()))
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
      group "Kernels over split storage"
        [
          test "an indexed write lands in each device's rows" (fun () ->
              let data = iota 24 and values = Array.map (fun v -> 100. +. v) (iota 9) in
              let rows = [ 6; 1; 3 ] in
              let t = split_storage devs4 [ 8; 3 ] 0 data in
              let index = broadcast (ints [ 3; 1 ] (Array.of_list rows)) [ 3; 3 ] in
              let on_each u = U.copy ~src:u ~device:(U.Multi (List.map Option.some devs4)) () in
              let written =
                Tolk_frontend.Op.scatter_indexed t ~dim:0
                  (Tolk_frontend.Tensor.of_uop (on_each index))
                  (Tolk_frontend.Tensor.of_uop (on_each (input "CPU" [ 3; 3 ] values)))
                  ~mode:`Set ~unique:true
              in
              equal (array float_exact) (with_rows ~c:3 data rows values)
                (gathered written));
          test "an indexed write off the split axis writes each device's lanes"
            (fun () ->
              let data = iota 32 and values = Array.map (fun v -> 100. +. v) (iota 16) in
              let rows = [ 3; 0 ] in
              let t = split_storage devs4 [ 4; 8 ] 1 data in
              let index =
                sharded (broadcast (ints [ 2; 1 ] (Array.of_list rows)) [ 2; 8 ])
                  [ 2; 8 ] devs4 1
              in
              let src = sharded (input "CPU" [ 2; 8 ] values) [ 2; 8 ] devs4 1 in
              let written =
                Tolk_frontend.Op.scatter_indexed t ~dim:0
                  (Tolk_frontend.Tensor.of_uop index)
                  (Tolk_frontend.Tensor.of_uop src) ~mode:`Set ~unique:true
              in
              equal (array float_exact) (with_rows ~c:8 data rows values)
                (gathered written));
          test "split blocks over split experts raise" (fun () ->
              (* A block's expert may live on another device, where its own
                 device cannot read it. *)
              let nb, m, k, e, n = (8, 2, 16, 8, 4) in
              raises_match
                (function Invalid_argument _ -> true | _ -> false)
                (fun () ->
                  ignore
                    (Tolk_frontend.Op.block_matmul
                       (spread ~axis:0 [ nb; m; k ]
                          (input "CPU" [ nb; m; k ] (wave (nb * m * k))))
                       (spread ~axis:0 [ e; k; n ]
                          (input "CPU" [ e; k; n ] (wave (e * k * n))))
                       ~ids:(spread ~axis:0 [ nb ] (ints [ nb ] (Array.make nb 0))))));
          test "each device multiplies its blocks, or its columns" (fun () ->
              let nb, m, k, e, n = (8, 2, 16, 3, 8) in
              let xs = wave (nb * m * k) and ws = wave (e * k * n) in
              let ids = [| 2; 0; 1; -1; 1; 2; 0; 0 |] in
              let expected = block_reference ~m ~n ~k xs ws ids in
              let x = input "CPU" [ nb; m; k ] xs and w = input "CPU" [ e; k; n ] ws in
              let ids = ints [ nb ] ids in
              let blocks =
                Tolk_frontend.Op.block_matmul
                  (spread ~axis:0 [ nb; m; k ] x) (spread [ e; k; n ] w)
                  ~ids:(spread ~axis:0 [ nb ] ids)
              in
              equal ~msg:"blocks" (array (float 1e-5)) expected (gathered blocks);
              let columns =
                Tolk_frontend.Op.block_matmul (spread [ nb; m; k ] x)
                  (spread ~axis:2 [ e; k; n ] w) ~ids:(spread [ nb ] ids)
              in
              equal ~msg:"columns" (array (float 1e-5)) expected (gathered columns));
          test "whole blocks sum each device's partial product" (fun () ->
              let nb, m, k, e, n = (8, 2, 16, 8, 4) in
              let xs = wave (nb * m * k) and ws = wave (e * k * n) in
              let ids = [| 1; 0; 3; -1; 4; 5; 7; 2 |] in
              let expected = block_reference ~m ~n ~k xs ws ids in
              let x = input "CPU" [ nb; m; k ] xs and w = input "CPU" [ e; k; n ] ws in
              let ids = spread [ nb ] (ints [ nb ] ids) in
              let matrices =
                Tolk_frontend.Op.block_matmul (spread [ nb; m; k ] x)
                  (spread ~axis:0 [ e; k; n ] w) ~ids
              in
              equal ~msg:"split matrices" (array (float 1e-5)) expected
                (gathered matrices);
              let inputs =
                Tolk_frontend.Op.block_matmul (spread ~axis:2 [ nb; m; k ] x)
                  (spread ~axis:1 [ e; k; n ] w) ~ids
              in
              equal ~msg:"split inputs" (array (float 1e-5)) expected
                (gathered inputs));
          test "each device decodes and multiplies its own matrices" (fun () ->
              let i, m, k, e, n = (8, 1, 32, 8, 4) in
              let xs = wave (i * m * k) in
              let codes =
                Bytes.init (e * n * k / 2) (fun b -> Char.chr ((b * 37) land 255))
              and scales = Bytes.init (e * n * k / 32) (fun b -> Char.chr (126 + (b mod 3))) in
              let ids = [| 1; 0; 3; -1; 4; 5; 7; 7 |] in
              let value c =
                let v = [| 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. |].(c land 7) in
                if c land 8 = 0 then v else -.v
              in
              let weight id o j =
                let byte = Char.code (Bytes.get codes ((((id * n) + o) * k / 2) + (j / 2))) in
                let code = if j land 1 = 0 then byte land 15 else byte lsr 4 in
                let s = Char.code (Bytes.get scales ((((id * n) + o) * k / 32) + (j / 32))) in
                value code *. Float.ldexp 1. (s - 127)
              in
              let expected =
                Array.init (i * m * n) (fun q ->
                    let t = q / (m * n) and r = q / n mod m and o = q mod n in
                    if ids.(t) < 0 then 0.
                    else
                      let acc = ref 0. in
                      for j = 0 to k - 1 do
                        acc := !acc +. (xs.((((t * m) + r) * k) + j) *. weight ids.(t) o j)
                      done;
                      !acc)
              in
              let product ?axis () =
                Tolk_frontend.Op.quant_matmul
                  ~ids:(spread ?axis [ i ] (ints [ i ] ids))
                  (spread ?axis [ i; m; k ] (input "CPU" [ i; m; k ] xs))
                  ~codes:(spread ~axis:0 [ e; n; k / 2 ] (bytes_node [ e; n; k / 2 ] codes))
                  ~scales:(spread ~axis:0 [ e; n; k / 32 ] (bytes_node [ e; n; k / 32 ] scales))
              in
              equal ~msg:"whole instances" (array (float 1e-4)) expected
                (gathered (product ()));
              raises_match ~msg:"split instances"
                (function Invalid_argument _ -> true | _ -> false)
                (fun () -> ignore (product ~axis:0 ())));
          test "a narrow index into a split axis past its range" (fun () ->
              (* A uint8 row into 512 rows over two devices: the second
                 device's first row, 256, does not fit the index's type. *)
              let devs = devs2 in
              let t = split_storage devs [ 512; 1 ] 0 (Array.make 512 0.0) in
              let on_each u = U.copy ~src:u ~device:(U.Multi (List.map Option.some devs)) () in
              let row =
                let buf =
                  Device.create_buffer ~size:1 ~dtype:Dtype.uint8
                    (Device.get "CPU")
                in
                Device.Buffer.ensure_allocated buf;
                Device.Buffer.copyin buf (Bytes.make 1 (Char.chr 5));
                U.reshape ~src:(U.from_buffer buf) ~shape:(shape_node [ 1; 1 ])
              in
              let written =
                Tolk_frontend.Op.scatter_indexed t ~dim:0
                  (Tolk_frontend.Tensor.of_uop (on_each row))
                  (Tolk_frontend.Tensor.of_uop
                     (on_each (input "CPU" [ 1; 1 ] [| 100.0 |])))
                  ~mode:`Set ~unique:true
              in
              let expected = Array.make 512 0.0 in
              expected.(5) <- 100.0;
              equal (array float_exact) expected (gathered written));
          test "updates split unlike their destination raise" (fun () ->
              let t = split_storage devs4 [ 4; 8 ] 1 (iota 32) in
              let index = broadcast (ints [ 2; 1 ] [| 3; 0 |]) [ 2; 8 ] in
              raises_match
                (function Invalid_argument _ -> true | _ -> false)
                (fun () ->
                  ignore
                    (Tolk_frontend.Op.scatter_indexed t ~dim:0
                       (Tolk_frontend.Tensor.of_uop index)
                       (Tolk_frontend.Tensor.of_uop (input "CPU" [ 2; 8 ] (iota 16)))
                       ~mode:`Set ~unique:true)));
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
                  let src = U.param ~slot:0 ~dtype:Dtype.float32 ~shape:v ~device:(U.Multi (List.map Option.some devs4)) () in
                  let result = Option.get (Allreduce.handle_allreduce src ~op:Ops.Add ~device:(U.Multi (List.map Option.some devs4))) in
                  is_true (List.for_all2 U.equal [v] (U.shape result));
                  let call = Option.get (Allreduce.create_allreduce_function src ~op:Ops.Add ~device:(U.Multi (List.map Option.some devs4))) in
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
