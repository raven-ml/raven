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

   The CPU peers use a synchronous shared copy queue. Its compiled submission
   copies the bytes and increments an owned traffic counter, so replay counts
   executed transfers. Ordinary kernels and the CPU host use the CPU runtime.
   The test runs alone because it replaces the CPU opener. *)

open Windtrap
open Rune_test_support.Support

(* Counting devices *)

module U = Tolk_uop.Uop
module D = Tolk_uop.Dtype
module B = Tolk.Device.Buffer

let transfers : (string * string, B.t) Hashtbl.t = Hashtbl.create 8

let zero_buffer name size =
  let buffer =
    B.create ~device:name ~size ~dtype:D.uint64 (Tolk.Device.allocator (Tolk.Device.get name))
  in
  B.ensure_allocated buffer;
  B.copyin buffer (Bytes.make (8 * size) '\000');
  buffer

let counter key =
  match Hashtbl.find_opt transfers key with
  | Some buffer -> buffer
  | None ->
      let buffer = zero_buffer "CPU" 1 in
      Hashtbl.add transfers key buffer;
      buffer

let create name =
  let cpu = Tolk_cpu.create ~aligned:false name in
  if name = "CPU" then cpu
  else
    let renderer = Tolk.Device.renderer cpu in
    let renderer_set =
      Tolk.Device.Renderer_set.make ~device:name
        ~arch:(Tolk.Renderer.target renderer).Tolk_uop.Target.arch
        [ ("CLANG", fun _ -> renderer) ]
    in
    let timeline = zero_buffer name 2 in
    let uint n = U.const (Tolk_uop.Const.int D.uint64 n) in
    let index ptr = U.index ~ptr ~idxs:[ U.const_int 0 ] () in
    let device node =
      match U.device_of node with
      | Some (U.Single name) -> name
      | _ -> fail "copy queue expected a single-device argument"
    in
    let encode node =
      match (U.op node, U.arg node, U.children node) with
      | ( Tolk_uop.Ops.Custom_function,
          U.Arg.String ("submit_cpu_copy_0" | "submit_cpu_compute_0"),
          [ linear; dependency ] ) ->
          let previous = ref [ dependency ] in
          let nodes =
            List.map
              (fun instruction ->
                let next =
                  match (U.as_call instruction, U.arg instruction) with
                  | Some { body; args = [ dst; src ] }, _
                    when U.op body = Tolk_uop.Ops.Store ->
                      let bytes =
                        uint (U.max_numel src * D.itemsize (U.dtype src))
                      in
                      let copy =
                        Tolk.Hcq2.ccall ~after:!previous ~name:"memcpy"
                          ~dtype:D.uint64
                          [
                            U.getaddr ~device:name ~src:dst ();
                            U.getaddr ~device:name ~src ();
                            bytes;
                          ]
                      in
                      let count =
                        U.placeholder ~shape:[ 1 ] ~dtype:D.uint64 ~slot:0
                          ~device:(U.Single name) ~volatile:true
                          ~allocation:
                            ( "test_transfer",
                              Marshal.to_string (device src, device dst) [] )
                          ()
                      in
                      let ptr = index (U.after ~src:count ~deps:[ copy ]) in
                      U.store ~dst:ptr
                        ~value:
                          (U.alu_binary ~op:Tolk_uop.Ops.Add
                             ~lhs:(U.load ~src:ptr ()) ~rhs:bytes)
                        ()
                  | _, U.Arg.Typed ("store", _) ->
                      U.store
                        ~dst:
                          (index
                             (U.after
                                ~src:(U.src instruction).(0)
                                ~deps:!previous))
                        ~value:(U.src instruction).(1)
                        ()
                  | _, U.Arg.Typed (("wait" | "barrier"), _) -> U.noop ()
                  | _ -> fail "unexpected synchronous copy queue instruction"
                in
                if U.op next <> Tolk_uop.Ops.Noop then previous := [ next ];
                next)
              (U.children linear)
          in
          Some (U.group nodes)
      | _ -> None
    in
    let queue =
      Tolk.Device.
        {
          timestamp_divider = 1.;
          profile_offset = (fun () -> 0.);
          completion = (fun () -> Fun.const ());
          prepare = (fun () -> ());
          host = "CPU";
          max_kernel_bindings = Some 0;
          copy = (fun _ -> Some "COPY:0");
          encode;
          lower = (fun _ -> None);
          compile =
            (fun sink ->
              let host = Tolk.Device.get "CPU" in
              Tolk.Codegen.to_program ~optimize:false (Tolk.Device.renderer host)
                sink);
          config = (fun () -> "COUNTED_HOST_COPY=1");
        }
    in
    let bufferize node =
      match (U.node_tag node, U.as_param node) with
      | Some "timeline", _ -> Some timeline
      | _, Some { param = { allocation = Some ("test_transfer", key); _ }; _ }
        ->
          Some (counter (Marshal.from_string key 0))
      | _ -> Tolk.Device.bufferize cpu node
    in
    (* Synchronous host queues cannot wait for another queue in the same
       submission. Separate peer groups let HCQ preserve those dependencies
       between submissions instead. *)
    Tolk.Device.make ~name ~peer_group:name ~allocator:(Tolk.Device.allocator cpu)
      ~renderer_set ~runtime:(Tolk.Device.runtime cpu)
      ~synchronize:(fun timeout -> Tolk.Device.synchronize ?timeout cpu)
      ~queue ~bufferize ()

let () = Tolk.Device.register "CPU" create

(* Bytes a call moves between two of rune's devices. *)
let peer_bytes f =
  let read buffer = Int64.to_int (Bytes.get_int64_le (B.as_bytes buffer) 0) in
  let before = Hashtbl.create (Hashtbl.length transfers) in
  Hashtbl.iter
    (fun key buffer -> Hashtbl.add before key (read buffer))
    transfers;
  let y = f () in
  let moved =
    Hashtbl.fold
      (fun ((src, dst) as key) buffer sum ->
        let n =
          read buffer - Option.value (Hashtbl.find_opt before key) ~default:0
        in
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

let test_rune_shared_reduce_replay () =
  let reduce = Rune.jit' (Nx.sum ~axes:[ 0 ]) in
  List.iter
    (fun replay ->
      let values =
        Array.init 64 (fun i ->
            float_of_int ((1000 * replay) + (100 * (i / 16)) + (i mod 16)))
      in
      let source =
        Nx.place
          (Nx.Placement.sharded ~axis:0 cpus)
          (Nx.create f32 [| 4; 16 |] values)
      in
      let reduced = reduce source in
      let expected =
        Array.init 16 (fun i -> float_of_int ((4000 * replay) + 600 + (4 * i)))
      in
      List.iteri
        (fun lane device ->
          equal
            ~msg:(Printf.sprintf "replay %d, lane %d" replay lane)
            (array float_exact) expected
            (to_arr (Nx.place (Nx.Placement.device device) reduced)))
        cpus)
    [ 0; 1; 2 ]

let tests =
  [
    test "Rune shared copy replay reduces every shard"
      test_rune_shared_reduce_replay;
    group "tensor parallelism"
      [ test "a column-then-row MLP" test_tensor_parallel_mlp ];
    group "expert parallelism"
      [
        test "each device multiplies its experts' routes"
          test_expert_parallel_product;
      ];
  ]

let () = exit (run "rune tensor parallelism" tests)
