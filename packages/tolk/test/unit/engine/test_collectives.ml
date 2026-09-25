(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Collectives on CPU:1..CPU:k: what each device holds and what crosses between
   devices. The host device CPU holds inputs and gathered results. The CPU
   opener is replaced by one whose allocator counts live bytes per device and
   records every buffer transfer, so the tests read memory and traffic off the
   run.

   Transfer counts follow from the collective alone, so the traffic tests pin
   today's lowering and name the target beside it; a lowering that moves fewer
   bytes updates them. Peaks also move with scheduling and memory planning, so
   the fully sharded step asserts its target bound instead, marked as an
   expected failure until the lowering meets it. *)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop
module T = Tolk_frontend.Tensor
module C = Tolk_frontend.Creation
module El = Tolk_frontend.Elementwise
module Mv = Tolk_frontend.Movement
module Op = Tolk_frontend.Op
module Rd = Tolk_frontend.Reduce
module Run = Tolk_frontend.Run

(* Counting devices *)

type usage = { mutable live : int; mutable peak : int }

let usages : (string, usage) Hashtbl.t = Hashtbl.create 8
let transfers : (string * string, int) Hashtbl.t = Hashtbl.create 8

let usage device =
  match Hashtbl.find_opt usages device with
  | Some u -> u
  | None ->
      let u = { live = 0; peak = 0 } in
      Hashtbl.replace usages device u;
      u

let count device n =
  let u = usage device in
  u.live <- u.live + n;
  u.peak <- max u.peak u.live

(* The counts sit above the LRU cache: a buffer returned to the cache no longer
   counts as live. The cache is flushed and the allocation retried when an
   allocation fails, so live bytes are what a device needs. *)
let counting device (Device.Allocator.Pack a) =
  let transfer ~dest ~src ~dest_device ~src_device n =
    let key = (src_device, dest_device) in
    let sum = Option.value (Hashtbl.find_opt transfers key) ~default:0 in
    Hashtbl.replace transfers key (sum + n);
    let bytes = Bytes.create n in
    a.copyout bytes src;
    a.copyin dest bytes;
    true
  in
  Device.Allocator.Pack
    {
      a with
      alloc =
        (fun n spec ->
          let buf = a.alloc n spec in
          count device n;
          buf);
      free =
        (fun buf n spec ->
          count device (-n);
          a.free buf n spec);
      transfer = Some transfer;
      supports_transfer = true;
    }

let create name =
  let cpu = Tolk_cpu.create name in
  let renderer = Device.renderer cpu in
  let renderer_set =
    Device.Renderer_set.make ~device:name
      ~arch:(Renderer.target renderer).Target.arch
      [ ("CLANG", fun _ -> renderer) ]
  in
  let allocator =
    counting name
      (Device.Allocator.Pack
         (Device.Lru_allocator.wrap
            (Storage.Host_allocator.make ~synchronize:ignore)))
  in
  Device.make ~name ~allocator ~renderer_set ~runtime:(Device.runtime cpu)
    ~synchronize:(fun () -> Device.synchronize cpu)
    ~bufferize:(Device.bufferize cpu) ()

let () = Device.register "CPU" create

(* Measurements *)

(* Buffers are freed by GC finalisers; the second cycle collects what the first
   cycle's finalisers released. *)
let settle () =
  Gc.full_major ();
  Gc.full_major ()

(* Bytes moved from one device to another, by (source, destination). *)
type flows = ((string * string) * int) list

(* [traffic f] is [f ()] with the transfers it made. *)
let traffic f : _ * flows =
  let before = Hashtbl.copy transfers in
  let result = f () in
  let moved key n = n - Option.value (Hashtbl.find_opt before key) ~default:0 in
  ( result,
    Hashtbl.fold
      (fun key n flows ->
        if moved key n = 0 then flows else (key, moved key n) :: flows)
      transfers [] )

let peer_bytes select (flows : flows) =
  List.fold_left
    (fun sum ((src, dst), n) ->
      if src <> dst && src <> "CPU" && dst <> "CPU" && select ~src ~dst then
        sum + n
      else sum)
    0 flows

let peer = peer_bytes (fun ~src:_ ~dst:_ -> true)
let received device = peer_bytes (fun ~src:_ ~dst -> dst = device)
let sent device = peer_bytes (fun ~src ~dst:_ -> src = device)

(* [peak_over devices f] is [f ()] with, per device, the highest live bytes
   while it ran minus the live bytes when it started. [f] settles between
   realizes itself when it spans several. *)
let peak_over devices f =
  settle ();
  let entry =
    List.map
      (fun d ->
        let u = usage d in
        let outer = u.peak in
        u.peak <- u.live;
        (d, u.live, outer))
      devices
  in
  let result = f () in
  ( result,
    List.map
      (fun (d, live, outer) ->
        let u = usage d in
        let peak = u.peak in
        u.peak <- max outer peak;
        (d, peak - live))
      entry )

(* [held devices f] is [f ()] with, per device, the live bytes it left
   allocated. *)
let held devices f =
  settle ();
  let before = List.map (fun d -> (d, (usage d).live)) devices in
  let result = f () in
  settle ();
  (result, List.map (fun (d, live) -> (d, (usage d).live - live)) before)

(* Each device's bytes of a realized tensor. *)
let device_bytes t =
  Run.realize_many [ t ];
  match U.Arg.as_param_arg (U.arg (U.buf_uop (T.uop t))) with
  | Some { buffer = Some bufs; _ } -> List.map Device.Buffer.as_bytes bufs
  | _ -> fail "tensor did not realize to storage"

(* Data *)

let host ~shape data = Run.of_float_array ~shape data
let devices n = List.init n (fun i -> Printf.sprintf "CPU:%d" (i + 1))

let f32_bytes values =
  let bytes = Bytes.create (Array.length values * 4) in
  Array.iteri
    (fun i v -> Bytes.set_int32_le bytes (i * 4) (Int32.bits_of_float v))
    values;
  bytes

(* Floats over 40 binades with mixed signs and no zero, so a sum's rounding
   depends on the order it folds them in and adding a zero pad is exact. *)
let spread n =
  Array.init n (fun i ->
      let v =
        Float.ldexp
          (1.0 +. (float_of_int (i * 37 mod 64) /. 64.0))
          ((i * 13 mod 40) - 20)
      in
      if i mod 3 = 0 then -.v else v)

(* Uniform in [-1, 1), drawn from its own generator, so a test's data does not
   depend on which tests ran before it. *)
let uniform ~seed n =
  let rng = Random.State.make seed in
  Array.init n (fun _ -> Random.State.float rng 2.0 -. 1.0)

(* Largest error relative to the largest expected magnitude. *)
let relative_error got expected =
  let err = ref 0.0 and magnitude = ref 0.0 in
  Array.iteri
    (fun i e ->
      err := Float.max !err (Float.abs (got.(i) -. e));
      magnitude := Float.max !magnitude (Float.abs e))
    expected;
  !err /. !magnitude

let check_error ~msg ~tolerance err =
  satisfies ~msg
    ~claim:(Printf.sprintf "at most %g" tolerance)
    float_exact
    (fun err -> err <= tolerance)
    err

(* Collectives *)

let strategies =
  let open Helpers in
  let forced ring_level all2all_level per_box =
    Context_var.
      [
        B (ring, ring_level);
        B (all2all, all2all_level);
        B (allreduce_node_ndevs, per_box);
      ]
  in
  [
    ("naive", forced 0 0 0);
    ("ring", forced 2 0 0);
    ("all2all", forced 0 2 0);
    ("hierarchical, 1 per box", forced 0 0 1);
    ("hierarchical, 2 per box", forced 0 0 2);
    ("hierarchical, 4 per box", forced 0 0 4);
    ("default", []);
  ]

let with_strategy (_, bindings) f = Helpers.Context_var.with_context bindings f
let strategy name = (name, List.assoc name strategies)

(* A copy of a split value to a device list: an all-gather. *)
let gather devices w =
  T.of_uop (U.copy ~src:(T.uop w) ~device:(U.Multi devices) ())

(* Column sums of a [rows; cols] array in float64. *)
let column_sums ~rows ~cols data =
  Array.init cols (fun c ->
      let sum = ref 0.0 and magnitude = ref 0.0 in
      for r = 0 to rows - 1 do
        sum := !sum +. data.((r * cols) + c);
        magnitude := !magnitude +. Float.abs data.((r * cols) + c)
      done;
      (!sum, !magnitude))

(* Every element of [bytes] within [1e-6] of its float64 sum, relative to the
   sum of magnitudes. *)
let check_sums ~msg sums bytes =
  Array.iteri
    (fun i (sum, magnitude) ->
      let got = Int32.float_of_bits (Bytes.get_int32_le bytes (i * 4)) in
      check_error ~msg ~tolerance:1e-6 (Float.abs (got -. sum) /. magnitude))
    sums

let check_replicas_equal ~msg = function
  | [] -> fail "no replicas"
  | first :: rest ->
      List.iteri
        (fun i replica ->
          is_true
            ~msg:(Printf.sprintf "%s: replica %d" msg (i + 2))
            (Bytes.equal first replica))
        rest

let allreduce_tests =
  group "allreduce"
    [
      cases "replicas agree bit for bit" ~name:fst strategies (fun strategy ->
          with_strategy strategy @@ fun () ->
          List.iter
            (fun ndev ->
              let cols = 4096 in
              let data = spread (ndev * cols) in
              let x =
                C.shard ~axis:0 ~devices:(devices ndev)
                  (host ~shape:[ ndev; cols ] data)
              in
              let msg = Printf.sprintf "%d devices" ndev in
              let replicas = device_bytes (Rd.sum ~axis:[ 0 ] x) in
              check_replicas_equal ~msg replicas;
              check_sums ~msg
                (column_sums ~rows:ndev ~cols data)
                (List.hd replicas))
            [ 2; 3; 4; 6; 8 ]);
      xfail ~reason:"an allreduce call is built after outputs are forwarded"
        (test "a realized allreduce holds no more than a consumed one"
           (fun () ->
             let devices = devices 4 and cols = 65536 in
             let w =
               C.shard ~axis:0 ~devices
                 (host ~shape:[ 4; cols ]
                    (uniform ~seed:[| 4; cols |] (4 * cols)))
             in
             Run.realize_many [ w ];
             let peaks f =
               snd (peak_over devices (fun () -> device_bytes (f ())))
             in
             let consumed =
               peaks (fun () -> Rd.sum ~axis:[ 0 ] (Rd.sum ~axis:[ 0 ] w))
             and realized = peaks (fun () -> Rd.sum ~axis:[ 0 ] w) in
             List.iter2
               (fun (device, consumed) (_, realized) ->
                 satisfies ~msg:device
                   ~claim:(Printf.sprintf "at most %d bytes" (consumed + 4))
                   int
                   (fun realized -> realized <= consumed + 4)
                   realized)
               consumed realized));
      test "a realized allreduce of a symbolic slice keeps its values"
        (fun () ->
          let data = Array.init 112 float_of_int in
          let w =
            C.shard ~axis:0 ~devices:(devices 4) (host ~shape:[ 4; 4; 7 ] data)
          in
          Run.realize_many [ w ];
          let cols =
            U.variable ~name:"allreduce_cols" ~min_val:1 ~max_val:7 ()
          in
          List.iter
            (fun n ->
              let sliced =
                U.shrink ~src:(T.uop w)
                  ~offset:(T.shape_uop [ 0; 0; 0 ])
                  ~size:
                    (U.stack
                       [
                         U.const_int 4;
                         U.const_int 4;
                         U.bind ~var:cols ~value:(U.const_int n);
                       ])
              in
              let reduced = Rd.sum ~axis:[ 0 ] (T.of_uop sliced) in
              Run.realize_many [ reduced ];
              let expected = ref 0.0 in
              for a = 0 to 3 do
                for b = 0 to 3 do
                  for c = 0 to n - 1 do
                    expected := !expected +. data.((a * 28) + (b * 7) + c)
                  done
                done
              done;
              equal
                ~msg:(Printf.sprintf "%d columns" n)
                (array float_exact) [| !expected |]
                (Run.to_float_array
                   (C.clone ~device:(U.Single "CPU")
                      (Rd.sum ~axis:[ 0; 1 ] reduced))))
            [ 3; 7 ]);
    ]

(* The per-device replicas of a [rows; cols] value split on axis 0 and gathered
   to its devices, and the transfers of the gather. *)
let gathered ~ndev ~rows ~cols data =
  let devices = devices ndev in
  let w = C.shard ~axis:0 ~devices (host ~shape:[ rows; cols ] data) in
  Run.realize_many [ w ];
  traffic (fun () -> device_bytes (gather devices w))

(* The peak of a 256 KiB value split on [axis] over 4 devices, gathered to them
   and summed. *)
let check_gather_peak ~axis =
  let ndev = 4 and rows = 64 and cols = 1024 in
  let devices = devices ndev in
  let w =
    C.shard ~axis ~devices
      (host ~shape:[ rows; cols ]
         (uniform ~seed:[| rows; cols |] (rows * cols)))
  in
  Run.realize_many [ w ];
  let _, peaks =
    peak_over devices (fun () ->
        device_bytes (Rd.sum ~axis:[ 0; 1 ] (gather devices w)))
  in
  let value = rows * cols * 4 in
  List.iter
    (fun (device, peak) ->
      satisfies ~msg:device
        ~claim:(Printf.sprintf "at most %d bytes" (value + (value / ndev)))
        int
        (fun peak -> peak <= value + (value / ndev))
        peak)
    peaks

let gather_tests =
  group "all-gather"
    [
      cases "replicas equal the source byte for byte" ~name:fst strategies
        (fun strategy ->
          with_strategy strategy @@ fun () ->
          List.iter
            (fun ndev ->
              let data = spread (24 * 256) in
              let replicas, _ = gathered ~ndev ~rows:24 ~cols:256 data in
              List.iteri
                (fun i replica ->
                  is_true
                    ~msg:(Printf.sprintf "%d devices: replica %d" ndev (i + 1))
                    (Bytes.equal (f32_bytes data) replica))
                replicas)
            [ 2; 3; 4; 8 ]);
      test "tiles on two axes reach every device" (fun () ->
          let devices = devices 6 and data = spread 48 in
          let alu op lhs rhs = Symbolic.simplify (U.alu_binary ~op ~lhs ~rhs) in
          let r =
            U.range ~size:(U.const_int 6) ~axis:(-1) ~kind:Axis_type.Device ()
          in
          let a = alu Ops.Floordiv r (U.const_int 3)
          and b = alu Ops.Floormod r (U.const_int 3) in
          let copied =
            U.copy
              ~src:(T.uop (host ~shape:[ 4; 12 ] data))
              ~device:(U.Multi devices) ()
          in
          let tile =
            U.shrink ~src:copied
              ~offset:
                (U.stack
                   [
                     alu Ops.Mul a (U.const_int 2);
                     alu Ops.Mul b (U.const_int 4);
                   ])
              ~size:(T.shape_uop [ 2; 4 ])
          in
          let tiled =
            T.of_uop (U.unshard ~src:tile ~axes:[ 0; 1 ] ~ranges:[ a; b ] ())
          in
          List.iteri
            (fun i replica ->
              is_true
                ~msg:(Printf.sprintf "replica %d" (i + 1))
                (Bytes.equal (f32_bytes data) replica))
            (device_bytes (gather devices tiled)));
      cases "each device receives (n-1)/n of the value" ~name:fst strategies
        (fun strategy ->
          with_strategy strategy @@ fun () ->
          List.iter
            (fun ndev ->
              let value = 48 * 256 * 4 in
              let _, flows =
                gathered ~ndev ~rows:48 ~cols:256
                  (uniform ~seed:[| ndev |] (48 * 256))
              in
              equal
                ~msg:(Printf.sprintf "%d devices" ndev)
                (list int)
                (List.init ndev (fun _ -> (ndev - 1) * value / ndev))
                (List.map (fun d -> received d flows) (devices ndev)))
            [ 2; 3; 4; 8 ]);
      test "a gather lowers to one call named allgather" (fun () ->
          let devices = devices 2 in
          let w = C.shard ~axis:0 ~devices (host ~shape:[ 4; 4 ] (spread 16)) in
          let lowered =
            U.graph_rewrite Multi.multi_pm
              (U.copy ~src:(T.uop w) ~device:(U.Multi devices) ())
          in
          match (U.op lowered, U.children lowered) with
          | Ops.After, [ output; call ] -> (
              match U.as_call call with
              | Some
                  {
                    info = { name = Some "allgather"; precompile = true; _ };
                    args = [ dst; _ ];
                    _;
                  } ->
                  is_true ~msg:"the call writes the output's storage"
                    (U.storage_base output == dst);
                  is_true ~msg:"on the target devices"
                    (U.device_of dst = Some (U.Multi devices))
              | _ -> fail "the AFTER does not wait on an allgather call")
          | _ -> fail "the gather is not an AFTER of one call");
      test "a gather of a symbolic slice keeps its values" (fun () ->
          let devices = devices 4 and data = Array.init 56 float_of_int in
          let w = C.shard ~axis:0 ~devices (host ~shape:[ 8; 7 ] data) in
          Run.realize_many [ w ];
          let cols = U.variable ~name:"gather_cols" ~min_val:1 ~max_val:7 () in
          List.iter
            (fun n ->
              let expected = ref 0.0 in
              for r = 0 to 7 do
                for c = 0 to n - 1 do
                  expected := !expected +. data.((r * 7) + c)
                done
              done;
              List.iter
                (fun device ->
                  let sliced =
                    U.shrink ~src:(T.uop w)
                      ~offset:(T.shape_uop [ 0; 0 ])
                      ~size:
                        (U.stack
                           [
                             U.const_int 8;
                             U.bind ~var:cols ~value:(U.const_int n);
                           ])
                  in
                  let gathered = T.of_uop (U.copy ~src:sliced ~device ()) in
                  equal
                    ~msg:(Printf.sprintf "%d columns" n)
                    (array float_exact) [| !expected |]
                    (Run.to_float_array
                       (C.clone ~device:(U.Single "CPU")
                          (Rd.sum ~axis:[ 0; 1 ] gathered))))
                [ U.Single "CPU:1"; U.Multi devices ])
            [ 3; 7 ]);
      (* The gather feeds a sum, as a gathered weight feeds a product. *)
      test "a row-split gather holds the value and at most one shard more"
        (fun () -> check_gather_peak ~axis:0);
      xfail ~reason:"inner-axis windows stage every foreign shard at once"
        (test "a column-split gather holds the value and at most one shard more"
           (fun () -> check_gather_peak ~axis:1));
      test "a realized gather holds one value per device" (fun () ->
          let devices = devices 4 and rows = 64 and cols = 1024 in
          let data = uniform ~seed:[| rows; cols |] (rows * cols) in
          let w = C.shard ~axis:0 ~devices (host ~shape:[ rows; cols ] data) in
          Run.realize_many [ w ];
          List.iter
            (fun device ->
              let replicas, peaks =
                peak_over devices (fun () ->
                    device_bytes (T.of_uop (U.copy ~src:(T.uop w) ~device ())))
              in
              List.iter
                (fun replica ->
                  is_true ~msg:"replica equals the source"
                    (Bytes.equal (f32_bytes data) replica))
                replicas;
              List.iter
                (fun (d, peak) ->
                  satisfies ~msg:d
                    ~claim:(Printf.sprintf "at most %d bytes" (rows * cols * 4))
                    int
                    (fun peak -> peak <= rows * cols * 4)
                    peak)
                peaks)
            [ U.Single "CPU:1"; U.Multi devices ]);
    ]

(* Today the reshard of an allreduce to split rows allreduces the whole value
   and each device keeps its rows. Target: a reduce-scatter, in which each
   device sends (n-1)/n of its partial. *)
let reshard_tests =
  cases "allreduce resharded to rows sends an allreduce's bytes" ~name:fst
    [ strategy "naive"; strategy "ring"; strategy "all2all" ]
    (fun ((name, _) as strategy) ->
      with_strategy strategy @@ fun () ->
      let ndev = 4 and rows = 64 and cols = 1024 in
      let devices = devices ndev in
      let data = spread (ndev * rows * cols) in
      let partials =
        C.shard ~axis:0 ~devices (host ~shape:[ ndev; rows; cols ] data)
      in
      Run.realize_many [ partials ];
      let shards, flows =
        traffic (fun () ->
            device_bytes
              (C.shard ~axis:0 ~devices (Rd.sum ~axis:[ 0 ] partials)))
      in
      let partial = rows * cols * 4 in
      let per_device =
        if name = "naive" then (ndev - 1) * partial
        else 2 * (ndev - 1) * partial / ndev
      in
      equal (list int)
        (List.init ndev (fun _ -> per_device))
        (List.map (fun d -> sent d flows) devices);
      let sums = column_sums ~rows:ndev ~cols:(rows * cols) data in
      let block = rows / ndev * cols in
      List.iteri
        (fun j shard ->
          check_sums
            ~msg:(Printf.sprintf "rows of device %d" (j + 1))
            (Array.sub sums (j * block) block)
            shard)
        shards)

(* A tensor-parallel MLP, [relu (x @ W1) @ W2] with [W1] split on its columns
   and [W2] on its rows, moves nothing between devices but the allreduce of its
   output. *)
let tensor_parallel ~batch name =
  test (Printf.sprintf "batch %d, %s" batch name) @@ fun () ->
  with_strategy (strategy name) @@ fun () ->
  let ndev = 4 and d = 512 and h = 2048 in
  let devices = devices ndev in
  let w1 = uniform ~seed:[| d; h |] (d * h)
  and w2 = uniform ~seed:[| h; d |] (h * d)
  and x = uniform ~seed:[| batch; d |] (batch * d) in
  let mlp x w1 w2 = Op.matmul (El.relu (Op.matmul x w1)) w2 in
  let split_w1 = C.shard ~axis:1 ~devices (host ~shape:[ d; h ] w1) in
  let split_w2 = C.shard ~axis:0 ~devices (host ~shape:[ h; d ] w2) in
  Run.realize_many [ split_w1; split_w2 ];
  let replicas, flows =
    traffic (fun () ->
        device_bytes
          (mlp
             (C.shard ~devices (host ~shape:[ batch; d ] x))
             split_w1 split_w2))
  in
  let _, allreduce =
    let partials =
      C.shard ~axis:0 ~devices
        (host ~shape:[ ndev; batch; d ]
           (uniform ~seed:[| ndev; batch; d |] (ndev * batch * d)))
    in
    Run.realize_many [ partials ];
    traffic (fun () -> device_bytes (Rd.sum ~axis:[ 0 ] partials))
  in
  equal int (peer allreduce) (peer flows);
  check_replicas_equal ~msg:"output" replicas;
  let expected =
    Run.to_float_array
      (mlp
         (host ~shape:[ batch; d ] x)
         (host ~shape:[ d; h ] w1)
         (host ~shape:[ h; d ] w2))
  in
  let got =
    Array.init (batch * d) (fun i ->
        Int32.float_of_bits (Bytes.get_int32_le (List.hd replicas) (i * 4)))
  in
  check_error ~msg:"output against one device" ~tolerance:1e-5
    (relative_error got expected)

(* Fully sharded step *)

(* A layer loop in which parameters, gradients and Adam's two moments are split
   on axis 0 over the devices, the batch is split on axis 0, and each weight is
   gathered to every device where it is used. *)
type layout = {
  place : T.t -> T.t;
  gather : T.t -> T.t;
  grad : T.t -> T.t -> T.t; (* [grad h da] is [h^T da], placed as [place]. *)
}

let transpose t = Mv.transpose ~dim0:0 ~dim1:1 t

let fsdp devices =
  {
    place = C.shard ~axis:0 ~devices;
    gather = gather devices;
    grad = (fun h da -> C.shard ~axis:0 ~devices (Op.matmul (transpose h) da));
  }

let one_device =
  {
    place = Fun.id;
    gather = Fun.id;
    grad = (fun h da -> Op.matmul (transpose h) da);
  }

type state = {
  x : T.t;
  w : T.t array;
  m : T.t array;
  v : T.t array;
  g : T.t array;
}

(* Weights [ws] and a batch [xs] placed by [layout], with zero gradients and
   moments, realized. *)
let setup layout ~dims ~batch ~ws ~xs =
  let place l data =
    layout.place (host ~shape:[ dims.(l); dims.(l + 1) ] data)
  in
  let zeros () =
    Array.mapi (fun l w -> place l (Array.make (Array.length w) 0.0)) ws
  in
  let s =
    {
      x = layout.place (host ~shape:[ batch; dims.(0) ] xs);
      w = Array.mapi place ws;
      m = zeros ();
      v = zeros ();
      g = zeros ();
    }
  in
  Run.realize_many
    ((s.x :: Array.to_list s.w) @ Array.to_list s.m @ Array.to_list s.v
   @ Array.to_list s.g);
  s

(* One training step of [relu (h @ W)] layers under the loss [0.5 sum h^2], then
   Adam, with one realize per layer and phase and the backward split into a
   gradient realize and a [dh] realize. Settles after each realize. *)
let step layout s =
  let layers = Array.length s.w in
  let realize ts =
    Run.realize_many ts;
    settle ()
  in
  let h = Array.make (layers + 1) s.x in
  for l = 0 to layers - 1 do
    h.(l + 1) <- El.relu (Op.matmul h.(l) (layout.gather s.w.(l)));
    realize [ h.(l + 1) ]
  done;
  let dh = ref h.(layers) in
  for l = layers - 1 downto 0 do
    let da = El.where (El.gt h.(l + 1) (T.f 0.0)) !dh (T.f 0.0) in
    ignore (Op.assign s.g.(l) (layout.grad h.(l) da));
    realize [ s.g.(l) ];
    dh := Op.matmul da (transpose (layout.gather s.w.(l)));
    realize [ !dh ];
    let b1 = 0.9 and b2 = 0.999 and lr = 1e-3 and eps = 1e-3 in
    let m' =
      El.add (El.mul s.m.(l) (T.f b1)) (El.mul s.g.(l) (T.f (1. -. b1)))
    in
    let v' =
      El.add
        (El.mul s.v.(l) (T.f b2))
        (El.mul (El.mul s.g.(l) s.g.(l)) (T.f (1. -. b2)))
    in
    let w' =
      El.sub s.w.(l)
        (El.mul (T.f lr) (El.div m' (El.add (El.sqrt v') (T.f eps))))
    in
    ignore (Op.assign s.m.(l) m');
    ignore (Op.assign s.v.(l) v');
    ignore (Op.assign s.w.(l) w');
    realize [ s.m.(l); s.v.(l); s.w.(l) ]
  done

let to_host t = Run.to_float_array (C.clone ~device:(U.Single "CPU") t)

type measure = {
  errors : float * float; (* Gradients and weights against one device. *)
  peer_bytes : int; (* Moved between devices by the step. *)
  state : (string * int) list; (* Per device, live bytes of the setup. *)
  over_state : (string * int) list;
      (* Per device, the highest live bytes during the step minus the live bytes
         after setup. *)
}

let measure ~ndev ~dims ~batch () =
  let layers = Array.length dims - 1 in
  let ws =
    Array.init layers (fun l ->
        let scale = sqrt (float_of_int dims.(l)) in
        Array.map
          (fun v -> v /. scale)
          (uniform
             ~seed:[| l; dims.(l); dims.(l + 1) |]
             (dims.(l) * dims.(l + 1))))
  in
  let xs = uniform ~seed:[| batch; dims.(0) |] (batch * dims.(0)) in
  let reference = setup one_device ~dims ~batch ~ws ~xs in
  step one_device reference;
  let devices = devices ndev in
  let layout = fsdp devices in
  let s, state = held devices (fun () -> setup layout ~dims ~batch ~ws ~xs) in
  let ((), flows), over_state =
    peak_over devices (fun () -> traffic (fun () -> step layout s))
  in
  let error got expected =
    Array.fold_left Float.max 0.0
      (Array.map2
         (fun got expected -> relative_error (to_host got) (to_host expected))
         got expected)
  in
  {
    errors = (error s.g reference.g, error s.w reference.w);
    peer_bytes = peer flows;
    state;
    over_state;
  }

let in_layers layer =
  Testable.make ~equal:Int.equal ~pp:(fun ppf n ->
      Format.fprintf ppf "%.3f layers" (float_of_int n /. float_of_int layer))

(* Fully sharded training fits when every device stays within two gathered
   layers plus its saved activations above its share of parameters, gradients
   and optimizer state. Today's lowering misses that bound for two reasons: the
   gradient is allreduced whole before each device keeps its rows, and every
   copy source is staged in its own buffer. *)
let fully_sharded (name, ndev, dims) =
  let batch = 64 and layers = Array.length dims - 1 in
  let layer_bytes = Array.init layers (fun l -> dims.(l) * dims.(l + 1) * 4) in
  let weight_bytes = Array.fold_left ( + ) 0 layer_bytes in
  let layer = Array.fold_left max 0 layer_bytes in
  let activations =
    Array.fold_left ( + ) 0
      (Array.init layers (fun l -> batch * dims.(l + 1) * 4 / ndev))
  in
  let bound = (2 * layer) + activations in
  let run = fixture (measure ~ndev ~dims ~batch) in
  group ~tags:[ "slow" ] name
    [
      test "gradients and weights match one device" (fun () ->
          let g, w = (run ()).errors in
          check_error ~msg:"gradients" ~tolerance:1e-5 g;
          check_error ~msg:"weights" ~tolerance:1e-5 w);
      test "each device holds its share of parameters, gradients and moments"
        (fun () ->
          let share =
            (4 * weight_bytes / ndev) + (batch * dims.(0) * 4 / ndev)
          in
          List.iter
            (fun (device, held) -> equal ~msg:device int share held)
            (run ()).state);
      (* Each weight is gathered twice, (n-1) layers each, and its gradient
         allreduced, 2(n-1) layers. Target: (n-1) for the gradient too, once it
         reduce-scatters. *)
      test "gathers move (n-1) layers and gradients 2(n-1)" (fun () ->
          equal int (4 * (ndev - 1) * weight_bytes) (run ()).peer_bytes);
      xfail
        ~reason:
          "gradients allreduce before the reshard, copies stage their sources"
        (test "each device holds at most two layers and activations over state"
           (fun () ->
             let device, over =
               List.fold_left
                 (fun worst (device, over) ->
                   if over > snd worst then (device, over) else worst)
                 ("", min_int) (run ()).over_state
             in
             satisfies
               ~msg:(Printf.sprintf "worst device, %s" device)
               ~claim:
                 (Printf.sprintf "at most %.3f layers"
                    (float_of_int bound /. float_of_int layer))
               (in_layers layer)
               (fun over -> over <= bound)
               over));
    ]

let () =
  Helpers.Context_var.with_context
    [ B (Helpers.dev, [ Target.of_string "CPU" ]) ]
  @@ fun () ->
  run "Collectives"
    [
      group "Harness"
        [
          test "a buffer counts while it is allocated" (fun () ->
              let buf =
                Device.create_buffer ~size:256 ~dtype:Dtype.float32
                  (Device.get "CPU:1")
              in
              (* Tolk's own count also holds buffers the harness does not see,
                 such as the linked-symbol slots the inner device allocates, so
                 the two are compared as changes. *)
              let counts () =
                ( (usage "CPU:1").live,
                  Option.value ~default:0
                    (Hashtbl.find_opt Storage.mem_used_per_device "CPU:1") )
              in
              let before = counts () in
              Device.Buffer.ensure_allocated buf;
              let live, used = counts () in
              equal int (fst before + 1024) live;
              equal ~msg:"tolk's own count" int (snd before + 1024) used;
              Device.Buffer.deallocate buf;
              equal (pair int int) before (counts ()));
          test "a copy between devices records its bytes" (fun () ->
              let data = uniform ~seed:[| 256 |] 256 in
              let x =
                C.clone ~device:(U.Single "CPU:1") (host ~shape:[ 256 ] data)
              in
              Run.realize_many [ x ];
              let copied, flows =
                traffic (fun () ->
                    device_bytes (C.clone ~device:(U.Single "CPU:2") x))
              in
              equal (list bytes) [ f32_bytes data ] copied;
              equal int 1024 (sent "CPU:1" flows);
              equal int 1024 (received "CPU:2" flows);
              equal int 1024 (peer flows));
          test "a peak is measured from the live bytes at entry" (fun () ->
              let buffer size =
                Device.create_buffer ~size ~dtype:Dtype.float32
                  (Device.get "CPU:1")
              in
              let kept = buffer 64 in
              Device.Buffer.ensure_allocated kept;
              let (), peaks =
                peak_over [ "CPU:1" ] (fun () ->
                    let transient = buffer 256 in
                    Device.Buffer.ensure_allocated transient;
                    Device.Buffer.deallocate transient)
              in
              equal (list (pair string int)) [ ("CPU:1", 1024) ] peaks;
              Device.Buffer.deallocate kept);
        ];
      allreduce_tests;
      gather_tests;
      reshard_tests;
      group "tensor-parallel MLP moves only its allreduce"
        [
          tensor_parallel ~batch:8 "naive";
          tensor_parallel ~batch:8 "ring";
          tensor_parallel ~batch:512 "naive";
          tensor_parallel ~batch:512 "ring";
        ];
      group "fully sharded step"
        (List.map fully_sharded
           [
             ("8 layers of 1024 x 1024 on 4 devices", 4, Array.make 9 1024);
             ("8 layers of 1024 x 1024 on 8 devices", 8, Array.make 9 1024);
             ( "4 layers of 2048 x 4096 on 4 devices",
               4,
               [| 2048; 4096; 2048; 4096; 2048 |] );
           ]);
    ]
