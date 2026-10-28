(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's per-call costs above the array layer, each row what a user calls: an
   operation on one-element host values, its kernel called as nx calls it and
   directly, a decode step's cache update, a view, creations, an operation
   over two devices, reading a value's shape, placing, and operations beside
   and under interpretations.

   Each row builds its inputs in its setup, in the worker that times it, so
   that a worker holds one row's inputs. *)

module A = Nx_array
module D = Nx_array.Dtype

(* A memory device, opened in the measuring worker: a device opened before the
   fork is lost in it. *)
let memory k = Result.get_ok (Rig.memory_device (Printf.sprintf "bench-m%d" k))

let filled dt s v =
  let n = Array.fold_left ( * ) 1 s in
  Nx.Repr.of_array Nx.Host.v
    (A.of_array dt (Iarray.of_array s) (Array.make n v))

let host n = filled D.Float32 [| n |] 1.
let one () = host 1
let on_host x = Nx.place Nx.Host.on x

(* A float32 add's program. *)
let add_f32 =
  let f = D.Any D.Float32 in
  Nx_kernel.Prog.of_node ~ins:[| f; f |] (Op2 (Binary Add, 0, 1))

(* K1's steps over nx.array and nx.cpu, the kernel called statically: the floor
   [dispatch/add-1]'s indirect call is judged against. *)
let add_direct a =
  let l = A.layout a in
  let dst =
    A.v D.Float32 l (Rig.Buffer.create (A.device a) (D.bytes D.Float32 1))
  in
  let loads = [| Nx_kernel.Plain a; Plain a |] in
  match Nx_cpu.map add_f32 ~dsts:[| A.Any dst |] loads with
  | Done -> Nx.Repr.of_array Nx.Host.v dst
  | refusal -> A.refused "add-1-direct" refusal [ A.Any dst; A.Any a ]

let chain half n =
  let rec go k x = if k = 0 then x else go (k - 1) (Nx.add x half) in
  go n (Nx.zeros D.Float32 [| 1 |])

let half () = Nx.scalar D.Float32 0.5

(* A decode step's cache update: one row written at a position held in data
   into gpt-oss's keys of one layer, a [1; 8; 4096; 64] bfloat16 cache, each
   call updating the last call's result as decode steps do. Donated, the row
   is written in place; copied, the cache is copied first. *)
let kv_cache () =
  let cache = Nx.copy (filled D.Bfloat16 [| 1; 8; 4096; 64 |] 0.) in
  let row = filled D.Bfloat16 [| 1; 8; 1; 64 |] 1. in
  let pos = on_host (Nx.scalar D.Int64 2048L) in
  (ref cache, row, Nx.[ A; A; D (pos, 1); A ])

(* A float32 sine of [x] built as an operation once, as an interpreter that
   evaluates a program holds it. *)
let sin_op x : ((float, D.float32_elt, Nx.host) Nx.t * unit) Nx.Prim.t =
  let prog =
    Nx_kernel.Prog.of_node ~ins:[| D.Any D.Float32 |]
      (Op1 (Unary Sin, D.Any D.Float32, 0))
  in
  Map
    {
      layout = Nx_array.Layout.contiguous (Iarray.of_array (Nx.shape x));
      prog;
      outs = Nx.Prim.[ D.Float32 ];
      loads = [| Plain x |];
      at = None;
    }

(* The sum of [xs], float32 arrays of one shape, built as one map of an
   output and a load each, as a compiler fusing maps or a derivative builds
   one. *)
let sum_op xs : ((float, D.float32_elt, Nx.host) Nx.t * unit) Nx.Prim.t =
  let n = Array.length xs and f = D.Any D.Float32 in
  let nodes =
    List.init n (fun k -> Nx_kernel.Prog.In k)
    @ List.init (n - 1) (fun i ->
          let acc = if i = 0 then 0 else n + i - 1 in
          Nx_kernel.Prog.Op2 (Binary Add, acc, i + 1))
  in
  let prog =
    Nx_kernel.Prog.v
      ~ins:(Iarray.init n (fun _ -> f))
      (Iarray.of_list nodes)
      ~outs:[| List.length nodes - 1 |]
  in
  Map
    {
      layout = Nx_array.Layout.contiguous (Iarray.of_array (Nx.shape xs.(0)));
      prog;
      outs = Nx.Prim.[ D.Float32 ];
      loads = Iarray.map (fun x -> Nx.Prim.Plain x) (Iarray.of_array xs);
      at = None;
    }

let dispatch_rows =
  Thumper.group "dispatch"
    [
      Thumper.bench_with_setup "shape" ~setup:one Nx.shape;
      Thumper.bench_with_setup "add-1" ~setup:one (fun x -> Nx.add x x);
      Thumper.bench_with_setup "sin-1" ~setup:one Nx.sin;
      Thumper.bench_with_setup "prim-eval-sin-1"
        ~setup:(fun () -> sin_op (one ()))
        (fun op -> fst (Nx.Prim.eval ~by:"bench" op));
      Thumper.bench_with_setup "add-1-direct"
        ~setup:(fun () -> Option.get (Nx.Repr.array (one ())))
        add_direct;
      Thumper.bench_with_setup "less-1" ~setup:one (fun x -> Nx.less x x);
      Thumper.bench_with_setup "where-1"
        ~setup:(fun () ->
          let x = one () in
          (Nx.less x x, x))
        (fun (b, x) -> Nx.where b x x);
      Thumper.bench_with_setup "cast-1" ~setup:one (Nx.cast D.Float64);
      Thumper.bench_with_setup "reshape-1" ~setup:one (Nx.reshape [| 1; 1 |]);
      Thumper.bench_with_setup "get-1-of-4x4"
        ~setup:(fun () -> Nx.reshape [| 4; 4 |] (host 16))
        (Nx.get [ 1; 2 ]);
      Thumper.bench_with_setup "zeros_like-1" ~setup:one Nx.zeros_like;
      Thumper.bench_with_setup "zeros_like-1M"
        ~setup:(fun () -> host (1 lsl 20))
        Nx.zeros_like;
      Thumper.bench "zeros-1M" (fun () ->
          Nx.place Nx.Host.on (Nx.zeros D.Float32 [| 1 lsl 20 |]));
      Thumper.bench_with_setup "add-1-twice" ~setup:one (fun x ->
          Nx.add (Nx.add x x) x);
      Thumper.bench_with_setup "add-1-twice-donated" ~setup:one (fun x ->
          Nx.add (Nx.donate (Nx.add x x)) x);
      Thumper.bench_with_setup "where-1-twice-donated"
        ~setup:(fun () ->
          let x = one () in
          (Nx.less x x, x))
        (fun (b, x) -> Nx.where b (Nx.donate (Nx.add x x)) x);
      Thumper.bench_with_setup "cast-1-twice-donated" ~setup:one (fun x ->
          Nx.cast D.Float64 (Nx.donate (Nx.add x x)));
      Thumper.bench_with_setup "add-scalar-1" ~setup:one (fun x ->
          Nx.add x (Nx.scalar D.Float32 1.));
      Thumper.bench_with_setup "add_s-1" ~setup:one (fun x -> Nx.add_s x 1.);
      Thumper.bench_with_setup "add-held-scalar-1"
        ~setup:(fun () -> (one (), half ()))
        (fun (x, half) -> Nx.add x half);
      Thumper.bench_with_setup "kv-append-4096" ~setup:kv_cache
        (fun (cache, row, at) -> cache := Nx.set at row (Nx.donate !cache));
      Thumper.bench_with_setup "kv-append-4096-copied" ~setup:kv_cache
        (fun (cache, row, at) -> cache := Nx.set at row !cache);
    ]

(* Calls on 16 elements, where a call's own cost outweighs its arithmetic,
   as a sampler's steps over a small state make them: elementwise kinds,
   a scalar form, a sum, and a leapfrog step x + dt (v + dt/2 g) of three
   such calls and two scalar forms. *)
let small_rows =
  let f32 () = filled D.Float32 [| 16 |] 1.5 in
  let f64 () = filled D.Float64 [| 16 |] 1.5 in
  let dt = 0.1 in
  let step (x, v, g) =
    Nx.add x (Nx.mul_s (Nx.add v (Nx.mul_s g (dt /. 2.))) dt)
  in
  Thumper.group "small"
    [
      Thumper.bench_with_setup "add-f32-16" ~setup:f32 (fun x -> Nx.add x x);
      Thumper.bench_with_setup "exp-f32-16" ~setup:f32 Nx.exp;
      Thumper.bench_with_setup "where-f32-16"
        ~setup:(fun () ->
          let x = f32 () in
          (Nx.less x x, x))
        (fun (c, x) -> Nx.where c x x);
      Thumper.bench_with_setup "mul_s-f32-1" ~setup:one (fun x ->
          Nx.mul_s x 0.5);
      Thumper.bench_with_setup "mul_s-f32-16" ~setup:f32 (fun x ->
          Nx.mul_s x 0.5);
      Thumper.bench_with_setup "mul_s-f64-16" ~setup:f64 (fun x ->
          Nx.mul_s x 0.5);
      Thumper.bench_with_setup "sum-f32-16" ~setup:f32 (fun x -> Nx.sum x);
      Thumper.bench_with_setup "sum-f64-16" ~setup:f64 (fun x -> Nx.sum x);
      Thumper.bench_with_setup "leapfrog-step-f32-16"
        ~setup:(fun () -> (f32 (), f32 (), f32 ()))
        step;
    ]

(* Elementwise operations of one element to past the caches: an add at
   float32, float16 and bfloat16, a where, host creations (a fill and two
   ranges), and adds of a transposed and of a broadcast
   operand. *)
let elementwise_rows =
  let m = 1 lsl 20 in
  let sizes = [ (1, "1"); (4096, "4K"); (m, "1M") ] in
  let add name dt =
    List.map
      (fun (n, size) ->
        Thumper.bench_with_setup
          (Printf.sprintf "add-%s-%s" name size)
          ~setup:(fun () -> filled dt [| n |] 1.)
          (fun x -> Nx.add x x))
      sizes
  in
  let where (n, size) =
    Thumper.bench_with_setup
      (Printf.sprintf "where-f32-%s" size)
      ~setup:(fun () ->
        let x = host n in
        (Nx.less x x, x))
      (fun (c, x) -> Nx.where c x x)
  in
  let square () = filled D.Float32 [| 1024; 1024 |] 1. in
  Thumper.group "elementwise"
    (add "f32" D.Float32 @ add "f16" D.Float16 @ add "bf16" D.Bfloat16
    @ List.map where sizes
    @ [
        Thumper.bench "full-f32-1M" (fun () ->
            on_host (Nx.full D.Float32 [| m |] 1.5));
        Thumper.bench "arange-i64-1M" (fun () ->
            on_host (Nx.arange D.Int64 0 m 1));
        Thumper.bench "arange-f32-1M" (fun () ->
            on_host (Nx.arange D.Float32 0 m 1));
        Thumper.bench_with_setup "add-f32-1M-transposed"
          ~setup:(fun () -> (Nx.transpose (square ()), square ()))
          (fun (t, y) -> Nx.add t y);
        Thumper.bench_with_setup "map-sum-24-f32-64K"
          ~setup:(fun () ->
            let x () = filled D.Float32 [| 1 lsl 16 |] 1. in
            sum_op (Array.init 23 (fun _ -> x ())))
          (fun op -> fst (Nx.Prim.eval ~by:"bench" op));
        Thumper.bench_with_setup "add-f32-1M-broadcast"
          ~setup:(fun () -> (square (), filled D.Float32 [| 1024 |] 1.))
          (fun (x, row) -> Nx.add x row);
      ])

let constant_rows =
  Thumper.group "constant"
    [
      Thumper.bench_with_setup "chain-1000" ~setup:half (fun half ->
          Nx.Repr.array (Nx.place Nx.Host.on (chain half 1000)));
    ]

(* A value on a set minted in the worker, of a brand the row does not name. *)
type value = Value : (float, D.float32_elt, 'd) Nx.t -> value

(* One element on each of two memory devices. *)
let split_two () =
  let module Two = (val Nx.devices [ memory 0; memory 1 ]) in
  Value (Nx.place (Two.split ~axis:0) (host 2))

(* Sixteen host elements, and a placement on a memory device. *)
type borrow =
  | Borrow : 'd Nx.Placement.t * (float, D.float32_elt, Nx.host) Nx.t -> borrow

let borrow_16 () =
  let module Mem = (val Nx.devices [ memory 0 ]) in
  Borrow (Mem.on, host 16)

(* A 1024 × 1024 float32 value, 4 MiB, and a split on axis 1 over four memory
   devices: from the host, and from a split on axis 0, where each device's
   column window meets every row window. *)
type reshard =
  | Reshard : 'd Nx.Placement.t * (float, D.float32_elt, 'e) Nx.t -> reshard

let square () = filled D.Float32 [| 1024; 1024 |] 1.

let host_to_columns () =
  let module Four = (val Nx.devices [ memory 0; memory 1; memory 2; memory 3 ])
  in
  Reshard (Four.split ~axis:1, square ())

let rows_to_columns () =
  let module Four = (val Nx.devices [ memory 0; memory 1; memory 2; memory 3 ])
  in
  Reshard (Four.split ~axis:1, Nx.place (Four.split ~axis:0) (square ()))

(* A 64 × 64 matmul's operands on two memory devices: the left split by rows,
   the right whole on each, so each device computes its rows. *)
type operands =
  | Operands :
      (float, D.float32_elt, 'd) Nx.t * (float, D.float32_elt, 'd) Nx.t
      -> operands

let rows_two () =
  let module Two = (val Nx.devices [ memory 0; memory 1 ]) in
  let x = host (64 * 64) |> Nx.reshape [| 64; 64 |] in
  Operands (Nx.place (Two.split ~axis:0) x, Nx.place Two.on x)

(* Ones of shape [s] split along [axis] over [n] memory devices. *)
let split_over n ~axis s () =
  let module Set = (val Nx.devices (List.init n memory)) in
  Value (Nx.place (Set.split ~axis) (filled D.Float32 s 1.))

(* A 256 × 4096 by 4096 × 256 matmul's operands split along k over four
   memory devices: each device holds a quarter of the contracted terms. *)
let k_four () =
  let module Four = (val Nx.devices [ memory 0; memory 1; memory 2; memory 3 ])
  in
  let a = filled D.Float32 [| 256; 4096 |] 1.
  and b = filled D.Float32 [| 4096; 256 |] 1. in
  Operands (Nx.place (Four.split ~axis:1) a, Nx.place (Four.split ~axis:0) b)

(* Reductions, scans and a contraction along cut axes, each device computing
   partial results where its terms lie: a column sum of rows over eight
   devices; a sum of all of a [8192; 1000] cut by columns, whose windows
   interleave along the reduced axes; a cumsum along a long cut axis and
   along rows of 8192 over four devices, windows shorter than a scan's
   chunk; and a matmul cut along k. *)
let partial_rows =
  [
    Thumper.bench_with_setup "sum-f32-8192x4096-cols-split8"
      ~setup:(split_over 8 ~axis:0 [| 8192; 4096 |])
      (fun (Value x) -> Value (Nx.sum ~axes:[ 0 ] x));
    Thumper.bench_with_setup "sum-f32-8192x1000-split4-axis1"
      ~setup:(split_over 4 ~axis:1 [| 8192; 1000 |])
      (fun (Value x) -> Value (Nx.sum x));
    Thumper.bench_with_setup "cumsum-f32-1M-split4"
      ~setup:(split_over 4 ~axis:0 [| 1 lsl 20 |])
      (fun (Value x) -> Value (Nx.cumsum ~axis:0 x));
    Thumper.bench_with_setup "cumsum-f32-256x8192-rows-split4"
      ~setup:(split_over 4 ~axis:1 [| 256; 8192 |])
      (fun (Value x) -> Value (Nx.cumsum ~axis:1 x));
    Thumper.bench_with_setup "matmul-f32-256x4096x256-k-split4" ~setup:k_four
      (fun (Operands (a, b)) -> Value (Nx.matmul a b));
  ]

let placed_rows =
  Thumper.group "placed"
    ([
       Thumper.bench_with_setup "add-1-two-memory-devices" ~setup:split_two
         (fun (Value two) -> Value (Nx.add two two));
       Thumper.bench_with_setup "matmul-f32-64-rows-two-memory-devices"
         ~setup:rows_two (fun (Operands (a, b)) -> Value (Nx.matmul a b));
       Thumper.bench_with_setup "item-f32-1M-split4"
         ~setup:(split_over 4 ~axis:0 [| 1 lsl 20 |])
         (fun (Value x) -> Nx.item [ 524_287 ] x);
     ]
    @ partial_rows)

let place_rows =
  Thumper.group "place"
    [
      Thumper.bench_with_setup "equal-1" ~setup:one (Nx.place Nx.Host.on);
      Thumper.bench_with_setup "borrow-memory-device-16" ~setup:borrow_16
        (fun (Borrow (on, x)) -> Value (Nx.place on x));
      Thumper.bench_with_setup "host-to-split-axis1" ~setup:host_to_columns
        (fun (Reshard (p, x)) -> Value (Nx.place p x));
      Thumper.bench_with_setup "split-to-split" ~setup:rows_to_columns
        (fun (Reshard (p, x)) -> Value (Nx.place p x));
    ]

(* Movements: a view costs the same at one element and at a million, and a
   movement no stride expresses costs a copy of its result, as copy-1M does. *)

let split_heads = Nx.Pattern.v "b t (h d) -> b h t d"
let rows () = Nx.reshape [| 1; 1024; 1024 |] (square ())

let move_rows =
  Thumper.group "move"
    [
      Thumper.bench_with_setup "transpose-1"
        ~setup:(fun () -> Nx.reshape [| 1; 1 |] (one ()))
        Nx.transpose;
      Thumper.bench_with_setup "transpose-1M" ~setup:square Nx.transpose;
      Thumper.bench_with_setup "rearrange-heads-1M" ~setup:rows
        (Nx.rearrange ~sizes:[ ("h", 16) ] split_heads);
      Thumper.bench_with_setup "reshape-swapaxes-heads-1M" ~setup:rows
        (fun x -> Nx.swapaxes 1 2 (Nx.reshape [| 1; 1024; 16; 64 |] x));
      Thumper.bench_with_setup "repeat-256K-4"
        ~setup:(fun () -> host (1 lsl 18))
        (Nx.repeat ~axis:0 4);
      Thumper.bench_with_setup "copy-1M"
        ~setup:(fun () -> host (1 lsl 20))
        Nx.copy;
    ]

(* Random draws computed on the host: each row places a draw of a fixed key,
   which computes it. The samplers' programs differ in length: one Threefry
   block per element for bits, uniform and float32 normal, and fixed rounds of
   proposals for the rejection samplers. bernoulli-f32-1Kx1K draws a
   probability array of rank 2. The one-element draws time a draw's fixed
   cost: a sampler's program is lowered once per dtype, and beta's has two
   thousand nodes. *)

let mib = 1 lsl 20
let key () = Nx.Rng.key 0
let draw x = Nx.Repr.array (Nx.place Nx.Host.on x)

(* A key and a float32 parameter of shape [s] filled with [v]. *)
let keyed s v () = (key (), filled D.Float32 s v)

(* The draws' floor: nx.cpu's Threefry called directly over 1M counters under
   one broadcast key, the one block per element that bits and uniform compute,
   without the counters' and words' arithmetic around it. *)
let threefry_operands () =
  let counters = A.of_array D.Uint64 [| mib |] (Array.init mib Int64.of_int) in
  let one_key =
    Option.get
      (A.move (A.Move.Broadcast [| mib |])
         (A.of_array D.Uint64 [| 1 |] [| 0L |]))
  in
  (counters, one_key)

let threefry =
  let u = D.Any D.Uint64 in
  Nx_kernel.Prog.of_node ~ins:[| u; u |] (Op2 (Binary Threefry, 0, 1))

let threefry_direct (c, one_key) =
  let dst = A.create (A.device c) D.Uint64 [| mib |] in
  let loads = [| Nx_kernel.Plain c; Plain one_key |] in
  match Nx_cpu.map threefry ~dsts:[| A.Any dst |] loads with
  | Done -> dst
  | refusal -> A.refused "threefry-1M-direct" refusal [ A.Any dst; A.Any c ]

let rng_rows =
  Thumper.group "rng"
    [
      Thumper.bench_with_setup "threefry-1M-direct" ~setup:threefry_operands
        threefry_direct;
      Thumper.bench_with_setup "bits-1M" ~setup:key (fun key ->
          draw (Nx.Rng.bits ~key [| mib |]));
      Thumper.bench_with_setup "uniform-f32-1M" ~setup:key (fun key ->
          draw (Nx.Rng.uniform ~key D.Float32 [| mib |]));
      Thumper.bench_with_setup "uniform-f64-1M" ~setup:key (fun key ->
          draw (Nx.Rng.uniform ~key D.Float64 [| mib |]));
      Thumper.bench_with_setup "normal-f32-1M" ~setup:key (fun key ->
          draw (Nx.Rng.normal ~key D.Float32 [| mib |]));
      Thumper.bench_with_setup "randint-1M" ~setup:key (fun key ->
          draw (Nx.Rng.randint ~key ~high:1000 [| mib |]));
      Thumper.bench_with_setup "bernoulli-f32-1M"
        ~setup:(keyed [| mib |] 0.3)
        (fun (key, p) -> draw (Nx.Rng.bernoulli ~key p));
      Thumper.bench_with_setup "bernoulli-f32-1Kx1K"
        ~setup:(keyed [| 1024; 1024 |] 0.3)
        (fun (key, p) -> draw (Nx.Rng.bernoulli ~key p));
      Thumper.bench_with_setup "gamma-f32-64K"
        ~setup:(keyed [| 1 lsl 16 |] 4.)
        (fun (key, rates) -> draw (Nx.Rng.gamma ~key rates));
      Thumper.bench_with_setup "poisson-f32-64K"
        ~setup:(keyed [| 1 lsl 16 |] 4.)
        (fun (key, rates) -> draw (Nx.Rng.poisson ~key rates));
      Thumper.bench_with_setup "normal-f32-1" ~setup:key (fun key ->
          draw (Nx.Rng.normal ~key D.Float32 [| 1 |]));
      Thumper.bench_with_setup "beta-f32-1" ~setup:(keyed [| 1 |] 2.)
        (fun (key, one) -> draw (Nx.Rng.beta ~key one one));
    ]

(* Contractions, each beside its kernel called directly, so that the
   frontend's cost over it shows: squares, from 4³, summed in lane order,
   through those a call's fixed cost and its choice of hardware decide, and a
   decode step's one row against a weight stored as [n × k]. *)

let matrix dt s = filled dt s 1.

(* Operands of [dt] filled with [v], of shapes [sa] and [sb]. *)
let pair dt sa sb v () = (filled dt sa v, filled dt sb v)
let matmul (a, b) = Nx.Repr.array (Nx.matmul a b)

(* [make]'s operands as arrays, and the product of those into a new [s]
   float32 array by nx.cpu's kernel. *)
let direct_operands make () =
  let a, b = make () in
  let plain x : Nx_kernel.operand =
    { array = A.Any (Option.get (Nx.Repr.array x)); scale = None }
  in
  (plain a, plain b)

let contract_direct ~contracting s (a, b) =
  let dst = A.create Rig.host D.Float32 s in
  match
    Nx_cpu.contract ~batch:[||] ~contracting ~segments:None
      ~acc:(D.Any D.Float32) ~dst ~init:None a b
  with
  | Done -> dst
  | refusal -> A.refused "contract-direct" refusal [ A.Any dst ]

let direct name ~contracting s make =
  Thumper.bench_with_setup name
    ~setup:(direct_operands make)
    (contract_direct ~contracting s)

let matmul_rows n =
  let operands = pair D.Float32 [| n; n |] [| n; n |] 1. in
  let name = Printf.sprintf "matmul-f32-%d" n in
  [
    Thumper.bench_with_setup name ~setup:operands matmul;
    direct (name ^ "-direct") ~contracting:[| (1, 0) |] [| n; n |] operands;
  ]

(* Nx.dot of two matrices, the product Nx.matmul times beside it. *)
let dot_rows =
  [
    Thumper.bench_with_setup "dot-f32-8"
      ~setup:(pair D.Float32 [| 8; 8 |] [| 8; 8 |] 1.)
      (fun (a, b) -> Nx.Repr.array (Nx.dot a b));
  ]

let rows_by_weight = Nx.Pattern.v "m k, n k -> m n | k"
let einsum p (a, b) = Nx.Repr.array (Nx.einsum p [ a; b ])

let decode_rows =
  let f32 = pair D.Float32 [| 1; 4096 |] [| 4096; 4096 |] 1. in
  [
    Thumper.bench_with_setup "einsum-f32-m1x4096x4096" ~setup:f32
      (einsum rows_by_weight);
    direct "einsum-f32-m1x4096x4096-direct" ~contracting:[| (1, 1) |]
      [| 1; 4096 |] f32;
    Thumper.bench_with_setup "einsum-bf16-m1x5120x2880"
      ~setup:(pair D.Bfloat16 [| 1; 2880 |] [| 5120; 2880 |] 1.)
      (einsum rows_by_weight);
    (* A transposed bfloat16 operand, which the product stages strided. *)
    Thumper.bench_with_setup "matmul-bf16-512-tn"
      ~setup:(pair D.Bfloat16 [| 512; 512 |] [| 512; 512 |] 1.)
      (fun (a, b) -> Nx.Repr.array (Nx.matmul (Nx.transpose a) b));
  ]

(* A matrix, a matrix and a vector: a (b v) does two matrix-vector products,
   (a b) v a matrix product first. *)
let chain_rows =
  let mmv = Nx.Pattern.v "i j, j k, k -> i | j k" in
  [
    Thumper.bench_with_setup "einsum-f32-mmv-1024"
      ~setup:(fun () ->
        [
          matrix D.Float32 [| 1024; 1024 |];
          matrix D.Float32 [| 1024; 1024 |];
          matrix D.Float32 [| 1024 |];
        ])
      (fun xs -> Nx.Repr.array (Nx.einsum mmv xs));
  ]

(* Attention's scores over heads stored [b; s; h; d], as a projection
   leaves them, read as [b; h; s; d]: their batch steps apart, so the
   product loops over b outside its batch of heads; and the same product
   on heads stored contiguous. *)
let attention_rows =
  let scores = Nx.Pattern.v "b h i d, b h j d -> b h i j | d" in
  let heads () =
    Nx.transpose ~axes:[ 0; 2; 1; 3 ] (matrix D.Float32 [| 2; 128; 8; 64 |])
  in
  [
    Thumper.bench_with_setup "einsum-f32-attention-2x8x128x64"
      ~setup:(fun () -> (heads (), heads ()))
      (einsum scores);
    Thumper.bench_with_setup "einsum-f32-attention-2x8x128x64-contiguous"
      ~setup:(fun () -> (Nx.copy (heads ()), Nx.copy (heads ())))
      (einsum scores);
  ]

(* Planning at the most operands einsum takes: [n] one-element operands in a
   ring, each sharing a summed name with the two before it and the two after,
   so that every split of a set meets names. *)
let plan_row n =
  let ring i = (i + n) mod n in
  let name i d = Printf.sprintf "r%d_%d" i d in
  let layout i =
    String.concat " "
      [ name i 1; name i 2; name (ring (i - 1)) 1; name (ring (i - 2)) 2 ]
  in
  let sums = List.concat (List.init n (fun i -> [ name i 1; name i 2 ])) in
  let p =
    Nx.Pattern.v
      (String.concat ", " (List.init n layout)
      ^ " -> | " ^ String.concat " " sums)
  in
  let operands () =
    let x = matrix D.Float32 [| 1; 1; 1; 1 |] in
    List.init n (fun _ -> x)
  in
  Thumper.bench_with_setup (Printf.sprintf "einsum-plan-%d" n)
    ~setup:operands (fun xs -> Nx.Repr.array (Nx.einsum p xs))

(* A weight gradient's product: 16 × 16 outputs over 65536 rows of
   activations read transposed, few outputs over a long depth, which the
   threads share. *)
let gradient_rows =
  [
    Thumper.bench_with_setup "matmul-f32-tn-16x16x65536"
      ~setup:(pair D.Float32 [| 65536; 16 |] [| 65536; 16 |] 1.)
      (fun (x, dy) -> Nx.Repr.array (Nx.matmul (Nx.transpose x) dy));
  ]

(* Products at the dtypes beside float32. *)
let square_rows (type v s) name (dt : (v, s) D.t) (v : v) sizes =
  List.map
    (fun n ->
      Thumper.bench_with_setup
        (Printf.sprintf "matmul-%s-%d" name n)
        ~setup:(pair dt [| n; n |] [| n; n |] v)
        matmul)
    sizes

(* 4 rows by a 64 × 64 matrix, which lane order sums in the target's tile. *)
let tile_rows =
  let f32 = pair D.Float32 [| 4; 64 |] [| 64; 64 |] 1. in
  [
    Thumper.bench_with_setup "matmul-f32-4x64x64" ~setup:f32 matmul;
    direct "matmul-f32-4x64x64-direct" ~contracting:[| (1, 0) |] [| 4; 64 |]
      f32;
    Thumper.bench_with_setup "matmul-f64-4x64x64"
      ~setup:(pair D.Float64 [| 4; 64 |] [| 64; 64 |] 1.)
      matmul;
  ]

(* Few rows on several threads, b's slivers fewer than the threads (24 ×
   24 × 512) or not a multiple of them (64³ float64 on six cores). *)
let split_rows =
  [
    Thumper.bench_with_setup "matmul-f32-24x24x512"
      ~setup:(pair D.Float32 [| 24; 512 |] [| 512; 24 |] 1.)
      matmul;
    Thumper.bench_with_setup "matmul-f64-64"
      ~setup:(fun () -> filled D.Float64 [| 64; 64 |] 1.)
      (fun x -> matmul (x, x));
  ]

let rows_by_columns = Nx.Pattern.v "m k, k n -> m n | k"
let weight_by_vector = Nx.Pattern.v "n k, k -> n | k"
let widened p (a, b) = Nx.Repr.array (Nx.contract ~acc:D.Int32 D.Int32 p a b)

(* int8 operands into an int32 accumulator, as quantized weights are: squares,
   a decode step and a prefill of 128 rows against a weight stored [n × k],
   a decode step against a weight stored [k × n] read from memory, a call
   each of six weights in turn (96 MiB, past the M1 Max's 48 MB system
   cache) as a model's layers are, 8 rows against the first weight stored
   [k × n], and the first weight by a vector; and int32's prefill, past a
   core's L2. *)
let widened_rows =
  List.map
    (fun n ->
      Thumper.bench_with_setup
        (Printf.sprintf "contract-i8-i32-%d" n)
        ~setup:(pair D.Int8 [| n; n |] [| n; n |] 1)
        (widened rows_by_columns))
    [ 64; 256; 1024 ]
  @
  let weights () =
    let w () = filled D.Int8 [| 4096; 4096 |] 1 in
    (filled D.Int8 [| 1; 4096 |] 1, Array.init 6 (fun _ -> w ()), ref 0)
  in
  [
    Thumper.bench_with_setup "contract-i8-i32-m1x4096x4096"
      ~setup:(pair D.Int8 [| 1; 4096 |] [| 4096; 4096 |] 1)
      (widened rows_by_weight);
    Thumper.bench_with_setup "contract-i8-i32-m128x4096x4096"
      ~setup:(pair D.Int8 [| 128; 4096 |] [| 4096; 4096 |] 1)
      (widened rows_by_weight);
    Thumper.bench_with_setup "contract-i32-m128x4096x4096"
      ~setup:(pair D.Int32 [| 128; 4096 |] [| 4096; 4096 |] 1l)
      (widened rows_by_weight);
    Thumper.bench_with_setup "contract-i8-i32-m1x4096x4096-kn-memory"
      ~setup:weights (fun (x, ws, turn) ->
        turn := (!turn + 1) mod Array.length ws;
        widened rows_by_columns (x, ws.(!turn)));
    Thumper.bench_with_setup "contract-i8-i32-m8x4096x4096-kn"
      ~setup:(pair D.Int8 [| 8; 4096 |] [| 4096; 4096 |] 1)
      (widened rows_by_columns);
    Thumper.bench_with_setup "contract-i8-i32-n1x4096x4096"
      ~setup:(pair D.Int8 [| 4096; 4096 |] [| 4096 |] 1)
      (widened weight_by_vector);
  ]

(* A complex vector by a matrix: the lane order's path where b, the larger
   operand, is copied into its blocks. *)
let vector_rows =
  [
    Thumper.bench_with_setup "matmul-c64-1x2048x2048"
      ~setup:(pair D.Complex64 [| 2048 |] [| 2048; 2048 |] Complex.one)
      matmul;
  ]

(* Squares of each dtype; bfloat16's are summed in float32 and rounded
   once into a bfloat16 result, as int8's in int32 into int8. *)
let dtype_rows =
  square_rows "bf16" D.Bfloat16 1. [ 64; 256; 1024 ]
  @ square_rows "c64" D.Complex64 Complex.one [ 64; 256; 1024 ]
  @ square_rows "c128" D.Complex128 Complex.one [ 64; 256; 1024 ]
  @ vector_rows
  @ square_rows "i32" D.Int32 1l [ 64; 256; 1024 ]
  @ square_rows "i64" D.Int64 1L [ 64; 256 ]
  @ square_rows "i8" D.Int8 1 [ 64; 256; 1024 ]
  @ widened_rows

let contract_rows =
  Thumper.group "contract"
    (matmul_rows 4 @ matmul_rows 8 @ matmul_rows 16 @ matmul_rows 32
   @ matmul_rows 64 @ matmul_rows 1024 @ dot_rows @ tile_rows @ split_rows
   @ decode_rows @ chain_rows @ attention_rows @ gradient_rows
   @ [ plan_row 11 ] @ dtype_rows)

(* Convolution and pooling over images of [8; 56; 56; 64]
   float32 channels last, as a user writes each today: a 3 × 3 convolution
   of 64 channels padded by one, in float32 and bfloat16, and into 3
   channels, as an image's last layer, which sums in lane order, as do an
   RGB image's into 3 and a complex64 one's into 4; its input gradient,
   conv_transpose, and its weight gradient, a contraction over
   the padded image's windows; the depthwise convolution, one channel a
   group; a 3 × 3 max-pool at stride 2 of the image channels first,
   padded by -∞, as pad, two window views and max; and its gradient, as
   those and where: the windows holding the maximum take the output's
   gradient, folded back onto the image. *)

let image dt = filled dt [| 8; 56; 56; 64 |] 1.
let pad1 = [| (1, 1); (1, 1) |]
let around = [| (0, 0); (1, 1); (1, 1); (0, 0) |]

let windows ~step x =
  Nx.sliding_window ~axis:2 ~window:3 ~step
    (Nx.sliding_window ~axis:1 ~window:3 ~step x)

let conv_by dt s () = (image dt, filled dt s 1.)

let conv ?groups (x, w) = Nx.Repr.array (Nx.conv ?groups ~padding:pad1 x w)

let weight_grad =
  let p = Nx.Pattern.v "n h w c i j, n h w o -> o i j c | n h w" in
  fun (x, dy) ->
    let v = windows ~step:1 (Nx.pad around 0. x) in
    Nx.Repr.array (Nx.contract D.Float32 p v dy)

let max_pool x =
  let x = Nx.pad [| (0, 0); (0, 0); (1, 1); (1, 1) |] Float.neg_infinity x in
  let v =
    Nx.sliding_window ~axis:3 ~window:3 ~step:2
      (Nx.sliding_window ~axis:2 ~window:3 ~step:2 x)
  in
  Nx.Repr.array (Nx.max ~axes:[ 4; 5 ] v)

(* [[n; c; 28; 28; 3; 3]] windows to [[n; c; 9; 784]] patches. *)
let max_pool_grad (x, dy) =
  let x = Nx.pad [| (0, 0); (0, 0); (1, 1); (1, 1) |] Float.neg_infinity x in
  let v =
    Nx.sliding_window ~axis:3 ~window:3 ~step:2
      (Nx.sliding_window ~axis:2 ~window:3 ~step:2 x)
  in
  let m = Nx.max ~axes:[ 4; 5 ] ~keepdims:true v in
  let dy = Nx.reshape [| 8; 64; 28; 28; 1; 1 |] dy in
  let g =
    Nx.where (Nx.equal v m) (Nx.broadcast_to (Nx.shape v) dy) (Nx.zeros_like v)
  in
  let p =
    Nx.reshape [| 8; 64; 9; 28 * 28 |]
      (Nx.transpose ~axes:[ 0; 1; 4; 5; 2; 3 ] g)
  in
  Nx.Repr.array
    (Nx.combine_patches ~output_size:[| 56; 56 |] ~kernel_size:[| 3; 3 |]
       ~stride:[| 2; 2 |] ~dilation:[| 1; 1 |] ~padding:pad1 p)

let conv_rows =
  Thumper.group "conv"
    [
      Thumper.bench_with_setup "conv-f32-8x56x56x64-3x3"
        ~setup:(conv_by D.Float32 [| 64; 3; 3; 64 |]) conv;
      Thumper.bench_with_setup "conv-f32-8x56x56x64-3x3-to-3"
        ~setup:(conv_by D.Float32 [| 3; 3; 3; 64 |]) conv;
      Thumper.bench_with_setup "conv-f32-8x112x112x3-3x3-to-3"
        ~setup:(fun () ->
          (filled D.Float32 [| 8; 112; 112; 3 |] 1., filled D.Float32 [| 3; 3; 3; 3 |] 1.))
        conv;
      Thumper.bench_with_setup "conv-c64-4x32x32x64-3x3-to-4"
        ~setup:(fun () ->
          (filled D.Complex64 [| 4; 32; 32; 64 |] Complex.one,
           filled D.Complex64 [| 4; 3; 3; 64 |] Complex.one))
        conv;
      Thumper.bench_with_setup "conv-bf16-8x56x56x64-3x3"
        ~setup:(conv_by D.Bfloat16 [| 64; 3; 3; 64 |]) conv;
      Thumper.bench_with_setup "conv-input-grad-f32-8x56x56x64-3x3"
        ~setup:(conv_by D.Float32 [| 64; 3; 3; 64 |]) (fun (dy, w) ->
          Nx.Repr.array (Nx.conv_transpose ~padding:pad1 dy w));
      Thumper.bench_with_setup "conv-weight-grad-f32-8x56x56x64-3x3"
        ~setup:(fun () -> (image D.Float32, image D.Float32)) weight_grad;
      Thumper.bench_with_setup "depthwise-f32-8x56x56x64-3x3"
        ~setup:(conv_by D.Float32 [| 64; 3; 3; 1 |]) (conv ~groups:64);
      Thumper.bench_with_setup "max-pool-f32-8x64x56x56-3x3s2"
        ~setup:(fun () -> filled D.Float32 [| 8; 64; 56; 56 |] 1.)
        max_pool;
      Thumper.bench_with_setup "max-pool-grad-f32-8x64x56x56-3x3s2"
        ~setup:(fun () ->
          (filled D.Float32 [| 8; 64; 56; 56 |] 1.,
           filled D.Float32 [| 8; 64; 28; 28 |] 1.))
        max_pool_grad;
    ]

(* Indexing: a row take reads its rows, an add-scatter its updates at
   positions drawn at random, as a histogram's or a gradient's land, a set
   writes its row into a copy of the cache, and concatenate copies its result
   once. *)

let take_1k () =
  let ids =
    A.of_array D.Int64 [| 1024 |]
      (Array.init 1024 (fun i -> Int64.of_int (i * 7 mod 8192)))
  in
  (Nx.Repr.of_array Nx.Host.v ids, filled D.Float32 [| 8192; 256 |] 1.)

(* 64K positions in 1024 targets, drawn once from a fixed key, their
   updates and the targets. *)
let scatter_64k () =
  let ids = Nx.Rng.randint ~key:(Nx.Rng.key 10) ~high:1024 [| 65536 |] in
  ( Nx.place Nx.Host.on (Nx.cast D.Int64 ids),
    host 65536,
    Nx.zeros D.Float32 [| 1024 |] )

(* A cache of 8 heads, 4096 positions and 64 features, 8 MiB, one
   position's row and the position. *)
let set_8m () =
  let cache =
    Nx.add
      (filled D.Float32 [| 1; 8; 4096; 64 |] 0.)
      (Nx.zeros D.Float32 [| 1; 8; 4096; 64 |])
  in
  let pos = Nx.Repr.of_array Nx.Host.v (A.of_array D.Int64 [||] [| 1000L |]) in
  (cache, filled D.Float32 [| 1; 8; 1; 64 |] 1., pos)

(* An embedding's gradient, the adjoint of a take of its rows: the rows of
   8192 tokens, 4096 wide, added into a table of 128256 rows, each call into
   the last call's table, donated. The ids are drawn once from a fixed seed,
   uniform, or by a Zipf law of exponent 1 as words in a text fall, so that
   a few rows take most of the tokens. *)
let vocab = 128256
let tokens = 8192
let width = 4096

let uniform_ids () =
  let rs = Random.State.make [| 7 |] in
  Array.init tokens (fun _ -> Int64.of_int (Random.State.int rs vocab))

(* Rank k weighs 1 / (k + 1): each id is the first rank whose cumulative
   weight reaches a uniform draw of the total. *)
let zipf_ids () =
  let rs = Random.State.make [| 7 |] in
  let cdf = Array.make vocab 0. in
  let total = ref 0. in
  for k = 0 to vocab - 1 do
    total := !total +. (1. /. float (k + 1));
    cdf.(k) <- !total
  done;
  Array.init tokens (fun _ ->
      let u = Random.State.float rs !total in
      let lo = ref 0 and hi = ref (vocab - 1) in
      while !lo < !hi do
        let mid = (!lo + !hi) / 2 in
        if cdf.(mid) < u then lo := mid + 1 else hi := mid
      done;
      Int64.of_int !lo)

let embedding_grad dt ids () =
  let ids = Nx.Repr.of_array Nx.Host.v (A.of_array D.Int64 [| tokens |] (ids ())) in
  let p = Nx.broadcast_to [| tokens; width |] (Nx.reshape [| tokens; 1 |] ids) in
  let g =
    on_host (Nx.Rng.uniform ~key:(Nx.Rng.key 11) dt [| tokens; width |])
  in
  let table = Nx.copy (on_host (Nx.zeros dt [| vocab; width |])) in
  (p, g, ref table)

let add_rows (p, g, table) =
  table := Nx.scatter ~combine:Add ~axis:0 p g (Nx.donate !table)

let embedding_rows =
  List.concat_map
    (fun (name, ids) ->
      [
        Thumper.bench_with_setup ("embedding-grad-bf16-" ^ name)
          ~setup:(embedding_grad D.Bfloat16 ids)
          add_rows;
        Thumper.bench_with_setup ("embedding-grad-f32-" ^ name)
          ~setup:(embedding_grad D.Float32 ids)
          add_rows;
      ])
    [ ("uniform", uniform_ids); ("zipf", zipf_ids) ]

(* The gradient of a take along axis 0 of a bfloat16 table of [rows] rows
   of [cols], a position drawn uniformly for each element: its updates
   added each into its own column, each call into the last call's table,
   donated. Of 256K rows of 64, the columns are more than the threads; of
   1M rows of 3, fewer, so that threads split each column's targets. *)
let take_along_grad rows cols () =
  let s = [| rows; cols |] in
  let p = Nx.Rng.randint ~key:(Nx.Rng.key 12) ~high:rows s in
  let g = on_host (Nx.Rng.uniform ~key:(Nx.Rng.key 11) D.Bfloat16 s) in
  let table = Nx.copy (on_host (Nx.zeros D.Bfloat16 s)) in
  (on_host (Nx.cast D.Int64 p), g, ref table)

let index_rows =
  Thumper.group "index"
    ([
      Thumper.bench_with_setup "take-rows-1K-of-8K" ~setup:take_1k
        (fun (ids, table) -> Nx.take ~axis:0 ids table);
      Thumper.bench_with_setup "scatter-add-64K-into-1K" ~setup:scatter_64k
        (fun (ids, updates, sums) ->
          Nx.scatter ~combine:Add ~axis:0 ids updates sums);
      Thumper.bench_with_setup "set-row-8M" ~setup:set_8m
        (fun (cache, row, pos) -> Nx.set Nx.[ A; A; D (pos, 1); A ] row cache);
      Thumper.bench_with_setup "concatenate-2x512K"
        ~setup:(fun () -> host (1 lsl 19))
        (fun x -> Nx.concatenate ~axis:0 [ x; x ]);
      Thumper.bench_with_setup "take-along-grad-bf16-256K-by-64"
        ~setup:(take_along_grad (1 lsl 18) 64)
        add_rows;
      Thumper.bench_with_setup "take-along-grad-bf16-1M-by-3"
        ~setup:(take_along_grad (1 lsl 20) 3)
        add_rows;
    ]
    @ embedding_rows)

(* Interpretations. Each row adds one-element host values 100 times: eagerly;
   under a Values interpretation that does not reach them; while an Extent lives
   on another domain, and while one lives on this domain around another fiber,
   either of which costs each add a perform; on a traced value, delivered to its
   interpretation's rule; and on an outer interpretation's traced value under an
   inner Extent, whose receiver asks each add whether the outer is around it. *)

let adds x =
  for _ = 1 to 100 do
    ignore (Sys.opaque_identity (Nx.add x x))
  done

type ('v, 's, 'd) Nx.Prim.payload += Traced : ('v, 's, 'd) Nx.Prim.payload

let tracing i ~by op =
  Nx.Prim.results ~by (fun _ form -> Nx.Prim.traced i form Traced) op

(* An Extent on another domain, live until [end_elsewhere], and a value to
   add. *)
let extent_elsewhere () =
  let started = Atomic.make false and stop = Atomic.make false in
  let d =
    Domain.spawn (fun () ->
        Nx.Prim.interpret ~name:"bench" Extent tracing (fun _ ->
            Atomic.set started true;
            while not (Atomic.get stop) do
              Domain.cpu_relax ()
            done))
  in
  while not (Atomic.get started) do
    Domain.cpu_relax ()
  done;
  ((stop, d), one ())

let end_elsewhere ((stop, d), _) =
  Atomic.set stop true;
  Domain.join d

(* An Extent on this domain, suspended in its own fiber until [end_here], and
   a value to add. *)
type _ Effect.t += Suspend : unit Effect.t

let extent_here () =
  let k = ref None in
  Effect.Deep.match_with
    (fun () ->
      Nx.Prim.interpret ~name:"bench" Extent tracing (fun _ ->
          Effect.perform Suspend))
    ()
    {
      retc = Fun.id;
      exnc = raise;
      effc =
        (fun (type a) (e : a Effect.t) ->
          match e with
          | Suspend ->
              Some
                (fun (c : (a, unit) Effect.Deep.continuation) ->
                  k := Some (fun () -> Effect.Deep.continue c ()))
          | _ -> None);
    };
  (k, one ())

let end_here (k, _) = Option.iter (fun resume -> resume ()) !k

let interp_rows =
  Thumper.group "interp"
    [
      Thumper.bench_with_setup "add-1-100-eager" ~setup:one adds;
      Thumper.bench_with_setup "add-1-100-under-values" ~setup:one (fun x ->
          Nx.Prim.interpret ~name:"bench" Values tracing (fun _ -> adds x));
      Thumper.bench_with_setup "add-1-100-beside-extent-on-another-domain"
        ~setup:extent_elsewhere ~teardown:end_elsewhere (fun (_, x) -> adds x);
      Thumper.bench_with_setup "add-1-100-fiber-beside-extent"
        ~setup:extent_here ~teardown:end_here (fun (_, x) -> adds x);
      Thumper.bench_with_setup "traced-add-1-100" ~setup:one (fun x ->
          Nx.Prim.interpret ~name:"bench" Values tracing (fun i ->
              let t = Nx.Prim.traced i (Nx.Prim.form x) Traced in
              for _ = 1 to 100 do
                ignore (Sys.opaque_identity (Nx.add t t))
              done));
      Thumper.bench_with_setup "traced-add-1-100-under-inner-extent" ~setup:one
        (fun x ->
          Nx.Prim.interpret ~name:"bench.outer" Values tracing (fun o ->
              let t = Nx.Prim.traced o (Nx.Prim.form x) Traced in
              Nx.Prim.interpret ~name:"bench.inner" Extent tracing (fun _ ->
                  for _ = 1 to 100 do
                    ignore (Sys.opaque_identity (Nx.add t t))
                  done)));
    ]

(* The frontend's compositions over 1M host elements, each what a user calls:
   creation of a leaf (placed on the host, which computes it), the crossings
   to OCaml values, the scalar forms and the functions nx composes from the
   kernels' kinds. *)

let n1m = 1 lsl 20

let ocaml_floats () =
  Array.init n1m (fun i -> Float.of_int (i mod 1000) /. 1000.)

let f1m () = Nx.create D.Float32 [| n1m |] (ocaml_floats ())
let above1 f = on_host (Nx.add_s f 1.)

(* [f1m] and [above1] of it, for the binary rows. *)
let f1m_above () =
  let f = f1m () in
  (f, above1 f)

let above1m () = above1 (f1m ())
let i1m () = on_host (Nx.arange D.Int32 (-(n1m / 2)) (n1m / 2) 1)
let b1m () = on_host (Nx.less_s (f1m ()) 0.5)

let c1m () =
  let re, im = f1m_above () in
  on_host (Nx.complex D.Complex64 ~re ~im)

let h1m () = on_host (Nx.cast D.Bfloat16 (f1m ()))
let m1024 () = Nx.reshape [| 1024; 1024 |] (f1m ())

let m1024x4096 () =
  let f = f1m () in
  let four = Nx.concatenate ~axis:0 [ f; f; f; f ] in
  on_host (Nx.reshape [| 1024; 4096 |] four)

(* A norm's batch: 8192 rows of 4096, 128 MiB of float32, past the
   caches. *)
let m8192x4096 () =
  let f = f1m () in
  let rows = Nx.concatenate ~axis:0 (List.init 32 (fun _ -> f)) in
  on_host (Nx.reshape [| 8192; 4096 |] rows)

let h8192x4096 () = on_host (Nx.cast D.Bfloat16 (m8192x4096 ()))

let m48 () = Nx.reshape [| 48; 1024 |] (Nx.slice [ R (0, 48 * 1024) ] (f1m ()))

(* Logits: 32 rows of a 32000-word vocabulary, and 8 rows of 64. *)
let logits () =
  Nx.reshape [| 32; 32000 |] (Nx.slice [ R (0, 32 * 32000) ] (f1m ()))

let small_logits () = Nx.reshape [| 8; 64 |] (Nx.slice [ R (0, 512) ] (f1m ()))

(* A batch of 64 rows of a 128256-word vocabulary's logits, 32.8 MB of
   float32. *)
let vocab () =
  let f = f1m () in
  let eight = Nx.concatenate ~axis:0 (List.init 8 (fun _ -> f)) in
  on_host (Nx.reshape [| 64; 128256 |] (Nx.slice [ R (0, 64 * 128256) ] eight))
let v1024 () = Nx.reshape [| 1024 |] (Nx.slice [ R (0, 1024) ] (f1m ()))
let p1000 () = Nx.slice [ R (0, 1000) ] (f1m ())

let frontend_rows =
  Thumper.group "frontend"
    [
      Thumper.bench_with_setup "create-f32-1M" ~setup:ocaml_floats
        (Nx.create D.Float32 [| n1m |]);
      Thumper.bench_with_setup "to_array-f32-1M" ~setup:f1m Nx.to_array;
      Thumper.bench_with_setup "item-f32-1M" ~setup:f1m (Nx.item [ 524_287 ]);
      Thumper.bench "arange-f32-1M" (fun () ->
          on_host (Nx.arange D.Float32 0 (Thumper.black_box n1m) 1));
      Thumper.bench "linspace-f32-1M" (fun () ->
          on_host (Nx.linspace D.Float32 0. 1. (Thumper.black_box n1m)));
      Thumper.bench "eye-f32-1024" (fun () ->
          on_host (Nx.eye D.Float32 (Thumper.black_box 1024)));
      Thumper.bench_with_setup "full_like-f32-1M" ~setup:f1m (fun x ->
          Nx.full_like x 2.);
      Thumper.bench_with_setup "add_s-f32-1M" ~setup:f1m (fun x ->
          Nx.add_s x 1.);
      Thumper.bench_with_setup "less_s-f32-1M" ~setup:f1m (fun x ->
          Nx.less_s x 0.5);
      Thumper.bench_with_setup "logical_and-f32-1M" ~setup:f1m (fun x ->
          Nx.logical_and x x);
      Thumper.bench_with_setup "logical_and-bool-1M" ~setup:b1m (fun x ->
          Nx.logical_and x x);
      Thumper.bench_with_setup "isnan-f32-1M" ~setup:f1m Nx.isnan;
      Thumper.bench_with_setup "isfinite-f32-1M" ~setup:f1m Nx.isfinite;
      Thumper.bench_with_setup "clamp-f32-1M" ~setup:f1m
        (Nx.clamp ~min:0.25 ~max:0.75);
      Thumper.bench_with_setup "lshift-i32-1M" ~setup:i1m (fun x ->
          Nx.lshift x 3);
      Thumper.bench_with_setup "rshift-i32-1M" ~setup:i1m (fun x ->
          Nx.rshift x 3);
      Thumper.bench_with_setup "bitwise_not-i32-1M" ~setup:i1m Nx.bitwise_not;
      Thumper.bench_with_setup "square-f32-1M" ~setup:f1m Nx.square;
      Thumper.bench_with_setup "rsqrt-f32-1M" ~setup:above1m Nx.rsqrt;
      Thumper.bench_with_setup "hypot-f32-1M" ~setup:f1m_above (fun (x, y) ->
          Nx.hypot x y);
      Thumper.bench_with_setup "asinh-f32-1M" ~setup:f1m Nx.asinh;
      Thumper.bench_with_setup "acosh-f32-1M" ~setup:above1m Nx.acosh;
      Thumper.bench_with_setup "atanh-f32-1M" ~setup:f1m Nx.atanh;
      Thumper.bench_with_setup "real-c64-1M" ~setup:c1m (Nx.real D.Float32);
      Thumper.bench_with_setup "magnitude-c64-1M" ~setup:c1m
        (Nx.magnitude D.Float32);
      Thumper.bench_with_setup "angle-c64-1M" ~setup:c1m (Nx.angle D.Float32);
      Thumper.bench_with_setup "complex-c64-1M" ~setup:f1m_above
        (fun (re, im) -> Nx.complex D.Complex64 ~re ~im);
      Thumper.bench_with_setup "conjugate-c64-1M" ~setup:c1m Nx.conjugate;
      Thumper.bench_with_setup "tril-f32-1024" ~setup:m1024 Nx.tril;
      Thumper.bench_with_setup "trace-f32-1024" ~setup:m1024 Nx.trace;
      Thumper.bench_with_setup "diag-f32-1024" ~setup:v1024 Nx.diag;
      Thumper.bench_with_setup "vdot-f32-1M" ~setup:f1m (fun x -> Nx.vdot x x);
      Thumper.bench_with_setup "to_string-f32-1000" ~setup:p1000 Nx.to_string;
    ]

(* Reductions and activations, each against its NumPy, SciPy or pandas
   counterpart at the same size: 1M float32 elements, rows of 1024; softmax
   also over a vocabulary's logits and a few short rows; softmax over a
   batch of 128256-word logits and RMS norm over a norm's batch, against
   PyTorch and llama.cpp, measured outside the repo. *)
let statistics_rows =
  Thumper.group "statistics"
    [
      Thumper.bench_with_setup "all-bool-1M" ~setup:b1m Nx.all;
      Thumper.bench_with_setup "count-bool-1M" ~setup:b1m Nx.count;
      Thumper.bench_with_setup "argmax-f32-1M" ~setup:f1m Nx.argmax;
      Thumper.bench_with_setup "argmax-f32-1024x1024-rows" ~setup:m1024
        (Nx.argmax ~axis:1);
      Thumper.bench_with_setup "var-f32-1M" ~setup:f1m Nx.var;
      Thumper.bench_with_setup "std-f32-1024x1024-rows" ~setup:m1024
        (Nx.std ~axes:[ 1 ]);
      Thumper.bench_with_setup "var-f32-8192x4096-rows" ~setup:m8192x4096
        (Nx.var ~axes:[ 1 ]);
      Thumper.bench_with_setup "var-bf16-8192x4096-rows" ~setup:h8192x4096
        (Nx.var ~axes:[ 1 ]);
      Thumper.bench_with_setup "logsumexp-f32-1M" ~setup:f1m Nx.logsumexp;
      (* Down columns, a reduced axis whose terms are far apart. *)
      Thumper.bench_with_setup "argmax-f32-1024x1024-cols" ~setup:m1024
        (Nx.argmax ~axis:0);
      Thumper.bench_with_setup "var-f32-1024x1024-cols" ~setup:m1024
        (Nx.var ~axes:[ 0 ]);
      Thumper.bench_with_setup "logsumexp-f32-1024x1024-cols" ~setup:m1024
        (Nx.logsumexp ~axes:[ 0 ]);
      Thumper.bench_with_setup "sum-c64-1M" ~setup:c1m Nx.sum;
      Thumper.bench_with_setup "cumsum-c64-1M" ~setup:c1m Nx.cumsum;
      Thumper.bench_with_setup "cumsum-f32-1024x4096-axis0" ~setup:m1024x4096
        (Nx.cumsum ~axis:0);
      Thumper.bench_with_setup "cummax-f32-1024x4096-axis0" ~setup:m1024x4096
        (Nx.cummax ~axis:0);
      Thumper.bench_with_setup "cumsum-f32-48x1024-axis0" ~setup:m48
        (Nx.cumsum ~axis:0);
      Thumper.bench_with_setup "sum-bf16-1M" ~setup:h1m Nx.sum;
      Thumper.bench_with_setup "max-bf16-1024x1024-cols" ~setup:h1m (fun x ->
          Nx.max ~axes:[ 0 ] (Nx.reshape [| 1024; 1024 |] x));
      Thumper.bench_with_setup "cumsum-bf16-1M" ~setup:h1m Nx.cumsum;
      Thumper.bench_with_setup "argmax-bf16-1M" ~setup:h1m Nx.argmax;
      Thumper.bench_with_setup "argmax-bool-1M" ~setup:b1m Nx.argmax;
      Thumper.bench_with_setup "logsumexp-bf16-1M" ~setup:h1m Nx.logsumexp;
      Thumper.bench_with_setup "var-bf16-1M" ~setup:h1m Nx.var;
      Thumper.bench_with_setup "sigmoid-f32-1M" ~setup:f1m Nx.sigmoid;
      Thumper.bench_with_setup "softplus-f32-1M" ~setup:f1m Nx.softplus;
      Thumper.bench_with_setup "softmax-f32-1024x1024" ~setup:m1024 Nx.softmax;
      Thumper.bench_with_setup "log_softmax-f32-1024x1024" ~setup:m1024
        Nx.log_softmax;
      Thumper.bench_with_setup "softmax-f32-32x32000" ~setup:logits Nx.softmax;
      Thumper.bench_with_setup "log_softmax-f32-32x32000" ~setup:logits
        Nx.log_softmax;
      Thumper.bench_with_setup "softmax-f32-8x64" ~setup:small_logits
        Nx.softmax;
      Thumper.bench_with_setup "softmax-f32-64x128256" ~setup:vocab Nx.softmax;
      Thumper.bench_with_setup "rms_norm-f32-8192x4096" ~setup:m8192x4096
        Nx.rms_norm;
      Thumper.bench_with_setup "log_softmax-f32-8x64" ~setup:small_logits
        Nx.log_softmax;
      Thumper.bench_with_setup "ewma-f32-1M" ~setup:f1m (Nx.ewma ~alpha:0.1);
      Thumper.bench_with_setup "ewma-f32-1024x1024-rows" ~setup:m1024
        (Nx.ewma ~axis:1 ~alpha:0.1);
    ]

let scan ?(axis = 1) reduction x =
  let dt = Nx.dtype x in
  Nx.Prim.eval ~by:"bench"
    (Scan
       {
         layout = A.Layout.contiguous (Iarray.of_array (Nx.shape x));
         axis;
         prog = Nx_kernel.Prog.v ~ins:[| D.Any dt |] [| In 0 |] ~outs:[| 0 |];
         reduction;
         loads = [| Plain x |];
         carry = None;
       })

let scan_rows =
  Thumper.group "scan"
    [
      (* Rows side by side, and one slice's chunks. *)
      Thumper.bench_with_setup "host-logsumexp-f32-1024x1024-rows"
        ~setup:m1024
        (scan (Logsumexp (0, D.Float32)));
      Thumper.bench_with_setup "host-logsumexp-f32-1024x1024-cols"
        ~setup:m1024
        (scan ~axis:0 (Logsumexp (0, D.Float32)));
      Thumper.bench_with_setup "host-logsumexp-f32-1M" ~setup:f1m
        (scan ~axis:0 (Logsumexp (0, D.Float32)));
      Thumper.bench_with_setup "host-argmax-f32-1024x1024-rows" ~setup:m1024
        (fun x -> fst (scan (Arg (Max, 0, D.Float32)) x));
      Thumper.bench_with_setup "host-argmax-f32-1M" ~setup:f1m (fun x ->
          fst (scan ~axis:0 (Arg (Max, 0, D.Float32)) x));
    ]

(* Order and counted lengths, each against its NumPy counterpart at the same
   size: 1M uniform draws, rows of 1024, a 64K-key table, 1000 groups. *)
let u1m () = on_host (Nx.Rng.uniform ~key:(Nx.Rng.key 7) D.Float32 [| n1m |])
let u1m64 () = on_host (Nx.cast D.Float64 (u1m ()))

let sorted_64k () =
  on_host
    (fst
       (Nx.sort (Nx.Rng.uniform ~key:(Nx.Rng.key 8) D.Float32 [| 1 lsl 16 |])))

let groups1m () =
  on_host (Nx.Rng.randint ~key:(Nx.Rng.key 9) ~high:1000 [| n1m |])
let quartiles = [| 0.25; 0.5; 0.75 |]

let order_rows =
  Thumper.group "order"
    [
      Thumper.bench_with_setup "sort-f32-1M" ~setup:u1m Nx.sort;
      Thumper.bench_with_setup "argsort-f32-1M" ~setup:u1m Nx.argsort;
      Thumper.bench_with_setup "sort-f32-1024x1024-rows"
        ~setup:(fun () -> Nx.reshape [| 1024; 1024 |] (u1m ()))
        Nx.sort;
      Thumper.bench_with_setup "top_k-8-f32-64x16384"
        ~setup:(fun () -> Nx.reshape [| 64; n1m / 64 |] (u1m ()))
        (Nx.top_k ~k:8);
      Thumper.bench_with_setup "searchsorted-f32-1M-in-64K"
        ~setup:(fun () -> (sorted_64k (), u1m ()))
        (fun (table, x) -> Nx.searchsorted ~side:`Left table x);
      Thumper.bench_with_setup "unique-i32-1M-1000" ~setup:groups1m Nx.unique;
      Thumper.bench_with_setup "positions-bool-1M" ~setup:b1m Nx.positions;
      Thumper.bench_with_setup "quantile-f64-1M-3" ~setup:u1m64
        (Nx.quantile quartiles);
      Thumper.bench_with_setup "histogram-f64-1M-100"
        ~setup:(fun () -> (on_host (Nx.linspace D.Float64 0. 1. 101), u1m64 ()))
        (fun (edges, x) -> Nx.histogram [ (edges, x) ]);
    ]

(* Fourier transforms of host values, each against vDSP and pocketfft at the
   same size: complex64 lines of 2^10, 2^16, 2^20 and 2^22 points, a float32
   line of 2^20 and its inverse, a complex64 line of the prime 65537 points
   (by Bluestein's convolution), and complex64 1024 × 1024 and 2048 × 2048
   along both axes: 2^22 points, 32 MiB, are past fudge's largest cache;
   16384 complex64 lines of 64 points (a batch by passes) and a complex128
   line of 2^16 points. Then rows without a vDSP counterpart: a complex64
   line of 3^10 points (radix 3), 4096 float32 lines of 255 points (odd
   real lines, which are not halved) and 32 complex64 lines of 2^16 points,
   16 MiB, whose four-step grids stay in cache while the result streams. *)
let fft f = Nx.Prim.eval ~by:"bench" (Fft f)

(* The first [n] points of [c1m], in memory of their own. *)
let c64 n () = on_host (Nx.slice [ R (0, n) ] (c1m ()))

let c4m () =
  let c = c1m () in
  on_host (Nx.concatenate ~axis:0 [ c; c; c; c ])

let c2c axes x = fft (C2c { direction = Forward; axes; x })

let fft_rows =
  Thumper.group "fft"
    [
      Thumper.bench_with_setup "c2c-1K-c64" ~setup:(c64 1024) (c2c [| 0 |]);
      Thumper.bench_with_setup "c2c-64K-c64" ~setup:(c64 (1 lsl 16))
        (c2c [| 0 |]);
      Thumper.bench_with_setup "c2c-1M-c64" ~setup:c1m (c2c [| 0 |]);
      Thumper.bench_with_setup "c2c-4M-c64" ~setup:c4m (c2c [| 0 |]);
      Thumper.bench_with_setup "rfft-1M-f32" ~setup:f1m (fun x ->
          fft (R2c { dtype = D.Complex64; axes = [| 0 |]; x }));
      Thumper.bench_with_setup "irfft-1M-f32"
        ~setup:(c64 ((1 lsl 19) + 1))
        (fun x ->
          fft (C2r { dtype = D.Float32; n = 1 lsl 20; axes = [| 0 |]; x }));
      Thumper.bench_with_setup "c2c-65537-c64" ~setup:(c64 65537) (c2c [| 0 |]);
      Thumper.bench_with_setup "fft2-1024x1024-c64"
        ~setup:(fun () -> Nx.reshape [| 1024; 1024 |] (c1m ()))
        (c2c [| 0; 1 |]);
      Thumper.bench_with_setup "fft2-2048x2048-c64"
        ~setup:(fun () -> Nx.reshape [| 2048; 2048 |] (c4m ()))
        (c2c [| 0; 1 |]);
      Thumper.bench_with_setup "c2c-16Kx64-c64"
        ~setup:(fun () -> Nx.reshape [| 16384; 64 |] (c1m ()))
        (c2c [| 1 |]);
      Thumper.bench_with_setup "c2c-32x64K-c64"
        ~setup:(fun () ->
          let c = c1m () in
          on_host
            (Nx.reshape [| 32; 65536 |] (Nx.concatenate ~axis:0 [ c; c ])))
        (c2c [| 1 |]);
      Thumper.bench_with_setup "c2c-59049-c64" ~setup:(c64 59049) (c2c [| 0 |]);
      Thumper.bench_with_setup "c2c-64K-c128"
        ~setup:(fun () -> on_host (Nx.cast D.Complex128 (c64 (1 lsl 16) ())))
        (c2c [| 0 |]);
      Thumper.bench_with_setup "rfft-4Kx255-f32"
        ~setup:(fun () ->
          on_host
            (Nx.reshape [| 4096; 255 |]
               (Nx.slice [ R (0, 4096 * 255) ] (f1m ()))))
        (fun x -> fft (R2c { dtype = D.Complex64; axes = [| 1 |]; x }));
    ]

(* Products with MXFP4 weights as a checkpoint stores them, codes apart from
   their scales, each copy in memory of its own: a decode step's 4096 × 4096
   projection of one token, cycling 16 copies (142 MB, past fudge's caches)
   so that each call reads its weight from memory, and a mixture of experts'
   layer, 64 tokens each through 8 of 32 experts of 4096 inputs to 1024
   outputs, token [i]'s experts [(5i + 4j) mod 32] for [j] below 8, over 2
   copies (71 MB each). *)

(* [copies] weights of shape [[| lead…; n; k |]], their rows alike: a fixed
   pattern of codes and scales near 1. *)
let mxfp4_weights ~copies lead n k =
  let bytes count f =
    on_host (Nx.cast D.Uint8 (f (Nx.arange D.Int32 0 count 1)))
  in
  let codes = bytes (n * k / 2) (fun i -> Nx.mod_s (Nx.mul_s i 37l) 256l) in
  let scales = bytes (n * k / 32) (fun i -> Nx.add_s (Nx.mod_s i 19l) 118l) in
  let rows last t =
    Nx.copy
      (Nx.broadcast_to
         (Array.append lead [| n; last |])
         (Nx.reshape [| n; last |] t))
  in
  Array.init copies (fun _ ->
      Nx_quant.mxfp4 ~scales:(rows (k / 32) scales) (rows (k / 2) codes))

(* [next ws turn] is the weight after the last call's. *)
let next ws turn =
  turn := (!turn + 1) mod Array.length ws;
  ws.(!turn)

let experts () =
  let ids =
    Nx.init D.Int64 [| 64; 8 |] (fun i ->
        Int64.of_int (((5 * i.(0)) + (4 * i.(1))) mod 32))
  in
  let ws = mxfp4_weights ~copies:2 [| 32 |] 1024 4096 in
  (ids, filled D.Float32 [| 64; 1; 4096 |] 1., ws, ref 0)

(* Sixteen weights of [4096; 4096] split along [axis] over four memory
   devices, and an input whole on each: each device sums its own rows, or
   its own 32 blocks of each row, from the codes where they lie. *)
type cut_weights =
  | Cut_weights :
      (float, D.float32_elt, 'd) Nx.t * 'd Nx_quant.t array * int ref
      -> cut_weights

let mxfp4_four ~axis () =
  let module Four = (val Nx.devices [ memory 0; memory 1; memory 2; memory 3 ])
  in
  let ws = mxfp4_weights ~copies:16 [||] 4096 4096 in
  let x = filled D.Float32 [| 1; 4096 |] 1. in
  Cut_weights
    ( Nx.place Four.on x,
      Array.map (Nx_quant.place (Four.split ~axis)) ws,
      ref 0 )

(* The mixture of experts' layer with each weight split along its 32 experts
   over four memory devices, the ids and tokens whole on each: each device
   sums the rows of its own 8 experts, and the devices' sums close by
   partials. *)
type cut_experts =
  | Cut_experts :
      'd Nx.int64_t
      * (float, D.float32_elt, 'd) Nx.t
      * 'd Nx_quant.t array
      * int ref
      -> cut_experts

let experts_four () =
  let module Four = (val Nx.devices [ memory 0; memory 1; memory 2; memory 3 ])
  in
  let ids, x, ws, turn = experts () in
  Cut_experts
    ( Nx.place Four.on ids,
      Nx.place Four.on x,
      Array.map (Nx_quant.place (Four.split ~axis:0)) ws,
      turn )

let quant_rows =
  Thumper.group "quant"
    [
      Thumper.bench_with_setup "mxfp4-gemv-4096"
        ~setup:(fun () ->
          let ws = mxfp4_weights ~copies:16 [||] 4096 4096 in
          (filled D.Float32 [| 1; 4096 |] 1., ws, ref 0))
        (fun (x, ws, turn) -> Nx.Repr.array (Nx_quant.apply (next ws turn) x));
      Thumper.bench_with_setup "mxfp4-gemv-4096-n-split4"
        ~setup:(mxfp4_four ~axis:0) (fun (Cut_weights (x, ws, turn)) ->
          Value (Nx_quant.apply (next ws turn) x));
      Thumper.bench_with_setup "mxfp4-gemv-4096-k-split4"
        ~setup:(mxfp4_four ~axis:1) (fun (Cut_weights (x, ws, turn)) ->
          Value (Nx_quant.apply (next ws turn) x));
      Thumper.bench_with_setup "mxfp4-moe-64x8of32" ~setup:experts
        (fun (ids, x, ws, turn) ->
          let w = next ws turn in
          Nx.Repr.array (Nx_quant.apply_experts ~ids w x));
      Thumper.bench_with_setup "mxfp4-moe-64x8of32-experts-split4"
        ~setup:experts_four (fun (Cut_experts (ids, x, ws, turn)) ->
          Value (Nx_quant.apply_experts ~ids (next ws turn) x));
    ]

let () =
  exit
  @@ Thumper.run "nx"
       [
         dispatch_rows;
         small_rows;
         elementwise_rows;
         constant_rows;
         placed_rows;
         place_rows;
         move_rows;
         rng_rows;
         contract_rows;
         conv_rows;
         index_rows;
         interp_rows;
         frontend_rows;
         statistics_rows;
         scan_rows;
         order_rows;
         fft_rows;
         quant_rows;
       ]
