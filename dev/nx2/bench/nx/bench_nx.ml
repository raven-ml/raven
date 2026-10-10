(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's per-call costs above the array layer, each row what a user calls: an
   operation on one-element host values, its kernel called as nx calls it and
   directly, a view, constants, an operation over two devices, reading a value's
   shape, placing, and operations beside and under interpretations. *)

module A = Nx_array
module D = Nx_array.Dtype

(* A memory device, opened in the measuring worker: a device opened before the
   fork is lost in it. *)
let memory k =
  match Rig.memory_device (Printf.sprintf "bench-m%d" k) with
  | Ok d -> d
  | Error e -> failwith e

let host n =
  Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 [| n |] (Array.make n 1.))

let x1 = host 1
let x16 = host 16
let x1m = host (1 lsl 20)
let b1 = Nx.less x1 x1
let a1 = Option.get (Nx.Repr.array x1)
let half = Nx.scalar D.Float32 0.5

(* K1's steps over nx.array and nx.cpu, the kernel called statically: the floor
   [dispatch/add-1]'s indirect call is judged against. *)
let add_direct a =
  let l = A.layout a in
  let dst =
    A.v D.Float32 l (Rig.Buffer.create (A.device a) (D.bytes D.Float32 1))
  in
  match Nx_cpu.apply2 (Binary Add) ~dst a a with
  | Done -> Nx.Repr.of_array Nx.Host.v dst
  | refusal -> A.refused "add-1-direct" refusal [ A.Any dst; A.Any a ]

let chain n =
  let rec go k x = if k = 0 then x else go (k - 1) (Nx.add x half) in
  go n (Nx.zeros D.Float32 [| 1 |])

let dispatch_rows =
  Thumper.group "dispatch"
    [
      Thumper.bench "shape" (fun () -> Nx.shape (Thumper.black_box x1));
      Thumper.bench "add-1" (fun () -> Nx.add (Thumper.black_box x1) x1);
      Thumper.bench "add-1-direct" (fun () -> add_direct (Thumper.black_box a1));
      Thumper.bench "less-1" (fun () -> Nx.less (Thumper.black_box x1) x1);
      Thumper.bench "where-1" (fun () -> Nx.where (Thumper.black_box b1) x1 x1);
      Thumper.bench "cast-1" (fun () ->
          Nx.cast D.Float64 (Thumper.black_box x1));
      Thumper.bench "reshape-1" (fun () ->
          Nx.reshape [| 1; 1 |] (Thumper.black_box x1));
      Thumper.bench "zeros_like-1" (fun () ->
          Nx.zeros_like (Thumper.black_box x1));
      Thumper.bench "zeros_like-1M" (fun () ->
          Nx.zeros_like (Thumper.black_box x1m));
      Thumper.bench "zeros-1M" (fun () ->
          Nx.place Nx.Host.on (Nx.zeros D.Float32 [| 1 lsl 20 |]));
      Thumper.bench "add-1-twice" (fun () ->
          let x = Thumper.black_box x1 in
          Nx.add (Nx.add x x) x);
      Thumper.bench "add-1-twice-donated" (fun () ->
          let x = Thumper.black_box x1 in
          Nx.add (Nx.donate (Nx.add x x)) x);
      Thumper.bench "where-1-twice-donated" (fun () ->
          let x = Thumper.black_box x1 in
          Nx.where b1 (Nx.donate (Nx.add x x)) x);
      Thumper.bench "cast-1-twice-donated" (fun () ->
          let x = Thumper.black_box x1 in
          Nx.cast D.Float64 (Nx.donate (Nx.add x x)));
      Thumper.bench "add-scalar-1" (fun () ->
          Nx.add (Thumper.black_box x1) (Nx.scalar D.Float32 1.));
      Thumper.bench "add-held-scalar-1" (fun () ->
          Nx.add (Thumper.black_box x1) half);
    ]

let constant_rows =
  Thumper.group "constant"
    [
      Thumper.bench "chain-1000" (fun () ->
          Nx.Repr.array (Nx.place Nx.Host.on (chain 1000)));
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
  Borrow (Mem.on, x16)

(* A 1024 × 1024 float32 value, 4 MiB, and a split on axis 1 over four memory
   devices: from the host, and from a split on axis 0, where each device's
   column window meets every row window. *)
type reshard =
  | Reshard : 'd Nx.Placement.t * (float, D.float32_elt, 'e) Nx.t -> reshard

let square =
  Nx.Repr.of_array Nx.Host.v
    (A.of_array D.Float32 [| 1024; 1024 |] (Array.make (1024 * 1024) 1.))

let host_to_columns () =
  let module Four = (val Nx.devices [ memory 0; memory 1; memory 2; memory 3 ])
  in
  Reshard (Four.split ~axis:1, square)

let rows_to_columns () =
  let module Four = (val Nx.devices [ memory 0; memory 1; memory 2; memory 3 ])
  in
  Reshard (Four.split ~axis:1, Nx.place (Four.split ~axis:0) square)

let placed_rows =
  Thumper.group "placed"
    [
      Thumper.bench_with_setup "add-1-two-memory-devices" ~setup:split_two
        (fun (Value two) -> Value (Nx.add two two));
    ]

let place_rows =
  Thumper.group "place"
    [
      Thumper.bench "equal-1" (fun () ->
          Nx.place Nx.Host.on (Thumper.black_box x1));
      Thumper.bench_with_setup "borrow-memory-device-16" ~setup:borrow_16
        (fun (Borrow (on, x)) -> Value (Nx.place on x));
      Thumper.bench_with_setup "host-to-split-axis1" ~setup:host_to_columns
        (fun (Reshard (p, x)) -> Value (Nx.place p x));
      Thumper.bench_with_setup "split-to-split" ~setup:rows_to_columns
        (fun (Reshard (p, x)) -> Value (Nx.place p x));
    ]

(* Movements: a view costs the same at one element and at a million, and a
   movement no stride expresses costs a copy of its result, as copy-1M does. *)

let x1_2d = Nx.reshape [| 1; 1 |] x1
let split_heads = Nx.Pattern.v "b t (h d) -> b h t d"
let rows = Nx.reshape [| 1; 1024; 1024 |] square
let quarter = host (1 lsl 18)

let move_rows =
  Thumper.group "move"
    [
      Thumper.bench "transpose-1" (fun () ->
          Nx.transpose (Thumper.black_box x1_2d));
      Thumper.bench "transpose-1M" (fun () ->
          Nx.transpose (Thumper.black_box square));
      Thumper.bench "rearrange-heads-1M" (fun () ->
          Nx.rearrange ~sizes:[ ("h", 16) ] split_heads (Thumper.black_box rows));
      Thumper.bench "reshape-swapaxes-heads-1M" (fun () ->
          Nx.swapaxes 1 2
            (Nx.reshape [| 1; 1024; 16; 64 |] (Thumper.black_box rows)));
      Thumper.bench "repeat-256K-4" (fun () ->
          Nx.repeat ~axis:0 4 (Thumper.black_box quarter));
      Thumper.bench "copy-1M" (fun () -> Nx.copy (Thumper.black_box x1m));
    ]

(* Random draws computed on the host: each row places a draw of a fixed key,
   which computes it. The samplers' programs differ in length: one Threefry
   block per element for bits and uniform, two for normal, and fixed rounds of
   proposals for the rejection samplers. *)

let mib = 1 lsl 20
let key = Nx.Rng.key 0

let filled n v =
  Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 [| n |] (Array.make n v))

let rates = filled (1 lsl 16) 4.
let probabilities = filled mib 0.3
let draw x = Nx.Repr.array (Nx.place Nx.Host.on x)

(* The draws' floor: nx.cpu's Threefry called directly over 1M counters under
   one broadcast key, the one block per element that bits and uniform compute,
   without the counters' and words' arithmetic around it. *)
let counters = A.of_array D.Uint64 [| mib |] (Array.init mib Int64.of_int)

let one_key =
  Option.get
    (A.move (A.Move.Broadcast [| mib |]) (A.of_array D.Uint64 [| 1 |] [| 0L |]))

let threefry_direct c =
  let dst = A.create (A.device c) D.Uint64 [| mib |] in
  match Nx_cpu.apply2 (Binary Threefry) ~dst c one_key with
  | Done -> dst
  | refusal -> A.refused "threefry-1M-direct" refusal [ A.Any dst; A.Any c ]

let rng_rows =
  Thumper.group "rng"
    [
      Thumper.bench "threefry-1M-direct" (fun () ->
          threefry_direct (Thumper.black_box counters));
      Thumper.bench "bits-1M" (fun () ->
          draw (Nx.Rng.bits ~key:(Thumper.black_box key) [| mib |]));
      Thumper.bench "uniform-f32-1M" (fun () ->
          draw (Nx.Rng.uniform ~key:(Thumper.black_box key) D.Float32 [| mib |]));
      Thumper.bench "uniform-f64-1M" (fun () ->
          draw (Nx.Rng.uniform ~key:(Thumper.black_box key) D.Float64 [| mib |]));
      Thumper.bench "normal-f32-1M" (fun () ->
          draw (Nx.Rng.normal ~key:(Thumper.black_box key) D.Float32 [| mib |]));
      Thumper.bench "randint-1M" (fun () ->
          draw
            (Nx.Rng.randint ~key:(Thumper.black_box key) ~high:1000 [| mib |]));
      Thumper.bench "bernoulli-f32-1M" (fun () ->
          draw (Nx.Rng.bernoulli ~key:(Thumper.black_box key) probabilities));
      Thumper.bench "gamma-f32-64K" (fun () ->
          draw (Nx.Rng.gamma ~key:(Thumper.black_box key) rates));
      Thumper.bench "poisson-f32-64K" (fun () ->
          draw (Nx.Rng.poisson ~key:(Thumper.black_box key) rates));
    ]

(* Contractions at the shapes of nx.cpu's and nx.cuda's rows, each beside its
   kernel called directly, so that the frontend's cost over it shows: squares,
   and a decode step's one row against a weight stored as [n × k]. *)

let matrix dt s =
  Nx.Repr.of_array Nx.Host.v
    (A.of_array dt s (Array.make (Array.fold_left ( * ) 1 s) 1.))

let gemm_direct ~contracting a b s =
  let spec =
    Nx_kernel.Spec.contract ~batch:[||] ~contracting ~acc:(D.Any D.Float32)
      ~out:(D.Any D.Float32) ~init:false
  in
  let a = Option.get (Nx.Repr.array a) and b = Option.get (Nx.Repr.array b) in
  fun () ->
    let dst = A.create Rig.host D.Float32 s in
    match Nx_cpu.contract spec ~dst:(A.Any dst) [| A.Any a; A.Any b |] with
    | Done -> dst
    | refusal -> A.refused "contract-direct" refusal [ A.Any dst ]

let matmul_rows n =
  let a = matrix D.Float32 [| n; n |] and b = matrix D.Float32 [| n; n |] in
  let name = Printf.sprintf "matmul-f32-%d" n in
  [
    Thumper.bench name (fun () ->
        Nx.Repr.array (Nx.matmul (Thumper.black_box a) b));
    Thumper.bench (name ^ "-direct")
      (gemm_direct ~contracting:[| (1, 0) |] a b [| n; n |]);
  ]

let rows_by_weight = Nx.Pattern.v "m k, n k -> m n | k"

let decode_rows =
  let x = matrix D.Float32 [| 1; 4096 |]
  and w = matrix D.Float32 [| 4096; 4096 |] in
  let xb = matrix D.Bfloat16 [| 1; 2880 |]
  and wb = matrix D.Bfloat16 [| 5120; 2880 |] in
  [
    Thumper.bench "einsum-f32-m1x4096x4096" (fun () ->
        Nx.Repr.array (Nx.einsum rows_by_weight (Thumper.black_box x) w));
    Thumper.bench "einsum-f32-m1x4096x4096-direct"
      (gemm_direct ~contracting:[| (1, 1) |] x w [| 1; 4096 |]);
    Thumper.bench "einsum-bf16-m1x5120x2880" (fun () ->
        Nx.Repr.array (Nx.einsum rows_by_weight (Thumper.black_box xb) wb));
  ]

let contract_rows =
  Thumper.group "contract"
    (matmul_rows 4 @ matmul_rows 64 @ matmul_rows 1024 @ decode_rows)

(* Indexing: a row take reads its rows, an add-scatter its updates, a set
   writes its row into a copy of the cache, and concatenate copies its result
   once. *)

let table =
  Nx.Repr.of_array Nx.Host.v
    (A.of_array D.Float32 [| 8192; 256 |] (Array.make (8192 * 256) 1.))

let ids_1k =
  Nx.Repr.of_array Nx.Host.v
    (A.of_array D.Int64 [| 1024 |]
       (Array.init 1024 (fun i -> Int64.of_int (i * 7 mod 8192))))

let ids_64k =
  Nx.Repr.of_array Nx.Host.v
    (A.of_array D.Int64 [| 65536 |]
       (Array.init 65536 (fun i -> Int64.of_int (i mod 1024))))

let ones_64k = host 65536
let sums = Nx.zeros D.Float32 [| 1024 |]
let half_1m = host (1 lsl 19)

(* A cache of 8 heads, 4096 positions and 64 features, 8 MiB, and one
   position's row. *)
let cache =
  Nx.add
    (Nx.Repr.of_array Nx.Host.v
       (A.of_array D.Float32 [| 1; 8; 4096; 64 |]
          (Array.make (8 * 4096 * 64) 0.)))
    (Nx.zeros D.Float32 [| 1; 8; 4096; 64 |])

let row =
  Nx.Repr.of_array Nx.Host.v
    (A.of_array D.Float32 [| 1; 8; 1; 64 |] (Array.make 512 1.))

let pos = Nx.Repr.of_array Nx.Host.v (A.of_array D.Int64 [||] [| 1000L |])

let index_rows =
  Thumper.group "index"
    [
      Thumper.bench "take-rows-1K-of-8K" (fun () ->
          Nx.take ~axis:0 (Thumper.black_box ids_1k) table);
      Thumper.bench "scatter-add-64K-into-1K" (fun () ->
          Nx.scatter ~combine:Add ~axis:0 (Thumper.black_box ids_64k) ones_64k
            sums);
      Thumper.bench "set-row-8M" (fun () ->
          Nx.set Nx.[ A; A; D (pos, 1); A ] row (Thumper.black_box cache));
      Thumper.bench "concatenate-2x512K" (fun () ->
          Nx.concatenate ~axis:0 [ Thumper.black_box half_1m; half_1m ]);
    ]

(* Interpretations. Each row adds one-element host values 100 times: eagerly;
   under a Values interpretation that does not reach them; while an Extent lives
   on another domain; while one lives on this domain around another fiber, which
   costs each add a perform; and on a traced value, delivered to its
   interpretation's rule. *)

let adds () =
  let x = Thumper.black_box x1 in
  for _ = 1 to 100 do
    ignore (Sys.opaque_identity (Nx.add x x))
  done

type ('v, 's, 'd) Nx.Prim.payload += Traced : ('v, 's, 'd) Nx.Prim.payload

let tracing i ~by op =
  Nx.Prim.results ~by (fun _ form -> Nx.Prim.traced i form Traced) op

(* An Extent on another domain, live until [end_elsewhere]. *)
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
  (stop, d)

let end_elsewhere (stop, d) =
  Atomic.set stop true;
  Domain.join d

(* An Extent on this domain, suspended in its own fiber until [end_here]. *)
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
  k

let end_here k = Option.iter (fun resume -> resume ()) !k

let interp_rows =
  Thumper.group "interp"
    [
      Thumper.bench "add-1-100-eager" adds;
      Thumper.bench "add-1-100-under-values" (fun () ->
          Nx.Prim.interpret ~name:"bench" Values tracing (fun _ -> adds ()));
      Thumper.bench_with_setup "add-1-100-beside-extent-on-another-domain"
        ~setup:extent_elsewhere ~teardown:end_elsewhere (fun _ -> adds ());
      Thumper.bench_with_setup "add-1-100-fiber-beside-extent"
        ~setup:extent_here ~teardown:end_here (fun _ -> adds ());
      Thumper.bench "traced-add-1-100" (fun () ->
          Nx.Prim.interpret ~name:"bench" Values tracing (fun i ->
              let t = Nx.Prim.traced i (Nx.Prim.form x1) Traced in
              for _ = 1 to 100 do
                ignore (Sys.opaque_identity (Nx.add t t))
              done));
    ]

(* The frontend's compositions over 1M host elements, each what a user calls:
   creation of a leaf (placed on the host, which computes it), the crossings
   to OCaml values, the scalar forms and the functions nx composes from the
   kernels' kinds. *)

let n1m = 1 lsl 20
let ocaml_floats = Array.init n1m (fun i -> Float.of_int (i mod 1000) /. 1000.)
let f1m = Nx.create D.Float32 [| n1m |] ocaml_floats
let above1 = Nx.place Nx.Host.on (Nx.add_s f1m 1.)
let i1m = Nx.place Nx.Host.on (Nx.arange D.Int32 (-(n1m / 2)) (n1m / 2) 1)
let b1m = Nx.place Nx.Host.on (Nx.less_s f1m 0.5)
let c1m = Nx.place Nx.Host.on (Nx.complex D.Complex64 ~re:f1m ~im:above1)
let m1024 = Nx.reshape [| 1024; 1024 |] f1m
let v1024 = Nx.reshape [| 1024 |] (Nx.slice [ R (0, 1024) ] f1m)
let p1000 = Nx.slice [ R (0, 1000) ] f1m
let on_host x = Nx.place Nx.Host.on x

let frontend_rows =
  Thumper.group "frontend"
    [
      Thumper.bench "create-f32-1M" (fun () ->
          Nx.create D.Float32 [| n1m |] (Thumper.black_box ocaml_floats));
      Thumper.bench "to_array-f32-1M" (fun () ->
          Nx.to_array (Thumper.black_box f1m));
      Thumper.bench "item-f32-1M" (fun () ->
          Nx.item [ 524_287 ] (Thumper.black_box f1m));
      Thumper.bench "arange-f32-1M" (fun () ->
          on_host (Nx.arange D.Float32 0 (Thumper.black_box n1m) 1));
      Thumper.bench "linspace-f32-1M" (fun () ->
          on_host (Nx.linspace D.Float32 0. 1. (Thumper.black_box n1m)));
      Thumper.bench "eye-f32-1024" (fun () ->
          on_host (Nx.eye D.Float32 (Thumper.black_box 1024)));
      Thumper.bench "full_like-f32-1M" (fun () ->
          Nx.full_like (Thumper.black_box f1m) 2.);
      Thumper.bench "add_s-f32-1M" (fun () ->
          Nx.add_s (Thumper.black_box f1m) 1.);
      Thumper.bench "less_s-f32-1M" (fun () ->
          Nx.less_s (Thumper.black_box f1m) 0.5);
      Thumper.bench "logical_and-f32-1M" (fun () ->
          Nx.logical_and (Thumper.black_box f1m) f1m);
      Thumper.bench "logical_and-bool-1M" (fun () ->
          Nx.logical_and (Thumper.black_box b1m) b1m);
      Thumper.bench "isnan-f32-1M" (fun () -> Nx.isnan (Thumper.black_box f1m));
      Thumper.bench "isfinite-f32-1M" (fun () ->
          Nx.isfinite (Thumper.black_box f1m));
      Thumper.bench "clamp-f32-1M" (fun () ->
          Nx.clamp ~min:0.25 ~max:0.75 (Thumper.black_box f1m));
      Thumper.bench "lshift-i32-1M" (fun () -> Nx.lshift (Thumper.black_box i1m) 3);
      Thumper.bench "rshift-i32-1M" (fun () -> Nx.rshift (Thumper.black_box i1m) 3);
      Thumper.bench "bitwise_not-i32-1M" (fun () ->
          Nx.bitwise_not (Thumper.black_box i1m));
      Thumper.bench "square-f32-1M" (fun () -> Nx.square (Thumper.black_box f1m));
      Thumper.bench "rsqrt-f32-1M" (fun () -> Nx.rsqrt (Thumper.black_box above1));
      Thumper.bench "hypot-f32-1M" (fun () ->
          Nx.hypot (Thumper.black_box f1m) above1);
      Thumper.bench "asinh-f32-1M" (fun () -> Nx.asinh (Thumper.black_box f1m));
      Thumper.bench "acosh-f32-1M" (fun () -> Nx.acosh (Thumper.black_box above1));
      Thumper.bench "atanh-f32-1M" (fun () -> Nx.atanh (Thumper.black_box f1m));
      Thumper.bench "real-c64-1M" (fun () ->
          Nx.real D.Float32 (Thumper.black_box c1m));
      Thumper.bench "magnitude-c64-1M" (fun () ->
          Nx.magnitude D.Float32 (Thumper.black_box c1m));
      Thumper.bench "angle-c64-1M" (fun () ->
          Nx.angle D.Float32 (Thumper.black_box c1m));
      Thumper.bench "complex-c64-1M" (fun () ->
          Nx.complex D.Complex64 ~re:(Thumper.black_box f1m) ~im:above1);
      Thumper.bench "conjugate-c64-1M" (fun () ->
          Nx.conjugate (Thumper.black_box c1m));
      Thumper.bench "tril-f32-1024" (fun () -> Nx.tril (Thumper.black_box m1024));
      Thumper.bench "trace-f32-1024" (fun () -> Nx.trace (Thumper.black_box m1024));
      Thumper.bench "diag-f32-1024" (fun () -> Nx.diag (Thumper.black_box v1024));
      Thumper.bench "vdot-f32-1M" (fun () -> Nx.vdot (Thumper.black_box f1m) f1m);
      Thumper.bench "to_string-f32-1000" (fun () ->
          Nx.to_string (Thumper.black_box p1000));
    ]

let () =
  exit
  @@ Thumper.run "nx"
       [
         dispatch_rows;
         constant_rows;
         placed_rows;
         place_rows;
         move_rows;
         rng_rows;
         contract_rows;
         index_rows;
         interp_rows;
         frontend_rows;
       ]
