(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx_kernel.Prog

type op0 = Fill of string | Iota of int

type kind =
  | K0 of op0
  | K1 of Nx_kernel.Prog.op1
  | K2 of Nx_kernel.Prog.op2
  | K3 of Nx_kernel.Prog.op3

type attention = {
  factor : Nx_array.Dtype.any -> Nx_array.Dtype.any -> int -> float;
  epsilon : Nx_array.Dtype.any -> int -> float -> float;
  zero_weights : Nx_array.Dtype.any -> int -> float;
  rounds_weights : bool;
}

type axis = Complex | R2c_last | C2r_last

type backend = {
  name : string;
  kernels : (module Nx_kernel.S);
  device : Rig.t;
  computes : kind -> Nx_array.Dtype.any -> bool;
  around : 'a. (unit -> 'a) -> 'a;
  attention : attention option;
  lane_order : bool;
  transform_factor : axis -> int -> float;
  scan_order : bool;
}

(* nx_cpu.mli's and nx_cuda.mli's promise: every kind at every dtype of its
   domain. *)
let every_kind _ _ = true

(* nx_cpu.mli's attention. Its error factor is [γ(k + 1)] in the
   accumulator, its [ε] is
   [e{^λu} (13n + 9 + ln t) u' + (e{^λu} - 1) ln t + u' |lse|], [n] the
   blocks of 64 keys and [λ] [3] in float32 and [5] in float64. Where every
   output is [0], its weights are exact: [exp (+0)] is [1] and [log 1] is
   [+0] (Nx_kernel.Prog's special values), [l] sums [tk] ones exactly, [α]
   is [1], and the division by a power of two is exact. It keeps its
   weights in the accumulator. *)
let cpu_attention =
  let unit acc =
    if acc = Nx_array.Dtype.Any Float32 then 0x1p-24 else 0x1p-53
  in
  let factor _ acc k =
    let u = unit acc and k = Float.of_int (k + 1) in
    k *. u /. (1. -. (k *. u))
  in
  let epsilon acc t lse =
    let u = unit acc in
    let u' = u /. (1. -. u) and ln = Float.log (Float.of_int t) in
    let lambda = if u = 0x1p-24 then 3. else 5. in
    let grow = Float.exp (lambda *. u) in
    (grow *. (Float.of_int ((13 * ((t + 63) / 64)) + 9) +. ln) *. u')
    +. ((grow -. 1.) *. ln)
    +. (u' *. Float.abs lse)
  in
  { factor; epsilon; zero_weights = (fun _ _ -> 0.); rounds_weights = false }

(* nx_cuda.mli's attention. With [p(k) = ⌈k / 256⌉ - 1], at least the
   merges of the parts a row of [k] keys spans, its error factor is
   [γ(k + 1 + p(k))], and [γ(k + 2⌈k/8⌉ + p(k))] for float16 and bfloat16
   queries, keys and values, which the matrix units may sum; its [ε] is
   nx.cpu's with [n + 2p(t)] blocks. Where every output is [0], its weights
   are exact, as nx.cpu's are, and a merge of parts multiplies by
   [exp (+0)]: [1]. It may apply weights through a rounding to the
   values' dtype. *)
let cuda_attention =
  let unit acc =
    if acc = Nx_array.Dtype.Any Float32 then 0x1p-24 else 0x1p-53
  in
  let parts t = Int.max 0 (((t + 255) / 256) - 1) in
  let factor q acc k =
    let u = unit acc in
    let units =
      match q with
      | Nx_array.Dtype.Any (Float16 | Bfloat16) ->
          k + (2 * ((k + 7) / 8)) + parts k
      | _ -> k + 1 + parts k
    in
    let k = Float.of_int units in
    k *. u /. (1. -. (k *. u))
  in
  let epsilon acc t lse =
    let u = unit acc in
    let u' = u /. (1. -. u) and ln = Float.log (Float.of_int t) in
    let lambda = if u = 0x1p-24 then 3. else 5. in
    let grow = Float.exp (lambda *. u) in
    let n = ((t + 63) / 64) + (2 * parts t) in
    (grow *. (Float.of_int ((13 * n) + 9) +. ln) *. u')
    +. ((grow -. 1.) *. ln)
    +. (u' *. Float.abs lse)
  in
  { factor; epsilon; zero_weights = (fun _ _ -> 0.); rounds_weights = true }

(* The prime factors of [n], least first. *)
let rec prime_factors n d =
  if n <= 1 then []
  else if d * d > n then [ n ]
  else if n mod d = 0 then d :: prime_factors (n / d) d
  else prime_factors n (d + 1)

let pow2_above n =
  let rec go p = if p >= n then p else go (2 * p) in
  go 1

(* nx_cpu.mli's transforms: e(n) = 8 Σ p^(3/2) + 8 over n's prime factors,
   or 6 e(m) + 26 for Bluestein's convolution of m points, the least power
   of two at least 2n - 1, where a prime factor exceeds 250; a real line of
   even n >= 512 through n / 2 complex points, e(n/2) + 8. *)
let cpu_transform =
  let rec e n =
    let ps = prime_factors n 2 in
    if List.exists (fun p -> p > 250) ps then (6. *. e (pow2_above ((2 * n) - 1))) +. 26.
    else List.fold_left (fun s p -> s +. (8. *. (Float.of_int p ** 1.5))) 8. ps
  in
  fun axis n ->
    match axis with
    | (R2c_last | C2r_last) when n mod 2 = 0 && n >= 512 -> e (n / 2) +. 8.
    | _ -> e n

(* nx_cuda.mli's transforms: e(n) = 5.35 log₂ n for a 7-smooth n and
   91 log₂ n + 236 for another; along a real last axis of even n,
   e(n/2) + 11 (r2c) or e(n/2) + 20 (c2r). *)
let cuda_transform =
  let e n =
    let log2 = Float.log2 (Float.of_int (Int.max n 1)) in
    if List.for_all (fun p -> p <= 7) (prime_factors n 2) then 5.35 *. log2
    else (91. *. log2) +. 236.
  in
  fun axis n ->
    match axis with
    | R2c_last when n mod 2 = 0 -> e (n / 2) +. 11.
    | C2r_last when n mod 2 = 0 -> e (n / 2) +. 20.
    | _ -> e n

let under target =
  {
    name = "cpu/" ^ target;
    kernels = (module Nx_cpu);
    device = Rig.host;
    computes = every_kind;
    around = (fun f -> Nx_cpu.with_target target f);
    attention = Some cpu_attention;
    lane_order = true;
    transform_factor = cpu_transform;
    scan_order = true;
  }

let cpus = List.map under Nx_cpu.targets

(* The best table, which nx.cpu runs from the start: the last. *)
let cpu = under (List.nth Nx_cpu.targets (List.length Nx_cpu.targets - 1))

(* nx.cuda on GPU 0 of the machine's NVIDIA GPUs, opened by each rig driver
   that reaches it: CUDA's library, and NVIDIA's kernel driver through rig.nv.
   nx_cuda.mli states it computes every kind at every dtype. *)
let nvidia ~name open_ =
  match open_ () with
  | Error why -> failwith (name ^ ": " ^ why)
  | Ok d when not (Nx_cuda.computes_on d) -> []
  | Ok d ->
      [
        {
          name;
          kernels = (module Nx_cuda);
          device = d;
          computes = every_kind;
          around = (fun f -> f ());
          attention = Some cuda_attention;
          lane_order = true;
          transform_factor = cuda_transform;
          scan_order = false;
        };
      ]

(* Devices *)

let on b a =
  if Rig.equal (Nx_array.device a) b.device then a
  else Nx_array.to_device b.device a

let host a =
  if Rig.equal (Nx_array.device a) Rig.host then a
  else Nx_array.to_device Rig.host a

let sees count = match count () with Ok n -> n > 0 | Error _ -> false

(* Opened at the first call, so that an executable that links this module
   but runs no nx.cuda group, such as nx.cpu's suites, which need not hold
   the GPU's lock, opens no GPU. *)
let gpus =
  lazy
    (let cuda () = Rig.open_ { (Rig_cuda.gpu 0) with name = "CUDA:0" } in
     if Sys.getenv_opt "NX_KERNELS" = Some "cpu" then []
     else if sees Rig_cuda.count then nvidia ~name:"cuda" cuda
     else [])

let gpus () = Lazy.force gpus
let backends () = cpu :: gpus ()

let nv =
  lazy
    (let nv () = Rig.open_ { (Rig_nv_nvidia.gpu 0) with name = "NV:0" } in
     match gpus () with
     | [] -> None
     | _ :: _ when not (sees Rig_nv_nvidia.count) -> None
     | _ :: _ -> List.nth_opt (nvidia ~name:"cuda/nv" nv) 0)

let nv () = Lazy.force nv

(* Elements as bits *)

let elements (Nx_array.Any x) =
  let module A = Nx_array in
  let module D = Nx_array.Dtype in
  let bytes (type v s) (c : (v, s) A.t) =
    match A.dtype c with
    | D.Bit -> Array.map (fun b -> if b then "\001" else "\000") (A.to_array c)
    | dt when D.bits dt = 4 ->
        Array.map
          (fun v -> String.make 1 (Char.chr v))
          (A.to_array (Option.get (A.bitcast D.Uint4 c)))
    | dt ->
        let w = D.bits dt / 8 in
        let b = A.to_array (Option.get (A.bitcast D.Uint8 c)) in
        Array.init
          (Array.length b / w)
          (fun i -> String.init w (fun j -> Char.chr b.((w * i) + j)))
  in
  bytes (A.copy (host x))

(* Whether [x] and [y], of one dtype and shape, hold the same elements bit
   for bit: their C-contiguous bytes, or a sub-byte array's elements one by
   one. *)
let same_bytes (Nx_array.Any x) (Nx_array.Any y) =
  let module A = Nx_array in
  if A.Dtype.bits (A.dtype x) < 8 then elements (A.Any x) = elements (A.Any y)
  else
    let x = A.copy (host x) and y = A.copy (host y) in
    Rig.Buffer.read (A.buffer x) Bigarray.char (fun a ->
        Rig.Buffer.read (A.buffer y) Bigarray.char (fun b -> a = b))

external x86_64 : unit -> bool = "nx_kernels_support_x86_64" [@@noalloc]

let x86_64 = x86_64 ()

(* Guard pages *)

type side = Before | After

external guarded_bytes :
  bool ->
  int ->
  int ->
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  * nativeint
  * int = "nx_kernels_support_guarded"

external unguard : nativeint -> int -> unit = "nx_kernels_support_unguard"

let guarded side dt l =
  let lo, hi = Nx_array.Layout.span l in
  let first = lo * Nx_array.Dtype.bits dt / 8 in
  let bytes = Nx_array.Dtype.bytes dt hi in
  let ba, base, total = guarded_bytes (side = After) first bytes in
  Bigarray.Array1.fill (Bigarray.Array1.sub ba first (bytes - first)) '\000';
  let a = Nx_array.v dt l (Rig.Buffer.of_bigarray ba) in
  (a, fun () -> unguard base total)

let guard side x =
  let dt = Nx_array.dtype x and l = Nx_array.layout x in
  let g, free = guarded side dt l in
  let lo, hi = Nx_array.Layout.span l in
  let first = lo * Nx_array.Dtype.bits dt / 8 in
  let bytes a =
    Nx_array.v Nx_array.Dtype.Uint8
      (Nx_array.Layout.v ~offset:first ~strides:[| 1 |]
         [| Nx_array.Dtype.bytes dt hi - first |])
      (Nx_array.buffer a)
  in
  Nx_array.transfer ~src:(bytes x) ~dst:(bytes g);
  (g, free)

(* One-node maps *)

let map (module K : Nx_kernel.S) ~ins n ~dst loads =
  K.map (P.of_node ~ins n) ~dsts:[| Nx_array.Any dst |] loads

let map0 k op ~dst =
  let dt = Nx_array.Dtype.Any (Nx_array.dtype dst) in
  let n = match op with Fill b -> P.Const (dt, b) | Iota i -> P.Coord i in
  map k ~ins:[||] n ~dst [||]

let map1 k op ~dst x =
  let y = Nx_array.Dtype.Any (Nx_array.dtype dst) in
  map k
    ~ins:[| Nx_array.Dtype.Any (Nx_array.dtype x) |]
    (Op1 (op, y, 0)) ~dst [| Nx_kernel.Plain x |]

let map2 k op ~dst x y =
  let d = Nx_array.Dtype.Any (Nx_array.dtype x) in
  map k ~ins:[| d; d |] (Op2 (op, 0, 1)) ~dst [| Nx_kernel.Plain x; Plain y |]

let map3 k op ~dst c x y =
  let d = Nx_array.Dtype.Any (Nx_array.dtype x) in
  map k
    ~ins:[| Nx_array.Dtype.Any (Nx_array.dtype c); d; d |]
    (Op3 (op, 0, 1, 2)) ~dst [| Nx_kernel.Plain c; Plain x; Plain y |]

(* Device memory *)

let settled d =
  for _ = 1 to 4 do
    Gc.full_major ();
    Rig.free_cache d
  done;
  Rig.allocated d

let loop_bytes d n f =
  f ();
  let before = settled d in
  Array.init n (fun _ ->
      f ();
      Rig.allocated d - before)

(* Address space *)

external cap_memory : int -> bool = "nx_kernels_support_cap_memory"
