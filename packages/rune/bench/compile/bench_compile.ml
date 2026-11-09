(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Cold compiles, and first calls over a warm disk cache.

   Each sample runs this executable again as [--cold case] over an empty disk
   cache, so a sample is a fresh process's whole first call of a compiled
   function: tracing, lowering, tolk's passes and the kernel compiler, and the
   process's start. A warm process cannot hold this cost: tolk's disk cache
   serves a program after its first compile. A case takes three samples. The
   cases are those whose compile time rune's lowering, tolk's vectors, a
   library's stores of its intermediates or the number of devices a program
   spans decide. The [cuda/] cases run on CUDA's first GPU, whose driver also
   loads each binary, the [amd/] cases on AMD's first GPU, through Linux's
   amdgpu driver, and the [metal/] cases on the Mac's GPU, whose pipelines
   Metal builds: each is left out without its GPU.

   A [warm/] case's sample is a later process's first call: the case's setup
   compiles it into a disk cache, which each sample's process reads its
   schedules and programs back from, on as many domains as tolk's default
   [PARALLEL] gives.

   A [first/] case's sample is only the first call of a process that set up
   its inputs first, with no compile yet: its setup starts processes that
   prepare the call and wait, and each sample has one of them make it. Such
   a row sees the cost of a process's first compile, a few milliseconds
   that a whole process's start and setup would hide.

   [--cold ID] runs one case's first call and prints its milliseconds over
   the disk cache [CACHEDB]: a cold sample runs it at an empty path and
   [PARALLEL=1], or the default [PARALLEL] for a case whose kernels compile
   at once, and a warm one over its setup's cache. [--first ID] prepares a
   [first/] case's call, prints "ready", makes the call when a line comes on
   its standard input, and prints "done". *)


let f64 = Nx.float64

(* [n] host float64 values in [[lo, hi)]. *)
let input ?(n = 1024) lo hi =
  Nx.place Nx.Host.on
    (Nx.add_s (Nx.mul_s (Nx.Rng.uniform f64 [| n |]) (hi -. lo)) lo)

let compile f x = ignore (Sys.opaque_identity (Rune.jit' f x))

(* A synthetic program of [ops] operations shaped as a special function:
   regions selected by [where] over a parameter and a variable that broadcast
   against each other, each region a chain whose intermediates fan out into
   two uses. A region is [region_ops] operations: its selection, its clamp,
   [region_steps] steps of four operations and the select. A compile time that
   grows faster than the program shows from about 4k operations (rune's
   workload). *)
let region_ops = 64
let region_steps = 14

let shaped ops a x =
  let result = ref (Nx.zeros_like x) in
  for r = 0 to (ops / region_ops) - 1 do
    let lo = Float.of_int r /. 8. in
    let inside =
      Nx.logical_and (Nx.greater_equal_s x lo) (Nx.less_s x (lo +. 0.125))
    in
    let xr = Nx.where inside x (Nx.full_like x (lo +. 0.0625)) in
    let t = ref xr and s = ref (Nx.broadcast_to (Nx.shape x) a) in
    for k = 1 to region_steps do
      t := Nx.add (Nx.mul !t xr) !s;
      s := Nx.sub_s (Nx.mul !s !t) (Float.of_int k)
    done;
    result := Nx.where inside !t !result
  done;
  !result

(* A dense solve of 32 float64 equations: an LU factorization with partial
   pivoting and two triangular solves, each a loop of one step per row. *)
let solve () =
  let m = 32 in
  let a =
    Nx.place Nx.Host.on
      (Nx.add
         (Nx.Rng.uniform f64 [| m; m |])
         (Nx.mul_s (Nx.eye f64 m) (Float.of_int m)))
  in
  let b = input ~n:m 0. 1. in
  compile (fun a -> Nx.solve a b) a

(* The eigendecomposition of 64 symmetric float32 matrices of 8 x 8: 56 rounds
   of Jacobi rotations, held in a loop. *)
let eigh () =
  let a = Nx.Rng.uniform Nx.float32 [| 64; 8; 8 |] in
  let a = Nx.place Nx.Host.on (Nx.add a (Nx.matrix_transpose a)) in
  ignore
    (Sys.opaque_identity
       (Rune.jit
          Nx.Ptree.(tensor @-> returns (pair tensor tensor))
          (fun a -> Nx.eigh Nx.float32 a)
          a))

(* The eigendecomposition of one symmetric float64 matrix of 32 x 32: its
   largest magnitude folds 32 lanes into a chain of maxima in a loop of 32
   trips, which LLVM 22's and 23's loop idiom recognition did not finish. *)
let eigh_32 () =
  let a = Nx.Rng.uniform f64 [| 32; 32 |] in
  let a = Nx.place Nx.Host.on (Nx.add a (Nx.matrix_transpose a)) in
  ignore
    (Sys.opaque_identity
       (Rune.jit
          Nx.Ptree.(tensor @-> returns (pair tensor tensor))
          (fun a -> Nx.eigh f64 a)
          a))

(* A tridiagonal solve of 100 float64 equations by Thomas's elimination as
   three associative scans: of 2 x 2 matrices as four arrays, for the pivots,
   then of affine maps as pairs, for the eliminated and the substituted
   values. A level is read at two or three positions: inlined whole, the
   levels made one kernel that grew some 3 times per level, and tolk stores
   one once its copies are many. *)
let tridiagonal () =
  let n = 100 in
  let matrix (a00, a01, a10, a11) (b00, b01, b10, b11) =
    let open Nx in
    let p00 = add (mul b00 a00) (mul b01 a10)
    and p01 = add (mul b00 a01) (mul b01 a11)
    and p10 = add (mul b10 a00) (mul b11 a10)
    and p11 = add (mul b10 a01) (mul b11 a11) in
    let s =
      maximum (maximum (abs p00) (abs p01)) (maximum (abs p10) (abs p11))
    in
    let s = where (equal s (zeros_like s)) (ones_like s) s in
    (div p00 s, div p01 s, div p10 s, div p11 s)
  in
  let affine (a1, b1) (a2, b2) = Nx.(mul a1 a2, add (mul a2 b1) b2) in
  let pair = Nx.Ptree.(pair tensor tensor) in
  let quad =
    Nx.Ptree.v (fun c (a, b, e, d) ->
        let open Nx.Ptree.Walk in
        let a = index c 0 tensor a in
        let b = index c 1 tensor b in
        let e = index c 2 tensor e in
        let d = index c 3 tensor d in
        (a, b, e, d))
  in
  let shift_down v =
    let first = Nx.slice [ Nx.R (0, 1) ] v in
    let rest = Nx.slice [ Nx.R (0, n - 1) ] v in
    Nx.concatenate ~axis:0 [ Nx.zeros_like first; rest ]
  in
  let sub = input ~n (-1.) 0. and sup = input ~n (-1.) 0. in
  let r = input ~n 0. 1. in
  let solve diag =
    let _, p01, _, p11 =
      Nx.associative_scan quad matrix
        (Nx.zeros_like diag, sup, Nx.neg sub, diag)
    in
    let ratio = Nx.div p01 p11 in
    let den = Nx.sub diag (Nx.mul sub (shift_down ratio)) in
    let _, d =
      Nx.associative_scan pair affine (Nx.neg (Nx.div sub den), Nx.div r den)
    in
    let flip v = Nx.flip ~axes:[ 0 ] v in
    let _, x = Nx.associative_scan pair affine (flip (Nx.neg ratio), flip d) in
    flip x
  in
  compile solve (input ~n 3. 4.)

(* A sampler's start and first steps on a model of 8 groups (eight schools): a
   location, a log scale and 8 effects, under normal and half-Cauchy priors.
   Each of 8 chains starts at the first of 100 uniform candidates whose log
   density is finite, which a check requires of one, with the last candidate's
   terms as its data, then takes 10 gradient steps, each checking each term
   finite. The checks' first failures are read by their data and the failed
   flag, and carried by the loop with their data, as a sampler's first call
   compiles them. *)
let schools () =
  let y = Nx.create f64 [| 8 |] [| 28.; 8.; -3.; 7.; -1.; 1.; 18.; 12. |] in
  let sigma =
    Nx.create f64 [| 8 |] [| 15.; 10.; 16.; 11.; 9.; 11.; 10.; 18. |]
  in
  let normal x m s =
    Nx.add_s (Nx.mul_s (Nx.square (Nx.div_s (Nx.sub_s x m) s)) (-0.5))
      (-.Float.log s)
  in
  (* Of rows [mu; log tau; eta_1 .. eta_8], one density each. *)
  let log_density q =
    let mu = Nx.slice [ Nx.A; Nx.R (0, 1) ] q in
    let log_tau = Nx.slice [ Nx.A; Nx.R (1, 2) ] q in
    let eta = Nx.slice [ Nx.A; Nx.R (2, 10) ] q in
    let tau = Nx.exp log_tau in
    let theta = Nx.add mu (Nx.mul tau eta) in
    let z = Nx.div (Nx.sub y theta) sigma in
    let cauchy = Nx.neg (Nx.log (Nx.add_s (Nx.square (Nx.div_s tau 5.)) 1.)) in
    let terms =
      [
        normal mu 0. 5.;
        Nx.add cauchy log_tau;
        normal eta 0. 1.;
        Nx.sub (Nx.mul_s (Nx.square z) (-0.5)) (Nx.log sigma);
      ]
    in
    List.map (Nx.sum ~axes:[ 1 ]) terms
  in
  let sum ts = List.fold_left Nx.add (List.hd ts) (List.tl ts) in
  let start k =
    let u = Nx.Rng.uniform_at Nx.Host.on ~key:k f64 [| 100; 10 |] in
    let cands = Nx.mul_s (Nx.sub_s u 0.5) 4. in
    let finite = Nx.isfinite (sum (log_density cands)) in
    let last = List.map (Nx.slice [ Nx.I 99 ]) (log_density cands) in
    Nx.check
      Nx.Ptree.(list tensor)
      (Nx.any finite) last
      (fun _ _ -> Failure "no finite start");
    Nx.take ~axis:0 (Nx.argmax (Nx.cast Nx.int32 finite)) cands
  in
  let step q _ =
    let terms = log_density q in
    let lp = sum terms in
    List.iter
      (fun t ->
        Nx.check
          Nx.Ptree.(pair tensor tensor)
          (Nx.isfinite t) (t, lp)
          (fun _ _ -> Failure "a step left the support"))
      terms;
    let g = Rune.grad' (fun q -> Nx.sum (sum (log_density q))) q in
    (Nx.add q (Nx.mul_s g 0.01), lp)
  in
  let sample k =
    let k = Nx.Rng.of_tensor k in
    let q =
      Rune.vmap
        Nx.Ptree.(Nx.Rng.ptree @-> returns tensor)
        start
        (Nx.Rng.split_batch ~n:8 k)
    in
    let q, lps = Rune.scan' step ~init:q (Nx.zeros f64 [| 10; 1 |]) in
    Nx.add (Nx.sum q) (Nx.sum lps)
  in
  compile sample (Nx.Rng.to_tensor (Nx.Rng.key 0))

(* A scan of 20 steps whose step checks each of its 48 values finite, with the
   value and the step's sum as data: a loop that carries each check's first
   failure and data, as a sampler's step carries its sites' checks. *)
let checked_scan () =
  let n = 48 in
  let step x _ =
    let y = Nx.add (Nx.mul_s x 0.5) (Nx.recip (Nx.add_s (Nx.square x) 1.)) in
    let total = Nx.sum y in
    for i = 0 to n - 1 do
      let v = Nx.slice [ Nx.I i ] y in
      Nx.check
        Nx.Ptree.(pair tensor tensor)
        (Nx.isfinite v) (v, total)
        (fun _ _ -> Failure "a value left the reals")
    done;
    (y, total)
  in
  let run x =
    let y, totals = Rune.scan' step ~init:x (Nx.zeros f64 [| 20; 1 |]) in
    Nx.add (Nx.sum y) (Nx.sum totals)
  in
  compile run (input ~n 0. 1.)

(* [shaped 4096] of a 2-element state padded to 3: the pad masks the state's
   axis, which the host upcasts whole, so each operation is of 3 lanes, a
   vector of the state's loads and the pad. *)
let shaped_padded () =
  compile
    (fun x -> shaped 4096 (Nx.scalar f64 0.5) (Nx.pad [| (0, 1) |] 0.5 x))
    (input ~n:2 0. 1.)

(* A map over 3 lanes of a loop on 4 by 3 float8 values, whose step selects
   between two polynomials of the values, one of them flipped: lane [i] steps
   [trips.(i)] times, so the lanes stop apart and each step keeps a stopped
   lane's values by a selection on its flag. The host upcasts the 3 by 4 by 3
   values whole, 36 lanes, and rune computes each float8 operation in float32
   between a decode and an encode. A lane's flag, broadcast across its 12
   values, left every operation after its selection lane by lane: 0.7 MB of C,
   and a first call of 3.2-7.2 s against 0.5-1.0 s. *)
let held_lanes_fp8 () =
  let x =
    Nx.place Nx.Host.on
      (Nx.cast Nx.float8_e4m3fn
         (Nx.add_s
            (Nx.mul_s (Nx.Rng.uniform Nx.float32 [| 3; 4; 3 |]) 3.)
            (-1.5)))
  in
  let trips =
    Nx.place Nx.Host.on (Nx.create Nx.int32 [| 3 |] [| 0l; 2l; 1l |])
  in
  let carry = Nx.Ptree.(pair tensor tensor) in
  let one = Nx.scalar Nx.int32 1l in
  let step (x, k) =
    let open Nx in
    let a = flip ~axes:[ 1 ] (sub (square (sub_s x 3.)) (square (mul x x))) in
    let b = mul (maximum (mul x x) (abs x)) (sub (maximum x (neg x)) x) in
    (where (less a b) (mul a b) (sub a b), add k one)
  in
  let iterated (x, t) =
    fst
      (Rune.iterate carry ~max:2
         ~until:(fun (_, k) -> Nx.greater_equal k t)
         step
         (x, Nx.scalar Nx.int32 0l))
  in
  let signature = Nx.Ptree.(carry @-> returns tensor) in
  ignore
    (Sys.opaque_identity
       (Rune.jit signature (Rune.vmap signature iterated) (x, trips)))

(* A chain of 4 float32 incomplete betas of 4 values, each link's point the
   last link's value, under shapes read from memory. The host upcasts the 4
   values whole, so each operation is a vector of 4 lanes over loads of the
   whole buffers. Its selections share their conditions on the shapes with
   every link. *)
let betainc_chain () =
  let f32 n lo hi =
    Nx.place Nx.Host.on
      (Nx.add_s (Nx.mul_s (Nx.Rng.uniform Nx.float32 [| n |]) (hi -. lo)) lo)
  in
  let a = f32 4 0.5 3.5 and b = f32 4 0.5 3.5 in
  let chain x =
    let y = ref x in
    for _ = 1 to 4 do
      y := Nx.betainc a b !y
    done;
    !y
  in
  compile chain (f32 4 0. 1.)

(* The gradient of a sum of 1021 float32 complemented beta quantiles in their
   shapes and probabilities, as rune's special-function suite compiles it: one
   scalar loop kernel of 0.9 MB of C, past the length from which Clang compiles
   without its loop vectorizer, whose time grows faster than the loop. *)
let betainccinv_grad () =
  let f32 n lo hi =
    Nx.place Nx.Host.on
      (Nx.add_s (Nx.mul_s (Nx.Rng.uniform Nx.float32 [| n |]) (hi -. lo)) lo)
  in
  let n = 1021 in
  let a = f32 n 0.5 3.5 and b = f32 n 0.5 3.5 and p = f32 n 0.01 0.99 in
  let s = Nx.Ptree.(pair (pair tensor tensor) tensor) in
  ignore
    (Sys.opaque_identity
       (Rune.jit
          Nx.Ptree.(s @-> returns s)
          (Rune.grad s (fun ((a, b), p) -> Nx.sum (Nx.betainccinv a b p)))
          ((a, b), p)))

(* The gradient of the float64 beta quantile of 4 values in its shapes and
   its probability: the guess and the Halley steps, which no derivative
   passes, each an incomplete beta whose regions every value computes, and
   the last step with its derivative. *)
let betaincinv_grad () =
  let s = Nx.Ptree.(pair (pair tensor tensor) tensor) in
  let grad = Rune.grad s (fun ((a, b), p) -> Nx.sum (Nx.betaincinv a b p)) in
  let a = input ~n:4 0.5 30. and b = input ~n:4 0.5 30. in
  ignore
    (Sys.opaque_identity
       (Rune.jit Nx.Ptree.(s @-> returns s) grad ((a, b), input ~n:4 0.01 0.99)))

(* The gradient of the float64 gamma quantile of 4 values in its shape and
   its probability: the guess and the Halley steps, which no derivative
   passes, and the last step with its derivative. *)
let gammaincinv_grad () =
  let s = Nx.Ptree.(pair tensor tensor) in
  let grad = Rune.grad s (fun (a, p) -> Nx.sum (Nx.gammaincinv a p)) in
  ignore
    (Sys.opaque_identity
       (Rune.jit
          Nx.Ptree.(s @-> returns s)
          grad
          (input ~n:4 0.5 30., input ~n:4 0.01 0.99)))

(* A normal draw of 1024 values of [dt] from a key argument: a Box-Muller
   transform, whose cosine and sine read an angle its bits bound below 2 pi. *)
let normal dt () =
  let draw k = Nx.Rng.normal ~key:(Nx.Rng.of_tensor k) dt [| 1024 |] in
  compile draw (Nx.Rng.to_tensor (Nx.Rng.key 0))

(* A double-word sum of 10^6 float64 numbers, both words returned: a tree of
   20 levels of double-word additions, which nx.wide cuts into programs of a
   few levels each. *)
let wide_sum () =
  let sum x =
    let s = Nx_wide.sum (Nx_wide.v x) in
    (Nx_wide.hi s, Nx_wide.lo s)
  in
  ignore
    (Sys.opaque_identity
       (Rune.jit
          Nx.Ptree.(tensor @-> returns (pair tensor tensor))
          sum
          (input ~n:1_000_000 0. 1.)))

(* A data-parallel step of a two-layer perceptron, its batch of 64 cut across
   16 memory devices of one set and its weights whole on each: the loss's
   gradient sums across the devices before the update. Each device computes
   its own kernels, which compile once, for the first device: the first
   call's cost past one device's is each device's share of the trace. *)
let data_parallel () =
  let devices =
    List.init 16 (fun i ->
        Result.get_ok (Rig.memory_device (Printf.sprintf "bench-dp-%d" i)))
  in
  let module S = (val Nx.devices devices) in
  let make shape seed =
    let n = Array.fold_left ( * ) 1 shape in
    Nx.create Nx.float32 shape
      (Array.init n (fun j -> sin (Float.of_int ((seed * 7919) + j)) *. 0.3))
  in
  let step (x, (w1, w2)) =
    let loss (w1, w2) =
      let y = Nx.matmul (Nx.tanh (Nx.matmul x w1)) w2 in
      Nx.mean (Nx.mul y y)
    in
    let g1, g2 = Rune.grad Nx.Ptree.(pair tensor tensor) loss (w1, w2) in
    (Nx.sub w1 (Nx.mul_s g1 0.1), Nx.sub w2 (Nx.mul_s g2 0.1))
  in
  let x = Nx.place (S.split ~axis:0) (make [| 64; 64 |] 3) in
  let w1 = Nx.place S.on (make [| 64; 128 |] 1) in
  let w = (w1, Nx.place S.on (make [| 128; 16 |] 2)) in
  ignore
    (Sys.opaque_identity
       (Rune.jit
          Nx.Ptree.(
            pair tensor (pair tensor tensor) @-> returns (pair tensor tensor))
          step (x, w)))

(* A fold, a running maximum's positions, a gather and an [Add] scatter over
   an axis of 64 rows cut across 16 memory devices of one set. Each computes
   by partials, each device over its tile, whose start each device's call
   binds, as it binds the start where a device stores its part of a closing:
   each device's kernels are the first device's programs. *)
let partials () =
  let devices =
    List.init 16 (fun i ->
        let name = Printf.sprintf "bench-partials-%d" i in
        Result.get_ok (Rig.memory_device name))
  in
  let module S = (val Nx.devices devices) in
  let cut = Nx.Placement.split ~axis:0 S.v and on = Nx.Placement.on S.v in
  let rows = 64 and width = 16 in
  let positions bound shape =
    let n = Array.fold_left ( * ) 1 shape in
    Nx.create Nx.int64 shape
      (Array.init n (fun j -> Int64.of_int (j * 5 mod bound)))
  in
  let x =
    Nx.place cut
      (Nx.create Nx.float32 [| rows; width |]
         (Array.init (rows * width) (fun j -> sin (Float.of_int j))))
  in
  let along = Nx.place on (positions rows [| 3; width |]) in
  let at = Nx.place cut (positions rows [| rows; width |]) in
  let into = Nx.place on (Nx.zeros Nx.float32 [| rows; width |]) in
  let f x =
    let first = Nx.cast Nx.float32 (Nx.argmax ~axis:0 x) in
    let peaks = snd (Rune.jvp' (Nx.cummax ~axis:0) x x) in
    let taken = Nx.take_along_axis ~axis:0 along peaks in
    let added = Nx.scatter ~combine:Add ~axis:0 at x into in
    Nx.add
      (Nx.add first (Nx.sum ~axes:[ 0 ] added))
      (Nx.sum ~axes:[ 0 ] taken)
  in
  compile f x

(* A product of 2880 float32 values by a bfloat16 matrix of 2880 by 512,
   widened to float32, as gpt-oss-20b projects its keys and values at a decode
   step: one kernel whose conversion's lanes, from loads gathered across the
   matrix's rows, set Clang's work. *)
let vecmat_bf16 () =
  let w = Nx.cast Nx.bfloat16 (Nx.Rng.uniform Nx.float32 [| 2880; 512 |]) in
  let w = Nx.place Nx.Host.on w in
  let x = Nx.place Nx.Host.on (Nx.Rng.uniform Nx.float32 [| 1; 2880 |]) in
  compile (fun x -> Nx.matmul x (Nx.cast Nx.float32 w)) x

(* The gradient in [w] of 256 rows of 16 float32 values by the 16 by 16
   matrix of [w] each row's id names, of 8: a contraction of each group's
   rows, read at positions its ids hold, whose host kernel's lanes, put
   together from many loaded rows, set Clang's work. *)
let segments_grad () =
  let rows = 256 and k = 16 and groups = 8 in
  let ids =
    Nx.place Nx.Host.on
      (Nx.create Nx.int64 [| rows |]
         (Array.init rows (fun r -> Int64.of_int (r mod groups))))
  in
  let x = Nx.place Nx.Host.on (Nx.Rng.uniform Nx.float32 [| rows; k |]) in
  let y = Nx.place Nx.Host.on (Nx.Rng.uniform Nx.float32 [| rows; k |]) in
  let w = Nx.place Nx.Host.on (Nx.Rng.uniform Nx.float32 [| groups; k; k |]) in
  compile
    (Rune.grad' (fun w -> Nx.sum (Nx.mul (Nx.matmul_segments ids x w) y)))
    w

(* Eight nested float64 sines of 1024 values on CUDA's first GPU: one kernel of
   some thousand lines, whose binary the GPU's driver loads. *)
let cuda_sines () =
  match Rig.open_ (Rig_cuda.gpu 0) with
  | Error e -> failwith e
  | Ok gpu ->
      let module D = (val Nx.devices [ gpu ] : Nx.Devices) in
      let x = Nx.place D.on (Nx.linspace f64 (-10.) 10. 1024) in
      compile
        (fun x -> List.fold_left (fun x _ -> Nx.sin x) x (List.init 8 Fun.id))
        x

(* A gradient step of a two-layer perceptron of 64 rows of 256 float32
   values, normalized between its layers, under a softmax cross-entropy, on
   AMD's first GPU: its forward and backward passes are some 15 kernels, each
   compiled by comgr, which sets the first call's cost. *)
let amd_step () =
  match Rig.open_ (Rig_amd_amdgpu.gpu 0) with
  | Error e -> failwith e
  | Ok gpu ->
      let module D = (val Nx.devices [ gpu ] : Nx.Devices) in
      let host shape = Nx.sub_s (Nx.Rng.uniform Nx.float32 shape) 0.5 in
      let f32 shape = Nx.place D.on (host shape) in
      let x = f32 [| 64; 256 |] in
      let target = Nx.place D.on (Nx.abs (host [| 64; 16 |])) in
      let loss (w1, w2) =
        let h = Nx.matmul x w1 in
        let mu = Nx.mean ~axes:[ 1 ] ~keepdims:true h in
        let d = Nx.sub h mu in
        let var = Nx.mean ~axes:[ 1 ] ~keepdims:true (Nx.square d) in
        let a = Nx.tanh (Nx.mul d (Nx.rsqrt (Nx.add_s var 1e-5))) in
        let logp = Nx.log_softmax ~axes:[ 1 ] (Nx.matmul a w2) in
        Nx.neg (Nx.mean (Nx.sum ~axes:[ 1 ] (Nx.mul target logp)))
      in
      let step (w1, w2) =
        let g1, g2 = Rune.grad Nx.Ptree.(pair tensor tensor) loss (w1, w2) in
        (Nx.sub w1 (Nx.mul_s g1 0.1), Nx.sub w2 (Nx.mul_s g2 0.1))
      in
      let w = Nx.Ptree.(pair tensor tensor) in
      ignore
        (Sys.opaque_identity
           (Rune.jit
              Nx.Ptree.(w @-> returns w)
              step
              (f32 [| 256; 128 |], f32 [| 128; 16 |])))

(* 32 sums of sines on the Mac's GPU, each of rows of 64 float32 values, of
   1 to 32 rows, scaled by its own constant offset by [salt]. Metal builds
   their pipelines in its own processes, several at once, and keeps them in
   its own cache, on the machine's disk. *)
let metal_sums salt () =
  match Rig.open_ (Rig_metal.gpu 0) with
  | Error e -> failwith e
  | Ok gpu ->
      let module D = (val Nx.devices [ gpu ] : Nx.Devices) in
      let k = 32 in
      let xs =
        List.init k (fun i ->
            Nx.place D.on (Nx.Rng.uniform Nx.float32 [| i + 1; 64 |]))
      in
      let sums xs =
        List.mapi
          (fun i x ->
            Nx.sum ~axes:[ 1 ] (Nx.sin (Nx.mul_s x (salt +. Float.of_int i))))
          xs
      in
      ignore
        (Sys.opaque_identity
           (Rune.jit Nx.Ptree.(list tensor @-> returns (list tensor)) sums xs))

(* One sum of sines of 8 rows of 64 float32 values on the Mac's GPU, scaled by
   the process's id, so that Metal's cache holds none of its pipeline: the
   device opens and the input is placed, then the call compiles one kernel,
   the process's first compile. *)
let metal_one () =
  match Rig.open_ (Rig_metal.gpu 0) with
  | Error e -> failwith e
  | Ok gpu ->
      let module D = (val Nx.devices [ gpu ] : Nx.Devices) in
      let x = Nx.place D.on (Nx.Rng.uniform Nx.float32 [| 8; 64 |]) in
      let salt = Float.of_int (Unix.getpid ()) in
      fun () ->
        compile (fun x -> Nx.sum ~axes:[ 1 ] (Nx.sin (Nx.mul_s x salt))) x

let pendulum () =
  let steps x =
    let theta = ref x and v = ref (Nx.zeros_like x) in
    for _ = 1 to 200 do
      v := Nx.sub !v (Nx.mul_s (Nx.sin !theta) 0.01);
      theta := Nx.add !theta (Nx.mul_s !v 0.01)
    done;
    !theta
  in
  compile steps (input ~n:64 0.1 0.9)

let cases =
  [
    ("linalg/solve-32", solve);
    ("linalg/eigh-8x8", eigh);
    ("linalg/eigh-f64-32x32", eigh_32);
    (* A float64 sine of an unbounded angle, reduced by the bits of 1/(2pi) at
       its exponent: its lowering's size. *)
    ("trig/sin", fun () -> compile Nx.sin (input (-10.) 10.));
    (* Eight nested sines: one elementwise kernel whose host lanes, a vector of
       64 bytes, set Clang's work. *)
    ( "trig/sin-nested-8",
      fun () ->
        compile
          (fun x -> List.fold_left (fun x _ -> Nx.sin x) x (List.init 8 Fun.id))
          (input (-10.) 10.) );
    ("scan/tridiagonal-100", tridiagonal);
    ("model/schools-start", schools);
    ("scan/checks-48", checked_scan);
    (* The same first compile at the default [PARALLEL]: its host kernels
       lowered, rendered and compiled together. *)
    ("parallel/checks-48", checked_scan);
    ("rng/normal-f64", normal f64);
    ("rng/normal-f32", normal Nx.float32);
    ( "shaped-4k",
      fun () -> compile (shaped 4096 (Nx.scalar f64 0.5)) (input 0. 1.) );
    ("shaped-4k-3-lanes", shaped_padded);
    ("lanes/held-fp8-36", held_lanes_fp8);
    ("special/betainc-chain-f32", betainc_chain);
    ("special/betainccinv-grad-f32", betainccinv_grad);
    ("special/betaincinv-grad-f64", betaincinv_grad);
    ("special/gammaincinv-grad-f64", gammaincinv_grad);
    ("wide/sum-1e6", wide_sum);
    (* A pendulum of 200 fixed steps written as an OCaml loop: one elementwise
       kernel of 3.6 MB of C, which the host writes as several functions, since
       Clang's time on one function grows faster than the function. *)
    ("long/pendulum-200", pendulum);
    ("products/vecmat-bf16", vecmat_bf16);
    ("products/segments-grad", segments_grad);
    ("devices/dp-step-16", data_parallel);
    ("devices/partials-16", partials);
    ("cuda/sin-nested-8", cuda_sines);
    ("amd/mlp-step", amd_step);
    (* Offset by the process's id: 32 kernels no earlier process compiled,
       which Metal's cache holds none of. Each sample leaves its 32 pipelines
       there. *)
    ("metal/sums-32", fun () -> metal_sums (Float.of_int (Unix.getpid ())) ());
    (* The same kernels in every process, whose pipelines Metal's cache holds
       after the first: timed warm only. *)
    ("metal/sums-32-cached", metal_sums 0.);
  ]

(* The [first/] cases: each prepares its call, which it returns. *)
let first_cases = [ ("metal/one-new", metal_one) ]

(* The cases whose cold samples run at the default [PARALLEL]: their kernels'
   compiles run at once. *)
let parallel_cases = [ "metal/sums-32"; "parallel/checks-48" ]

(* The cases timed again over a warm disk cache, as a user's next run of a
   program makes its first call: a model's first steps, whose schedule reads 40
   programs back, a scan that reads 112, a decomposition, a kernel whose
   binary CUDA's driver loads, and 32 kernels whose pipelines rig prepares
   together and Metal finds in its cache. *)
let warm_cases =
  [
    "model/schools-start";
    "scan/checks-48";
    "linalg/eigh-8x8";
    "cuda/sin-nested-8";
    "metal/sums-32-cached";
  ]

(* The cases timed warm only. *)
let warm_only = [ "metal/sums-32-cached" ]

(* Whether [gpu]'s first GPU opens, [gpu] being ["cuda"], ["amd"] or
   ["metal"], asked of a fresh process. *)
let opens gpu =
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process exe [| exe; "--opens"; gpu |] Unix.stdin Unix.stdout
      Unix.stderr
  in
  match Unix.waitpid [] pid with _, Unix.WEXITED 0 -> true | _ -> false

let rec remove path =
  if Sys.is_directory path then (
    Array.iter (fun f -> remove (Filename.concat path f)) (Sys.readdir path);
    Sys.rmdir path)
  else Sys.remove path

(* A fresh process's first call of [id] over the disk cache in [dir], with the
   variables [env] set. The GPU driver's own cache of the code it translates is
   off, so that a warm call's binaries come from the disk cache alone, as they
   do where the driver's cache misses. *)
let first_call ~env dir id =
  let exe = Sys.executable_name in
  let env =
    Array.concat
      [
        [|
          "CACHEDB=" ^ Filename.concat dir "cache.db";
          "CUDA_CACHE_DISABLE=1";
        |];
        env;
        Unix.environment ();
      ]
  in
  let null = Unix.openfile Filename.null [ Unix.O_WRONLY ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close null) @@ fun () ->
  let pid =
    Unix.create_process_env exe [| exe; "--cold"; id |] env Unix.stdin null
      Unix.stderr
  in
  match snd (Unix.waitpid [] pid) with
  | Unix.WEXITED 0 -> ()
  | _ -> failwith ("bench_compile: the first call of " ^ id ^ " failed")

let temp_dir () = Filename.temp_dir "bench-compile" ""

(* One sample: a fresh process compiling [id] over an empty disk cache. *)
let cold id () =
  let dir = temp_dir () in
  Fun.protect ~finally:(fun () -> remove dir) @@ fun () ->
  let env = if List.mem id parallel_cases then [||] else [| "PARALLEL=1" |] in
  first_call ~env dir id

(* A process ready to make a [first/] case's call over an empty disk cache:
   a line on [go] starts the call, and [said] says "ready", then "done". *)
type ready = { pid : int; go : out_channel; said : in_channel; dir : string }

let start_ready id =
  let dir = temp_dir () and exe = Sys.executable_name in
  let env =
    Array.append
      [| "CACHEDB=" ^ Filename.concat dir "cache.db"; "PARALLEL=1" |]
      (Unix.environment ())
  in
  let child_in, go = Unix.pipe ~cloexec:true () in
  let said, child_out = Unix.pipe ~cloexec:true () in
  let pid =
    Unix.create_process_env exe [| exe; "--first"; id |] env child_in child_out
      Unix.stderr
  in
  Unix.close child_in;
  Unix.close child_out;
  {
    pid;
    go = Unix.out_channel_of_descr go;
    said = Unix.in_channel_of_descr said;
    dir;
  }

let expect r line =
  match In_channel.input_line r.said with
  | Some l when l = line -> ()
  | _ -> failwith ("bench_compile: a first call's process did not say " ^ line)

let stop r =
  (try Unix.kill r.pid Sys.sigkill with Unix.Unix_error (ESRCH, _, _) -> ());
  ignore (Unix.waitpid [] r.pid);
  close_out_noerr r.go;
  close_in_noerr r.said;
  remove r.dir

(* A trial calls a case once to size its batches, at least 3 times to warm up
   and 3 times for its samples: a process each. This holds while one call
   passes the batch floor, so that every batch is one call: a first call of a
   few milliseconds under the floor would double the batch and run out of
   processes, which fails the case. *)
let ready_processes = 8

(* A [first/] case: its setup starts [ready_processes] processes and waits
   until each has prepared [id]'s call; a sample has the next one make it. *)
let first id =
  Thumper.bench_with_setup ("first/" ^ id)
    ~setup:(fun () ->
      let all = List.init ready_processes (fun _ -> start_ready id) in
      List.iter (fun r -> expect r "ready") all;
      (all, Queue.of_seq (List.to_seq all)))
    ~teardown:(fun (all, _) -> List.iter stop all)
    (fun (_, waiting) ->
      match Queue.take_opt waiting with
      | None -> failwith "bench_compile: no process left ready"
      | Some r ->
          output_string r.go "go\n";
          flush r.go;
          expect r "done")

(* A case whose samples read [id]'s programs back from the disk cache its
   setup fills. *)
let warm id =
  Thumper.bench_with_setup ("warm/" ^ id)
    ~setup:(fun () ->
      let dir = temp_dir () in
      first_call ~env:[||] dir id;
      dir)
    ~teardown:remove
    (fun dir -> first_call ~env:[||] dir id)

let config =
  Thumper.Config.(
    default |> samples 3 |> warmup 0. |> deadline 300.
    |> metrics [ Thumper.Metric.wall_time ])

let () =
  match Array.to_list Sys.argv with
  | [ _; "--cold"; id ] -> (
      match List.assoc_opt id cases with
      | Some f ->
          let t0 = Unix.gettimeofday () in
          Nx.Rng.with_key (Nx.Rng.key 42) f;
          Printf.printf "%.3f\n" ((Unix.gettimeofday () -. t0) *. 1000.)
      | None ->
          prerr_endline ("bench_compile: no case " ^ id);
          exit 2)
  | [ _; "--first"; id ] -> (
      match List.assoc_opt id first_cases with
      | Some prepare ->
          let call = Nx.Rng.with_key (Nx.Rng.key 42) prepare in
          print_endline "ready";
          ignore (In_channel.input_line stdin);
          call ();
          print_endline "done"
      | None ->
          prerr_endline ("bench_compile: no case " ^ id);
          exit 2)
  | [ _; "--opens"; gpu ] ->
      let opened =
        match gpu with
        | "amd" -> Result.is_ok (Rig.open_ (Rig_amd_amdgpu.gpu 0))
        | "metal" -> Result.is_ok (Rig.open_ (Rig_metal.gpu 0))
        | _ -> Result.is_ok (Rig.open_ (Rig_cuda.gpu 0))
      in
      exit (if opened then 0 else 1)
  | _ ->
      let vendors = [ "cuda"; "amd"; "metal" ] in
      let gpus = List.filter opens vendors in
      let runs id =
        match String.index_opt id '/' with
        | Some i when List.mem (String.sub id 0 i) vendors ->
            List.mem (String.sub id 0 i) gpus
        | _ -> true
      in
      Thumper.run "compile" ~config
        ~budgets:[ Thumper.Budget.no_slower_than 0.05 ]
        (List.filter_map
           (fun (id, _) ->
             if runs id && not (List.mem id warm_only) then
               Some (Thumper.bench id (cold id))
             else None)
           cases
        @ List.map warm (List.filter runs warm_cases)
        @ List.filter_map
            (fun (id, _) -> if runs id then Some (first id) else None)
            first_cases)
      |> exit
