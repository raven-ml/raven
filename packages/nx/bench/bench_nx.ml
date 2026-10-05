(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx core operations across the regimes a perf change must not silently
   regress: elementwise binary and unary, reductions along each axis, and
   structural materialization. Inputs are allocated in the group builders and
   captured in each closure, so only the operation and its output allocation are
   timed.

   The [lab] subset is the fast, representative slice the perf loop optimizes.
   It keeps the documented anomalies -- the f64 elementwise cliff at 100x100,
   the reduction threshold near 128x128 -- and the non-contiguous paths
   (operating on a transposed view, reducing along a strided axis, materializing
   a transpose) so wins on those stay findable. *)

let lab = [ "lab" ]

let binary_benchmarks () =
  let f32_1m_a = Nx.rand Nx.Float32 [| 1_000_000 |] in
  let f32_1m_b = Nx.rand Nx.Float32 [| 1_000_000 |] in
  let f32_512_a = Nx.rand Nx.Float32 [| 512; 512 |] in
  let f32_512_b = Nx.rand Nx.Float32 [| 512; 512 |] in
  let f64_512_a = Nx.rand Nx.Float64 [| 512; 512 |] in
  let f64_512_b = Nx.rand Nx.Float64 [| 512; 512 |] in
  let f64_100_a = Nx.rand Nx.Float64 [| 100; 100 |] in
  let f64_100_b = Nx.rand Nx.Float64 [| 100; 100 |] in
  let bc_lhs = Nx.rand Nx.Float32 [| 1024; 1024 |] in
  let bc_rhs = Nx.rand Nx.Float32 [| 1024; 1 |] in
  let nc_dense = Nx.rand Nx.Float32 [| 512; 512 |] in
  let nc_view = Nx.transpose (Nx.rand Nx.Float32 [| 512; 512 |]) in
  [
    Thumper.bench ~tags:lab "add 1M" (fun () -> Nx.add f32_1m_a f32_1m_b);
    Thumper.bench ~tags:lab "add 512x512" (fun () -> Nx.add f32_512_a f32_512_b);
    Thumper.bench ~tags:lab "add 100x100 f64" (fun () ->
        Nx.add f64_100_a f64_100_b);
    Thumper.bench "add 512x512 f64" (fun () -> Nx.add f64_512_a f64_512_b);
    Thumper.bench "sub 512x512" (fun () -> Nx.sub f32_512_a f32_512_b);
    Thumper.bench ~tags:lab "mul 1M" (fun () -> Nx.mul f32_1m_a f32_1m_b);
    Thumper.bench ~tags:lab "mul 100x100 f64" (fun () ->
        Nx.mul f64_100_a f64_100_b);
    Thumper.bench "mul 512x512" (fun () -> Nx.mul f32_512_a f32_512_b);
    Thumper.bench "div 512x512" (fun () -> Nx.div f32_512_a f32_512_b);
    Thumper.bench ~tags:lab "add broadcast [1024x1024]+[1024x1]" (fun () ->
        Nx.add bc_lhs bc_rhs);
    Thumper.bench ~tags:lab "add noncontig (transpose view) 512x512" (fun () ->
        Nx.add nc_dense nc_view);
  ]

let unary_benchmarks () =
  let flat = Nx.rand Nx.Float32 [| 1_000_000 |] in
  let mat = Nx.rand Nx.Float32 [| 512; 512 |] in
  [
    Thumper.bench ~tags:lab "exp 1M" (fun () -> Nx.exp flat);
    Thumper.bench "log 512x512" (fun () -> Nx.log mat);
    Thumper.bench "sqrt 1M" (fun () -> Nx.sqrt flat);
    Thumper.bench "neg 512x512" (fun () -> Nx.neg mat);
    Thumper.bench "abs 512x512" (fun () -> Nx.abs mat);
  ]

let reduce_benchmarks () =
  let small = Nx.rand Nx.Float32 [| 128; 128 |] in
  let flat = Nx.rand Nx.Float32 [| 1_000_000 |] in
  let mat = Nx.rand Nx.Float32 [| 512; 512 |] in
  let transposed = Nx.transpose (Nx.rand Nx.Float32 [| 2048; 2048 |]) in
  let wide = Nx.rand Nx.Float32 [| 32; 262144 |] in
  let short_runs = Nx.rand Nx.Float32 [| 65536; 16; 2 |] in
  [
    Thumper.bench ~tags:lab "sum 128x128" (fun () -> Nx.sum small);
    Thumper.bench "sum transposed 2048x2048" (fun () -> Nx.sum transposed);
    Thumper.bench "sum axis0 32x262144" (fun () -> Nx.sum ~axes:[ 0 ] wide);
    Thumper.bench "sum axes02 65536x16x2" (fun () ->
        Nx.sum ~axes:[ 0; 2 ] short_runs);
    Thumper.bench ~tags:lab "sum full 1M" (fun () -> Nx.sum flat);
    Thumper.bench ~tags:lab "sum axis0 512x512" (fun () ->
        Nx.sum ~axes:[ 0 ] mat);
    Thumper.bench ~tags:lab "sum axis1 512x512" (fun () ->
        Nx.sum ~axes:[ 1 ] mat);
    Thumper.bench "max axis1 512x512" (fun () -> Nx.max ~axes:[ 1 ] mat);
    Thumper.bench "mean axis0 512x512" (fun () -> Nx.mean ~axes:[ 0 ] mat);
    Thumper.bench "argmax axis1 512x512" (fun () -> Nx.argmax ~axis:1 mat);
  ]

let structural_benchmarks () =
  let flat = Nx.rand Nx.Float32 [| 1_000_000 |] in
  let transpose_view = Nx.transpose (Nx.rand Nx.Float32 [| 512; 512 |]) in
  let cat_a = Nx.rand Nx.Float32 [| 512; 512 |] in
  let cat_b = Nx.rand Nx.Float32 [| 512; 512 |] in
  let gather_source = Nx.rand Nx.Float32 [| 4096; 256 |] in
  let gather_indices =
    Nx.create Nx.Int64 [| 1024 |]
      (Array.init 1024 (fun i -> Int64.of_int (i * 37 mod 4096)))
  in
  let sort_input = Nx.rand Nx.Float32 [| 512; 512 |] in
  [
    Thumper.bench ~tags:lab "contiguous of transpose 512x512" (fun () ->
        Nx.contiguous transpose_view);
    Thumper.bench "reshape 1M→1000x1000" (fun () ->
        Nx.reshape [| 1000; 1000 |] flat);
    Thumper.bench ~tags:lab "concatenate axis0 two 512x512" (fun () ->
        Nx.concatenate ~axis:0 [ cat_a; cat_b ]);
    Thumper.bench "cast f32→f16 1M" (fun () -> Nx.cast Nx.Float16 flat);
    Thumper.bench "cast f32→i32 1M" (fun () -> Nx.cast Nx.Int32 flat);
    Thumper.bench "copy 1M" (fun () -> Nx.copy flat);
    Thumper.bench "gather 1024 rows from 4096x256" (fun () ->
        Nx.take ~axis:0 ~indices:gather_indices gather_source);
    Thumper.bench "sort rows 512x512" (fun () -> Nx.sort sort_input);
  ]

(* The fixed cost of an operation: routing, effects and allocation around a
   kernel that has almost nothing to do. One element exposes that cost alone;
   1,024 elements show it beside a small kernel. *)
let dispatch_benchmarks () =
  let one = Nx.rand Nx.Float32 [| 1 |] and one' = Nx.rand Nx.Float32 [| 1 |] in
  let row = Nx.rand Nx.Float32 [| 1024 |] in
  let row' = Nx.rand Nx.Float32 [| 1024 |] in
  let one_mat = Nx.rand Nx.Float32 [| 1; 1 |] in
  let mat = Nx.rand Nx.Float32 [| 32; 32 |] in
  let one_mask = Nx.less one one' and mask = Nx.less row row' in
  [
    Thumper.bench "add 1" (fun () -> Nx.add one one');
    Thumper.bench "add 1024" (fun () -> Nx.add row row');
    Thumper.bench "less 1" (fun () -> Nx.less one one');
    Thumper.bench "less 1024" (fun () -> Nx.less row row');
    Thumper.bench "where 1" (fun () -> Nx.where one_mask one one');
    Thumper.bench "where 1024" (fun () -> Nx.where mask row row');
    Thumper.bench "sum 1" (fun () -> Nx.sum one);
    Thumper.bench "sum 1024" (fun () -> Nx.sum row);
    Thumper.bench "matmul 1x1" (fun () -> Nx.matmul one_mat one_mat);
    Thumper.bench "matmul 32x32" (fun () -> Nx.matmul mat mat);
    Thumper.bench "zeros 1" (fun () -> Nx.zeros Nx.Float32 [| 1 |]);
    Thumper.bench "shape" (fun () -> Nx.shape row);
  ]

(* Samplers at the sizes a training step draws: a dropout mask or an init at a
   million elements, and the rejection samplers, whose cost per element is the
   question, at fewer. Poisson is measured at one rate per regime of its
   algorithm. *)
let random_benchmarks () =
  let key = Nx.Rng.key 7 in
  let f32 = Nx.Float32 in
  let large = [| 1_000_000 |] in
  let param shape v = Nx.broadcast_to shape (Nx.scalar f32 v) in
  let p = param large 0.9 in
  let concentration = param [| 100_000 |] 2.5 in
  let rate r = param [| 10_000 |] r in
  let rate_1 = rate 1.0 and rate_30 = rate 30.0 and rate_100 = rate 100.0 in
  [
    Thumper.bench "uniform 1M" (fun () -> Nx.Rng.uniform key f32 large);
    Thumper.bench "normal 1M" (fun () -> Nx.Rng.normal key f32 large);
    Thumper.bench "bernoulli 1M" (fun () -> Nx.Rng.bernoulli key p);
    Thumper.bench "gamma 100k" (fun () -> Nx.Rng.gamma key concentration);
    Thumper.bench "poisson rate 1 10k" (fun () -> Nx.Rng.poisson key rate_1);
    Thumper.bench "poisson rate 30 10k" (fun () -> Nx.Rng.poisson key rate_30);
    Thumper.bench "poisson rate 100 10k" (fun () -> Nx.Rng.poisson key rate_100);
  ]

let () =
  Nx.Rng.with_key (Nx.Rng.key 42) @@ fun () ->
  Thumper.run "nx"
    ~budgets:
      [
        Thumper.Budget.no_slower_than 0.05;
        Thumper.Budget.no_more_alloc_than 0.01;
      ]
    [
      Thumper.group "dispatch" (dispatch_benchmarks ());
      Thumper.group "binary" (binary_benchmarks ());
      Thumper.group "unary" (unary_benchmarks ());
      Thumper.group "reduce" (reduce_benchmarks ());
      Thumper.group "structural" (structural_benchmarks ());
      Thumper.group "random" (random_benchmarks ());
    ]
  |> exit
