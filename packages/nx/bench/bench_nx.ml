(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx core operations across the regimes a perf change must not silently
   regress: elementwise binary and unary, reductions along each axis, and
   structural materialization. Inputs are allocated in the group builders and
   captured in each closure, so only the operation and its output allocation
   are timed.

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
    Thumper.bench ~tags:lab "relu 1M" (fun () -> Nx.relu flat);
    Thumper.bench "abs 512x512" (fun () -> Nx.abs mat);
  ]

let reduce_benchmarks () =
  let small = Nx.rand Nx.Float32 [| 128; 128 |] in
  let flat = Nx.rand Nx.Float32 [| 1_000_000 |] in
  let mat = Nx.rand Nx.Float32 [| 512; 512 |] in
  [
    Thumper.bench ~tags:lab "sum 128x128" (fun () -> Nx.sum small);
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
  ]

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  Thumper.run "nx"
    ~budgets:
      [
        Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05;
        Thumper.Budget.no_more_alloc_than 0.01;
      ]
    [
      Thumper.group "binary" (binary_benchmarks ());
      Thumper.group "unary" (unary_benchmarks ());
      Thumper.group "reduce" (reduce_benchmarks ());
      Thumper.group "structural" (structural_benchmarks ());
    ]
