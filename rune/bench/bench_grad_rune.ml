(** Minimal gradient benchmark suite for Rune autodiff *)

open Rune

(** Benchmark sizes - focus on realistic ML workload sizes *)
let sizes =
  [
    ("Small", 100);
    (* Small batch/feature size *)
    ("Medium", 500);
    (* Medium neural network layer *)
    ("Large", 1000);
    (* Large neural network layer *)
  ]

let backend_name = "Rune"

(** Helper to create benchmark name *)
let benchmark_name op_name size_name =
  Printf.sprintf "%s %s (%s)" op_name size_name backend_name

(** 1. Scalar→Scalar: f(x) = x^2 *)
let scalar_grad_benchmarks () =
  let f x = square x in

  List.map
    (fun (size_name, _) ->
      let x = scalar float32 5.0 in
      let bench_name = benchmark_name "ScalarGrad" size_name in
      Ubench.bench bench_name (fun () -> ignore (grad f x)))
    sizes

(** 2. Vector→Scalar: f(x) = sum(x^2) (L2 norm squared) *)
let vector_scalar_grad_benchmarks () =
  List.map
    (fun (size_name, size) ->
      let x = randn float32 [| size |] in
      let f x = sum (square x) in
      let bench_name = benchmark_name "VectorGrad" size_name in
      Ubench.bench bench_name (fun () -> ignore (grad f x)))
    sizes

(** 3. MatMul gradient: f(x) = sum(matmul(x, W)) *)
let matmul_grad_benchmarks () =
  List.map
    (fun (size_name, size) ->
      let x = randn float32 [| size; size |] in
      let w = randn float32 [| size; size |] in
      let f x = sum (matmul x w) in
      let bench_name = benchmark_name "MatMulGrad" size_name in
      Ubench.bench bench_name (fun () -> ignore (grad f x)))
    sizes

(** 4. Chain of operations: f(x) = sum(exp(tanh(x^2))) *)
let chain_grad_benchmarks () =
  List.map
    (fun (size_name, size) ->
      let x = randn float32 [| size; size |] in
      let f x = sum (exp (tanh (square x))) in
      let bench_name = benchmark_name "ChainGrad" size_name in
      Ubench.bench bench_name (fun () -> ignore (grad f x)))
    sizes

(** 5. Higher-order gradient: grad(grad(f)) where f(x) = sum(x^3) *)
let higher_order_grad_benchmarks () =
  List.map
    (fun (size_name, size) ->
      let x = randn float32 [| size |] in
      let f x = sum (mul (mul x x) x) in
      (* x^3 as x * x * x *)
      let grad_f = grad f in
      let grad_grad_f = grad (fun x -> sum (grad_f x)) in
      let bench_name = benchmark_name "HigherOrderGrad" size_name in
      Ubench.bench bench_name (fun () -> ignore (grad_grad_f x)))
    sizes

(** Build all benchmarks *)
let build_benchmarks () =
  List.concat
    [
      scalar_grad_benchmarks ();
      vector_scalar_grad_benchmarks ();
      matmul_grad_benchmarks ();
      chain_grad_benchmarks ();
      higher_order_grad_benchmarks ();
    ]

(** Main entry point *)
let () =
  let benchmarks = build_benchmarks () in
  Ubench.run_cli benchmarks
