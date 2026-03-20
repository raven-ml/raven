(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Configuration *)
let sizes = [ 50; 100; 200; 500 ]
let backend_name = "Nx"

let benchmark_name op_name size dtype_label =
  Printf.sprintf "%s %dx%d %s (%s)" op_name size size dtype_label backend_name

let nx_operations_f32 ~size =
  let shape = [| size; size |] in
  let a = Nx.rand Nx.Float32 shape in
  let b = Nx.rand Nx.Float32 shape in

  let ops =
    [
      ("Add", fun () -> Thumper.consume (Nx.add a b));
      ("Mul", fun () -> Thumper.consume (Nx.mul a b));
    ]
  in

  let ops =
    ops
    @ [
        ("Sum", fun () -> Thumper.consume (Nx.sum a));
        ("Transpose", fun () -> Thumper.consume (Nx.transpose a));
      ]
  in

  ops

let nx_operations_f64 ~size =
  let shape = [| size; size |] in
  let a = Nx.rand Nx.Float64 shape in
  let b = Nx.rand Nx.Float64 shape in

  let ops =
    [
      ("Add", fun () -> Thumper.consume (Nx.add a b));
      ("Mul", fun () -> Thumper.consume (Nx.mul a b));
    ]
  in

  let ops =
    ops
    @ [
        ("Sum", fun () -> Thumper.consume (Nx.sum a));
        ("Transpose", fun () -> Thumper.consume (Nx.transpose a));
      ]
  in

  ops

let build_benchmarks () =
  let f32_benches = ref [] in
  let f64_benches = ref [] in
  List.iter
    (fun size ->
      let ops_f32 = nx_operations_f32 ~size in
      List.iter
        (fun (op_name, fn) ->
          let bench_name = benchmark_name op_name size "f32" in
          f32_benches := Thumper.bench bench_name fn :: !f32_benches)
        ops_f32;

      let ops_f64 = nx_operations_f64 ~size in
      List.iter
        (fun (op_name, fn) ->
          let bench_name = benchmark_name op_name size "f64" in
          f64_benches := Thumper.bench bench_name fn :: !f64_benches)
        ops_f64)
    sizes;
  [
    Thumper.group "f32" (List.rev !f32_benches);
    Thumper.group "f64" (List.rev !f64_benches);
  ]

let () =
  let benchmarks = build_benchmarks () in
  Thumper.run "nx" benchmarks
