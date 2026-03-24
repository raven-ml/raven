(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let sizes = [ 500; 1000 ]

let bench_name op size dtype =
  Printf.sprintf "%s %dx%d %s" op size size dtype

let ops_f32 ~size =
  let shape = [| size; size |] in
  let a = Nx.rand Nx.Float32 shape in
  let b = Nx.rand Nx.Float32 shape in
  [
    ("Add", fun () -> (Nx.add a b));
    ("Matmul", fun () -> (Nx.matmul a b));
  ]

let ops_f64 ~size =
  let shape = [| size; size |] in
  let a = Nx.rand Nx.Float64 shape in
  let b = Nx.rand Nx.Float64 shape in
  [
    ("Add", fun () -> (Nx.add a b));
    ("Matmul", fun () -> (Nx.matmul a b));
  ]

let benchmarks () =
  List.concat_map
    (fun size ->
      let f32 =
        List.map
          (fun (op, fn) -> Thumper.bench (bench_name op size "f32") fn)
          (ops_f32 ~size)
      in
      let f64 =
        List.map
          (fun (op, fn) -> Thumper.bench (bench_name op size "f64") fn)
          (ops_f64 ~size)
      in
      f32 @ f64)
    sizes
