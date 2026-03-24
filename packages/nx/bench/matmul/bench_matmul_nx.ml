(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let backend_name = "Nx"

type matmul_case = { name : string; m : int; k : int; n : int }

let cases =
  [
    { name = "SquareSmall"; m = 64; k = 64; n = 64 };
    { name = "TallSkinny"; m = 256; k = 64; n = 256 };
    { name = "Wide"; m = 128; k = 256; n = 64 };
    { name = "SquareLarge"; m = 512; k = 512; n = 512 };
  ]

let benchmark_name case dtype_label suffix =
  Printf.sprintf "MatMul %s %dx%d @ %dx%d %s%s (%s)" case.name case.m case.k
    case.k case.n dtype_label suffix backend_name

let setup_operands (type a b) (dtype : (a, b) Nx.dtype) case =
  let lhs = Nx.rand dtype [| case.m; case.k |] in
  let rhs = Nx.rand dtype [| case.k; case.n |] in
  (lhs, rhs)

let add_case (type a b) benches case (dtype : (a, b) Nx.dtype) dtype_label =
  let lhs, rhs = setup_operands dtype case in
  let name = benchmark_name case dtype_label "" in
  let fn () = (Nx.matmul lhs rhs) in
  benches := Thumper.bench name fn :: !benches

let build_benchmarks () =
  let f32_benches = ref [] in
  let f64_benches = ref [] in
  List.iter
    (fun case ->
      add_case f32_benches case Nx.Float32 "f32";
      add_case f64_benches case Nx.Float64 "f64")
    cases;
  [
    Thumper.group "f32" (List.rev !f32_benches);
    Thumper.group "f64" (List.rev !f64_benches);
  ]

let () =
  let benchmarks = build_benchmarks () in
  Thumper.run "nx_matmul" benchmarks
