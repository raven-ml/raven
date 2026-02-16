(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let backend_name = "Nx"

type matmul_case = { name : string; m : int; k : int; n : int; seed : int }

let cases =
  [
    { name = "SquareSmall"; m = 64; k = 64; n = 64; seed = 11 };
    { name = "TallSkinny"; m = 256; k = 64; n = 256; seed = 17 };
    { name = "Wide"; m = 128; k = 256; n = 64; seed = 23 };
    { name = "SquareLarge"; m = 512; k = 512; n = 512; seed = 29 };
  ]

let benchmark_name case dtype_label suffix =
  Printf.sprintf "MatMul %s %dx%d @ %dx%d %s%s (%s)" case.name case.m case.k
    case.k case.n dtype_label suffix backend_name

let setup_operands (type a b) (dtype : (a, b) Nx.dtype) case =
  let lhs = Nx.rand dtype ~key:(Nx.Rng.key case.seed) [| case.m; case.k |] in
  let rhs =
    Nx.rand dtype ~key:(Nx.Rng.key (case.seed + 1)) [| case.k; case.n |]
  in
  (lhs, rhs)

let add_case (type a b) benches case (dtype : (a, b) Nx.dtype) dtype_label =
  let lhs, rhs = setup_operands dtype case in
  let name = benchmark_name case dtype_label "" in
  let fn () = ignore (Nx.matmul lhs rhs) in
  benches := Ubench.bench name fn :: !benches;
  let out = Nx.empty dtype [| case.m; case.n |] in
  let name_reuse = benchmark_name case dtype_label " reuse" in
  let fn_reuse () = ignore (Nx.matmul ~out lhs rhs) in
  benches := Ubench.bench name_reuse fn_reuse :: !benches

let build_benchmarks () =
  let benches = ref [] in
  List.iter
    (fun case ->
      add_case benches case Nx.Float32 "f32";
      add_case benches case Nx.Float64 "f64")
    cases;
  List.rev !benches

let default_config () =
  let open Ubench.Config in
  default |> time_limit 1.0 |> warmup 1 |> min_measurements 5
  |> geometric_scale 1.3 |> gc_stabilization false |> build

let () =
  let benchmarks = build_benchmarks () in
  let config = default_config () in
  ignore (Ubench.run ~config benchmarks)
