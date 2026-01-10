(*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Dtype = Nx_core.Dtype
module Oxfe = Nx_core.Make_frontend (Nx_oxcaml)

let sizes = [ 200; 500; 1000 ]

let bench_name op size dtype backend =
  Printf.sprintf "%s %dx%d %s (%s)" op size size dtype backend

let ops_f32 ~size =
  let shape = [| size; size |] in
  let numel = size * size in
  let a = Nx.rand Nx.Float32 ~key:(Nx.Rng.key (size * 3)) shape in
  let b = Nx.rand Nx.Float32 ~key:(Nx.Rng.key ((size * 3) + 1)) shape in
  let ctx = Nx_oxcaml.create_context () in
  let a_ox = Nx_oxcaml.op_buffer ctx Dtype.Float32 numel in
  let b_ox = Nx_oxcaml.op_buffer ctx Dtype.Float32 numel in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 numel in
  let a_fe = Oxfe.empty ctx Oxfe.float32 shape in
  let b_fe = Oxfe.empty ctx Oxfe.float32 shape in
  [
    ("Add", "Nx", fun () -> ignore (Nx.add a b));
    ("Add", "Nx_oxcaml", fun () -> Nx_oxcaml.op_add ~out a_ox b_ox);
    ("Add", "Nx_oxcaml_frontend", fun () -> ignore (Oxfe.add a_fe b_fe));
    ("Sub", "Nx", fun () -> ignore (Nx.sub a b));
    ("Sub", "Nx_oxcaml", fun () -> Nx_oxcaml.op_sub ~out a_ox b_ox);
    ("Sub", "Nx_oxcaml_frontend", fun () -> ignore (Oxfe.sub a_fe b_fe));
  ]

let ops_f64 ~size =
  let shape = [| size; size |] in
  let numel = size * size in
  let a = Nx.rand Nx.Float64 ~key:(Nx.Rng.key (size * 3)) shape in
  let b = Nx.rand Nx.Float64 ~key:(Nx.Rng.key ((size * 3) + 1)) shape in
  let ctx = Nx_oxcaml.create_context () in
  let a_ox = Nx_oxcaml.op_buffer ctx Dtype.Float64 numel in
  let b_ox = Nx_oxcaml.op_buffer ctx Dtype.Float64 numel in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 numel in
  let a_fe = Oxfe.empty ctx Oxfe.float64 shape in
  let b_fe = Oxfe.empty ctx Oxfe.float64 shape in
  [
    ("Add", "Nx", fun () -> ignore (Nx.add a b));
    ("Add", "Nx_oxcaml", fun () -> Nx_oxcaml.op_add ~out a_ox b_ox);
    ("Add", "Nx_oxcaml_frontend", fun () -> ignore (Oxfe.add a_fe b_fe));
    ("Sub", "Nx", fun () -> ignore (Nx.sub a b));
    ("Sub", "Nx_oxcaml", fun () -> Nx_oxcaml.op_sub ~out a_ox b_ox);
    ("Mod", "Nx_oxcaml", fun () -> Nx_oxcaml.op_mod ~out a_ox b_ox);
    ("Pow", "Nx_oxcaml", fun () -> Nx_oxcaml.op_pow ~out a_ox b_ox);
    ("Sub", "Nx_oxcaml_frontend", fun () -> ignore (Oxfe.sub a_fe b_fe));
  ]

let build_benchmarks () =
  List.concat_map
    (fun size ->
      let f32 =
        List.map
          (fun (op, backend, fn) ->
            Ubench.bench (bench_name op size "f32" backend) fn)
          (ops_f32 ~size)
      in
      let f64 =
        List.map
          (fun (op, backend, fn) ->
            Ubench.bench (bench_name op size "f64" backend) fn)
          (ops_f64 ~size)
      in
      f32 @ f64)
    sizes

let config =
  Ubench.Config.(
    default |> time_limit 1.0 |> warmup 1 |> min_measurements 5
    |> geometric_scale 1.3 |> gc_stabilization false |> build)

let () = ignore (Ubench.run ~config (build_benchmarks ()))
