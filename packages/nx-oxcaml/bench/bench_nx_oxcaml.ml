(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

[@@@warning "-26"]

module Nx_ox = Nx_core.Make_frontend (Nx_backend)

let sizes = [ 500; 1000 ]

let bench_name op size dtype backend =
  Printf.sprintf "%s %dx%d %s (%s)" op size size dtype backend

let ops_f32 ~size =
  let shape = [| size; size |] in
  let a = Nx.rand Nx.Float32 shape in
  let b = Nx.rand Nx.Float32 shape in
  let ctx = Nx_backend.create_context () in
  let a_fe = Nx_ox.empty ctx Nx_ox.float32 shape in
  let b_fe = Nx_ox.empty ctx Nx_ox.float32 shape in
  let bin_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op a b));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op a_fe b_fe));
    ]
  in
  let _unary_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op a));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op a_fe));
    ]
  in
  let _reduce_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op a));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op a_fe));
    ]
  in
  [
    bin_pair "Add" (fun a b -> Nx.add a b) (fun a b -> Nx_ox.add a b);
    bin_pair "Matmul" (fun a b -> Nx.matmul a b) (fun a b -> Nx_ox.matmul a b);
  ]
  |> List.concat

let ops_f64 ~size =
  let shape = [| size; size |] in
  let a = Nx.rand Nx.Float64 shape in
  let b = Nx.rand Nx.Float64 shape in
  let ctx = Nx_backend.create_context () in
  let a_fe = Nx_ox.empty ctx Nx_ox.float64 shape in
  let b_fe = Nx_ox.empty ctx Nx_ox.float64 shape in
  let bin_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op a b));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op a_fe b_fe));
    ]
  in
  let _unary_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op a));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op a_fe));
    ]
  in
  let _reduce_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op a));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op a_fe));
    ]
  in
  [
    bin_pair "Add" (fun a b -> Nx.add a b) (fun a b -> Nx_ox.add a b);
    bin_pair "Matmul" (fun a b -> Nx.matmul a b) (fun a b -> Nx_ox.matmul a b);
  ]
  |> List.concat

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

let () = ignore (Ubench.run ~config ~sort_by_wall:false (build_benchmarks ()))
