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
  let a = Nx.rand Nx.Float32 ~key:(Nx.Rng.key (size * 3)) shape in
  let b = Nx.rand Nx.Float32 ~key:(Nx.Rng.key ((size * 3) + 1)) shape in
  let out_c = Nx.empty Nx.float32 shape in
  let out_c_scalar = Nx.empty Nx.float32 [||] in
  let ctx = Nx_backend.create_context () in
  let a_fe = Nx_ox.empty ctx Nx_ox.float32 shape in
  let b_fe = Nx_ox.empty ctx Nx_ox.float32 shape in
  let out_fe = Nx_ox.empty ctx Nx_ox.float32 shape in
  let out_fe_scalar = Nx_ox.empty ctx Nx_ox.float32 [||] in
  let bin_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c a b));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe a_fe b_fe));
    ]
  in
  let unary_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c a));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe a_fe));
    ]
  in
  let reduce_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c_scalar a));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe_scalar a_fe));
    ]
  in
  [
    bin_pair "Add" (fun ~out a b -> Nx.add ~out a b) (fun ~out a b -> Nx_ox.add ~out a b);
    bin_pair "Matmul" (fun ~out a b -> Nx.matmul ~out a b) (fun ~out a b -> Nx_ox.matmul ~out a b);
    (* bin_pair "Sub" (fun ~out a b -> Nx.sub ~out a b) (fun ~out a b -> Nx_ox.sub ~out a b); *)
    (* bin_pair "Mul" (fun ~out a b -> Nx.mul ~out a b) (fun ~out a b -> Nx_ox.mul ~out a b); *)
    (* bin_pair "Div" (fun ~out a b -> Nx.div ~out a b) (fun ~out a b -> Nx_ox.div ~out a b); *)
    (* bin_pair "Mod" (fun ~out a b -> Nx.mod_ ~out a b) (fun ~out a b -> Nx_ox.mod_ ~out a b); *)
    (* bin_pair "Pow" (fun ~out a b -> Nx.pow ~out a b) (fun ~out a b -> Nx_ox.pow ~out a b); *)
    (* bin_pair "Max" (fun ~out a b -> Nx.maximum ~out a b) (fun ~out a b -> Nx_ox.maximum ~out a b); *)
    (* bin_pair "Min" (fun ~out a b -> Nx.minimum ~out a b) (fun ~out a b -> Nx_ox.minimum ~out a b); *)
    (* unary_pair "Neg" (fun ~out a -> Nx.neg ~out a) (fun ~out a -> Nx_ox.neg ~out a); *)
    (* unary_pair "Abs" (fun ~out a -> Nx.abs ~out a) (fun ~out a -> Nx_ox.abs ~out a); *)
    (* unary_pair "Sqrt" (fun ~out a -> Nx.sqrt ~out a) (fun ~out a -> Nx_ox.sqrt ~out a); *)
    (* unary_pair "Exp" (fun ~out a -> Nx.exp ~out a) (fun ~out a -> Nx_ox.exp ~out a); *)
    (* unary_pair "Log" (fun ~out a -> Nx.log ~out a) (fun ~out a -> Nx_ox.log ~out a); *)
    (* unary_pair "Sin" (fun ~out a -> Nx.sin ~out a) (fun ~out a -> Nx_ox.sin ~out a); *)
    (* unary_pair "Cos" (fun ~out a -> Nx.cos ~out a) (fun ~out a -> Nx_ox.cos ~out a); *)
    (* reduce_pair "Reduce_sum" (fun ~out a -> Nx.sum ~out a) (fun ~out a -> Nx_ox.sum ~out a); *)
    (* reduce_pair "Reduce_prod" (fun ~out a -> Nx.prod ~out a) (fun ~out a -> Nx_ox.prod ~out a); *)
    (* reduce_pair "Reduce_max" (fun ~out a -> Nx.max ~out a) (fun ~out a -> Nx_ox.max ~out a); *)
    (* reduce_pair "Reduce_min" (fun ~out a -> Nx.min ~out a) (fun ~out a -> Nx_ox.min ~out a); *)
  ]
  |> List.concat

let ops_f64 ~size =
  let shape = [| size; size |] in
  let a = Nx.rand Nx.Float64 ~key:(Nx.Rng.key (size * 3)) shape in
  let b = Nx.rand Nx.Float64 ~key:(Nx.Rng.key ((size * 3) + 1)) shape in
  let out_c = Nx.empty Nx.float64 shape in
  let out_c_scalar = Nx.empty Nx.float64 [||] in
  let ctx = Nx_backend.create_context () in
  let a_fe = Nx_ox.empty ctx Nx_ox.float64 shape in
  let b_fe = Nx_ox.empty ctx Nx_ox.float64 shape in
  let out_fe = Nx_ox.empty ctx Nx_ox.float64 shape in
  let out_fe_scalar = Nx_ox.empty ctx Nx_ox.float64 [||] in
  let bin_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c a b));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe a_fe b_fe));
    ]
  in
  let unary_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c a));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe a_fe));
    ]
  in
  let reduce_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c_scalar a));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe_scalar a_fe));
    ]
  in
  [
    bin_pair "Add" (fun ~out a b -> Nx.add ~out a b) (fun ~out a b -> Nx_ox.add ~out a b);
    bin_pair "Matmul" (fun ~out a b -> Nx.matmul ~out a b) (fun ~out a b -> Nx_ox.matmul ~out a b);
    (* bin_pair "Sub" (fun ~out a b -> Nx.sub ~out a b) (fun ~out a b -> Nx_ox.sub ~out a b); *)
    (* bin_pair "Mul" (fun ~out a b -> Nx.mul ~out a b) (fun ~out a b -> Nx_ox.mul ~out a b); *)
    (* bin_pair "Div" (fun ~out a b -> Nx.div ~out a b) (fun ~out a b -> Nx_ox.div ~out a b); *)
    (* bin_pair "Mod" (fun ~out a b -> Nx.mod_ ~out a b) (fun ~out a b -> Nx_ox.mod_ ~out a b); *)
    (* bin_pair "Pow" (fun ~out a b -> Nx.pow ~out a b) (fun ~out a b -> Nx_ox.pow ~out a b); *)
    (* bin_pair "Max" (fun ~out a b -> Nx.maximum ~out a b) (fun ~out a b -> Nx_ox.maximum ~out a b); *)
    (* bin_pair "Min" (fun ~out a b -> Nx.minimum ~out a b) (fun ~out a b -> Nx_ox.minimum ~out a b); *)
    (* unary_pair "Neg" (fun ~out a -> Nx.neg ~out a) (fun ~out a -> Nx_ox.neg ~out a); *)
    (* unary_pair "Abs" (fun ~out a -> Nx.abs ~out a) (fun ~out a -> Nx_ox.abs ~out a); *)
    (* unary_pair "Sqrt" (fun ~out a -> Nx.sqrt ~out a) (fun ~out a -> Nx_ox.sqrt ~out a); *)
    (* unary_pair "Exp" (fun ~out a -> Nx.exp ~out a) (fun ~out a -> Nx_ox.exp ~out a); *)
    (* unary_pair "Log" (fun ~out a -> Nx.log ~out a) (fun ~out a -> Nx_ox.log ~out a); *)
    (* unary_pair "Sin" (fun ~out a -> Nx.sin ~out a) (fun ~out a -> Nx_ox.sin ~out a); *)
    (* unary_pair "Cos" (fun ~out a -> Nx.cos ~out a) (fun ~out a -> Nx_ox.cos ~out a); *)
    (* reduce_pair "Reduce_sum" (fun ~out a -> Nx.sum ~out a) (fun ~out a -> Nx_ox.sum ~out a); *)
    (* reduce_pair "Reduce_prod" (fun ~out a -> Nx.prod ~out a) (fun ~out a -> Nx_ox.prod ~out a); *)
    (* reduce_pair "Reduce_max" (fun ~out a -> Nx.max ~out a) (fun ~out a -> Nx_ox.max ~out a); *)
    (* reduce_pair "Reduce_min" (fun ~out a -> Nx.min ~out a) (fun ~out a -> Nx_ox.min ~out a); *)
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
