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
  let cond_c = Nx.less a b in
  let out_c = Nx.empty Nx.float32 shape in
  let out_c_scalar = Nx.empty Nx.float32 [||] in
  let out_c_bool = Nx.empty Nx.bool shape in
  let ctx = Nx_backend.create_context () in
  let a_fe = Nx_ox.empty ctx Nx_ox.float32 shape in
  let b_fe = Nx_ox.empty ctx Nx_ox.float32 shape in
  let cond_fe = Nx_ox.less a_fe b_fe in
  let out_fe = Nx_ox.empty ctx Nx_ox.float32 shape in
  let out_fe_scalar = Nx_ox.empty ctx Nx_ox.float32 [||] in
  let out_fe_bool = Nx_ox.empty ctx Nx_ox.bool shape in
  let bin_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c a b));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe a_fe b_fe));
    ]
  in
  let cmp_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c_bool a b));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe_bool a_fe b_fe));
    ]
  in
  let unary_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c a));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe a_fe));
    ]
  in
  let ternary_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c cond_c a b));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe cond_fe a_fe b_fe));
    ]
  in
  let reduce_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c_scalar a));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe_scalar a_fe));
    ]
  in
  let no_out_unary_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op a));
      (name, "Nx (OxCaml)", fun () -> ignore (ox_op a_fe));
    ]
  in
  let no_out_sort_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op a));
      (name, "Nx (OxCaml)", fun () -> ignore (ox_op a_fe));
    ]
  in
  [
    bin_pair "Add" (fun ~out a b -> Nx.add ~out a b) (fun ~out a b -> Nx_ox.add ~out a b);
    bin_pair "Matmul" (fun ~out a b -> Nx.matmul ~out a b) (fun ~out a b -> Nx_ox.matmul ~out a b);
    bin_pair "Sub" (fun ~out a b -> Nx.sub ~out a b) (fun ~out a b -> Nx_ox.sub ~out a b);
    bin_pair "Mul" (fun ~out a b -> Nx.mul ~out a b) (fun ~out a b -> Nx_ox.mul ~out a b);
    bin_pair "Div" (fun ~out a b -> Nx.div ~out a b) (fun ~out a b -> Nx_ox.div ~out a b);
    bin_pair "Mod" (fun ~out a b -> Nx.mod_ ~out a b) (fun ~out a b -> Nx_ox.mod_ ~out a b);
    bin_pair "Pow" (fun ~out a b -> Nx.pow ~out a b) (fun ~out a b -> Nx_ox.pow ~out a b);
    bin_pair "Atan2" (fun ~out a b -> Nx.atan2 ~out a b) (fun ~out a b -> Nx_ox.atan2 ~out a b);
    bin_pair "Max" (fun ~out a b -> Nx.maximum ~out a b) (fun ~out a b -> Nx_ox.maximum ~out a b);
    bin_pair "Min" (fun ~out a b -> Nx.minimum ~out a b) (fun ~out a b -> Nx_ox.minimum ~out a b);
    cmp_pair "Cmp_eq" (fun ~out a b -> Nx.cmpeq ~out a b) (fun ~out a b -> Nx_ox.cmpeq ~out a b);
    cmp_pair "Cmp_ne" (fun ~out a b -> Nx.cmpne ~out a b) (fun ~out a b -> Nx_ox.cmpne ~out a b);
    cmp_pair "Cmp_lt" (fun ~out a b -> Nx.cmplt ~out a b) (fun ~out a b -> Nx_ox.cmplt ~out a b);
    cmp_pair "Cmp_le" (fun ~out a b -> Nx.cmple ~out a b) (fun ~out a b -> Nx_ox.cmple ~out a b);
    ternary_pair "Where" (fun ~out c t f -> Nx.where ~out c t f) (fun ~out c t f -> Nx_ox.where ~out c t f);
    unary_pair "Neg" (fun ~out a -> Nx.neg ~out a) (fun ~out a -> Nx_ox.neg ~out a);
    unary_pair "Abs" (fun ~out a -> Nx.abs ~out a) (fun ~out a -> Nx_ox.abs ~out a);
    unary_pair "Recip" (fun ~out a -> Nx.recip ~out a) (fun ~out a -> Nx_ox.recip ~out a);
    unary_pair "Sqrt" (fun ~out a -> Nx.sqrt ~out a) (fun ~out a -> Nx_ox.sqrt ~out a);
    unary_pair "Exp" (fun ~out a -> Nx.exp ~out a) (fun ~out a -> Nx_ox.exp ~out a);
    unary_pair "Log" (fun ~out a -> Nx.log ~out a) (fun ~out a -> Nx_ox.log ~out a);
    unary_pair "Sign" (fun ~out a -> Nx.sign ~out a) (fun ~out a -> Nx_ox.sign ~out a);
    unary_pair "Sin" (fun ~out a -> Nx.sin ~out a) (fun ~out a -> Nx_ox.sin ~out a);
    unary_pair "Cos" (fun ~out a -> Nx.cos ~out a) (fun ~out a -> Nx_ox.cos ~out a);
    unary_pair "Tan" (fun ~out a -> Nx.tan ~out a) (fun ~out a -> Nx_ox.tan ~out a);
    unary_pair "Asin" (fun ~out a -> Nx.asin ~out a) (fun ~out a -> Nx_ox.asin ~out a);
    unary_pair "Acos" (fun ~out a -> Nx.acos ~out a) (fun ~out a -> Nx_ox.acos ~out a);
    unary_pair "Atan" (fun ~out a -> Nx.atan ~out a) (fun ~out a -> Nx_ox.atan ~out a);
    unary_pair "Sinh" (fun ~out a -> Nx.sinh ~out a) (fun ~out a -> Nx_ox.sinh ~out a);
    unary_pair "Cosh" (fun ~out a -> Nx.cosh ~out a) (fun ~out a -> Nx_ox.cosh ~out a);
    unary_pair "Tanh" (fun ~out a -> Nx.tanh ~out a) (fun ~out a -> Nx_ox.tanh ~out a);
    unary_pair "Trunc" (fun ~out a -> Nx.trunc ~out a) (fun ~out a -> Nx_ox.trunc ~out a);
    unary_pair "Ceil" (fun ~out a -> Nx.ceil ~out a) (fun ~out a -> Nx_ox.ceil ~out a);
    unary_pair "Floor" (fun ~out a -> Nx.floor ~out a) (fun ~out a -> Nx_ox.floor ~out a);
    unary_pair "Round" (fun ~out a -> Nx.round ~out a) (fun ~out a -> Nx_ox.round ~out a);
    unary_pair "Erf" (fun ~out a -> Nx.erf ~out a) (fun ~out a -> Nx_ox.erf ~out a);
    reduce_pair "Reduce_sum" (fun ~out a -> Nx.sum ~out a) (fun ~out a -> Nx_ox.sum ~out a);
    reduce_pair "Reduce_prod" (fun ~out a -> Nx.prod ~out a) (fun ~out a -> Nx_ox.prod ~out a);
    reduce_pair "Reduce_max" (fun ~out a -> Nx.max ~out a) (fun ~out a -> Nx_ox.max ~out a);
    reduce_pair "Reduce_min" (fun ~out a -> Nx.min ~out a) (fun ~out a -> Nx_ox.min ~out a);
    no_out_unary_pair "Cum_sum" (fun a -> Nx.cumsum a) (fun a -> Nx_ox.cumsum a);
    no_out_unary_pair "Cum_prod" (fun a -> Nx.cumprod a) (fun a -> Nx_ox.cumprod a);
    no_out_unary_pair "Cum_max" (fun a -> Nx.cummax a) (fun a -> Nx_ox.cummax a);
    no_out_unary_pair "Cum_min" (fun a -> Nx.cummin a) (fun a -> Nx_ox.cummin a);
    no_out_unary_pair "Argmax" (fun a -> Nx.argmax a) (fun a -> Nx_ox.argmax a);
    no_out_unary_pair "Argmin" (fun a -> Nx.argmin a) (fun a -> Nx_ox.argmin a);
    no_out_sort_pair "Sort" (fun a -> Nx.sort a) (fun a -> Nx_ox.sort a);
    no_out_unary_pair "Argsort" (fun a -> Nx.argsort a) (fun a -> Nx_ox.argsort a);
  ]
  |> List.concat

let ops_f64 ~size =
  let shape = [| size; size |] in
  let a = Nx.rand Nx.Float64 shape in
  let b = Nx.rand Nx.Float64 shape in
  let cond_c = Nx.less a b in
  let out_c = Nx.empty Nx.float64 shape in
  let out_c_scalar = Nx.empty Nx.float64 [||] in
  let out_c_bool = Nx.empty Nx.bool shape in
  let ctx = Nx_backend.create_context () in
  let a_fe = Nx_ox.empty ctx Nx_ox.float64 shape in
  let b_fe = Nx_ox.empty ctx Nx_ox.float64 shape in
  let cond_fe = Nx_ox.less a_fe b_fe in
  let out_fe = Nx_ox.empty ctx Nx_ox.float64 shape in
  let out_fe_scalar = Nx_ox.empty ctx Nx_ox.float64 [||] in
  let out_fe_bool = Nx_ox.empty ctx Nx_ox.bool shape in
  let bin_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c a b));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe a_fe b_fe));
    ]
  in
  let cmp_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c_bool a b));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe_bool a_fe b_fe));
    ]
  in
  let unary_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c a));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe a_fe));
    ]
  in
  let ternary_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c cond_c a b));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe cond_fe a_fe b_fe));
    ]
  in
  let reduce_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op ~out:out_c_scalar a));
      (name, "Nx (OxCaml)", fun () ->
          ignore (ox_op ~out:out_fe_scalar a_fe));
    ]
  in
  let no_out_unary_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op a));
      (name, "Nx (OxCaml)", fun () -> ignore (ox_op a_fe));
    ]
  in
  let no_out_sort_pair name nx_op ox_op =
    [
      (name, "Nx (C)", fun () -> ignore (nx_op a));
      (name, "Nx (OxCaml)", fun () -> ignore (ox_op a_fe));
    ]
  in
  [
    bin_pair "Add" (fun ~out a b -> Nx.add ~out a b) (fun ~out a b -> Nx_ox.add ~out a b);
    bin_pair "Matmul" (fun ~out a b -> Nx.matmul ~out a b) (fun ~out a b -> Nx_ox.matmul ~out a b);
    bin_pair "Sub" (fun ~out a b -> Nx.sub ~out a b) (fun ~out a b -> Nx_ox.sub ~out a b);
    bin_pair "Mul" (fun ~out a b -> Nx.mul ~out a b) (fun ~out a b -> Nx_ox.mul ~out a b);
    bin_pair "Div" (fun ~out a b -> Nx.div ~out a b) (fun ~out a b -> Nx_ox.div ~out a b);
    bin_pair "Mod" (fun ~out a b -> Nx.mod_ ~out a b) (fun ~out a b -> Nx_ox.mod_ ~out a b);
    bin_pair "Pow" (fun ~out a b -> Nx.pow ~out a b) (fun ~out a b -> Nx_ox.pow ~out a b);
    bin_pair "Atan2" (fun ~out a b -> Nx.atan2 ~out a b) (fun ~out a b -> Nx_ox.atan2 ~out a b);
    bin_pair "Max" (fun ~out a b -> Nx.maximum ~out a b) (fun ~out a b -> Nx_ox.maximum ~out a b);
    bin_pair "Min" (fun ~out a b -> Nx.minimum ~out a b) (fun ~out a b -> Nx_ox.minimum ~out a b);
    cmp_pair "Cmp_eq" (fun ~out a b -> Nx.cmpeq ~out a b) (fun ~out a b -> Nx_ox.cmpeq ~out a b);
    cmp_pair "Cmp_ne" (fun ~out a b -> Nx.cmpne ~out a b) (fun ~out a b -> Nx_ox.cmpne ~out a b);
    cmp_pair "Cmp_lt" (fun ~out a b -> Nx.cmplt ~out a b) (fun ~out a b -> Nx_ox.cmplt ~out a b);
    cmp_pair "Cmp_le" (fun ~out a b -> Nx.cmple ~out a b) (fun ~out a b -> Nx_ox.cmple ~out a b);
    ternary_pair "Where" (fun ~out c t f -> Nx.where ~out c t f) (fun ~out c t f -> Nx_ox.where ~out c t f);
    unary_pair "Neg" (fun ~out a -> Nx.neg ~out a) (fun ~out a -> Nx_ox.neg ~out a);
    unary_pair "Abs" (fun ~out a -> Nx.abs ~out a) (fun ~out a -> Nx_ox.abs ~out a);
    unary_pair "Recip" (fun ~out a -> Nx.recip ~out a) (fun ~out a -> Nx_ox.recip ~out a);
    unary_pair "Sqrt" (fun ~out a -> Nx.sqrt ~out a) (fun ~out a -> Nx_ox.sqrt ~out a);
    unary_pair "Exp" (fun ~out a -> Nx.exp ~out a) (fun ~out a -> Nx_ox.exp ~out a);
    unary_pair "Log" (fun ~out a -> Nx.log ~out a) (fun ~out a -> Nx_ox.log ~out a);
    unary_pair "Sign" (fun ~out a -> Nx.sign ~out a) (fun ~out a -> Nx_ox.sign ~out a);
    unary_pair "Sin" (fun ~out a -> Nx.sin ~out a) (fun ~out a -> Nx_ox.sin ~out a);
    unary_pair "Cos" (fun ~out a -> Nx.cos ~out a) (fun ~out a -> Nx_ox.cos ~out a);
    unary_pair "Tan" (fun ~out a -> Nx.tan ~out a) (fun ~out a -> Nx_ox.tan ~out a);
    unary_pair "Asin" (fun ~out a -> Nx.asin ~out a) (fun ~out a -> Nx_ox.asin ~out a);
    unary_pair "Acos" (fun ~out a -> Nx.acos ~out a) (fun ~out a -> Nx_ox.acos ~out a);
    unary_pair "Atan" (fun ~out a -> Nx.atan ~out a) (fun ~out a -> Nx_ox.atan ~out a);
    unary_pair "Sinh" (fun ~out a -> Nx.sinh ~out a) (fun ~out a -> Nx_ox.sinh ~out a);
    unary_pair "Cosh" (fun ~out a -> Nx.cosh ~out a) (fun ~out a -> Nx_ox.cosh ~out a);
    unary_pair "Tanh" (fun ~out a -> Nx.tanh ~out a) (fun ~out a -> Nx_ox.tanh ~out a);
    unary_pair "Trunc" (fun ~out a -> Nx.trunc ~out a) (fun ~out a -> Nx_ox.trunc ~out a);
    unary_pair "Ceil" (fun ~out a -> Nx.ceil ~out a) (fun ~out a -> Nx_ox.ceil ~out a);
    unary_pair "Floor" (fun ~out a -> Nx.floor ~out a) (fun ~out a -> Nx_ox.floor ~out a);
    unary_pair "Round" (fun ~out a -> Nx.round ~out a) (fun ~out a -> Nx_ox.round ~out a);
    unary_pair "Erf" (fun ~out a -> Nx.erf ~out a) (fun ~out a -> Nx_ox.erf ~out a);
    reduce_pair "Reduce_sum" (fun ~out a -> Nx.sum ~out a) (fun ~out a -> Nx_ox.sum ~out a);
    reduce_pair "Reduce_prod" (fun ~out a -> Nx.prod ~out a) (fun ~out a -> Nx_ox.prod ~out a);
    reduce_pair "Reduce_max" (fun ~out a -> Nx.max ~out a) (fun ~out a -> Nx_ox.max ~out a);
    reduce_pair "Reduce_min" (fun ~out a -> Nx.min ~out a) (fun ~out a -> Nx_ox.min ~out a);
    no_out_unary_pair "Cum_sum" (fun a -> Nx.cumsum a) (fun a -> Nx_ox.cumsum a);
    no_out_unary_pair "Cum_prod" (fun a -> Nx.cumprod a) (fun a -> Nx_ox.cumprod a);
    no_out_unary_pair "Cum_max" (fun a -> Nx.cummax a) (fun a -> Nx_ox.cummax a);
    no_out_unary_pair "Cum_min" (fun a -> Nx.cummin a) (fun a -> Nx_ox.cummin a);
    no_out_unary_pair "Argmax" (fun a -> Nx.argmax a) (fun a -> Nx_ox.argmax a);
    no_out_unary_pair "Argmin" (fun a -> Nx.argmin a) (fun a -> Nx_ox.argmin a);
    no_out_sort_pair "Sort" (fun a -> Nx.sort a) (fun a -> Nx_ox.sort a);
    no_out_unary_pair "Argsort" (fun a -> Nx.argsort a) (fun a -> Nx_ox.argsort a);
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
