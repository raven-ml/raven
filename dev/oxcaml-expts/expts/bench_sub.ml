module Dtype = Nx_core.Dtype
module Par_ox = Oxcaml_expts.Nx_oxcaml
(** Benchmark suite for Nx tensor operations *)

let sizes = [ 200; 500; 1000]

(** Helper to create benchmark name *)
let benchmark_name op_name size dtype_label backend_name =
  Printf.sprintf "%s %dx%d %s (%s)" op_name size size dtype_label backend_name

(** Generate benchmark operations for Float32 *)
let nx_operations_f32 ~size =
  let shape = [| size; size |] in
  let a = Nx.rand Nx.Float32 ~key:(Nx.Rng.key (size * 3)) shape in
  let b = Nx.rand Nx.Float32 ~key:(Nx.Rng.key ((size * 3) + 1)) shape in

  let ctx_ox = Par_ox.create_context () in

  let a_p_ox = Par_ox.op_buffer ctx_ox Dtype.Float32 size in
  let b_p_ox = Par_ox.op_buffer ctx_ox Dtype.Float32 size in
  let out_ox = Par_ox.op_buffer ctx_ox Dtype.Float32 size in

  let ops =
    [
      ("Sub", "Nx", fun () -> ignore (Nx.sub a b));
      ( "sub",
        "Nx parallel",
        fun () -> ignore (Par_ox.op_sub ~out:out_ox a_p_ox b_p_ox) );
    ]
  in

  ops

(** Generate benchmark operations for Float64 *)
let nx_operations_f64 ~size =
  let shape = [| size; size |] in
  let a = Nx.rand Nx.Float64 ~key:(Nx.Rng.key (size * 3)) shape in
  let b = Nx.rand Nx.Float64 ~key:(Nx.Rng.key ((size * 3) + 1)) shape in

  let ctx_ox = Par_ox.create_context () in

  let a_p_ox = Par_ox.op_buffer ctx_ox Dtype.Float64 size in
  let b_p_ox = Par_ox.op_buffer ctx_ox Dtype.Float64 size in
  let out_ox = Par_ox.op_buffer ctx_ox Dtype.Float64 size in

  let ops =
    [
      ("Sub", "Nx", fun () -> ignore (Nx.sub a b));
      ( "Sub",
        "Nx parallel",
        fun () -> ignore (Par_ox.op_sub ~out:out_ox a_p_ox b_p_ox) );
    ]
  in

  ops

(** Build all benchmarks *)
let build_benchmarks () =
  let benchmarks = ref [] in
  List.iter
    (fun size ->
      (* Float32 benchmarks *)
      let ops_f32 = nx_operations_f32 ~size in
      List.iter
        (fun (op_name, backend_name, fn) ->
          let bench_name = benchmark_name op_name size "f32" backend_name in
          benchmarks := Ubench.bench bench_name fn :: !benchmarks)
        ops_f32;

      (* Float64 benchmarks *)
      let ops_f64 = nx_operations_f64 ~size in
      List.iter
        (fun (op_name, backend_name, fn) ->
          let bench_name = benchmark_name op_name size "f64" backend_name in
          benchmarks := Ubench.bench bench_name fn :: !benchmarks)
        ops_f64)
    sizes;
  List.rev !benchmarks

(** Default configuration matching NumPy benchmark *)
let default_config () =
  let open Ubench.Config in
  default |> time_limit 1.0 |> warmup 1 |> min_measurements 5
  |> geometric_scale 1.3 |> gc_stabilization false |> build

(** Main entry point *)
let () =
  let benchmarks = build_benchmarks () in
  let config = default_config () in
  (* Mirror the Python defaults for fair comparisons with NumPy benchmarks. *)
  ignore (Ubench.run ~config benchmarks)
