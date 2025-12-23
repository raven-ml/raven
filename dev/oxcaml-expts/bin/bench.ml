(** Benchmark suite for Nx tensor operations *)

(** Configuration *)
let sizes = [ 50; 100; 200; 500 ]

let backend_name = "Nx"

(** Helper to create benchmark name *)
let benchmark_name op_name size dtype_label =
  Printf.sprintf "%s %dx%d %s (%s)" op_name size size dtype_label backend_name

(** Generate benchmark operations for Float32 *)
let nx_operations_f32 ~size =
  let shape = [| size; size |] in
  let a = Nx.rand Nx.Float32 ~key:(Nx.Rng.key (size * 2)) shape in
  let b = Nx.rand Nx.Float32 ~key:(Nx.Rng.key ((size * 2) + 1)) shape in

  let ops =
    [
      ("Add", fun () -> ignore (Nx.add a b));
      (* ("Mul", fun () -> ignore (Nx.mul a b)); *)
    ]
  in

  let ops =
    ops
    @ [
        ("Sum", fun () -> ignore (Nx.sum a));
        (* ("Transpose", fun () -> ignore (Nx.transpose a)); *)
      ]
  in

  ops

(** Generate benchmark operations for Float64 *)
let nx_operations_f64 ~size =
  let shape = [| size; size |] in
  let a = Nx.rand Nx.Float64 ~key:(Nx.Rng.key (size * 3)) shape in
  let b = Nx.rand Nx.Float64 ~key:(Nx.Rng.key ((size * 3) + 1)) shape in

  let ops =
    [
      ("Add", fun () -> ignore (Nx.add a b));
      (* ("Mul", fun () -> ignore (Nx.mul a b)); *)
    ]
  in

  (* let ops =
    ops
    @ [
        ("Sum", fun () -> ignore (Nx.sum a));
        ("Transpose", fun () -> ignore (Nx.transpose a));
      ]
  in *)

  ops

(** Build all benchmarks *)
let build_benchmarks () =
  let benchmarks = ref [] in
  List.iter
    (fun size ->
      (* Float32 benchmarks *)
      let ops_f32 = nx_operations_f32 ~size in
      List.iter
        (fun (op_name, fn) ->
          let bench_name = benchmark_name op_name size "f32" in
          benchmarks := Ubench.bench bench_name fn :: !benchmarks)
        ops_f32;

      (* Float64 benchmarks *)
      let ops_f64 = nx_operations_f64 ~size in
      List.iter
        (fun (op_name, fn) ->
          let bench_name = benchmark_name op_name size "f64" in
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

