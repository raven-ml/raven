(** Benchmark suite for Nx einsum operations *)

(** Configuration *)
let sizes = [ 50; 100; 200 ]

let backend_name = "Nx"

(** Helper to create benchmark name *)
let benchmark_name op_name size dtype_label =
  Printf.sprintf "%s %dx%d %s (%s)" op_name size size dtype_label backend_name

type einsum_spec = { name : string; subscripts : string }
(** Einsum operation specification *)

(** Common einsum operations to benchmark - covering key use cases *)
let einsum_specs =
  [
    { name = "MatMul"; subscripts = "ij,jk->ik" };
    { name = "BatchMatMul"; subscripts = "bij,bjk->bik" };
    { name = "InnerProduct"; subscripts = "i,i->" };
  ]

(** Setup tensors for a given operation and dtype *)
let setup_f32 spec size =
  match spec.name with
  | "MatMul" ->
      let shape = [| size; size |] in
      [| Nx.rand Nx.Float32 shape; Nx.rand Nx.Float32 shape |]
  | "BatchMatMul" ->
      let shape = [| 4; size; size |] in
      [| Nx.rand Nx.Float32 shape; Nx.rand Nx.Float32 shape |]
  | "InnerProduct" ->
      let shape = [| size |] in
      [| Nx.rand Nx.Float32 shape; Nx.rand Nx.Float32 shape |]
  | _ -> failwith ("Unknown einsum operation: " ^ spec.name)

let setup_f64 spec size =
  match spec.name with
  | "MatMul" ->
      let shape = [| size; size |] in
      [| Nx.rand Nx.Float64 shape; Nx.rand Nx.Float64 shape |]
  | "BatchMatMul" ->
      let shape = [| 4; size; size |] in
      [| Nx.rand Nx.Float64 shape; Nx.rand Nx.Float64 shape |]
  | "InnerProduct" ->
      let shape = [| size |] in
      [| Nx.rand Nx.Float64 shape; Nx.rand Nx.Float64 shape |]
  | _ -> failwith ("Unknown einsum operation: " ^ spec.name)

(** Build all benchmarks *)
let build_benchmarks () =
  let benchmarks = ref [] in

  (* Float32 benchmarks *)
  List.iter
    (fun size ->
      List.iter
        (fun spec ->
          let operands = setup_f32 spec size in
          let bench_name = benchmark_name spec.name size "f32" in
          let fn () = ignore (Nx.einsum spec.subscripts operands) in
          benchmarks := Ubench.bench bench_name fn :: !benchmarks)
        einsum_specs)
    sizes;

  (* Float64 benchmarks *)
  List.iter
    (fun size ->
      List.iter
        (fun spec ->
          let operands = setup_f64 spec size in
          let bench_name = benchmark_name spec.name size "f64" in
          let fn () = ignore (Nx.einsum spec.subscripts operands) in
          benchmarks := Ubench.bench bench_name fn :: !benchmarks)
        einsum_specs)
    sizes;

  List.rev !benchmarks

(** Default configuration *)
let default_config () =
  let open Ubench.Config in
  default |> time_limit 1.0 |> warmup 1 |> min_measurements 5
  |> geometric_scale 1.3 |> gc_stabilization false |> build

(** Main entry point *)
let () =
  let benchmarks = build_benchmarks () in
  let config = default_config () in
  ignore (Ubench.run ~config benchmarks)
