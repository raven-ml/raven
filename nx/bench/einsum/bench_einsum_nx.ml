(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Configuration *)
let sizes = [ 50; 100; 200 ]
let backend_name = "Nx"

let benchmark_name op_name size dtype_label =
  Printf.sprintf "%s %dx%d %s (%s)" op_name size size dtype_label backend_name

type einsum_spec = { name : string; subscripts : string }

let einsum_specs =
  [
    { name = "MatMul"; subscripts = "ij,jk->ik" };
    { name = "BatchMatMul"; subscripts = "bij,bjk->bik" };
    { name = "InnerProduct"; subscripts = "i,i->" };
    (* Critical contraction-reduction patterns (known to be slow) *)
    { name = "ContractReduce1"; subscripts = "ij,kj->" };
    { name = "ContractReduce2"; subscripts = "ij,jk->" };
  ]

let make_key spec size offset =
  Nx.Rng.key (Hashtbl.hash (spec.name, size, offset))

let setup_f32 spec size =
  match spec.name with
  | "MatMul" | "ContractReduce2" ->
      let shape = [| size; size |] in
      [
        Nx.rand Nx.Float32 ~key:(make_key spec size 0) shape;
        Nx.rand Nx.Float32 ~key:(make_key spec size 1) shape;
      ]
  | "BatchMatMul" ->
      let shape = [| 4; size; size |] in
      [
        Nx.rand Nx.Float32 ~key:(make_key spec size 2) shape;
        Nx.rand Nx.Float32 ~key:(make_key spec size 3) shape;
      ]
  | "InnerProduct" ->
      let shape = [| size |] in
      [
        Nx.rand Nx.Float32 ~key:(make_key spec size 4) shape;
        Nx.rand Nx.Float32 ~key:(make_key spec size 5) shape;
      ]
  | "ContractReduce1" ->
      (* ij,kj-> needs two (size, size) matrices *)
      let shape = [| size; size |] in
      [
        Nx.rand Nx.Float32 ~key:(make_key spec size 6) shape;
        Nx.rand Nx.Float32 ~key:(make_key spec size 7) shape;
      ]
  | _ -> failwith ("Unknown einsum operation: " ^ spec.name)

let setup_f64 spec size =
  match spec.name with
  | "MatMul" | "ContractReduce2" ->
      let shape = [| size; size |] in
      [
        Nx.rand Nx.Float64 ~key:(make_key spec size 8) shape;
        Nx.rand Nx.Float64 ~key:(make_key spec size 9) shape;
      ]
  | "BatchMatMul" ->
      let shape = [| 4; size; size |] in
      [
        Nx.rand Nx.Float64 ~key:(make_key spec size 10) shape;
        Nx.rand Nx.Float64 ~key:(make_key spec size 11) shape;
      ]
  | "InnerProduct" ->
      let shape = [| size |] in
      [
        Nx.rand Nx.Float64 ~key:(make_key spec size 12) shape;
        Nx.rand Nx.Float64 ~key:(make_key spec size 13) shape;
      ]
  | "ContractReduce1" ->
      (* ij,kj-> needs two (size, size) matrices *)
      let shape = [| size; size |] in
      [
        Nx.rand Nx.Float64 ~key:(make_key spec size 14) shape;
        Nx.rand Nx.Float64 ~key:(make_key spec size 15) shape;
      ]
  | _ -> failwith ("Unknown einsum operation: " ^ spec.name)

let build_benchmarks () =
  let benchmarks = ref [] in

  (* Float32 benchmarks *)
  List.iter
    (fun size ->
      List.iter
        (fun spec ->
          let operands = setup_f32 spec size |> Array.of_list in
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
          let operands = setup_f64 spec size |> Array.of_list in
          let bench_name = benchmark_name spec.name size "f64" in
          let fn () = ignore (Nx.einsum spec.subscripts operands) in
          benchmarks := Ubench.bench bench_name fn :: !benchmarks)
        einsum_specs)
    sizes;

  List.rev !benchmarks

let default_config () =
  let open Ubench.Config in
  default |> time_limit 1.0 |> warmup 1 |> min_measurements 5
  |> geometric_scale 1.3 |> gc_stabilization false |> build

let () =
  let benchmarks = build_benchmarks () in
  let config = default_config () in
  ignore (Ubench.run ~config benchmarks)
