(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Configuration *)
let sizes = [ 50; 100; 200; 512 ]
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
    (* Independent contraction: no shared axes, sum everything *)
    { name = "IndependentSum"; subscripts = "ab,cd->" };
  ]

let setup_f32 spec size =
  match spec.name with
  | "MatMul" | "ContractReduce2" ->
      let shape = [| size; size |] in
      [ Nx.rand Nx.Float32 shape; Nx.rand Nx.Float32 shape ]
  | "BatchMatMul" ->
      let shape = [| 4; size; size |] in
      [ Nx.rand Nx.Float32 shape; Nx.rand Nx.Float32 shape ]
  | "InnerProduct" ->
      let shape = [| size |] in
      [ Nx.rand Nx.Float32 shape; Nx.rand Nx.Float32 shape ]
  | "ContractReduce1" ->
      let shape = [| size; size |] in
      [ Nx.rand Nx.Float32 shape; Nx.rand Nx.Float32 shape ]
  | "IndependentSum" ->
      let shape = [| size; size |] in
      [ Nx.rand Nx.Float32 shape; Nx.rand Nx.Float32 shape ]
  | _ -> failwith ("Unknown einsum operation: " ^ spec.name)

let setup_f64 spec size =
  match spec.name with
  | "MatMul" | "ContractReduce2" ->
      let shape = [| size; size |] in
      [ Nx.rand Nx.Float64 shape; Nx.rand Nx.Float64 shape ]
  | "BatchMatMul" ->
      let shape = [| 4; size; size |] in
      [ Nx.rand Nx.Float64 shape; Nx.rand Nx.Float64 shape ]
  | "InnerProduct" ->
      let shape = [| size |] in
      [ Nx.rand Nx.Float64 shape; Nx.rand Nx.Float64 shape ]
  | "ContractReduce1" ->
      let shape = [| size; size |] in
      [ Nx.rand Nx.Float64 shape; Nx.rand Nx.Float64 shape ]
  | "IndependentSum" ->
      let shape = [| size; size |] in
      [ Nx.rand Nx.Float64 shape; Nx.rand Nx.Float64 shape ]
  | _ -> failwith ("Unknown einsum operation: " ^ spec.name)

let build_benchmarks () =
  let f32_benches = ref [] in
  let f64_benches = ref [] in

  List.iter
    (fun size ->
      List.iter
        (fun spec ->
          let operands = setup_f32 spec size |> Array.of_list in
          let bench_name = benchmark_name spec.name size "f32" in
          let fn () = Thumper.consume (Nx.einsum spec.subscripts operands) in
          f32_benches := Thumper.bench bench_name fn :: !f32_benches)
        einsum_specs)
    sizes;

  List.iter
    (fun size ->
      List.iter
        (fun spec ->
          let operands = setup_f64 spec size |> Array.of_list in
          let bench_name = benchmark_name spec.name size "f64" in
          let fn () = Thumper.consume (Nx.einsum spec.subscripts operands) in
          f64_benches := Thumper.bench bench_name fn :: !f64_benches)
        einsum_specs)
    sizes;

  [
    Thumper.group "f32" (List.rev !f32_benches);
    Thumper.group "f64" (List.rev !f64_benches);
  ]

let () =
  let benchmarks = build_benchmarks () in
  Thumper.run "nx_einsum" benchmarks
