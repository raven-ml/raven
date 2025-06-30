(* bench_nx_unit.ml - Nx benchmarking framework *)

open Ubench

(* Simple benchmark configuration *)
let config =
  {
    quota = Time_limit 2.0;
    warmup_iterations = 3;
    min_measurements = 10;
    stabilize_gc = true;
    geometric_scale = 1.3;
    fork_benchmarks = false;
    regressions = [ (Time_per_run, [ Runs ]) ];
  }

(* Test parameters *)
let sizes = [ 50; 100; 200; 500 ]
let dtypes = [ Nx_core.Dtype.Pack Float32 ]

(* Helper to create benchmark name *)
let make_name op_name size dtype backend_name =
  Printf.sprintf "%s %dx%d %s (%s)" op_name size size
    (Nx_core.Dtype.to_string dtype)
    backend_name

(* Generate benchmarks for a specific backend *)
module Make (Backend : Nx_core.Backend_intf.S) = struct
  module Nx = Nx_core.Make_frontend (Backend)

  let create_benchmarks ctx backend_name =
    let benchmarks = ref [] in

    (* Element-wise operations *)
    List.iter
      (fun size ->
        List.iter
          (fun (Nx_core.Dtype.Pack dtype) ->
            let shape = [| size; size |] in

            (* Addition *)
            let add_bench =
              let a = Nx.rand ctx dtype shape in
              let b = Nx.rand ctx dtype shape in
              create (make_name "Add" size dtype backend_name) (fun () ->
                  ignore (Nx.add a b))
            in

            (* Multiplication *)
            let mul_bench =
              let a = Nx.rand ctx dtype shape in
              let b = Nx.rand ctx dtype shape in
              create (make_name "Mul" size dtype backend_name) (fun () ->
                  ignore (Nx.mul a b))
            in

            (* MatMul *)
            let matmul_bench =
              let a = Nx.rand ctx dtype shape in
              let b = Nx.rand ctx dtype shape in
              create (make_name "MatMul" size dtype backend_name) (fun () ->
                  ignore (Nx.matmul a b))
            in

            (* Sum *)
            let sum_bench =
              let a = Nx.rand ctx dtype shape in
              create (make_name "Sum" size dtype backend_name) (fun () ->
                  ignore (Nx.sum a))
            in

            (* Conv2D for size >= 100 *)
            let conv_bench =
              if size >= 100 then
                let input = Nx.rand ctx dtype [| 1; 3; size; size |] in
                let kernel = Nx.rand ctx dtype [| 8; 3; 3; 3 |] in
                Some
                  (create (make_name "Conv2D" size dtype backend_name)
                     (fun () ->
                       ignore (Nx.convolve2d input kernel ~padding_mode:`Same)))
              else None
            in

            benchmarks :=
              add_bench :: mul_bench :: matmul_bench :: sum_bench
              :: List.filter_map (fun x -> x) [ conv_bench ]
              @ !benchmarks)
          dtypes)
      sizes;

    List.rev !benchmarks

  let run ~backend_name ctx =
    Printf.printf "Running %s backend benchmarks...\n" backend_name;
    let benchmarks = create_benchmarks ctx backend_name in
    let results = run ~config benchmarks in
    (backend_name, results)
end

(* Helper to summarize results *)
let summarize_results all_results =
  Printf.printf "\n=== Performance Summary ===\n";

  (* Find fastest backend for each operation *)
  let operations = [ "Add"; "MatMul"; "Sum"; "Conv2D" ] in

  List.iter
    (fun op ->
      Printf.printf "\n%s Performance:\n" op;
      let op_results =
        List.concat_map snd all_results
        |> List.filter (fun r ->
               try
                 ignore (Str.search_forward (Str.regexp op) r.name 0);
                 true
               with Not_found -> false)
      in
      let sorted =
        List.sort
          (fun a b -> Float.compare a.time_stats.avg b.time_stats.avg)
          op_results
      in
      List.iteri
        (fun i result ->
          if i < 3 then (* Show top 3 *)
            Printf.printf "  %d. %s: %.2f ns/op\n" (i + 1) result.name
              result.time_stats.avg)
        sorted)
    operations
