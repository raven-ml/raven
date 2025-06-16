(* bench_nx.ml - Simple Nx benchmarking *)

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
module Make_benchmarks (Backend : Nx_core.Backend_intf.S) = struct
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

            (* Square *)
            let square_bench =
              let a = Nx.rand ctx dtype shape in
              create (make_name "Square" size dtype backend_name) (fun () ->
                  ignore (Nx.square a))
            in

            (* Sum *)
            let sum_bench =
              let a = Nx.rand ctx dtype shape in
              create (make_name "Sum" size dtype backend_name) (fun () ->
                  ignore (Nx.sum a))
            in

            benchmarks :=
              [ add_bench; mul_bench; square_bench; sum_bench ] @ !benchmarks)
          dtypes)
      sizes;

    (* Convolution benchmarks *)
    let conv_sizes = List.filter (fun s -> s <= 200) sizes in
    List.iter
      (fun size ->
        List.iter
          (fun (Nx_core.Dtype.Pack dtype) ->
            (* Conv2D benchmark *)
            let conv2d_bench =
              let input_shape = [| 1; 3; size; size |] in  (* NCHW format *)
              let kernel_shape = [| 16; 3; 3; 3 |] in      (* Out_channels x In_channels x H x W *)
              let input = Nx.rand ctx dtype input_shape in
              let kernel = Nx.rand ctx dtype kernel_shape in
              create (make_name "Conv2D 3x3" size dtype backend_name) (fun () ->
                  ignore (Nx.convolve2d input kernel))
            in
            
            (* Conv2D with larger kernel *)
            let conv2d_5x5_bench =
              let input_shape = [| 1; 3; size; size |] in
              let kernel_shape = [| 16; 3; 5; 5 |] in
              let input = Nx.rand ctx dtype input_shape in
              let kernel = Nx.rand ctx dtype kernel_shape in
              create (make_name "Conv2D 5x5" size dtype backend_name) (fun () ->
                  ignore (Nx.convolve2d input kernel))
            in
            
            benchmarks := [ conv2d_bench; conv2d_5x5_bench ] @ !benchmarks)
          dtypes)
      conv_sizes;

    (* Matrix multiplication (smaller sizes only) *)
    let matmul_sizes = List.filter (fun s -> s <= 200) sizes in
    List.iter
      (fun size ->
        List.iter
          (fun (Nx_core.Dtype.Pack dtype) ->
            let shape = [| size; size |] in
            let matmul_bench =
              let a = Nx.rand ctx dtype shape in
              let b = Nx.rand ctx dtype shape in
              create (make_name "MatMul" size dtype backend_name) (fun () ->
                  ignore (Nx.matmul a b))
            in
            benchmarks := matmul_bench :: !benchmarks)
          dtypes)
      matmul_sizes;

    !benchmarks
end

let string_contains s1 s2 =
  let re = Str.regexp_string s2 in
  try
    ignore (Str.search_forward re s1 0);
    true
  with Not_found -> false

(* Run benchmarks for all backends *)
let run_all_backends () =
  Printf.printf "Nx Benchmarking Suite\n";
  Printf.printf "=====================\n\n";

  (* Native backend *)
  Printf.printf "Running Native backend...\n";
  let module Native_bench = Make_benchmarks (Nx_native) in
  let native_ctx = Nx_native.create_context () in
  let native_benchmarks = Native_bench.create_benchmarks native_ctx "Native" in
  let native_results = run ~config native_benchmarks in

  (* CBLAS backend *)
  Printf.printf "\nRunning CBLAS backend...\n";
  let module CBLAS_bench = Make_benchmarks (Nx_cblas) in
  let cblas_ctx = Nx_cblas.create_context () in
  let cblas_benchmarks = CBLAS_bench.create_benchmarks cblas_ctx "CBLAS" in
  let cblas_results = run ~config cblas_benchmarks in

  (* Metal backend *)
  Printf.printf "\nRunning Metal backend...\n";
  let module Metal_bench = Make_benchmarks (Nx_metal) in
  let metal_ctx = Nx_metal.create_context () in
  let metal_benchmarks = Metal_bench.create_benchmarks metal_ctx "Metal" in
  let metal_results = run ~config metal_benchmarks in

  let all_results = native_results @ cblas_results @ metal_results in

  (* Simple analysis *)
  Printf.printf "\n=== Performance Summary ===\n";

  (* Find fastest backend for each operation *)
  let operations = [ "Add"; "MatMul"; "Sum"; "Conv2D" ] in
  List.iter
    (fun op ->
      Printf.printf "\n%s Performance:\n" op;
      let op_results =
        List.filter (fun r -> string_contains r.name op) all_results
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

(* Just run everything *)
let () = run_all_backends ()
