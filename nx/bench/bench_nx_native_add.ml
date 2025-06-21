(* bench_nx_native_add.ml - Benchmark for add operation on all types *)

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

(* All supported data types *)
let dtypes = [
  Nx_core.Dtype.Pack Float16; 
  Nx_core.Dtype.Pack Float32;
  Nx_core.Dtype.Pack Float64;
  Nx_core.Dtype.Pack Int8;
  Nx_core.Dtype.Pack UInt8;
  Nx_core.Dtype.Pack Int16;
  Nx_core.Dtype.Pack UInt16;
  Nx_core.Dtype.Pack Int32;
  Nx_core.Dtype.Pack Int64;
  Nx_core.Dtype.Pack Complex32;  
  Nx_core.Dtype.Pack Complex64; 
]

(* Helper to create benchmark name *)
let make_name size dtype =
  Printf.sprintf "Add %dx%d %s" size size (Nx_core.Dtype.to_string dtype)

(* Create benchmarks for add operation *)
let create_add_benchmarks () =
  let module Nx = Nx_core.Make_frontend (Nx_native) in
  let ctx = Nx_native.create_context () in
  let benchmarks = ref [] in

  List.iter
    (fun size ->
      List.iter
        (fun (Nx_core.Dtype.Pack dtype) ->
          let shape = [| size; size |] in
          let add_bench =
            let a, b = match dtype with
              | Float16 | Float32 | Float64  ->
                  let a = Nx.rand ctx dtype shape in
                  let b = Nx.rand ctx dtype shape in
                  (a, b)
              | _ ->
                  let a = Nx.ones ctx dtype shape in
                  let b = Nx.ones ctx dtype shape in
                  (a, b)
            in
            create (make_name size dtype) (fun () ->
                ignore (Nx.add a b))
          in
          benchmarks := add_bench :: !benchmarks)
        dtypes)
    sizes;

  !benchmarks

(* Analyze results by data type *)
let analyze_by_dtype results =
  Printf.printf "\n=== Add Performance by Data Type ===\n";
  
  List.iter
    (fun (Nx_core.Dtype.Pack dtype) ->
      Printf.printf "\n%s:\n" (Nx_core.Dtype.to_string dtype);
      let dtype_results =
        List.filter (fun r -> 
          String.contains r.name (Nx_core.Dtype.to_string dtype).[0]) results
      in
      let sorted =
        List.sort
          (fun a b -> Float.compare a.time_stats.avg b.time_stats.avg)
          dtype_results
      in
      List.iter
        (fun result ->
          Printf.printf "  %s: %.2f ns/op (±%.2f%%)\n" 
            result.name
            result.time_stats.avg
            (result.time_stats.std_dev /. result.time_stats.avg *. 100.0))
        sorted)
    dtypes

(* Analyze results by size *)
let analyze_by_size results =
  Printf.printf "\n=== Add Performance by Size ===\n";
  
  List.iter
    (fun size ->
      Printf.printf "\n%dx%d matrices:\n" size size;
      let size_str = Printf.sprintf "%dx%d" size size in
      let size_results =
        List.filter (fun r -> String.contains r.name size_str.[0]) results
      in
      let sorted =
        List.sort
          (fun a b -> Float.compare a.time_stats.avg b.time_stats.avg)
          size_results
      in
      List.iter
        (fun result ->
          Printf.printf "  %s: %.2f ns/op\n" 
            result.name
            result.time_stats.avg)
        sorted)
    sizes

(* Find fastest and slowest for each size *)
let analyze_extremes results =
  Printf.printf "\n=== Fastest vs Slowest by Size ===\n";
  
  List.iter
    (fun size ->
      let size_str = Printf.sprintf "%dx%d" size size in
      let size_results =
        List.filter (fun r -> String.contains r.name size_str.[0]) results
      in
      if List.length size_results > 0 then begin
        let sorted =
          List.sort
            (fun a b -> Float.compare a.time_stats.avg b.time_stats.avg)
            size_results
        in
        let fastest = List.hd sorted in
        let slowest = List.hd (List.rev sorted) in
        let speedup = slowest.time_stats.avg /. fastest.time_stats.avg in
        Printf.printf "\n%s:\n" size_str;
        Printf.printf "  Fastest: %s (%.2f ns/op)\n" 
          fastest.name fastest.time_stats.avg;
        Printf.printf "  Slowest: %s (%.2f ns/op)\n" 
          slowest.name slowest.time_stats.avg;
        Printf.printf "  Speedup: %.2fx\n" speedup
      end)
    sizes

(* Main benchmark execution *)
let run_benchmarks () =
  Printf.printf "Creating add operation benchmarks...\n";
  let benchmarks = create_add_benchmarks () in
  Printf.printf "Running %d benchmarks...\n" (List.length benchmarks);
  
  let results = run ~config benchmarks in
  
  (* Multiple analysis views *)
  analyze_by_dtype results;
  analyze_by_size results;
  analyze_extremes results;
  
  (* Overall summary *)
  Printf.printf "\n=== Overall Summary ===\n";
  let all_times = List.map (fun r -> r.time_stats.avg) results in
  let min_time = List.fold_left Float.min Float.max_float all_times in
  let max_time = List.fold_left Float.max 0.0 all_times in
  let avg_time = List.fold_left (+.) 0.0 all_times /. float (List.length all_times) in
  
  Printf.printf "Min time: %.2f ns/op\n" min_time;
  Printf.printf "Max time: %.2f ns/op\n" max_time;
  Printf.printf "Avg time: %.2f ns/op\n" avg_time;
  Printf.printf "Range: %.2fx\n" (max_time /. min_time)

(* Run the benchmarks *)
let () = run_benchmarks ()
