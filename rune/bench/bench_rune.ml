(* bench_rune.ml - Comprehensive Rune benchmarking suite *)

open Ubench

(* Benchmark configuration following nx/bench pattern *)
let config =
  Config.default |> Config.time_limit 2.0 |> Config.warmup 3
  |> Config.min_measurements 10
  |> Config.gc_stabilization true
  |> Config.geometric_scale 1.3 |> Config.fork false
  |> Config.regressions [ (Time_per_run, [ Runs ], false) ]
  |> Config.progress_callback (fun _ -> ())  (* Suppress progress output *)
  |> Config.build

(* Test parameters as specified *)
let sizes = [ 50; 100; 200 ]

(* Helper to create benchmark name *)
let make_name op_name size pass_type =
  Printf.sprintf "%s %dx%d f32 (%s)" op_name size size pass_type

(* Helper to create random tensors *)
let create_rand shape =
  let key = Rune.Rng.key 42 in
  Rune.Rng.uniform key Rune.float32 shape

(* Safe JIT wrapper to handle compilation failures *)
let safe_jit_bench name f input =
  try
    let jit_f = Rune.jit f in
    Some (bench name (fun () -> ignore (jit_f input)))
  with
  | _ -> 
    Printf.eprintf "Warning: JIT compilation failed for %s, skipping...\n" name;
    None

(* Element-wise operations benchmarks *)
let bench_elementwise () =
  List.fold_left (fun acc size ->
    let shape = [| size; size |] in
    let a = create_rand shape in
    let b = create_rand shape in
    
    (* Forward pass benchmarks *)
    let add_bench = bench (make_name "Add" size "forward") 
      (fun () -> ignore (Rune.add a b)) in
    let mul_bench = bench (make_name "Mul" size "forward") 
      (fun () -> ignore (Rune.mul a b)) in
    let square_bench = bench (make_name "Square" size "forward") 
      (fun () -> ignore (Rune.square a)) in
    let sqrt_bench = bench (make_name "Sqrt" size "forward") 
      (fun () -> ignore (Rune.sqrt (Rune.abs a))) in
    let exp_bench = bench (make_name "Exp" size "forward") 
      (fun () -> ignore (Rune.exp a)) in
    let log_bench = bench (make_name "Log" size "forward") 
      (fun () -> ignore (Rune.log (Rune.add (Rune.abs a) (Rune.scalar Rune.float32 1e-8)))) in
    let sin_bench = bench (make_name "Sin" size "forward") 
      (fun () -> ignore (Rune.sin a)) in
    let cos_bench = bench (make_name "Cos" size "forward") 
      (fun () -> ignore (Rune.cos a)) in
    
    (* Backward pass benchmarks using Rune.grad *)
    let add_grad_bench = bench (make_name "Add" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.add x b)) a)) in
    let mul_grad_bench = bench (make_name "Mul" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.mul x b)) a)) in
    let square_grad_bench = bench (make_name "Square" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.square x)) a)) in
    let sqrt_grad_bench = bench (make_name "Sqrt" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.sqrt (Rune.abs x))) a)) in
    let exp_grad_bench = bench (make_name "Exp" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.exp x)) a)) in
    let log_grad_bench = bench (make_name "Log" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.log (Rune.add (Rune.abs x) (Rune.scalar Rune.float32 1e-8)))) a)) in
    let sin_grad_bench = bench (make_name "Sin" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.sin x)) a)) in
    let cos_grad_bench = bench (make_name "Cos" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.cos x)) a)) in
      
    (* JIT compilation benchmarks with error handling *)
    let jit_benchmarks = List.filter_map (fun x -> x) [
      safe_jit_bench (make_name "Add" size "jit") (fun x -> Rune.add x b) a;
      safe_jit_bench (make_name "Mul" size "jit") (fun x -> Rune.mul x b) a;
      safe_jit_bench (make_name "Square" size "jit") (fun x -> Rune.square x) a;
      safe_jit_bench (make_name "Exp" size "jit") (fun x -> Rune.exp x) a;
    ] in
      
    add_bench :: mul_bench :: square_bench :: sqrt_bench :: exp_bench :: 
    log_bench :: sin_bench :: cos_bench ::
    add_grad_bench :: mul_grad_bench :: square_grad_bench :: sqrt_grad_bench :: 
    exp_grad_bench :: log_grad_bench :: sin_grad_bench :: cos_grad_bench ::
    jit_benchmarks @ acc
  ) [] sizes

(* Reduction operations benchmarks *)
let bench_reductions () =
  List.fold_left (fun acc size ->
    let shape = [| size; size |] in
    let a = create_rand shape in
    
    (* Forward pass benchmarks *)
    let sum_bench = bench (make_name "Sum" size "forward") 
      (fun () -> ignore (Rune.sum a)) in
    let mean_bench = bench (make_name "Mean" size "forward") 
      (fun () -> ignore (Rune.mean a)) in
    let max_bench = bench (make_name "Max" size "forward") 
      (fun () -> ignore (Rune.max a)) in
      
    (* Backward pass benchmarks *)
    let sum_grad_bench = bench (make_name "Sum" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum x) a)) in
    let mean_grad_bench = bench (make_name "Mean" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.mean x) a)) in
      
    (* JIT compilation benchmarks with error handling *)
    let jit_benchmarks = List.filter_map (fun x -> x) [
      safe_jit_bench (make_name "Sum" size "jit") (fun x -> Rune.sum x) a;
      safe_jit_bench (make_name "Mean" size "jit") (fun x -> Rune.mean x) a;
    ] in
      
    sum_bench :: mean_bench :: max_bench :: 
    sum_grad_bench :: mean_grad_bench ::
    jit_benchmarks @ acc
  ) [] sizes

(* Linear algebra operations benchmarks *)
let bench_linalg () =
  List.fold_left (fun acc size ->
    let shape = [| size; size |] in
    let a = create_rand shape in
    let b = create_rand shape in
    
    (* Batched matmul shapes *)
    let batch_shape_a = [| 4; size; size |] in
    let batch_shape_b = [| 4; size; size |] in
    let batch_a = create_rand batch_shape_a in
    let batch_b = create_rand batch_shape_b in
    
    (* Forward pass benchmarks *)
    let matmul_bench = bench (make_name "Matmul" size "forward") 
      (fun () -> ignore (Rune.matmul a b)) in
    let batched_matmul_bench = bench (make_name "BatchedMatmul" size "forward") 
      (fun () -> ignore (Rune.matmul batch_a batch_b)) in
      
    (* Backward pass benchmarks *)
    let matmul_grad_bench = bench (make_name "Matmul" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.matmul x b)) a)) in
    let batched_matmul_grad_bench = bench (make_name "BatchedMatmul" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.matmul x batch_b)) batch_a)) in
      
    (* JIT compilation benchmarks with error handling *)
    let jit_benchmarks = List.filter_map (fun x -> x) [
      safe_jit_bench (make_name "Matmul" size "jit") (fun x -> Rune.matmul x b) a;
      safe_jit_bench (make_name "BatchedMatmul" size "jit") (fun x -> Rune.matmul x batch_b) batch_a;
    ] in
      
    matmul_bench :: batched_matmul_bench :: 
    matmul_grad_bench :: batched_matmul_grad_bench ::
    jit_benchmarks @ acc
  ) [] sizes

(* Shape operations benchmarks *)
let bench_shape_ops () =
  List.fold_left (fun acc size ->
    let shape = [| size; size |] in
    let a = create_rand shape in
    
    (* Forward pass benchmarks *)
    let reshape_bench = bench (make_name "Reshape" size "forward") 
      (fun () -> ignore (Rune.reshape [| size * size |] a)) in
    let transpose_bench = bench (make_name "Transpose" size "forward") 
      (fun () -> ignore (Rune.transpose a ~axes:[1; 0])) in
    let slice_bench = bench (make_name "Slice" size "forward") 
      (fun () -> ignore (Rune.slice [Rune.R (0, size/2); Rune.R (0, size/2)] a)) in
    let broadcast_bench = bench (make_name "Broadcast" size "forward") 
      (fun () -> 
        let b = create_rand [| 1; size |] in
        ignore (Rune.add a b)) in
        
    (* Backward pass benchmarks *)
    let reshape_grad_bench = bench (make_name "Reshape" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.reshape [| size * size |] x)) a)) in
    let transpose_grad_bench = bench (make_name "Transpose" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.transpose x ~axes:[1; 0])) a)) in
    let broadcast_grad_bench = bench (make_name "Broadcast" size "backward") 
      (fun () -> 
        let b = create_rand [| 1; size |] in
        ignore (Rune.grad (fun x -> Rune.sum (Rune.add x b)) a)) in
        
    (* JIT compilation benchmarks with error handling *)
    let jit_benchmarks = List.filter_map (fun x -> x) [
      safe_jit_bench (make_name "Reshape" size "jit") (fun x -> Rune.reshape [| size * size |] x) a;
      safe_jit_bench (make_name "Transpose" size "jit") (fun x -> Rune.transpose x ~axes:[1; 0]) a;
    ] in
      
    reshape_bench :: transpose_bench :: slice_bench :: broadcast_bench ::
    reshape_grad_bench :: transpose_grad_bench :: broadcast_grad_bench ::
    jit_benchmarks @ acc
  ) [] sizes

(* Neural network operations benchmarks *)
let bench_neural_ops () =
  List.fold_left (fun acc size ->
    let shape = [| size; size |] in
    let a = create_rand shape in
    
    (* Forward pass benchmarks *)
    let relu_bench = bench (make_name "ReLU" size "forward") 
      (fun () -> ignore (Rune.relu a)) in
    let sigmoid_bench = bench (make_name "Sigmoid" size "forward") 
      (fun () -> ignore (Rune.sigmoid a)) in
    let tanh_bench = bench (make_name "Tanh" size "forward") 
      (fun () -> ignore (Rune.tanh a)) in
    let softmax_bench = bench (make_name "Softmax" size "forward") 
      (fun () -> ignore (Rune.softmax a ~axes:[-1])) in
      
    (* Backward pass benchmarks *)
    let relu_grad_bench = bench (make_name "ReLU" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.relu x)) a)) in
    let sigmoid_grad_bench = bench (make_name "Sigmoid" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.sigmoid x)) a)) in
    let tanh_grad_bench = bench (make_name "Tanh" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.tanh x)) a)) in
    let softmax_grad_bench = bench (make_name "Softmax" size "backward") 
      (fun () -> ignore (Rune.grad (fun x -> Rune.sum (Rune.softmax x ~axes:[-1])) a)) in
      
    (* JIT compilation benchmarks with error handling *)
    let jit_benchmarks = List.filter_map (fun x -> x) [
      safe_jit_bench (make_name "ReLU" size "jit") (fun x -> Rune.relu x) a;
      safe_jit_bench (make_name "Sigmoid" size "jit") (fun x -> Rune.sigmoid x) a;
      safe_jit_bench (make_name "Tanh" size "jit") (fun x -> Rune.tanh x) a;
      safe_jit_bench (make_name "Softmax" size "jit") (fun x -> Rune.softmax x ~axes:[-1]) a;
    ] in
      
    relu_bench :: sigmoid_bench :: tanh_bench :: softmax_bench ::
    relu_grad_bench :: sigmoid_grad_bench :: tanh_grad_bench :: softmax_grad_bench ::
    jit_benchmarks @ acc
  ) [] sizes

(* Main benchmark runner *)
let () =
  let benchmarks = 
    bench_elementwise () @ 
    bench_reductions () @ 
    bench_linalg () @ 
    bench_shape_ops () @ 
    bench_neural_ops () 
  in
  
  (* Run benchmarks with ubench's built-in output *)
  ignore (run ~config benchmarks);
  
  (* Print system information *)
  Printf.printf "\n## System Information\n\n";
  Printf.printf "- OCaml version: %s\n" (Sys.ocaml_version);
  Printf.printf "- Rune version: Development\n";
  Printf.printf "- Test sizes: %s\n" (String.concat ", " (List.map string_of_int sizes));
  Printf.printf "- Data types: Float32\n";
  Printf.printf "- Measurements per benchmark: %d\n" 10;
  Printf.printf "- Warmup iterations: %d\n" 3