open Nx

(* Test data specification *)
let test_specs =
  [
    (* ("tiny_4x4", [| 1; 1; 4; 4 |], [| 1; 1; 3; 3 |]); *)
    (* ("small_8x8", [| 1; 1; 8; 8 |], [| 1; 1; 3; 3 |]); *)
    ("medium_16x16", [| 1; 4; 16; 16 |], [| 8; 4; 3; 3 |]);
    (* Skip large tests for now - they're too slow and might cause memory issues *)
    (* ("channels_32x32", [| 1; 8; 32; 32 |], [| 16; 8; 3; 3 |]); *)
    (* ("kernel_5x5", [| 1; 4; 16; 16 |], [| 8; 4; 5; 5 |]); *)
    (* ("batch_16x16", [| 4; 4; 16; 16 |], [| 8; 4; 3; 3 |]); *)
  ]

(* Create all test data upfront and keep references *)
let test_data =
  List.map
    (fun (name, x_shape, k_shape) ->
      let x = ones float32 x_shape in
      let k = ones float32 k_shape in
      (name, x, k))
    test_specs

(* Benchmark original implementation *)
let bench_original () =
  List.map
    (fun (name, x, k) ->
      Ubench.create ("orig_" ^ name) (fun () ->
          Nx.convolve2d ~padding_mode:`Valid x k |> ignore))
    test_data

(* Benchmark optimized implementation *)
let bench_optimized () =
  List.map
    (fun (name, x, k) ->
      Ubench.create ("opt_" ^ name) (fun () ->
          Nx_conv.convolve2d ~padding_mode:`Valid x k |> ignore))
    test_data

let () =
  Printf.printf "Convolution Benchmarks\n";
  Printf.printf "=====================\n\n";

  let tests = bench_original () @ bench_optimized () in

  Printf.printf "Running %d benchmarks...\n" (List.length tests);
  flush stdout;

  let results = Ubench.run ~warmup:1 ~trials:3 ~min_time:0.01 tests in
  Ubench.print_report results
