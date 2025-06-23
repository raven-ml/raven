module Nx = Nx_core.Make_frontend (Nx_cblas)

let test_pooling () =
  Printf.printf "Testing avg_pool2d for non-deterministic behavior...\n\n";

  (* Create a simple input *)
  let input_shape = [| 1; 1; 4; 4 |] in
  let input_data = Array.init 16 (fun i -> float_of_int (i + 1)) in
  let x = Nx.create () Nx.float32 input_shape input_data in

  Printf.printf "Input:\n";
  for i = 0 to 3 do
    for j = 0 to 3 do
      Printf.printf "%2.0f " (Nx.unsafe_get [ 0; 0; i; j ] x)
    done;
    Printf.printf "\n"
  done;
  Printf.printf "\n";

  (* Run pooling multiple times *)
  Printf.printf "Running avg_pool2d 10 times with kernel_size=(2,2):\n";
  for run = 1 to 10 do
    let pooled = Nx.avg_pool2d x ~kernel_size:(2, 2) ~stride:(2, 2) in
    Printf.printf "Run %2d: " run;

    (* Check output values *)
    let has_extreme = ref false in
    for i = 0 to 1 do
      for j = 0 to 1 do
        let v = Nx.unsafe_get [ 0; 0; i; j ] pooled in
        Printf.printf "%8.2f " v;
        if abs_float v > 1000.0 || Float.is_nan v || Float.is_infinite v then
          has_extreme := true
      done
    done;

    if !has_extreme then Printf.printf " <- EXTREME VALUES!";
    Printf.printf "\n"
  done

let test_pooling_with_conv () =
  Printf.printf "\n\nTesting conv2d -> avg_pool2d sequence:\n";

  (* Simulate what happens in the model *)
  let x = Nx.randn () Nx.float32 ~seed:42 [| 32; 1; 28; 28 |] in
  let w = Nx.randn () Nx.float32 ~seed:43 [| 8; 1; 3; 3 |] in
  let b = Nx.zeros () Nx.float32 [| 8 |] in

  for run = 1 to 50 do
    (* Conv2d *)
    let conv = Nx.convolve2d x w ~stride:(1, 1) ~padding_mode:`Same in
    let b_reshaped = Nx.reshape [| 1; 8; 1; 1 |] b in
    let conv_biased = Nx.add conv b_reshaped in

    (* ReLU *)
    let relu = Nx.maximum conv_biased (Nx.scalar () Nx.float32 0.0) in

    (* Pooling *)
    let pooled = Nx.avg_pool2d relu ~kernel_size:(2, 2) ~stride:(2, 2) in

    (* Check for extreme values *)
    let max_val = Nx.unsafe_get [] (Nx.max (Nx.abs pooled)) in
    Printf.printf "Run %d: max abs value after pooling = %.2e" run max_val;

    if max_val > 1000.0 || Float.is_nan max_val || Float.is_infinite max_val
    then Printf.printf " <- EXTREME!";
    Printf.printf "\n"
  done

let () =
  test_pooling ();
  test_pooling_with_conv ()
