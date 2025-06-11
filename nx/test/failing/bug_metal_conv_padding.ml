open Nx_core
module Context = Nx_metal

let shape_to_string t =
  let shape = View.shape (Context.view t) in
  Printf.sprintf "[%s]" (String.concat "; " (Array.to_list (Array.map string_of_int shape)))

let print_array name t =
  let ba = Context.data t in
  let offset = View.offset (Context.view t) in
  let numel = View.numel (Context.view t) in
  Printf.printf "%s: " name;
  for i = 0 to numel - 1 do
    Printf.printf "%.1f " (Bigarray.Array1.get ba (offset + i))
  done;
  Printf.printf "\n"

let () =
  let ctx = Context.create_context () in
  
  (* Test case from failing test *)
  let input_ba = Bigarray.Array1.of_array Bigarray.Float32 Bigarray.c_layout [| 1.; 2.; 3.; 4.; 5. |] in
  let input = Context.op_const_array ctx input_ba in
  let input = Context.op_reshape input [| 1; 1; 5 |] in
  
  let kernel_ba = Bigarray.Array1.of_array Bigarray.Float32 Bigarray.c_layout [| 1.; 1.; 1. |] in
  let kernel = Context.op_const_array ctx kernel_ba in
  let kernel = Context.op_reshape kernel [| 1; 1; 3 |] in
  
  Printf.printf "Input shape: %s\n" (shape_to_string input);
  print_array "Input" input;
  Printf.printf "Kernel shape: %s\n" (shape_to_string kernel);
  print_array "Kernel" kernel;
  
  (* Test same padding *)
  Printf.printf "\nTesting same padding:\n";
  (* For same padding with kernel size 3, we need padding (1,1) *)
  let padding = (1, 1) in
  
  (* Let's manually pad the input to see what happens *)
  let padded_ba = Bigarray.Array1.of_array Bigarray.Float32 Bigarray.c_layout [| 0.; 1.; 2.; 3.; 4.; 5.; 0. |] in
  let padded = Context.op_const_array ctx padded_ba in
  let padded = Context.op_reshape padded [| 1; 1; 7 |] in
  Printf.printf "Manually padded input shape: %s\n" (shape_to_string padded);
  print_array "Manually padded" padded;
  
  (* Now unfold without padding since input is already padded *)
  let unfolded_manual = Context.op_unfold padded 
    ~kernel_size:[|3|] 
    ~stride:[|1|] 
    ~dilation:[|1|] 
    ~padding:[|(0, 0)|] in
  Printf.printf "Unfolded manual shape: %s\n" (shape_to_string unfolded_manual);
  print_array "Unfolded manual" unfolded_manual;
  
  (* Also test the automatic padding *)
  let unfolded = Context.op_unfold input 
    ~kernel_size:[|3|] 
    ~stride:[|1|] 
    ~dilation:[|1|] 
    ~padding:[|padding|] in
  Printf.printf "Unfolded with padding shape: %s\n" (shape_to_string unfolded);
  print_array "Unfolded" unfolded;
  Printf.printf "Expected unfold: [[0,1,2], [1,2,3], [2,3,4], [3,4,5], [4,5,0]] -> 0,1,2,1,2,3,2,3,4,3,4,5,4,5,0\n";
  
  (* Let's also test without padding to see if that works *)
  Printf.printf "\nTesting without padding:\n";
  let unfolded_no_pad = Context.op_unfold input 
    ~kernel_size:[|3|] 
    ~stride:[|1|] 
    ~dilation:[|1|] 
    ~padding:[|(0, 0)|] in
  Printf.printf "Unfolded no padding shape: %s\n" (shape_to_string unfolded_no_pad);
  print_array "Unfolded no padding" unfolded_no_pad;
  Printf.printf "Expected unfold no padding: [[1,2,3], [2,3,4], [3,4,5]] -> 1,2,3,2,3,4,3,4,5\n";
  
  (* Now do the convolution manually using matmul *)
  (* Flip kernel for convolution *)
  let kernel_flipped_ba = Bigarray.Array1.of_array Bigarray.Float32 Bigarray.c_layout [| 1.; 1.; 1. |] in
  let kernel_flipped = Context.op_const_array ctx kernel_flipped_ba in
  let kernel_flipped = Context.op_reshape kernel_flipped [| 1; 3 |] in
  
  (* Matmul: [1,3] @ [3,5] -> [1,5] *)
  let result = Context.op_matmul kernel_flipped unfolded in
  let result = Context.op_reshape result [| 1; 1; 5 |] in
  
  Printf.printf "\nConvolution result shape: %s\n" (shape_to_string result);
  print_array "Result" result;
  Printf.printf "Expected: [3.0, 6.0, 9.0, 12.0, 9.0]\n"