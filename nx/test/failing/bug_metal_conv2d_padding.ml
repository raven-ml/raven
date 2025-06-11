open Nx_core
module Context = Nx_metal

let shape_to_string t =
  let shape = View.shape (Context.view t) in
  Printf.sprintf "[%s]" (String.concat "; " (Array.to_list (Array.map string_of_int shape)))

let print_array name t =
  let ba = Context.data t in
  let offset = View.offset (Context.view t) in
  let numel = View.numel (Context.view t) in
  Printf.printf "%s (shape=%s, numel=%d): " name (shape_to_string t) numel;
  for i = 0 to numel - 1 do
    Printf.printf "%.1f " (Bigarray.Array1.get ba (offset + i))
  done;
  Printf.printf "\n"

let () =
  let ctx = Context.create_context () in
  
  (* Test case from failing test - 2D convolution with same padding *)
  let input_ba = Bigarray.Array1.of_array Bigarray.Float32 Bigarray.c_layout 
    [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |] in
  let input = Context.op_const_array ctx input_ba in
  let input = Context.op_reshape input [| 1; 1; 3; 3 |] in
  
  let kernel_ba = Bigarray.Array1.of_array Bigarray.Float32 Bigarray.c_layout 
    [| 1.; 1.; 1.; 1. |] in
  let kernel = Context.op_const_array ctx kernel_ba in
  let kernel = Context.op_reshape kernel [| 1; 1; 2; 2 |] in
  
  Printf.printf "Input shape: %s\n" (shape_to_string input);
  print_array "Input" input;
  Printf.printf "Kernel shape: %s\n" (shape_to_string kernel);
  print_array "Kernel" kernel;
  
  (* Test unfold with same padding *)
  Printf.printf "\nTesting 2D unfold with same padding:\n";
  (* For kernel size (2,2) and input (3,3), same padding should give output (3,3) *)
  (* Padding needed: pad_h = (2-1)/2 = 0.5 -> (0,1), pad_w = (2-1)/2 = 0.5 -> (0,1) *)
  let padding = [|(0, 1); (0, 1)|] in
  let unfolded = Context.op_unfold input 
    ~kernel_size:[|2; 2|] 
    ~stride:[|1; 1|] 
    ~dilation:[|1; 1|] 
    ~padding:padding in
  Printf.printf "Unfolded with padding shape: %s\n" (shape_to_string unfolded);
  print_array "Unfolded" unfolded;
  
  (* Also test without padding *)
  Printf.printf "\nTesting 2D unfold without padding:\n";
  let unfolded_no_pad = Context.op_unfold input 
    ~kernel_size:[|2; 2|] 
    ~stride:[|1; 1|] 
    ~dilation:[|1; 1|] 
    ~padding:[|(0, 0); (0, 0)|] in
  Printf.printf "Unfolded no padding shape: %s\n" (shape_to_string unfolded_no_pad);
  print_array "Unfolded no padding" unfolded_no_pad;
  
  (* Expected for no padding: 
     Windows: [[1,2,4,5], [2,3,5,6], [4,5,7,8], [5,6,8,9]]
     As a flat array: 1,2,4,5,2,3,5,6,4,5,7,8,5,6,8,9
  *)
  Printf.printf "Expected unfold no padding: [[1,2,4,5], [2,3,5,6], [4,5,7,8], [5,6,8,9]]\n"