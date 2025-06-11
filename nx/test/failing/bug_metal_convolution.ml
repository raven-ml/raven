open Nx_core
module Context = Nx_metal

let shape_to_string t =
  let shape = View.shape (Context.view t) in
  Printf.sprintf "[%s]" (String.concat "; " (Array.to_list (Array.map string_of_int shape)))

let numel t = View.numel (Context.view t)

let () =
  let ctx = Context.create_context () in
  
  (* Simple 1D convolution test - convert from bigarray *)
  let input_ba = Bigarray.Array1.of_array Bigarray.Float32 Bigarray.c_layout [| 1.; 2.; 3.; 4.; 5. |] in
  let input = Context.op_const_array ctx input_ba in
  let input = Context.op_reshape input [| 1; 1; 5 |] in
  
  let kernel_ba = Bigarray.Array1.of_array Bigarray.Float32 Bigarray.c_layout [| 1.; 0.; -1. |] in
  let kernel = Context.op_const_array ctx kernel_ba in
  let kernel = Context.op_reshape kernel [| 1; 1; 3 |] in
  
  Printf.printf "Input shape: %s\n" (shape_to_string input);
  Printf.printf "Kernel shape: %s\n" (shape_to_string kernel);
  
  (* Test unfold directly *)
  let unfolded = Context.op_unfold input 
    ~kernel_size:[|3|] 
    ~stride:[|1|] 
    ~dilation:[|1|] 
    ~padding:[|(0, 0)|] in
    
  Printf.printf "\nUnfolded shape: %s\n" (shape_to_string unfolded);
  
  (* Read back unfold result *)
  let unfolded_ba = Context.data unfolded in
  let offset = View.offset (Context.view unfolded) in
  Printf.printf "Unfolded values (numel=%d, offset=%d):\n" (numel unfolded) offset;
  for i = 0 to numel unfolded - 1 do
    Printf.printf "  [%d] = %.1f\n" i (Bigarray.Array1.get unfolded_ba (offset + i))
  done;
  Printf.printf "Expected unfold: [[1,2,3], [2,3,4], [3,4,5]] -> [1,2,3,2,3,4,3,4,5]\n"