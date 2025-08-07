(* Test extended types through Nx API *)

let () =
  Printf.printf "Testing extended types through Nx Native...\n";
  flush stdout;
  
  (* Create context *)
  let ctx = Nx_native.create_context () in
  Printf.printf "Created context\n";
  flush stdout;
  
  (* Test 1: Create bfloat16 constant *)
  Printf.printf "Test 1: Creating bfloat16 scalar...\n";
  flush stdout;
  
  let t = Nx_native.op_const_scalar ctx 42.5 Nx_core.Dtype.bfloat16 in
  Printf.printf "  Created bfloat16 scalar tensor\n";
  flush stdout;
  
  (* Test 2: Get shape *)
  Printf.printf "Test 2: Getting shape...\n";
  flush stdout;
  
  let view = Nx_native.view t in
  let shape = Nx_core.View.shape view in
  Printf.printf "  Shape: [|%s|]\n" 
    (String.concat "; " (Array.to_list (Array.map string_of_int shape)));
  flush stdout;
  
  (* Test 3: Get data *)
  Printf.printf "Test 3: Getting data...\n";
  flush stdout;
  
  let data = Nx_native.data t in
  Printf.printf "  Data type matches bfloat16: %b\n" 
    (Bigarray_ext.Array1.kind data = Bigarray_ext.Bfloat16);
  flush stdout;
  
  if Bigarray_ext.Array1.dim data > 0 then begin
    let value = Bigarray_ext.Array1.get data 0 in
    Printf.printf "  Data[0]: %f\n" value;
    flush stdout
  end;
  
  (* Test 4: Create through frontend *)
  Printf.printf "Test 4: Creating through frontend...\n";
  flush stdout;
  
  let module Frontend = Nx_core.Make_frontend(Nx_native) in
  Printf.printf "  Frontend module created\n";
  flush stdout;
  
  Printf.printf "  Calling Frontend.create...\n";
  flush stdout;
  
  (* Test with simpler case first *)
  Printf.printf "  Creating scalar first...\n";
  flush stdout;
  
  let _t_scalar = Frontend.scalar ctx Nx_core.Dtype.bfloat16 1.0 in
  Printf.printf "  Created scalar successfully\n";
  flush stdout;
  
  Printf.printf "  Now creating array manually...\n";
  flush stdout;
  
  (* Test the issue directly *)
  let kind = Nx_core.Dtype.to_bigarray_ext_kind Nx_core.Dtype.bfloat16 in
  Printf.printf "  Got kind\n";
  flush stdout;
  
  let ba = Bigarray_ext.Array1.create kind Bigarray_ext.c_layout 3 in
  Printf.printf "  Created bigarray\n";
  flush stdout;
  
  Printf.printf "  Setting element 0...\n";
  flush stdout;
  Bigarray_ext.Array1.unsafe_set ba 0 1.0;
  Printf.printf "  Set element 0\n";
  flush stdout;
  
  Bigarray_ext.Array1.unsafe_set ba 1 2.0;
  Bigarray_ext.Array1.unsafe_set ba 2 3.0;
  Printf.printf "  Set all elements\n";
  flush stdout;
  
  Printf.printf "  Now testing blit directly...\n";
  flush stdout;
  
  let dst = Bigarray_ext.Array1.create kind Bigarray_ext.c_layout 3 in
  Printf.printf "  Created destination array\n";
  flush stdout;
  
  Printf.printf "  Calling blit...\n";
  flush stdout;
  
  Bigarray_ext.Array1.blit ba dst;
  Printf.printf "  Blit succeeded\n";
  flush stdout;
  
  Printf.printf "  Now testing op_const_array directly...\n";
  flush stdout;
  
  let _t_direct = Nx_native.op_const_array ctx ba in
  Printf.printf "  op_const_array succeeded\n";
  flush stdout;
  
  Printf.printf "  Now using Frontend.create...\n";
  flush stdout;
  
  let t2 = Frontend.create ctx Nx_core.Dtype.bfloat16 [|3|] [|1.0; 2.0; 3.0|] in
  Printf.printf "  Created bfloat16 array [1.0; 2.0; 3.0]\n";
  flush stdout;
  
  Printf.printf "  Getting shape...\n";
  flush stdout;
  
  let shape = Frontend.shape t2 in
  Printf.printf "  Shape: [|%s|]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int shape)));
  flush stdout;
  
  Printf.printf "\nAll tests passed!\n"