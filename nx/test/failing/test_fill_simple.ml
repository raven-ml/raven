(* Simple test for fill operation *)

let () =
  Printf.printf "Testing simple fill...\n";
  flush stdout;
  
  (* Test with standard float32 first *)
  let arr_f32 = Bigarray_ext.Array1.create Bigarray_ext.Float32 Bigarray_ext.c_layout 3 in
  Printf.printf "Created Float32 array\n";
  flush stdout;
  
  Bigarray_ext.Array1.fill arr_f32 1.5;
  Printf.printf "Filled Float32 array with 1.5\n";
  flush stdout;
  
  let v = Bigarray_ext.Array1.get arr_f32 0 in
  Printf.printf "arr_f32[0] = %f\n" v;
  flush stdout;
  
  (* Now test with bfloat16 *)
  Printf.printf "\nTesting BFloat16 fill...\n";
  flush stdout;
  
  let arr_bf16 = Bigarray_ext.Array1.create Bigarray_ext.Bfloat16 Bigarray_ext.c_layout 3 in
  Printf.printf "Created BFloat16 array\n";
  flush stdout;
  
  Printf.printf "About to call fill on BFloat16...\n";
  flush stdout;
  
  Printf.printf "Calling fill now\n%!";
  Bigarray_ext.Array1.fill arr_bf16 2.5;
  Printf.printf "Fill returned\n%!";
  
  Printf.printf "Filled BFloat16 array with 2.5\n";
  flush stdout;
  
  let v2 = Bigarray_ext.Array1.get arr_bf16 0 in
  Printf.printf "arr_bf16[0] = %f\n" v2;
  
  Printf.printf "\nTest completed!\n"