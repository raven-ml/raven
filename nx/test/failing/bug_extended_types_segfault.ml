(* Minimal reproduction of segfault with extended types *)

let () =
  Printf.printf "Testing extended types segfault...\n";
  flush stdout;
  
  (* Test 1: Create a simple bfloat16 array *)
  Printf.printf "Test 1: Creating bfloat16 array with Bigarray_ext...\n";
  flush stdout;
  
  let bf16_array = Bigarray_ext.Array1.create Bigarray_ext.Bfloat16 Bigarray_ext.c_layout 3 in
  Printf.printf "  Created bfloat16 array of size 3\n";
  flush stdout;
  
  (* Test 2: Set values *)
  Printf.printf "Test 2: Setting values...\n";
  flush stdout;
  
  Bigarray_ext.Array1.set bf16_array 0 1.0;
  Printf.printf "  Set index 0 to 1.0\n";
  flush stdout;
  
  Bigarray_ext.Array1.set bf16_array 1 2.0;
  Printf.printf "  Set index 1 to 2.0\n";
  flush stdout;
  
  Bigarray_ext.Array1.set bf16_array 2 3.0;
  Printf.printf "  Set index 2 to 3.0\n";
  flush stdout;
  
  (* Test 3: Get values *)
  Printf.printf "Test 3: Getting values...\n";
  flush stdout;
  
  let v0 = Bigarray_ext.Array1.get bf16_array 0 in
  Printf.printf "  Got index 0: %f\n" v0;
  flush stdout;
  
  let v1 = Bigarray_ext.Array1.get bf16_array 1 in
  Printf.printf "  Got index 1: %f\n" v1;
  flush stdout;
  
  let v2 = Bigarray_ext.Array1.get bf16_array 2 in
  Printf.printf "  Got index 2: %f\n" v2;
  flush stdout;
  
  (* Test 4: Test the kind function *)
  Printf.printf "Test 4: Testing kind function...\n";
  flush stdout;
  
  let kind = Bigarray_ext.Array1.kind bf16_array in
  Printf.printf "  Kind matches Bfloat16: %b\n" (kind = Bigarray_ext.Bfloat16);
  flush stdout;
  
  (* Test 5: Create other extended types *)
  Printf.printf "Test 5: Creating other extended types...\n";
  flush stdout;
  
  let bool_array = Bigarray_ext.Array1.create Bigarray_ext.Bool Bigarray_ext.c_layout 2 in
  Printf.printf "  Created bool array\n";
  flush stdout;
  ignore bool_array;
  
  let int4_array = Bigarray_ext.Array1.create Bigarray_ext.Int4_signed Bigarray_ext.c_layout 2 in
  Printf.printf "  Created int4_signed array\n";
  flush stdout;
  ignore int4_array;
  
  let uint4_array = Bigarray_ext.Array1.create Bigarray_ext.Int4_unsigned Bigarray_ext.c_layout 2 in
  Printf.printf "  Created int4_unsigned array\n";
  flush stdout;
  ignore uint4_array;
  
  let f8e4m3_array = Bigarray_ext.Array1.create Bigarray_ext.Float8_e4m3 Bigarray_ext.c_layout 2 in
  Printf.printf "  Created float8_e4m3 array\n";
  flush stdout;
  ignore f8e4m3_array;
  
  let f8e5m2_array = Bigarray_ext.Array1.create Bigarray_ext.Float8_e5m2 Bigarray_ext.c_layout 2 in
  Printf.printf "  Created float8_e5m2 array\n";
  flush stdout;
  ignore f8e5m2_array;
  
  let c16_array = Bigarray_ext.Array1.create Bigarray_ext.Complex16 Bigarray_ext.c_layout 2 in
  Printf.printf "  Created complex16 array\n";
  flush stdout;
  ignore c16_array;
  
  let qint8_array = Bigarray_ext.Array1.create Bigarray_ext.Qint8 Bigarray_ext.c_layout 2 in
  Printf.printf "  Created qint8 array\n";
  flush stdout;
  ignore qint8_array;
  
  let quint8_array = Bigarray_ext.Array1.create Bigarray_ext.Quint8 Bigarray_ext.c_layout 2 in
  Printf.printf "  Created quint8 array\n";
  flush stdout;
  ignore quint8_array;
  
  Printf.printf "\nAll tests passed without segfault!\n"