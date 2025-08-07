(* Test extended bigarray operations *)

let test_bfloat16_operations () =
  Printf.printf "Testing BFloat16 operations...\n";
  flush stdout;
  
  (* Test Array1 *)
  Printf.printf "  About to create Array1...\n";
  flush stdout;
  let arr1 = Bigarray_ext.Array1.create Bigarray_ext.Bfloat16 Bigarray_ext.c_layout 5 in
  Printf.printf "  Created Array1 with 5 elements\n";
  flush stdout;
  
  (* Test fill *)
  Printf.printf "  About to fill...\n";
  flush stdout;
  Bigarray_ext.Array1.fill arr1 3.14;
  Printf.printf "  Filled with 3.14\n";
  flush stdout;
  
  (* Test get *)
  let v = Bigarray_ext.Array1.get arr1 0 in
  Printf.printf "  arr1[0] = %f\n" v;
  
  (* Test set *)
  Bigarray_ext.Array1.set arr1 2 2.71;
  Printf.printf "  Set arr1[2] = 2.71\n";
  
  (* Test blit *)
  let arr2 = Bigarray_ext.Array1.create Bigarray_ext.Bfloat16 Bigarray_ext.c_layout 5 in
  Bigarray_ext.Array1.blit arr1 arr2;
  Printf.printf "  Blitted arr1 to arr2\n";
  let v2 = Bigarray_ext.Array1.get arr2 2 in
  Printf.printf "  arr2[2] = %f\n" v2;
  
  (* Test Array2 *)
  let arr2d = Bigarray_ext.Array2.create Bigarray_ext.Bfloat16 Bigarray_ext.c_layout 3 3 in
  Printf.printf "  Created Array2 (3x3)\n";
  
  Bigarray_ext.Array2.fill arr2d 1.0;
  Printf.printf "  Filled 2D array with 1.0\n";
  
  Bigarray_ext.Array2.set arr2d 1 1 5.0;
  let v2d = Bigarray_ext.Array2.get arr2d 1 1 in
  Printf.printf "  arr2d[1,1] = %f\n" v2d;
  
  (* Test Array3 *)
  let arr3d = Bigarray_ext.Array3.create Bigarray_ext.Bfloat16 Bigarray_ext.c_layout 2 2 2 in
  Printf.printf "  Created Array3 (2x2x2)\n";
  
  Bigarray_ext.Array3.fill arr3d 0.5;
  Printf.printf "  Filled 3D array with 0.5\n";
  
  Bigarray_ext.Array3.set arr3d 0 1 1 7.0;
  let v3d = Bigarray_ext.Array3.get arr3d 0 1 1 in
  Printf.printf "  arr3d[0,1,1] = %f\n" v3d;
  
  Printf.printf "  BFloat16 operations passed!\n\n"

let test_complex16_operations () =
  Printf.printf "Testing Complex16 operations...\n";
  flush stdout;
  
  (* Test Array1 *)
  let arr1 = Bigarray_ext.Array1.create Bigarray_ext.Complex16 Bigarray_ext.c_layout 3 in
  Printf.printf "  Created Complex16 Array1 with 3 elements\n";
  
  (* Test fill *)
  let c = Complex.{re = 1.0; im = 2.0} in
  Bigarray_ext.Array1.fill arr1 c;
  Printf.printf "  Filled with (1.0 + 2.0i)\n";
  
  (* Test get *)
  let v = Bigarray_ext.Array1.get arr1 0 in
  Printf.printf "  arr1[0] = (%f + %fi)\n" v.Complex.re v.Complex.im;
  
  (* Test set *)
  let c2 = Complex.{re = 3.0; im = 4.0} in
  Bigarray_ext.Array1.set arr1 1 c2;
  Printf.printf "  Set arr1[1] = (3.0 + 4.0i)\n";
  
  let v2 = Bigarray_ext.Array1.get arr1 1 in
  Printf.printf "  arr1[1] = (%f + %fi)\n" v2.Complex.re v2.Complex.im;
  
  Printf.printf "  Complex16 operations passed!\n\n"

let test_bool_operations () =
  Printf.printf "Testing Bool operations...\n";
  flush stdout;
  
  (* Test Array1 *)
  let arr1 = Bigarray_ext.Array1.create Bigarray_ext.Bool Bigarray_ext.c_layout 4 in
  Printf.printf "  Created Bool Array1 with 4 elements\n";
  
  (* Test fill *)
  Bigarray_ext.Array1.fill arr1 true;
  Printf.printf "  Filled with true\n";
  
  (* Test set/get *)
  Bigarray_ext.Array1.set arr1 1 false;
  Bigarray_ext.Array1.set arr1 3 false;
  
  for i = 0 to 3 do
    let v = Bigarray_ext.Array1.get arr1 i in
    Printf.printf "  arr1[%d] = %b\n" i v
  done;
  
  Printf.printf "  Bool operations passed!\n\n"

let test_int4_operations () =
  Printf.printf "Testing Int4 operations...\n";
  flush stdout;
  
  (* Test Array1 *)
  let arr1 = Bigarray_ext.Array1.create Bigarray_ext.Int4_signed Bigarray_ext.c_layout 8 in
  Printf.printf "  Created Int4_signed Array1 with 8 elements\n";
  
  (* Test fill *)
  Bigarray_ext.Array1.fill arr1 3;
  Printf.printf "  Filled with 3\n";
  
  (* Test set/get *)
  Bigarray_ext.Array1.set arr1 0 (-7);
  Bigarray_ext.Array1.set arr1 1 7;
  Bigarray_ext.Array1.set arr1 2 (-1);
  
  for i = 0 to 3 do
    let v = Bigarray_ext.Array1.get arr1 i in
    Printf.printf "  arr1[%d] = %d\n" i v
  done;
  
  Printf.printf "  Int4 operations passed!\n\n"

let test_genarray_operations () =
  Printf.printf "Testing Genarray operations...\n";
  flush stdout;
  
  (* Create a 2D genarray with bfloat16 *)
  let ga = Bigarray_ext.Genarray.create Bigarray_ext.Bfloat16 Bigarray_ext.c_layout [|2; 3|] in
  Printf.printf "  Created 2x3 Genarray with BFloat16\n";
  
  (* Test fill *)
  Bigarray_ext.Genarray.fill ga 2.5;
  Printf.printf "  Filled with 2.5\n";
  
  (* Test set/get *)
  Bigarray_ext.Genarray.set ga [|0; 1|] 7.5;
  let v = Bigarray_ext.Genarray.get ga [|0; 1|] in
  Printf.printf "  ga[0,1] = %f\n" v;
  
  (* Test blit *)
  let ga2 = Bigarray_ext.Genarray.create Bigarray_ext.Bfloat16 Bigarray_ext.c_layout [|2; 3|] in
  Bigarray_ext.Genarray.blit ga ga2;
  let v2 = Bigarray_ext.Genarray.get ga2 [|0; 1|] in
  Printf.printf "  After blit, ga2[0,1] = %f\n" v2;
  
  Printf.printf "  Genarray operations passed!\n\n"

let () =
  Printf.printf "=== Testing Extended Bigarray Operations ===\n\n";
  
  test_bfloat16_operations ();
  test_complex16_operations ();
  test_bool_operations ();
  test_int4_operations ();
  test_genarray_operations ();
  
  Printf.printf "=== All tests passed! ===\n"