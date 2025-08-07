(* Debug test for fill operation *)

let () =
  Printf.printf "Testing fill debug...\n";
  flush stdout;
  
  (* Create a bfloat16 array *)
  let arr = Bigarray_ext.Array1.create Bigarray_ext.Bfloat16 Bigarray_ext.c_layout 3 in
  Printf.printf "Created BFloat16 array\n";
  flush stdout;
  
  (* Set values manually first *)
  Bigarray_ext.Array1.set arr 0 1.0;
  Printf.printf "Set arr[0] = 1.0\n";
  flush stdout;
  
  let v = Bigarray_ext.Array1.get arr 0 in
  Printf.printf "Got arr[0] = %f\n" v;
  flush stdout;
  
  (* Try to call the fill function directly *)
  Printf.printf "About to call Bigarray_ext.Array1.fill\n";
  flush stdout;
  
  (* This is the line that hangs *)
  Bigarray_ext.Array1.fill arr 2.5;
  
  Printf.printf "Fill completed\n";
  flush stdout