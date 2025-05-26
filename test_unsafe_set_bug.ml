let () =
  Printexc.record_backtrace true;

  (* Test 1: Direct unsafe_set on original array - should work *)
  let t1 = Nx.create Nx.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Printf.printf "Test 1: Direct unsafe_set on original array\n";
  Printf.printf "Before: ";
  Nx.print_data t1;
  Nx.set_item [ 0 ] 99.0 t1;
  Printf.printf "After: ";
  Nx.print_data t1;
  Printf.printf "\n";

  (* Test 2: unsafe_set on split view - currently fails *)
  let t2 = Nx.create Nx.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let parts = Nx.split ~axis:0 2 t2 in
  let p1 = List.nth parts 0 in

  Printf.printf "Test 2: unsafe_set on split view\n";
  Printf.printf "Original: ";
  Nx.print_data t2;
  Printf.printf "Split part (shape %s): "
    (Nx.shape p1 |> Array.to_list |> List.map string_of_int |> String.concat "x");
  Nx.print_data p1;

  (* Debug info *)
  Printf.printf "\nDebug info for split part:\n";
  Printf.printf "  Shape: %s\n"
    (Nx.shape p1 |> Array.to_list |> List.map string_of_int |> String.concat "x");
  Printf.printf "  Offset: %d\n" (Nx.offset p1);
  Printf.printf "  Strides: %s\n"
    (Nx.strides p1 |> Array.to_list |> List.map string_of_int
   |> String.concat ", ");
  Printf.printf "  Is contiguous: %b\n" (Nx.is_contiguous p1);
  Printf.printf "  Total elements: %d\n" (Nx.numel p1);

  (* Check the underlying data *)
  let data = Nx.unsafe_data p1 in
  Printf.printf "  Bigarray dims: %d\n" (Bigarray.Array1.dim data);
  Printf.printf "  First few elements of data: ";
  for i = 0 to min 3 (Bigarray.Array1.dim data - 1) do
    Printf.printf "%.1f " data.{i}
  done;
  Printf.printf "\n";

  try
    Printf.printf "\nTrying unsafe_set [0] 99.0 on split part...\n";
    Nx.set_item [ 0 ] 99.0 p1;
    Printf.printf "Success! After: ";
    Nx.print_data p1;
    Printf.printf "Original after modification: ";
    Nx.print_data t2
  with e ->
    Printf.printf "Failed with: %s\n" (Printexc.to_string e);
    Printf.printf "Backtrace:\n%s\n" (Printexc.get_backtrace ())
