open Nx

let () =
  Printf.printf "Testing set_slice behavior:\n";
  let a = zeros float32 [| 2; 3 |] in
  let value = create float32 [| 3 |] [| 7.; 8.; 9. |] in

  Printf.printf "Original array:\n";
  print_data a;
  Printf.printf "\nValue to set: ";
  print_data value;

  (* Try different slice specifications *)
  Printf.printf "\n\n1. Using set_slice [I 1] (should set row 1):\n";
  let a1 = copy a in
  set_slice [ I 1 ] a1 value;
  print_data a1;

  Printf.printf
    "\n\n2. Using set_slice [I 1; R []] (explicit full range for columns):\n";
  let a2 = copy a in
  let value2 = reshape [| 1; 3 |] value in
  set_slice [ I 1; R [] ] a2 value2;
  print_data a2;

  Printf.printf "\n\n3. Using set_slice [R [1; 1]; R []] (range for row):\n";
  let a3 = copy a in
  let value3 = reshape [| 1; 3 |] value in
  set_slice [ R [ 1; 1 ]; R [] ] a3 value3;
  print_data a3;

  Printf.printf "\n\n4. Direct shrink and blit (what should happen):\n";
  let a4 = copy a in
  let row1_view = shrink [| (1, 2); (0, 3) |] a4 in
  let value4 = reshape [| 1; 3 |] value in
  blit value4 row1_view;
  print_data a4
