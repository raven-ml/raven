open Nx

(* Broadcasting and Vectorized Operations *)

let () =
  Printf.printf "Nx Broadcasting Examples\n\n";

  (* 1. Basic broadcasting: matrix + scalar *)
  Printf.printf "1. Basic Broadcasting: Matrix + Scalar\n";
  let matrix =
    init float64 [| 3; 3 |] (fun idx ->
        float_of_int ((idx.(0) * 3) + idx.(1) + 1))
  in
  Printf.printf "Original matrix:\n%s\n" (to_string matrix);

  (* We can use mul_scalar for this specific case *)
  let scaled = mul_s matrix 2.0 in
  Printf.printf "Matrix x 2.0 (using mul_scalar):\n%s\n\n" (to_string scaled);

  (* 2. Matrix and vector operations *)
  Printf.printf "2. Broadcasting: Matrix + Row Vector\n";
  let matrix =
    init float64 [| 3; 4 |] (fun idx ->
        float_of_int ((idx.(0) * 4) + idx.(1) + 1))
  in
  let row_vector =
    init float64 [| 1; 4 |] (fun idx -> float_of_int (idx.(1) + 1))
  in

  Printf.printf "Matrix (3x4):\n%s\n" (to_string matrix);
  Printf.printf "Row vector (1x4):\n%s\n" (to_string row_vector);

  (* Broadcasting happens automatically in operations *)
  let result = add matrix row_vector in
  Printf.printf "Matrix + row_vector:\n%s\n\n" (to_string result);

  (* 3. Matrix and column vector operations *)
  Printf.printf "3. Broadcasting: Matrix + Column Vector\n";
  let column_vector =
    init float64 [| 3; 1 |] (fun idx -> float_of_int (idx.(0) + 1) *. 10.0)
  in

  Printf.printf "Matrix (3x4):\n%s\n" (to_string matrix);
  Printf.printf "Column vector (3x1):\n%s\n" (to_string column_vector);

  let result = add matrix column_vector in
  Printf.printf "Matrix + column_vector:\n%s\n\n" (to_string result);

  (* 4. Row vector + column vector (creates a matrix) *)
  Printf.printf "4. Broadcasting: Row Vector + Column Vector\n";
  Printf.printf "Row vector (1x4):\n%s\n" (to_string row_vector);
  Printf.printf "Column vector (3x1):\n%s\n" (to_string column_vector);

  let outer_sum = add row_vector column_vector in
  Printf.printf "Row vector + column vector (outer sum):\n%s\n\n"
    (to_string outer_sum);

  (* 5. Explicitly broadcasting arrays *)
  Printf.printf "5. Explicit Broadcasting with broadcast_to\n";
  let small_matrix =
    init float64 [| 2; 2 |] (fun idx ->
        float_of_int ((idx.(0) * 2) + idx.(1) + 1))
  in
  Printf.printf "Small matrix (2x2):\n%s\n" (to_string small_matrix);

  let broadcasted = broadcast_to [| 2; 2; 2 |] small_matrix in
  Printf.printf "Broadcasted to shape [2,2,2]:\n%s\n\n" (to_string broadcasted);

  (* 6. Broadcasting with arrays of different shapes *)
  Printf.printf "6. Broadcasting Arrays with Different Dimensions\n";
  let arr_3d =
    init float64 [| 2; 2; 2 |] (fun idx ->
        float_of_int ((idx.(0) * 4) + (idx.(1) * 2) + idx.(2) + 1))
  in
  let arr_1d = arange_f float64 1.0 3.0 1.0 in

  Printf.printf "3D array (2x2x2):\n%s\n" (to_string arr_3d);
  Printf.printf "1D array (2):\n%s\n" (to_string arr_1d);

  (* Reshape the 1D array to make broadcasting clearer *)
  let arr_1d_reshaped = reshape [| 2; 1; 1 |] arr_1d in
  Printf.printf "1D array reshaped to (2x1x1):\n%s\n"
    (to_string arr_1d_reshaped);

  let result_3d = mul arr_3d arr_1d_reshaped in
  Printf.printf "3D result of multiplication:\n%s\n\n" (to_string result_3d);

  (* 7. Broadcasting with explicit broadcast_arrays *)
  Printf.printf "7. Using broadcast_arrays\n";
  let a = arange float64 0 3 1 in
  (* [0,1,2] *)
  let b = arange float64 0 4 1 in
  (* [0,1,2,3] *)

  Printf.printf "Array a:\n%s\n" (to_string a);
  Printf.printf "Array b:\n%s\n" (to_string b);

  (* Reshape to make broadcasting work *)
  let a' = reshape [| 3; 1 |] a in
  let b' = reshape [| 1; 4 |] b in

  Printf.printf "Array a reshaped (3x1):\n%s\n" (to_string a');
  Printf.printf "Array b reshaped (1x4):\n%s\n" (to_string b');

  let[@warning "-8"] [ c'; d' ] = broadcast_arrays [ a'; b' ] in

  Printf.printf "Broadcast a':\n%s\n" (to_string c');
  Printf.printf "Broadcast b':\n%s\n" (to_string d')
