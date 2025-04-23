open Ndarray

(* Array Manipulation: Reshaping, Slicing, and Transforming Arrays *)

let print_separator () = Printf.printf "\n%s\n\n" (String.make 50 '-')

let () =
  (* Create a sample array for demonstrations *)
  let data =
    init float64 [| 3; 4 |] (fun idx ->
        float_of_int ((idx.(0) * 4) + idx.(1) + 1))
  in

  Printf.printf "Original array (3x4):\n%s\n" (to_string data);
  print_separator ();

  (* 1. Reshaping arrays *)
  Printf.printf "1. Reshaping Arrays\n";

  let reshaped_2_6 = reshape [| 2; 6 |] data in
  Printf.printf "Reshaped to (2x6):\n%s\n" (to_string reshaped_2_6);

  let reshaped_6_2 = reshape [| 6; 2 |] data in
  Printf.printf "Reshaped to (6x2):\n%s\n" (to_string reshaped_6_2);

  let flattened = flatten data in
  Printf.printf "Flattened to 1D:\n%s\n" (to_string flattened);

  let raveled = ravel data in
  Printf.printf "Raveled (same as flatten):\n%s\n" (to_string raveled);

  print_separator ();

  (* 2. Slicing and indexing *)
  Printf.printf "2. Slicing and Indexing\n";

  (* Get the first row *)
  let first_row = get [| 0 |] data in
  Printf.printf "First row:\n%s\n" (to_string first_row);

  (* Get the second column *)
  let second_col = get [| 1 |] data in
  Printf.printf "Second column:\n%s\n" (to_string second_col);

  print_separator ();

  (* 3. Transposing and permuting axes *)
  Printf.printf "3. Transposing and Permuting Axes\n";

  let transposed = transpose data in
  Printf.printf "Transposed array:\n%s\n" (to_string transposed);

  (* Create a 3D array *)
  let array_3d = reshape [| 2; 3; 2 |] (arange float64 0 12 1) in
  Printf.printf "3D array (2x3x2):\n%s\n" (to_string array_3d);

  let permuted = transpose ~axes:[| 2; 0; 1 |] array_3d in
  Printf.printf "Permuted axes [2,0,1]:\n%s\n" (to_string permuted);

  print_separator ();

  (* 4. Stacking and concatenating arrays *)
  Printf.printf "4. Stacking and Concatenating Arrays\n";

  let a =
    init float64 [| 2; 2 |] (fun idx ->
        float_of_int ((idx.(0) * 2) + idx.(1) + 1))
  in
  let b =
    init float64 [| 2; 2 |] (fun idx ->
        float_of_int ((idx.(0) * 2) + idx.(1) + 5))
  in

  Printf.printf "Array A:\n%s\n" (to_string a);
  Printf.printf "Array B:\n%s\n" (to_string b);

  let vertical = vstack [ a; b ] in
  Printf.printf "Vertical stack (vstack):\n%s\n" (to_string vertical);

  let horizontal = hstack [ a; b ] in
  Printf.printf "Horizontal stack (hstack):\n%s\n" (to_string horizontal);

  let depth = dstack [ a; b ] in
  Printf.printf "Depth stack (dstack):\n%s\n" (to_string depth);

  let concatenated = concatenate ~axis:0 [ a; b ] in
  Printf.printf "Concatenated along axis 0:\n%s\n" (to_string concatenated);

  print_separator ();

  (* 5. Splitting arrays *)
  Printf.printf "5. Splitting Arrays\n";

  let to_split = arange float64 0 9 1 in
  Printf.printf "Array to split:\n%s\n" (to_string to_split);

  let splits = split 3 to_split in
  Printf.printf "Split into 3 equal parts:\n";
  List.iteri
    (fun i arr -> Printf.printf "Part %d:\n%s\n" (i + 1) (to_string arr))
    splits;

  let arr_2d = reshape [| 3; 3 |] to_split in
  Printf.printf "2D array (3x3):\n%s\n" (to_string arr_2d);

  let col_splits = split ~axis:1 3 arr_2d in
  Printf.printf "Split columns into 3 parts:\n";
  List.iteri
    (fun i arr -> Printf.printf "Column %d:\n%s\n" (i + 1) (to_string arr))
    col_splits;

  print_separator ();

  (* 6. Expanding and squeezing dimensions *)
  Printf.printf "6. Expanding and Squeezing Dimensions\n";

  let vector = arange float64 0 5 1 in
  Printf.printf "Original vector (1D):\n%s\n" (to_string vector);

  let expanded = expand_dims 0 vector in
  Printf.printf "Expanded at axis 0 (row vector):\n%s\n" (to_string expanded);

  let expanded_col = expand_dims 1 vector in
  Printf.printf "Expanded at axis 1 (column vector):\n%s\n"
    (to_string expanded_col);

  (* Create a 3D array with a singleton dimension *)
  let array_with_singleton = reshape [| 2; 1; 3 |] (arange float64 0 6 1) in
  Printf.printf "Array with singleton dimension (2x1x3):\n%s\n"
    (to_string array_with_singleton);

  let squeezed = squeeze array_with_singleton in
  Printf.printf "After squeezing singleton dimensions:\n%s\n"
    (to_string squeezed);

  print_separator ();

  (* 7. Tiling and repeating arrays *)
  Printf.printf "7. Tiling and Repeating Arrays\n";

  let small = create float64 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Printf.printf "Original small array (2x2):\n%s\n" (to_string small);

  let tiled = tile [| 2; 3 |] small in
  Printf.printf "Tiled (2x3 times):\n%s\n" (to_string tiled);

  let repeated = repeat 3 small in
  Printf.printf "Elements repeated 3 times (flattened):\n%s\n"
    (to_string repeated);

  let repeated_axis = repeat ~axis:0 2 small in
  Printf.printf "Rows repeated 2 times:\n%s\n" (to_string repeated_axis);

  print_separator ();

  (* 8. Padding arrays *)
  Printf.printf "8. Padding Arrays\n";

  let to_pad = create float64 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  Printf.printf "Original array to pad (2x3):\n%s\n" (to_string to_pad);

  let padded = pad [| (1, 1); (2, 1) |] 0.0 to_pad in
  Printf.printf "Padded with zeros (pad_width=[(1,1), (2,1)]):\n%s\n"
    (to_string padded);

  print_separator ();

  (* 9. Flipping and rolling arrays *)
  Printf.printf "9. Flipping and Rolling Arrays\n";

  let to_flip = reshape [| 3; 3 |] (arange float64 1 10 1) in
  Printf.printf "Original array (3x3):\n%s\n" (to_string to_flip);

  let flipped_ud = flip ~axes:[| 0 |] to_flip in
  Printf.printf "Flipped up-down (axis 0):\n%s\n" (to_string flipped_ud);

  let flipped_lr = flip ~axes:[| 1 |] to_flip in
  Printf.printf "Flipped left-right (axis 1):\n%s\n" (to_string flipped_lr);

  let rolled = roll 1 to_flip in
  Printf.printf "Rolled by 1 (flattened):\n%s\n" (to_string rolled);

  let rolled_axis = roll ~axis:0 1 to_flip in
  Printf.printf "Rolled rows down by 1:\n%s\n" (to_string rolled_axis)
