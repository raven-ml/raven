(** Change array shapes and let broadcasting align dimensions automatically.

    Reshape a flat signal into frames, center data by subtracting column means
    (broadcasting in action), and build an outer product without any loops. *)

open Nx
open Nx.Infix

let () =
  (* --- Reshape: flat signal → frames --- *)
  let signal = arange_f float64 0.0 12.0 1.0 in
  Printf.printf "Flat signal (12 samples):\n%s\n\n" (data_to_string signal);

  let frames = reshape [| 3; 4 |] signal in
  Printf.printf "Reshaped into 3 frames of 4:\n%s\n\n" (data_to_string frames);

  let flat_again = flatten frames in
  Printf.printf "Flattened back: %s\n\n" (data_to_string flat_again);

  (* --- Transpose: swap rows and columns --- *)
  Printf.printf "Transposed:\n%s\n\n" (data_to_string (transpose frames));

  (* --- Stacking arrays --- *)
  let a = create float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let b = create float64 [| 3 |] [| 4.0; 5.0; 6.0 |] in
  Printf.printf "vstack [a; b]:\n%s\n" (data_to_string (vstack [ a; b ]));
  Printf.printf "hstack [a; b]: %s\n\n" (data_to_string (hstack [ a; b ]));

  (* --- Broadcasting: subtract column means to center data --- *)
  let data =
    create float64 [| 4; 3 |]
      [|
        10.0; 200.0; 3000.0;
        20.0; 400.0; 1000.0;
        30.0; 100.0; 2000.0;
        40.0; 300.0; 4000.0;
      |]
  in
  Printf.printf "Raw data (4 samples × 3 features):\n%s\n" (data_to_string data);

  (* Mean along axis 0 with keepdims — shape [1; 3] broadcasts against [4; 3]. *)
  let col_means = mean ~axes:[ 0 ] ~keepdims:true data in
  Printf.printf "Column means: %s\n" (data_to_string col_means);

  let centered = data - col_means in
  Printf.printf "Centered (zero-mean columns):\n%s\n\n" (data_to_string centered);

  (* --- Outer product via broadcasting --- *)
  let x = create float64 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let y = create float64 [| 3 |] [| 10.0; 20.0; 30.0 |] in

  (* x as column [4;1], y as row [1;3] → result is [4;3]. *)
  let outer = reshape [| 4; 1 |] x * reshape [| 1; 3 |] y in
  Printf.printf "x = %s\n" (data_to_string x);
  Printf.printf "y = %s\n" (data_to_string y);
  Printf.printf "Outer product (x × y):\n%s\n\n" (data_to_string outer);

  (* --- expand_dims / squeeze --- *)
  let v = arange float64 0 4 1 in
  let row = expand_dims [ 0 ] v in
  let col = expand_dims [ 1 ] v in
  Printf.printf "Vector:     shape %s → %s\n" (shape_to_string (shape v))
    (data_to_string v);
  Printf.printf "Row vector: shape %s → %s\n" (shape_to_string (shape row))
    (data_to_string row);
  Printf.printf "Col vector: shape %s\n%s\n"
    (shape_to_string (shape col))
    (data_to_string col)
