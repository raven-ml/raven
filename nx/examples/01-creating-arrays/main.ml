(** Build arrays from scratch — constants, ranges, grids, and custom data.

    This example walks through the most common ways to create arrays. By the end
    you'll know how to pick a dtype, fill arrays with constants, generate
    ranges, and build grids and triangular matrices. *)

open Nx

let () =
  (* Constant-filled arrays: zeros, ones, and an arbitrary fill value. *)
  let z = zeros float32 [| 2; 3 |] in
  Printf.printf "zeros (2×3):\n%s\n\n" (data_to_string z);

  let o = ones float64 [| 3 |] in
  Printf.printf "ones (3):\n%s\n\n" (data_to_string o);

  let pi = full float64 [| 2; 2 |] Float.pi in
  Printf.printf "full π (2×2):\n%s\n\n" (data_to_string pi);

  (* Ranges: integer steps and evenly-spaced floats. *)
  let ints = arange int32 0 10 1 in
  Printf.printf "arange 0..9:\n%s\n\n" (data_to_string ints);

  let spaced = linspace float64 0.0 1.0 5 in
  Printf.printf "linspace 0..1 (5 points):\n%s\n\n" (data_to_string spaced);

  let decades = logspace float64 1.0 4.0 4 in
  Printf.printf "logspace 10¹..10⁴:\n%s\n\n" (data_to_string decades);

  (* From raw data: pack an OCaml array into a 2×3 matrix. *)
  let data = create float64 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  Printf.printf "create from data (2×3):\n%s\n\n" (data_to_string data);

  (* Build a multiplication table with [init]. *)
  let mul_table =
    init int32 [| 5; 5 |] (fun idx ->
        Int32.of_int ((idx.(0) + 1) * (idx.(1) + 1)))
  in
  Printf.printf "5×5 multiplication table:\n%s\n\n" (data_to_string mul_table);

  (* Identity and eye: diagonal matrices. *)
  let id = identity float64 3 in
  Printf.printf "identity 3×3:\n%s\n\n" (data_to_string id);

  let e = eye ~k:1 float64 3 in
  Printf.printf "eye (k=1, superdiagonal):\n%s\n\n" (data_to_string e);

  (* Coordinate grids with meshgrid. *)
  let xs = arange_f float64 0.0 3.0 1.0 in
  let ys = arange_f float64 0.0 2.0 1.0 in
  let grid_x, grid_y = meshgrid xs ys in
  Printf.printf "meshgrid X:\n%s\n" (data_to_string grid_x);
  Printf.printf "meshgrid Y:\n%s\n\n" (data_to_string grid_y);

  (* Triangular matrices: tril and triu. *)
  let m = ones float64 [| 4; 4 |] in
  Printf.printf "tril (lower triangle):\n%s\n" (data_to_string (tril m));
  Printf.printf "triu (upper triangle):\n%s\n" (data_to_string (triu m))
