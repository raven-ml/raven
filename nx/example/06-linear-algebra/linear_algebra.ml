open Nx

(* Linear Algebra Operations with Nx *)

let print_separator () = Printf.printf "\n%s\n\n" (String.make 50 '-')

let () =
  (* Create sample matrices for linear algebra operations *)
  Printf.printf "Nx Linear Algebra Examples\n";
  print_separator ();

  (* 1. Matrix multiplication *)
  Printf.printf "1. Matrix Multiplication\n";

  let a =
    init float64 [| 2; 3 |] (fun idx ->
        float_of_int ((idx.(0) * 3) + idx.(1) + 1))
  in

  let b =
    init float64 [| 3; 2 |] (fun idx ->
        float_of_int ((idx.(0) * 2) + idx.(1) + 1))
  in

  Printf.printf "Matrix A (2x3):\n%s\n" (to_string a);
  Printf.printf "Matrix B (3x2):\n%s\n" (to_string b);

  let matrix_product = matmul a b in
  Printf.printf "Matrix product A x B (2x2):\n%s\n" (to_string matrix_product);

  (* Dot product of vectors *)
  let v1 = create float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let v2 = create float64 [| 3 |] [| 4.0; 5.0; 6.0 |] in

  Printf.printf "Vector v1:\n%s\n" (to_string v1);
  Printf.printf "Vector v2:\n%s\n" (to_string v2);

  let dot_product = dot v1 v2 in
  Printf.printf "Dot product v1 · v2:\n%s\n" (to_string dot_product)
