open Ndarray

(* Hello World: Introduction to creating and displaying ndarrays *)

let () =
  (* Create arrays using different creation functions *)

  (* 1. Create a zeros array - 3x3 matrix filled with zeros *)
  let zeros = zeros float64 [| 3; 3 |] in
  Printf.printf "Zeros array (3x3):\n%s\n\n" (to_string zeros);

  (* 2. Create a ones array - 2x4 matrix filled with ones *)
  let ones = ones float64 [| 2; 4 |] in
  Printf.printf "Ones array (2x4):\n%s\n\n" (to_string ones);

  (* 3. Create an array filled with a specific value (3.14) *)
  let pi_array = full float64 [| 2; 2 |] 3.14 in
  Printf.printf "Array filled with π (2x2):\n%s\n\n" (to_string pi_array);

  (* 4. Create a range from 0 to 9 *)
  let range = arange int32 0 10 1 in
  Printf.printf "Range from 0 to 9:\n%s\n\n" (to_string range);

  (* 5. Create a range of floats with a specific step *)
  let float_range = arange_f float64 0.0 1.0 0.1 in
  Printf.printf "Float range from 0.0 to 0.9 (step 0.1):\n%s\n\n"
    (to_string float_range);

  (* 6. Create an identity matrix - 3x3 *)
  let identity = identity float64 3 in
  Printf.printf "Identity matrix (3x3):\n%s\n\n" (to_string identity);

  (* 7. Create a custom array using the 'create' function *)
  let data = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let custom = create float64 [| 2; 3 |] data in
  Printf.printf "Custom array from data (2x3):\n%s\n\n" (to_string custom);

  (* 8. Initialize an array with a function *)
  let init_array =
    init float64 [| 3; 3 |] (fun idx ->
        float_of_int (idx.(0) + idx.(1)) (* Sum of row and column indices *))
  in
  Printf.printf "Array initialized with function (3x3):\n%s\n\n"
    (to_string init_array)
