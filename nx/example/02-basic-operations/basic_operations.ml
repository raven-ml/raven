open Nx

(* Basic Nx Operations: Arithmetic, indexing, and transformations *)

let () =
  (* Create some sample arrays *)
  let a =
    init float64 [| 2; 3 |] (fun idx ->
        float_of_int ((idx.(0) * 3) + idx.(1) + 1)
        (* Values 1-6 in row-major order *))
  in

  let b =
    init float64 [| 2; 3 |] (fun idx ->
        float_of_int (idx.(0) + idx.(1) + 1) (* Values based on row+col+1 *))
  in

  (* Display our sample arrays *)
  Printf.printf "Array A:\n%s\n\n" (to_string a);
  Printf.printf "Array B:\n%s\n\n" (to_string b);

  (* 1. Accessing elements *)
  let element = get_item [| 0; 1 |] a in
  Printf.printf "Element A[0,1]: %.1f\n\n" element;

  (* 2. Setting elements *)
  let a_modified = copy a in
  set_item [| 0; 1 |] 10.0 a_modified;
  Printf.printf "Modified A (changed A[0,1] to 10.0):\n%s\n\n"
    (to_string a_modified);

  (* 3. Basic arithmetic operations *)
  let sum = add a b in
  Printf.printf "A + B:\n%s\n\n" (to_string sum);

  let diff = sub a b in
  Printf.printf "A - B:\n%s\n\n" (to_string diff);

  let product = mul a b in
  Printf.printf "A * B (element-wise):\n%s\n\n" (to_string product);

  let quotient = div a b in
  Printf.printf "A / B (element-wise):\n%s\n\n" (to_string quotient);

  (* 4. Scalar operations *)
  let scaled = mul_scalar a 2.0 in
  Printf.printf "A * 2.0:\n%s\n\n" (to_string scaled);

  (* 5. In-place operations *)
  let a_inplace = copy a in
  let _ = add_inplace a_inplace b in
  (* a_inplace += b *)
  Printf.printf "A += B (in-place):\n%s\n\n" (to_string a_inplace);

  (* 6. Mathematical functions *)
  let squared = square a in
  Printf.printf "A² (element-wise):\n%s\n\n" (to_string squared);

  let sqrt_a = sqrt (abs a) in
  Printf.printf "√|A| (element-wise):\n%s\n\n" (to_string sqrt_a);

  (* 7. Matrix transformations *)
  let transposed = transpose a in
  Printf.printf "A transposed:\n%s\n\n" (to_string transposed);

  let reshaped = reshape [| 3; 2 |] a in
  Printf.printf "A reshaped to (3x2):\n%s\n\n" (to_string reshaped);

  (* 8. Flattening arrays *)
  let flattened = flatten a in
  Printf.printf "A flattened:\n%s\n\n" (to_string flattened);

  (* 9. Functional operations *)
  let mapped = map (fun x -> (x *. x) +. 1.0) a in
  Printf.printf "A mapped with f(x) = x² + 1:\n%s\n\n" (to_string mapped);

  (* 10. Comparison operations *)
  let greater_than = greater a b in
  Printf.printf "A > B (element-wise):\n%s\n\n" (to_string greater_than);

  (* 11. Clipping values *)
  let clipped = clip ~min:2.0 ~max:4.0 a in
  Printf.printf "A clipped to range [2.0, 4.0]:\n%s\n" (to_string clipped)
