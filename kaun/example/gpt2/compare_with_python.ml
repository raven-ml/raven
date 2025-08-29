(* Compare GPT-2 output with Python reference *)

open Kaun_models.GPT2

let () =
  Printf.printf "=== Comparing GPT-2 Output with Python ===\n\n";

  let device = Rune.c in
  let dtype = Rune.float32 in

  (* Load model *)
  let gpt2 = from_pretrained ~model_id:"gpt2" ~device ~dtype () in

  (* Create tokenizer *)
  let tokenizer = Tokenizer.create () in

  (* Test with same input as Python *)
  let test_text = "Hello world" in
  Printf.printf "Input: %S\n" test_text;

  (* Encode and run forward pass *)
  let inputs = Tokenizer.encode tokenizer test_text ~device in
  let output = forward gpt2 inputs () in

  (* Get output values *)
  let output_shape = Rune.shape output.last_hidden_state in
  Printf.printf "Output shape: [%d, %d, %d]\n" output_shape.(0) output_shape.(1)
    output_shape.(2);

  (* Get first 10 values and compare with Python reference *)
  (* Reshape to get a 1D view of first position *)
  let first_pos = Rune.slice [ I 0; I 0; A ] output.last_hidden_state in
  let first_10 = Rune.slice [ R (0, 10) ] first_pos in

  (* Convert to array for easier access *)
  let first_10_array = Rune.to_array first_10 in

  (* Print first 10 values *)
  Printf.printf "\nFirst 10 values from OCaml:\n";
  Array.iteri
    (fun i v -> Printf.printf "  [%d]: %.6f\n" i v)
    (Array.sub first_10_array 0 10);

  (* Python reference values from earlier *)
  let python_values =
    [|
      -9.088777005672455e-06;
      -0.14020976424217224;
      -0.20845119655132294;
      -0.028111519291996956;
      -0.09017819911241531;
      -0.19850760698318481;
      4.072483062744141;
      -0.24899107217788696;
      -0.17168566584587097;
      -0.01689625345170498;
    |]
  in

  Printf.printf "\nFirst 10 values from Python:\n";
  Array.iteri (fun i v -> Printf.printf "  [%d]: %.6f\n" i v) python_values;

  (* Compare values *)
  Printf.printf "\nComparison (tolerance = 1e-4):\n";
  let all_close = ref true in
  for i = 0 to 9 do
    let ocaml_val = first_10_array.(i) in
    let python_val = python_values.(i) in
    let diff = abs_float (ocaml_val -. python_val) in
    let close = diff < 1e-4 in
    if not close then all_close := false;
    Printf.printf "  [%d]: %s (diff = %.6e)\n" i
      (if close then "✓" else "✗")
      diff
  done;

  if !all_close then Printf.printf "\n✓ All values match within tolerance!\n"
  else
    Printf.printf
      "\n✗ Some values don't match - may need to check implementation\n"
