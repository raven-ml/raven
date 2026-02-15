(** Element-wise math with operators — the Infix module makes array code read
    like algebra.

    Temperature conversions, BMI calculations, and score normalization, all
    expressed with clean infix operators instead of verbose function calls. *)

open Nx
open Nx.Infix

let () =
  (* --- Temperature conversion: C → F --- *)
  let celsius =
    create float64 [| 5 |] [| 0.0; 20.0; 37.0; 100.0; -40.0 |]
  in
  let fahrenheit = celsius *$ 1.8 +$ 32.0 in
  Printf.printf "Celsius:    %s\n" (data_to_string celsius);
  Printf.printf "Fahrenheit: %s\n\n" (data_to_string fahrenheit);

  (* --- BMI from height and weight arrays --- *)
  let height_m = create float64 [| 4 |] [| 1.65; 1.80; 1.72; 1.55 |] in
  let weight_kg = create float64 [| 4 |] [| 68.0; 90.0; 75.0; 52.0 |] in
  let bmi = weight_kg / (height_m * height_m) in
  Printf.printf "Heights (m): %s\n" (data_to_string height_m);
  Printf.printf "Weights (kg): %s\n" (data_to_string weight_kg);
  Printf.printf "BMI:          %s\n\n" (data_to_string bmi);

  (* --- Exam score normalization (min-max scaling to [0, 1]) --- *)
  let scores =
    create float64 [| 6 |] [| 72.0; 85.0; 60.0; 93.0; 78.0; 55.0 |]
  in
  let lo = min scores in
  let hi = max scores in
  let normalized = (scores - lo) / (hi - lo) in
  Printf.printf "Raw scores:  %s\n" (data_to_string scores);
  Printf.printf "Normalized:  %s\n\n" (data_to_string normalized);

  (* --- Math functions: exp, log, sqrt, abs --- *)
  let x = create float64 [| 5 |] [| -2.0; -1.0; 0.0; 1.0; 2.0 |] in
  Printf.printf "x:       %s\n" (data_to_string x);
  Printf.printf "abs(x):  %s\n" (data_to_string (abs x));
  Printf.printf "x²:      %s\n" (data_to_string (square x));
  Printf.printf "√|x|:    %s\n" (data_to_string (sqrt (abs x)));
  Printf.printf "exp(x):  %s\n" (data_to_string (exp x));
  Printf.printf "sign(x): %s\n\n" (data_to_string (sign x));

  (* --- Clamp: cap sensor readings to a valid range --- *)
  let readings =
    create float64 [| 6 |] [| -5.0; 12.0; 105.0; 42.0; -1.0; 99.0 |]
  in
  let clamped = clamp ~min:0.0 ~max:100.0 readings in
  Printf.printf "Sensor readings: %s\n" (data_to_string readings);
  Printf.printf "Clamped [0,100]: %s\n" (data_to_string clamped)
