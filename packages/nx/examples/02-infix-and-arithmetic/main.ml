(** Element-wise math with operators: inside [Nx.Infix.( … )], array code
    reads like algebra.

    Temperature conversions, BMI calculations, and score normalization, all
    written with infix operators. [Nx.Infix] defines [+], [-], [*] and [/] on
    arrays, so a local open [Nx.Infix.( … )] applies them to arrays and
    leaves OCaml's integer arithmetic as it is outside the parentheses. *)

let () =
  (* --- Temperature conversion: C → F --- *)
  let celsius =
    Nx.create Nx.float64 [| 5 |] [| 0.0; 20.0; 37.0; 100.0; -40.0 |]
  in
  let fahrenheit = Nx.Infix.((celsius *$ 1.8) +$ 32.0) in
  Printf.printf "Celsius:    %s\n" (Nx.to_string celsius);
  Printf.printf "Fahrenheit: %s\n\n" (Nx.to_string fahrenheit);

  (* --- BMI from height and weight arrays --- *)
  let height_m = Nx.create Nx.float64 [| 4 |] [| 1.65; 1.80; 1.72; 1.55 |] in
  let weight_kg = Nx.create Nx.float64 [| 4 |] [| 68.0; 90.0; 75.0; 52.0 |] in
  let bmi = Nx.Infix.(weight_kg / (height_m * height_m)) in
  Printf.printf "Heights (m): %s\n" (Nx.to_string height_m);
  Printf.printf "Weights (kg): %s\n" (Nx.to_string weight_kg);
  Printf.printf "BMI:          %s\n\n" (Nx.to_string bmi);

  (* --- Exam score normalization (min-max scaling to [0, 1]) --- *)
  let scores =
    Nx.create Nx.float64 [| 6 |] [| 72.0; 85.0; 60.0; 93.0; 78.0; 55.0 |]
  in
  let lo = Nx.min scores in
  let hi = Nx.max scores in
  let normalized = Nx.Infix.((scores - lo) / (hi - lo)) in
  Printf.printf "Raw scores:  %s\n" (Nx.to_string scores);
  Printf.printf "Normalized:  %s\n\n" (Nx.to_string normalized);

  (* --- Math functions: exp, log, sqrt, abs --- *)
  let x = Nx.create Nx.float64 [| 5 |] [| -2.0; -1.0; 0.0; 1.0; 2.0 |] in
  Printf.printf "x:       %s\n" (Nx.to_string x);
  Printf.printf "abs(x):  %s\n" (Nx.to_string (Nx.abs x));
  Printf.printf "x²:      %s\n" (Nx.to_string (Nx.square x));
  Printf.printf "√|x|:    %s\n" (Nx.to_string (Nx.sqrt (Nx.abs x)));
  Printf.printf "exp(x):  %s\n" (Nx.to_string (Nx.exp x));
  Printf.printf "sign(x): %s\n\n" (Nx.to_string (Nx.sign x));

  (* --- Clamp: cap sensor readings to a valid range --- *)
  let readings =
    Nx.create Nx.float64 [| 6 |] [| -5.0; 12.0; 105.0; 42.0; -1.0; 99.0 |]
  in
  let clamped = Nx.clamp ~min:0.0 ~max:100.0 readings in
  Printf.printf "Sensor readings: %s\n" (Nx.to_string readings);
  Printf.printf "Clamped [0,100]: %s\n" (Nx.to_string clamped)
