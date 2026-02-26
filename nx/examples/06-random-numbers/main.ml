(** Implicit RNG with reproducible scopes — generate distributions, sample, and
    shuffle.

    Roll dice, estimate pi with Monte Carlo, and generate noisy training data.
    Every result is reproducible inside an [Rng.run] scope: same seed, same
    numbers. Outside any scope the global fallback provides convenient but
    non-reproducible randomness. *)

open Nx
open Nx.Infix

let () =
  (* --- Dice simulation: roll 10 six-sided dice --- *)
  let dice = Rng.randint Int32 ~high:7 [| 10 |] 1 in
  Printf.printf "10 dice rolls: %s\n\n" (data_to_string dice);

  (* --- Uniform random floats in [0, 1) --- *)
  let uniform = Rng.uniform Float64 [| 5 |] in
  Printf.printf "Uniform [0,1): %s\n\n" (data_to_string uniform);

  (* --- Normal distribution (mean=0, std=1) --- *)
  let normal = Rng.normal Float64 [| 5 |] in
  Printf.printf "Normal(0,1):   %s\n\n" (data_to_string normal);

  (* --- Monte Carlo pi estimation ---

     Drop N random points in a unit square. The fraction landing inside the unit
     circle (distance from origin < 1) approximates pi/4. *)
  let n = 100_000 in
  let xs = rand float64 [| n |] in
  let ys = rand float64 [| n |] in
  let inside = less_s ((xs * xs) + (ys * ys)) 1.0 in
  let count = sum (cast Float64 inside) in
  let pi_est = item [] count *. 4.0 /. Float.of_int n in
  Printf.printf "Monte Carlo pi (%d points): %.4f  (actual: %.4f)\n\n" n pi_est
    Float.pi;

  (* --- Synthetic training data: y = 3x + 2 + noise --- *)
  let x = Rng.uniform Float64 [| 8 |] in
  let noise = Rng.normal Float64 [| 8 |] *$ 0.1 in
  let y = (x *$ 3.0) +$ 2.0 + noise in
  Printf.printf "x:     %s\n" (data_to_string x);
  Printf.printf "y ~ 3x+2: %s\n\n" (data_to_string y);

  (* --- Reproducibility: Rng.run ~seed gives the same result --- *)
  let a = Rng.run ~seed:99 (fun () -> Rng.normal Float64 [| 3 |]) in
  let b = Rng.run ~seed:99 (fun () -> Rng.normal Float64 [| 3 |]) in
  Printf.printf "Same seed, run 1: %s\n" (data_to_string a);
  Printf.printf "Same seed, run 2: %s\n" (data_to_string b);
  Printf.printf "Identical? %b\n\n" (item [] (all (equal a b)));

  (* --- Shuffle: random permutation of a dataset --- *)
  let data = arange int32 0 8 1 in
  let shuffled = Rng.shuffle data in
  Printf.printf "Original:  %s\n" (data_to_string data);
  Printf.printf "Shuffled:  %s\n" (data_to_string shuffled)
