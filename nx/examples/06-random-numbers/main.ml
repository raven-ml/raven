(** Deterministic randomness with splittable keys — generate distributions,
    sample, and shuffle.

    Roll dice, estimate π with Monte Carlo, and generate noisy training data.
    Every result is reproducible: same seed, same numbers. *)

open Nx
open Nx.Infix

let () =
  (* Keys are the source of randomness. Same seed → same results. *)
  let key = Rng.key 42 in

  (* Split into independent subkeys for different experiments. *)
  let keys = Rng.split ~n:4 key in

  (* --- Dice simulation: roll 10 six-sided dice --- *)
  let dice = Rng.randint Int32 ~key:keys.(0) ~high:7 [| 10 |] 1 in
  Printf.printf "10 dice rolls: %s\n\n" (data_to_string dice);

  (* --- Uniform random floats in [0, 1) --- *)
  let uniform = Rng.uniform ~key:keys.(1) Float64 [| 5 |] in
  Printf.printf "Uniform [0,1): %s\n\n" (data_to_string uniform);

  (* --- Normal distribution (mean=0, std=1) --- *)
  let normal = Rng.normal ~key:keys.(2) Float64 [| 5 |] in
  Printf.printf "Normal(0,1):   %s\n\n" (data_to_string normal);

  (* --- Monte Carlo π estimation ---

     Drop N random points in a unit square. The fraction landing inside the
     unit circle (distance from origin < 1) approximates π/4. *)
  let n = 100_000 in
  let mc_keys = Rng.split ~n:2 keys.(3) in
  let xs = rand float64 ~key:mc_keys.(0) [| n |] in
  let ys = rand float64 ~key:mc_keys.(1) [| n |] in
  let inside = less_s ((xs * xs) + (ys * ys)) 1.0 in
  let count = sum (cast Float64 inside) in
  let pi_est = item [] count *. 4.0 /. Float.of_int n in
  Printf.printf "Monte Carlo π (%d points): %.4f  (actual: %.4f)\n\n" n pi_est
    Float.pi;

  (* --- Synthetic training data: y = 3x + 2 + noise --- *)
  let data_keys = Rng.split ~n:2 (Rng.key 7) in
  let x = Rng.uniform ~key:data_keys.(0) Float64 [| 8 |] in
  let noise = Rng.normal ~key:data_keys.(1) Float64 [| 8 |] *$ 0.1 in
  let y = x *$ 3.0 +$ 2.0 + noise in
  Printf.printf "x:     %s\n" (data_to_string x);
  Printf.printf "y ≈ 3x+2: %s\n\n" (data_to_string y);

  (* --- Reproducibility: same key always gives the same result --- *)
  let a = Rng.normal ~key:(Rng.key 99) Float64 [| 3 |] in
  let b = Rng.normal ~key:(Rng.key 99) Float64 [| 3 |] in
  Printf.printf "Same key, run 1: %s\n" (data_to_string a);
  Printf.printf "Same key, run 2: %s\n" (data_to_string b);
  Printf.printf "Identical? %b\n\n" (item [] (all (equal a b)));

  (* --- Shuffle: random permutation of a dataset --- *)
  let data = arange int32 0 8 1 in
  let shuffled = Rng.shuffle ~key:(Rng.key 0) data in
  Printf.printf "Original:  %s\n" (data_to_string data);
  Printf.printf "Shuffled:  %s\n" (data_to_string shuffled)
