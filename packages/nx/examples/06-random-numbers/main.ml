(** Random numbers from keys and scopes: generate distributions, sample, and
    shuffle.

    Roll dice, estimate pi with Monte Carlo, and generate noisy training data.
    Every draw is a function of a key: a sampler given [~key] draws from it, and
    one called without it takes the next key of the enclosing [Rng.with_key]
    scope, so the same seed gives the same numbers. Outside any scope, keys are
    seeded from the system's entropy and runs differ. *)

let () =
  (* --- Dice simulation: roll 10 six-sided dice --- *)
  let dice = Nx.Rng.randint ~low:1 ~high:7 [| 10 |] in
  Printf.printf "10 dice rolls: %s\n\n" (Nx.to_string dice);

  (* --- Uniform random floats in [0, 1) --- *)
  let uniform = Nx.Rng.uniform Nx.float64 [| 5 |] in
  Printf.printf "Uniform [0,1): %s\n\n" (Nx.to_string uniform);

  (* --- Normal distribution (mean=0, std=1) --- *)
  let normal = Nx.Rng.normal Nx.float64 [| 5 |] in
  Printf.printf "Normal(0,1):   %s\n\n" (Nx.to_string normal);

  (* --- Monte Carlo pi estimation ---

     Drop N random points in a unit square. The fraction landing inside the unit
     circle (distance from origin < 1) approximates pi/4. *)
  let n = 100_000 in
  let xs = Nx.Rng.uniform Nx.float64 [| n |] in
  let ys = Nx.Rng.uniform Nx.float64 [| n |] in
  let inside = Nx.(Infix.(less_s ((xs * xs) + (ys * ys)) 1.0)) in
  let count = Int64.to_float (Nx.item [] (Nx.count inside)) in
  let pi_est = count *. 4.0 /. Float.of_int n in
  Printf.printf "Monte Carlo pi (%d points): %.4f  (actual: %.4f)\n\n" n pi_est
    Float.pi;

  (* --- Synthetic training data: y = 3x + 2 + noise --- *)
  let x = Nx.Rng.uniform Nx.float64 [| 8 |] in
  let noise = Nx.(Infix.(Rng.normal float64 [| 8 |] *$ 0.1)) in
  let y = Nx.Infix.((x *$ 3.0) +$ 2.0 + noise) in
  Printf.printf "x:     %s\n" (Nx.to_string x);
  Printf.printf "y ~ 3x+2: %s\n\n" (Nx.to_string y);

  (* --- Reproducibility: one seed gives one result --- *)
  let draw () = Nx.Rng.normal Nx.float64 [| 3 |] in
  let a = Nx.Rng.with_key (Nx.Rng.key 99) draw in
  let b = Nx.Rng.with_key (Nx.Rng.key 99) draw in
  Printf.printf "Same seed, run 1: %s\n" (Nx.to_string a);
  Printf.printf "Same seed, run 2: %s\n" (Nx.to_string b);
  Printf.printf "Identical? %b\n\n" (Nx.item [] (Nx.array_equal a b));

  (* --- Explicit keys: split one key into independent ones --- *)
  let keys = Nx.Rng.split ~n:2 (Nx.Rng.key 7) in
  let c = Nx.Rng.normal ~key:keys.(0) Nx.float64 [| 3 |] in
  let d = Nx.Rng.normal ~key:keys.(1) Nx.float64 [| 3 |] in
  Printf.printf "Key 0: %s\n" (Nx.to_string c);
  Printf.printf "Key 1: %s\n\n" (Nx.to_string d);

  (* --- Shuffle: gather a dataset at a random permutation of its rows --- *)
  let data = Nx.arange Nx.int32 0 8 1 in
  let shuffled = Nx.take (Nx.Rng.permutation 8) data in
  Printf.printf "Original:  %s\n" (Nx.to_string data);
  Printf.printf "Shuffled:  %s\n" (Nx.to_string shuffled)
