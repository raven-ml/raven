(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Explicit splittable RNG keys: sampler purity and statistics, key
   derivation (split, fold_in), bit-exact parity between eager and compiled
   execution (the C threefry kernel and Tolk's decomposition are the same
   function), the constant-key refusal inside jit, and composition with grad,
   vmap and pmap. *)

open Windtrap
open Rune_test_support.Support

module Key = struct
  type t = Rune.Rng.key

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

let raises_jit_error f =
  raises_match
    (fun exn -> match exn with Rune.Jit_error _ -> true | _ -> false)
    (fun () -> ignore (f ()))

(* Bitwise equality of float tensors: the parity claims are exact, not
   approximate. *)
let check_bits ~msg expected actual =
  let e = to_arr expected and a = to_arr actual in
  equal ~msg:(msg ^ " (length)") int (Array.length e) (Array.length a);
  is_true ~msg
    (Array.for_all2
       (fun x y -> Int32.bits_of_float x = Int32.bits_of_float y)
       e a)

let mean a = Array.fold_left ( +. ) 0.0 a /. float_of_int (Array.length a)

let correlation x y =
  let mx = mean x and my = mean y in
  let cov = ref 0.0 and vx = ref 0.0 and vy = ref 0.0 in
  Array.iteri
    (fun i xi ->
      cov := !cov +. ((xi -. mx) *. (y.(i) -. my));
      vx := !vx +. ((xi -. mx) ** 2.0);
      vy := !vy +. ((y.(i) -. my) ** 2.0))
    x;
  !cov /. sqrt (!vx *. !vy)

(* Samplers are pure *)

let test_same_key_same_values () =
  let u () = Rune.Rng.uniform (Rune.Rng.key 42) Nx.float32 [| 32 |] in
  check_bits ~msg:"same key, same values" (u ()) (u ())

let test_different_keys_differ () =
  let u seed = to_arr (Rune.Rng.uniform (Rune.Rng.key seed) f32 [| 32 |]) in
  is_true ~msg:"different keys, different values" (u 1 <> u 2)

let test_uniform_range_and_mean () =
  let a = to_arr (Rune.Rng.uniform (Rune.Rng.key 0) f32 [| 10_000 |]) in
  is_true ~msg:"in [0, 1)" (Array.for_all (fun v -> v >= 0.0 && v < 1.0) a);
  equal ~msg:"mean near 1/2" (float 0.02) 0.5 (mean a)

let test_normal_moments () =
  let a = to_arr (Rune.Rng.normal (Rune.Rng.key 3) f32 [| 10_000 |]) in
  let m = mean a in
  let var = mean (Array.map (fun v -> (v -. m) ** 2.0) a) in
  equal ~msg:"mean near 0" (float 0.05) 0.0 m;
  equal ~msg:"variance near 1" (float 0.05) 1.0 var

let test_randint_range () =
  let a = Nx.to_array (Rune.Rng.randint (Rune.Rng.key 5) ~high:9 [| 1000 |] 3) in
  is_true ~msg:"in [3, 9)" (Array.for_all (fun v -> v >= 3l && v < 9l) a)

let test_bernoulli_probability () =
  let a =
    to_arr
      (Nx.cast f32 (Rune.Rng.bernoulli (Rune.Rng.key 6) ~p:0.3 [| 10_000 |]))
  in
  equal ~msg:"fraction of ones near p" (float 0.02) 0.3 (mean a)

let test_sampler_argument_errors () =
  raises_invalid_arg (fun () ->
      ignore (Rune.Rng.randint (Rune.Rng.key 0) ~high:3 [| 4 |] 3));
  raises_invalid_arg (fun () ->
      ignore (Rune.Rng.bernoulli (Rune.Rng.key 0) ~p:1.5 [| 4 |]));
  raises_invalid_arg (fun () -> ignore (Rune.Rng.split ~n:0 (Rune.Rng.key 0)));
  raises_invalid_arg (fun () ->
      ignore (Rune.Rng.uniform (Nx.zeros Nx.int32 [| 3 |]) f32 [| 4 |]))

(* Key derivation *)

let test_split_is_deterministic () =
  let sub i = Nx.to_array (Rune.Rng.split (Rune.Rng.key 42)).(i) in
  is_true ~msg:"splitting twice gives the same subkeys" (sub 0 = sub 0);
  is_true ~msg:"subkeys differ" (sub 0 <> sub 1)

let test_split_independence () =
  let ks = Rune.Rng.split (Rune.Rng.key 42) in
  let draw k = to_arr (Rune.Rng.uniform k f32 [| 10_000 |]) in
  let x = draw ks.(0) and y = draw ks.(1) in
  is_true ~msg:"correlation near zero" (abs_float (correlation x y) < 0.03);
  equal ~msg:"subkey 0 mean sane" (float 0.02) 0.5 (mean x);
  equal ~msg:"subkey 1 mean sane" (float 0.02) 0.5 (mean y)

let test_fold_in_distinct_and_reproducible () =
  let root = Rune.Rng.key 7 in
  let draw i = to_arr (Rune.Rng.uniform (Rune.Rng.fold_in root i) f32 [| 8 |]) in
  let outs = Array.init 5 draw in
  for i = 0 to 4 do
    for j = i + 1 to 4 do
      is_true ~msg:(Printf.sprintf "steps %d and %d differ" i j)
        (outs.(i) <> outs.(j))
    done
  done;
  is_true ~msg:"reproducible from the root" (draw 3 = outs.(3))

(* Jit: bit parity and key threading *)

let test_jit_uniform_bit_parity () =
  let f key = Rune.Rng.uniform key Nx.float32 [| 1000 |] in
  let k = Rune.Rng.key 42 in
  let g = Rune.jit (module Key) f in
  check_bits ~msg:"eager == jit, bitwise" (f k) (g k);
  check_bits ~msg:"replay" (f k) (g k)

let test_jit_int_samplers_bit_parity () =
  let k = Rune.Rng.key 9 in
  let fr key = Nx.cast f32 (Rune.Rng.randint key ~high:9 [| 64 |] 3) in
  check_bits ~msg:"randint" (fr k) (Rune.jit (module Key) fr k);
  let fb key = Nx.cast f32 (Rune.Rng.bernoulli key ~p:0.3 [| 64 |]) in
  check_bits ~msg:"bernoulli" (fb k) (Rune.jit (module Key) fb k)

(* The threefry bits agree exactly; cos/log/sqrt in Box-Muller land within
   float32 ulps of eager (the compiler decomposes transcendentals). *)
let test_jit_normal_matches_eager () =
  let f key = Rune.Rng.normal key Nx.float32 [| 1000 |] in
  let k = Rune.Rng.key 42 in
  check_arr ~msg:"normal" (to_arr (f k)) (Rune.jit (module Key) f k)

let test_jit_split_derived_key_traces () =
  let f key =
    let ks = Rune.Rng.split key in
    Nx.add
      (Rune.Rng.uniform ks.(0) Nx.float32 [| 8 |])
      (Rune.Rng.uniform ks.(1) Nx.float32 [| 8 |])
  in
  let k = Rune.Rng.key 11 in
  check_bits ~msg:"keys split inside the trace" (f k) (Rune.jit (module Key) f k)

let test_jit_fold_in_driven_steps () =
  let root = Rune.Rng.key 3 in
  let g = Rune.jit (module Key) (fun key -> Rune.Rng.uniform key f32 [| 8 |]) in
  let outs = Array.init 5 (fun i -> to_arr (g (Rune.Rng.fold_in root i))) in
  for i = 0 to 4 do
    for j = i + 1 to 4 do
      is_true ~msg:(Printf.sprintf "steps %d and %d differ" i j)
        (outs.(i) <> outs.(j))
    done
  done;
  is_true ~msg:"reproducible from the root"
    (to_arr (g (Rune.Rng.fold_in root 3)) = outs.(3))

let test_jit_implicit_rng_raises () =
  let g = Rune.jit' (fun x -> Nx.add x (Nx.rand f32 [| 3 |])) in
  raises_jit_error (fun () -> g (vec32 [| 1.0; 2.0; 3.0 |]))

let test_jit_captured_key_raises () =
  let k = Rune.Rng.key 42 in
  let g = Rune.jit' (fun x -> Nx.add x (Rune.Rng.uniform k f32 [| 3 |])) in
  raises_jit_error (fun () -> g (vec32 [| 1.0; 2.0; 3.0 |]))

(* Autodiff: samples are constants of the tape *)

let test_grad_dropout_mask_is_constant () =
  let key = Rune.Rng.key 7 in
  let x = vec32 [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |] in
  let forward_mask = ref None in
  let f x =
    let m = Nx.cast f32 (Rune.Rng.bernoulli key ~p:0.5 [| 8 |]) in
    forward_mask := Some m;
    Nx.sum (Nx.mul x m)
  in
  let g = Rune.grad' f x in
  let m =
    match !forward_mask with Some m -> m | None -> assert false
  in
  (* The gradient is exactly the mask: the same values flowed through the
     forward and the backward pass, and no gradient flowed into the draw. *)
  check_bits ~msg:"gradient equals the forward mask" m g

(* Vmap: per-lane keys decorrelate lanes *)

let test_vmap_per_lane_keys () =
  let ks = Rune.Rng.split ~n:4 (Rune.Rng.key 42) in
  let stacked = Nx.stack ~axis:0 (Array.to_list ks) in
  let out =
    Rune.vmap' (fun key -> Rune.Rng.uniform key Nx.float32 [| 8 |]) stacked
  in
  equal ~msg:"one row per lane" int 4 (Nx.shape out).(0);
  (* Each lane draws what its key draws unbatched, and lanes differ. *)
  for i = 0 to 3 do
    check_bits
      ~msg:(Printf.sprintf "lane %d matches its key" i)
      (Rune.Rng.uniform ks.(i) Nx.float32 [| 8 |])
      (Nx.slice [ Nx.I i ] out)
  done;
  is_true ~msg:"lanes are decorrelated"
    (to_arr (Nx.slice [ Nx.I 0 ] out) <> to_arr (Nx.slice [ Nx.I 1 ] out))

(* Pmap: a replicated key replicates the samples. Per-device decorrelation
   (fold_in of a device index) is future work; today every device of the
   tuple draws the same values from a replicated key. *)

let test_pmap_replicated_key_smoke () =
  let k = Rune.Rng.key 42 in
  let g =
    Rune.pmap
      ~devices:[ "CPU:1"; "CPU:2" ]
      ~in_axes:[ None ]
      (module Key)
      (fun key -> Rune.Rng.uniform key Nx.float32 [| 8 |])
  in
  check_bits ~msg:"replicated key, replicated samples"
    (Rune.Rng.uniform k Nx.float32 [| 8 |])
    (g k)

let tests =
  [
    group "samplers"
      [
        test "same key, same values" test_same_key_same_values;
        test "different keys differ" test_different_keys_differ;
        test "uniform range and mean" test_uniform_range_and_mean;
        test "normal moments" test_normal_moments;
        test "randint range" test_randint_range;
        test "bernoulli probability" test_bernoulli_probability;
        test "argument validation" test_sampler_argument_errors;
      ];
    group "keys"
      [
        test "split is deterministic" test_split_is_deterministic;
        test "split subkeys are independent" test_split_independence;
        test "fold_in is distinct and reproducible"
          test_fold_in_distinct_and_reproducible;
      ];
    group "jit"
      [
        test "uniform is bit-identical under jit" test_jit_uniform_bit_parity;
        test "randint and bernoulli are bit-identical under jit"
          test_jit_int_samplers_bit_parity;
        test "normal matches eager" test_jit_normal_matches_eager;
        test "keys split inside the trace compile"
          test_jit_split_derived_key_traces;
        test "fold_in drives fresh values through a jitted step"
          test_jit_fold_in_driven_steps;
        test "implicit RNG raises" test_jit_implicit_rng_raises;
        test "a captured key raises" test_jit_captured_key_raises;
      ];
    group "transformations"
      [
        test "samples are constants of the tape"
          test_grad_dropout_mask_is_constant;
        test "vmap over per-lane keys decorrelates lanes"
          test_vmap_per_lane_keys;
        test "pmap replicates samples from a replicated key"
          test_pmap_replicated_key_smoke;
      ];
  ]

let () = run "rune rng" tests
