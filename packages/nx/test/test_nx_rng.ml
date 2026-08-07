(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx
open Windtrap

(* A key is a transparent [|2|] int32 tensor; compare keys by their words. *)
let key_words k = Nx.to_array k

let test_key_creation () =
  let key1 = Rng.key 42 in
  let key2 = Rng.key 42 in
  let key3 = Rng.key 43 in
  equal ~msg:"same seed produces same key" bool true
    (key_words key1 = key_words key2);
  equal ~msg:"different seeds produce different keys" bool true
    (key_words key1 <> key_words key3)

let test_key_splitting () =
  let key = Rng.key 42 in
  let keys = Rng.split key in
  equal ~msg:"default split produces 2 keys" int 2 (Array.length keys);

  let keys3 = Rng.split ~n:3 key in
  equal ~msg:"split with n=3 produces 3 keys" int 3 (Array.length keys3);

  (* Check keys are different *)
  equal ~msg:"split keys are different" bool true
    (key_words keys.(0) <> key_words keys.(1));

  (* Check deterministic *)
  let keys2 = Rng.split key in
  equal ~msg:"split is deterministic" bool true
    (key_words keys.(0) = key_words keys2.(0))

let test_fold_in () =
  let key = Rng.key 42 in
  let key1 = Rng.fold_in key 1 in
  let key2 = Rng.fold_in key 2 in
  let key1_again = Rng.fold_in key 1 in

  equal ~msg:"fold_in with different data produces different keys" bool true
    (key_words key1 <> key_words key2);
  equal ~msg:"fold_in is deterministic" bool true
    (key_words key1 = key_words key1_again)

let test_rand () =
  let shape = [| 3; 4 |] in
  let t = Rng.with_key (Rng.key 42) (fun () -> rand float32 shape) in

  equal ~msg:"rand produces correct shape" (array int) shape (Nx.shape t);

  (* Check values are in [0, 1) *)
  let values = Nx.to_array (Nx.reshape [| 12 |] t) in
  Array.iter
    (fun v -> equal ~msg:"rand values in [0, 1)" bool true (v >= 0. && v < 1.))
    values;

  (* Check deterministic *)
  let t2 = Rng.with_key (Rng.key 42) (fun () -> rand float32 shape) in
  let is_equal = Nx.all (Nx.equal t t2) in
  let is_equal_val = Nx.to_array is_equal in
  equal ~msg:"rand is deterministic" bool true is_equal_val.(0)

let test_randn () =
  let shape = [| 10_000 |] in
  let t = Rng.with_key (Rng.key 42) (fun () -> randn float32 shape) in

  equal ~msg:"randn produces correct shape" (array int) shape (Nx.shape t);

  (* Check roughly normal distribution (mean ~0, std ~1) *)
  let values = Nx.to_array t in
  let mean =
    Array.fold_left ( +. ) 0. values /. float_of_int (Array.length values)
  in
  let variance =
    Array.fold_left (fun acc v -> acc +. ((v -. mean) ** 2.)) 0. values
    /. float_of_int (Array.length values)
  in
  let std = Stdlib.sqrt variance in

  equal ~msg:"randn mean ~0" (float 0.05) 0. mean;
  equal ~msg:"randn std ~1" (float 0.05) 1. std

(* Box-Muller fills the draw two samples at a time, so a mistake can leave the
   second half or the odd trailing sample wrong while the first half looks
   right. Check the halves of an odd-length draw separately. *)
let test_randn_fills_the_whole_draw () =
  let n = 20_001 in
  let values = Nx.to_array (Rng.normal (Rng.key 5) float32 [| n |]) in
  equal ~msg:"randn honours an odd length" int n (Array.length values);
  equal ~msg:"randn draws are finite" bool true
    (Array.for_all Float.is_finite values);
  let moments a =
    let n = float_of_int (Array.length a) in
    let mean = Array.fold_left ( +. ) 0. a /. n in
    let variance =
      Array.fold_left (fun acc v -> acc +. ((v -. mean) ** 2.)) 0. a /. n
    in
    (mean, Stdlib.sqrt variance)
  in
  let half = n / 2 in
  let m0, s0 = moments (Array.sub values 0 half) in
  let m1, s1 = moments (Array.sub values half (n - half)) in
  equal ~msg:"first half mean ~0" (float 0.05) 0. m0;
  equal ~msg:"first half std ~1" (float 0.05) 1. s0;
  equal ~msg:"second half mean ~0" (float 0.05) 0. m1;
  equal ~msg:"second half std ~1" (float 0.05) 1. s1

(* A draw is an exact multiple of 2^-p in [0, 1 - 2^-p], where p is the
   destination's significand width. Statistics cannot see the endpoints — the
   old closed-interval bug hit 1.0 about once in 2^24 float32 draws — so check
   the reachable value set instead: at a narrow dtype the whole grid shows up in
   a few thousand draws, which pins both ends exactly. *)
let test_uniform_half_open () =
  let key = Rng.key 4242 in
  let scan (type b) name (dt : (float, b) Nx.dtype) p n =
    let values = Nx.to_array (Nx.cast float64 (Rng.uniform key dt [| n |])) in
    let step = Float.ldexp 1.0 (-p) in
    let seen = Hashtbl.create (1 lsl Stdlib.min p 12) in
    let out_of_range = ref 0 and off_grid = ref 0 in
    Array.iter
      (fun v ->
        if v < 0.0 || v >= 1.0 then incr out_of_range;
        if not (Float.is_integer (v /. step)) then incr off_grid;
        Hashtbl.replace seen v ())
      values;
    equal ~msg:(name ^ " draws lie in [0, 1)") int 0 !out_of_range;
    equal ~msg:(name ^ " draws are multiples of 2^-p") int 0 !off_grid;
    let extreme pick = Hashtbl.fold (fun v () acc -> pick v acc) seen in
    (extreme Float.max neg_infinity, extreme Float.min infinity)
  in
  (* Narrow dtypes: every grid point is drawn, so the bounds are exact. *)
  let grid (type b) name (dt : (float, b) Nx.dtype) p n =
    let hi, lo = scan name dt p n in
    equal
      ~msg:(name ^ " reaches 1 - 2^-p")
      (float 0.0)
      (1.0 -. Float.ldexp 1.0 (-p))
      hi;
    equal ~msg:(name ^ " reaches 0") (float 0.0) 0.0 lo
  in
  grid "float8_e5m2" float8_e5m2 3 2_000;
  grid "float8_e4m3" float8_e4m3 4 2_000;
  grid "bfloat16" bfloat16 8 20_000;
  grid "float16" float16 11 200_000;
  (* float32: 2^24 grid points are too many to enumerate, so bound both ends
     instead — the upper bound is exact, the lower one only rules out a
     collapsed range (missing it has probability e^-100). *)
  let hi, lo = scan "float32" float32 24 1_000_000 in
  equal ~msg:"float32 stays at or below 1 - 2^-24" bool true
    (hi <= 1.0 -. Float.ldexp 1.0 (-24));
  equal ~msg:"float32 approaches 1" bool true (hi >= 1.0 -. 1e-4);
  equal ~msg:"float32 approaches 0" bool true (lo <= 1e-4);
  (* float64 lands on the 2^-53 grid, not the 2^-24 one a widened float32 draw
     would sit on. The grid check inside [scan] is the whole claim: a draw off
     the 2^-53 grid is impossible, and one that is a multiple of 2^-24 as well
     would betray a float32 draw in disguise — so also count how many draws are
     multiples of 2^-24, which should be about n * 2^-29, i.e. none. *)
  let hi, lo = scan "float64" float64 53 200_000 in
  equal ~msg:"float64 stays at or below 1 - 2^-53" bool true
    (hi <= 1.0 -. Float.ldexp 1.0 (-53));
  equal ~msg:"float64 approaches 1" bool true (hi >= 1.0 -. 1e-4);
  equal ~msg:"float64 approaches 0" bool true (lo <= 1e-4);
  let coarse = Float.ldexp 1.0 (-24) in
  let on_float32_grid =
    Nx.to_array (Rng.uniform (Rng.key 4242) float64 [| 200_000 |])
    |> Array.to_list
    |> List.filter (fun v -> Float.is_integer (v /. coarse))
    |> List.length
  in
  equal ~msg:"float64 draws do not sit on the float32 grid" int 0
    on_float32_grid

let test_keyless_float_sampler_dtypes () =
  let check (type b) name (dt : (float, b) Nx.dtype) =
    let uniform_sample : (float, b) Nx.t = rand dt [| 2 |] in
    let normal_sample : (float, b) Nx.t = randn dt [| 2 |] in
    equal ~msg:(name ^ " rand dtype") bool true (Nx.dtype uniform_sample = dt);
    equal ~msg:(name ^ " randn dtype") bool true (Nx.dtype normal_sample = dt)
  in
  Rng.with_key (Rng.key 42) (fun () ->
      check "float32" float32;
      check "float64" float64)

let test_randint () =
  let shape = [| 10 |] in
  let t = Rng.with_key (Rng.key 42) (fun () -> randint ~low:5 ~high:15 shape) in

  equal ~msg:"randint produces correct shape" (array int) shape (Nx.shape t);

  (* Check values are in [min, max) *)
  let values = Nx.to_array t in
  Array.iter
    (fun v ->
      let v = Int32.to_int v in
      equal ~msg:"randint values in [5, 15)" bool true (v >= 5 && v < 15))
    values

(* [low, high) must be half-open and flat all the way to its ends, including a
   negative [low]: rounding the affine map towards zero used to drop [low]
   entirely and give 0 twice its share. *)
let test_randint_covers_range_uniformly () =
  let low = -5 and high = 5 in
  let n = 200_000 in
  let values = Nx.to_array (Rng.randint (Rng.key 91) ~low ~high [| n |]) in
  let span = high - low in
  let counts = Array.make span 0 in
  let out_of_range = ref 0 in
  Array.iter
    (fun v ->
      let v = Int32.to_int v in
      if v < low || v >= high then incr out_of_range
      else counts.(v - low) <- counts.(v - low) + 1)
    values;
  equal ~msg:"randint draws lie in [low, high)" int 0 !out_of_range;
  let expected = float_of_int n /. float_of_int span in
  let tol =
    5.0 *. Stdlib.sqrt (expected *. (1.0 -. (1.0 /. float_of_int span)))
  in
  Array.iteri
    (fun i c ->
      equal
        ~msg:(Printf.sprintf "randint value %d is drawn its share" (low + i))
        (float tol) expected (float_of_int c))
    counts

(* [p = 1] must accept every draw and [p = 0] reject every one: the comparison
   is [u < p], so these hold exactly when [u] lives in [0, 1). *)
let test_bernoulli_extremes () =
  let n = 100_000 in
  let count p =
    let t = Nx.cast uint8 (Rng.bernoulli (Rng.key 17) ~p [| n |]) in
    Array.fold_left
      (fun acc v -> acc + if v > 0 then 1 else 0)
      0 (Nx.to_array t)
  in
  equal ~msg:"bernoulli p=1 always accepts" int n (count 1.0);
  equal ~msg:"bernoulli p=0 never accepts" int 0 (count 0.0)

let test_bernoulli () =
  let shape = [| 1000 |] in
  let p = 0.3 in
  let t = Rng.with_key (Rng.key 42) (fun () -> bernoulli ~p shape) in

  equal ~msg:"bernoulli produces correct shape" (array int) shape (Nx.shape t);
  let t_int = cast uint8 t in
  (* Check proportion roughly matches p *)
  let values = Nx.to_array t_int in
  let ones =
    Array.fold_left (fun acc v -> acc + if v > 0 then 1 else 0) 0 values
  in
  let prop = float_of_int ones /. float_of_int (Array.length values) in
  equal ~msg:"bernoulli proportion ~p" (float 0.05) p prop

let test_shuffle_preserves_shape () =
  let shape = [| 6; 4 |] in
  let data =
    Array.init (shape.(0) * shape.(1)) (fun i -> float_of_int (i + 1))
  in
  let x = Nx.create float32 shape data in
  let shuffled = Rng.with_key (Rng.key 7) (fun () -> shuffle x) in

  equal ~msg:"shuffle preserves leading axis" (array int) shape
    (Nx.shape shuffled);

  let flatten t =
    let dims = Nx.shape t in
    let total = Array.fold_left ( * ) 1 dims in
    let reshaped = Nx.reshape [| total |] t in
    Nx.to_array reshaped
  in
  let orig_flat = flatten x in
  let shuffled_flat = flatten shuffled in

  let sorted_orig = Array.copy orig_flat in
  let sorted_shuffled = Array.copy shuffled_flat in
  Array.sort compare sorted_orig;
  Array.sort compare sorted_shuffled;
  equal ~msg:"shuffle preserves multiset"
    (array (float 0.0))
    sorted_orig sorted_shuffled;

  let shuffled_again = Rng.with_key (Rng.key 7) (fun () -> shuffle x) in
  let equality = Nx.equal shuffled shuffled_again |> Nx.all |> Nx.to_array in
  equal ~msg:"shuffle deterministic with same seed" bool true equality.(0)

let test_truncated_normal () =
  let shape = [| 100 |] in
  let lower = -1.5 in
  let upper = 2.0 in
  let t =
    Rng.with_key (Rng.key 42) (fun () -> truncated_normal float32 ~lower ~upper shape)
  in

  equal ~msg:"truncated_normal produces correct shape" (array int) shape
    (Nx.shape t);

  (* Check all values are within bounds *)
  let values = Nx.to_array t in
  Array.iter
    (fun v ->
      equal
        ~msg:
          (Printf.sprintf "truncated_normal values in [%.1f, %.1f]: %.3f" lower
             upper v)
        bool true
        (v >= lower && v <= upper))
    values

let test_truncated_normal_distribution () =
  let shape = [| 20_000 |] in
  let lower = -0.75 in
  let upper = 1.25 in
  let samples =
    Rng.with_key (Rng.key 123) (fun () -> truncated_normal float32 ~lower ~upper shape)
  in

  equal ~msg:"truncated_normal produces correct shape" (array int) shape
    (Nx.shape samples);

  let values = Nx.to_array samples in
  let total = Array.length values in
  let boundary_hits =
    Array.fold_left
      (fun acc v ->
        if Float.abs (v -. lower) < 1e-6 || Float.abs (v -. upper) < 1e-6 then
          acc + 1
        else acc)
      0 values
  in

  equal
    ~msg:
      (Printf.sprintf
         "truncated normal rarely clips to bounds (%d / %d clipped)"
         boundary_hits total)
    bool true
    (boundary_hits < total / 1000);

  let mean = Array.fold_left ( +. ) 0. values /. float_of_int total in
  equal ~msg:"truncated normal mean lies within interval" bool true
    (mean > lower && mean < upper)

(* The inverse-CDF draw must reproduce the conditional distribution exactly, not
   merely land inside the bounds. Both moments of a standard normal restricted
   to [a, b] are closed-form, so check against them:

   mean = (phi a - phi b) / Z var = 1 + (a * phi a - b * phi b) / Z - mean^2
   with Z = Phi b - Phi a

   A rejection sampler and an inverse-CDF sampler agree here; a botched inverse
   — wrong scale, wrong interval, a sign slip in erfinv — does not. *)
let test_truncated_normal_matches_the_conditional_moments () =
  let lower = -0.75 and upper = 1.25 in
  let n = 200_000 in
  let phi x = Float.exp (-0.5 *. x *. x) /. Stdlib.sqrt (2.0 *. Float.pi) in
  let cdf x = 0.5 *. (1.0 +. Float.erf (x /. Stdlib.sqrt 2.0)) in
  let z = cdf upper -. cdf lower in
  let expected_mean = (phi lower -. phi upper) /. z in
  let expected_var =
    1.0
    +. (((lower *. phi lower) -. (upper *. phi upper)) /. z)
    -. (expected_mean *. expected_mean)
  in
  let values =
    Nx.to_array
      (Rng.truncated_normal (Rng.key 4242) ~lower ~upper float64 [| n |])
  in
  let mean = Array.fold_left ( +. ) 0.0 values /. float_of_int n in
  let var =
    Array.fold_left (fun acc v -> acc +. ((v -. mean) ** 2.0)) 0.0 values
    /. float_of_int n
  in
  (* 5 standard errors of the sample mean, and a matching band for the
     variance. *)
  let se = Stdlib.sqrt (expected_var /. float_of_int n) in
  equal ~msg:"truncated normal mean matches the closed form"
    (float (5.0 *. se))
    expected_mean mean;
  equal ~msg:"truncated normal variance matches the closed form" (float 0.01)
    expected_var var;
  equal ~msg:"every draw lies inside the bounds" bool true
    (Array.for_all (fun v -> v >= lower && v <= upper) values)

(* Narrow bounds are where rejection sampling used to spend its budget and
   eventually give up; inverting costs one draw whatever the acceptance rate. *)
let test_truncated_normal_handles_narrow_bounds () =
  let lower = 3.0 and upper = 3.01 in
  let values =
    Nx.to_array
      (Rng.truncated_normal (Rng.key 7) ~lower ~upper float64 [| 512 |])
  in
  equal ~msg:"a 0.06%-mass interval still fills" bool true
    (Array.for_all (fun v -> v >= lower && v <= upper) values);
  equal ~msg:"the draws are not all identical" bool true
    (Array.exists (fun v -> v <> values.(0)) values)

(* Every keyed sampler is a pure function of its key: same key, same values;
   different keys, different values. *)
let test_keyed_samplers_are_pure () =
  let check name draw =
    equal
      ~msg:(name ^ " is deterministic")
      bool true
      (draw (Rng.key 3) = draw (Rng.key 3));
    equal
      ~msg:(name ^ " follows its key")
      bool true
      (draw (Rng.key 3) <> draw (Rng.key 4))
  in
  let logits = Nx.create float32 [| 6 |] [| 0.; 1.; 2.; 0.5; 1.5; 0.2 |] in
  check "truncated_normal" (fun k ->
      Nx.to_array
        (Rng.truncated_normal k ~lower:(-2.) ~upper:2. float32 [| 32 |]));
  check "permutation" (fun k -> Nx.to_array (Rng.permutation k 64));
  check "shuffle" (fun k ->
      Nx.to_array (Rng.shuffle k (Nx.arange float32 0 64 1)));
  check "categorical" (fun k ->
      Nx.to_array (Rng.categorical k ~shape:[| 64 |] logits))

(* [permutation] must be a permutation, not merely a tensor of indices. *)
let test_permutation_is_a_permutation () =
  let n = 4096 in
  let p =
    Array.map Int32.to_int (Nx.to_array (Rng.permutation (Rng.key 11) n))
  in
  let seen = Array.make n false in
  Array.iter (fun i -> if i >= 0 && i < n then seen.(i) <- true) p;
  equal ~msg:"permutation has the right length" int n (Array.length p);
  equal ~msg:"permutation hits every index exactly once" bool true
    (Array.for_all Fun.id seen)

(* The sort keys are assembled from two Threefry words by masking and shifting,
   which is easy to get subtly wrong — a bad mask biases the high word, and
   every draw then sorts nearly by its low word alone. Check the ordering is
   uniform: over many keys each element must reach each position about equally
   often. A skewed key construction shows up here as a heavy diagonal.

   (This does not test the tie bias the 64-bit keys were introduced for. That
   one is not observable at any feasible sample size: it shifts P(i before j)
   by 2^-24, which needs ~1e16 trials to see. The argument for it is the
   collision count, in the comment on [permutation].) *)
let test_permutation_positions_are_uniform () =
  let n = 8 and trials = 20_000 in
  let counts = Array.make_matrix n n 0 in
  for t = 0 to trials - 1 do
    let p = Nx.to_array (Rng.permutation (Rng.key t) n) in
    Array.iteri (fun pos v ->
        let v = Int32.to_int v in
        counts.(pos).(v) <- counts.(pos).(v) + 1)
      p
  done;
  let expected = float_of_int trials /. float_of_int n in
  (* Binomial standard deviation, times five. *)
  let tol =
    5.0
    *. Stdlib.sqrt
         (expected *. (1.0 -. (1.0 /. float_of_int n)))
  in
  Array.iteri
    (fun pos row ->
      Array.iteri
        (fun v c ->
          equal
            ~msg:(Printf.sprintf "element %d reaches position %d evenly" v pos)
            (float tol) expected (float_of_int c))
        row)
    counts

(* Both are inverse-CDF draws whose closed-form moments pin them exactly.
   Gumbel(0,1): mean = Euler-Mascheroni, variance = pi^2/6. Exponential(1):
   mean = variance = 1. A sign slip or a missing negation moves both. *)
let test_gumbel_and_exponential_moments () =
  let n = 200_000 in
  let moments a =
    let len = float_of_int (Array.length a) in
    let mean = Array.fold_left ( +. ) 0.0 a /. len in
    let var =
      Array.fold_left (fun acc v -> acc +. ((v -. mean) ** 2.0)) 0.0 a /. len
    in
    (mean, var)
  in
  let g = Nx.to_array (Rng.gumbel (Rng.key 1) float64 [| n |]) in
  let gm, gv = moments g in
  equal ~msg:"gumbel mean is Euler-Mascheroni" (float 0.02) 0.5772156649 gm;
  equal ~msg:"gumbel variance is pi^2/6" (float 0.05)
    (Float.pi *. Float.pi /. 6.0)
    gv;
  let e = Nx.to_array (Rng.exponential (Rng.key 2) float64 [| n |]) in
  let em, ev = moments e in
  equal ~msg:"exponential mean is 1" (float 0.02) 1.0 em;
  equal ~msg:"exponential variance is 1" (float 0.05) 1.0 ev;
  equal ~msg:"exponential draws are non-negative" bool true
    (Array.for_all (fun v -> v >= 0.0) e);
  (* Both poles are the real risk: a draw of exactly 0 or of the largest
     representable uniform must stay finite. *)
  equal ~msg:"gumbel draws are finite" bool true
    (Array.for_all Float.is_finite g);
  equal ~msg:"exponential draws are finite" bool true
    (Array.for_all Float.is_finite e)

(* Gamma(a, 1) has mean a and variance a, and — the part that catches a broken
   acceptance test rather than a broken constant — skewness 2/sqrt a. A sampler
   that accepted everything would still land the mean roughly right while
   getting the tail badly wrong, so check the third moment too. Both sides of
   the concentration = 1 boundary are exercised, since below it the draw goes
   through the Gamma(a) = Gamma(a+1) * U^(1/a) shift. *)
let test_gamma_moments () =
  let n = 200_000 in
  let check concentration =
    let v =
      Nx.to_array (Rng.gamma (Rng.key 31) ~concentration float64 [| n |])
    in
    let len = float_of_int n in
    let mean = Array.fold_left ( +. ) 0.0 v /. len in
    let central p =
      Array.fold_left (fun acc x -> acc +. ((x -. mean) ** p)) 0.0 v /. len
    in
    let var = central 2.0 in
    let skew = central 3.0 /. (var ** 1.5) in
    let label = Printf.sprintf "gamma(%g)" concentration in
    equal ~msg:(label ^ " mean") (float (0.03 *. concentration)) concentration
      mean;
    equal ~msg:(label ^ " variance") (float (0.06 *. concentration))
      concentration var;
    equal ~msg:(label ^ " skewness")
      (float 0.12)
      (2.0 /. Stdlib.sqrt concentration)
      skew;
    equal ~msg:(label ^ " draws are positive") bool true
      (Array.for_all (fun x -> x > 0.0) v)
  in
  check 0.4;
  check 1.0;
  check 3.7;
  check 20.0

let invalid_arg_raised ~msg f =
  raises_match ~msg (function Invalid_argument _ -> true | _ -> false) f

let test_gamma_validates_concentration () =
  invalid_arg_raised ~msg:"zero concentration" (fun () ->
      ignore (Rng.gamma (Rng.key 0) ~concentration:0.0 float32 [| 4 |]));
  invalid_arg_raised ~msg:"negative concentration" (fun () ->
      ignore (Rng.gamma (Rng.key 0) ~concentration:(-1.0) float32 [| 4 |]))

(* Poisson is exact here, so it can be held to its whole distribution rather
   than a couple of moments: compare the empirical frequency of each count
   against the analytic pmf. Mean and variance both equal the rate, which a
   truncated round count would break at the tail. *)
let test_poisson_matches_the_pmf () =
  let n = 200_000 in
  let check rate =
    let v =
      Array.map Int32.to_int
        (Nx.to_array (Rng.poisson (Rng.key 77) ~rate [| n |]))
    in
    let len = float_of_int n in
    let mean = Array.fold_left (fun a x -> a +. float_of_int x) 0.0 v /. len in
    let var =
      Array.fold_left
        (fun a x -> a +. ((float_of_int x -. mean) ** 2.0))
        0.0 v
      /. len
    in
    let label = Printf.sprintf "poisson(%g)" rate in
    equal ~msg:(label ^ " mean") (float (0.03 *. Stdlib.sqrt rate)) rate mean;
    equal ~msg:(label ^ " variance") (float (0.1 *. rate)) rate var;
    equal ~msg:(label ^ " counts are non-negative") bool true
      (Array.for_all (fun x -> x >= 0) v);
    (* Frequencies against the pmf, over the counts carrying real mass. *)
    let top = int_of_float (Float.ceil (rate +. (4.0 *. Stdlib.sqrt rate))) in
    let counts = Array.make (top + 1) 0 in
    Array.iter (fun x -> if x <= top then counts.(x) <- counts.(x) + 1) v;
    let pmf = ref (Float.exp (-.rate)) in
    for c = 0 to top do
      let expected = !pmf *. len in
      if expected > 50.0 then
        equal
          ~msg:(Printf.sprintf "%s frequency of %d" label c)
          (float (5.0 *. Stdlib.sqrt expected))
          expected
          (float_of_int counts.(c));
      pmf := !pmf *. rate /. float_of_int (c + 1)
    done
  in
  check 0.7;
  check 4.0;
  check 30.0

let test_poisson_validates_rate () =
  invalid_arg_raised ~msg:"zero rate" (fun () ->
      ignore (Rng.poisson (Rng.key 0) ~rate:0.0 [| 4 |]));
  invalid_arg_raised ~msg:"rate beyond the ceiling" (fun () ->
      ignore (Rng.poisson (Rng.key 0) ~rate:250.0 [| 4 |]))

let test_categorical () =
  (* Test with simple 1D logits: [0.0, 1.0, 2.0] *)
  (* Expected probabilities after softmax: [0.090, 0.245, 0.665] approximately *)
  let logits = Nx.create float32 [| 3 |] [| 0.0; 1.0; 2.0 |] in
  let samples = Rng.with_key (Rng.key 42) (fun () -> categorical logits) in

  (* Check output shape *)
  let output_shape = Nx.shape samples in
  equal ~msg:"categorical produces correct shape" (array int) [||] output_shape;

  (* Check that output is a scalar int32 *)
  let sample_val = Nx.to_array samples in
  equal ~msg:"categorical produces single value" int 1 (Array.length sample_val);

  (* Check value is in valid range [0, 2] *)
  let sample_idx = Int32.to_int sample_val.(0) in
  equal ~msg:"categorical value in valid range" bool true
    (sample_idx >= 0 && sample_idx <= 2);

  (* Test determinism *)
  let samples2 = Rng.with_key (Rng.key 42) (fun () -> categorical logits) in
  let is_equal = Nx.all (Nx.equal samples samples2) in
  let is_equal_val = Nx.to_array is_equal in
  equal ~msg:"categorical is deterministic" bool true is_equal_val.(0);

  (* Test with Float64 *)
  let logits64 = Nx.create float64 [| 3 |] [| 0.0; 1.0; 2.0 |] in
  let samples64 = Rng.with_key (Rng.key 42) (fun () -> categorical logits64) in
  let is_equal64 = Nx.all (Nx.equal samples samples64) in
  let is_equal_val64 = Nx.to_array is_equal64 in
  equal ~msg:"categorical is type agnostic" bool true is_equal_val64.(0)

let test_categorical_2d () =
  (* Test with 2D logits: [[0.0, 1.0], [2.0, 0.0]] *)
  (* Expected probabilities after softmax: [[0.269, 0.731], [0.881, 0.119]] approximately *)
  let logits = Nx.create float32 [| 2; 2 |] [| 0.0; 1.0; 2.0; 0.0 |] in
  let samples = Rng.with_key (Rng.key 42) (fun () -> categorical logits) in

  (* Check output shape (should be [2] - one sample per row) *)
  let output_shape = Nx.shape samples in
  equal ~msg:"categorical 2D produces correct shape" (array int) [| 2 |]
    output_shape;

  (* Check values are in valid range [0, 1] for each row *)
  let sample_vals = Nx.to_array samples in
  equal ~msg:"categorical 2D produces 2 values" int 2 (Array.length sample_vals);

  Array.iter
    (fun v ->
      let idx = Int32.to_int v in
      equal ~msg:"categorical 2D value in valid range" bool true
        (idx >= 0 && idx <= 1))
    sample_vals

let test_categorical_axis_handling () =
  (* 2D logits: shape [2; 3] Row 0 -> [0.0, 1.0, 2.0] Row 1 -> [2.0, 0.5, -1.0]
     This ensures all probabilities differ. *)
  let logits =
    Nx.create float32 [| 2; 3 |] [| 0.0; 1.0; 2.0; 2.0; 0.5; -1.0 |]
  in

  (* axis=1 -> sample across columns for each row -> shape [2] *)
  let samples_axis_1 =
    Rng.with_key (Rng.key 42) (fun () -> categorical ~axis:1 logits)
  in

  (* axis=-1 -> equivalent to axis=1 -> shape [2] *)
  let samples_axis_neg_1 =
    Rng.with_key (Rng.key 42) (fun () -> categorical ~axis:(-1) logits)
  in

  (* axis=0 -> sample across rows for each column -> shape [3] *)
  let samples_axis_0 =
    Rng.with_key (Rng.key 42) (fun () -> categorical ~axis:0 logits)
  in

  (* Check shape for axis=1 *)
  let shape_axis_1 = Nx.shape samples_axis_1 in
  equal ~msg:"categorical axis=1 produces correct shape" (array int) [| 2 |]
    shape_axis_1;

  (* Check shape for axis=-1 (should match axis=1) *)
  let shape_axis_neg_1 = Nx.shape samples_axis_neg_1 in
  equal ~msg:"categorical axis=-1 matches axis=1 shape" (array int) [| 2 |]
    shape_axis_neg_1;

  (* Check shape for axis=0 *)
  let shape_axis_0 = Nx.shape samples_axis_0 in
  equal ~msg:"categorical axis=0 produces correct shape" (array int) [| 3 |]
    shape_axis_0;

  (* Check that axis=1 and axis=-1 give identical results *)
  let is_equal = Nx.all (Nx.equal samples_axis_1 samples_axis_neg_1) in
  let is_equal_val = Nx.to_array is_equal in
  equal ~msg:"categorical axis=-1 behaves like axis=1" bool true
    is_equal_val.(0);

  (* Sanity check: ensure sampled indices are in valid range *)
  let vals_axis_0 = Nx.to_array samples_axis_0 in
  Array.iter
    (fun i ->
      equal ~msg:"axis=0 value in valid range" bool true
        (Int32.to_int i >= 0 && Int32.to_int i < 2))
    vals_axis_0;

  let vals_axis_1 = Nx.to_array samples_axis_1 in
  Array.iter
    (fun i ->
      equal ~msg:"axis=1 value in valid range" bool true
        (Int32.to_int i >= 0 && Int32.to_int i < 3))
    vals_axis_1

let test_categorical_shape_prefix_axis () =
  let logits =
    Nx.create float64 [| 2; 3; 4 |]
      [|
        0.0;
        0.5;
        1.0;
        1.5;
        2.0;
        2.5;
        3.0;
        -0.5;
        0.25;
        1.25;
        -1.0;
        0.75;
        -0.25;
        0.4;
        1.8;
        -1.5;
        0.2;
        1.1;
        0.3;
        -0.8;
        0.6;
        1.4;
        -0.2;
        0.9;
      |]
  in

  let prefix_shape = [| 5; 6 |] in
  let samples =
    Rng.with_key (Rng.key 314) (fun () ->
        categorical ~shape:prefix_shape ~axis:(-2) logits)
  in

  let expected_shape = [| 5; 6; 2; 4 |] in
  equal ~msg:"categorical shape prefix keeps axis semantics" (array int)
    expected_shape (Nx.shape samples);

  let values = Nx.to_array samples |> Array.map Int32.to_int in
  Array.iter
    (fun v ->
      equal ~msg:"categorical indices within axis range" bool true
        (v >= 0 && v < 3))
    values

let test_categorical_distribution () =
  let logits = Nx.create float32 [| 3 |] [| 0.0; 1.0; 2.0 |] in

  let n_samples = 20000 in
  let inds =
    Rng.with_key (Rng.key 123) (fun () -> categorical ~shape:[| n_samples |] logits)
  in

  equal ~msg:"categorical produces correct shape" (array int) [| n_samples |]
    (Nx.shape inds);

  let values = Nx.to_array inds |> Array.map Int32.to_int in

  (* Histogram counts *)
  let n_classes = 3 in
  let counts = Array.make n_classes 0 in
  Array.iter (fun v -> counts.(v) <- counts.(v) + 1) values;

  (* Compute softmax probabilities from logits_arr *)
  let logits_arr = [| 0.0; 1.0; 2.0 |] in
  let max_logit =
    Array.fold_left
      (fun acc x -> if x > acc then x else acc)
      neg_infinity logits_arr
  in
  let exps = Array.map (fun x -> Stdlib.exp (x -. max_logit)) logits_arr in
  let sum_exps = Array.fold_left ( +. ) 0. exps in
  let probs = Array.map (fun e -> e /. sum_exps) exps in

  (* Check each bucket is within a reasonable statistical tolerance *)
  Array.iteri
    (fun i p ->
      let prop = float_of_int counts.(i) /. float_of_int n_samples in
      let se = Stdlib.sqrt (p *. (1. -. p) /. float_of_int n_samples) in
      let tol = Stdlib.max (4. *. se) 0.01 in
      equal
        ~msg:(Printf.sprintf "categorical bucket %d ~ p" i)
        (float tol) p prop)
    probs

let () =
  run "Nx.Rng"
    [
      group "key"
        [
          test "creation" test_key_creation;
          test "splitting" test_key_splitting;
          test "fold_in" test_fold_in;
        ];
      group "sampling"
        [
          test "rand" test_rand;
          test "randn" test_randn;
          test "randn_fills_the_whole_draw" test_randn_fills_the_whole_draw;
          test "keyless float sampler dtypes" test_keyless_float_sampler_dtypes;
          test "randint" test_randint;
          test "uniform_half_open" test_uniform_half_open;
          test "randint_covers_range_uniformly"
            test_randint_covers_range_uniformly;
          test "bernoulli" test_bernoulli;
          test "bernoulli_extremes" test_bernoulli_extremes;
          test "shuffle_preserves_shape" test_shuffle_preserves_shape;
          test "truncated_normal" test_truncated_normal;
          test "truncated_normal_distribution"
            test_truncated_normal_distribution;
          test "truncated_normal matches the conditional moments"
            test_truncated_normal_matches_the_conditional_moments;
          test "truncated_normal handles narrow bounds"
            test_truncated_normal_handles_narrow_bounds;
          test "keyed samplers are pure" test_keyed_samplers_are_pure;
          test "permutation is a permutation" test_permutation_is_a_permutation;
          test "permutation positions are uniform"
            test_permutation_positions_are_uniform;
          test "gumbel and exponential moments"
            test_gumbel_and_exponential_moments;
          test "gamma moments" test_gamma_moments;
          test "gamma validates concentration"
            test_gamma_validates_concentration;
          test "poisson matches the pmf" test_poisson_matches_the_pmf;
          test "poisson validates rate" test_poisson_validates_rate;
          test "categorical" test_categorical;
          test "categorical_2d" test_categorical_2d;
          test "categorical_axis_handling" test_categorical_axis_handling;
          test "categorical_shape_prefix_axis"
            test_categorical_shape_prefix_axis;
          test "categorical_distribution" test_categorical_distribution;
        ];
    ]
