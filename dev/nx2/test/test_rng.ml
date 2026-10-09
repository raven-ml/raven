(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.Rng: Threefry's known answers, the laws between keys and draws, and each
   sampler's distribution within stated bounds. *)

open Windtrap
module A = Nx_array
module Rng = Nx.Rng

let m = Nx_support.memory

module S2 = (val Nx.devices [ m 0; m 1 ])

let message f =
  match f () with _ -> "no exception" | exception Invalid_argument m -> m

(* [x]'s elements in row-major order, computed on the host. *)
let read x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))

(* [x]'s elements on the first device of its set. *)
let shard x = A.to_array (Option.get (Nx.Repr.shards x)).(0)
let host_of dt s xs = Nx.Repr.of_array Nx.Host.v (A.of_array dt s xs)
let words k = read (Rng.to_tensor k)
let word = Int32.of_int
let numel s = Array.fold_left ( * ) 1 s

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let shapes =
  Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 5)
  |> Gen.with_pp pp_shape

let seeds = Gen.int

(* Keys *)

(* Threefry-2x32-20, Random123's known answers: [fold_in k d] encrypts the
   counter [(d asr 32, d)] under [k]'s words [(seed asr 32, seed)]. *)
let known =
  cases
    ~name:(fun (s, d, _) -> Printf.sprintf "seed %x data %x" s d)
    "known answers"
    [
      (0, 0, [| 0x6b200159; 0x99ba4efe |]);
      (-1, -1, [| 0x1cb996fc; 0xbb002be7 |]);
      ( (0x13198a2e lsl 32) lor 0x03707344,
        (0x243f6a88 lsl 32) lor 0x85a308d3,
        [| 0xc4923a9c; 0x483df7a0 |] );
    ]
    (fun (seed, data, want) ->
      equal (array int32) (Array.map word want)
        (words (Rng.fold_in (Rng.key seed) data)))

let keys =
  group "keys"
    [
      prop "a seed's words are its high and low halves" seeds (fun s ->
          equal (array int32) [| word (s asr 32); word s |] (words (Rng.key s)));
      test "a key is a value of every set" (fun () ->
          is_none (Nx.placement (Rng.to_tensor (Rng.key 7))));
      prop "split's keys are split_batch's rows" (Gen.int_range 1 6) (fun n ->
          let k = Rng.key 3 in
          let rows = words (Rng.split_batch ~n k) in
          let each =
            Array.concat (Array.to_list (Array.map words (Rng.split ~n k)))
          in
          equal (array int32) rows each);
      test "split keys are distinct and differ from their key" (fun () ->
          let k = Rng.key 11 in
          let ks = Array.map words (Rng.split ~n:8 k) in
          let all = Array.to_list (Array.append [| words k |] ks) in
          equal int (List.length all) (List.length (List.sort_uniq compare all)));
      prop "fold_in_tensor of [n] is fold_in of n"
        (Gen.int32_range Int32.min_int Int32.max_int) (fun n ->
          let k = Rng.key 5 in
          let i =
            Nx.Repr.of_array Nx.Host.v (A.of_array A.Dtype.Int32 [||] [| n |])
          in
          equal (array int32)
            (words (Rng.fold_in k (Int32.to_int n)))
            (words (Rng.fold_in_tensor k i)));
      test "fold_in_tensor of a batch of indices is a batch of keys" (fun () ->
          let k = Rng.key 5 in
          let i =
            Nx.Repr.of_array Nx.Host.v
              (A.of_array A.Dtype.Int32 [| 3 |] [| 0l; 1l; -4l |])
          in
          let each =
            Array.concat
              (List.map (fun n -> words (Rng.fold_in k n)) [ 0; 1; -4 ])
          in
          equal (array int32) each (words (Rng.fold_in_tensor k i)));
      prop "of_tensor undoes to_tensor" seeds (fun s ->
          let k = Rng.split_batch ~n:3 (Rng.key s) in
          equal (array int32) (words k)
            (words (Rng.of_tensor (Rng.to_tensor k))));
      test "of_tensor refuses a last axis other than 2" (fun () ->
          let t =
            Nx.Repr.of_array Nx.Host.v
              (A.of_array A.Dtype.Int32 [| 3 |] [| 0l; 1l; 2l |])
          in
          equal string "Nx.Rng.of_tensor: a key has shape [...; 2], not [3]"
            (message (fun () -> Rng.of_tensor t)));
      test "split refuses n below 1" (fun () ->
          equal string "Nx.Rng.split: n is 0, not at least 1"
            (message (fun () -> Rng.split ~n:0 (Rng.key 1))));
      test "split refuses a batch" (fun () ->
          let b = Rng.split_batch ~n:2 (Rng.key 1) in
          equal string
            "Nx.Rng.split: a batch of keys of shape [2; 2]; draw from one key"
            (message (fun () -> Rng.split b)));
    ]

(* Draws *)

let draws =
  group "draws"
    [
      prop "bits are split_batch's words in order" (Gen.int_range 0 9) (fun n ->
          let k = Rng.key 21 in
          cover "odd" (n mod 2 = 1);
          cover "none" (n = 0);
          let blocks = words (Rng.split_batch ~n:(((n + 1) / 2) + 1) k) in
          equal (array int32) (Array.sub blocks 0 n)
            (read (Rng.bits ~key:k [| n |])));
      prop "a draw is its flat draw reshaped" shapes (fun s ->
          let k = Rng.key 2 in
          equal (array int32)
            (read (Rng.bits ~key:k [| numel s |]))
            (read (Rng.bits ~key:k s)));
      prop "a draw is the same on every set" shapes (fun s ->
          let k = Rng.key 9 in
          let x = Rng.normal ~key:k Nx.float32 s in
          equal (array float_exact) (read x) (shard (Nx.place S2.on x)));
      test "a placed key draws where it lies" (fun () ->
          let k = Rng.place S2.on (Rng.key 9) in
          let x = Rng.uniform ~key:k Nx.float32 [| 5 |] in
          equal bool true
            (Nx.Placement.equal S2.on (Option.get (Nx.placement x)));
          equal (array float_exact)
            (read (Rng.uniform ~key:(Rng.key 9) Nx.float32 [| 5 |]))
            (shard x));
      prop "a batch of keys draws each key's draw"
        (Gen.pair (Gen.int_range 1 4) shapes)
        (fun (n, s) ->
          let k = Rng.key 4 in
          let batch =
            read (Rng.uniform ~key:(Rng.split_batch ~n k) Nx.float64 s)
          in
          let each =
            Array.concat
              (Array.to_list
                 (Array.map
                    (fun k -> read (Rng.uniform ~key:k Nx.float64 s))
                    (Rng.split ~n k)))
          in
          equal (array float_exact) each batch);
      test "float32 uniform is the low 24 bits of bits, scaled" (fun () ->
          let k = Rng.key 13 in
          let b = read (Rng.bits ~key:k [| 64 |]) in
          let want =
            Array.map
              (fun w ->
                Float.ldexp (Int32.to_float (Int32.logand w 0xFF_FFFFl)) (-24))
              b
          in
          equal (array float_exact) want
            (read (Rng.uniform ~key:k Nx.float32 [| 64 |])));
      test "a negative extent raises" (fun () ->
          equal string "Nx.Rng.uniform: shape [2; -1] has a negative extent"
            (message (fun () -> Rng.uniform Nx.float32 [| 2; -1 |])));
    ]

(* Distributions. Each moment is checked within 6 standard errors of its value,
   at fixed keys. *)

let n = 1 lsl 16
let mean xs = Array.fold_left ( +. ) 0. xs /. Float.of_int (Array.length xs)

let variance xs =
  let m = mean xs in
  Array.fold_left (fun a x -> a +. ((x -. m) *. (x -. m))) 0. xs
  /. Float.of_int (Array.length xs)

(* [x] within [k] standard errors [se] of [want]. *)
let near ~se want x =
  at_most float_exact ~than:(6. *. se) (Float.abs (x -. want))

let unit_grid name dt p =
  test name (fun () ->
      let xs = read (Rng.uniform ~key:(Rng.key 1) dt [| 4096 |]) in
      Array.iter
        (fun x ->
          if
            not
              (x >= 0. && x < 1.
              && Float.equal (Float.ldexp x p) (Float.round (Float.ldexp x p)))
          then failf "%h is not a multiple of 2^-%d in [0, 1)" x p)
        xs)

let distributions =
  group "distributions"
    [
      unit_grid "float32 uniform draws are multiples of 2^-24 in [0, 1)"
        Nx.float32 24;
      unit_grid "float64 uniform draws are multiples of 2^-53 in [0, 1)"
        Nx.float64 53;
      unit_grid "bfloat16 uniform draws are multiples of 2^-8 in [0, 1)"
        Nx.bfloat16 8;
      test "uniform: mean 1/2, variance 1/12" (fun () ->
          let xs = read (Rng.uniform ~key:(Rng.key 100) Nx.float64 [| n |]) in
          near ~se:(sqrt (1. /. 12. /. Float.of_int n)) 0.5 (mean xs);
          near
            ~se:(sqrt (1. /. 180. /. Float.of_int n))
            (1. /. 12.) (variance xs));
      test "normal: mean 0, variance 1" (fun () ->
          let xs = read (Rng.normal ~key:(Rng.key 101) Nx.float32 [| n |]) in
          near ~se:(sqrt (1. /. Float.of_int n)) 0. (mean xs);
          near ~se:(sqrt (2. /. Float.of_int n)) 1. (variance xs));
      test "normal at float64: mean 0, variance 1" (fun () ->
          let xs =
            read (Rng.normal ~key:(Rng.key 102) Nx.float64 [| n + 1 |])
          in
          near ~se:(sqrt (1. /. Float.of_int n)) 0. (mean xs);
          near ~se:(sqrt (2. /. Float.of_int n)) 1. (variance xs));
      test "exponential: mean 1, variance 1, never negative" (fun () ->
          let xs =
            read (Rng.exponential ~key:(Rng.key 103) Nx.float32 [| n |])
          in
          Array.iter
            (fun x ->
              if not (x >= 0. && Float.is_finite x) then failf "draw %g" x)
            xs;
          near ~se:(sqrt (1. /. Float.of_int n)) 1. (mean xs);
          near ~se:(sqrt (8. /. Float.of_int n)) 1. (variance xs));
      test "split keys draw uncorrelated values" (fun () ->
          let ks = Rng.split (Rng.key 104) in
          let a = read (Rng.normal ~key:ks.(0) Nx.float64 [| n |]) in
          let b = read (Rng.normal ~key:ks.(1) Nx.float64 [| n |]) in
          let c = mean (Array.map2 ( *. ) a b) in
          near ~se:(sqrt (1. /. Float.of_int n)) 0. c);
      test "randint: every value of [low, high), uniformly" (fun () ->
          let xs =
            read (Rng.randint ~key:(Rng.key 105) ~low:(-3) ~high:5 [| n |])
          in
          let counts = Array.make 8 0 in
          Array.iter
            (fun x ->
              let x = Int32.to_int x in
              if x < -3 || x >= 5 then failf "draw %d outside [-3, 5)" x;
              counts.(x + 3) <- counts.(x + 3) + 1)
            xs;
          let p = 1. /. 8. in
          Array.iter
            (fun c ->
              near
                ~se:(sqrt (p *. (1. -. p) /. Float.of_int n))
                p
                (Float.of_int c /. Float.of_int n))
            counts);
      test "randint over int32's range" (fun () ->
          let xs =
            read
              (Rng.randint ~key:(Rng.key 106) ~low:(-0x8000_0000)
                 ~high:0x7FFF_FFFF [| 4096 |])
          in
          let m = mean (Array.map Int32.to_float xs) in
          near ~se:(4294967296. *. sqrt (1. /. 12. /. 4096.)) (-0.5) m);
      (* A range wider than 2^24, not a power of two: the draws land above 2^24,
         and both their high bins and their low six bits are uniform. With 64
         bins of 1024 expected draws a chi-square has 63 degrees of freedom,
         mean 63 and standard deviation √126. *)
      test "randint is uniform over a range wider than 2^24" (fun () ->
          let range = 3 lsl 28 in
          let xs =
            read (Rng.randint ~key:(Rng.key 108) ~high:range [| n |])
            |> Array.map Int32.to_int
          in
          let chi2 bin =
            let counts = Array.make 64 0 in
            Array.iter (fun x -> counts.(bin x) <- counts.(bin x) + 1) xs;
            let e = Float.of_int n /. 64. in
            Array.fold_left
              (fun a c -> a +. (((Float.of_int c -. e) ** 2.) /. e))
              0. counts
          in
          let bound = 63. +. (6. *. sqrt 126.) in
          Array.iter (fun x -> if x < 0 || x >= range then failf "draw %d" x) xs;
          greater int ~than:(1 lsl 24) (Array.fold_left max 0 xs);
          at_most float_exact ~than:bound (chi2 (fun x -> x / (range / 64)));
          at_most float_exact ~than:bound (chi2 (fun x -> x land 63)));
      test "randint refuses an empty range" (fun () ->
          equal string "Nx.Rng.randint: low 3 is not below high 3"
            (message (fun () -> Rng.randint ~low:3 ~high:3 [| 1 |])));
      test "randint refuses a bound outside int32" (fun () ->
          equal string "Nx.Rng.randint: [0, 4294967296) does not fit in int32"
            (message (fun () -> Rng.randint ~high:0x1_0000_0000 [| 1 |])));
      test "bernoulli: true at rate p" (fun () ->
          let p =
            Nx.Repr.of_array Nx.Host.v
              (A.of_array A.Dtype.Float32 [| 3; 1 |] [| 0.; 0.25; 1. |])
          in
          let p =
            Nx.mul p
              (Nx.Repr.of_array Nx.Host.v
                 (A.of_array A.Dtype.Float32 [| 1; n |] (Array.make n 1.)))
          in
          let xs = read (Rng.bernoulli ~key:(Rng.key 107) p) in
          let rate r =
            mean
              (Array.map
                 (fun b -> if b then 1. else 0.)
                 (Array.sub xs (r * n) n))
          in
          equal float_exact 0. (rate 0);
          near ~se:(sqrt (0.25 *. 0.75 /. Float.of_int n)) 0.25 (rate 1);
          equal float_exact 1. (rate 2));
      test "bernoulli refuses p outside [0, 1]" (fun () ->
          let p =
            Nx.Repr.of_array Nx.Host.v
              (A.of_array A.Dtype.Float64 [| 4 |] [| 0.5; 1.; 1.5; Float.nan |])
          in
          equal string "Nx.Rng.bernoulli: p at [2] is 1.5, not in [0, 1]"
            (message (fun () -> Rng.bernoulli p)));
    ]

(* The rejection samplers, each at parameters across its regimes. A gamma of
   concentration [a] has mean [a] and variance [a], its sample variance a
   standard error of [a √((2 + 6/a) / n)]. *)

let full dt v = host_of dt [| n |] (Array.make n v)

(* The modified Bessel function of the first kind [I_k(x)], by its series. *)
let bessel k x =
  let rec go m term acc =
    if m > 200 then acc
    else
      let acc = acc +. term in
      let m' = m + 1 in
      go m' (term *. (x /. 2.) *. (x /. 2.) /. Float.of_int (m' * (m' + k))) acc
  in
  let rec fact i = if i <= 1 then 1. else Float.of_int i *. fact (i - 1) in
  go 0 (((x /. 2.) ** Float.of_int k) /. fact k) 0.

let rejection =
  group "rejection samplers"
    [
      cases
        ~name:(fun a -> Printf.sprintf "gamma %g: mean and variance a" a)
        "gamma" [ 0.25; 1.; 2.5; 30. ]
        (fun a ->
          let xs = read (Rng.gamma ~key:(Rng.key 300) (full Nx.float64 a)) in
          Array.iter (fun x -> if not (x >= 0.) then failf "draw %g" x) xs;
          let nf = Float.of_int n in
          near ~se:(sqrt (a /. nf)) a (mean xs);
          near ~se:(a *. sqrt ((2. +. (6. /. a)) /. nf)) a (variance xs));
      test "gamma at float32 has the float64 mean" (fun () ->
          let xs = read (Rng.gamma ~key:(Rng.key 301) (full Nx.float32 3.)) in
          near ~se:(sqrt (3. /. Float.of_int n)) 3. (mean xs));
      test "beta 2 3: mean 2/5, in [0, 1]" (fun () ->
          let xs =
            read
              (Rng.beta ~key:(Rng.key 302) (full Nx.float64 2.)
                 (full Nx.float64 3.))
          in
          Array.iter
            (fun x -> if not (x >= 0. && x <= 1.) then failf "draw %g" x)
            xs;
          near ~se:(0.2 /. sqrt (Float.of_int n)) 0.4 (mean xs);
          near ~se:(1. /. sqrt (Float.of_int n)) 0.04 (variance xs));
      test "beta of tiny concentrations stays in [0, 1]" (fun () ->
          let xs =
            read
              (Rng.beta ~key:(Rng.key 303) (full Nx.float32 0.01)
                 (full Nx.float32 0.01))
          in
          Array.iter
            (fun x -> if not (x >= 0. && x <= 1.) then failf "draw %g" x)
            xs);
      cases
        ~name:(fun k -> Printf.sprintf "von_mises %g: E cos = I1/I0" k)
        "von_mises" [ 0.; 0.5; 2.; 50. ]
        (fun k ->
          let xs =
            read (Rng.von_mises ~key:(Rng.key 304) (full Nx.float64 k))
          in
          Array.iter
            (fun x ->
              if not (x >= -.Float.pi && x <= Float.pi) then failf "draw %g" x)
            xs;
          let se = 1. /. sqrt (Float.of_int n) in
          near ~se (bessel 1 k /. bessel 0 k) (mean (Array.map cos xs));
          near ~se 0. (mean (Array.map sin xs)));
      test "gamma refuses a concentration outside (0, inf)" (fun () ->
          let a = host_of Nx.float64 [| 3 |] [| 1.; -1.; Float.infinity |] in
          equal string "Nx.Rng.gamma: a at [1] is -1, not in (0, inf)"
            (message (fun () -> Rng.gamma a)));
      test "von_mises refuses an infinite concentration" (fun () ->
          let k = host_of Nx.float64 [| 2 |] [| 0.; Float.infinity |] in
          equal string "Nx.Rng.von_mises: k at [1] is inf, not in [0, inf)"
            (message (fun () -> Rng.von_mises k)));
      (* A count's sample variance has the standard error [√((μ4 - σ⁴) / n)],
         [μ4] its fourth central moment. *)
      cases
        ~name:(fun l -> Printf.sprintf "poisson %g: mean and variance rate" l)
        "poisson"
        [ 0.5; 4.; 10.; 37.; 1000. ]
        (fun l ->
          let xs =
            read (Rng.poisson ~key:(Rng.key 305) (full Nx.float64 l))
            |> Array.map Int32.to_float
          in
          Array.iter (fun x -> if x < 0. then failf "count %g" x) xs;
          let nf = Float.of_int n in
          near ~se:(sqrt (l /. nf)) l (mean xs);
          near ~se:(sqrt ((l +. (2. *. l *. l)) /. nf)) l (variance xs));
      test "poisson of rate 0 is 0" (fun () ->
          equal (array int32) (Array.make n 0l)
            (read (Rng.poisson ~key:(Rng.key 306) (full Nx.float32 0.))));
      cases
        ~name:(fun (t, p) -> Printf.sprintf "binomial %d %g: mean n p" t p)
        "binomial"
        [ (10, 0.3); (100, 0.5); (1000, 0.9); (7, 0.); (7, 1.) ]
        (fun (t, p) ->
          let count =
            host_of Nx.int32 [| n |] (Array.make n (Int32.of_int t))
          in
          let xs =
            read (Rng.binomial ~key:(Rng.key 307) count (full Nx.float64 p))
            |> Array.map Int32.to_float
          in
          let tf = Float.of_int t in
          Array.iter (fun x -> if x < 0. || x > tf then failf "count %g" x) xs;
          let nf = Float.of_int n and q = 1. -. p in
          let var = tf *. p *. q in
          let mu4 = var *. (1. +. (3. *. (tf -. 2.) *. p *. q)) in
          near ~se:(sqrt (var /. nf)) (tf *. p) (mean xs);
          near ~se:(sqrt ((mu4 -. (var *. var)) /. nf)) var (variance xs));
      test "poisson refuses a rate whose counts int32 cannot hold" (fun () ->
          let rate = host_of Nx.float64 [| 2 |] [| 1.; 2147483648. |] in
          equal string
            "Nx.Rng.poisson: rate at [1] is 2.14748e+09, not in [0, 2^31)"
            (message (fun () -> Rng.poisson rate)));
      test "binomial refuses a negative n" (fun () ->
          let count = host_of Nx.int32 [| 2 |] [| 3l; -2l |] in
          equal string "Nx.Rng.binomial: n at [1] is -2, not in [0, inf)"
            (message (fun () -> Rng.binomial count (full Nx.float32 0.5))));
    ]

(* The scope *)

let scope =
  group "scope"
    [
      test "keyless draws take the scope's keys in order" (fun () ->
          let k = Rng.key 200 in
          let a, b =
            Rng.with_key k (fun () ->
                let a = read (Rng.bits [| 3 |]) in
                (a, read (Rng.bits [| 3 |])))
          in
          equal (array int32) (read (Rng.bits ~key:(Rng.fold_in k 0) [| 3 |])) a;
          equal (array int32) (read (Rng.bits ~key:(Rng.fold_in k 1) [| 3 |])) b);
      test "an inner scope replaces the outer for its extent" (fun () ->
          let k = Rng.key 201 and k' = Rng.key 202 in
          let inner, after =
            Rng.with_key k (fun () ->
                let inner =
                  Rng.with_key k' (fun () -> read (Rng.bits [| 2 |]))
                in
                (inner, read (Rng.bits [| 2 |])))
          in
          equal (array int32)
            (read (Rng.bits ~key:(Rng.fold_in k' 0) [| 2 |]))
            inner;
          equal (array int32)
            (read (Rng.bits ~key:(Rng.fold_in k 0) [| 2 |]))
            after);
      test "next_key is the key the next draw takes" (fun () ->
          let k = Rng.key 203 in
          let key = Rng.with_key k (fun () -> words (Rng.next_key ())) in
          equal (array int32) (words (Rng.fold_in k 0)) key);
      test "a key with bytes roots no scope" (fun () ->
          let k = Rng.place Nx.Host.on (Rng.key 204) in
          equal string
            "Nx.Rng.with_key: the key has bytes; a scope's key is of every set"
            (message (fun () -> Rng.with_key k ignore)));
      test "unscoped draws differ" (fun () ->
          let a = read (Rng.bits [| 4 |]) and b = read (Rng.bits [| 4 |]) in
          not_equal (array int32) a b);
    ]

let () =
  exit (run "nx random" [ known; keys; draws; distributions; rejection; scope ])
