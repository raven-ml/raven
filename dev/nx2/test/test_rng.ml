(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.Rng: Threefry's known answers, the laws between keys and draws, and each
   sampler's distribution within stated bounds, over drawn keys. *)

open Windtrap
module A = Nx_array
module Rng = Nx.Rng

let m = Nx_support.memory

module S2 = (val Nx.devices [ m 0; m 1 ])
module Count = (val Nx.devices ~kernels:(module Nx_support.Counting) [ m 3 ])

let message f =
  match f () with _ -> "no exception" | exception Invalid_argument m -> m

(* [x]'s elements in row-major order, computed on the host. *)
let read x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))

(* [x]'s elements on the first device of its set. *)
let shard x = A.to_array (Option.get (Nx.Repr.shards x)).(0)
let host_of dt s xs = Nx.Repr.of_array Nx.Host.v (A.of_array dt s xs)
let words k = read (Rng.to_tensor k)

(* An int32 word as the unsigned 32-bit value it holds. *)
let u32 w = Int64.logand (Int64.of_int32 w) 0xFFFF_FFFFL
let word = Int32.of_int
let numel s = Array.fold_left ( * ) 1 s

(* Block [i] of the words [ws] of [split_batch]: a uint64, the first word
   low. *)
let block ws i =
  Int64.logor (u32 ws.(2 * i)) (Int64.shift_left (u32 ws.((2 * i) + 1)) 32)

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let shapes =
  Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 5)
  |> Gen.with_pp pp_shape

let seeds = Gen.int

(* The key of any two words. *)
let keys =
  Gen.map
    (fun (w0, w1) -> Rng.of_tensor (host_of Nx.int32 [| 2 |] [| w0; w1 |]))
    (Gen.pair Gen.int32 Gen.int32)

type float_dtype = F : (float, 's) A.Dtype.t -> float_dtype

let pp_dtype ppf (F dt) = Format.pp_print_string ppf (A.Dtype.name dt)

let wide = Gen.of_list ~pp:pp_dtype [ F Nx.float32; F Nx.float64 ]

(* Each float dtype with its uniform draw's precision [p], from its format:
   its significand's width, and 1 for float4_e2m1fn, whose values below 1 are 0
   and 1/2. *)
let floats =
  Gen.of_list
    ~pp:(fun ppf (dt, _) -> pp_dtype ppf dt)
    [
      (F Nx.float32, 24);
      (F Nx.float64, 53);
      (F Nx.bfloat16, 8);
      (F Nx.float16, 11);
      (F Nx.float8_e4m3fn, 4);
      (F Nx.float8_e5m2, 3);
      (F Nx.float4_e2m1fn, 1);
    ]

let pp_float ppf x = Format.fprintf ppf "%h" x
let floats_of xs = Gen.of_list ~pp:pp_float xs

(* A parameter: a value, or the least positive value of the dtype it is
   given in. *)
type param = Least | V of float

let pp_param ppf = function
  | Least -> Format.pp_print_string ppf "Least"
  | V x -> Format.fprintf ppf "V %h" x

let params xs = Gen.of_list ~pp:pp_param xs

(* [x] as [dt] holds it. *)
let at (type s) (dt : (float, s) A.Dtype.t) = function
  | Least ->
      let f = A.Dtype.float_format dt in
      f.min_normal *. f.epsilon
  | V x -> A.Dtype.of_float dt x

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

let n = 1 lsl 14
let mean xs = Array.fold_left ( +. ) 0. xs /. Float.of_int (Array.length xs)

let variance xs =
  let m = mean xs in
  Array.fold_left (fun a x -> a +. ((x -. m) *. (x -. m))) 0. xs
  /. Float.of_int (Array.length xs)

(* [x], a statistic of [n] draws of variance [var / n], within 6 standard
   errors of [want]: [√var / √n], which does not underflow at a tiny [var]. *)
let near ~var want x =
  let se = sqrt var /. sqrt (Float.of_int n) in
  at_most float_exact ~than:(6. *. se) (Float.abs (x -. want))

(* Two draws are uncorrelated: the mean product of two standard normal draws of
   [n] is within 6 standard errors, [1 / √n], of 0. *)
let uncorrelated k k' =
  let a = read (Rng.normal ~key:k Nx.float64 [| n |]) in
  let b = read (Rng.normal ~key:k' Nx.float64 [| n |]) in
  near ~var:1. 0. (mean (Array.map2 ( *. ) a b))

let keys_group =
  group "keys"
    [
      prop "a seed's words are its high and low halves" seeds (fun s ->
          equal (array int32) [| word (s asr 32); word s |] (words (Rng.key s)));
      test "a key is a value of every set" (fun () ->
          is_none (Nx.placement (Rng.to_tensor (Rng.key 7))));
      prop "split's keys are split_batch's rows"
        (Gen.pair keys (Gen.int_range 1 6))
        (fun (k, n) ->
          let rows = words (Rng.split_batch ~n k) in
          let each =
            Array.concat (Array.to_list (Array.map words (Rng.split ~n k)))
          in
          equal (array int32) rows each);
      prop "split keys are distinct and differ from their key"
        (Gen.pair keys (Gen.int_range 1 16))
        (fun (k, n) ->
          let split = Array.to_list (Array.map words (Rng.split ~n k)) in
          let all = words k :: split in
          equal int (n + 1) (List.length (List.sort_uniq compare all)));
      prop "fold_in of distinct indices gives distinct keys"
        (Gen.triple keys Gen.int Gen.int)
        (fun (k, i, i') ->
          assume (i <> i');
          not_equal (array int32) (words (Rng.fold_in k i))
            (words (Rng.fold_in k i')));
      prop "split keys draw uncorrelated values" keys (fun k ->
          let ks = Rng.split k in
          uncorrelated ks.(0) ks.(1));
      prop "fold_in keys draw uncorrelated values"
        (Gen.triple keys Gen.int Gen.int)
        (fun (k, i, i') ->
          assume (i <> i');
          uncorrelated (Rng.fold_in k i) (Rng.fold_in k i'));
      prop "fold_in_tensor of [n] is fold_in of n"
        (Gen.pair keys (Gen.int32_range Int32.min_int Int32.max_int))
        (fun (k, n) ->
          let i = host_of A.Dtype.Int32 [||] [| n |] in
          equal (array int32)
            (words (Rng.fold_in k (Int32.to_int n)))
            (words (Rng.fold_in_tensor k i)));
      test "fold_in_tensor of a batch of indices is a batch of keys" (fun () ->
          let k = Rng.key 5 in
          let i = host_of A.Dtype.Int32 [| 3 |] [| 0l; 1l; -4l |] in
          let each =
            Array.concat
              (List.map (fun n -> words (Rng.fold_in k n)) [ 0; 1; -4 ])
          in
          equal (array int32) each (words (Rng.fold_in_tensor k i)));
      test "fold_in_tensor joins a key batch and indices that both broadcast"
        (fun () ->
          let ks = Rng.split ~n:2 (Rng.key 5) in
          let batch =
            Rng.of_tensor
              (Nx.reshape [| 2; 1; 2 |]
                 (Rng.to_tensor (Rng.split_batch ~n:2 (Rng.key 5))))
          in
          let i = host_of A.Dtype.Int32 [| 3 |] [| 0l; 7l; -4l |] in
          let each =
            Array.concat
              (List.concat_map
                 (fun k ->
                   List.map (fun n -> words (Rng.fold_in k n)) [ 0; 7; -4 ])
                 (Array.to_list ks))
          in
          let y = Rng.fold_in_tensor batch i in
          equal (array int) [| 2; 3; 2 |] (Nx.shape (Rng.to_tensor y));
          equal (array int32) each (words y));
      test "of_tensor computes nothing, from reversed words too" (fun () ->
          let t = Nx.place Count.on (host_of Nx.int32 [| 2 |] [| 1l; 2l |]) in
          let reversed = Nx.flip t in
          Nx_support.Counting.reset ();
          ignore (Rng.of_tensor reversed);
          equal int 0 (Nx_support.Counting.calls ()));
      prop "of_tensor undoes to_tensor" keys (fun k ->
          let k = Rng.split_batch ~n:3 k in
          equal (array int32) (words k)
            (words (Rng.of_tensor (Rng.to_tensor k))));
      test "of_tensor refuses a last axis other than 2" (fun () ->
          let t = host_of A.Dtype.Int32 [| 3 |] [| 0l; 1l; 2l |] in
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

(* Each sampler at fixed parameters, its draw of a shape read as floats. *)
let samplers =
  let of_ints xs = Array.map Int32.to_float xs in
  let full dt s v = host_of dt s (Array.make (numel s) v) in
  let counts s = host_of Nx.int32 s (Array.make (numel s) 40l) in
  Gen.of_list
    ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name)
    [
      ("bits", fun k s -> of_ints (read (Rng.bits ~key:k s)));
      ("uniform", fun k s -> read (Rng.uniform ~key:k Nx.float16 s));
      ("normal", fun k s -> read (Rng.normal ~key:k Nx.float32 s));
      ("exponential", fun k s -> read (Rng.exponential ~key:k Nx.float64 s));
      ( "randint",
        fun k s -> of_ints (read (Rng.randint ~key:k ~low:(-9) ~high:1000 s)) );
      ( "bernoulli",
        fun k s ->
          Array.map Bool.to_float
            (read (Rng.bernoulli ~key:k (full Nx.float32 s 0.3))) );
      ("gamma", fun k s -> read (Rng.gamma ~key:k (full Nx.float32 s 0.7)));
      ( "beta",
        fun k s ->
          read (Rng.beta ~key:k (full Nx.float64 s 2.) (full Nx.float64 s 0.5))
      );
      ( "von_mises",
        fun k s -> read (Rng.von_mises ~key:k (full Nx.float32 s 3.)) );
      ( "poisson",
        fun k s -> of_ints (read (Rng.poisson ~key:k (full Nx.float64 s 30.)))
      );
      ( "binomial",
        fun k s ->
          of_ints
            (read (Rng.binomial ~key:k (counts s) (full Nx.float32 s 0.25))) );
    ]

(* The words [w] at their uniform's precision [p] of 24 or less: their low [p]
   bits scaled by [2^-p]. *)
let low_bits p w =
  Float.ldexp (Int32.to_float (Int32.logand w (word ((1 lsl p) - 1)))) (-p)

(* 53 bits of block [i] of [ws]: its first word's low 21 bits over its
   second. *)
let bits53 ws i =
  Int64.logor
    (Int64.shift_left (Int64.logand (u32 ws.(2 * i)) 0x1F_FFFFL) 32)
    (u32 ws.((2 * i) + 1))

(* Lemire's multiply-shift of the range [low, high) at element [j] of [n]:
   block [j]'s 64 bits times the range, shifted down by 64, unless its low 64
   bits fall below [2^64 mod range], where block [n + j] stands whatever its
   low bits. The flag is whether element [j] took the second block. *)
let lemire ws ~low ~high ~n j =
  let r = Int64.of_int (high - low) in
  let t = Int64.unsigned_rem (Int64.neg r) r in
  let pick i =
    let w = block ws i in
    let wh = Int64.shift_right_logical w 32 and wl = u32 (Int64.to_int32 w) in
    let hi =
      Int64.shift_right_logical
        (Int64.add (Int64.mul wh r)
           (Int64.shift_right_logical (Int64.mul wl r) 32))
        32
    in
    (hi, Int64.unsigned_compare (Int64.mul w r) t < 0)
  in
  let hi, rejected = pick j in
  let hi = if rejected then fst (pick (n + j)) else hi in
  (Int32.of_int (low + Int64.to_int hi), rejected)

(* A range [low, high) inside int32: small ones, ones at its extremes, and the
   whole of it. *)
let ranges =
  let open Gen in
  let fits v = v >= -0x8000_0000 && v <= 0x7FFF_FFFF in
  let pp ppf (l, h) = Format.fprintf ppf "[%d, %d)" l h in
  one_of
    [
      (let+ low = int_range (-0x8000_0000) 0x7FFF_FFFE
       and+ width = int_range 1 64 in
       (low, min (low + width) 0x7FFF_FFFF));
      (let+ a = int_range (-0x8000_0000) 0x7FFF_FFFF
       and+ b = int_range (-0x8000_0000) 0x7FFF_FFFF in
       if a = b then (a, a + 1) else (min a b, max a b));
      of_list [ (-0x8000_0000, 0x7FFF_FFFF); (-0x8000_0000, -0x7FFF_FFFF) ];
    ]
  |> such_that (fun (l, h) -> fits l && fits h && l < h)
  |> with_pp pp

(* Block 0 of the key of words [(1343584651, 2)], times [0xFFFF0001], falls
   below [2^64 mod 0xFFFF0001]: its first round is rejected. *)
let second_round =
  ( host_of Nx.int32 [| 2 |] [| 1343584651l; 2l |],
    (-0x8000_0000, 0x7FFF_0001),
    1 )

let probabilities =
  let open Gen in
  one_of
    [
      of_list [ 0.; 1.; 0.5 ];
      float_range 0. 1.;
      map (fun e -> Float.ldexp 1. e) (int_range (-60) (-1));
    ]

let draws =
  group "draws"
    [
      prop "bits are split_batch's words in order"
        (Gen.pair keys (Gen.int_range 0 9))
        (fun (k, n) ->
          cover "odd" (n mod 2 = 1);
          cover "none" (n = 0);
          let blocks = words (Rng.split_batch ~n:(((n + 1) / 2) + 1) k) in
          equal (array int32) (Array.sub blocks 0 n)
            (read (Rng.bits ~key:k [| n |])));
      prop "a draw is its flat draw reshaped" (Gen.pair keys shapes)
        (fun (k, s) ->
          equal (array int32)
            (read (Rng.bits ~key:k [| numel s |]))
            (read (Rng.bits ~key:k s)));
      prop "equal keys give equal draws"
        (Gen.triple seeds samplers shapes)
        (fun (seed, (_, draw), s) ->
          let k = Rng.key seed in
          let k' = Rng.of_tensor (host_of Nx.int32 [| 2 |] (words k)) in
          equal (array float_exact) (draw k s) (draw k' s));
      prop "a draw is the same on every set" (Gen.pair seeds shapes)
        (fun (seed, s) ->
          let k = Rng.key seed in
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
        (Gen.triple keys (Gen.int_range 1 4) shapes)
        (fun (k, n, s) ->
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
      prop "a narrow float uniform is the low p bits of bits, scaled"
        (Gen.pair keys floats)
        (fun (k, (F dt, p)) ->
          assume (p <= 24);
          let want = Array.map (low_bits p) (read (Rng.bits ~key:k [| 64 |])) in
          equal (array float_exact) want
            (read (Rng.uniform ~key:k dt [| 64 |])));
      prop "float64 uniform is 53 bits of each block, scaled" keys (fun k ->
          let ws = words (Rng.split_batch ~n:64 k) in
          let want =
            Array.init 64 (fun i ->
                Float.ldexp (Int64.to_float (bits53 ws i)) (-53))
          in
          equal (array float_exact) want
            (read (Rng.uniform ~key:k Nx.float64 [| 64 |])));
      prop "randint is Lemire's multiply-shift of each block"
        ~examples:[ second_round ]
        (Gen.triple
           (Gen.map Rng.to_tensor keys)
           ranges (Gen.int_range 0 64))
        (fun (k, (low, high), n) ->
          let k = Rng.of_tensor k in
          let ws = words (Rng.split_batch ~n:(max 1 (2 * n)) k) in
          let want = Array.init n (lemire ws ~low ~high ~n) in
          cover "a second round" (Array.exists snd want);
          equal (array int32) (Array.map fst want)
            (read (Rng.randint ~key:k ~low ~high [| n |])));
      prop "bernoulli is a 53-bit draw below p"
        (Gen.triple keys wide
           (Gen.array ~size:(Gen.int_range 0 64) probabilities))
        (fun (k, F dt, ps) ->
          let ps = Array.map (A.Dtype.of_float dt) ps in
          let p = host_of dt [| Array.length ps |] ps in
          let ws = words (Rng.split_batch ~n:(max 1 (Array.length ps)) k) in
          let want =
            Array.mapi
              (fun j p ->
                Int64.compare (bits53 ws j)
                  (Int64.of_float (Float.ceil (Float.ldexp p 53)))
                < 0)
              ps
          in
          equal (array bool) want (read (Rng.bernoulli ~key:k p)));
      test "a negative extent raises" (fun () ->
          equal string "Nx.Rng.uniform: shape [2; -1] has a negative extent"
            (message (fun () -> Rng.uniform Nx.float32 [| 2; -1 |])));
    ]

(* Distributions. Each moment of [n] draws is checked within 6 standard errors
   of its value, over drawn keys. *)

let distributions =
  group "distributions"
    [
      prop "uniform draws are equally likely multiples of 2^-p in [0, 1)"
        (Gen.pair keys floats)
        (fun (k, (F dt, p)) ->
          let xs = read (Rng.uniform ~key:k dt [| n |]) in
          Array.iter
            (fun x ->
              let scaled = Float.ldexp x p in
              let whole = Float.equal scaled (Float.round scaled) in
              if not (x >= 0. && x < 1. && whole) then
                failf "%h is not a multiple of 2^-%d in [0, 1)" x p)
            xs;
          if p <= 8 then begin
            let bins = 1 lsl p in
            let counts = Array.make bins 0 in
            Array.iter
              (fun x ->
                let i = int_of_float (Float.ldexp x p) in
                counts.(i) <- counts.(i) + 1)
              xs;
            let q = 1. /. Float.of_int bins and nf = Float.of_int n in
            Array.iter
              (fun c -> near ~var:(q *. (1. -. q)) q (Float.of_int c /. nf))
              counts
          end);
      prop "uniform: mean 1/2, variance 1/12" (Gen.pair keys wide)
        (fun (k, F dt) ->
          let xs = read (Rng.uniform ~key:k dt [| n |]) in
          near ~var:(1. /. 12.) 0.5 (mean xs);
          near ~var:(1. /. 180.) (1. /. 12.) (variance xs));
      prop "normal: mean 0, variance 1" (Gen.pair keys wide)
        (fun (k, F dt) ->
          let xs = read (Rng.normal ~key:k dt [| n |]) in
          Array.iter
            (fun x -> if not (Float.is_finite x) then failf "draw %g" x)
            xs;
          near ~var:1. 0. (mean xs);
          near ~var:2. 1. (variance xs));
      (* Key 3743's float32 uniform draw is 0 at index 1386, where the
         exponential is -log 1: +0, never -0. *)
      test "exponential of a zero uniform draw is +0" (fun () ->
          let k = Rng.key 3743 in
          equal float_exact 0.
            (read (Rng.uniform ~key:k Nx.float32 [| 4096 |])).(1386);
          let x =
            (read (Rng.exponential ~key:k Nx.float32 [| 4096 |])).(1386)
          in
          equal (pair float_exact bool) (0., false) (x, Float.sign_bit x));
      prop "exponential: finite, never negative, mean 1, variance 1"
        (Gen.pair keys wide)
        (fun (k, F dt) ->
          let xs = read (Rng.exponential ~key:k dt [| n |]) in
          Array.iter
            (fun x ->
              if not (x >= 0. && Float.is_finite x && not (Float.sign_bit x))
              then failf "draw %g" x)
            xs;
          near ~var:1. 1. (mean xs);
          near ~var:8. 1. (variance xs));
      prop "randint draws every value of [low, high), uniformly"
        (Gen.pair keys ranges)
        (fun (k, (low, high)) ->
          let xs =
            read (Rng.randint ~key:k ~low ~high [| n |])
            |> Array.map Int32.to_int
          in
          Array.iter
            (fun x ->
              if x < low || x >= high then
                failf "draw %d outside [%d, %d)" x low high)
            xs;
          let width = high - low and nf = Float.of_int n in
          if width <= 64 then begin
            cover "one value" (width = 1);
            let counts = Array.make width 0 in
            Array.iter (fun x -> counts.(x - low) <- counts.(x - low) + 1) xs;
            let p = 1. /. Float.of_int width in
            Array.iter
              (fun c -> near ~var:(p *. (1. -. p)) p (Float.of_int c /. nf))
              counts
          end
          else begin
            (* A uniform draw over [w] values has variance [(w² - 1) / 12]. *)
            let w = Float.of_int width in
            let centre = Float.of_int low +. ((w -. 1.) /. 2.) in
            near ~var:(((w *. w) -. 1.) /. 12.) centre
              (mean (Array.map Float.of_int xs))
          end);
      (* A range wider than 2^24, not a power of two: the draws land above 2^24,
         and both their high bins and their low six bits are uniform. With 64
         bins of 1024 expected draws a chi-square has 63 degrees of freedom,
         mean 63 and standard deviation √126. *)
      prop "randint is uniform over a range wider than 2^24" keys (fun k ->
          let range = 3 lsl 28 and n = 1 lsl 16 in
          let xs =
            read (Rng.randint ~key:k ~high:range [| n |])
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
      (* The rate's bound is a normal one, so [p] keeps [n p] and [n (1 - p)]
         above 16; "a 53-bit draw below p" pins the rarer events. *)
      prop "bernoulli: true at rate p"
        (let edge = Float.ldexp 1. (-10) in
         Gen.triple keys wide
           (Gen.one_of
              [ Gen.of_list [ 0.; 1. ]; Gen.float_range edge (1. -. edge) ]))
        (fun (k, F dt, p) ->
          let p = A.Dtype.of_float dt p in
          let ps = host_of dt [| n |] (Array.make n p) in
          let xs = read (Rng.bernoulli ~key:k ps) in
          let rate = mean (Array.map Bool.to_float xs) in
          cover "never" (p = 0.);
          cover "always" (p = 1.);
          if p = 0. || p = 1. then equal float_exact p rate
          else near ~var:(p *. (1. -. p)) p rate);
      test "bernoulli refuses p outside [0, 1]" (fun () ->
          let p =
            host_of A.Dtype.Float64 [| 4 |] [| 0.5; 1.; 1.5; Float.nan |]
          in
          equal string "Nx.Rng.bernoulli: p at [2] is 1.5, not in [0, 1]"
            (message (fun () -> Rng.bernoulli p)));
    ]

(* The rejection samplers, each at parameters across its regimes and at the
   ends of its domain. *)

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

(* A gamma of concentration [a] has mean [a], variance [a] and excess kurtosis
   [6 / a], so its sample variance has variance [(2 a² + 6 a) / n]. *)
let gamma_moments a xs =
  Array.iter
    (fun x -> if not (x >= 0. && Float.is_finite x) then failf "draw %g" x)
    xs;
  near ~var:a a (mean xs);
  near ~var:((2. *. a *. a) +. (6. *. a)) a (variance xs)

(* A count's sample variance has variance [(μ4 - σ⁴) / n], [μ4] its fourth
   central moment, which is 0 for a constant count, where rounding can give
   -0. *)
let count_moments ~mean:mu ~var ~mu4 xs =
  near ~var mu (mean xs);
  near ~var:(Float.max 0. (mu4 -. (var *. var))) var (variance xs)

(* A draw of two parameters, beta's or binomial's, has more operands than an
   nx.cpu map holds and runs node by node, at 20 to 60 times the cost of a
   draw of one: its laws run on [few] cases. *)
let few = 25

let rejection =
  group "rejection samplers"
    [
      prop "gamma: mean and variance a"
        (Gen.triple keys wide
           (params
              [ Least; V 1e-30; V 0.01; V 0.25; V 1.; V 2.5; V 30.; V 1e6 ]))
        (fun (k, F dt, a) ->
          let a = at dt a in
          gamma_moments a (read (Rng.gamma ~key:k (full dt a))));
      (* beta's draw is x = 1 / (1 + exp d), d = log G(b) - log G(a), for the
         gammas its key's two split keys draw. d is a sum of logarithms, each
         rounded, so its error is a few ulps of their magnitudes, and x moves by
         x (1 - x) per unit of d: the draw lies within 16 ε x (1 - x) (2 + |log
         G(a)| + |log G(b)|), plus 4 ulps for the exponential, the sum, the
         quotient and the reference's own two roundings, of the gammas' ratio
         formed in float64 from their draws. *)
      prop ~count:few "beta is the gammas' ratio"
        (Gen.pair keys
           (Gen.of_list
              ~pp:(fun ppf (a, b) -> Format.fprintf ppf "(%g, %g)" a b)
              [ (0.5, 0.5); (0.3, 4.); (2., 3.); (9., 0.7); (40., 25.) ]))
        (fun (k, (a, b)) ->
          let ks = Rng.split k in
          let ga = read (Rng.gamma ~key:ks.(0) (full Nx.float64 a)) in
          let gb = read (Rng.gamma ~key:ks.(1) (full Nx.float64 b)) in
          let xs =
            read (Rng.beta ~key:k (full Nx.float64 a) (full Nx.float64 b))
          in
          (* The error of each draw as a fraction of its bound. *)
          let share i x =
            let r = ga.(i) /. (ga.(i) +. gb.(i)) in
            let logs = 2. +. Float.abs (log ga.(i)) +. Float.abs (log gb.(i)) in
            let bound =
              (16. *. epsilon_float *. r *. (1. -. r) *. logs)
              +. (4. *. (Float.succ r -. r))
            in
            Float.abs (x -. r) /. bound
          in
          let worst = ref 0. in
          Array.iteri (fun i x -> worst := Float.max !worst (share i x)) xs;
          at_most float_exact ~than:1. !worst);
      (* Beta(a, b) has mean a / (a + b) and variance a b / ((a + b)² (a + b +
         1)). At the least positive concentrations both gammas underflow and a
         draw is 0 or 1, each with probability 1/2, from the logarithms of
         their shifts. *)
      prop ~count:few "beta: in [0, 1], mean a / (a + b)"
        (Gen.triple keys wide
           (Gen.of_list
              ~pp:(fun ppf (a, b) ->
                Format.fprintf ppf "(%a, %a)" pp_param a pp_param b)
              [
                (Least, Least);
                (V 0.01, V 0.01);
                (V 0.5, V 0.5);
                (V 2., V 3.);
                (V 1e-3, V 1e3);
                (V 1e6, V 1e6);
              ]))
        (fun (k, F dt, (a, b)) ->
          let a = at dt a and b = at dt b in
          let xs = read (Rng.beta ~key:k (full dt a) (full dt b)) in
          Array.iter
            (fun x -> if not (x >= 0. && x <= 1.) then failf "draw %g" x)
            xs;
          let s = a +. b in
          let var = a /. s *. (b /. s) /. (s +. 1.) in
          near ~var (a /. s) (mean xs));
      test "beta over parameters that both broadcast is beta over their join"
        (fun () ->
          let a = host_of Nx.float64 [| 2; 1 |] [| 0.5; 4. |] in
          let b = host_of Nx.float64 [| 1; 3 |] [| 1.; 2.; 9. |] in
          let k = Rng.key 309 in
          equal (array float_exact)
            (read
               (Rng.beta ~key:k
                  (Nx.broadcast_to [| 2; 3 |] a)
                  (Nx.broadcast_to [| 2; 3 |] b)))
            (read (Rng.beta ~key:k a b)));
      test
        "binomial over parameters that both broadcast is binomial over their \
         join" (fun () ->
          let t = host_of Nx.int32 [| 2; 1 |] [| 5l; 400l |] in
          let p = host_of Nx.float64 [| 1; 3 |] [| 0.1; 0.5; 0.95 |] in
          let k = Rng.key 310 in
          equal (array int32)
            (read
               (Rng.binomial ~key:k
                  (Nx.broadcast_to [| 2; 3 |] t)
                  (Nx.broadcast_to [| 2; 3 |] p)))
            (read (Rng.binomial ~key:k t p)));
      (* E cos θ = I1(κ) / I0(κ) and E sin θ = 0, each draw's cosine and sine of
         variance at most 1. π is the dtype's, which at float32 lies above
         float64's. *)
      prop "von_mises: in [-π, π], E cos = I1/I0"
        (Gen.triple keys wide
           (params [ V 0.; Least; V 1e-8; V 0.5; V 1.; V 2.; V 50. ]))
        (fun (k, F dt, kappa) ->
          let kappa = at dt kappa in
          let xs = read (Rng.von_mises ~key:k (full dt kappa)) in
          let pi = A.Dtype.of_float dt Float.pi in
          Array.iter
            (fun x -> if not (x >= -.pi && x <= pi) then failf "draw %h" x)
            xs;
          near ~var:1.
            (bessel 1 kappa /. bessel 0 kappa)
            (mean (Array.map cos xs));
          near ~var:1. 0. (mean (Array.map sin xs)));
      prop "von_mises of a huge concentration stays in [-π, π]"
        (Gen.pair keys wide)
        (fun (k, F dt) ->
          let xs = read (Rng.von_mises ~key:k (full dt 1e30)) in
          let pi = A.Dtype.of_float dt Float.pi in
          Array.iter
            (fun x -> if not (x >= -.pi && x <= pi) then failf "draw %h" x)
            xs);
      test "gamma refuses a concentration outside (0, inf)" (fun () ->
          let a = host_of Nx.float64 [| 3 |] [| 1.; -1.; Float.infinity |] in
          equal string "Nx.Rng.gamma: a at [1] is -1, not in (0, inf)"
            (message (fun () -> Rng.gamma a)));
      test "von_mises refuses an infinite concentration" (fun () ->
          let k = host_of Nx.float64 [| 2 |] [| 0.; Float.infinity |] in
          equal string "Nx.Rng.von_mises: k at [1] is inf, not in [0, inf)"
            (message (fun () -> Rng.von_mises k)));
      (* A Poisson count of rate [l] has mean and variance [l] and fourth
         central moment [l + 3 l²]. The bounds are normal ones, so [n l] is
         above 16, or so small that a count above 0 has no chance. *)
      prop "poisson: mean and variance rate"
        (Gen.pair keys
           (floats_of
              [
                Float.ldexp 1. (-1074);
                1e-3;
                0.5;
                4.;
                9.99;
                10.;
                37.;
                1000.;
                1e6;
              ]))
        (fun (k, l) ->
          let xs =
            read (Rng.poisson ~key:k (full Nx.float64 l))
            |> Array.map Int32.to_float
          in
          Array.iter (fun x -> if x < 0. then failf "count %g" x) xs;
          count_moments ~mean:l ~var:l ~mu4:(l +. (3. *. l *. l)) xs);
      prop "poisson of rate 0 is 0" (Gen.triple keys floats shapes)
        (fun (k, (F dt, _), s) ->
          let rate = host_of dt s (Array.make (numel s) 0.) in
          equal (array int32) (Array.make (numel s) 0l)
            (read (Rng.poisson ~key:k rate)));
      prop "poisson near int32's bound counts within [0, 2^31)"
        (Gen.pair keys (floats_of [ 2147483647.; 2147400000. ]))
        (fun (k, l) ->
          let rate = host_of Nx.float64 [| 64 |] (Array.make 64 l) in
          let xs = read (Rng.poisson ~key:k rate) in
          Array.iter (fun x -> if x < 0l then failf "count %ld" x) xs);
      (* A binomial count of [t] trials of probability [p] has mean [t p],
         variance [t p q] and fourth central moment [t p q (1 + 3 (t - 2) p
         q)]. *)
      prop ~count:few "binomial: in [0, n], mean n p, variance n p q"
        (Gen.pair keys
           (Gen.of_list
              ~pp:(fun ppf (t, p) -> Format.fprintf ppf "(%d, %h)" t p)
              [
                (0, 0.5);
                (7, 0.);
                (7, 1.);
                (1, 0.3);
                (10, 0.3);
                (100, 0.5);
                (1000, 0.9);
                (0x7FFF_FFFF, 1e-12);
                (0x7FFF_FFFF, 0.5);
                (40, Float.ldexp 1. (-1074));
              ]))
        (fun (k, (t, p)) ->
          let count =
            host_of Nx.int32 [| n |] (Array.make n (Int32.of_int t))
          in
          let xs =
            read (Rng.binomial ~key:k count (full Nx.float64 p))
            |> Array.map Int32.to_float
          in
          let tf = Float.of_int t in
          Array.iter (fun x -> if x < 0. || x > tf then failf "count %g" x) xs;
          let q = 1. -. p in
          let var = tf *. p *. q in
          count_moments ~mean:(tf *. p) ~var
            ~mu4:(var *. (1. +. (3. *. (tf -. 2.) *. p *. q)))
            xs);
      (* A narrow float's largest finite value is a rate like any other: the
         domain is checked at the compute dtype, where 2^31 is a value. *)
      cases
        ~name:(fun (n, _) -> "poisson accepts " ^ n ^ "'s largest rate")
        "narrow rates"
        [
          ( "float4_e2m1fn",
            fun () -> Rng.poisson (host_of Nx.float4_e2m1fn [| 1 |] [| 6. |]) );
          ( "float8_e4m3fn",
            fun () -> Rng.poisson (host_of Nx.float8_e4m3fn [| 1 |] [| 448. |])
          );
          ( "float8_e5m2",
            fun () -> Rng.poisson (host_of Nx.float8_e5m2 [| 1 |] [| 57344. |])
          );
        ]
        (fun (_, f) -> equal int 1 (Array.length (read (f ()))));
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

(* An interpretation that traces every operation it reaches. *)
type ('v, 's, 'd) Nx.Prim.payload += Traced : ('v, 's, 'd) Nx.Prim.payload

let tracing i ~by op =
  Nx.Prim.results ~by (fun _ form -> Nx.Prim.traced i form Traced) op

let scope =
  group "scope"
    [
      prop "keyless draws take the scope's keys in order" seeds (fun s ->
          let k = Rng.key s in
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
      test "a scoped keyless draw under a tracing extent is traced" (fun () ->
          let k = Rng.key 205 in
          Nx.Prim.interpret ~name:"test.trace" Extent tracing (fun i ->
              Rng.with_key k (fun () ->
                  is_some (Nx.Prim.payload i (Rng.bits [| 2 |])))));
      test
        "an unscoped keyless draw under a tracing extent leaves the domain's \
         generator usable" (fun () ->
          let run () =
            let traced =
              Nx.Prim.interpret ~name:"test.trace" Extent tracing (fun i ->
                  Option.is_some (Nx.Prim.payload i (Rng.bits [| 2 |])))
            in
            (traced, Array.length (read (Rng.bits [| 2 |])))
          in
          equal (pair bool int) (true, 2) (Domain.join (Domain.spawn run)));
    ]

let () =
  exit
    (run "nx random"
       [ known; keys_group; draws; distributions; rejection; scope ])
