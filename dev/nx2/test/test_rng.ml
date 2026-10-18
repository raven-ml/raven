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
let shard x = A.to_array (Iarray.get (Option.get (Nx.Repr.shards x)) 0)
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

(* Shapes up to rank 6, half of them of rank 5 or 6 with no extent below 2,
   whose positions weigh every axis. *)
let shapes =
  Gen.one_of
    [
      Gen.array ~size:(Gen.int_range 0 6) (Gen.int_range 0 4);
      Gen.array ~size:(Gen.int_range 5 6) (Gen.int_range 2 3);
    ]
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

(* The [n] draws [xs] follow the continuous CDF [cdf]: their
   Kolmogorov-Smirnov statistic is below [3 / √n], which a sample of the
   distribution exceeds with probability about [2 e^-18]. *)
let ks cdf xs =
  let xs = Array.copy xs in
  Array.sort Float.compare xs;
  let nf = Float.of_int (Array.length xs) in
  let d = ref 0. in
  Array.iteri
    (fun i x ->
      let f = cdf x in
      d := Float.max !d (Float.max (Float.of_int (i + 1) /. nf -. f) (f -. Float.of_int i /. nf)))
    xs;
  at_most float_exact ~than:(3. /. sqrt nf) !d

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
      test "a key's walk is one array at the root, checked on rebuild"
        (fun () ->
          let k = Rng.key 7 and ptree = Nx.Ptree.v Rng.walk in
          equal ~msg:"visits" (list string) [ "the root: a leaf" ]
            (List.map
               (Format.asprintf "%a" Nx.Ptree.pp_visit)
               (Nx.Ptree.visits ptree k));
          let k' = Nx.Ptree.map ptree (fun _ a -> a) k in
          equal ~msg:"map" (array int32) (words k) (words k');
          let word = Nx.P (host_of Nx.int32 [| 3 |] [| 1l; 2l; 3l |]) in
          raises_match
            (Exn.invalid_arg ~substring:"Nx.Rng.of_tensor: ")
            (fun () -> Nx.Ptree.rebuild ptree ~like:k [ word ]));
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
      ( "truncated_normal",
        fun k s ->
          read
            (Rng.truncated_normal ~key:k (full Nx.float32 s (-1.))
               (full Nx.float32 s 2.)) );
      ( "dirichlet",
        fun k s ->
          let a = full Nx.float64 (Array.append s [| 3 |]) 0.7 in
          read (Rng.dirichlet ~key:k a) );
      ( "categorical",
        fun k s ->
          let logits = full Nx.float32 (Array.append s [| 4 |]) 0. in
          Array.map Int64.to_float (read (Rng.categorical ~key:k logits)) );
      ( "permutation",
        fun k s ->
          Array.map Int64.to_float (read (Rng.permutation ~key:k (numel s))) );
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

(* Lemire's multiply-shift of the range [low, high) at element [j]: block
   [2j]'s 64 bits times the range, shifted down by 64, unless its low 64 bits
   fall below [2^64 mod range], where block [2j + 1] stands whatever its low
   bits. The flag is whether element [j] took the second block. *)
let lemire ws ~low ~high j =
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
  let hi, rejected = pick (2 * j) in
  let hi = if rejected then fst (pick ((2 * j) + 1)) else hi in
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

(* Block 0 of the key of words [(-1235405770, 2)], times [0xFFFF0001], falls
   below [2^64 mod 0xFFFF0001]: its first round is rejected. *)
let second_round =
  ( host_of Nx.int32 [| 2 |] [| -1235405770l; 2l |],
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
          cover "rank 5 or more, no extent below 2"
            (Array.length s >= 5 && Array.for_all (fun d -> d >= 2) s);
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
      (* A cut of an inner axis has no contiguous image in a draw flattened
         to rows: a draw keeps its parameters' shape, and their placement. *)
      test "a draw over a split parameter is split as it is" (fun () ->
          let p =
            host_of Nx.float32 [| 4; 6 |]
              (Array.init 24 (fun i -> Float.of_int i /. 24.))
          in
          let k = Rng.key 3 in
          let at = S2.split ~axis:1 in
          let x = Rng.bernoulli ~key:k (Nx.place at p) in
          equal bool true (Nx.Placement.equal at (Option.get (Nx.placement x)));
          equal (array bool) (read (Rng.bernoulli ~key:k p)) (read x));
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
      prop "a batch of keys draws each key's draw, for every sampler"
        (Gen.triple keys (Gen.int_range 1 3) samplers)
        (fun (k, n, (_, draw)) ->
          let s = [| 2; 3 |] in
          let each =
            Array.concat
              (Array.to_list (Array.map (fun k -> draw k s) (Rng.split ~n k)))
          in
          equal (array float_exact) each (draw (Rng.split_batch ~n k) s));
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
          let want = Array.init n (lemire ws ~low ~high) in
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
      prop "normal: Kolmogorov-Smirnov against the normal CDF"
        (Gen.pair keys wide)
        (fun (k, F dt) ->
          let cdf x = 0.5 *. (1. +. Float.erf (x /. Float.sqrt 2.)) in
          ks cdf (read (Rng.normal ~key:k dt [| n |])));
      (* Draws 2q and 2q + 1 are r cos θ and r sin θ, r = √(-2 log u1) and θ =
         2π u2, of the uniform draws u1 = 2q and u2 = 2q + 1, u1 floored at
         2^-53: within 2^-46 of the transform computed here, 8 ulps of the
         radius's bound √(-2 log 2^-53) < 9. *)
      prop "float64 normal is the Box-Muller transform of uniform pairs" keys
        (fun k ->
          let u = read (Rng.uniform ~key:k Nx.float64 [| 64 |]) in
          let want =
            Array.init 64 (fun j ->
                let q = j / 2 * 2 in
                let u1 = Float.max u.(q) (Float.ldexp 1. (-53)) in
                let r = sqrt (-2. *. log u1) in
                let angle = 2. *. Float.pi *. u.(q + 1) in
                if j mod 2 = 0 then r *. cos angle else r *. sin angle)
          in
          equal (array (float (Float.ldexp 1. (-46)))) want
            (read (Rng.normal ~key:k Nx.float64 [| 64 |])));
      (* Elements 2q and 2q + 1 are one Box-Muller pair: independent standard
         normals, whose mean product is 0 within 6 standard errors. *)
      prop "normal: a draw's pairs are uncorrelated" (Gen.pair keys wide)
        (fun (k, F dt) ->
          let xs = read (Rng.normal ~key:k dt [| 2 * n |]) in
          let products = Array.init n (fun q -> xs.(2 * q) *. xs.((2 * q) + 1)) in
          near ~var:1. 0. (mean products));
      (* Key 443's float32 uniform draw is 0 at index 646, where the
         exponential is -log 1: +0, never -0. *)
      test "exponential of a zero uniform draw is +0" (fun () ->
          let k = Rng.key 443 in
          equal float_exact 0.
            (read (Rng.uniform ~key:k Nx.float32 [| 4096 |])).(646);
          let x =
            (read (Rng.exponential ~key:k Nx.float32 [| 4096 |])).(646)
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
          near ~var:8. 1. (variance xs);
          ks (fun x -> -.Float.expm1 (-.x)) xs);
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

(* [log Γ(x)] for [x > 0], by Lanczos' approximation of [g = 7] and nine
   terms, of relative error below 10^-15, and below 1/2 by the reflection
   [Γ(x) Γ(1 - x) = π / sin (π x)]. *)
let rec log_gamma x =
  if x < 0.5 then log (Float.pi /. sin (Float.pi *. x)) -. log_gamma (1. -. x)
  else
    let c =
      [|
        0.99999999999980993; 676.5203681218851; -1259.1392167224028;
        771.32342877765313; -176.61502916214059; 12.507343278686905;
        -0.13857109526572012; 9.9843695780195716e-6; 1.5056327351493116e-7;
      |]
    in
    let x = x -. 1. in
    let t = x +. 7.5 in
    let sum = ref c.(0) in
    for i = 1 to 8 do
      sum := !sum +. (c.(i) /. (x +. Float.of_int i))
    done;
    (0.5 *. log (2. *. Float.pi)) +. ((x +. 0.5) *. log t) -. t +. log !sum

(* The regularized lower incomplete gamma [P(a, x)], the CDF of a gamma of
   concentration [a] at [x > 0] given with its logarithm [lx], so that [x]
   may lie below the least float: below [a + 1] by its series
   [x^a e^-x / Γ(a + 1) Σ x^k / ((a + 1) … (a + k))], above as [1 - Q(a, x)],
   [Q]'s continued fraction evaluated by Lentz's method, each to the
   float64's epsilon. *)
let gamma_p a x lx =
  let eps = epsilon_float in
  if x < a +. 1. then begin
    let term = ref 1. and sum = ref 1. and k = ref 1. in
    while Float.abs !term > Float.abs !sum *. eps do
      term := !term *. x /. (a +. !k);
      sum := !sum +. !term;
      k := !k +. 1.
    done;
    !sum *. exp ((a *. lx) -. x -. log_gamma (a +. 1.))
  end
  else begin
    let tiny = 1e-300 in
    let b = ref (x +. 1. -. a) in
    let c = ref (1. /. tiny) and d = ref (1. /. !b) in
    let h = ref !d and i = ref 1. and fin = ref false in
    while not !fin do
      let an = -. !i *. (!i -. a) in
      b := !b +. 2.;
      d := (an *. !d) +. !b;
      if Float.abs !d < tiny then d := tiny;
      c := !b +. (an /. !c);
      if Float.abs !c < tiny then c := tiny;
      d := 1. /. !d;
      let del = !d *. !c in
      h := !h *. del;
      i := !i +. 1.;
      fin := Float.abs (del -. 1.) <= eps
    done;
    1. -. exp (log !h -. x +. (a *. lx) -. log_gamma a)
  end

(* A gamma of concentration [a] has mean [a], variance [a] and central
   moments [μ4 = 3a² + 6a] and [μ6 = 15a³ + 130a² + 120a], so its sample
   variance has variance [v / n], [v = 2a² + 6a], and skewness
   [(8a³ + 112a² + 120a) / (v^(3/2) √n)]. Six standard errors bound the
   mean, and the variance where its skewness is at most 1/4. Below that,
   about [a < 0.07] at [n = 2^14], the sample variance's upper tail is that
   of a few large draws, far heavier than a normal's: at [a = 0.01] six
   standard errors are passed in about one sample of 2·10^4. *)
let gamma_moments a xs =
  Array.iter
    (fun x -> if not (x >= 0. && Float.is_finite x) then failf "draw %g" x)
    xs;
  near ~var:a a (mean xs);
  let v = (2. *. a *. a) +. (6. *. a) in
  let mu = (8. *. a *. a *. a) +. (112. *. a *. a) +. (120. *. a) in
  if mu /. (v *. sqrt v *. sqrt (Float.of_int n)) <= 0.25 then
    near ~var:v a (variance xs)

(* Gamma draws rounded to floats of [bits] bits, 32 or 64, follow the
   gamma's CDF rounded too: a draw [x] holds the mass between the midpoints
   to its neighbours, from 0 for the least. Their empirical CDF, ties
   included, is within [ε = √(ln (2 / δ) / 2n)] of that CDF at every cell's
   ends but with probability [δ = 2·10^-9], by the Dvoretzky-Kiefer-Wolfowitz
   inequality with Massart's constant, which holds for every distribution.
   It is a test of power at every concentration: at [a = 0.01] a draw of
   [a = 0.012] moves the CDF by about 0.07 near 0, three times [ε]. A
   float64 draw's cell is taken as the draw itself, but 0's, which ends at
   half the least float64, [2^-1075].

   The cells' ends increase, so above a concentration of 100, where the
   series and the continued fraction take [O(√a)] terms each, the CDF at
   each end is the last one's plus Simpson's rule for the density over the
   step between them: draws lie about [√a / 2000] apart, and the density
   varies on the scale of its standard deviation [√a]. *)
let gamma_cdf ~bits a xs =
  let xs = Array.copy xs in
  Array.sort Float.compare xs;
  let nf = Float.of_int n in
  let next x d = Int32.float_of_bits (Int32.add (Int32.bits_of_float x) d) in
  let lg = log_gamma a in
  let density t = exp (((a -. 1.) *. log t) -. t -. lg) in
  (* The CDF at an end [x] given with its logarithm, after the last end
     [prev] where it was [cdf]. *)
  let prev = ref 0. and cdf = ref 0. in
  let at x lx =
    (if x = 0. && lx = neg_infinity then cdf := 0.
     else if a <= 100. || !prev = 0. then cdf := gamma_p a x lx
     else
       let h = x -. !prev in
       cdf :=
         !cdf
         +. h /. 6.
            *. (density !prev +. (4. *. density (!prev +. (h /. 2.)))
               +. density x));
    prev := x;
    !cdf
  in
  let p x = if x = 0. then at 0. neg_infinity else at x (log x) in
  (* The chance of a rounded draw below [x], and of one at most [x]. *)
  let below x =
    if bits = 32 && x > 0. then p ((next x (-1l) +. x) /. 2.) else p x
  in
  let through x =
    if bits = 32 then p ((x +. next x 1l) /. 2.)
    else if x = 0. then at 0. (log (Float.ldexp 1. (-1074)) -. log 2.)
    else p x
  in
  let d = ref 0. and i = ref 0 in
  while !i < n do
    let x = xs.(!i) and j = ref (!i + 1) in
    while !j < n && xs.(!j) = x do
      incr j
    done;
    d := Float.max !d (Float.abs ((Float.of_int !i /. nf) -. below x));
    d := Float.max !d (Float.abs ((Float.of_int !j /. nf) -. through x));
    i := !j
  done;
  at_most float_exact ~than:(sqrt (log 1e9 /. (2. *. nf))) !d

(* A gamma of concentration [a] has Laplace transform
   [E e^(-sx) = (1 + s)^-a]. Each [e^(-sx)] lies in [0, 1], of variance
   [w = (1 + 2s)^-a - (1 + s)^-2a], so by Bernstein's inequality the mean of
   [n] of them is within [√(2 w L / n) + 2 L / (3n)], [L = ln (2 / δ)], of
   the transform but with probability [δ = 2·10^-9]: a check of the whole
   distribution that no few large draws upset, at [s = 1 / (1 + a)] and
   [4 / (1 + a)], where the transform is between [e^-4] and 1. *)
let gamma_laplace a xs =
  let nf = Float.of_int n and l = log 1e9 in
  List.iter
    (fun s ->
      let want = (1. +. s) ** -.a in
      let w =
        Float.max 0. (((1. +. (2. *. s)) ** -.a) -. ((1. +. s) ** (-2. *. a)))
      in
      let within = sqrt (2. *. w *. l /. nf) +. (2. *. l /. (3. *. nf)) in
      let got = mean (Array.map (fun x -> exp (-.s *. x)) xs) in
      at_most float_exact ~than:within (Float.abs (got -. want)))
    [ 1. /. (1. +. a); 4. /. (1. +. a) ]

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
      (* The CDF the gamma law checks against, to 10^-8 of mpmath's values
         at 40 digits. *)
      cases
        ~name:(fun (a, x, _) -> Printf.sprintf "P(%g, %g)" a x)
        "gamma's CDF is the regularized lower incomplete gamma"
        [
          (0.01, 1e-30, 0.50404727285999545286);
          (0.01, 0.5, 0.99437324380603281585);
          (0.01, 3., 0.99986702864342984452);
          (0.25, 0.1, 0.60833884572896606698);
          (1., 1., 0.6321205588285576784);
          (2.5, 2., 0.45058404864721976739);
          (30., 25., 0.1821039159774551098);
          (30., 40., 0.95677131784826443307);
          (1e6, 999000., 0.15865521357430365246);
          (1e6, 1001500., 0.93313889576410223751);
        ]
        (fun (a, x, want) ->
          at_most float_exact ~than:1e-8
            (Float.abs (gamma_p a x (log x) -. want)));
      prop "gamma: mean and variance a, Laplace transform and CDF"
        (Gen.triple keys wide
           (params
              [ Least; V 1e-30; V 0.01; V 0.25; V 1.; V 2.5; V 30.; V 1e6 ]))
        (fun (k, F dt, a) ->
          let a = at dt a in
          let xs = read (Rng.gamma ~key:k (full dt a)) in
          gamma_moments a xs;
          gamma_laplace a xs;
          gamma_cdf ~bits:(A.Dtype.bits dt) a xs);
      (* beta's draw j is x = 1 / (1 + exp d), d = log G(b) - log G(a), for
         the gammas 2j and 2j + 1 its key draws from a and b interleaved. d is a sum of logarithms, each
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
          let ab = Array.init (2 * n) (fun i -> if i mod 2 = 0 then a else b) in
          let g = read (Rng.gamma ~key:k (host_of Nx.float64 [| 2 * n |] ab)) in
          let ga = Array.init n (fun i -> g.(2 * i)) in
          let gb = Array.init n (fun i -> g.((2 * i) + 1)) in
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

(* Inversions and normalisations *)

let inv_sqrt_2pi = 1. /. sqrt (2. *. Float.pi)

let pdf x =
  if Float.is_finite x then inv_sqrt_2pi *. exp (-0.5 *. x *. x) else 0.

(* The standard normal's masses below and above [x], each to its relative
   precision. *)
let below x = 0.5 *. Float.erfc (-.x /. sqrt 2.)
let above x = 0.5 *. Float.erfc (x /. sqrt 2.)

(* The standard normal conditioned on [[a, b]], [a <= b]: its mean, variance
   and CDF, from the masses on the side of 0 that keeps their digits. *)
let truncated a b =
  let right = a >= 0. in
  let z = if right then above a -. above b else below b -. below a in
  let xpdf x = if Float.is_finite x then x *. pdf x else 0. in
  let mean = (pdf a -. pdf b) /. z in
  let var = 1. +. ((xpdf a -. xpdf b) /. z) -. (mean *. mean) in
  let cdf x =
    if x <= a then 0.
    else if x >= b then 1.
    else if right then (above a -. above x) /. z
    else (below x -. below a) /. z
  in
  (mean, var, cdf)

let pp_bounds ppf (a, b) = Format.fprintf ppf "[%g, %g]" a b

let intervals =
  Gen.of_list ~pp:pp_bounds
    [
      (Float.neg_infinity, Float.infinity);
      (0., Float.infinity);
      (-1., 2.);
      (3., -1.);
      (-0.5, 0.5);
      (2., 3.);
      (5., Float.infinity);
      (Float.neg_infinity, -6.);
      (15., 16.);
      (-40., -38.);
    ]

(* A row of [k] concentrations, [x] each. *)
let rows dt k x = host_of dt [| n; k |] (Array.make (n * k) x)

(* [xs] as rows of [k]. *)
let rows_of k xs =
  Array.init (Array.length xs / k) (fun i -> Array.sub xs (i * k) k)

let others =
  group "inversions and normalisations"
    [
      prop "truncated_normal draws lie in their interval"
        (Gen.triple keys floats
           (Gen.of_list ~pp:pp_bounds
              [
                (Float.neg_infinity, Float.infinity);
                (-1., 2.);
                (2., -1.);
                (0.5, 0.5);
                (40., Float.infinity);
                (Float.neg_infinity, -40.);
                (30., 30.5);
              ]))
        (fun (k, (F dt, _), (a, b)) ->
          let a = A.Dtype.of_float dt a and b = A.Dtype.of_float dt b in
          let lo = Float.min a b and hi = Float.max a b in
          let xs =
            read
              (Rng.truncated_normal ~key:k (host_of dt [||] [| a |])
                 (host_of dt [| n |] (Array.make n b)))
          in
          Array.iter
            (fun x -> if not (x >= lo && x <= hi) then failf "draw %h" x)
            xs);
      (* A sample's KS statistic against the conditioned CDF, and its mean
         within 6 standard errors, for intervals on both sides and far out,
         where float32's masses underflow. *)
      prop "truncated_normal: mean and CDF of the conditioned normal"
        (Gen.triple keys wide intervals) (fun (k, F dt, (a, b)) ->
          let lo = Float.min a b and hi = Float.max a b in
          let xs =
            read
              (Rng.truncated_normal ~key:k
                 (host_of dt [| n |] (Array.make n a))
                 (host_of dt [| n |] (Array.make n b)))
          in
          let want, var, cdf = truncated lo hi in
          near ~var want (mean xs);
          ks cdf xs);
      (* A float32 draw against the quantile of its own uniform [u], floored at
         2^-24, formed in float64 from the smaller of the masses below and above
         it: within 6 ε + 3 ulps of float32. *)
      prop "a float32 truncated_normal draw inverts its uniform"
        (Gen.pair keys
           (Gen.of_list ~pp:pp_bounds
              [
                (Float.neg_infinity, Float.infinity);
                (0., Float.infinity);
                (-1., 2.);
                (-0.5, 0.5);
                (2., 3.);
                (5., Float.infinity);
                (Float.neg_infinity, -6.);
                (15., 16.);
              ]))
        (fun (k, (a, b)) ->
          let full v = host_of Nx.float32 [| n |] (Array.make n v) in
          let xs = read (Rng.truncated_normal ~key:k (full a) (full b)) in
          let us = read (Rng.uniform ~key:k Nx.float32 [| n |]) in
          let us = Array.map (Float.max (Float.ldexp 1. (-24))) us in
          let right u = ((1. -. u) *. above a) +. (u *. above b) in
          let left u = ((1. -. u) *. below a) +. (u *. below b) in
          let qs = Array.map (fun u -> Float.min (right u) (left u)) us in
          let ys = read (Nx.ndtri (host_of Nx.float64 [| n |] qs)) in
          let eps = Float.ldexp 1. (-23) in
          let worst = ref 0. in
          Array.iteri
            (fun i x ->
              let u = us.(i) in
              let want = if right u < left u then -.ys.(i) else ys.(i) in
              let want =
                Float.min (Float.max want (Float.min a b)) (Float.max a b)
              in
              let ulp = Float.ldexp eps (snd (Float.frexp want) - 1) in
              let bound = (6. *. eps) +. (3. *. ulp) in
              worst := Float.max !worst (Float.abs (x -. want) /. bound))
            xs;
          at_most float_exact ~than:1. !worst);
      prop "dirichlet rows sum to 1, each with an element of at least 1 / k"
        (Gen.triple keys wide
           (Gen.pair (Gen.int_range 1 5)
              (params [ Least; V 1e-30; V 0.01; V 0.5; V 1.; V 30.; V 1e6 ])))
        (fun (key, F dt, (k, a)) ->
          let a = at dt a in
          let eps = (A.Dtype.float_format dt).epsilon in
          let xs = read (Rng.dirichlet ~key (rows dt k a)) in
          Array.iter
            (fun row ->
              let sum = Array.fold_left ( +. ) 0. row in
              let top = Array.fold_left Float.max 0. row in
              Array.iter
                (fun x -> if not (x >= 0. && x <= 1.) then failf "element %h" x)
                row;
              at_most float_exact
                ~than:(Float.of_int (k + 2) *. eps)
                (Float.abs (sum -. 1.));
              at_least float_exact
                ~than:((1. -. (4. *. eps)) /. Float.of_int k)
                top)
            (rows_of k xs));
      (* Component i of Dirichlet(a) has mean a_i / s and variance a_i (s - a_i)
         / (s^2 (s + 1)), s the concentrations' sum. *)
      prop ~count:few "dirichlet: each component's mean"
        (Gen.pair keys
           (Gen.of_list
              ~pp:(fun ppf a ->
                Format.fprintf ppf "[%s]"
                  (String.concat "; "
                     (Array.to_list (Array.map string_of_float a))))
              [
                [| 1.; 1.; 1. |];
                [| 0.3; 2.; 5. |];
                [| 0.05; 0.05 |];
                [| 40.; 2. |];
              ]))
        (fun (key, a) ->
          let k = Array.length a in
          let conc =
            host_of Nx.float64 [| n; k |]
              (Array.concat (List.init n (fun _ -> a)))
          in
          let xs = rows_of k (read (Rng.dirichlet ~key conc)) in
          let s = Array.fold_left ( +. ) 0. a in
          Array.iteri
            (fun i ai ->
              let var = ai *. (s -. ai) /. (s *. s *. (s +. 1.)) in
              near ~var (ai /. s) (mean (Array.map (fun row -> row.(i)) xs)))
            a);
      (* Dirichlet draw j is gamma draw j of the same key, divided by its row's
         sum: each logarithm is rounded, so each element lies within a few ulps
         of the logarithms' magnitudes of the ratio formed in float64 from the
         gamma draws. *)
      prop ~count:few "dirichlet is the gammas' row ratios"
        (Gen.pair keys
           (Gen.of_list
              ~pp:(fun ppf a ->
                Format.fprintf ppf "[%s]"
                  (String.concat "; "
                     (Array.to_list (Array.map string_of_float a))))
              [ [| 0.5; 0.5 |]; [| 0.3; 4.; 2. |]; [| 9.; 0.7; 1.; 1. |] ]))
        (fun (key, a) ->
          let k = Array.length a in
          let conc =
            host_of Nx.float64 [| n; k |]
              (Array.concat (List.init n (fun _ -> a)))
          in
          let g = rows_of k (read (Rng.gamma ~key conc)) in
          let d = rows_of k (read (Rng.dirichlet ~key conc)) in
          let worst = ref 0. in
          Array.iteri
            (fun r row ->
              let sum = Array.fold_left ( +. ) 0. row in
              let logs =
                Array.fold_left
                  (fun m x -> Float.max m (Float.abs (log x)))
                  0. row
              in
              Array.iteri
                (fun i gi ->
                  let want = gi /. sum in
                  let bound =
                    (16. *. epsilon_float *. want *. (2. +. (2. *. logs)))
                    +. (4. *. (Float.succ want -. want))
                  in
                  worst :=
                    Float.max !worst (Float.abs (d.(r).(i) -. want) /. bound))
                row)
            g;
          at_most float_exact ~than:1. !worst);
      test "dirichlet of no component is empty, of one is 1" (fun () ->
          let k = Rng.key 401 in
          equal (array float_exact) [||]
            (read (Rng.dirichlet ~key:k (host_of Nx.float32 [| 3; 0 |] [||])));
          equal (array float_exact) [| 1.; 1. |]
            (read
               (Rng.dirichlet ~key:k
                  (host_of Nx.float32 [| 2; 1 |] [| 0.2; 7. |]))));
      (* Each category's frequency within 6 standard errors of its probability,
         a logit of -inf never. *)
      prop "categorical: frequencies follow the softmax"
        (Gen.triple keys
           (Gen.of_list ~pp:pp_dtype
              [ F Nx.float32; F Nx.float64; F Nx.float16 ])
           (Gen.of_list ~pp:(fun ppf a -> Format.pp_print_int ppf a) [ 0; 1 ]))
        (fun (key, F dt, axis) ->
          let logits = [| 0.; log 2.; log 3.; Float.neg_infinity; -1. |] in
          let k = Array.length logits in
          let rows =
            host_of dt [| n; k |] (Array.concat (List.init n (fun _ -> logits)))
          in
          let rows = if axis = 0 then Nx.transpose rows else rows in
          let axis = if axis = 0 then 0 else -1 in
          let draws = read (Rng.categorical ~key ~axis rows) in
          equal int n (Array.length draws);
          let counts = Array.make k 0 in
          Array.iter
            (fun i -> counts.(Int64.to_int i) <- counts.(Int64.to_int i) + 1)
            draws;
          let ws = Array.map (fun l -> exp (A.Dtype.of_float dt l)) logits in
          let total = Array.fold_left ( +. ) 0. ws in
          Array.iteri
            (fun i w ->
              let p = w /. total in
              near
                ~var:(p *. (1. -. p))
                p
                (Float.of_int counts.(i) /. Float.of_int n))
            ws);
      prop "permutation is an ordering of 0 to n - 1"
        (Gen.pair keys (Gen.int_range 0 70))
        (fun (key, m) ->
          cover "none" (m = 0);
          let xs = read (Rng.permutation ~key m) in
          let sorted = Array.copy xs in
          Array.sort Int64.compare sorted;
          equal (array int64) (Array.init m Int64.of_int) sorted);
      (* Each of the 6 orderings of 3 within 6 standard errors of 1/6, over a
         batch of keys. *)
      prop ~count:few "permutation: orderings are equally likely" keys
        (fun key ->
          let xs = read (Rng.permutation ~key:(Rng.split_batch ~n key) 3) in
          let counts = Hashtbl.create 6 in
          for i = 0 to n - 1 do
            let o = (xs.(3 * i), xs.((3 * i) + 1)) in
            Hashtbl.replace counts o
              (1 + Option.value ~default:0 (Hashtbl.find_opt counts o))
          done;
          equal int 6 (Hashtbl.length counts);
          Hashtbl.iter
            (fun _ c ->
              let f = Float.of_int c /. Float.of_int n in
              near ~var:(5. /. 36.) (1. /. 6.) f)
            counts);
      cases
        ~name:(fun (name, _, _) -> name)
        "refusals at the call"
        [
          ( "truncated_normal of a NaN bound",
            (fun () ->
              Rng.truncated_normal
                (host_of Nx.float64 [| 2 |] [| 0.; Float.nan |])
                (host_of Nx.float64 [||] [| 1. |])),
            "Nx.Rng.truncated_normal: lower at [1] is nan, not in [-inf, inf]"
          );
          ( "truncated_normal of bounds that do not broadcast",
            (fun () ->
              Rng.truncated_normal
                (host_of Nx.float64 [| 2 |] [| 0.; 1. |])
                (host_of Nx.float64 [| 3 |] [| 1.; 2.; 3. |])),
            "Nx.Rng.truncated_normal: shapes [2] and [3] do not broadcast" );
          ( "dirichlet of no axis",
            (fun () -> Rng.dirichlet (host_of Nx.float64 [||] [| 1. |])),
            "Nx.Rng.dirichlet: concentrations of shape [] have no axis" );
          ( "dirichlet of a zero concentration",
            (fun () ->
              Rng.dirichlet (host_of Nx.float64 [| 1; 2 |] [| 1.; 0. |])),
            "Nx.Rng.dirichlet: a at [0; 1] is 0, not in (0, inf)" );
        ]
        (fun (_, f, want) ->
          equal string want (message (fun () -> ignore (f ()))));
      cases
        ~name:(fun (name, _, _) -> name)
        "categorical refusals at the call"
        [
          ( "an axis it lacks",
            (fun () ->
              Rng.categorical ~axis:2
                (host_of Nx.float32 [| 2; 3 |] (Array.make 6 0.))),
            "Nx.Rng.categorical: 2 is not an axis of float32 [2; 3]" );
          ( "an empty axis",
            (fun () -> Rng.categorical (host_of Nx.float32 [| 2; 0 |] [||])),
            "Nx.Rng.categorical: axis -1 of float32 [2; 0] has no category" );
          ( "an infinite logit",
            (fun () ->
              let inf = Float.infinity in
              let logits = [| 0.; 1.; 0.; inf |] in
              Rng.categorical (host_of Nx.float32 [| 2; 2 |] logits)),
            "Nx.Rng.categorical: the largest logit at [1] is inf, not in \
             (-inf, inf)" );
          ( "a NaN logit",
            (fun () ->
              Rng.categorical
                (host_of Nx.float32 [| 1; 2 |] [| 0.; Float.nan |])),
            "Nx.Rng.categorical: the largest logit at [0] is nan, not in \
             (-inf, inf)" );
          ( "a row of -inf",
            (fun () ->
              Rng.categorical
                (host_of Nx.float32 [| 1; 2 |]
                   [| Float.neg_infinity; Float.neg_infinity |])),
            "Nx.Rng.categorical: the largest logit at [0] is -inf, not in \
             (-inf, inf)" );
          ( "permutation of a negative n",
            (fun () -> Rng.permutation (-1)),
            "Nx.Rng.permutation: n is -1, not at least 0" );
        ]
        (fun (_, f, want) ->
          equal string want (message (fun () -> ignore (f ()))));
    ]
(* Parameter checks: a sampler names the first parameter outside its domain, in
   C order, whatever the dtype, shape and placement: at the call on the host,
   and on another set at the next wait there, the draw itself waiting for
   nothing. *)

(* The index of the flat position [i] of a value of shape [s], in C order. *)
let index s i =
  let idx = Array.make (Array.length s) 0 and k = ref i in
  for a = Array.length s - 1 downto 0 do
    idx.(a) <- !k mod s.(a);
    k := !k / s.(a)
  done;
  idx

(* A parameter's shape, whether it is also checked split over [S2] along its
   first axis, doubled to divide evenly, and two flat positions in it, which may
   coincide. *)
let bad_params =
  Gen.triple
    (Gen.array ~size:(Gen.int_range 0 4) (Gen.int_range 1 4))
    Gen.bool (Gen.pair Gen.nat Gen.nat)
  |> Gen.map (fun (s, split, (i, j)) ->
      let split = split && Array.length s > 0 in
      let s =
        if split then Array.mapi (fun a e -> if a = 0 then 2 * e else e) s
        else s
      in
      let n = numel s in
      (s, split, (i mod n, j mod n)))

let split x = Nx.place (S2.split ~axis:0) x

(* The message naming position [i] of shape [s] for [name]'s value [v]. *)
let refusal by name s i v text =
  Format.asprintf "%s: %s at %a is %s, not in %s" by name pp_shape (index s i) v
    text

let checks =
  group "parameter checks"
    [
      prop "bernoulli names its first p outside [0, 1]"
        (Gen.triple floats bad_params (floats_of [ -1.; 1.5; 4. ]))
        (fun ((F dt, _), (s, on_s2, (i, j)), v) ->
          cover "split" on_s2;
          cover "two outside" (i <> j);
          cover "rank 0" (Array.length s = 0);
          cover "rank above 1" (Array.length s > 1);
          let p =
            host_of dt s
              (Array.init (numel s) (fun k ->
                   A.Dtype.of_float dt (if k = i || k = j then v else 0.5)))
          in
          let want =
            refusal "Nx.Rng.bernoulli" "p" s (Int.min i j)
              (Printf.sprintf "%g" v) "[0, 1]"
          in
          equal string want (message (fun () -> Rng.bernoulli p));
          if on_s2 then begin
            let x = Rng.bernoulli (split p) in
            equal string want (message (fun () -> Nx.sync x))
          end);
      prop "binomial names its first negative n" bad_params
        (fun (s, on_s2, (i, j)) ->
          cover "split" on_s2;
          cover "two outside" (i <> j);
          let n =
            host_of Nx.int32 s
              (Array.init (numel s) (fun k ->
                   if k = i || k = j then -2l else 3l))
          in
          let p = host_of Nx.float32 s (Array.make (numel s) 0.5) in
          let want =
            refusal "Nx.Rng.binomial" "n" s (Int.min i j) "-2" "[0, inf)"
          in
          equal string want (message (fun () -> Rng.binomial n p));
          if on_s2 then begin
            let x = Rng.binomial (split n) (split p) in
            equal string want (message (fun () -> Nx.sync x))
          end);
      (let on xs = Nx.place S2.on (host_of Nx.float32 [| 2 |] xs) in
       (* The draw returns; the wait on its set raises. *)
       let waits x () = Nx.sync x in
       let good = on [| 0.5; 0.5 |] in
       cases
         ~name:(fun (name, _, _) -> name)
         "a draw on another set checks its parameters at the next wait"
         [
           ( "bernoulli",
             (fun () -> waits (Rng.bernoulli (on [| 0.5; 2. |]))),
             "Nx.Rng.bernoulli: p at [1] is 2, not in [0, 1]" );
           ( "gamma",
             (fun () -> waits (Rng.gamma (on [| 1.; 0. |]))),
             "Nx.Rng.gamma: a at [1] is 0, not in (0, inf)" );
           ( "beta",
             (fun () -> waits (Rng.beta good (on [| -1.; 1. |]))),
             "Nx.Rng.beta: b at [0] is -1, not in (0, inf)" );
           ( "von_mises",
             (fun () -> waits (Rng.von_mises (on [| Float.nan; 1. |]))),
             "Nx.Rng.von_mises: k at [0] is nan, not in [0, inf)" );
           ( "poisson",
             (fun () -> waits (Rng.poisson (on [| 1.; -3. |]))),
             "Nx.Rng.poisson: rate at [1] is -3, not in [0, 2^31)" );
           ( "binomial",
             (fun () ->
               let n = host_of Nx.int32 [| 2 |] [| -1l; 2l |] in
               let n = Nx.place S2.on n in
               waits (Rng.binomial n good)),
             "Nx.Rng.binomial: n at [0] is -1, not in [0, inf)" );
           ( "truncated_normal",
             (fun () ->
               waits (Rng.truncated_normal good (on [| 1.; Float.nan |]))),
             "Nx.Rng.truncated_normal: upper at [1] is nan, not in \
              [-inf, inf]" );
           ( "dirichlet",
             (fun () -> waits (Rng.dirichlet (on [| 1.; -2. |]))),
             "Nx.Rng.dirichlet: a at [1] is -2, not in (0, inf)" );
           ( "categorical",
             (fun () -> waits (Rng.categorical (on [| 0.; Float.infinity |]))),
             "Nx.Rng.categorical: the largest logit at [] is inf, not in \
              (-inf, inf)" );
         ]
         (fun (_, draw, want) ->
           let wait = draw () in
           equal string want (message wait)));
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

(* Names *)

(* The [by]s a rule receives while [f] runs under it, each once, sorted. *)
let observed_names f =
  let names = ref [] in
  let observe _ ~by op =
    if not (List.mem by !names) then names := by :: !names;
    Nx.Prim.eval ~by op
  in
  Nx.Prim.interpret ~name:"test.names" Extent observe (fun _ -> ignore (f ()));
  List.sort String.compare !names

(* [x] dead: donated to [Nx.copy], which consumed it. *)
let kill x =
  ignore (Nx.copy (Nx.donate x));
  x

(* Each sampler as [name, (operands, make)]: [make ~dead] makes its operands,
   the first dead when [dead], and gives the draw from a key, or keyless at
   [None]. *)
let calls =
  let f64 = Nx.float64 in
  let arg ~dead x = if dead then kill x else x in
  let floats ~dead xs = arg ~dead (host_of f64 [| Array.length xs |] xs) in
  let pos ~dead = floats ~dead [| 0.5; 2. |] in
  let unit ~dead = floats ~dead [| 0.25; 0.75 |] in
  let none f = (0, fun ~dead:_ -> f) in
  [
    ("bits", none (fun key -> ignore (Rng.bits ?key [| 3 |])));
    ("uniform", none (fun key -> ignore (Rng.uniform ?key f64 [| 3 |])));
    ("normal", none (fun key -> ignore (Rng.normal ?key f64 [| 3 |])));
    ( "exponential",
      none (fun key -> ignore (Rng.exponential ?key f64 [| 3 |])) );
    ("randint", none (fun key -> ignore (Rng.randint ?key ~high:5 [| 3 |])));
    ("permutation", none (fun key -> ignore (Rng.permutation ?key 4)));
    ( "bernoulli",
      ( 1,
        fun ~dead ->
          let p = unit ~dead in
          fun key -> ignore (Rng.bernoulli ?key p) ) );
    ( "gamma",
      ( 1,
        fun ~dead ->
          let a = pos ~dead in
          fun key -> ignore (Rng.gamma ?key a) ) );
    ( "beta",
      ( 2,
        fun ~dead ->
          let a = pos ~dead and b = pos ~dead:false in
          fun key -> ignore (Rng.beta ?key a b) ) );
    ( "von_mises",
      ( 1,
        fun ~dead ->
          let k = pos ~dead in
          fun key -> ignore (Rng.von_mises ?key k) ) );
    ( "poisson",
      ( 1,
        fun ~dead ->
          let rate = pos ~dead in
          fun key -> ignore (Rng.poisson ?key rate) ) );
    ( "binomial",
      ( 2,
        fun ~dead ->
          let n = arg ~dead (host_of Nx.int32 [| 2 |] [| 3l; 5l |]) in
          let p = unit ~dead:false in
          fun key -> ignore (Rng.binomial ?key n p) ) );
    ( "truncated_normal",
      ( 2,
        fun ~dead ->
          let lower = floats ~dead [| -1.; 0. |] in
          let upper = floats ~dead:false [| 1.; 2. |] in
          fun key -> ignore (Rng.truncated_normal ?key lower upper) ) );
    ( "dirichlet",
      ( 1,
        fun ~dead ->
          let a = pos ~dead in
          fun key -> ignore (Rng.dirichlet ?key a) ) );
    ( "categorical",
      ( 1,
        fun ~dead ->
          let logits = pos ~dead in
          fun key -> ignore (Rng.categorical ?key logits) ) );
  ]

(* Keyless, unscoped and in a scope, and with a key: each as [name, (made,
   under)], [made] the functions that made its key, which a rule receives when
   the key, a value of every set, is computed under it. *)
let keyings =
  let k = Rng.key 301 in
  let placed = Rng.place Nx.Host.on k in
  [
    ("unscoped", ([], fun draw -> draw None));
    ( "scoped",
      ([ "Nx.Rng.key" ], fun draw -> Rng.with_key k (fun () -> draw None)) );
    ("keyed", ([], fun draw -> draw (Some placed)));
  ]

(* [f by keying made operands make under] for each sampler [by] and keying. *)
let each_sampler f =
  List.iter
    (fun (name, (operands, make)) ->
      List.iter
        (fun (keying, (made, under)) ->
          f ("Nx.Rng." ^ name) keying made operands make under)
        keyings)
    calls

let names =
  group "names"
    [
      test "every operation a sampler makes reaches a rule as the sampler"
        (fun () ->
          each_sampler (fun by keying made _ make under ->
              let draw = make ~dead:false in
              equal ~msg:(by ^ ", " ^ keying) (list string)
                (List.sort String.compare (by :: made))
                (observed_names (fun () -> under draw))));
      test "a dead first operand names the sampler and the operand" (fun () ->
          each_sampler (fun by keying _ operands make under ->
              if operands > 0 then
                let draw = make ~dead:true in
                let want = by ^ ": operand 1 was donated to Nx.copy" in
                raises ~msg:(by ^ ", " ^ keying) (Invalid_argument want)
                  (fun () -> under draw)));
    ]

(* Goldens *)

(* Each sampler's first draws under key 0, as exact bits: a baseline that a
   change to the generator's layout or a sampler's arithmetic moves. *)
let goldens =
  test "each sampler's first draws under key 0" (fun () ->
      let k = Rng.key 0 and s = [| 4 |] in
      let full dt v = host_of dt s (Array.make 4 v) in
      let counts v = host_of Nx.int32 s (Array.make 4 v) in
      let floats name xs =
        Printf.printf "%s:%s\n" name
          (String.concat "" (Array.to_list (Array.map (Printf.sprintf " %h") xs)))
      in
      let ints name xs =
        Printf.printf "%s:%s\n" name
          (String.concat ""
             (Array.to_list (Array.map (Printf.sprintf " %ld") xs)))
      in
      ints "bits" (read (Rng.bits ~key:k s));
      floats "uniform f16" (read (Rng.uniform ~key:k Nx.float16 s));
      floats "uniform f32" (read (Rng.uniform ~key:k Nx.float32 s));
      floats "uniform f64" (read (Rng.uniform ~key:k Nx.float64 s));
      floats "normal f32" (read (Rng.normal ~key:k Nx.float32 s));
      floats "normal f64" (read (Rng.normal ~key:k Nx.float64 s));
      floats "exponential f32" (read (Rng.exponential ~key:k Nx.float32 s));
      floats "exponential f64" (read (Rng.exponential ~key:k Nx.float64 s));
      ints "randint" (read (Rng.randint ~key:k ~low:(-9) ~high:1000 s));
      Printf.printf "bernoulli:%s\n"
        (String.concat ""
           (Array.to_list
              (Array.map
                 (fun b -> if b then " 1" else " 0")
                 (read (Rng.bernoulli ~key:k (full Nx.float32 0.3))))));
      floats "gamma f32" (read (Rng.gamma ~key:k (full Nx.float32 0.7)));
      floats "gamma f64" (read (Rng.gamma ~key:k (full Nx.float64 2.5)));
      floats "beta f64"
        (read (Rng.beta ~key:k (full Nx.float64 2.) (full Nx.float64 0.5)));
      floats "von_mises f32" (read (Rng.von_mises ~key:k (full Nx.float32 3.)));
      ints "poisson f32" (read (Rng.poisson ~key:k (full Nx.float32 4.)));
      ints "poisson f64" (read (Rng.poisson ~key:k (full Nx.float64 30.)));
      ints "binomial f32"
        (read (Rng.binomial ~key:k (counts 40l) (full Nx.float32 0.25)));
      ints "binomial f64"
        (read (Rng.binomial ~key:k (counts 10l) (full Nx.float64 0.3)));
      floats "truncated_normal f32"
        (read
           (Rng.truncated_normal ~key:k (full Nx.float32 (-1.))
              (full Nx.float32 2.)));
      floats "truncated_normal f64"
        (read
           (Rng.truncated_normal ~key:k (full Nx.float64 5.)
              (full Nx.float64 Float.infinity)));
      floats "dirichlet f64"
        (read (Rng.dirichlet ~key:k (full Nx.float64 0.5)));
      let positions name xs =
        Printf.printf "%s:%s\n" name
          (String.concat ""
             (Array.to_list (Array.map (Printf.sprintf " %Ld") xs)))
      in
      let logits = host_of Nx.float32 [| 4; 3 |] (Array.make 12 0.) in
      positions "categorical" (read (Rng.categorical ~key:k logits));
      positions "permutation" (read (Rng.permutation ~key:k 8));
      expect (output ()) @@ __POS_OF__ {|
        bits: 1797259609 -1715843330 1351547692 -1059176654
        uniform f16: 0x1.59p-3 0x1.bf8p-1 0x1.96p-2 0x1.cc8p-1
        uniform f32: 0x1.000ac8p-3 0x1.749dfcp-1 0x1.1df658p-1 0x1.bc7e64p-1
        uniform f64: 0x1.5999ba4efep-13 0x1.df65981bc7e64p-2 0x1.89bb3f05795ccp-3 0x1.789a20a6fd70cp-2
        normal f32: -0x1.22c006p-2 -0x1.027b5cp+1 0x1.7595cap-1 -0x1.9733b4p-1
        normal f64: -0x1.05cdacf43155ap+2 0x1.a8abde6ea3f96p-1 -0x1.397a40a15a62ep+0 0x1.574fd27204757p+0
        exponential f32: 0x1.11853ap-3 0x1.4d14e4p+0 0x1.a29f72p-1 0x1.03576p+1
        exponential f64: 0x1.59a104c24e8a4p-13 0x1.4348c112b2bcdp-1 0x1.b541fa2f2a75dp-3 0x1.d583d0aa77246p-2
        randint: 596 984 651 668
        bernoulli: 1 0 1 0
        gamma f32: 0x1.00f66cp-1 0x1.8618a8p-2 0x1.121b52p-2 0x1.34a13cp+2
        gamma f64: 0x1.d0cc931992a18p+1 0x1.339beb47338d6p+1 0x1.15078c6b06a44p-1 0x1.130a3f54a8a4bp+1
        beta f64: 0x1.8e9a8ced5c868p-1 0x1.d346b21cc65a2p-3 0x1.c9ec7abf3da78p-1 0x1.9faa399316489p-2
        von_mises f32: 0x1.b5cdd6p-4 0x1.5b2948p+0 0x1.aaf66ep-2 0x1.8a843ap-1
        poisson f32: 2 1 6 4
        poisson f64: 29 37 35 24
        binomial f32: 12 4 13 10
        binomial f64: 0 2 2 5
        truncated_normal f32: -0x1.47d256p-1 0x1.607708p-1 0x1.2db01ap-2 0x1.1f88c4p+0
        truncated_normal f64: 0x1.400085479d28p+2 0x1.47b476d59b514p+2 0x1.429fe288808d7p+2 0x1.459cab42bf8a3p+2
        dirichlet f64: 0x1.31936afa5218ep-6 0x1.b6c005bb54a9cp-2 0x1.e60e2452158e3p-9 0x1.192d53a63104cp-1
        categorical: 2 1 1 2
        permutation: 3 5 7 0 4 6 1 2
        |})

let () =
  exit
    (run "nx random"
       [
         known;
         keys_group;
         draws;
         distributions;
         rejection;
         others;
         checks;
         scope;
         names;
         goldens;
       ])
