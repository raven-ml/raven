(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx on a set whose kernels are nx.cuda's: Nx's contractions, reductions and
   scans of values placed on CUDA's GPU 0 compute there. A matmul or a sum of
   small integers, which every association computes exactly, gives the host's
   values; a float32 sum along rows, NaNs and infinities among its terms,
   nx.cpu's bits; and a float32 cumsum along rows, as nx.mli states it on one
   device, the prefix of its rows' cumsum for a prefix of them, bit for bit, and
   within its bound. Shapes of zero, one and many elements along each axis, rows
   on either side of a block of 1024 terms and of a chunk of 4096. Every test
   skips where CUDA opens no GPU. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype

let gpu =
  let t = lazy (Rig.open_ { (Rig_cuda.gpu 0) with name = "CUDA:nx" }) in
  fun () ->
    match Lazy.force t with Ok d -> d | Error why -> skip ~reason:why ()

let set =
  lazy
    (let d = gpu () in
     Nx.devices ~kernels:(module Nx_cuda) [ d ])

let host dt s xs =
  Nx.Repr.of_array Nx.Host.v (A.of_array dt (Iarray.of_array s) xs)

let read x =
  A.to_array (Option.get (Nx.Repr.array ~by:"test" (Nx.place Nx.Host.on x)))

let pp_ints sep =
  Format.(
    pp_print_list ~pp_sep:(fun p () -> pp_print_string p sep) pp_print_int)

(* Extents: none, one, a few, and past a tile. *)
let extent =
  Gen.frequency
    [
      (1, Gen.constant 0);
      (2, Gen.int_range 1 4);
      (2, Gen.int_range 5 70);
      (1, Gen.int_range 100 300);
    ]

(* [n] small integers from [seed], as floats: exact in any float32 sum of
   fewer than 2^22 of them. *)
let small n seed =
  let rs = Random.State.make [| seed |] in
  Array.init n (fun _ -> Float.of_int (Random.State.int rs 9 - 4))

let total s = Array.fold_left ( * ) 1 s

(* A matmul of [batch] (none, or that many) matrices m x k times k x n. *)
let matmuls =
  let open Gen in
  let* batch = of_list [ None; Some 0; Some 1; Some 3 ] in
  let* m = extent and+ k = extent and+ n = extent in
  let+ seed = int in
  (batch, m, k, n, seed)

let pp_matmul ppf (batch, m, k, n, seed) =
  Format.fprintf ppf "%s%dx%d by %dx%d (seed %d)"
    (match batch with None -> "" | Some b -> Printf.sprintf "%d of " b)
    m k k n seed

let law_matmul (batch, m, k, n, seed) =
  cover "an empty product" (m = 0 || k = 0 || n = 0 || batch = Some 0);
  cover "a batch" (Option.is_some batch);
  cover "past a tile" (m > 64 || n > 64 || k > 64);
  let lead = match batch with None -> [||] | Some b -> [| b |] in
  let sa = Array.append lead [| m; k |] and sb = Array.append lead [| k; n |] in
  let a = host D.Float32 sa (small (total sa) seed) in
  let b = host D.Float32 sb (small (total sb) (seed + 1)) in
  let module G = (val Lazy.force set) in
  let want = Nx.matmul a b in
  let got = Nx.matmul (Nx.place G.on a) (Nx.place G.on b) in
  equal (array int) (Nx.shape want) (Nx.shape got);
  equal (array float_exact) (read want) (read got)

(* A shape of rank 0 to 3 and the axes a sum reduces, from none to all;
   one time in three a shape with an empty axis that the sum reduces or
   keeps, which the others draw one time in 11 and 14. *)
let sums =
  let open Gen in
  let any =
    let* r = int_range 0 3 in
    let* s = array ~size:(constant r) extent in
    let* axes = subsequence (List.init r Fun.id) in
    let+ seed = int in
    (s, axes, seed)
  in
  let emptied =
    let* r = int_range 1 3 in
    let* s = array ~size:(constant r) extent and+ z = int_range 0 (r - 1) in
    let s = Array.mapi (fun a n -> if a = z then 0 else n) s in
    let with_z axes = List.sort_uniq compare (z :: axes) in
    let* axes =
      frequency
        [
          (1, constant []);
          (1, map with_z (subsequence (List.init r Fun.id)));
        ]
    in
    let+ seed = int in
    (s, axes, seed)
  in
  frequency [ (2, any); (1, emptied) ]

let pp_sum ppf (s, axes, seed) =
  Format.fprintf ppf "%a along [%a] (seed %d)" (pp_ints "x") (Array.to_list s)
    (pp_ints "; ") axes seed

let law_sum (s, axes, seed) =
  cover "an empty sum" (List.exists (fun a -> s.(a) = 0) axes);
  cover "an empty result" (Array.exists (( = ) 0) s && axes = []);
  cover "every axis" (Array.length s > 0 && List.length axes = Array.length s);
  let x = host D.Float32 s (small (total s) seed) in
  let module G = (val Lazy.force set) in
  let want = Nx.sum ~axes x and got = Nx.sum ~axes (Nx.place G.on x) in
  equal (array int) (Nx.shape want) (Nx.shape got);
  equal (array float_exact) (read want) (read got)

(* [rows] rows of [r] float32 terms in [-1, 1), -0 among them, and NaNs
   and infinities where [specials]. *)
let rows =
  let open Gen in
  let* rows = int_range 1 3 in
  let* r =
    frequency
      [
        (1, int_range 0 2);
        (2, int_range 3 1100);
        (2, int_range 4000 4200);
        (1, int_range 9000 10007);
      ]
  in
  let* specials = bool in
  let+ seed = int in
  (rows, r, specials, seed)

let pp_rows ppf (rows, r, specials, seed) =
  Format.fprintf ppf "%d rows of %d%s (seed %d)" rows r
    (if specials then ", NaNs and infinities among them" else "")
    seed

let terms (rows, r, specials, seed) =
  let rs = Random.State.make [| seed |] in
  Array.init (rows * r) (fun _ ->
      match Random.State.int rs 64 with
      | 0 when specials -> Float.nan
      | 1 when specials -> Float.infinity
      | 2 when specials -> Float.neg_infinity
      | 3 -> -0.
      | _ ->
          let x = Random.State.float rs 2. -. 1. in
          Int32.float_of_bits (Int32.bits_of_float x))

(* float32 bits, every NaN as one: a NaN's sign and payload are not
   promised. *)
let bits a =
  Array.map
    (fun x -> if Float.is_nan x then 0x7fc00000l else Int32.bits_of_float x)
    a

let law_sum_bits ((rows, r, specials, _) as c) =
  cover "rows past a block" (r > 1024);
  cover "rows past a chunk" (r > 4096);
  cover "a NaN or an infinity among the terms" specials;
  let module G = (val Lazy.force set) in
  let x = host D.Float32 [| rows; r |] (terms c) in
  equal (array int32)
    (bits (read (Nx.sum ~axes:[ 1 ] x)))
    (bits (read (Nx.sum ~axes:[ 1 ] (Nx.place G.on x))))

(* A row case and a prefix of its rows to scan alone: anywhere along the row, or
   one of its last three terms' ends one time in two, so that a prefix past a
   chunk of 4096 is drawn about one case in five. *)
let prefixes =
  let open Gen in
  let* ((_, r, _, _) as c) = rows in
  let+ p =
    frequency
      [
        (1, int_range 0 r); (1, map (fun j -> Int.max 0 (r - j)) (int_range 0 2));
      ]
  in
  (c, p)

let pp_prefix ppf (c, p) = Format.fprintf ppf "%a, its first %d" pp_rows c p

(* The first [p] terms of each of [c]'s rows. *)
let first p ((rows, r, _, _) as c) =
  let xs = terms c in
  host D.Float32 [| rows; p |]
    (Array.init (rows * p) (fun k -> xs.((k / p * r) + (k mod p))))

(* nx.mli's cumsum: on one device, the cumsum of a prefix of the axis is the
   prefix of the cumsum, bit for bit, NaNs as a class. *)
let law_prefix (((rows, r, _, _) as c), p) =
  cover "rows past a chunk" (r > 4096);
  cover "a prefix past a chunk" (p > 4096);
  let module G = (val Lazy.force set) in
  let whole =
    read
      (Nx.cumsum ~axis:1
         (Nx.place G.on (host D.Float32 [| rows; r |] (terms c))))
  in
  let got = read (Nx.cumsum ~axis:1 (Nx.place G.on (first p c))) in
  let want = Array.init (rows * p) (fun k -> whole.((k / p * r) + (k mod p))) in
  equal (array int32) (bits want) (bits got)

(* nx.mli's cumsum: a float result is within [γ(n - 1) Σ|x|] of the exact sum of
   its [n] terms. The float64 sum of float32 terms in [-1, 1) errs by at most [n
   2^-53 Σ|x|], far below the bound, which it is held to. *)
let law_scan (rows, r, _, seed) =
  cover "rows past a chunk" (r > 4096);
  let xs = terms (rows, r, false, seed) in
  let module G = (val Lazy.force set) in
  let got =
    read (Nx.cumsum ~axis:1 (Nx.place G.on (host D.Float32 [| rows; r |] xs)))
  in
  let u = ldexp 1. (-24) and w = ldexp 1. (-53) in
  let gamma k u = Float.of_int k *. u /. (1. -. (Float.of_int k *. u)) in
  let worst = ref 0. and sum = ref 0. and mag = ref 0. in
  Array.iteri
    (fun i x ->
      let n = (i mod r) + 1 in
      if n = 1 then begin
        sum := 0.;
        mag := 0.
      end;
      sum := !sum +. x;
      mag := !mag +. Float.abs x;
      (* A NaN for finite terms is an infinite error, never a pass. *)
      let err =
        if Float.is_nan got.(i) then Float.infinity
        else Float.abs (got.(i) -. !sum)
      in
      let bound = (gamma (n - 1) u +. gamma n w) *. !mag in
      if err > 0. then worst := Float.max !worst (err /. bound))
    xs;
  at_most ~msg:"the largest error over its bound" float_exact ~than:1. !worst

let () =
  exit
    (run "nx on nx.cuda"
       [
         group "a set of nx.cuda's kernels"
           [
             prop "a matmul of small integers is the host's"
               (Gen.with_pp pp_matmul matmuls)
               law_matmul;
             prop "a sum of small integers is the host's"
               (Gen.with_pp pp_sum sums) law_sum;
             prop "a float32 sum along rows has nx.cpu's bits"
               (Gen.with_pp pp_rows rows) law_sum_bits;
             prop ~count:200
               "a float32 cumsum of a prefix of the rows is the prefix of \
                their cumsum"
               (Gen.with_pp pp_prefix prefixes)
               law_prefix;
             prop
               "a float32 cumsum is within γ(n - 1) Σ|x| of the exact sum of \
                its n terms"
               (Gen.with_pp pp_rows rows) law_scan;
           ];
       ])
