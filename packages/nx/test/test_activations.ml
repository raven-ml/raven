(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx's activations: sigmoid and softplus within their stated ulps of correctly
   rounded rows (gen/special.py) and, over float32 drawn as bits, of a float64
   reference; softmax and log_softmax within their stated bounds of their
   definitions; rms_norm and standardize within their bounds of their
   definitions computed to twice float64's digits, with the bits of the
   compositions and steps their docs name; the narrow floats at float32, rounded
   once. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

let value dt s xs : (float, _, Nx.host) Nx.t =
  Nx.Repr.of_array Nx.Host.v (A.of_array dt (Iarray.of_array s) xs)

let float_pp ppf v = Format.fprintf ppf "%h" v
let floats = array float_exact
let f32 xs = Nx.create Nx.float32 [| Array.length xs |] xs
let f64 xs = Nx.create Nx.float64 [| Array.length xs |] xs
let fl32 x = Int32.float_of_bits (Int32.bits_of_float x)
let host y = Nx.to_array (Nx.place Nx.Host.on y)

let near ~msg ~tol want got =
  equal ~msg (float (Float.max tol Float.min_float)) want got

(* [got] within [tol] of the pair [hi + lo]: the error taken before any rounding
   of the pair, [got - hi] exact where they are close. *)
let off_pair ~msg ~tol (hi, lo) got =
  near
    ~msg:(Printf.sprintf "%s: %h, by definition %h" msg got hi)
    ~tol 0.
    (got -. hi -. lo)

(* Sigmoid and softplus *)

(* Rows of [name]'s golden at float32 and float64: points about where each
   underflows, saturates and changes formula, each value correctly rounded. *)
let rows name =
  List.map (fun r -> ("f32", r)) (Nx_support.golden name D.Float32)
  @ List.map (fun r -> ("f64", r)) (Nx_support.golden name D.Float64)

let row_id (d, (r : Nx_support.row)) = Printf.sprintf "%s %h" d r.args.(0)

(* [f] of the row's argument at its dtype, read as float64, and that dtype. *)
let at_row (f : 's. (float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t)
    (d, (r : Nx_support.row)) =
  if d = "f32" then
    ( Nx_support.ulps D.Float32,
      (Nx.to_array (Nx.cast Nx.float64 (f (f32 r.args)))).(0) )
  else (Nx_support.ulps D.Float64, (Nx.to_array (f (f64 r.args))).(0))

let within_ulps k
    (f : 's. (float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t)
    ((_, (r : Nx_support.row)) as row) =
  let ulps, got = at_row f row in
  at_most
    ~msg:(Printf.sprintf "%h, exactly %h" got r.value)
    float_exact ~than:k (ulps r.value got)

(* Where a result is one of sigmoid's ends. *)
let ends v =
  if Int64.bits_of_float v = 0L then "+0."
  else if v = 1. then "1."
  else if Float.is_nan v then "nan"
  else "between"

let check_ends ((_, (r : Nx_support.row)) as row) =
  let _, got = at_row (fun x -> Nx.sigmoid x) row in
  equal
    ~msg:(Printf.sprintf "%h, exactly %h" got r.value)
    string (ends r.value) (ends got)

(* float32 drawn as bits, half of them about 0, against float64's [1 / (1 + exp
   (-x))] and [log1p (exp x)], [x + log1p (exp (-x))] above 30: within about
   [2^-51] of the exact value, which rounds to float32 other than the reference
   only within that of a tie. *)
let float32 =
  let open Gen in
  with_pp float_pp
    (frequency
       [
         (1, map Int32.float_of_bits int32);
         (1, map fl32 (float_range (-120.) 120.));
       ])

(* Float32's greatest of both signs, and points whose sigmoid is subnormal,
   which a draw reaches one case in fourteen. *)
let float32_examples =
  let top = Int32.float_of_bits 0x7f7fffffl in
  [ top; -.top; -100.; -95.5 ]

let sigmoid64 x = 1. /. (1. +. exp (-.x))

let softplus64 x =
  if x > 30. then x +. Float.log1p (exp (-.x)) else Float.log1p (exp x)

let law_sigmoid32 x =
  let got = (Nx.to_array (Nx.cast Nx.float64 (Nx.sigmoid (f32 [| x |])))).(0) in
  let r = sigmoid64 x in
  cover "a subnormal result" (r > 0. && r < 0x1p-126);
  cover "a result of 1." (fl32 r = 1.);
  cover "a NaN" (Float.is_nan x);
  if not (Float.is_nan x) then
    satisfies ~claim:"in [0, 1]" float_exact (fun v -> v >= 0. && v <= 1.) got;
  at_most
    ~msg:(Printf.sprintf "%h, about %h" got r)
    float_exact ~than:3.
    (Nx_support.ulps D.Float32 r got)

let law_softplus32 x =
  let got =
    (Nx.to_array (Nx.cast Nx.float64 (Nx.softplus (f32 [| x |])))).(0)
  in
  let r = softplus64 x in
  cover "far below 0" (x < -20.);
  cover "far above 0" (x > 20.);
  cover "the greatest floats" (Float.abs x > 1e30);
  at_most
    ~msg:(Printf.sprintf "%h, about %h" got r)
    float_exact ~than:3.
    (Nx_support.ulps D.Float32 r got)

let test_sigmoid_specials () =
  equal floats
    [| 0.5; 0.5; 1.; 0.; nan |]
    (Nx.to_array (Nx.sigmoid (f64 [| 0.; -0.; infinity; neg_infinity; nan |])))

let test_softplus_specials () =
  equal floats
    [| Float.log 2.; Float.log 2.; infinity; 0.; nan |]
    (Nx.to_array (Nx.softplus (f64 [| 0.; -0.; infinity; neg_infinity; nan |])))

(* At a float narrower than float32, [f x] is [f] at float32 rounded once, over
   every code of the format. *)
let codes (type s) (dt : (float, s) D.t) : (float, s, Nx.host) Nx.t =
  let every =
    match D.bits dt with
    | 4 -> Nx.bitcast dt (Nx.arange Nx.uint4 0 16 1)
    | 8 -> Nx.bitcast dt (Nx.arange Nx.uint8 0 256 1)
    | _ -> Nx.bitcast dt (Nx.arange Nx.uint16 0 65536 1)
  in
  Nx.place Nx.Host.on every

let narrow_dtypes =
  List.filter (fun (D.Any dt) -> D.is D.Float dt && D.bits dt < 32) D.all

let narrow name (f : 's. (float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t)
    =
  cases
    (name ^ " at a narrow float is float32's, rounded once")
    ~id:(fun (D.Any dt) -> D.name dt)
    narrow_dtypes
    (fun (D.Any dt) ->
      match D.kind dt with
      | D.Float ->
          let x = codes dt in
          equal floats
            (Nx.to_array
               (Nx.cast Nx.float64 (Nx.cast dt (f (Nx.cast Nx.float32 x)))))
            (Nx.to_array (Nx.cast Nx.float64 (f x)))
      | _ -> ())

(* Rows of 16, so that a NaN code spoils its own row alone. *)
let in_rows x = Nx.reshape [| Nx.numel x / 16; 16 |] x

(* A row of 16 affine values at [x]'s dtype, which every format holds. *)
let row_of x vs = Nx.cast (Nx.dtype x) (Nx.create Nx.float32 [| 16 |] vs)

let gains =
  [| 1.; -1.; 0.5; 1.5; 2.; -3.; 4.; 6.; 1.; 0.5; -0.5; 3.; -2.; 1.; 0.; 1. |]

let offsets =
  [| 0.; 0.5; -1.; 2.; -0.5; 1.; 0.; -2.; 3.; 0.; 1.5; -1.5; 0.; 6.; -6.; 4. |]

(* Pairs of twice float64's digits *)

(* [a + b] as a float64 and its rounding error. *)
let two_sum a b =
  let s = a +. b in
  let v = s -. a in
  (s, a -. (s -. v) +. (b -. v))

(* [a + b] as [two_sum], for [|a| >= |b|]. *)
let fast_two_sum a b =
  let s = a +. b in
  (s, b -. (s -. a))

(* Pairs [hi + lo] of twice float64's digits: a sum and a product within about
   [2^-104] of the exact one, relatively, a quotient and a root a little
   more. *)
let dd_add (ah, al) (bh, bl) =
  let s, e = two_sum ah bh in
  fast_two_sum s (e +. al +. bl)

let dd_mul (ah, al) (bh, bl) =
  let p = ah *. bh in
  fast_two_sum p (Float.fma ah bh (-.p) +. (ah *. bl) +. (al *. bh))

let dd_div (ah, al) (bh, bl) =
  let q = ah /. bh in
  fast_two_sum q ((Float.fma (-.q) bh ah +. al -. (q *. bl)) /. bh)

let dd_sqrt (ah, al) =
  if ah = 0. then (0., 0.)
  else
    let r = Float.sqrt ah in
    fast_two_sum r ((Float.fma (-.r) r ah +. al) /. (2. *. r))

(* Lanes *)

(* The lanes of [xs], of shape [s], along its last axis: positions in C
   order. *)
let lanes s =
  let r = Array.length s in
  let n = s.(r - 1) in
  let total = Array.fold_left ( * ) 1 s in
  List.init
    (if n = 0 then 0 else total / n)
    (fun l -> Array.init n (fun j -> (l * n) + j))

(* The full blocks of 1024 terms of one of [outputs] lanes of [n] that the CPU
   kernels reduce in one call, before its cap of 32 KiB of terms: below 64
   outputs, a lane's blocks fall into groups of a power of two, about 64 groups
   in all, and a call takes one group's full blocks. *)
let blocks_a_call ~outputs n =
  let blocks = (n + 1023) / 1024 in
  let group =
    if outputs >= 64 || blocks <= 1 then blocks
    else
      let want = (64 + outputs - 1) / outputs in
      let each = (blocks + want - 1) / want in
      let rec up g = if g < each then up (2 * g) else g in
      up 1
  in
  min group (n / 1024)

(* Covers the CPU kernels' calls of several full blocks of a lane of [n] in [s]
   at [dt], and of more than the 32 KiB a call holds. *)
let covers_blocks (type s) (dt : (float, s) D.t) s n =
  let outputs = Array.fold_left ( * ) 1 s / n in
  let b = blocks_a_call ~outputs n in
  cover "several full blocks a call" (b >= 2);
  cover "more full blocks than a call holds" (b > 256 / D.bits dt)

(* As many lanes as the CPU kernels aim their calls at, each of a few more full
   blocks than a call of float32 holds: some of every lane's calls take several
   blocks, and some reach the cap. Every law that draws lanes takes it as an
   example, as no draw is this large. *)
let wide = [| 64; 9217 |]

let wide_xs =
  Array.init (64 * 9217) (fun i ->
      if i mod 10 = 3 then -0.
      else Float.of_int ((i * 7919 mod 2001) - 1000) /. 16.)

(* [n] values within [bound], zeros of either sign among them. *)
let values_within bound n =
  let open Gen in
  array ~size:(constant n)
    (frequency
       [
         (8, float_range (-.bound) bound); (1, constant 0.); (1, constant (-0.));
       ])

(* Each device a window of the lanes, along the last axis. *)
module Mem = (val Nx.devices (List.init 4 Nx_support.memory))

let cut_lanes x = Nx.place (Nx.Placement.split ~axis:(Nx.ndim x - 1) Mem.v) x

(* [b] of a lane of [n]: [β(n)], or over four windows [β(m) + t - 1]. *)
let depth_of ~cut n =
  if cut then Nx_softmax_support.beta ((n + 3) / 4) + Stdlib.min n 4 - 1
  else Nx_softmax_support.beta n

(* Softmax and log_softmax *)

module Sm = Nx_softmax_support

type special = Plain | Neg_inf | Nan | Inf

(* Lanes of one element to several blocks, of values within a few units to past
   [exp]'s range, about [0.] or far from it, some cut across four devices; [-∞]
   elements, or a NaN or a [+∞] in every other lane. *)
let softmax_case =
  let open Gen in
  let* s =
    frequency
      [
        (3, array ~size:(int_range 1 3) (int_range 1 5));
        (1, map (fun r -> [| r; 1 |]) (int_range 1 4));
        ( 1,
          let* r = int_range 1 3 in
          let+ k = of_list ~pp:Format.pp_print_int [ 1025; 2048; 3000 ] in
          [| r; k |] );
      ]
  in
  let n = Array.fold_left ( * ) 1 s in
  let* bound =
    frequency
      [ (2, of_list ~pp:float_pp [ 1.; 60.; 300. ]); (1, constant 1000.) ]
  in
  let* xs = values_within bound n in
  let* offset =
    frequency [ (3, constant 0.); (1, of_list ~pp:float_pp [ 1e3; -1e5; 3e6 ]) ]
  in
  let* special =
    frequency
      [
        (5, constant Plain);
        (4, constant Neg_inf);
        (1, constant Nan);
        (1, constant Inf);
      ]
  in
  let* cut = frequency [ (3, constant false); (1, constant true) ] in
  let* seed = int in
  let+ scale = of_list ~pp:float_pp [ 1.; -0.5; 3.; 0.1 ] in
  (s, xs, offset, special, seed, cut, scale)

(* The wide lanes with a NaN in every other one; lanes far from 0 under a scale
   not a power of two; and a lane whose exact softmax is below the least normal,
   which a draw reaches one case in fourteen at float64. *)
let softmax_examples =
  [
    (wide, wide_xs, 0., Nan, 7, false, 1.);
    ([| 2; 3 |], [| 1.; -2.; 0.5; 3.; 0.; -1. |], 1e3, Plain, 0, false, 3.);
    ([| 1; 2 |], [| 1000.; 0. |], 0., Plain, 0, false, 1.);
  ]

(* [xs] shifted by [offset], with [special]'s elements: a quarter [-∞], or a NaN
   or [+∞] in every other lane of [k]. *)
let with_special special seed offset k xs =
  let st = Random.State.make [| seed |] in
  Array.mapi
    (fun i x ->
      let lane = i / k and j = i mod k in
      let marked = lane mod 2 = 0 && j = lane * 7 mod k in
      match special with
      | Plain -> x +. offset
      | Neg_inf ->
          if Random.State.int st 4 = 0 then neg_infinity else x +. offset
      | Nan -> if marked then nan else x +. offset
      | Inf -> if marked then infinity else x +. offset)
    xs

(* Softmax and log_softmax within the bounds their docs state of their
   definitions, [b] [β(k)] for a lane of [k], or [β(m) + t - 1] cut into [t]
   windows of at most [m]; [+0.] and [-∞] at [-∞]; NaN where the doc makes a
   lane NaN. [u] is [dt]'s unit roundoff and [normal] its least normal float. *)
let law_softmax (type s) (dt : (float, s) D.t) u normal
    (s, xs, offset, special, seed, cut, scale) =
  let k = s.(Array.length s - 1) in
  covers_blocks dt s k;
  cover "one element a lane" (k = 1);
  cover "lanes cut across devices" cut;
  cover "lanes far from 0" (offset <> 0.);
  cover "a scale other than 1" (scale <> 1.);
  cover "far from 0 under a scale not a power of two"
    (offset <> 0. && fst (Float.frexp (Float.abs scale)) <> 0.5);
  cover "-inf elements" (special = Neg_inf);
  let xs = with_special special seed offset k xs in
  let xs = Array.map (D.of_float dt) xs in
  let x = value dt s xs in
  let run (type d) (x : (float, s, d) Nx.t) =
    (host (Nx.softmax ~scale x), host (Nx.log_softmax ~scale x))
  in
  let sm, lsm = if cut then run (cut_lanes x) else run x in
  let b = depth_of ~cut k in
  List.iter
    (fun lane ->
      let lx = Array.map (fun i -> xs.(i)) lane in
      let exact = Sm.lane ~scale lx in
      cover "a NaN lane" (Option.is_none exact);
      match exact with
      | None ->
          let nans = Array.make (Array.length lane) nan in
          equal ~msg:"softmax of a NaN lane" floats nans
            (Array.map (fun i -> sm.(i)) lane);
          equal ~msg:"log_softmax of a NaN lane" floats nans
            (Array.map (fun i -> lsm.(i)) lane)
      | Some l ->
          Sm.covers ~normal l;
          let at j = lane.(j) in
          Sm.check u ~normal ~exp_ulps:2 ~b ~n:k ~scale l ~at sm;
          Sm.check_log u ~exp_ulps:2 ~log_ulps:2 ~b ~n:k ~scale l ~at lsm)
    (lanes s)

(* Under [~where], each lane is the softmax of the elements it holds, by its
   definition over those alone, within the bound its doc states, as is its
   log_softmax, [n] counting every element of the lane; an element it does not
   hold is exactly [+0.] (log_softmax: [-∞]) whatever its value, NaN included,
   in a lane holding a NaN too, and a lane holding none is all [+0.] ([-∞]). A
   NaN held makes its lane's held elements NaN. *)
let where_case =
  let open Gen in
  let* r = int_range 1 4 in
  let* n = frequency [ (4, int_range 1 8); (1, of_list [ 1025; 2048 ]) ] in
  let total = r * n in
  (* Per lane: none held, all held, or each element drawn. *)
  let* kinds = array ~size:(constant r) (int_range 0 3) in
  let* bits = array ~size:(constant total) bool in
  let held =
    Array.init total (fun i ->
        match kinds.(i / n) with 0 -> false | 1 -> true | _ -> bits.(i))
  in
  let* xs = values_within 60. total in
  let* nans =
    array ~size:(constant total)
      (frequency [ (1, constant true); (12, constant false) ])
  in
  let xs = Array.mapi (fun i x -> if nans.(i) then Float.nan else x) xs in
  let+ scale = of_list [ 1.; -0.5; 3.; 0.1 ] in
  ([| r; n |], xs, held, scale)

let law_where (type s) (dt : (float, s) D.t) u normal (s, xs, held, scale) =
  let xs = Array.map (D.of_float dt) xs in
  let x = value dt s xs in
  let where = Nx.create Nx.bool s held in
  let sm = Nx.to_array (Nx.softmax ~scale ~where x)
  and lsm = Nx.to_array (Nx.log_softmax ~scale ~where x) in
  let n = s.(1) in
  List.iter
    (fun lane ->
      let mine = List.filter (fun i -> held.(i)) (Array.to_list lane) in
      let nan_held = List.exists (fun i -> Float.is_nan xs.(i)) mine in
      let outside = List.filter (fun i -> not held.(i)) (Array.to_list lane) in
      cover "a lane holding none" (mine = []);
      cover "a lane holding some" (mine <> [] && List.length mine < n);
      cover "a NaN outside where"
        (List.exists (fun i -> Float.is_nan xs.(i)) outside);
      cover "a NaN held" nan_held;
      let at l = Array.of_list (List.map (fun i -> sm.(i)) l) in
      let lat l = Array.of_list (List.map (fun i -> lsm.(i)) l) in
      equal ~msg:"softmax outside where" floats
        (Array.make (List.length outside) 0.)
        (at outside);
      equal ~msg:"log_softmax outside where" floats
        (Array.make (List.length outside) neg_infinity)
        (lat outside);
      if nan_held then begin
        equal ~msg:"softmax in a lane holding a NaN" floats
          (Array.make (List.length mine) nan)
          (at mine);
        equal ~msg:"log_softmax in a lane holding a NaN" floats
          (Array.make (List.length mine) nan)
          (lat mine)
      end
      else if mine <> [] then
        let mine = Array.of_list mine in
        match Sm.lane ~scale (Array.map (fun i -> xs.(i)) mine) with
        | None -> fail "a lane of finite elements by its definition NaN"
        | Some l ->
            let b = Sm.beta n and at j = mine.(j) in
            Sm.check u ~normal ~exp_ulps:2 ~b ~n ~scale l ~at sm;
            Sm.check_log u ~exp_ulps:2 ~log_ulps:2 ~b ~n ~scale l ~at lsm)
    (lanes s)

(* Every other element of each row of 16 of [x], but of every third row, which
   holds none. *)
let some_held x =
  let r = Nx.dim 0 x in
  Nx.create Nx.bool [| r; 16 |]
    (Array.init (r * 16) (fun i -> i / 16 mod 3 <> 2 && i mod 2 = 0))

(* Along [axes], the lane is every element the axes reach: each element is
   within its bound of the softmax of those elements. *)
let test_axes () =
  let x =
    Nx.create Nx.float64 [| 2; 3; 2 |]
      (Array.init 12 (fun i -> Float.of_int (i * 7 mod 5)))
  in
  let got = Nx.to_array (Nx.softmax ~axes:[ 0; 2 ] x) in
  let xs = Nx.to_array x in
  for j = 0 to 2 do
    let lane =
      Array.of_list
        (List.concat_map
           (fun i -> [ (i * 6) + (j * 2); (i * 6) + (j * 2) + 1 ])
           [ 0; 1 ])
    in
    match Sm.lane ~scale:1. (Array.map (fun i -> xs.(i)) lane) with
    | None -> fail "a lane of finite elements by its definition NaN"
    | Some l ->
        Sm.check 0x1p-53 ~normal:0x1p-1022 ~exp_ulps:2 ~b:(Sm.beta 4) ~n:4
          ~scale:1. l
          ~at:(fun k -> lane.(k))
          got
  done

let test_softmax_values () =
  equal ~msg:"a lane holding a NaN" floats [| nan; nan; nan |]
    (Nx.to_array (Nx.softmax (f64 [| 1.; nan; 2. |])));
  equal ~msg:"a lane of greatest +∞" floats [| nan; nan |]
    (Nx.to_array (Nx.softmax (f64 [| infinity; 1. |])));
  (* [d = [0; -1000]], whose [exp] is [[1; 0]]: their sum [1], its log [0]. *)
  equal ~msg:"log_softmax keeps a far entry's digits" floats [| 0.; -1000. |]
    (Nx.to_array (Nx.log_softmax (f64 [| 0.; -1000. |])));
  equal ~msg:"softmax of the same" floats [| 1.; 0. |]
    (Nx.to_array (Nx.softmax (f64 [| 0.; -1000. |])));
  equal ~msg:"lanes of none" (array int) [| 0; 3 |]
    (Nx.shape (Nx.softmax (Nx.zeros Nx.float32 [| 0; 3 |])));
  let m = Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let rows = Nx.create Nx.bool [| 2; 1 |] [| true; false |]
  and cols = Nx.create Nx.bool [| 3 |] [| true; false; true |] in
  equal ~msg:"a lane holding none is 0." floats
    (Array.append (Array.sub (Nx.to_array (Nx.softmax m)) 0 3) [| 0.; 0.; 0. |])
    (Nx.to_array (Nx.softmax ~where:rows m));
  let p = Nx.to_array (Nx.softmax (f64 [| 1.; 3. |])) in
  let x = Nx.create Nx.float64 [| 2; 3 |] [| 1.; 9.; 3.; 4.; nan; 6. |] in
  equal ~msg:"an element outside where takes no part, NaN or not" floats
    [| p.(0); 0.; p.(1); p.(0); 0.; p.(1) |]
    (Nx.to_array (Nx.softmax ~where:cols x));
  (* A zero scale makes the held elements equal, at the stored float64 and the
     narrow float16 alike, and leaves out the others. *)
  let held = Nx.create Nx.bool [| 3 |] [| true; false; true |] in
  equal ~msg:"a zero scale" floats [| 0.5; 0.; 0.5 |]
    (Nx.to_array (Nx.softmax ~scale:0. ~where:held (f64 [| 1.; 9.; -3. |])));
  equal ~msg:"a zero scale at float16" floats [| 0.5; 0.; 0.5 |]
    (Nx.to_array
       (Nx.cast Nx.float64
          (Nx.softmax ~scale:0. ~where:held
             (Nx.cast Nx.float16 (f64 [| 1.; 9.; -3. |])))));
  (* [-log 2] within log_softmax's bound at [n = 3], [d = 0]. *)
  let got =
    Nx.to_array (Nx.log_softmax ~scale:0. ~where:held (f64 [| 1.; 9.; -3. |]))
  in
  let r = -.Float.log 2. and u = 0x1p-53 in
  let tol =
    ((Float.of_int (Sm.beta 3) +. (6. *. Float.log 3.) +. 5.) *. u)
    +. (u *. 2. *. Float.abs r)
  in
  near ~msg:"log_softmax at a zero scale" ~tol r got.(0);
  equal ~msg:"log_softmax outside where" float_exact neg_infinity got.(1);
  (* A negative scale shifts by the least element. *)
  equal ~msg:"a negative scale" floats [| 1.; 0. |]
    (Nx.to_array (Nx.softmax ~scale:(-1.) (f64 [| -1e6; 1e6 |])))

let raise_rows =
  let one = f64 [| 1. |] and three = f32 [| 1.; 2.; 3. |] in
  let m = Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let row id by f = (id, by, fun () -> ignore (f ())) in
  [
    row "softmax, an axis past the rank" "Nx.softmax" (fun () ->
        Nx.softmax ~axes:[ 2 ] one);
    row "log_softmax, an axis repeated" "Nx.log_softmax" (fun () ->
        Nx.log_softmax ~axes:[ 0; 0 ] one);
    row "softmax, a where of another shape" "Nx.softmax" (fun () ->
        Nx.softmax ~where:(Nx.create Nx.bool [| 2 |] [| true; false |]) m);
    row "rms_norm, an axis past the rank" "Nx.rms_norm" (fun () ->
        Nx.rms_norm ~axes:[ 1 ] (f32 [| 1. |]));
    row "rms_norm, a negative eps" "Nx.rms_norm" (fun () ->
        Nx.rms_norm ~eps:(-1.) (f32 [| 1. |]));
    row "rms_norm, a NaN eps" "Nx.rms_norm" (fun () ->
        Nx.rms_norm ~eps:nan (f32 [| 1. |]));
    row "rms_norm, a scale of another shape" "Nx.rms_norm" (fun () ->
        Nx.rms_norm ~scale:(f32 [| 1.; 2. |]) three);
    row "standardize, an axis past the rank" "Nx.standardize" (fun () ->
        Nx.standardize ~axes:[ 1 ] (f32 [| 1. |]));
    row "standardize, an axis repeated" "Nx.standardize" (fun () ->
        Nx.standardize ~axes:[ 0; 0 ] (f32 [| 1. |]));
    row "standardize, a negative eps" "Nx.standardize" (fun () ->
        Nx.standardize ~eps:(-1.) (f32 [| 1. |]));
    row "standardize, a NaN eps" "Nx.standardize" (fun () ->
        Nx.standardize ~eps:nan (f32 [| 1. |]));
    row "standardize, a scale of another shape" "Nx.standardize" (fun () ->
        Nx.standardize ~scale:(f32 [| 1.; 2. |]) three);
    row "standardize, a shift of another shape" "Nx.standardize" (fun () ->
        Nx.standardize ~shift:(f32 [| 1.; 2. |]) three);
  ]

(* The bytes the host holds after four rounds of collection, each of which
   returns to it what it collected. *)
let settled () =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create Rig.host 0);
    Rig.free_cache Rig.host
  done;
  Rig.allocated Rig.host

(* The most bytes a host buffer of [n] bytes counts: its own, up to 112 more
   where it starts on 128 bytes, and on Windows, where one of 64 KiB starts on a
   page of 4 KiB, that page less 16 more. *)
let host_bytes n = n + if Sys.win32 && n >= 64 * 1024 then 4096 - 16 else 112

(* At every float dtype a softmax holds, after the call and before any
   collection, the result alone of its operand's size, with and without
   [~where]: the bytes the host holds more than settled, its cache included,
   count every buffer the call took. Lanes of 4096, so that each lane's greatest
   element and sum, held too, are under a page between them, and a page is less
   than any other array of the operand's size, 8 KiB at float4. One uncounted
   run comes first. *)
let test_softmax_memory () =
  List.iter
    (fun (D.Any dt) ->
      match D.kind dt with
      | D.Float ->
          let x =
            Nx.cast dt
              (Nx.Rng.normal ~key:(Nx.Rng.key 5) Nx.float32 [| 4; 4096 |])
          in
          let where =
            Nx.create Nx.bool [| 1; 4096 |]
              (Array.init 4096 (fun i -> i mod 3 <> 0))
          in
          Nx.sync x;
          List.iter
            (fun where ->
              let call () =
                let y = Nx.softmax ?where x in
                Nx.sync y;
                y
              in
              ignore (call ());
              let before = settled () in
              let y = call () in
              let grew = Rig.allocated Rig.host - before in
              let result = D.bytes dt (4 * 4096) in
              less
                ~msg:
                  (Printf.sprintf "bytes held above the settled ones, %s%s"
                     (D.name dt)
                     (if where = None then "" else ", under where"))
                int
                ~than:(host_bytes result + 4096)
                grew;
              equal (array int) [| 4; 4096 |] (Nx.shape y))
            [ None; Some where ]
      | _ -> ())
    D.all

(* RMS norm *)

(* [(hi + lo) b] as a pair, within about [2^-104] of it, relatively; an infinite
   or NaN product alone. *)
let dd_mul_float (hi, lo) b =
  let p = hi *. b in
  if not (Float.is_finite p) then (p, 0.)
  else fast_two_sum p (Float.fma hi b (-.p) +. (lo *. b))

(* Values of a lane's scale or shift that every float format holds. *)
let affine_value =
  Gen.of_list ~pp:float_pp [ 1.; -1.; 0.5; -0.5; 1.5; 2.; -3.; 4.; 6.; 0. ]

(* [xs] times the power of two that brings its greatest magnitude into [[1, 2)]:
   at [eps = 0], a lane's normalisation is that of the scaled lane, whose
   squares do not underflow. *)
let unit_scale xs =
  let m = Array.fold_left (fun a x -> Float.max a (Float.abs x)) 0. xs in
  if m = 0. || not (Float.is_finite m) then xs
  else
    let _, e = Float.frexp m in
    Array.map (fun x -> Float.ldexp x (1 - e)) xs

(* The power of two a case of small lanes takes at [dt], whose lanes are then
   subnormal. *)
let small_scale (type s) (dt : (float, s) D.t) =
  match dt with D.Float32 -> 0x1p-140 | _ -> 0x1p-1060

(* Whether [x] is subnormal at [dt]. *)
let subnormal (type s) (dt : (float, s) D.t) x =
  let least_normal = match dt with D.Float32 -> 0x1p-126 | _ -> 0x1p-1022 in
  x <> 0. && Float.abs x < least_normal

(* A lane of [n] values by the definition, each a pair [hi + lo] within about
   [n² 2^-106] of the exact value, relatively, far below either format's
   tolerance: the squares and their sum compensated, then the mean, [eps], the
   root and the quotient carried to twice float64's digits. *)
let lane_rms eps xs =
  let xs = if eps = 0. then unit_scale xs else xs in
  let n = Float.of_int (Array.length xs) in
  let hi = ref 0. and lo = ref 0. in
  Array.iter
    (fun x ->
      let p = x *. x in
      let s, e = two_sum !hi p in
      hi := s;
      lo := !lo +. e +. Float.fma x x (-.p))
    xs;
  let s, l = fast_two_sum !hi !lo in
  let q = s /. n in
  let m, ml = fast_two_sum q ((Float.fma (-.q) n s +. l) /. n) in
  let a, e = two_sum m eps in
  let a, al = fast_two_sum a (e +. ml) in
  let r = Float.sqrt a in
  let rh, rl =
    if a = 0. then (0., 0.)
    else fast_two_sum r ((Float.fma (-.r) r a +. al) /. (2. *. r))
  in
  Array.map
    (fun x ->
      if rh = 0. then (x /. 0., 0.)
      else
        let q = x /. rh in
        fast_two_sum q ((Float.fma (-.q) rh x -. (q *. rl)) /. rh))
    xs

(* Lanes along the last axis of a few to several blocks of terms, each drawn
   often; values of either sign, zeros of both; a quarter cut across four
   devices, which take no scale. *)
let rms_case =
  let open Gen in
  let* s =
    frequency
      [
        (2, array ~size:(int_range 1 3) (int_range 1 5));
        ( 3,
          let* r = int_range 1 3 in
          let+ k =
            of_list ~pp:Format.pp_print_int
              [ 1; 16; 17; 1024; 1025; 2048; 3000; 5000 ]
          in
          [| r; k |] );
      ]
  in
  let n = Array.fold_left ( * ) 1 s in
  let* xs = values_within 100. n in
  let* eps = of_list [ 1e-6; 0.; 0.5 ] in
  (* A fifth of the cases, subnormal lanes, at [eps = 0]. *)
  let* small = frequency [ (1, constant true); (4, constant false) ] in
  let eps = if small then 0. else eps in
  let k = s.(Array.length s - 1) in
  let* cut = frequency [ (3, constant false); (1, constant true) ] in
  let+ g = option (array ~size:(constant k) affine_value) in
  (s, xs, eps, (if cut then None else g), small, cut)

(* The wide lanes, and a lane of one term, which a draw reaches one case in
   eight. *)
let rms_examples =
  [
    (wide, wide_xs, 1e-6, None, false, false);
    ([| 3; 1 |], [| 1.; -2.; 0. |], 1e-6, Some [| -3. |], false, false);
  ]

(* [xs] at [dt], subnormal for a case of small lanes. *)
let drawn (type s) (dt : (float, s) D.t) small xs =
  let at = if small then small_scale dt else 1. in
  Array.map (fun x -> D.of_float dt (x *. at)) xs

(* [g], the values of a lane's affine operand, at [dt] along the last axis of
   [x]. *)
let affine_of (type s) (dt : (float, s) D.t) g =
  Option.map
    (fun g -> value dt [| Array.length g |] (Array.map (D.of_float dt) g))
    g

(* Within [k u / (1 - 4 k u)] of the definition, relatively, for [k = (b + 9) /
   2], or [(b + 11) / 2] with a scale, [b] [β(n)] or over [t] windows of at most
   [m] [β(m) + t - 1], as its doc states. An output below the least normal
   rounds by half a least subnormal [tiny], and a scale [g]'s product by [|g|]
   times that and half again: [(|g| + 1) tiny] bounds both. NaN exactly where
   the definition is. *)
let law_rms (type s) (dt : (float, s) D.t) u tiny
    ((s, xs, eps, g, small, cut) as _c) =
  let n = s.(Array.length s - 1) in
  cover "one term a lane" (n = 1);
  covers_blocks dt s n;
  cover "no eps" (eps = 0.);
  cover "a scale" (Option.is_some g);
  cover "subnormal lanes" small;
  cover "lanes cut across devices" cut;
  let xs = drawn dt small xs in
  let scale = affine_of dt g in
  let x = value dt s xs in
  let y =
    if cut then host (Nx.rms_norm ~eps (cut_lanes x))
    else host (Nx.rms_norm ~eps ?scale x)
  in
  let g = Option.map (Array.map (D.of_float dt)) g in
  let b = Float.of_int (depth_of ~cut n) in
  let k = (b +. if Option.is_some g then 11. else 9.) /. 2. in
  let tol = k *. u /. (1. -. (4. *. k *. u)) in
  List.iter
    (fun lane ->
      let want = lane_rms eps (Array.map (fun i -> xs.(i)) lane) in
      let want =
        match g with
        | None -> want
        | Some g -> Array.mapi (fun j w -> dd_mul_float w g.(j)) want
      in
      Array.iteri
        (fun j i ->
          let gj = match g with Some g -> g.(j) | None -> 1. in
          let hi, lo = want.(j) in
          let msg = Printf.sprintf "rms_norm at %d" i in
          if Float.is_nan hi then equal ~msg float_exact nan y.(i)
          else if y.(i) <> hi then
            off_pair ~msg
              ~tol:((tol *. Float.abs hi) +. ((Float.abs gj +. 1.) *. tiny))
              (hi, lo) y.(i))
        lane)
    (lanes s)

(* At float32 and float64, the bits of the composition its doc names, on lanes
   whose squares neither overflow nor, at [eps = 0], sum to [0]: at [eps = 0],
   each lane is zeros or its greatest square normal. *)
let law_rms_composed (type s) (dt : (float, s) D.t) (s, xs, eps, g, small, _) =
  let xs = drawn dt small xs in
  let least = match dt with D.Float32 -> 0x1p-63 | _ -> 0x1p-511 in
  let kept lane =
    let big m i = Float.max m (Float.abs xs.(i)) in
    let m = Array.fold_left big 0. lane in
    m = 0. || m >= least
  in
  assume (eps > 0. || List.for_all kept (lanes s));
  let x = value dt s xs in
  let scale = affine_of dt g in
  let ms = Nx.mean ~axes:[ -1 ] ~keepdims:true (Nx.square x) in
  let want = Nx.mul x (Nx.rsqrt (Nx.add_s ms eps)) in
  let want = match scale with None -> want | Some g -> Nx.mul want g in
  equal floats (Nx.to_array want) (Nx.to_array (Nx.rms_norm ~eps ?scale x))

let test_rms_values () =
  equal ~msg:"a lane holding a NaN is NaN" floats [| nan; nan; nan |]
    (Nx.to_array (Nx.rms_norm (f32 [| 1.; nan; 2. |])));
  equal ~msg:"a lane holding an infinity is x · 0" floats [| 0.; -0.; nan |]
    (Nx.to_array (Nx.rms_norm (f32 [| 1e30; -1.; infinity |])));
  equal ~msg:"a lane of zeros" floats [| 0.; 0. |]
    (Nx.to_array (Nx.rms_norm (f32 [| 0.; 0. |])));
  equal ~msg:"at eps 0, [3; -3]" floats [| 1.; -1. |]
    (Nx.to_array (Nx.rms_norm ~eps:0. (f32 [| 3.; -3. |])));
  equal ~msg:"at eps 0, a lane of zeros is x · ∞" floats [| nan; nan |]
    (Nx.to_array (Nx.rms_norm ~eps:0. (f32 [| 0.; 0. |])));
  equal ~msg:"at eps 0, [1e-30] at float32 is [1.]" floats [| 1. |]
    (Nx.to_array (Nx.rms_norm ~eps:0. (f32 [| 1e-30 |])));
  equal ~msg:"at eps 0, subnormal squares, float32" floats [| 1.; -1. |]
    (Nx.to_array (Nx.rms_norm ~eps:0. (f32 [| 0x1p-140; -0x1p-140 |])));
  equal ~msg:"at eps 0, subnormal squares, float64" floats [| 1.; -1. |]
    (Nx.to_array (Nx.rms_norm ~eps:0. (f64 [| 0x1p-1060; -0x1p-1060 |])));
  equal ~msg:"squares that overflow" floats [| 1.; -1. |]
    (Nx.to_array (Nx.rms_norm (f32 [| 1e20; -1e20 |])));
  equal ~msg:"an overflowing lane beside one that keeps its bits" floats
    (Nx.to_array (Nx.rms_norm (f32 [| 3.; 5. |])))
    (Array.sub
       (Nx.to_array
          (Nx.rms_norm
             (Nx.create Nx.float32 [| 2; 2 |] [| 3.; 5.; 1e30; 1e30 |])))
       0 2);
  equal ~msg:"lanes of none" (array int) [| 0; 3 |]
    (Nx.shape (Nx.rms_norm (Nx.zeros Nx.float32 [| 0; 3 |])));
  (* Small integers, whose squares sum exactly in every association. *)
  let x =
    Nx.create Nx.float64 [| 2; 3; 2 |]
      (Array.init 12 (fun i -> Float.of_int (i * 7 mod 5)))
  in
  let moved = Nx.reshape [| 3; 4 |] (Nx.transpose ~axes:[ 1; 0; 2 ] x) in
  let back y = Nx.transpose ~axes:[ 1; 0; 2 ] (Nx.reshape [| 3; 2; 2 |] y) in
  equal ~msg:"along several axes, the lane of every element they reach" floats
    (Nx.to_array (back (Nx.rms_norm moved)))
    (Nx.to_array (Nx.rms_norm ~axes:[ 0; 2 ] x))

(* Standardize *)

(* A lane by the definition, [(x - μ) / sqrt (V + eps)], each element a pair far
   within either format's tolerance, with the lane's exact mean [μ], variance
   [V], mean absolute deviation and mean magnitude, to the precision a bound
   needs. *)
let lane_z eps xs =
  let xs = if eps = 0. then unit_scale xs else xs in
  let n = (Float.of_int (Array.length xs), 0.) in
  let sum = Array.fold_left (fun a x -> dd_add a (x, 0.)) (0., 0.) xs in
  let mu = dd_div sum n in
  let ds = Array.map (fun x -> dd_add (x, 0.) (-.fst mu, -.snd mu)) xs in
  let sq = Array.fold_left (fun a d -> dd_add a (dd_mul d d)) (0., 0.) ds in
  let v = dd_div sq n in
  let root = dd_sqrt (dd_add v (eps, 0.)) in
  let y =
    Array.map
      (fun d -> if fst root = 0. then (fst d /. 0., 0.) else dd_div d root)
      ds
  in
  let mean f = Array.fold_left (fun a x -> a +. f x) 0. xs /. fst n in
  let dev = mean (fun x -> Float.abs (x -. fst mu)) in
  (y, fst mu, fst v, dev, mean Float.abs)

(* The variance's depth: nx.cpu's [L + D] for a lane of [n] terms on one device
   (nx_cpu.mli), [β(m) + t - 1] over cut axes, as var states. *)
let z_depth ~cut n =
  if cut then Float.of_int (depth_of ~cut n)
  else
    let l = ((Stdlib.min n 1024 + 15) / 16) + Stdlib.min 3 (n - 1) in
    let rec depth b d = if b <= 1 then d else depth ((b + 1) / 2) (d + 1) in
    Float.of_int (l + depth ((n + 1023) / 1024) 0)

(* The bound [standardize]'s doc states for an element of exact value [y] of a
   lane, before the narrow rounding and the affine: [(F / 2s + 11u/2) |y| + (b +
   1) u A / √s], [F] the variance's bound in its depth [b]. *)
let z_bound u b eps (mu, v, dev, mag) y =
  let e = (b *. u *. mag) +. (u *. Float.abs mu) in
  let f =
    ((b +. 5.) *. u *. v)
    +. (2. *. (b +. 1.) *. u *. e *. Float.sqrt v)
    +. (((3. *. b) +. 7.) *. u *. e *. e)
  in
  let s = v +. eps in
  (((f /. (2. *. s)) +. (5.5 *. u)) *. Float.abs y)
  +. ((b +. 1.) *. u *. dev /. Float.sqrt s)

(* Lanes of a few to several blocks, drawn about a mean far from their spread or
   none, constant ones among them; an affine half the time, a quarter cut across
   four devices without one. *)
let z_case =
  let open Gen in
  let* s =
    frequency
      [
        (2, array ~size:(int_range 1 3) (int_range 1 5));
        ( 3,
          let* r = int_range 1 3 in
          let+ k =
            of_list ~pp:Format.pp_print_int
              [ 1; 2; 16; 17; 768; 1024; 1025; 3000; 5000 ]
          in
          [| r; k |] );
      ]
  in
  let n = Array.fold_left ( * ) 1 s and k = s.(Array.length s - 1) in
  let* mu = of_list ~pp:float_pp [ 0.; 64.; -1000.; 1e4; 3e5 ] in
  let* sigma = of_list ~pp:float_pp [ 1.; 1e-3; 100.; 0. ] in
  let* zs = values_within 3. n in
  (* Deviations of [2^-10] spreads at least: the bound holds barring underflow,
     which a drawn [z] near zero would reach. *)
  let grid z = Float.round (z *. 1024.) /. 1024. in
  let at z = if mu = 0. then sigma *. grid z else mu +. (sigma *. grid z) in
  (* An eighth of the cases, lanes of zeros of both signs. *)
  let* zeros = frequency [ (1, constant true); (7, constant false) ] in
  (* A fifth, subnormal lanes at [eps = 0], their spread their size. *)
  let* small = frequency [ (1, constant true); (4, constant false) ] in
  let xs =
    Array.map
      (if zeros then fun z -> 0. *. z else if small then grid else at)
      zs
  in
  let* eps = of_list [ 1e-5; 0.; 0.5 ] in
  let eps = if small then 0. else eps in
  let* cut = frequency [ (3, constant false); (1, constant true) ] in
  let* g = option (array ~size:(constant k) affine_value) in
  let+ b = option (array ~size:(constant k) affine_value) in
  let g, b = if cut then (None, None) else (g, b) in
  (s, xs, eps, g, b, small, cut)

(* The wide lanes; a lane of one term; a constant lane of zeros of both signs,
   which draws reach about one case in eight. *)
let z_examples =
  [
    (wide, wide_xs, 1e-5, None, None, false, false);
    ( [| 2; 1 |],
      [| 3.; -1e4 |],
      1e-5,
      Some [| 2. |],
      Some [| -1. |],
      false,
      false );
    ([| 1; 4 |], [| -0.; 0.; -0.; 0. |], 0.5, None, None, false, false);
  ]

(* [y] through the affine where there is one: no step where there is none, so
   that a [+0.] stays one. *)
let affine g b j y =
  let y = match g with None -> y | Some g -> y *. g.(j) in
  match b with None -> y | Some b -> y +. b.(j)

(* Within the bound its doc states of the definition, the affine's [|g|] times
   it plus [u |g y|] and [u |g y + b|], and a subnormal's rounding carried by
   [g]; a lane of no spread, [-0.] among its elements or not, is the affine of
   [+0.] exactly, at every [eps]. *)
let law_z (type s) (dt : (float, s) D.t) u tiny (s, xs, eps, g, b, small, cut) =
  let k = s.(Array.length s - 1) in
  cover "one term a lane" (k = 1);
  covers_blocks dt s k;
  cover "no eps" (eps = 0.);
  cover "a scale and a shift" (Option.is_some g && Option.is_some b);
  cover "lanes cut across devices" cut;
  let xs = drawn dt small xs in
  cover "a subnormal lane" (Array.exists (subnormal dt) xs);
  let x = value dt s xs in
  let y =
    if cut then host (Nx.standardize ~eps (cut_lanes x))
    else
      host
        (Nx.standardize ~eps ?scale:(affine_of dt g) ?shift:(affine_of dt b) x)
  in
  let lane_g = Option.value g ~default:(Array.make k 1.) in
  let lane_b = Option.value b ~default:(Array.make k 0.) in
  let depth = z_depth ~cut k in
  List.iter
    (fun lane ->
      let lx = Array.map (fun i -> xs.(i)) lane in
      let want, mu, v, dev, mag = lane_z eps lx in
      let constant = Array.for_all (fun x -> x = lx.(0)) lx in
      cover "a constant lane" constant;
      cover "a -0. in a constant lane"
        (constant
        && Array.exists
             (fun x -> Int64.bits_of_float x = Int64.bits_of_float (-0.))
             lx);
      cover "a lane far from its mean" (dev > 0. && Float.abs mu > 1e3 *. dev);
      Array.iteri
        (fun j i ->
          let yr, yl = want.(j) and got = y.(i) in
          let gj = lane_g.(j) and bj = lane_b.(j) in
          let msg = Printf.sprintf "standardize at %d" i in
          if v = 0. then
            equal
              ~msg:(msg ^ ", a lane of no spread")
              float_exact (affine g b j 0.) got
          else
            let zh, zl = dd_add (dd_mul_float (yr, yl) gj) (bj, 0.) in
            let tol =
              (Float.abs gj *. z_bound u depth eps (mu, v, dev, mag) yr)
              +. (u *. (Float.abs (gj *. yr) +. Float.abs zh))
              +. ((Float.abs gj +. 1.) *. tiny)
            in
            off_pair ~msg ~tol (zh, zl) got)
        lane)
    (lanes s)

(* At float32 and float64, the bits of the composition its doc names. *)
let law_z_composed (type s) (dt : (float, s) D.t) (s, xs, eps, g, b, small, _) =
  let x = value dt s (drawn dt small xs) in
  let scale = affine_of dt g and shift = affine_of dt b in
  let want = Nx.standardize ~eps x in
  let want = match scale with None -> want | Some g -> Nx.mul want g in
  let want = match shift with None -> want | Some b -> Nx.add want b in
  equal floats (Nx.to_array want)
    (Nx.to_array (Nx.standardize ~eps ?scale ?shift x))

(* [standardize ~eps x] along the last axis by the steps its doc names, each one
   operation: [c = Σ x / n], the sums [s₁] of [d = x - c] and [s₂] of [d²], [c₂
   = s₁ / n], [v = (s₂ - s₁ c₂) / n] as the kernels' moments take it, at least
   [0.] and [s₂] where that is [∞], then [(d - c₂) r] for [r = 1 / sqrt (v +
   eps)], [+0.] where [d = c₂]. *)
let z_steps ~eps x =
  let n = Float.of_int (Nx.dim (Nx.ndim x - 1) x) in
  let lane y = Nx.sum ~axes:[ -1 ] ~keepdims:true y in
  let d = Nx.sub x (Nx.div_s (lane x) n) in
  let s1 = lane d and s2 = lane (Nx.mul d d) in
  let c2 = Nx.div_s s1 n in
  let v = Nx.div_s (Nx.sub s2 (Nx.mul s1 c2)) n in
  let v = Nx.where (Nx.less_s v 0.) (Nx.zeros_like v) v in
  let v = Nx.where (Nx.equal_s s2 Float.infinity) s2 v in
  let r = Nx.recip (Nx.sqrt (Nx.add_s v eps)) in
  let q = Nx.sub d c2 in
  Nx.where (Nx.equal_s q 0.) (Nx.zeros_like q) (Nx.mul q r)

(* At float32 and float64, a lane that needs no power of two, as no case but the
   subnormal ones does, has the bits of its doc's steps. *)
let law_z_steps (type s) (dt : (float, s) D.t) (s, xs, eps, _, _, small, _) =
  assume (not small);
  let xs = drawn dt small xs in
  let lanes = List.map (Array.map (fun i -> xs.(i))) (lanes s) in
  let equal_elements l = Array.for_all (fun x -> x = l.(0)) l in
  (* A lane whose mean is a thousand times its greatest deviation or more. *)
  let far l =
    let n = Float.of_int (Array.length l) in
    let m = Array.fold_left ( +. ) 0. l /. n in
    let dev =
      Array.fold_left (fun a x -> Float.max a (Float.abs (x -. m))) 0. l
    in
    dev > 0. && Float.abs m >= 1000. *. dev
  in
  cover "a lane of equal elements, +0. throughout"
    (List.exists equal_elements lanes);
  cover "no eps" (eps = 0.);
  cover "a lane whose mean is far from its spread" (List.exists far lanes);
  let x = value dt s xs in
  equal floats
    (Nx.to_array (z_steps ~eps x))
    (Nx.to_array (Nx.standardize ~eps x))

(* A lane of equal elements is [+0.], whatever their mean's rounding, up to
   [2^17] of them at float32: the old composition gave [0.21] for [1000.1]. *)
let equal_lanes =
  [
    ("1024 of 1000.1", Array.make 1024 1000.1);
    ("768 of 1000.1", Array.make 768 1000.1);
    ("4096 of 0.1", Array.make 4096 0.1);
    ("5000 of -3.3", Array.make 5000 (-3.3));
    ("one 7.", [| 7. |]);
    ("3000 of 1e-30", Array.make 3000 1e-30);
    ("2^17 of 12345.678", Array.make 0x20000 12345.678);
    ("2^17 of 0.1", Array.make 0x20000 0.1);
    ("four zeros", Array.make 4 0.);
    ("-0. twice", [| -0.; -0. |]);
    ("0. and -0.", [| 0.; -0. |]);
    ("one -0.", [| -0. |]);
    ("1025 of -0.", Array.make 1025 (-0.));
    ("1e30 twice", [| 1e30; 1e30 |]);
  ]

let check_equal_lane (_, xs) =
  let n = Array.length xs in
  let at (type s) (dt : (float, s) D.t) eps =
    let x = Nx.create dt [| 1; n |] (Array.map (D.of_float dt) xs) in
    equal
      ~msg:(Printf.sprintf "%s, eps %g" (D.name dt) eps)
      floats (Array.make n 0.)
      (Nx.to_array (Nx.cast Nx.float64 (Nx.standardize ~eps x)))
  in
  List.iter
    (fun eps ->
      at D.Float32 eps;
      at D.Float64 eps;
      at D.Bfloat16 eps)
    [ 1e-5; 0. ]

let test_z_values () =
  let f16 xs = Nx.create Nx.float16 [| Array.length xs |] xs in
  equal ~msg:"a lane holding a NaN" floats [| nan; nan; nan |]
    (Nx.to_array (Nx.standardize (f32 [| 1.; nan; 2. |])));
  equal ~msg:"a lane holding an infinity" floats [| nan; nan; nan |]
    (Nx.to_array (Nx.standardize (f32 [| 1.; infinity; 2. |])));
  equal ~msg:"a lane of -∞" floats [| nan; nan |]
    (Nx.to_array (Nx.standardize (f32 [| neg_infinity; neg_infinity |])));
  equal ~msg:"a float16 lane whose variance passes float16's greatest" floats
    [| -1.; 1.; -1.; 1. |]
    (Nx.to_array
       (Nx.cast Nx.float64
          (Nx.standardize (f16 [| -300.; 300.; -300.; 300. |]))));
  equal ~msg:"at eps 0, [3; 5]" floats [| -1.; 1. |]
    (Nx.to_array (Nx.standardize ~eps:0. (f32 [| 3.; 5. |])));
  equal ~msg:"at eps 0, [2; 2]" floats [| 0.; 0. |]
    (Nx.to_array (Nx.standardize ~eps:0. (f32 [| 2.; 2. |])));
  equal ~msg:"squares that underflow at eps 0, beside a lane of zeros" floats
    [| 1.; -1.; 0.; 0. |]
    (Nx.to_array
       (Nx.standardize ~eps:0.
          (Nx.create Nx.float32 [| 2; 2 |] [| 0x1p-140; -0x1p-140; 0.; -0. |])));
  equal ~msg:"float64 squares that underflow" floats [| 1.; -1. |]
    (Nx.to_array (Nx.standardize ~eps:0. (f64 [| 0x1p-1060; -0x1p-1060 |])));
  equal ~msg:"[1e-40; -1e-40] at float32 is [1; -1]" floats [| 1.; -1. |]
    (Nx.to_array (Nx.standardize ~eps:0. (f32 [| 1e-40; -1e-40 |])));
  equal ~msg:"bfloat16 squares that underflow" floats [| 1.; -1. |]
    (Nx.to_array
       (Nx.cast Nx.float64
          (Nx.standardize ~eps:0.
             (Nx.create Nx.bfloat16 [| 2 |] [| 0x1p-130; -0x1p-130 |]))));
  let wide = Nx.create Nx.float32 [| 2; 2 |] [| -3e19; 3e19; 3.; 5. |] in
  equal
    ~msg:
      "squared deviations past the greatest beside a lane that keeps its bits"
    floats
    (Array.append [| -1.; 1. |]
       (Nx.to_array (Nx.standardize (f32 [| 3.; 5. |]))))
    (Nx.to_array (Nx.standardize wide));
  equal ~msg:"[1e38; -1e38] at float32" floats [| 1.; -1. |]
    (Nx.to_array (Nx.standardize (f32 [| 1e38; -1e38 |])));
  equal ~msg:"[1e300; -1e300] at float64" floats [| 1.; -1. |]
    (Nx.to_array (Nx.standardize (f64 [| 1e300; -1e300 |])));
  equal ~msg:"[1e30; -1e30] at bfloat16" floats [| 1.; -1. |]
    (Nx.to_array
       (Nx.cast Nx.float64
          (Nx.standardize (Nx.create Nx.bfloat16 [| 2 |] [| 1e30; -1e30 |]))));
  equal ~msg:"+0. before the affine" floats [| -0.; 0. |]
    (Nx.to_array (Nx.standardize ~scale:(f32 [| -1.; 1. |]) (f32 [| 4.; 4. |])));
  equal ~msg:"then scale and shift" floats [| 0.5; -1.5 |]
    (Nx.to_array
       (Nx.standardize
          ~scale:(f32 [| 2.; -1. |])
          ~shift:(f32 [| 0.5; -1.5 |])
          (f32 [| 4.; 4. |])));
  equal ~msg:"lanes of none" (array int) [| 0; 3 |]
    (Nx.shape (Nx.standardize (Nx.zeros Nx.float32 [| 0; 3 |])));
  equal ~msg:"lanes of no element" (array int) [| 3; 0 |]
    (Nx.shape (Nx.standardize (Nx.zeros Nx.float32 [| 3; 0 |])));
  (* Small integers, whose sums are exact in every association. *)
  let x =
    Nx.create Nx.float64 [| 2; 3; 2 |]
      (Array.init 12 (fun i -> Float.of_int (i * 7 mod 5)))
  in
  let moved = Nx.reshape [| 3; 4 |] (Nx.transpose ~axes:[ 1; 0; 2 ] x) in
  let back y = Nx.transpose ~axes:[ 1; 0; 2 ] (Nx.reshape [| 3; 2; 2 |] y) in
  equal ~msg:"along several axes, the lane of every element they reach" floats
    (Nx.to_array (back (Nx.standardize moved)))
    (Nx.to_array (Nx.standardize ~axes:[ 0; 2 ] x))

(* Over a strided or broadcast view, and with a broadcast scale and shift, the
   bits of the same values contiguous: nx.cpu's association depends on the shape
   alone. *)
let test_z_views () =
  let x =
    Nx.create Nx.float32 [| 4; 6 |]
      (Array.init 24 (fun i -> 100. +. (Float.of_int (i * 7 mod 11) /. 4.)))
  in
  let g = Nx.create Nx.float32 [| 3 |] [| 1.5; -0.5; 2. |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 0.25; 1.; -1. |] in
  let strided = Nx.slice [ Rs (0, 4, 2); Rs (1, 6, 2) ] x in
  let wide v = Nx.copy (Nx.broadcast_to [| 2; 3 |] v) in
  equal ~msg:"strided, a broadcast scale and shift" floats
    (Nx.to_array
       (Nx.standardize ~scale:(wide g) ~shift:(wide b) (Nx.copy strided)))
    (Nx.to_array (Nx.standardize ~scale:g ~shift:b strided));
  let row = Nx.slice [ R (1, 2) ] x in
  let one = Nx.to_array (Nx.standardize row) in
  equal ~msg:"rows broadcast" floats
    (Array.concat [ one; one; one ])
    (Nx.to_array (Nx.standardize (Nx.broadcast_to [| 3; 6 |] row)))

let () =
  exit
    (run "nx activations"
       [
         group "sigmoid"
           [
             cases "is within 3 ulps of correctly rounded rows" ~id:row_id
               (rows "sigmoid")
               (within_ulps 3. (fun x -> Nx.sigmoid x));
             cases "is +0. and 1. only where the exact value rounds there"
               ~id:row_id (rows "sigmoid") check_ends;
             prop "over float32 is in [0, 1] and within 3 ulps"
               ~examples:float32_examples float32 law_sigmoid32;
             test "has its values at 0, the infinities and NaN"
               test_sigmoid_specials;
             narrow "sigmoid" Nx.sigmoid;
           ];
         group "softplus"
           [
             cases "is within 3 ulps of correctly rounded rows" ~id:row_id
               (rows "softplus")
               (within_ulps 3. (fun x -> Nx.softplus x));
             prop "over float32 is within 3 ulps" ~examples:float32_examples
               float32 law_softplus32;
             test "has its values at 0, the infinities and NaN"
               test_softplus_specials;
             narrow "softplus" Nx.softplus;
           ];
         group "softmax and log_softmax"
           [
             prop "float64 are within their bounds" ~examples:softmax_examples
               softmax_case
               (law_softmax D.Float64 0x1p-53 0x1p-1022);
             prop "float32 are within their bounds" ~examples:softmax_examples
               softmax_case
               (law_softmax D.Float32 0x1p-24 0x1p-126);
             prop "float64 under where are within their bounds" where_case
               (law_where D.Float64 0x1p-53 0x1p-1022);
             prop "float32 under where are within their bounds" where_case
               (law_where D.Float32 0x1p-24 0x1p-126);
             test "along several axes, the lane is every element they reach"
               test_axes;
             test "have the values their definition gives" test_softmax_values;
             test "softmax holds its result alone at every float dtype"
               test_softmax_memory;
             narrow "softmax" (fun x -> Nx.softmax (in_rows x));
             narrow "log_softmax" (fun x -> Nx.log_softmax (in_rows x));
             narrow "softmax under where" (fun x ->
                 let x = in_rows x in
                 Nx.softmax ~where:(some_held x) x);
             narrow "log_softmax under where" (fun x ->
                 let x = in_rows x in
                 Nx.log_softmax ~where:(some_held x) x);
           ];
         group "rms_norm"
           [
             prop "float64 is within its bound of its definition"
               ~examples:rms_examples rms_case
               (law_rms D.Float64 0x1p-53 0x1p-1074);
             prop "float32 is within its bound of its definition"
               ~examples:rms_examples rms_case
               (law_rms D.Float32 0x1p-24 0x1p-149);
             prop "float32 has the bits of its composition"
               ~examples:rms_examples rms_case
               (law_rms_composed D.Float32);
             prop "float64 has the bits of its composition"
               ~examples:rms_examples rms_case
               (law_rms_composed D.Float64);
             test "has the values its doc states" test_rms_values;
             narrow "rms_norm" (fun x -> Nx.rms_norm (in_rows x));
             narrow "rms_norm with a scale" (fun x ->
                 Nx.rms_norm ~scale:(row_of x gains) (in_rows x));
           ];
         group "standardize"
           [
             prop "float64 is within its bound" ~examples:z_examples z_case
               (law_z D.Float64 0x1p-53 0x1p-1074);
             prop "float32 is within its bound" ~examples:z_examples z_case
               (law_z D.Float32 0x1p-24 0x1p-149);
             prop "float32's affine has the bits of its composition"
               ~examples:z_examples z_case (law_z_composed D.Float32);
             prop "float64's affine has the bits of its composition"
               ~examples:z_examples z_case (law_z_composed D.Float64);
             prop "float32 has the bits of its doc's steps" ~examples:z_examples
               z_case (law_z_steps D.Float32);
             prop "float64 has the bits of its doc's steps" ~examples:z_examples
               z_case (law_z_steps D.Float64);
             cases "a lane of equal elements is +0." ~id:fst equal_lanes
               check_equal_lane;
             test "has the values its doc states" test_z_values;
             test "a view has its copy's bits" test_z_views;
             narrow "standardize" (fun x -> Nx.standardize (in_rows x));
             narrow "standardize with a scale and a shift" (fun x ->
                 Nx.standardize ~scale:(row_of x gains)
                   ~shift:(row_of x offsets) (in_rows x));
           ];
         cases
           "each raises naming itself for an axis outside the rank or \
            repeated, an eps negative or NaN, or an operand that does not \
            broadcast"
           ~id:(fun (id, _, _) -> id)
           raise_rows
           (fun (_, by, f) -> invalid ~by f);
       ])
