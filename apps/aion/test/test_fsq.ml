(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Aion

let image = [ 7; 5; 5; 5; 5 ]
let segmap = [ 8; 5; 5; 5 ]
let ints a = Nx.to_array (Nx.cast Nx.int64 a) |> Array.map Int64.to_int
let vec dt a = Nx.create dt [| 1; Array.length a |] a
let codes a = Nx.create Nx.int64 [| Array.length a |] (Array.map Int64.of_int a)

(* The preamble's constants of a level [l]: [h], [o] and [s]. *)
let half l = Float.of_int (l - 1) *. (1. +. 1e-3) /. 2.
let offset l = if l mod 2 = 0 then 0.5 else 0.
let shift l = Float.atanh (offset l /. half l)

(* The code of the digits [d], the first least significant. *)
let code levels d =
  fst
    (List.fold_left2
       (fun (k, b) l d -> (k + (d * b), b * l))
       (0, 1) levels (Array.to_list d))

(* The integer nearest [y], halves to even. *)
let round_even y =
  let r = Float.round y in
  if Float.abs (y -. Float.trunc y) = 0.5 then 2. *. Float.round (y /. 2.)
  else r

let digits levels y =
  Array.of_list
    (List.mapi (fun i l -> int_of_float (round_even y.(i)) + (l / 2)) levels)

(* Codes the preamble states: far above zero every number takes its top digit,
   far below its bottom one, and at zero digit [⌊L/2⌋]. *)
let known =
  cases "codes of known vectors"
    ~id:(fun (n, _, _, _) -> n)
    [
      ("image, all high", image, [| 20.; 20.; 20.; 20.; 20. |], 4374);
      ("image, all low", image, [| -20.; -20.; -20.; -20.; -20. |], 0);
      ("image, zeros", image, [| 0.; 0.; 0.; 0.; 0. |], 2187);
      ("image, mixed", image, [| 20.; -20.; 0.; 0.; 0. |], 2176);
      ("segmap, all high", segmap, [| 20.; 20.; 20.; 20. |], 999);
      ("segmap, all low", segmap, [| -20.; -20.; -20.; -20. |], 0);
      ("segmap, zeros", segmap, [| 0.; 0.; 0.; 0. |], 500);
    ]
    (fun (_, levels, z, k) ->
      let q = Fsq.v levels in
      equal (array int) [| k |] (ints (Fsq.encode q (vec Nx.float32 z)));
      equal (array int) [| k |] (ints (Fsq.encode q (vec Nx.float64 z))))

(* At float32 the bound's half width is PyTorch's: [(L - 1) * (1 + eps) / 2] in
   float32 operation order, read from torch 2.14.1's FiniteScalarQuantizer
   constants. A number far above zero bounds to [h - o]. *)
let half_widths =
  cases "float32 half widths are PyTorch's"
    ~id:(fun (l, _) -> string_of_int l)
    [
      (7, Float.of_string "0x1.806250p+1");
      (5, Float.of_string "0x1.00418ap+1");
      (8, Float.of_string "0x1.c072b2p+1" -. 0.5);
    ]
    (fun (l, h) ->
      let q = Fsq.v [ l ] in
      equal (array float_exact) [| h |]
        (Nx.to_array (Fsq.bound q (vec Nx.float32 [| 1e30 |]))))

(* Levels of up to five numbers, each of 2 to 9 values. *)
let levels_gen = Gen.(list ~size:(int_range 1 5) (int_range 2 9))

let bound_formula =
  prop "bound is tanh (z + s) h - o"
    Gen.(pair levels_gen (array ~size:(constant 5) (float_range (-4.) 4.)))
    (fun (levels, zs) ->
      let d = List.length levels in
      let z = Array.sub zs 0 d in
      let q = Fsq.v levels in
      let got = Nx.to_array (Fsq.bound q (vec Nx.float64 z)) in
      List.iteri
        (fun i l ->
          cover "an even level" (l mod 2 = 0);
          let want = (Float.tanh (z.(i) +. shift l) *. half l) -. offset l in
          (* tanh within 2 ulps, the shift, product and difference rounded once
             each: within 8 ulps of [h + o]. *)
          let tol = 8. *. epsilon_float *. (half l +. offset l) in
          equal (float tol) want got.(i))
        levels)

let rounds =
  prop "a code holds the bounded numbers rounded, halves to even"
    Gen.(
      pair levels_gen
        (array ~size:(constant 5) (map Bounds.f32 (float_range (-3.) 3.))))
    (fun (levels, zs) ->
      let d = List.length levels in
      let z = vec Nx.float32 (Array.sub zs 0 d) in
      let q = Fsq.v levels in
      let y = Nx.to_array (Fsq.bound q z) in
      equal (array int)
        [| code levels (digits levels y) |]
        (ints (Fsq.encode q z)))

(* The float32 next to [z] toward [+∞]. *)
let next z = Bounds.move 1 z

(* A float32 [z] whose bound is [y] exactly, searched from [z] upward. *)
let rec onto q y z n =
  if n = 0 then None
  else
    let b = Nx.to_array (Fsq.bound q (vec Nx.float32 [| z |])) in
    if b.(0) = y then Some z else onto q y (next z) (n - 1)

let ties =
  cases "a bounded number half way between two integers takes the even one"
    ~id:(fun (y, _) -> string_of_float y)
    [ (0.5, 3); (1.5, 5); (-0.5, 3); (-1.5, 1); (2.5, 5) ]
    (fun (y, digit) ->
      let q = Fsq.v [ 7 ] in
      let start = Bounds.move (-64) (Bounds.f32 (Float.atanh (y /. half 7))) in
      match onto q y start 128 with
      | None -> fail (Printf.sprintf "no float32 bounds onto %g" y)
      | Some z ->
          equal (array int) [| digit |]
            (ints (Fsq.encode q (vec Nx.float32 [| z |]))))

(* The float64 [z] whose bound is the integer [k - ⌊L/2⌋] of digit [k]. *)
let preimage levels v =
  Array.of_list
    (List.mapi
       (fun i l ->
         let q = Float.round (v.(i) *. Float.of_int (l / 2)) in
         Float.atanh ((q +. offset l) /. half l) -. shift l)
       levels)

let round_trip =
  prop "a code's vector, bounded back, is the code"
    Gen.(
      let* levels = levels_gen in
      let n = List.fold_left ( * ) 1 levels in
      let+ k =
        frequency
          [ (1, constant 0); (1, constant (n - 1)); (2, int_range 0 (n - 1)) ]
      in
      (levels, k))
    (fun (levels, k) ->
      let q = Fsq.v levels in
      cover "the first code" (k = 0);
      cover "the last code" (k = Fsq.size q - 1);
      let v = Nx.to_array (Fsq.decode Nx.float32 q (codes [| k |])) in
      Array.iter (fun x -> at_most float_exact ~than:1. (Float.abs x)) v;
      let z = vec Nx.float64 (preimage levels v) in
      equal (array int) [| k |] (ints (Fsq.encode q z)))

let decoded =
  cases "codes decode to their vectors"
    ~id:(fun (k, _) -> string_of_int k)
    [
      (0, [| -1.; -1.; -1.; -1.; -1. |]);
      (2187, [| 0.; 0.; 0.; 0.; 0. |]);
      (4374, [| 1.; 1.; 1.; 1.; 1. |]);
      (1, [| Bounds.f32 (-2. /. 3.); -1.; -1.; -1.; -1. |]);
      (7, [| -1.; -0.5; -1.; -1.; -1. |]);
      (875, [| -1.; -1.; -1.; -1.; -0.5 |]);
    ]
    (fun (k, v) ->
      let q = Fsq.v image in
      equal (array float_exact) v
        (Nx.to_array (Fsq.decode Nx.float32 q (codes [| k |]))))

let refusals =
  group "refusals"
    [
      cases "v refuses degenerate levels"
        ~id:(fun l -> String.concat ";" (List.map string_of_int l))
        [ []; [ 1 ]; [ 5; 0 ]; [ 7; -3 ]; List.init 64 (fun _ -> 2) ]
        (fun l -> raises_match Exn.invalid_arg (fun () -> Fsq.v l));
      cases "decode refuses a code outside [0, size)" ~id:string_of_int
        [ -1; 4375; 10_000 ] (fun k ->
          raises_match (Exn.invalid_arg ~sub:"outside [0, 4375)") (fun () ->
              Fsq.decode Nx.float32 (Fsq.v image) (codes [| 0; k |])));
      test "encode refuses NaN" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Fsq.encode (Fsq.v image)
                (vec Nx.float32 [| 0.; Float.nan; 0.; 0.; 0. |])));
      test "the last axis must hold one entry per level" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Fsq.bound (Fsq.v image) (Nx.zeros Nx.float32 [| 3; 4 |]));
          raises_match Exn.invalid_arg (fun () ->
              Fsq.encode (Fsq.v image) (Nx.zeros Nx.float32 [| 3; 6 |])));
    ]

let shapes =
  test "encode drops the last axis, decode adds it" (fun () ->
      let q = Fsq.v image in
      let k = Fsq.encode q (Nx.zeros Nx.float32 [| 2; 3; 5 |]) in
      equal (array int) [| 2; 3 |] (Nx.shape k);
      equal (array int) [| 2; 3; 5 |] (Nx.shape (Fsq.decode Nx.float32 q k));
      equal int 4375 (Fsq.size q);
      equal (list int) image (Fsq.levels q))

(* The walk reports the levels: other levels of one length key other
   programs, equal levels the same. *)
let walk =
  let visits l =
    List.map
      (Format.asprintf "%a" Nx.Ptree.pp_visit)
      (Nx.Ptree.visits (Nx.Ptree.v Fsq.walk) (Fsq.v l))
  in
  group "walk"
    [
      test "other levels give other visits" (fun () ->
          not_equal (list string) (visits segmap) (visits [ 7; 5; 5; 5 ]));
      test "equal levels give equal visits" (fun () ->
          equal (list string) (visits segmap) (visits [ 8; 5; 5; 5 ]));
    ]

(* Compiled with the quantizer as an argument, one function serves every
   quantizer: its walk reports the levels, so quantizers of as many levels key
   programs of their own. Bounding is tanh and IEEE arithmetic, which Rune.jit
   computes exactly on the host, so codes and vectors agree bit for bit. *)
let signature () = Nx.Ptree.(v Fsq.walk @-> tensor @-> returns tensor)
let encode32 = Rune.jit (signature ()) Fsq.encode
let encode64 = Rune.jit (signature ()) Fsq.encode
let decode32 = Rune.jit (signature ()) (Fsq.decode Nx.float32)
let bound32 = Rune.jit (signature ()) Fsq.bound
let bound64 = Rune.jit (signature ()) Fsq.bound

(* Quantizers of four levels, so that their programs share a shape. *)
let fours = [ segmap; [ 7; 5; 5; 5 ]; [ 5; 5; 5; 5 ] ]

(* A value's bits, every NaN as one: which NaN a function gives is
   unspecified. *)
let bits x =
  let bits v = if Float.is_nan v then 0x7ff8L else Int64.bits_of_float v in
  Array.map bits (Nx.to_array (Nx.cast Nx.float64 x))

(* Numbers near the bound's rounding steps, large magnitudes whose tanh is
   ±1, infinities, -0. and NaN, which bound keeps and encode refuses. *)
let number =
  Gen.(
    frequency
      [
        (6, float_range (-3.) 3.);
        (1, of_list [ 1e30; -1e30; 1e300; -1e300 ]);
        (1, of_list [ Float.infinity; Float.neg_infinity; -0.; 0. ]);
        (1, constant Float.nan);
      ])

let compiled =
  prop "compiled equals eager, bit for bit"
    Gen.(
      let* levels = of_list fours in
      let n = List.fold_left ( * ) 1 levels in
      let* z = array ~size:(constant 12) number in
      let+ k = array ~size:(constant 3) (int_range 0 (n - 1)) in
      (levels, z, k))
    (fun (levels, z, k) ->
      cover "the segmentation map's levels" (levels = segmap);
      cover "levels of another quantizer of four" (levels <> segmap);
      let nan = Array.exists Float.is_nan z in
      cover "NaN" nan;
      cover "no NaN" (not nan);
      cover "an infinity" (Array.exists (fun x -> Float.abs x = infinity) z);
      let q = Fsq.v levels in
      let z64 = Nx.create Nx.float64 [| 3; 4 |] z in
      let z32 = Nx.cast Nx.float32 z64 in
      equal (array int64) (bits (Fsq.bound q z32)) (bits (bound32 q z32));
      equal (array int64) (bits (Fsq.bound q z64)) (bits (bound64 q z64));
      if not nan then begin
        equal (array int) (ints (Fsq.encode q z32)) (ints (encode32 q z32));
        equal (array int) (ints (Fsq.encode q z64)) (ints (encode64 q z64))
      end;
      let k = codes k in
      equal (array int64)
        (bits (Fsq.decode Nx.float32 q k))
        (bits (decode32 q k)))

(* On the host a compiled call raises a refusal at the call, as eagerly. *)
let compiled_refusals =
  group "compiled refusals"
    [
      test "decode refuses a code outside [0, size)" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"code 4375 at [1] outside [0, 4375)")
            (fun () -> decode32 (Fsq.v image) (codes [| 0; 4375 |])));
      test "encode refuses NaN" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"NaN at [0; 1]") (fun () ->
              encode32 (Fsq.v image)
                (vec Nx.float32 [| 0.; Float.nan; 0.; 0.; 0. |])));
    ]

let () =
  exit
    (run "aion.fsq"
       [
         group "encode"
           [ known; half_widths; bound_formula; rounds; ties; round_trip ];
         group "decode" [ decoded ];
         refusals;
         shapes;
         walk;
         group "compiled" [ compiled; compiled_refusals ];
       ])
