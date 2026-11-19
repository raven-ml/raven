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

(* A set over a software device, without kernels: it computes only under
   Rune.jit, on this machine's CPU. *)
module Soft = (val Nx.devices [ Result.get_ok (Rig.software "aion-fsq") ])

let compiled =
  test "encode and decode compile to the host's codes and vectors" (fun () ->
      let q = Fsq.v segmap in
      let z =
        Nx.create Nx.float32 [| 3; 4 |]
          [| 0.; 0.3; -0.2; 2.; -1.; 0.1; 0.7; -3.; 5.; -5.; 0.25; 0. |]
      in
      let host x = Nx.place Nx.Host.on x in
      let k = Fsq.encode q z in
      let kj = Rune.jit' (Fsq.encode q) (Nx.place Soft.on z) in
      equal (array int) (ints k) (ints (host kj));
      let dj = Rune.jit' (Fsq.decode Nx.float32 q) (Nx.place Soft.on k) in
      equal (array float_exact)
        (Nx.to_array (Fsq.decode Nx.float32 q k))
        (Nx.to_array (host dj)))

let () =
  exit
    (run "aion.fsq"
       [
         group "encode"
           [ known; half_widths; bound_formula; rounds; ties; round_trip ];
         group "decode" [ decoded ];
         refusals;
         shapes;
         group "devices" [ compiled ];
       ])
