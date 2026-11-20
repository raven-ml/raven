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
  if Float.abs (y -. Float.trunc y) = 0.5 then 2. *. Float.round (y /. 2.)
  else Float.round y

let digits levels y =
  Array.of_list
    (List.mapi (fun i l -> int_of_float (round_even y.(i)) + (l / 2)) levels)

(* At float32 the half width is PyTorch's [(L - 1) * (1 + eps) / 2] in float32
   operation order, read from torch 2.14.1's FiniteScalarQuantizer. A number far
   above zero bounds to [h - o]. *)
let half_widths =
  cases "float32 half widths are PyTorch's"
    ~id:(fun (l, _) -> string_of_int l)
    [
      (7, Float.of_string "0x1.806250p+1");
      (5, Float.of_string "0x1.00418ap+1");
      (8, Float.of_string "0x1.c072b2p+1" -. 0.5);
    ]
    (fun (l, h) ->
      equal (array float_exact) [| h |]
        (Nx.to_array (Fsq.bound (Fsq.v [ l ]) (vec Nx.float32 [| 1e30 |]))))

(* AION-1's levels, or levels of up to five numbers, each of 2 to 9 values. *)
let levels_gen =
  Gen.(
    frequency
      [
        (1, of_list [ image; segmap ]);
        (3, list ~size:(int_range 1 5) (int_range 2 9));
      ])

let bound_formula =
  prop "bound is tanh (z + s) h - o"
    Gen.(pair levels_gen (array ~size:(constant 5) (float_range (-4.) 4.)))
    (fun (levels, zs) ->
      let z = Array.sub zs 0 (List.length levels) in
      let got = Nx.to_array (Fsq.bound (Fsq.v levels) (vec Nx.float64 z)) in
      List.iteri
        (fun i l ->
          cover "an even level" (l mod 2 = 0);
          let want = (Float.tanh (z.(i) +. shift l) *. half l) -. offset l in
          (* tanh within 2 ulps, the shift, product and difference rounded once
             each: within 8 ulps of [h + o]. *)
          let tol = 8. *. epsilon_float *. (half l +. offset l) in
          equal (float tol) want got.(i))
        levels)

(* Numbers far above zero, which take their top digit, far below, which take
   their bottom one, zero, which takes digit [⌊L/2⌋], and between. *)
let rounds =
  prop "a code holds the bounded numbers rounded, halves to even"
    Gen.(
      let* levels = levels_gen in
      let* dt64 = bool in
      let+ z =
        array
          ~size:(constant (List.length levels))
          (frequency
             [
               (4, map Bounds.f32 (float_range (-3.) 3.));
               (1, of_list [ -20.; 0.; 20. ]);
             ])
      in
      (levels, dt64, z))
    (fun (levels, dt64, z) ->
      cover "AION-1's levels" (levels = image || levels = segmap);
      cover "a number far from zero"
        (Array.exists (fun x -> Float.abs x = 20.) z);
      let q = Fsq.v levels in
      let y, k =
        if dt64 then
          let z = vec Nx.float64 z in
          (Nx.to_array (Fsq.bound q z), Fsq.encode q z)
        else
          let z = vec Nx.float32 z in
          (Nx.to_array (Nx.cast Nx.float64 (Fsq.bound q z)), Fsq.encode q z)
      in
      equal (array int) [| code levels (digits levels y) |] (ints k))

(* A float32 [z] whose bound is [y] exactly, searched upward from [z]. *)
let rec onto q y z n =
  if n = 0 then None
  else if Nx.to_array (Fsq.bound q (vec Nx.float32 [| z |])) = [| y |] then
    Some z
  else onto q y (Bounds.move 1 z) (n - 1)

let ties =
  cases "a bounded number half way between two integers takes the even one"
    ~id:(fun (y, _) -> string_of_float y)
    [ (0.5, 3); (1.5, 5); (-0.5, 3); (-1.5, 1); (2.5, 5) ]
    (fun (y, digit) ->
      let q = Fsq.v [ 7 ] in
      let start = Bounds.move (-64) (Bounds.f32 (Float.atanh (y /. half 7))) in
      match onto q y start 128 with
      | None -> failf "no float32 bounds onto %g" y
      | Some z ->
          equal (array int) [| digit |]
            (ints (Fsq.encode q (vec Nx.float32 [| z |]))))

let decodes =
  prop "a code decodes to its digits' (k - ⌊L/2⌋) / ⌊L/2⌋"
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
      let rec split k = function
        | [] -> []
        | l :: ls -> (k mod l) :: split (k / l) ls
      in
      let want =
        List.map2
          (fun l d ->
            let h = l / 2 in
            Bounds.f32 (Float.of_int (d - h) /. Float.of_int h))
          levels (split k levels)
      in
      let v = Nx.to_array (Fsq.decode Nx.float32 q (codes [| k |])) in
      equal (array float_exact) (Array.of_list want) v)

let shapes =
  test "encode drops the last axis, decode adds it" (fun () ->
      let q = Fsq.v image in
      let k = Fsq.encode q (Nx.zeros Nx.float32 [| 2; 3; 5 |]) in
      equal (array int) [| 2; 3 |] (Nx.shape k);
      equal (array int) [| 2; 3; 5 |] (Nx.shape (Fsq.decode Nx.float32 q k));
      equal int 4375 (Fsq.size q);
      equal (list int) image (Fsq.levels q))

(* The walk reports the levels: a program for each list of levels. *)
let walk =
  test "the walk reports the levels" (fun () ->
      let visits l =
        List.map
          (Format.asprintf "%a" Nx.Ptree.pp_visit)
          (Nx.Ptree.visits (Nx.Ptree.v Fsq.walk) (Fsq.v l))
      in
      not_equal (list string) (visits segmap) (visits [ 7; 5; 5; 5 ]);
      equal (list string) (visits segmap) (visits [ 8; 5; 5; 5 ]))

(* Compiled with the quantizer as an argument, one function serves every
   quantizer: the walk reports the levels, so quantizers of four levels, whose
   arrays share shapes, key programs of their own. Bounding is tanh and IEEE
   arithmetic, which Rune.jit computes exactly on the host: bit for bit. *)
let signature () = Nx.Ptree.(v Fsq.walk @-> tensor @-> returns tensor)
let encode32 = Rune.jit (signature ()) Fsq.encode
let encode64 = Rune.jit (signature ()) Fsq.encode
let decode32 = Rune.jit (signature ()) (Fsq.decode Nx.float32)
let bound32 = Rune.jit (signature ()) Fsq.bound
let bound64 = Rune.jit (signature ()) Fsq.bound

(* A value's bits, every NaN as one: which NaN a function gives is
   unspecified. *)
let bits x =
  let bits v = if Float.is_nan v then 0x7ff8L else Int64.bits_of_float v in
  Array.map bits (Nx.to_array (Nx.cast Nx.float64 x))

(* Numbers near the bound's rounding steps, large magnitudes whose tanh is ±1,
   infinities, -0. and NaN, which bound keeps and encode refuses. *)
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
      let* levels = of_list [ segmap; [ 7; 5; 5; 5 ]; [ 5; 5; 5; 5 ] ] in
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

(* Each value refusal raises at the call on the host, eagerly and compiled. *)
let refusals =
  let refuses ?sub f = raises_match (Exn.invalid_arg ?sub) f in
  let q = Fsq.v image in
  group "refusals"
    [
      cases "v refuses degenerate levels"
        ~id:(fun l -> String.concat ";" (List.map string_of_int l))
        [ []; [ 1 ]; [ 5; 0 ]; [ 7; -3 ]; List.init 64 (fun _ -> 2) ]
        (fun l -> refuses (fun () -> Fsq.v l));
      cases "decode refuses a code outside [0, size)" ~id:string_of_int
        [ -1; 4375; 10_000 ] (fun k ->
          let sub = Printf.sprintf "code %d at [1] outside [0, 4375)" k in
          refuses ~sub (fun () -> Fsq.decode Nx.float32 q (codes [| 0; k |]));
          refuses ~sub (fun () -> decode32 q (codes [| 0; k |])));
      test "encode refuses NaN" (fun () ->
          let z = vec Nx.float32 [| 0.; Float.nan; 0.; 0.; 0. |] in
          refuses ~sub:"NaN at [0; 1]" (fun () -> Fsq.encode q z);
          refuses ~sub:"NaN at [0; 1]" (fun () -> encode32 q z));
      test "the last axis must hold one entry per level" (fun () ->
          refuses (fun () -> Fsq.bound q (Nx.zeros Nx.float32 [| 3; 4 |]));
          refuses (fun () -> Fsq.encode q (Nx.zeros Nx.float32 [| 3; 6 |])));
    ]

let () =
  exit
    (run "aion.fsq"
       [
         group "encode" [ half_widths; bound_formula; rounds; ties ];
         group "decode" [ decodes ];
         shapes;
         walk;
         refusals;
         compiled;
       ])
