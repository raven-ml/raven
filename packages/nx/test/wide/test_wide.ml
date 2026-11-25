(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Double-word numbers. Each operation is held to its stated bound against
   mpmath's exact results (golden/, written by gen/generate.py), and the
   representation, the constructor and the walk to their laws. *)

open Windtrap

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

(* Dtypes: a name, the dtype and its unit roundoff [u]. *)

type dtype = D : string * (float, 's) Nx.dtype * float -> dtype

let f32 = D ("f32", Nx.float32, Float.ldexp 1. (-24))
let f64 = D ("f64", Nx.float64, Float.ldexp 1. (-53))
let dtypes = [ f32; f64 ]
let name (D (n, _, _)) = n

let dtype =
  Gen.of_list ~pp:(fun ppf d -> Format.pp_print_string ppf (name d)) dtypes

(* Goldens: rows of a dtype's tag and hexadecimal floats. *)

let golden file (D (tag, _, _)) =
  In_channel.with_open_text
    ("golden/" ^ file ^ ".golden")
    In_channel.input_lines
  |> List.filter_map (fun l ->
      match String.split_on_char ' ' l with
      | t :: fields when t = tag -> Some (Array.of_list fields)
      | _ -> None)

let column rows i =
  Array.of_list (List.map (fun r -> float_of_string r.(i)) rows)

let vector dt xs = Nx.create dt [| Array.length xs |] xs

(* The numbers of columns [i] and [i + 1], high and low words. *)
let operand dt rows i =
  Nx_wide.v ~lo:(vector dt (column rows (i + 1))) (vector dt (column rows i))

let words w = (Nx.to_array (Nx_wide.hi w), Nx.to_array (Nx_wide.lo w))

(* Errors against exact results *)

let two_sum a b =
  let s = a +. b in
  let a' = s -. b in
  let b' = s -. a' in
  (s, a -. a' +. (b -. b'))

(* [fast_two_sum a b] is [a + b] rounded and its rounding error, exactly where
   [|a| >= |b|], as the drawn words and every number's words are: unlike
   [two_sum], it forms no intermediate past the sum. *)
let fast_two_sum a b =
  let s = a +. b in
  (s, b -. (s -. a))

(* [ordered_sum a b] is the exact [a + b] as a rounded sum and its error: the
   fast two-sum of the larger magnitude first. *)
let ordered_sum a b =
  if Float.abs a >= Float.abs b then fast_two_sum a b else fast_two_sum b a

(* [error (zh, zl) (r1, r2, r3)] is [zh + zl - (r1 + r2 + r3)], to a relative
   [2^-50] of itself: [zh - r1] is exact, as [zh] and [r1] round one number, and
   the rest is summed with each rounding error kept. *)
let error (zh, zl) (r1, r2, r3) =
  let s, c =
    List.fold_left
      (fun (s, c) t ->
        let s, e = two_sum s t in
        (s, c +. e))
      (0., 0.)
      [ zh -. r1; zl; -.r2; -.r3 ]
  in
  s +. c

(* The slack of measuring an error in float64: far below any bound's margin. *)
let slack = 1. +. Float.ldexp 1. (-40)

(* [within ~bound ~u rows z] checks that each number of [z] is within [bound]
   relative of the exact result in [rows]' columns 4 to 6, the worst row
   reported in units of [u²]. *)
let within ~bound ~u rows z =
  let zh, zl = words z in
  let r1 = column rows 4 and r2 = column rows 5 and r3 = column rows 6 in
  let worst = ref (0., -1) in
  Array.iteri
    (fun k r1 ->
      (* [gap] makes a NaN or infinite error infinite, which the worst row keeps
         and the bound fails. *)
      let e = Nx_support.gap (error (zh.(k), zl.(k)) (r1, r2.(k), r3.(k))) 0. in
      let rel =
        if r1 = 0. then if e = 0. then 0. else Float.infinity
        else e /. Float.abs r1
      in
      if rel > fst !worst || snd !worst < 0 then worst := (rel, k))
    r1;
  at_most
    ~msg:(Printf.sprintf "row %d, in units of u²" (snd !worst))
    float_exact
    ~than:(bound *. slack /. (u *. u))
    (fst !worst /. (u *. u))

let sum_bound u = 3. *. u *. u /. (1. -. (4. *. u))

(* A float's value: zero unsigned, every NaN as one. A number's words are its
   high word to its bits and its low word to its value, whose zero's sign a
   number does not carry. *)
let unsigned =
  Testable.contramap
    (fun x -> if x = 0. then 0. else if Float.is_nan x then Float.nan else x)
    float_exact

let number = pair float_exact unsigned
let exact_words = pair (array float_exact) (array float_exact)

(* Drawn numbers *)

(* Numbers drawn for the laws: a shape of up to three axes, zero-size and
   one-element ones included, a layout, and per element a high word and a low
   word [m 2^-k] times it for [|m| <= 1] and [k] from [-80] to [80], so up to
   [2^80] times larger, which [v] normalises. High words take signed zeros,
   infinities, NaN, subnormals and the format's extremes. *)
type drawn = {
  shape : int array;
  layout : Nx_support.layout;
  his : float array;
  los : float array;
}

let special =
  [
    0.;
    -0.;
    Float.infinity;
    Float.neg_infinity;
    Float.nan;
    Float.min_float;
    Float.ldexp 1. (-1074);
    Float.max_float;
    -.Float.max_float;
    Float.ldexp 1. 1020;
    1.;
    -1.;
  ]

let high =
  Gen.frequency
    [
      (2, Gen.float_range (-1e6) 1e6);
      (1, Gen.of_list ~pp:Format.pp_print_float special);
    ]

let low hi (m, k) =
  let lo = Float.ldexp (m *. hi) (-k) in
  if Float.is_finite lo then lo else 0.

let numel s = Array.fold_left ( * ) 1 s

let numbers count =
  let open Gen in
  let* shape = array ~size:(int_range 0 3) (int_range 0 4) in
  let* layout = Nx_support.layouts in
  let n = numel shape in
  let element = pair high (pair (float_range (-1.) 1.) (int_range (-80) 80)) in
  let+ draws = list ~size:(constant count) (array ~size:(constant n) element) in
  List.map
    (fun xs ->
      {
        shape;
        layout;
        his = Array.map fst xs;
        los = Array.map (fun (h, l) -> low h l) xs;
      })
    draws

let one = Gen.pair dtype (Gen.map List.hd (numbers 1))

let two =
  Gen.pair dtype
    (Gen.map (function [ a; b ] -> (a, b) | _ -> assert false) (numbers 2))

(* [d]'s words at [dt], laid out as drawn. *)
let laid dt d =
  let hi, _ = Nx_support.laid d.layout dt d.shape d.his in
  let lo, _ = Nx_support.laid d.layout dt d.shape d.los in
  (hi, lo)

let drawn dt d =
  let hi, lo = laid dt d in
  Nx_wide.v ~lo hi

(* [d]'s number at [dt] in C order, of the words the laid out ones read. *)
let contiguous dt d =
  let hi, lo = laid dt d in
  let c t = Nx.create dt d.shape (Nx.to_array t) in
  Nx_wide.v ~lo:(c lo) (c hi)

(* What a law over drawn numbers needs to reach, at each dtype. *)
let cover_drawn (D (tag, dt, _)) d =
  let hi, lo = laid dt d in
  let his = Nx.to_array hi and los = Nx.to_array lo in
  let at s = s ^ " at " ^ tag in
  cover "no element" (numel d.shape = 0);
  cover "one element" (numel d.shape = 1);
  cover "a strided or broadcast layout"
    (d.layout <> Nx_support.C_order && numel d.shape > 1);
  cover
    (at "a non-finite high word")
    (Array.exists (fun x -> not (Float.is_finite x)) his);
  cover (at "a zero high word") (Array.exists (fun x -> x = 0.) his);
  cover
    (at "a finite low word larger than its high word")
    (Array.exists2
       (fun h l -> Float.is_finite l && Float.abs l > Float.abs h)
       his los)

(* [normal ~msg w] checks that each [hi] is [hi + lo] rounded, and that a
   non-finite [hi] has a zero [lo]. *)
let normal ~msg w =
  let h = Nx_wide.hi w and l = Nx_wide.lo w in
  equal
    ~msg:(msg ^ ": hi is hi + lo rounded")
    (array unsigned)
    (Nx.to_array (Nx.add h l))
    (Nx.to_array h);
  equal
    ~msg:(msg ^ ": a non-finite hi's lo is zero")
    (array unsigned)
    (Nx.to_array (Nx.where (Nx.isfinite h) l (Nx.zeros_like l)))
    (Nx.to_array l)

(* [w] summed along every axis, against each column of [w] and its negation
   stacked on a new last axis: the same shape, laid out apart. *)
let associates_by_shape w =
  let r = Nx.ndim (Nx_wide.hi w) in
  let column f = Nx.stack ~axis:r [ f w; Nx.neg (f w) ] in
  let twice = Nx_wide.v ~lo:(column Nx_wide.lo) (column Nx_wide.hi) in
  let zh, zl = words (Nx_wide.sum ~axes:(List.init r Fun.id) twice) in
  let sh, sl = words (Nx_wide.sum w) in
  (* A zero's sign is the plain sum's, which is [+0]. *)
  let value = pair unsigned unsigned in
  equal value (sh.(0), sl.(0)) (zh.(0), zl.(0));
  equal value (-.sh.(0), -.sl.(0)) (zh.(1), zl.(1))

(* Scalars *)

let f x = Nx.scalar Nx.float64 x
let scalar w = (Nx.item [] (Nx_wide.hi w), Nx.item [] (Nx_wide.lo w))
let same = pair float_exact float_exact
let tiny = Float.ldexp 1. (-60)

(* A NaN's sign and payload are the operation's: any NaN compares as one. *)
let nan_aware =
  Testable.contramap
    (fun (h, l) -> ((if Float.is_nan h then Float.nan else h), l))
    same

(* Numbers *)

(* The binary operations, by name, at float64. *)
let binaries =
  [
    ("add", Nx_wide.add);
    ("sub", Nx_wide.sub);
    ("mul", Nx_wide.mul);
    ("div", Nx_wide.div);
  ]

(* A number of float64 as a structure's module, which moves between sets. *)
module Number = struct
  type ('a, 'd) t = (Nx.float64_elt, 'd) Nx_wide.t

  let walk = Nx_wide.walk
end

let numbers_group =
  let w () = Nx_wide.v ~lo:(f tiny) (f 1.) and z x = Nx_wide.v (f x) in
  group "numbers"
    [
      prop ~count:400
        "every operation's result is normalised, a non-finite one with a zero \
         low word"
        two (fun ((D (_, dt, _) as t), (a, b)) ->
          cover_drawn t a;
          let x = drawn dt a and y = drawn dt b in
          List.iter
            (fun (op, z) -> normal ~msg:op z)
            [
              ("v", x);
              ("add", Nx_wide.add x y);
              ("sub", Nx_wide.sub x y);
              ("mul", Nx_wide.mul x y);
              ("div", Nx_wide.div x y);
              ("floor", Nx_wide.floor x);
              ("sum", Nx_wide.sum x);
            ]);
      prop ~count:400
        "an operation's words do not depend on its operands' layout" two
        (fun ((D (_, dt, _) as t), (a, b)) ->
          cover_drawn t a;
          let x = drawn dt a and y = drawn dt b in
          let x' = contiguous dt a and y' = contiguous dt b in
          let check op z z' = equal ~msg:op exact_words (words z') (words z) in
          check "add" (Nx_wide.add x y) (Nx_wide.add x' y');
          check "sub" (Nx_wide.sub x y) (Nx_wide.sub x' y');
          check "mul" (Nx_wide.mul x y) (Nx_wide.mul x' y');
          check "div" (Nx_wide.div x y) (Nx_wide.div x' y');
          check "floor" (Nx_wide.floor x) (Nx_wide.floor x');
          check "sum" (Nx_wide.sum x) (Nx_wide.sum x'));
      cases "operands broadcast as nx's arithmetic does" ~id:fst binaries
        (fun (_, op) ->
          let a =
            Nx_wide.v
              ~lo:(Nx.create Nx.float64 [| 2; 1 |] [| tiny; -.tiny |])
              (Nx.create Nx.float64 [| 2; 1 |] [| 1.; 2. |])
          and b =
            Nx_wide.v
              ~lo:(Nx.full Nx.float64 [| 3 |] (Float.ldexp 1. (-61)))
              (Nx.create Nx.float64 [| 3 |] [| 3.; 5.; 7. |])
          in
          let wide w =
            let s = [| 2; 3 |] in
            Nx_wide.v
              ~lo:(Nx.copy (Nx.broadcast_to s (Nx_wide.lo w)))
              (Nx.copy (Nx.broadcast_to s (Nx_wide.hi w)))
          in
          equal exact_words (words (op (wide a) (wide b))) (words (op a b)));
      cases
        "where the float operation on the high words is not finite, the result \
         is that float with a zero low word"
        ~id:(fun (n, _, _) -> n)
        [
          ( "an addition that overflows",
            (fun () -> Nx_wide.add (z Float.max_float) (z Float.max_float)),
            Float.infinity );
          ( "a product that overflows",
            (fun () -> Nx_wide.mul (z 1e200) (z (-1e200))),
            Float.neg_infinity );
          ( "a division by zero",
            (fun () -> Nx_wide.div (w ()) (z 0.)),
            Float.infinity );
          ("zero by zero", (fun () -> Nx_wide.div (z 0.) (z 0.)), Float.nan);
          ( "an infinity times one",
            (fun () -> Nx_wide.mul (z Float.infinity) (z 1.)),
            Float.infinity );
          ( "one by an infinity",
            (fun () -> Nx_wide.div (z 1.) (z Float.infinity)),
            0. );
          ( "an infinity less itself",
            (fun () -> Nx_wide.sub (z Float.infinity) (z Float.infinity)),
            Float.nan );
          ( "NaN plus a number",
            (fun () -> Nx_wide.add (z Float.nan) (w ())),
            Float.nan );
          ( "the floor of an infinity",
            (fun () -> Nx_wide.floor (z Float.neg_infinity)),
            Float.neg_infinity );
          ( "the floor of NaN",
            (fun () -> Nx_wide.floor (z Float.nan)),
            Float.nan );
        ]
        (fun (_, op, expected) ->
          equal nan_aware (expected, 0.) (scalar (op ())));
      (* The floor of [1 - 2^-60] is a zero whose high word's floor, [1], is
         not: it is [+0]. *)
      cases
        "a zero result has the sign the float operation on the high words \
         gives a zero, and is +0 otherwise"
        ~id:(fun (n, _, _) -> n)
        [
          ("x - x", (fun () -> Nx_wide.sub (w ()) (w ())), 0.);
          ("-0 * x", (fun () -> Nx_wide.mul (z (-0.)) (w ())), -0.);
          ("-0 / x", (fun () -> Nx_wide.div (z (-0.)) (w ())), -0.);
          ("-0 + -0", (fun () -> Nx_wide.add (z (-0.)) (z (-0.))), -0.);
          ("the floor of -0", (fun () -> Nx_wide.floor (z (-0.))), -0.);
          ("the floor of 0.5", (fun () -> Nx_wide.floor (z 0.5)), 0.);
          ( "the floor of 1 - 2^-60",
            (fun () -> Nx_wide.floor (Nx_wide.v ~lo:(f (-.tiny)) (f 1.))),
            0. );
        ]
        (fun (_, op, expected) -> equal same (expected, 0.) (scalar (op ())));
      test "a placed number computes on its set" (fun () ->
          let module M = (val Nx.devices [ Nx_support.memory 0 ]) in
          let p = Nx.Ptree.place (module Number) M.on (w ()) in
          let lies msg z =
            equal ~msg (Nx_support.placement ()) M.on
              (Nx.placement (Nx_wide.hi z))
          in
          lies "add" (Nx_wide.add p p);
          lies "sub" (Nx_wide.sub p p);
          lies "mul" (Nx_wide.mul p p);
          lies "div" (Nx_wide.div p p);
          lies "floor" (Nx_wide.floor p);
          lies "sum" (Nx_wide.sum p);
          equal ~msg:"value" same
            (2., Float.ldexp 1. (-59))
            (scalar (Nx_wide.add p p)));
    ]

(* Constructing *)

type narrow = N : (float, 's) Nx.dtype -> narrow

let v_group =
  group "v"
    [
      (* [max_float] beside a low word of [-0.375 max_float] lost its low word
         where the words were normalised by a two-sum, whose difference
         overflowed. [1e-20] beside a low word of [1] is the case a fast two-sum
         taken in the words' order loses. *)
      prop ~count:400 "is hi + lo exactly"
        ~examples:
          [
            ( f64,
              {
                shape = [| 1 |];
                layout = Nx_support.C_order;
                his = [| Float.max_float |];
                los = [| -0.375 *. Float.max_float |];
              } );
            ( f64,
              {
                shape = [| 1 |];
                layout = Nx_support.C_order;
                his = [| 1e-20 |];
                los = [| 1. |];
              } );
          ]
        one
        (fun ((D (_, dt, _) as t), d) ->
          cover_drawn t d;
          let hi, lo = laid dt d in
          let given =
            Array.map2 ordered_sum (Nx.to_array hi) (Nx.to_array lo)
          in
          let zh, zl = words (Nx_wide.v ~lo hi) in
          (* The float64 sums of the given words and of the result's, by
             magnitude, are exact, so they agree where the numbers do. A sum
             past the dtype's range is held as that float, an infinity. *)
          let round x = Nx.item [] (Nx.cast dt (Nx.scalar Nx.float64 x)) in
          Array.iteri
            (fun k (s, e) ->
              let msg = Printf.sprintf "element %d" k in
              if Float.is_finite (round s) then
                equal ~msg (pair unsigned unsigned) (s, e)
                  (ordered_sum zh.(k) zl.(k))
              else equal ~msg number (round s, 0.) (zh.(k), zl.(k)))
            given);
      prop ~count:400 "returns a normalised pair's words unchanged" one
        (fun ((D (_, dt, _) as t), d) ->
          cover_drawn t d;
          let w = drawn dt d in
          let w' = Nx_wide.v ~lo:(Nx_wide.lo w) (Nx_wide.hi w) in
          equal exact_words (words w) (words w'));
      test "broadcasts its words against each other" (fun () ->
          let hi = Nx.create Nx.float64 [| 2; 1 |] [| 1.; 2. |]
          and lo = Nx.create Nx.float64 [| 3 |] [| tiny; 0.; -.tiny |] in
          let s = [| 2; 3 |] in
          let w' =
            Nx_wide.v
              ~lo:(Nx.copy (Nx.broadcast_to s lo))
              (Nx.copy (Nx.broadcast_to s hi))
          in
          equal exact_words (words w') (words (Nx_wide.v ~lo hi)));
      cases "holds a non-finite number with a zero low word"
        ~id:(fun (h, l, _) -> Printf.sprintf "%h + %h" h l)
        [
          (Float.infinity, 1., Float.infinity);
          (Float.neg_infinity, 1., Float.neg_infinity);
          (Float.nan, 1., Float.nan);
          (1., Float.infinity, Float.infinity);
        ]
        (fun (h, l, expected) ->
          equal nan_aware (expected, 0.) (scalar (Nx_wide.v ~lo:(f l) (f h))));
      cases "refuses a dtype other than float32 or float64"
        ~id:(fun (N dt) -> Nx.Dtype.name dt)
        [
          N Nx.float16;
          N Nx.bfloat16;
          N Nx.float8_e4m3fn;
          N Nx.float8_e5m2;
          N Nx.float4_e2m1fn;
        ]
        (fun (N dt) ->
          invalid ~by:"Nx_wide.v" (fun () -> Nx_wide.v (Nx.zeros dt [| 2 |])));
      test "refuses words that do not broadcast" (fun () ->
          invalid ~by:"Nx_wide.v" (fun () ->
              Nx_wide.v
                ~lo:(Nx.zeros Nx.float64 [| 3 |])
                (Nx.zeros Nx.float64 [| 2 |])));
    ]

(* Arithmetic *)

let add_group =
  group "add"
    [
      cases "is a + b within 3u² / (1 - 4u) relative" ~id:name dtypes
        (fun (D (_, dt, u) as d) ->
          let rows = golden "add" d in
          within ~bound:(sum_bound u) ~u rows
            (Nx_wide.add (operand dt rows 0) (operand dt rows 2)));
    ]

let sub_group =
  group "sub"
    [
      cases "is a - b within 3u² / (1 - 4u) relative" ~id:name dtypes
        (fun (D (_, dt, u) as d) ->
          let rows = golden "sub" d in
          within ~bound:(sum_bound u) ~u rows
            (Nx_wide.sub (operand dt rows 0) (operand dt rows 2)));
      prop ~count:400 "a number less itself is +0, a non-finite one NaN" one
        (fun ((D (_, dt, _) as t), d) ->
          cover_drawn t d;
          let w = drawn dt d in
          let h, _ = words w in
          let zh, zl = words (Nx_wide.sub w w) in
          equal exact_words
            ( Array.map (fun x -> if Float.is_finite x then 0. else Float.nan) h,
              Array.make (Array.length zl) 0. )
            (zh, zl));
    ]

let mul_group =
  group "mul"
    [
      cases "is a * b within 4u² relative" ~id:name dtypes
        (fun (D (_, dt, u) as d) ->
          let rows = golden "mul" d in
          within
            ~bound:(4. *. u *. u)
            ~u rows
            (Nx_wide.mul (operand dt rows 0) (operand dt rows 2)));
      prop ~count:400 "a product of words with a zero low word is exact" two
        (fun (D (tag, dt, _), (a, b)) ->
          let x = Nx.to_array (Nx_wide.hi (drawn dt a))
          and y = Nx.to_array (Nx_wide.hi (drawn dt b)) in
          let edge = if tag = "f64" then 969 else 102 in
          let inside v =
            v = 0.
            || Float.abs v >= Float.ldexp 1. (-edge)
               && Float.abs v <= Float.ldexp 1. edge
          in
          let keep =
            List.filter
              (fun i -> inside x.(i) && inside y.(i) && inside (x.(i) *. y.(i)))
              (List.init (Int.min (Array.length x) (Array.length y)) Fun.id)
          in
          cover "a product in the domain" (keep <> []);
          let pick a = Array.of_list (List.map (fun i -> a.(i)) keep) in
          let x = pick x and y = pick y in
          let zh, zl =
            words
              (Nx_wide.mul (Nx_wide.v (vector dt x)) (Nx_wide.v (vector dt y)))
          in
          let round v = Nx.item [] (Nx.cast dt (Nx.scalar Nx.float64 v)) in
          Array.iteri
            (fun i zh ->
              let ph = round (x.(i) *. y.(i)) in
              (* The product's rounding error: exact at float64 by [fma], and by
                 the float64 product of float32 words. *)
              let pl =
                if tag = "f64" then Float.fma x.(i) y.(i) (-.ph)
                else (x.(i) *. y.(i)) -. ph
              in
              equal number (ph, pl) (zh, zl.(i)))
            zh);
    ]

let div_group =
  group "div"
    [
      cases "is a / b within 10u² relative" ~id:name dtypes
        (fun (D (_, dt, u) as d) ->
          let rows = golden "div" d in
          within
            ~bound:(10. *. u *. u)
            ~u rows
            (Nx_wide.div (operand dt rows 0) (operand dt rows 2)));
    ]

let floor_group =
  group "floor"
    [
      cases "is the exact floor" ~id:name dtypes (fun (D (_, dt, _) as d) ->
          let rows = golden "floor" d in
          let zh, zl = words (Nx_wide.floor (operand dt rows 0)) in
          (* mpmath's zero has no sign; the numbers group checks zeros'. *)
          equal (array number)
            (Array.map2 (fun h l -> (h, l)) (column rows 2) (column rows 3))
            (Array.map2 (fun h l -> ((if h = 0. then 0. else h), l)) zh zl));
    ]

(* Comparisons *)

let comparisons =
  let flags rows i = Array.map (fun v -> v = 1.) (column rows i) in
  group "comparisons"
    [
      cases "less is a < b exactly" ~id:name dtypes (fun (D (_, dt, _) as d) ->
          let rows = golden "compare" d in
          equal (array bool) (flags rows 4)
            (Nx.to_array (Nx_wide.less (operand dt rows 0) (operand dt rows 2))));
      cases "equal is a = b exactly" ~id:name dtypes (fun (D (_, dt, _) as d) ->
          let rows = golden "compare" d in
          equal (array bool) (flags rows 5)
            (Nx.to_array
               (Nx_wide.equal (operand dt rows 0) (operand dt rows 2))));
      cases "infinities and NaN compare by their high words, NaN false"
        ~id:(fun (a, b, _) -> Printf.sprintf "%h and %h" a b)
        [
          (Float.infinity, Float.infinity, (false, true));
          (Float.neg_infinity, 1., (true, false));
          (1., Float.infinity, (true, false));
          (Float.infinity, 1., (false, false));
          (Float.nan, Float.nan, (false, false));
          (Float.nan, 1., (false, false));
          (1., Float.nan, (false, false));
        ]
        (fun (a, b, expected) ->
          let a = Nx_wide.v (f a) and b = Nx_wide.v (f b) in
          equal (pair bool bool) expected
            (Nx.item [] (Nx_wide.less a b), Nx.item [] (Nx_wide.equal a b)));
    ]

(* Sums *)

(* Rows of one summand count, summed along their last axis as a batch. *)
let sum_rows (D (_, dt, u) as d) =
  let rows = golden "sum" d in
  let counts =
    List.sort_uniq Int.compare (List.map (fun r -> int_of_string r.(0)) rows)
  in
  List.iter
    (fun n ->
      let rows = List.filter (fun r -> int_of_string r.(0) = n) rows in
      let m = List.length rows in
      let word j =
        Nx.create dt [| m; n |]
          (Array.concat
             (List.map
                (fun r ->
                  Array.init n (fun i -> float_of_string r.(1 + (2 * i) + j)))
                rows))
      in
      let zh, zl =
        words (Nx_wide.sum ~axes:[ 1 ] (Nx_wide.v ~lo:(word 1) (word 0)))
      in
      let at i =
        column (List.map (fun r -> Array.sub r (1 + (2 * n)) 4) rows) i
      in
      let r1 = at 0 and r2 = at 1 and r3 = at 2 and s = at 3 in
      let levels = Float.ceil (Float.log2 (float_of_int n)) in
      let gamma = Float.expm1 (levels *. Float.log1p (sum_bound u)) *. slack in
      Array.iteri
        (fun k r1 ->
          at_most
            ~msg:(Printf.sprintf "%d summands, row %d" n k)
            float_exact
            ~than:(gamma *. s.(k))
            (Nx_support.gap (error (zh.(k), zl.(k)) (r1, r2.(k), r3.(k))) 0.))
        r1)
    counts

let sum_group =
  group "sum"
    [
      cases "is within its bound of the exact sum" ~id:name dtypes sum_rows;
      prop ~count:400 "its association depends only on the shape" one
        (fun ((D (_, dt, _) as t), d) ->
          cover_drawn t d;
          associates_by_shape (drawn dt d));
      test "a sum overflowing then meeting an infinity associates by shape"
        (fun () ->
          let m = Float.max_float in
          associates_by_shape
            (Nx_wide.v
               (Nx.create Nx.float64 [| 4; 2; 1 |]
                  [| 0.; 0.; m; 0.; m; 0.; 0.; Float.neg_infinity |])));
      test "a sum of opposite extremes associates by shape" (fun () ->
          let m = Float.max_float in
          let x =
            Nx.create Nx.float64 [| 1; 3; 4 |]
              (Array.init 12 (fun i ->
                   if i = 6 then -.m else if i = 11 then m else 0.))
          in
          associates_by_shape
            (Nx_wide.v
               (Nx.transpose
                  (Nx.broadcast_to [| 2; 4; 3; 1 |] (Nx.transpose x)))));
      (* Every association of these summands gives the one non-finite float. *)
      cases "a sum holding one infinity or NaN among finite summands is it"
        ~id:(fun (n, _, _) -> n)
        [
          ("an infinity", [| 1.; Float.infinity; -2.; tiny |], Float.infinity);
          ("a NaN", [| 1.; Float.nan; -2. |], Float.nan);
        ]
        (fun (_, xs, expected) ->
          List.iter
            (fun (D (tag, dt, _)) ->
              let w = Nx_wide.v (vector dt xs) in
              equal ~msg:tag nan_aware (expected, 0.) (scalar (Nx_wide.sum w)))
            dtypes);
      test "a sum of no summand is +0" (fun () ->
          let w = Nx_wide.v (Nx.zeros Nx.float64 [| 2; 0 |]) in
          equal exact_words
            ([| 0.; 0. |], [| 0.; 0. |])
            (words (Nx_wide.sum ~axes:[ 1 ] w)));
      cases "a sum of -0 terms is +0" ~id:string_of_int [ 1; 2; 3; 4; 5 ]
        (fun n ->
          let w = Nx_wide.v (Nx.full Nx.float64 [| n |] (-0.)) in
          equal same (0., 0.) (scalar (Nx_wide.sum w)));
      test "a negative axis counts from the end" (fun () ->
          let w =
            Nx_wide.v
              (Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |])
          in
          equal (array float_exact) [| 6.; 15. |]
            (fst (words (Nx_wide.sum ~axes:[ -1 ] w))));
      cases "refuses axes it cannot sum" ~id:fst
        [ ("an axis out of bounds", [ 2 ]); ("a repeated axis", [ 0; -2 ]) ]
        (fun (_, axes) ->
          invalid ~by:"Nx_wide.sum" (fun () ->
              Nx_wide.sum ~axes (Nx_wide.v (Nx.zeros Nx.float64 [| 2; 2 |]))));
    ]

(* Structures *)

(* A number alone, described by its walk. *)
let alone () = Nx.Ptree.v Nx_wide.walk

let paths s x =
  List.rev
    (Nx.Ptree.fold s (fun p _ acc -> Nx.Ptree.Path.to_string p :: acc) x [])

let walk_group =
  group "walk"
    [
      test "the words are at hi and lo" (fun () ->
          equal (list string) [ "hi"; "lo" ]
            (paths (alone ()) (Nx_wide.v (Nx.ones Nx.float64 [| 2 |]))));
      test "a structure walks a number at its field's path" (fun () ->
          let s =
            Nx.Ptree.v (fun c (a, b) ->
                let open Nx.Ptree.Walk in
                let a = field c "a" Nx_wide.walk a in
                let b = field c "b" tensor b in
                (a, b))
          in
          equal (list string) [ "a.hi"; "a.lo"; "b" ]
            (paths s (Nx_wide.v (f 1.), f 2.)));
      prop ~count:400 "rebuilds a value it walks unchanged" one
        (fun ((D (_, dt, _) as t), d) ->
          cover_drawn t d;
          let w = drawn dt d in
          let w' = Nx.Ptree.map (alone ()) (fun _ t -> t) w in
          equal exact_words (words w) (words w'));
      test "a value rebuilt from leafwise sums is the number they make"
        (fun () ->
          (* hi words 1 + 1 and low words 2^-60 + 2^-60 make 2 + 2^-59. *)
          let w = Nx_wide.v ~lo:(f tiny) (f 1.) in
          let s = Nx.Ptree.map2 (alone ()) (fun _ a b -> Nx.add a b) w w in
          equal same (2., Float.ldexp 1. (-59)) (scalar s));
      test
        "a rebuilt value walks as the words it was rebuilt from, and reads \
         normalised" (fun () ->
          let p = alone () in
          let s =
            Nx.Ptree.map p
              (fun _ t -> Nx.add t (Nx.ones_like t))
              (Nx_wide.v (f 1.))
          in
          let walked =
            List.rev
              (Nx.Ptree.fold p
                 (fun _ t acc -> Nx.item [] (Nx.cast Nx.float64 t) :: acc)
                 s [])
          in
          (* Words 1 and 0 become 2 and 1, held as 3 and 0. *)
          equal ~msg:"walked" (list float_exact) [ 2.; 1. ] walked;
          equal ~msg:"read" same (3., 0.) (scalar s));
      test
        "a value rebuilt from a low word larger than its high word reads as \
         their sum" (fun () ->
          let s =
            Nx.Ptree.map (alone ())
              (fun p t ->
                Nx.cast (Nx.dtype t)
                  (Nx.full Nx.float64 (Nx.shape t)
                     (if Nx.Ptree.Path.to_string p = "hi" then 1e-20 else 1.)))
              (Nx_wide.v (f 0.))
          in
          equal same (1., 1e-20) (scalar s));
      test "axpy over a number scales and adds the number" (fun () ->
          (* 3 (1 + 2^-60) + (1 + 2^-60) is 4 + 2^-58. *)
          let w = Nx_wide.v ~lo:(f tiny) (f 1.) in
          let s = Nx.Ptree.axpy (alone ()) (f 3.) w w in
          equal same (4., Float.ldexp 1. (-58)) (scalar s));
    ]

let () =
  exit
    (run "nx wide"
       [
         numbers_group;
         v_group;
         add_group;
         sub_group;
         mul_group;
         div_group;
         floor_group;
         comparisons;
         sum_group;
         walk_group;
       ])
