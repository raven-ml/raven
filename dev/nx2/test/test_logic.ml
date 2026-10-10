(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Truth, bits and float classes, elementwise: each function against the
   element its doc states, computed in OCaml from the operands' elements, at
   every dtype it takes over values drawn from bytes. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let same_float a b =
  (Float.is_nan a && Float.is_nan b)
  || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let same (type v s) (dt : (v, s) D.t) (a : v) (b : v) =
  match D.kind dt with
  | D.Float -> same_float a b
  | D.Complex -> same_float a.Complex.re b.Complex.re && same_float a.im b.im
  | D.Signed | D.Unsigned | D.Boolean -> a = b

let elements dt =
  Testable.make ~pp:(Format.pp_print_list (D.pp_value dt)) ~equal:(fun a b ->
      List.length a = List.length b && List.for_all2 (same dt) a b)

let equal_elements dt a b =
  equal (elements dt) (Array.to_list a) (Array.to_list b)

(* A host value of [dt] and shape [s] over drawn bytes: NaNs, signed zeros,
   infinities and extremes among its elements. *)
let drawn (type v s) (dt : (v, s) D.t) s : (v, s, Nx.host) Nx.t Gen.t =
  let open Gen in
  let n = Array.fold_left ( * ) 1 s in
  let+ data = string_of ~size:(constant (D.bytes dt n)) char in
  let data =
    match dt with
    | D.Bool -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | _ -> data
  in
  Nx.Repr.of_array Nx.Host.v
    (A.v dt (L.contiguous s)
       (Rig.Buffer.of_string (if data = "" then "\000" else data)))

let shape =
  Gen.array ~size:(Gen.int_range 0 3)
    (Gen.frequency
       [ (1, Gen.constant 0); (1, Gen.constant 1); (4, Gen.int_range 2 4) ])

type one = One : ('v, 's) D.t * ('v, 's, Nx.host) Nx.t -> one
type two = Two : ('v, 's) D.t * ('v, 's, Nx.host) Nx.t * ('v, 's, Nx.host) Nx.t -> two

let pp_value ppf x =
  Format.fprintf ppf "%a %a" D.pp (Nx.dtype x) pp_ints (Nx.shape x)

let of_dtypes dts =
  Gen.of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) dts

let one dts =
  Gen.with_pp
    (fun ppf (One (_, x)) -> pp_value ppf x)
    (let open Gen in
     let* (D.Any dt) = of_dtypes dts in
     let* s = shape in
     let+ x = drawn dt s in
     One (dt, x))

(* Two operands, the second of the first's shape or one that broadcasts to
   it. *)
let two =
  Gen.with_pp
    (fun ppf (Two (_, a, b)) -> Format.fprintf ppf "%a, %a" pp_value a pp_value b)
    (let open Gen in
     let* (D.Any dt) = of_dtypes D.all in
     let* s = shape in
     let* narrow = bool in
     let s' = if narrow then Array.map (fun _ -> 1) s else s in
     let* a = drawn dt s in
     let+ b = drawn dt s' in
     Two (dt, a, b))

let all = D.all
let integers = List.filter (fun (D.Any dt) -> D.is D.Signed dt || D.is D.Unsigned dt) all

(* Truth *)

(* [v] is not zero: a NaN is not, nor a complex number with a part that is not
   zero. *)
let truth (type v s) (dt : (v, s) D.t) (v : v) =
  match D.kind dt with
  | D.Float -> v <> 0.
  | D.Complex -> v.Complex.re <> 0. || v.im <> 0.
  | D.Boolean -> v
  | D.Signed | D.Unsigned -> v <> D.zero dt

let of_truth dt t = if t then D.one dt else D.zero dt

(* [a] and [b] broadcast together, as arrays of elements in C order. *)
let broadcast2 a b =
  match Nx.broadcast_arrays [ a; b ] with
  | [ a; b ] -> (Nx.to_array a, Nx.to_array b)
  | _ -> assert false

let law_logical
    (f : 'v 's. ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t)
    op (Two (dt, a, b)) =
  let xs, ys = broadcast2 a b in
  let expected = Array.map2 (fun x y -> of_truth dt (op (truth dt x) (truth dt y))) xs ys in
  equal_elements dt expected (Nx.to_array (f a b))

let law_not (One (dt, x)) =
  equal_elements dt
    (Array.map (fun v -> of_truth dt (not (truth dt v))) (Nx.to_array x))
    (Nx.to_array (Nx.logical_not x))

let truths =
  group "truth"
    [
      prop "logical_and is one where both are true" two
        (law_logical (fun a b -> Nx.logical_and a b) ( && ));
      prop "logical_or is one where either is true" two
        (law_logical (fun a b -> Nx.logical_or a b) ( || ));
      prop "logical_xor is one where exactly one is true" two
        (law_logical (fun a b -> Nx.logical_xor a b) ( <> ));
      prop "logical_not is one where x is zero" (one all) law_not;
      test "a NaN and a negative zero" (fun () ->
          let x = Nx.create Nx.float32 [| 3 |] [| nan; -0.; 2. |] in
          equal (array float_exact) [| 1.; 0.; 1. |]
            (Nx.to_array (Nx.logical_or x x)));
    ]

(* Bits *)

(* [x]'s elements as int64 integers, by the cast's modulo rule, and back. *)
let int64s x = Nx.to_array (Nx.cast Nx.int64 x)

let of_int64s dt s vs = Nx.to_array (Nx.cast dt (Nx.create Nx.int64 s vs))

let law_not_bits (One (dt, x)) =
  let flipped = Array.map Int64.lognot (int64s x) in
  equal_elements dt (of_int64s dt (Nx.shape x) flipped)
    (Nx.to_array (Nx.bitwise_not x))

let law_lshift (One (dt, x), n) =
  cover "past the width" (n >= D.bits dt);
  let shifted =
    Array.map (fun v -> if n >= 64 then 0L else Int64.shift_left v n) (int64s x)
  in
  equal_elements dt (of_int64s dt (Nx.shape x) shifted) (Nx.to_array (Nx.lshift x n))

(* A signed integer shifts arithmetically, an unsigned one logically: its
   int64 is never negative but for uint64's, which shifts logically there. *)
let law_rshift (One (dt, x), n) =
  cover "past the width" (n >= D.bits dt);
  let unsigned = D.is D.Unsigned dt in
  let shift v =
    if n >= 64 then if unsigned || Int64.compare v 0L >= 0 then 0L else -1L
    else if unsigned then Int64.shift_right_logical v n
    else Int64.shift_right v n
  in
  equal_elements dt
    (of_int64s dt (Nx.shape x) (Array.map shift (int64s x)))
    (Nx.to_array (Nx.rshift x n))

let integer_shift = Gen.pair (one integers) (Gen.int_range 0 70)

let bits =
  group "bits"
    [
      prop "bitwise_not flips every bit"
        (one (List.filter (fun (D.Any dt) -> not (D.is D.Boolean dt)) integers))
        law_not_bits;
      test "bitwise_not of booleans is not" (fun () ->
          equal (array bool) [| true; false |]
            (Nx.to_array (Nx.bitwise_not (Nx.create Nx.bool [| 2 |] [| false; true |]))));
      test "bitwise_not of a bit is not" (fun () ->
          equal (array bool) [| true; false |]
            (Nx.to_array (Nx.bitwise_not (Nx.create Nx.bit [| 2 |] [| false; true |]))));
      prop "lshift is multiplication by 2^n modulo the width" integer_shift
        law_lshift;
      prop "rshift divides by 2^n toward negative infinity" integer_shift
        law_rshift;
      cases "refusals"
        ~name:(fun (n, _, _) -> n)
        [
          ("bitwise_not of a float", "Nx.bitwise_not",
            fun () -> ignore (Nx.bitwise_not (Nx.zeros Nx.float32 [| 1 |])));
          ("lshift of a negative count", "Nx.lshift",
            fun () -> ignore (Nx.lshift (Nx.zeros Nx.int32 [| 1 |]) (-1)));
          ("lshift of a boolean", "Nx.lshift",
            fun () -> ignore (Nx.lshift (Nx.zeros Nx.bool [| 1 |]) 1));
          ("rshift of a negative count", "Nx.rshift",
            fun () -> ignore (Nx.rshift (Nx.zeros Nx.uint8 [| 1 |]) (-2)));
          ("rshift of a float", "Nx.rshift",
            fun () -> ignore (Nx.rshift (Nx.zeros Nx.float64 [| 1 |]) 2));
        ]
        (fun (_, by, f) -> invalid ~by f);
    ]

(* Float classes *)

let classify (type v s) (dt : (v, s) D.t) (p : float -> bool) join other (v : v) =
  match D.kind dt with
  | D.Float -> p v
  | D.Complex -> join (p v.Complex.re) (p v.im)
  | D.Signed | D.Unsigned | D.Boolean -> other

let law_class (f : 'v 's. ('v, 's, Nx.host) Nx.t -> Nx.host Nx.bool_t) p join
    other (One (dt, x)) =
  equal (array bool)
    (Array.map (classify dt p join other) (Nx.to_array x))
    (Nx.to_array (f x))

let classes =
  group "float classes"
    [
      prop "isnan" (one all) (law_class (fun x -> Nx.isnan x) Float.is_nan ( || ) false);
      prop "isinf" (one all)
        (law_class (fun x -> Nx.isinf x) (fun v -> Float.abs v = infinity) ( || ) false);
      prop "isfinite" (one all) (law_class (fun x -> Nx.isfinite x) Float.is_finite ( && ) true);
      test "float8_e5m2's largest finite value and its infinity" (fun () ->
          let x =
            Nx.bitcast Nx.float8_e5m2 (Nx.create Nx.uint8 [| 3 |] [| 0x7b; 0x7c; 0xfc |])
          in
          equal (array bool) [| false; true; true |] (Nx.to_array (Nx.isinf x));
          equal (array bool) [| true; false; false |] (Nx.to_array (Nx.isfinite x)));
      test "a format without infinities saturates, so nothing is infinite"
        (fun () ->
          let x = Nx.create Nx.float8_e4m3fn [| 2 |] [| 448.; nan |] in
          equal (array bool) [| false; false |] (Nx.to_array (Nx.isinf x));
          equal (array bool) [| true; false |] (Nx.to_array (Nx.isfinite x)));
    ]

(* Clamp *)

let law_clamp (One (dt, x), (lo, hi)) =
  let pick = function
    | None -> None
    | Some j ->
        let vs = Nx.to_array x in
        if Array.length vs = 0 then None else Some vs.(j mod Array.length vs)
  in
  let lo = pick lo and hi = pick hi in
  let expected =
    let v = match lo with None -> x | Some lo -> Nx.maximum x (Nx.scalar dt lo) in
    match hi with None -> v | Some hi -> Nx.minimum v (Nx.scalar dt hi)
  in
  equal_elements dt (Nx.to_array expected) (Nx.to_array (Nx.clamp ?min:lo ?max:hi x))

let clamps =
  group "clamp"
    [
      prop "clamp is minimum of maximum, each bound left out not applied"
        (Gen.pair (one all) (Gen.pair (Gen.option Gen.nat) (Gen.option Gen.nat)))
        law_clamp;
      test "a crossed pair of bounds gives the upper" (fun () ->
          equal (array int) [| 2; 2 |]
            (Nx.to_array (Nx.clamp ~min:5 ~max:2 (Nx.create Nx.int8 [| 2 |] [| 0; 9 |]))));
      test "clamp keeps a NaN" (fun () ->
          let y = Nx.clamp ~min:0. ~max:1. (Nx.create Nx.float32 [| 3 |] [| nan; -2.; 3. |]) in
          equal (array float_exact) [| nan; 0.; 1. |] (Nx.to_array y));
      test "clamp refuses an int outside the dtype" (fun () ->
          invalid ~by:"Nx.clamp" (fun () ->
              Nx.clamp ~max:300 (Nx.zeros Nx.uint8 [| 1 |])));
    ]

let () = exit (run "nx logic" [ truths; bits; classes; clamps ])
