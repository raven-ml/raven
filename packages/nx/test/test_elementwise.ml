(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Element-wise arithmetic, math, comparison and bitwise operations. Each is
   checked against OCaml's own function on every element, over every layout and
   over the values that break arithmetic: NaN, infinities, signed zeros,
   subnormals and the extremes of each type. *)

open Windtrap
open Nx_test

let pp_float ppf x = Format.fprintf ppf "%.17g" x
let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let floats dtype = viewed ~pp:pp_float dtype Gen.any_float

(* Unary float operations *)

type unary = {
  name : string;
  nx : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t;
  ocaml : float -> float;
  exact : bool;  (** correctly rounded, as IEEE requires of it *)
}

let sign x =
  if Float.is_nan x then x else if x > 0. then 1. else if x < 0. then -1. else x

let unary =
  [
    { name = "abs"; nx = Nx.abs; ocaml = Float.abs; exact = true };
    { name = "neg"; nx = Nx.neg; ocaml = Float.neg; exact = true };
    { name = "sign"; nx = Nx.sign; ocaml = sign; exact = true };
    { name = "square"; nx = Nx.square; ocaml = (fun x -> x *. x); exact = true };
    { name = "sqrt"; nx = Nx.sqrt; ocaml = Float.sqrt; exact = true };
    { name = "recip"; nx = Nx.recip; ocaml = (fun x -> 1. /. x); exact = true };
    { name = "trunc"; nx = Nx.trunc; ocaml = Float.trunc; exact = true };
    { name = "ceil"; nx = Nx.ceil; ocaml = Float.ceil; exact = true };
    { name = "floor"; nx = Nx.floor; ocaml = Float.floor; exact = true };
    { name = "round"; nx = Nx.round; ocaml = Float.round; exact = true };
    {
      name = "relu";
      nx = Nx.relu;
      ocaml = (fun x -> Float.max x 0.);
      exact = true;
    };
    {
      name = "rsqrt";
      nx = Nx.rsqrt;
      ocaml = (fun x -> 1. /. Float.sqrt x);
      exact = false;
    };
    { name = "log"; nx = Nx.log; ocaml = Float.log; exact = false };
    { name = "log2"; nx = Nx.log2; ocaml = Float.log2; exact = false };
    { name = "exp"; nx = Nx.exp; ocaml = Float.exp; exact = false };
    { name = "exp2"; nx = Nx.exp2; ocaml = Float.exp2; exact = false };
    { name = "sin"; nx = Nx.sin; ocaml = Float.sin; exact = false };
    { name = "cos"; nx = Nx.cos; ocaml = Float.cos; exact = false };
    { name = "tan"; nx = Nx.tan; ocaml = Float.tan; exact = false };
    { name = "asin"; nx = Nx.asin; ocaml = Float.asin; exact = false };
    { name = "acos"; nx = Nx.acos; ocaml = Float.acos; exact = false };
    { name = "atan"; nx = Nx.atan; ocaml = Float.atan; exact = false };
    { name = "sinh"; nx = Nx.sinh; ocaml = Float.sinh; exact = false };
    { name = "cosh"; nx = Nx.cosh; ocaml = Float.cosh; exact = false };
    { name = "tanh"; nx = Nx.tanh; ocaml = Float.tanh; exact = false };
    { name = "asinh"; nx = Nx.asinh; ocaml = Float.asinh; exact = false };
    { name = "acosh"; nx = Nx.acosh; ocaml = Float.acosh; exact = false };
    { name = "atanh"; nx = Nx.atanh; ocaml = Float.atanh; exact = false };
    { name = "erf"; nx = Nx.erf; ocaml = Float.erf; exact = false };
    {
      name = "sigmoid";
      nx = Nx.sigmoid;
      ocaml = (fun x -> 1. /. (1. +. Float.exp (-.x)));
      exact = false;
    };
  ]

(* A transcendental function is within a few units in the last place of the
   correctly rounded result. *)
let tolerance ~f32 exact =
  if exact then 0.
  else if f32 then 4. *. epsilon_float *. (2. ** 29.)
  else 4. *. epsilon_float

let agrees ~f32 (u : unary) t =
  let rel = tolerance ~f32 u.exact in
  let round = if f32 then to_f32 else Fun.id in
  let r = Ref.of_nx t in
  equal
    (Ref.witness (close ~rel ()))
    { r with data = Array.map (fun x -> round (u.ocaml x)) r.data }
    (Ref.of_nx (u.nx t))

let unary_ops =
  group "unary operations"
    (List.concat_map
       (fun u ->
         [
           prop
             (u.name ^ " at float64 agrees with OCaml's")
             (floats Nx.float64) (agrees ~f32:false u);
           prop
             (u.name ^ " at float32 agrees with OCaml's, rounded")
             (floats Nx.float32) (agrees ~f32:true u);
         ])
       unary)

let classifiers =
  group "classifiers"
    [
      prop "isnan, isinf and isfinite classify every float" (floats Nx.float64)
        (fun t ->
          let r = Ref.of_nx t in
          let classify f = { r with data = Array.map f r.data } in
          let bools = Ref.witness bool in
          equal bools (classify Float.is_nan) (Ref.of_nx (Nx.isnan t));
          equal bools
            (classify (fun x -> Float.abs x = infinity))
            (Ref.of_nx (Nx.isinf t));
          equal bools (classify Float.is_finite) (Ref.of_nx (Nx.isfinite t)));
      test "isnan, isinf and isfinite of integers say finite" (fun () ->
          let t = Nx.create Nx.int32 [| 2 |] [| 0l; Int32.max_int |] in
          equal (tensor bool) (Nx.zeros Nx.bool [| 2 |]) (Nx.isnan t);
          equal (tensor bool) (Nx.zeros Nx.bool [| 2 |]) (Nx.isinf t);
          equal (tensor bool) (Nx.ones Nx.bool [| 2 |]) (Nx.isfinite t));
      test
        "erfinv inverts erf on (-1, 1), is infinite at the ends and NaN beyond"
        (fun () ->
          let t = Nx.create Nx.float64 [| 5 |] [| -1.; 1.; 1.5; -2.; nan |] in
          equal
            (tensor (close ~rel:0. ()))
            (Nx.create Nx.float64 [| 5 |]
               [| neg_infinity; infinity; nan; nan; nan |])
            (Nx.erfinv t));
      prop "erf of erfinv keeps about seven digits at float32"
        (Gen.array ~size:(Gen.int_range 0 8) (Gen.float_range (-0.999) 0.999))
        (fun xs ->
          let t = Nx.create Nx.float32 [| Array.length xs |] xs in
          equal (tensor (close ~rel:1e-6 ())) t (Nx.erf (Nx.erfinv t)));
      prop "erf of erfinv is the identity at float64"
        (Gen.array ~size:(Gen.int_range 0 8)
           (Gen.float_range (-0.999999) 0.999999))
        (fun xs ->
          let t = Nx.create Nx.float64 [| Array.length xs |] xs in
          equal (tensor (close ~rel:1e-14 ())) t (Nx.erf (Nx.erfinv t)));
    ]

(* Binary float operations, over operands that broadcast. *)

let broadcast_pair dtype =
  let open Gen in
  let* a = floats dtype in
  let s = Nx.shape a in
  let* kind = int_range 0 2 in
  let b_shape =
    match kind with
    | 0 -> s
    | 1 -> [||]
    | _ -> Array.mapi (fun i d -> if i mod 2 = 0 then 1 else d) s
  in
  let+ xs = array ~size:(constant (Ref.numel b_shape)) any_float
  and+ swap = bool in
  let b = Nx.create dtype b_shape xs in
  if swap then (b, a) else (a, b)

type binary = {
  bname : string;
  bnx : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t;
  bocaml : float -> float -> float;
  bexact : bool;
}

let binary =
  [
    { bname = "add"; bnx = Nx.add; bocaml = ( +. ); bexact = true };
    { bname = "sub"; bnx = Nx.sub; bocaml = ( -. ); bexact = true };
    { bname = "mul"; bnx = Nx.mul; bocaml = ( *. ); bexact = true };
    { bname = "div"; bnx = Nx.div; bocaml = ( /. ); bexact = true };
    { bname = "maximum"; bnx = Nx.maximum; bocaml = Float.max; bexact = true };
    { bname = "minimum"; bnx = Nx.minimum; bocaml = Float.min; bexact = true };
    {
      bname = "mod_ (the sign of the dividend, as C's fmod; nx.mli is silent)";
      bnx = Nx.mod_;
      bocaml = Float.rem;
      bexact = false;
    };
    { bname = "pow"; bnx = Nx.pow; bocaml = Float.pow; bexact = false };
    { bname = "atan2"; bnx = Nx.atan2; bocaml = Float.atan2; bexact = false };
    { bname = "hypot"; bnx = Nx.hypot; bocaml = Float.hypot; bexact = false };
  ]

let binary_ops =
  group "binary operations"
    (List.concat_map
       (fun b ->
         let check ~f32 (x, y) =
           let rel = tolerance ~f32 b.bexact in
           let round = if f32 then to_f32 else Fun.id in
           equal
             (Ref.witness (close ~rel ()))
             (Ref.map2
                (fun u v -> round (b.bocaml u v))
                (Ref.of_nx x) (Ref.of_nx y))
             (Ref.of_nx (b.bnx x y))
         in
         [
           prop
             (b.bname ^ " at float64 agrees with OCaml's")
             (broadcast_pair Nx.float64)
             (check ~f32:false);
           prop
             (b.bname ^ " at float32 agrees with OCaml's, rounded")
             (broadcast_pair Nx.float32)
             (check ~f32:true);
         ])
       binary)

let refusals =
  test "a binary operation refuses shapes that do not broadcast" (fun () ->
      raises_invalid_arg (fun () ->
          Nx.add
            (Nx.zeros Nx.float32 [| 2; 3 |])
            (Nx.zeros Nx.float32 [| 3; 2 |])))

let comparisons =
  let cmp name nx ocaml =
    prop (name ^ " agrees with OCaml's IEEE comparison")
      (broadcast_pair Nx.float64) (fun (x, y) ->
        equal (Ref.witness bool)
          (Ref.map2 ocaml (Ref.of_nx x) (Ref.of_nx y))
          (Ref.of_nx (nx x y)))
  in
  group "comparisons"
    [
      cmp "less" Nx.less (fun (a : float) b -> a < b);
      cmp "less_equal" Nx.less_equal (fun (a : float) b -> a <= b);
      cmp "greater" Nx.greater (fun (a : float) b -> a > b);
      cmp "greater_equal" Nx.greater_equal (fun (a : float) b -> a >= b);
      cmp "equal" Nx.equal (fun (a : float) b -> a = b);
      cmp "not_equal" Nx.not_equal (fun (a : float) b -> a <> b);
      prop "where picks from the first where the condition holds"
        (broadcast_pair Nx.float64) (fun (x, y) ->
          let r = Ref.map2 (fun a b -> (a, b)) (Ref.of_nx x) (Ref.of_nx y) in
          let cond =
            { r with data = Array.map (fun (a, _) -> a < 0.) r.data }
          in
          equal
            (Ref.witness (close ~rel:0. ()))
            {
              r with
              data = Array.map (fun (a, b) -> if a < 0. then a else b) r.data;
            }
            (Ref.of_nx (Nx.where (Nx.create Nx.bool cond.shape cond.data) x y)));
      prop "clamp is minimum of the upper bound and maximum of the lower"
        (Gen.triple (floats Nx.float64)
           (Gen.float_range (-10.) 0.)
           (Gen.float_range 0. 10.))
        (fun (t, lo, hi) ->
          equal
            (tensor (close ~rel:0. ()))
            (Nx.minimum_s (Nx.maximum_s t lo) hi)
            (Nx.clamp ~min:lo ~max:hi t));
      prop "lerp is a + w (b - a)" (broadcast_pair Nx.float64) (fun (a, b) ->
          let w = Nx.scalar Nx.float64 0.25 in
          equal
            (tensor (close ~rel:0. ()))
            (Nx.add a (Nx.mul w (Nx.sub b a)))
            (Nx.lerp a b w));
    ]

(* Integers wrap, as OCaml's fixed-width integers do. *)

let int32_value =
  Gen.frequency
    [
      (4, Gen.map Int32.of_int (Gen.int_range (-9) 9));
      (2, Gen.int32);
      ( 1,
        Gen.of_list
          ~pp:(fun ppf v -> Format.fprintf ppf "%ld" v)
          [ Int32.min_int; Int32.max_int; -1l; 0l ] );
    ]

let pp_int32 ppf v = Format.fprintf ppf "%ld" v
let int32s = viewed ~pp:pp_int32 Nx.int32 int32_value

(* Tensors of one shape, for the laws. *)
let int32_tuple n =
  let open Gen in
  let* s = array ~size:(int_range 0 3) (int_range 0 3) in
  let* steps = layout in
  let one = viewed ~shape:(constant s) ~layout:(constant ~pp:pp_layout steps) in
  list ~size:(constant n) (one ~pp:pp_int32 Nx.int32 int32_value)

let pair_of g =
  Gen.map (function [ a; b ] -> (a, b) | _ -> assert false) (g 2)

let triple_of g =
  Gen.map (function [ a; b; c ] -> (a, b, c) | _ -> assert false) (g 3)

let ints = tensor int32

let int_laws =
  let pairs = pair_of int32_tuple and triples = triple_of int32_tuple in
  group "integer laws"
    [
      prop "add is associative" triples (Law.associative ints Nx.add);
      prop "add is commutative" pairs (Law.commutative ints Nx.add);
      prop "zero is neutral for add" int32s (fun x ->
          Law.neutral ints Nx.add (Nx.zeros_like x) x);
      prop "neg inverts add" int32s (fun x ->
          Law.invertible ints Nx.add (Nx.zeros_like x) Nx.neg x);
      prop "mul is associative" triples (Law.associative ints Nx.mul);
      prop "mul is commutative" pairs (Law.commutative ints Nx.mul);
      prop "one is neutral for mul" int32s (fun x ->
          Law.neutral ints Nx.mul (Nx.ones_like x) x);
      prop "zero absorbs mul" int32s (fun x ->
          Law.absorbing ints Nx.mul (Nx.zeros_like x) x);
      prop "mul distributes over add" triples
        (Law.distributive ints Nx.mul ~over:Nx.add);
      prop "bitwise_and is associative" triples
        (Law.associative ints Nx.bitwise_and);
      prop "bitwise_or is associative" triples
        (Law.associative ints Nx.bitwise_or);
      prop "bitwise_xor is associative" triples
        (Law.associative ints Nx.bitwise_xor);
      prop "bitwise_xor is its own inverse" int32s (fun x ->
          Law.invertible ints Nx.bitwise_xor (Nx.zeros_like x) Fun.id x);
      prop "bitwise_and distributes over bitwise_or" triples
        (Law.distributive ints Nx.bitwise_and ~over:Nx.bitwise_or);
      prop "neg is involutive" int32s (Law.involutive ints Nx.neg);
      prop "bitwise_not is involutive" int32s
        (Law.involutive ints Nx.bitwise_not);
      prop "maximum is associative" triples (Law.associative ints Nx.maximum);
      prop "minimum is commutative" pairs (Law.commutative ints Nx.minimum);
    ]

let int_ops =
  let agree name nx ocaml =
    prop (name ^ " agrees with Int32's") (pair_of int32_tuple) (fun (a, b) ->
        equal (Ref.witness int32)
          (Ref.map2 ocaml (Ref.of_nx a) (Ref.of_nx b))
          (Ref.of_nx (nx a b)))
  in
  let nonzero =
    let open Gen in
    let* s = array ~size:(int_range 0 3) (int_range 0 3) in
    let* steps = layout in
    let one =
      viewed ~shape:(constant s) ~layout:(constant ~pp:pp_layout steps)
    in
    pair
      (one ~pp:pp_int32 Nx.int32 int32_value)
      (one ~pp:pp_int32 Nx.int32 (such_that (fun v -> v <> 0l) int32_value))
  in
  group "integer operations"
    [
      agree "add" Nx.add Int32.add;
      agree "sub" Nx.sub Int32.sub;
      agree "mul" Nx.mul Int32.mul;
      agree "maximum" Nx.maximum max;
      agree "bitwise_and" Nx.bitwise_and Int32.logand;
      agree "bitwise_or" Nx.bitwise_or Int32.logor;
      agree "bitwise_xor" Nx.bitwise_xor Int32.logxor;
      prop "div truncates toward zero, and mod_ has the sign of the dividend"
        nonzero (fun (a, b) ->
          let ra = Ref.of_nx a and rb = Ref.of_nx b in
          equal (Ref.witness int32) (Ref.map2 Int32.div ra rb)
            (Ref.of_nx (Nx.div a b));
          equal (Ref.witness int32) (Ref.map2 Int32.rem ra rb)
            (Ref.of_nx (Nx.mod_ a b)));
      prop "neg, abs and bitwise_not agree with Int32's" int32s (fun t ->
          let r = Ref.of_nx t in
          let map f = { r with data = Array.map f r.data } in
          equal (Ref.witness int32) (map Int32.neg) (Ref.of_nx (Nx.neg t));
          equal (Ref.witness int32) (map Int32.abs) (Ref.of_nx (Nx.abs t));
          equal (Ref.witness int32) (map Int32.lognot)
            (Ref.of_nx (Nx.bitwise_not t)));
      prop "lshift and rshift agree with Int32's arithmetic shifts"
        (Gen.pair int32s (Gen.int_range 0 31))
        (fun (t, n) ->
          let r = Ref.of_nx t in
          let map f = { r with data = Array.map f r.data } in
          equal (Ref.witness int32)
            (map (fun v -> Int32.shift_left v n))
            (Ref.of_nx (Nx.lshift t n));
          equal (Ref.witness int32)
            (map (fun v -> Int32.shift_right v n))
            (Ref.of_nx (Nx.rshift t n)));
      test "shifts refuse a negative count and a float dtype" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.lshift (Nx.zeros Nx.int32 [| 2 |]) (-1));
          raises_invalid_arg (fun () ->
              Nx.rshift (Nx.zeros Nx.int32 [| 2 |]) (-1));
          raises_invalid_arg (fun () ->
              Nx.lshift (Nx.zeros Nx.float32 [| 2 |]) 1));
      prop "logical operations read non-zero as true" (pair_of int32_tuple)
        (fun (a, b) ->
          let truth v = v <> 0l and of_bool c = if c then 1l else 0l in
          let ra = Ref.of_nx a and rb = Ref.of_nx b in
          let both f =
            Ref.map2 (fun x y -> of_bool (f (truth x) (truth y))) ra rb
          in
          equal (Ref.witness int32) (both ( && ))
            (Ref.of_nx (Nx.logical_and a b));
          equal (Ref.witness int32) (both ( || ))
            (Ref.of_nx (Nx.logical_or a b));
          equal (Ref.witness int32) (both ( <> ))
            (Ref.of_nx (Nx.logical_xor a b));
          equal (Ref.witness int32)
            {
              ra with
              data = Array.map (fun v -> of_bool (not (truth v))) ra.data;
            }
            (Ref.of_nx (Nx.logical_not a)));
    ]

(* Narrow integers wrap at their width. *)

type small = Small : string * (int, 'b) Nx.dtype * int * bool -> small

let wrap bits signed v =
  let m = 1 lsl bits in
  let v = ((v mod m) + m) mod m in
  if signed && v >= m / 2 then v - m else v

let narrow_ints =
  group "narrow integers"
    (List.map
       (fun (Small (name, dtype, bits, signed)) ->
         let lo = if signed then -(1 lsl (bits - 1)) else 0 in
         let hi = if signed then (1 lsl (bits - 1)) - 1 else (1 lsl bits) - 1 in
         let values =
           Gen.array ~size:(Gen.int_range 0 9)
             (Gen.frequency
                [
                  (3, Gen.int_range lo hi);
                  (1, Gen.of_list ~pp:Format.pp_print_int [ lo; hi; 0 ]);
                ])
         in
         prop (name ^ " arithmetic and bits wrap at its width")
           (Gen.pair values values) (fun (xs, ys) ->
             let n = Int.min (Array.length xs) (Array.length ys) in
             let t v = Nx.create dtype [| n |] (Array.sub v 0 n) in
             let expect f =
               Array.init n (fun i -> wrap bits signed (f xs.(i) ys.(i)))
             in
             let check msg f nx =
               equal ~msg (array int) (expect f)
                 (Nx.to_array (nx (t xs) (t ys)))
             in
             check "add" ( + ) Nx.add;
             check "sub" ( - ) Nx.sub;
             check "mul" ( * ) Nx.mul;
             check "bitwise_and" ( land ) Nx.bitwise_and;
             check "bitwise_or" ( lor ) Nx.bitwise_or;
             check "bitwise_xor" ( lxor ) Nx.bitwise_xor;
             check "neg" (fun x _ -> -x) (fun a _ -> Nx.neg a)))
       [
         Small ("int8", Nx.int8, 8, true);
         Small ("uint8", Nx.uint8, 8, false);
         Small ("int16", Nx.int16, 16, true);
         Small ("uint16", Nx.uint16, 16, false);
       ])

let packed_ints =
  test "int4 and uint4 refuse arithmetic (nx.mli is silent)" (fun () ->
      raises_match
        (function Failure _ | Invalid_argument _ -> true | _ -> false)
        (fun () -> Nx.add (Nx.zeros Nx.int4 [| 2 |]) (Nx.zeros Nx.int4 [| 2 |])))

(* Complex numbers *)

let pp_complex ppf (z : Complex.t) = Format.fprintf ppf "(%g, %g)" z.re z.im
let component = Gen.float_range (-100.) 100.

let complex_value =
  Gen.(
    let+ re = component and+ im = component in
    Complex.{ re; im })

let complexes = viewed ~pp:pp_complex Nx.complex128 complex_value

(* Complex numbers within [rel] of the larger modulus. *)
let complex_close ~rel =
  Testable.make
    ~pp:(fun ppf (z : Complex.t) ->
      Format.fprintf ppf "(%.17g, %.17g)" z.re z.im)
    ~equal:(fun (a : Complex.t) (b : Complex.t) ->
      let same x y = (Float.is_nan x && Float.is_nan y) || x = y in
      (same a.re b.re && same a.im b.im)
      || Complex.norm (Complex.sub a b)
         <= rel *. Float.max (Complex.norm a) (Complex.norm b))

let complex_pair =
  let open Gen in
  let* s = array ~size:(int_range 0 3) (int_range 0 3) in
  let* steps = layout in
  let one = viewed ~shape:(constant s) ~layout:(constant ~pp:pp_layout steps) in
  pair
    (one ~pp:pp_complex Nx.complex128 complex_value)
    (one ~pp:pp_complex Nx.complex128 complex_value)

let complex_numbers =
  let agree name rel nx ocaml =
    prop (name ^ " agrees with Stdlib.Complex") complex_pair (fun (a, b) ->
        equal
          (Ref.witness (complex_close ~rel))
          (Ref.map2 ocaml (Ref.of_nx a) (Ref.of_nx b))
          (Ref.of_nx (nx a b)))
  in
  let parts t = (Nx.real Nx.float64 t, Nx.imag Nx.float64 t) in
  let floats64 = tensor (close ~rel:0. ()) in
  group "complex numbers"
    [
      agree "add" 0. Nx.add Complex.add;
      agree "sub" 0. Nx.sub Complex.sub;
      agree "mul" 1e-15 Nx.mul Complex.mul;
      (* Division by zero has no one convention: the divisors are non-zero. *)
      prop "div agrees with Stdlib.Complex" complex_pair (fun (a, b) ->
          let rb = Ref.of_nx b in
          assume (Array.for_all (fun z -> z <> Complex.zero) rb.data);
          equal
            (Ref.witness (complex_close ~rel:1e-14))
            (Ref.map2 Complex.div (Ref.of_nx a) rb)
            (Ref.of_nx (Nx.div a b)));
      prop
        "real, imag, magnitude, angle and conjugate agree with Stdlib.Complex"
        complexes (fun z ->
          let r = Ref.of_nx z in
          let map f = { r with data = Array.map f r.data } in
          let reals = Ref.witness (close ~rel:1e-15 ()) in
          equal reals
            (map (fun z -> z.Complex.re))
            (Ref.of_nx (Nx.real Nx.float64 z));
          equal reals
            (map (fun z -> z.Complex.im))
            (Ref.of_nx (Nx.imag Nx.float64 z));
          equal reals (map Complex.norm) (Ref.of_nx (Nx.magnitude Nx.float64 z));
          equal reals (map Complex.arg) (Ref.of_nx (Nx.angle Nx.float64 z));
          equal
            (Ref.witness (complex_close ~rel:0.))
            (map Complex.conj)
            (Ref.of_nx (Nx.conjugate z)));
      prop "complex assembles what real and imag take apart" complexes
        (Law.round_trip
           (tensor (complex_close ~rel:0.))
           (pair floats64 floats64) parts
           (fun (re, im) -> Nx.complex Nx.complex128 ~re ~im));
      cases "the sign of a zero imaginary part picks the side of the branch cut"
        ~name:(fun (re, im, _) -> Printf.sprintf "angle (%g, %g)" re im)
        [
          (-1., 0., Float.pi);
          (-1., -0., -.Float.pi);
          (-0., 0., Float.pi);
          (0., 0., 0.);
        ]
        (fun (re, im, expected) ->
          equal (close ~rel:0. ()) expected
            (Nx.item []
               (Nx.angle Nx.float64 (Nx.scalar Nx.complex128 { re; im }))));
      test "real and magnitude read non-finite components exactly" (fun () ->
          let z =
            Nx.create Nx.complex128 [| 3 |]
              [|
                { re = infinity; im = nan };
                { re = -.infinity; im = 0. };
                { re = 1e300; im = 1e300 };
              |]
          in
          equal floats64
            (Nx.create Nx.float64 [| 3 |] [| infinity; -.infinity; 1e300 |])
            (Nx.real Nx.float64 z);
          equal
            (tensor (close ~rel:1e-15 ()))
            (Nx.create Nx.float64 [| 3 |]
               [| infinity; infinity; 1e300 *. Float.sqrt 2. |])
            (Nx.magnitude Nx.float64 z));
      test
        "imag, angle and conjugate of a non-finite real part are NaN, as \
         documented" (fun () ->
          let z =
            Nx.create Nx.complex128 [| 2 |]
              [| { re = infinity; im = 1. }; { re = nan; im = 1. } |]
          in
          equal floats64 (Nx.full Nx.float64 [| 2 |] nan) (Nx.imag Nx.float64 z);
          equal floats64
            (Nx.full Nx.float64 [| 2 |] nan)
            (Nx.angle Nx.float64 z));
    ]

(* float16 and bfloat16 compute as float32 and round once, which gives the
   correctly rounded result: float32 carries more than twice their precision. *)

type narrow = Narrow : string * (float, 'b) Nx.dtype -> narrow

let narrow_floats =
  let narrow_pair (type b) (dt : (float, b) Nx.dtype) =
    Gen.map
      (fun (a, b) -> (Nx.cast dt a, Nx.cast dt b))
      (broadcast_pair Nx.float32)
  in
  group "narrow floats"
    (List.concat_map
       (fun (Narrow (name, dt)) ->
         let exact = tensor (close ~rel:0. ()) in
         let wide t = Nx.cast Nx.float32 t in
         [
           prop (name ^ " arithmetic is float32's, rounded once")
             (narrow_pair dt) (fun (a, b) ->
               let once f = Nx.cast dt (f (wide a) (wide b)) in
               equal ~msg:"add" exact (once Nx.add) (Nx.add a b);
               equal ~msg:"sub" exact (once Nx.sub) (Nx.sub a b);
               equal ~msg:"mul" exact (once Nx.mul) (Nx.mul a b);
               equal ~msg:"div" exact (once Nx.div) (Nx.div a b);
               equal ~msg:"maximum" exact (once Nx.maximum) (Nx.maximum a b);
               equal ~msg:"less" (tensor bool)
                 (Nx.less (wide a) (wide b))
                 (Nx.less a b);
               equal ~msg:"sqrt" exact
                 (Nx.cast dt (Nx.sqrt (wide a)))
                 (Nx.sqrt a));
         ])
       [ Narrow ("float16", Nx.float16); Narrow ("bfloat16", Nx.bfloat16) ]
    @ [
        test "float16 and bfloat16 sums accumulate wider than they store"
          (fun () ->
            equal float_exact 4096.
              (Nx.item [] (Nx.sum (Nx.ones Nx.float16 [| 4096 |])));
            equal float_exact 1024.
              (Nx.item [] (Nx.sum (Nx.ones Nx.bfloat16 [| 1024 |]))));
      ])

(* Data types *)

type packed = D : string * ('a, 'b) Nx.dtype -> packed

let dtypes =
  group "data types"
    [
      cases "each dtype says whether it is float, complex, int or unsigned"
        ~name:(fun (D (name, _), _) -> name)
        [
          (D ("float16", Nx.float16), (true, false, false, false));
          (D ("bfloat16", Nx.bfloat16), (true, false, false, false));
          (D ("float8_e4m3", Nx.float8_e4m3), (true, false, false, false));
          (D ("float8_e5m2", Nx.float8_e5m2), (true, false, false, false));
          (D ("float64", Nx.float64), (true, false, false, false));
          (D ("complex64", Nx.complex64), (false, true, false, false));
          (D ("int4", Nx.int4), (false, false, true, false));
          (D ("uint4", Nx.uint4), (false, false, true, true));
          (D ("uint32", Nx.uint32), (false, false, true, true));
          (D ("uint64", Nx.uint64), (false, false, true, true));
          (D ("bool", Nx.bool), (false, false, false, false));
        ]
        (fun (D (_, dt), expected) ->
          equal (quad bool bool bool bool) expected
            ( Nx_dtype.is_float dt,
              Nx_dtype.is_complex dt,
              Nx_dtype.is_int dt,
              Nx_dtype.is_uint dt ));
      test "narrow integers hold their width's range" (fun () ->
          equal (pair int int) (-8, 7)
            (Nx_dtype.min_value Nx.int4, Nx_dtype.max_value Nx.int4);
          equal (pair int int) (0, 15)
            (Nx_dtype.min_value Nx.uint4, Nx_dtype.max_value Nx.uint4);
          equal (pair int32 int32) (0l, -1l)
            (Nx_dtype.min_value Nx.uint32, Nx_dtype.max_value Nx.uint32);
          equal (pair int64 int64) (0L, -1L)
            (Nx_dtype.min_value Nx.uint64, Nx_dtype.max_value Nx.uint64);
          equal (pair bool bool) (false, true)
            (Nx_dtype.min_value Nx.bool, Nx_dtype.max_value Nx.bool));
    ]

let () =
  exit
    (run "nx elementwise"
       [
         unary_ops;
         classifiers;
         binary_ops;
         refusals;
         comparisons;
         int_laws;
         int_ops;
         narrow_ints;
         packed_ints;
         complex_numbers;
         narrow_floats;
         dtypes;
       ])
