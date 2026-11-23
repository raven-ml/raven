(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A kind of constants, lowered by Rune.Lower.op, is its value computed while it
   lowers: its graph has no more nodes than the constant of its result's type
   and shape, which [Nx.full] lowers to, whatever the kind, the constants and
   the shape. That graph is each part's literal moved to the shape, so its size
   depends on the shape's rank alone, never on its elements. Unfolded, a
   [lgamma] of a constant links over a thousand nodes, and a complex [asinh] as
   many. Eager's bits for these constants are the suites' "of a constant"
   tests. *)

open Windtrap

type real =
  | One of { f1 : 's 'd. (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t }
  | Two of {
      f2 :
        's 'd.
        (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t;
    }
  | Three of {
      f3 :
        's 'd.
        (float, 's, 'd) Nx.t ->
        (float, 's, 'd) Nx.t ->
        (float, 's, 'd) Nx.t ->
        (float, 's, 'd) Nx.t;
    }

type complex =
  | C1 of { c1 : 's 'd. (Complex.t, 's, 'd) Nx.t -> (Complex.t, 's, 'd) Nx.t }
  | C2 of {
      c2 :
        's 'd.
        (Complex.t, 's, 'd) Nx.t ->
        (Complex.t, 's, 'd) Nx.t ->
        (Complex.t, 's, 'd) Nx.t;
    }

let reals =
  [
    ("sin", One { f1 = Nx.sin });
    ("cos", One { f1 = Nx.cos });
    ("tan", One { f1 = Nx.tan });
    ("exp", One { f1 = Nx.exp });
    ("exp2", One { f1 = Nx.exp2 });
    ("expm1", One { f1 = Nx.expm1 });
    ("log", One { f1 = Nx.log });
    ("log2", One { f1 = Nx.log2 });
    ("log1p", One { f1 = Nx.log1p });
    ("sqrt", One { f1 = Nx.sqrt });
    ("rsqrt", One { f1 = Nx.rsqrt });
    ("sinh", One { f1 = Nx.sinh });
    ("cosh", One { f1 = Nx.cosh });
    ("tanh", One { f1 = Nx.tanh });
    ("asin", One { f1 = Nx.asin });
    ("acos", One { f1 = Nx.acos });
    ("atan", One { f1 = Nx.atan });
    ("asinh", One { f1 = Nx.asinh });
    ("acosh", One { f1 = Nx.acosh });
    ("atanh", One { f1 = Nx.atanh });
    ("sinc", One { f1 = Nx.sinc });
    ("erf", One { f1 = Nx.erf });
    ("erfc", One { f1 = Nx.erfc });
    ("erfinv", One { f1 = Nx.erfinv });
    ("ndtr", One { f1 = Nx.ndtr });
    ("log_ndtr", One { f1 = Nx.log_ndtr });
    ("ndtri", One { f1 = Nx.ndtri });
    ("lgamma", One { f1 = Nx.lgamma });
    ("digamma", One { f1 = Nx.digamma });
    ("i0e", One { f1 = Nx.i0e });
    ("i1e", One { f1 = Nx.i1e });
    ("atan2", Two { f2 = Nx.atan2 });
    ("pow", Two { f2 = Nx.pow });
    ("hypot", Two { f2 = Nx.hypot });
    ("lbeta", Two { f2 = Nx.lbeta });
    ("gammainc", Two { f2 = Nx.gammainc });
    ("gammaincc", Two { f2 = Nx.gammaincc });
    ("log_gammainc", Two { f2 = Nx.log_gammainc });
    ("gammaincinv", Two { f2 = Nx.gammaincinv });
    ("gammainccinv", Two { f2 = Nx.gammainccinv });
    ("betainc", Three { f3 = Nx.betainc });
    ("betaincc", Three { f3 = Nx.betaincc });
    ("log_betainc", Three { f3 = Nx.log_betainc });
    ("betaincinv", Three { f3 = Nx.betaincinv });
  ]

let complexes =
  [
    ("sqrt", C1 { c1 = Nx.sqrt });
    ("sign", C1 { c1 = Nx.sign });
    ("recip", C1 { c1 = Nx.recip });
    ("exp", C1 { c1 = Nx.exp });
    ("exp2", C1 { c1 = Nx.exp2 });
    ("log", C1 { c1 = Nx.log });
    ("log2", C1 { c1 = Nx.log2 });
    ("sin", C1 { c1 = Nx.sin });
    ("cos", C1 { c1 = Nx.cos });
    ("tan", C1 { c1 = Nx.tan });
    ("sinh", C1 { c1 = Nx.sinh });
    ("cosh", C1 { c1 = Nx.cosh });
    ("tanh", C1 { c1 = Nx.tanh });
    ("asin", C1 { c1 = Nx.asin });
    ("acos", C1 { c1 = Nx.acos });
    ("atan", C1 { c1 = Nx.atan });
    ("asinh", C1 { c1 = Nx.asinh });
    ("acosh", C1 { c1 = Nx.acosh });
    ("atanh", C1 { c1 = Nx.atanh });
    ("mul", C2 { c2 = Nx.mul });
    ("div", C2 { c2 = Nx.div });
    ("pow", C2 { c2 = Nx.pow });
  ]

(* Cases *)

(* The constants (a part each for a complex kind, the real part first), at
   [float64] or [complex128] where [wide], in the shape [shape]. *)
type case = { wide : bool; values : float list; shape : int array }

let pp_case ppf { wide; values; shape } =
  Format.fprintf ppf "%s [%s] of shape [|%s|]"
    (if wide then "wide" else "narrow")
    (String.concat "; " (List.map (Printf.sprintf "%h") values))
    (String.concat "; " (Array.to_list (Array.map string_of_int shape)))

let specials =
  [ 0.; -0.; 1.; -1.; 0.5; Float.infinity; Float.neg_infinity; Float.nan ]

let gen =
  let open Gen in
  with_pp pp_case
    (let+ wide = bool
     and+ values = list ~size:(constant 6) any_float
     and+ shape = array ~size:(int_range 0 3) (int_range 0 4) in
     { wide; values; shape })

(* Each special value at every operand, at both widths, half of them of no
   element or of a single one. *)
let examples =
  let shapes = [| [||]; [| 0 |]; [| 2; 3 |]; [| 1 |] |] in
  List.concat
    (List.mapi
       (fun i v ->
         List.map
           (fun j ->
             let shape = shapes.(((2 * i) + j) mod 4) in
             { wide = j = 1; values = List.init 6 (fun _ -> v); shape })
           [ 0; 1 ])
       specials)

(* Laws *)

let real_nodes (type s) (dt : (float, s) Nx.dtype) k { values; shape; _ } =
  let x i = Nx.full dt shape (List.nth values i) in
  Lowered.nodes (fun () ->
      match k with
      | One { f1 } -> f1 (x 0)
      | Two { f2 } -> f2 (x 0) (x 1)
      | Three { f3 } -> f3 (x 0) (x 1) (x 2))

let complex_nodes (type s) (dt : (Complex.t, s) Nx.dtype) k { values; shape; _ }
    =
  let z i =
    Nx.full dt shape
      {
        Complex.re = List.nth values (2 * i);
        im = List.nth values ((2 * i) + 1);
      }
  in
  Lowered.nodes (fun () ->
      match k with C1 { c1 } -> c1 (z 0) | C2 { c2 } -> c2 (z 0) (z 1))

(* The most nodes a constant of [shape] lowers to: one of distinct parts. *)
let real_bound dt shape = Lowered.nodes (fun () -> Nx.full dt shape 2.)

let complex_bound dt shape =
  Lowered.nodes (fun () -> Nx.full dt shape { Complex.re = 2.; im = 3. })

let real (name, k) =
  prop ~count:8 ~examples name gen (fun ({ wide; shape; _ } as c) ->
      if wide then
        at_most int
          ~than:(real_bound Nx.float64 shape)
          (real_nodes Nx.float64 k c)
      else
        at_most int
          ~than:(real_bound Nx.float32 shape)
          (real_nodes Nx.float32 k c))

let complex (name, k) =
  prop ~count:8 ~examples name gen (fun ({ wide; shape; _ } as c) ->
      if wide then
        at_most int
          ~than:(complex_bound Nx.complex128 shape)
          (complex_nodes Nx.complex128 k c)
      else
        at_most int
          ~than:(complex_bound Nx.complex64 shape)
          (complex_nodes Nx.complex64 k c))

(* Values

   A kind of constants compiled by Rune.jit gives the values it gives eagerly.
   One compile per dtype holds every kind at every tuple, each result read
   through the argument [x], a one, so that the function reads it. A NaN equals
   any NaN. *)

let tuples arity =
  let vs = specials @ [ 2.5; -3.75; 0x1p-10; 1e30 ] in
  let n = List.length vs in
  List.mapi
    (fun i v ->
      List.init arity (fun j ->
          if j = 0 then v else List.nth vs ((i + (3 * j)) mod n)))
    vs

let arity = function One _ -> 1 | Two _ -> 2 | Three _ -> 3
let carity = function C1 _ -> 1 | C2 _ -> 2

let same a b =
  (Float.is_nan a && Float.is_nan b)
  || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* The names and tuples of [rows] whose compiled and eager values differ. *)
let differing rows ~eq compiled eager =
  List.concat
    (List.mapi
       (fun i row -> if eq compiled.(i) eager.(i) then [] else [ row ])
       rows)

let real_values (type s) (dt : (float, s) Nx.dtype) () =
  let rows =
    List.concat_map
      (fun (name, k) -> List.map (fun vs -> (name, vs)) (tuples (arity k)))
      reals
  in
  let f x =
    let c = Nx.full_like x in
    Nx.concatenate ~axis:0
      (List.concat_map
         (fun (_, k) ->
           List.map
             (fun vs ->
               Nx.mul x
                 (match (k, vs) with
                 | One { f1 }, [ a ] -> f1 (c a)
                 | Two { f2 }, [ a; b ] -> f2 (c a) (c b)
                 | Three { f3 }, [ a; b; d ] -> f3 (c a) (c b) (c d)
                 | _ -> assert false))
             (tuples (arity k)))
         reals)
  in
  let x = Nx.full dt [| 1 |] 1. in
  let eager = Nx.to_array (f x) and compiled = Nx.to_array (Rune.jit' f x) in
  let show (name, vs) =
    Printf.sprintf "%s %s" name
      (String.concat " " (List.map (Printf.sprintf "%h") vs))
  in
  equal (list string) []
    (List.map show (differing rows ~eq:same compiled eager))

let complex_values (type s) (dt : (Complex.t, s) Nx.dtype) () =
  let pair = function
    | a :: b :: rest -> ({ Complex.re = a; im = b }, rest)
    | _ -> assert false
  in
  let rows =
    List.concat_map
      (fun (name, k) -> List.map (fun vs -> (name, vs)) (tuples (2 * carity k)))
      complexes
  in
  let f x =
    let c = Nx.full_like x in
    Nx.concatenate ~axis:0
      (List.concat_map
         (fun (_, k) ->
           List.map
             (fun vs ->
               let a, rest = pair vs in
               Nx.where (Nx.equal x x)
                 (match k with
                 | C1 { c1 } -> c1 (c a)
                 | C2 { c2 } -> c2 (c a) (c (fst (pair rest))))
                 x)
             (tuples (2 * carity k)))
         complexes)
  in
  let x = Nx.full dt [| 1 |] Complex.one in
  let eager = Nx.to_array (f x) and compiled = Nx.to_array (Rune.jit' f x) in
  let eq (a : Complex.t) (b : Complex.t) = same a.re b.re && same a.im b.im in
  let show (name, vs) =
    Printf.sprintf "%s %s" name
      (String.concat " " (List.map (Printf.sprintf "%h") vs))
  in
  equal (list string) [] (List.map show (differing rows ~eq compiled eager))

(* Integers: each operand an int64 constant converted to the dtype, which wraps
   it, and each result converted back. *)

type int_op = {
  op : 'v 's 'd. ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t;
}

(* Each operation, and whether it reads one operand alone. *)
let int_ops =
  [
    ("add", false, { op = Nx.add });
    ("sub", false, { op = Nx.sub });
    ("mul", false, { op = Nx.mul });
    ("div", false, { op = Nx.div });
    ("mod", false, { op = Nx.mod_ });
    ("maximum", false, { op = Nx.maximum });
    ("minimum", false, { op = Nx.minimum });
    ("and", false, { op = Nx.bitwise_and });
    ("or", false, { op = Nx.bitwise_or });
    ("xor", false, { op = Nx.bitwise_xor });
    ("neg", true, { op = (fun a _ -> Nx.neg a) });
    ("abs", true, { op = (fun a _ -> Nx.abs a) });
    ("not", true, { op = (fun a _ -> Nx.bitwise_not a) });
    ("lshift 1", true, { op = (fun a _ -> Nx.lshift a 1) });
    ("lshift 5", true, { op = (fun a _ -> Nx.lshift a 5) });
    ("rshift 1", true, { op = (fun a _ -> Nx.rshift a 1) });
    ("rshift 7", true, { op = (fun a _ -> Nx.rshift a 7) });
  ]

let int_specials =
  [
    0L;
    1L;
    -1L;
    2L;
    7L;
    -7L;
    100L;
    255L;
    256L;
    32767L;
    -32768L;
    65535L;
    2147483647L;
    -2147483648L;
    4294967295L;
    Int64.max_int;
    Int64.min_int;
  ]

(* Each of [int_specials] beside the one after it and two others, for an
   operation of two operands: [0] and [-1] are each a divisor of every value,
   whose quotient Nx defines. *)
let int_pairs unary =
  let n = List.length int_specials in
  List.concat
    (List.mapi
       (fun i a ->
         if unary then [ (a, a) ]
         else
           List.init 3 (fun j ->
               (a, List.nth int_specials ((i + 1 + (j * 6)) mod n))))
       int_specials)
  @
  if unary then []
  else
    List.map (fun a -> (a, 0L)) int_specials
    @ List.map (fun a -> (a, -1L)) int_specials

let int_values (type v s) (dt : (v, s) Nx.dtype) () =
  let rows =
    List.concat_map
      (fun (name, unary, _) ->
        List.map (fun ab -> (name, ab)) (int_pairs unary))
      int_ops
  in
  let f x =
    let c v = Nx.cast dt (Nx.full_like x v) in
    Nx.concatenate ~axis:0
      (List.concat_map
         (fun (_, unary, { op }) ->
           List.map
             (fun (a, b) -> Nx.mul x (Nx.cast Nx.int64 (op (c a) (c b))))
             (int_pairs unary))
         int_ops)
  in
  let x = Nx.full Nx.int64 [| 1 |] 1L in
  let eager = Nx.to_array (f x) and compiled = Nx.to_array (Rune.jit' f x) in
  let show (name, (a, b)) = Printf.sprintf "%s %Ld %Ld" name a b in
  equal (list string) []
    (List.map show (differing rows ~eq:Int64.equal compiled eager))

(* Conversions: each float of [float_casts] to each integer dtype that holds its
   truncation, a conversion C defines, and each integer of [int_specials] to
   each float dtype. *)
let float_casts =
  [
    0.;
    -0.;
    0.5;
    -0.5;
    0.99;
    127.9;
    -128.9;
    255.5;
    3e4;
    -3e4;
    65535.;
    2e9;
    -2e9;
    4e9;
    9e18;
    -9.2e18;
  ]

type int_dtype =
  | Int_dtype : string * ('v, 's) Nx.dtype * float * float -> int_dtype

let int_dtypes =
  [
    Int_dtype ("int8", Nx.int8, -128., 127.);
    Int_dtype ("uint8", Nx.uint8, 0., 255.);
    Int_dtype ("int16", Nx.int16, -32768., 32767.);
    Int_dtype ("uint16", Nx.uint16, 0., 65535.);
    Int_dtype ("int32", Nx.int32, -2147483648., 2147483647.);
    Int_dtype ("uint32", Nx.uint32, 0., 4294967295.);
    Int_dtype ("int64", Nx.int64, -0x1p63, 0x1.fffffffffffffp62);
    Int_dtype ("uint64", Nx.uint64, 0., 0x1.fffffffffffffp63);
  ]

let conversions () =
  let held lo hi v = lo <= Float.trunc v && Float.trunc v <= hi in
  let to_ints =
    List.concat_map
      (fun (Int_dtype (name, dt, lo, hi)) ->
        List.filter_map
          (fun v ->
            if held lo hi v then
              Some
                ( Printf.sprintf "%h to %s" v name,
                  fun x ->
                    Nx.mul x
                      (Nx.cast Nx.int64
                         (Nx.cast dt (Nx.full_like (Nx.cast Nx.float64 x) v)))
                )
            else None)
          float_casts)
      int_dtypes
  in
  let to_floats =
    List.concat_map
      (fun v ->
        List.map
          (fun (name, cast) ->
            (Printf.sprintf "%Ld to %s" v name, fun x -> cast x v))
          [
            ( "float64",
              fun x v ->
                Nx.mul x
                  (Nx.bitcast Nx.int64 (Nx.cast Nx.float64 (Nx.full_like x v)))
            );
            ( "float32",
              fun x v ->
                Nx.mul x
                  (Nx.cast Nx.int64
                     (Nx.bitcast Nx.int32
                        (Nx.cast Nx.float32 (Nx.full_like x v)))) );
            ( "float16",
              fun x v ->
                Nx.mul x
                  (Nx.cast Nx.int64
                     (Nx.bitcast Nx.int16
                        (Nx.cast Nx.float16 (Nx.full_like x v)))) );
          ])
      int_specials
  in
  let rows = to_ints @ to_floats in
  let f x = Nx.concatenate ~axis:0 (List.map (fun (_, g) -> g x) rows) in
  let x = Nx.full Nx.int64 [| 1 |] 1L in
  let eager = Nx.to_array (f x) and compiled = Nx.to_array (Rune.jit' f x) in
  equal (list string) []
    (List.map fst (differing rows ~eq:Int64.equal compiled eager))

let () =
  exit
    (run "Folded"
       [
         group "a real kind of constants is a constant" (List.map real reals);
         group "a complex kind of constants is a constant"
           (List.map complex complexes);
         group "a kind of constants compiles to its eager values"
           [
             test "float32" (real_values Nx.float32);
             test "float64" (real_values Nx.float64);
             test "float16" (real_values Nx.float16);
             test "complex64" (complex_values Nx.complex64);
             test "complex128" (complex_values Nx.complex128);
             test "int8" (int_values Nx.int8);
             test "uint8" (int_values Nx.uint8);
             test "int16" (int_values Nx.int16);
             test "uint16" (int_values Nx.uint16);
             test "int32" (int_values Nx.int32);
             test "uint32" (int_values Nx.uint32);
             test "int64" (int_values Nx.int64);
             test "uint64" (int_values Nx.uint64);
             test "conversions" conversions;
           ];
       ])
