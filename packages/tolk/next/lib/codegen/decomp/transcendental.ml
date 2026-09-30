(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let transcendental_dtypes = Dtype.[ Float16; Float32; Float64 ]
let float_like u x = const_like u (`Float x)
let int_like u n = const_like u (`Int (Z.of_int n))

let not_in what dt =
  invalid_arg (Format.asprintf "%a is not %s" Dtype.pp dt what)

let not_float = not_in "a float16, float32 or float64"
let not_int = not_in "an int16, int32 or int64"

let check d =
  if not (List.mem (dtype d) transcendental_dtypes) then not_float (dtype d)

let int_of_width : Dtype.t -> Dtype.t = function
  | Float64 -> Int64
  | Float32 -> Int32
  | Float16 -> Int16
  | dt -> not_float dt

(* Only frexp reads it, after checking its argument. *)
let uint_of_width : Dtype.t -> Dtype.t = function
  | Float64 -> Uint64
  | Float32 -> Uint32
  | _ -> Uint16

(* inf to inf, -inf to neg_inf, NaN to nan, anything else to ratio. *)
let lazy_map_numbers x ~inf ~neg_inf ~nan ratio =
  where
    O.(x <> float infinity)
    (where O.(x <> x) nan (where O.(x <> float neg_infinity) ratio neg_inf))
    inf

(* Helper functions for bit manipulation *)

let mantissa_bits dt = snd (Dtype.finfo dt)

let exponent_bias dt =
  let fnuz = List.mem dt Dtype.fp8_fnuz in
  (1 lsl (fst (Dtype.finfo dt) - 1)) - if fnuz then 0 else 1

let exponent_mask dt = (1 lsl fst (Dtype.finfo dt)) - 1

(* Utils *)

let pow2 y =
  if y < 0 then invalid_arg (Printf.sprintf "negative shift %d" y);
  const (`Int (Z.shift_left Z.one y))

let shr x y = O.(x // pow2 y)
let shl x y = O.(x * pow2 y)

let rintk d =
  let half = where O.(d < float 0.) (float_like d (-0.5)) (float_like d 0.5) in
  cast O.(d + half) (int_of_width (dtype d))

let pow2if q float_dtype =
  let out : Dtype.t =
    match dtype q with
    | Int64 -> Float64
    | Int32 -> Float32
    | Int16 -> float_dtype
    | dt -> not_int dt
  in
  bitcast (shl O.(q + int (exponent_bias out)) (mantissa_bits out)) out

let ilogb2k d =
  let dt = dtype d in
  let dint = bitcast d (int_of_width dt) in
  O.(
    (shr dint (mantissa_bits dt) land int (exponent_mask dt))
    - int (exponent_bias dt))

let ldexp3k d e =
  let dt = dtype d in
  let m2 = shl (cast e (int_of_width dt)) (mantissa_bits dt) in
  bitcast O.(bitcast d (int_of_width dt) + m2) dt

let ldexp2k d e =
  O.(d * pow2if (shr e 1) (dtype d) * pow2if (e - shr e 1) (dtype d))

let frexp v =
  check v;
  let dt = dtype v in
  (* m1 masks the sign and the mantissa, m2 sets the exponent that normalizes
     the mantissa into [0.5, 1). *)
  let m1, m2 =
    match dt with
    | Float64 -> (0x000FFFFFFFFFFFFF, 0x3FE0000000000000)
    | Float32 -> (0x807FFFFF, 0x3F000000)
    | _ -> (0x83FF, 0x3800)
  in
  let bits = bitcast v (uint_of_width dt) in
  let exponent = O.(shr bits (mantissa_bits dt) land int (exponent_mask dt)) in
  let mantissa = bitcast O.(bits land int m1 lor int m2) dt in
  (mantissa, O.(exponent - int (exponent_bias dt) + int 1))

(* Reduction algorithms for sine *)

(* 1/(2pi) in 32-bit words, the first its integer part: 1312 bits, as a
   float64's greatest exponent needs *)
let one_over_two_pi =
  [|
    0x00000000;
    0x28be60db;
    0x9391054a;
    0x7f09d5f4;
    0x7d4d3770;
    0x36d8a566;
    0x4f10e410;
    0x7f9458ea;
    0xf7aef158;
    0x6dc91b8e;
    0x909374b8;
    0x01924bba;
    0x82746487;
    0x3f877ac7;
    0x2c4a69cf;
    0xba208d7d;
    0x4baed121;
    0x3a671c09;
    0xad17df90;
    0x4e64758e;
    0x60d4ce7d;
    0x272117e2;
    0xef7e4a0e;
    0xc7fe25ff;
    0xf7816603;
    0xfbcbc462;
    0xd6829b47;
    0xdb4d9fb3;
    0xc9f2c26d;
    0xd3d18fd9;
    0xa797fa8b;
    0x5d49eeb1;
    0xfaf97c5e;
    0xcf41ce7d;
    0xe294a4ba;
    0x9afed7ec;
    0x47e35742;
    0x1580cc11;
    0xbf1edaea;
    0xfc33ef08;
    0x26bd0d87;
    0x6a78e458;
  |]

let payne_hanek_reduction d =
  check d;
  let dt = dtype d in
  let intermediate_dtype : Dtype.t = if dt = Float16 then Float32 else dt in
  (* d = m * 2^(e - w), the integer m of w bits in k words of 32 bits *)
  let w, k, max_exp =
    if intermediate_dtype = Float64 then (53, 2, 1024) else (24, 1, 128)
  in
  let f, e = frexp d in
  let m =
    cast O.(cast f intermediate_dtype * float (Float.ldexp 1. w)) Uint64
  in
  let ms =
    List.filteri (fun i _ -> i < k) [ O.(m land int 0xffffffff); shr m 32 ]
  in
  (* d/(2pi) * 2^128 mod 2^128 comes from the words of 1/(2pi) at d's exponent,
     with g words below it for the carries *)
  let g = 2 in
  let l = 4 + k + g in
  let shift = 128 - w in
  let e = maximum O.(cast e Int32 + int shift) (int 0) in
  let i = O.(e // int 32) and s = cast O.(e land int 31) Uint64 in
  let table =
    Array.sub one_over_two_pi 0 (((max_exp - w + 128) / 32) - 3 + l + 1)
  in
  let word offset =
    let ret = ref (int ~dtype:Uint64 0) in
    for count = Array.length table - offset - 1 downto max 0 (-offset) do
      ret :=
        where O.(i <> int count) !ret (int_like !ret table.(count + offset))
    done;
    !ret
  in
  let a = Array.init (l + 1) (fun j -> word (j - 3)) in
  let b =
    List.init l (fun j ->
        let next = a.(j + 1) in
        O.((a.(j) lsl s) lor (next lsr (int 32 - s)) land int 0xffffffff))
  in
  (* columns of 32 bits, from g + 1 words below the binary point to the top
     word *)
  let terms = Array.make (g + 5) [] in
  let push c t = terms.(c + g + 1) <- terms.(c + g + 1) @ [ t ] in
  List.iteri
    (fun k mk ->
      List.iteri
        (fun j bj ->
          let p = O.(mk * bj) and c = k - j + 3 in
          if -g - 1 <= c && c <= 3 then push c O.(p land int 0xffffffff);
          if -g - 1 <= c + 1 && c + 1 <= 3 then push (c + 1) (shr p 32))
        b)
    ms;
  let z, _ =
    Array.fold_left
      (fun (z, carry) ts ->
        let col =
          match ts @ carry with
          | t :: rest -> List.fold_left add t rest
          | [] -> invalid_arg "an empty column"
        in
        (z @ [ O.(col land int 0xffffffff) ], [ shr col 32 ]))
      ([], []) terms
  in
  let zh, zl =
    match List.rev z with
    | z3 :: z2 :: z1 :: z0 :: _ -> (O.(shl z3 32 lor z2), O.(shl z1 32 lor z0))
    | _ -> invalid_arg "fewer than four columns"
  in
  (* The quotient rounds to the nearest quadrant: from half a quadrant up, the
     remainder is the fraction less a quadrant. *)
  let half = O.(shr zh 61 land int 1) in
  let q = cast O.(shr zh 62 + half) Int32 in
  let hi =
    O.(cast (zh land int 0x3fffffffffffffff) Int64 - shl (cast half Int64) 62)
  in
  let r =
    O.(
      (cast hi intermediate_dtype * float (2. *. Float.pi /. Float.ldexp 1. 64))
      + cast zl intermediate_dtype
        * float (2. *. Float.pi /. Float.ldexp 1. 128))
  in
  (cast r dt, q)

let cody_waite_reduction d =
  let m_1_pi = 0.318309886183790671537767526745028724 in
  let dt = dtype d in
  let qdh =
    O.(
      cast (cast (d * float (Float.ldexp m_1_pi (-24))) Int64) dt
      * float (Float.ldexp 1. 24))
  in
  let rec reduce_d x q =
    match dtype x with
    | Float64 ->
        let pi_a, pi_b = (3.1415926218032836914, 3.1786509424591713469e-08) in
        let pi_c, pi_d =
          (1.2246467864107188502e-16, 1.2736634327021899816e-24)
        in
        let d = O.((qdh * float (-.pi_a)) + x) in
        let d = O.((q * float (-.pi_a)) + d) in
        let d = O.((qdh * float (-.pi_b)) + d) in
        let d = O.((q * float (-.pi_b)) + d) in
        let d = O.((qdh * float (-.pi_c)) + d) in
        let d = O.((q * float (-.pi_c)) + d) in
        O.(((qdh + q) * float (-.pi_d)) + d)
    | Float16 ->
        (* float16 reaches 1 ulp only when reduced in float32. *)
        cast (reduce_d (cast x Float32) (cast q Float32)) Float16
    | _ ->
        let d = O.((q * float (-3.1414794921875)) + x) in
        let d = O.((q * float (-0.00011315941810607910156)) + d) in
        let d = O.((q * float (-1.9841872589410058936e-09)) + d) in
        O.((q * float (-1.2154201256553420762e-10)) + d)
  in
  let quadrant =
    if dt = Float64 then rintk O.((d * float m_1_pi) - qdh)
    else rintk O.(d * float m_1_pi)
  in
  (reduce_d d (cast quadrant dt), cast quadrant Int32)

(* Approximate sine on small angle *)

let poly_n x = function
  | c :: cs ->
      List.fold_left (fun acc c -> O.((acc * x) + float c)) (float c) cs
  | [] -> invalid_arg "a polynomial without coefficients"

let trig_poly d coeff32 coeff64 =
  O.(d * poly_n (d * d) (if dtype d = Float64 then coeff64 else coeff32))

(* Approximates sine on [-pi/2, pi/2]. *)
let sin_poly d =
  trig_poly d
    [
      2.6083159809786593541503e-06;
      -0.0001981069071916863322258;
      0.00833307858556509017944336;
      -0.166666597127914428710938;
      1.0;
    ]
    [
      -7.97255955009037868891952e-18;
      2.81009972710863200091251e-15;
      -7.64712219118158833288484e-13;
      1.60590430605664501629054e-10;
      -2.50521083763502045810755e-08;
      2.75573192239198747630416e-06;
      -0.000198412698412696162806809;
      0.00833333333333332974823815;
      -0.166666666666666657414808;
      1.0;
    ]

let ifand q n = O.(q land int n <> int 0)
let sign_if cond r = O.(r * where cond (int_like r (-1)) (int_like r 1))
let sin_poly_small d q = sign_if (ifand q 1) (sin_poly d)

(* An odd quadrant's sine is the cosine of the remainder, sin (pi/2 - |d|),
   within the polynomial's range. *)
let sin_poly_large d q =
  let magnitude = where O.(d < int 0) O.(-d) d in
  let d = where (ifand q 1) O.(float (Float.pi /. 2.) - magnitude) d in
  sign_if (ifand q 2) (sin_poly d)

(* Toplevel functions for xsin, xlog2 and xexp2 *)

let xsin ?(fast = false) ?(switch_over = 30.0) d =
  check d;
  let zero = float_like d 0. in
  let x = lazy_map_numbers d ~inf:zero ~neg_inf:zero ~nan:zero d in
  let x_sign =
    where
      O.(x <> int 0)
      (where O.(x < int 0) (int_like x (-1)) (int_like x 1))
      (int_like x 0)
  in
  let x_abs = O.(x * x_sign) in
  let result =
    if fast then
      let r, q = cody_waite_reduction x_abs in
      sin_poly_small r q
    else
      let r, q = payne_hanek_reduction x_abs in
      (* Payne-Hanek assumes |x| >= pi/4, so smaller angles use Cody-Waite. *)
      let r_small, q_small = cody_waite_reduction x_abs in
      where
        O.(x_abs < float switch_over)
        (sin_poly_small r_small q_small)
        (sin_poly_large r q)
  in
  let nan = float_like d Dtype.nan in
  lazy_map_numbers d ~inf:nan ~neg_inf:nan ~nan O.(result * x_sign)

let xexp2 d =
  check d;
  let zero = float_like d 0. in
  let x = lazy_map_numbers d ~inf:zero ~neg_inf:zero ~nan:zero d in
  let q = rintk x in
  let s = O.(x - q) in
  (* A polynomial with 13 non-zero terms on [-(log 2)/2, (log 2)/2]. *)
  let u =
    if dtype d = Float64 then
      poly_n s
        [
          0.4434359082926529454e-9;
          0.7073164598085707425e-8;
          0.1017819260921760451e-6;
          0.1321543872511327615e-5;
          0.1525273353517584730e-4;
          0.1540353045101147808e-3;
          0.1333355814670499073e-2;
          0.9618129107597600536e-2;
          0.5550410866482046596e-1;
          0.2402265069591012214e+0;
          0.6931471805599452862e+0;
          0.1000000000000000000e+1;
        ]
    else
      poly_n s
        [
          0.1535920892e-3;
          0.1339262701e-2;
          0.9618384764e-2;
          0.5550347269e-1;
          0.2402264476e+0;
          0.6931471825e+0;
          1.0;
        ]
  in
  let u = ldexp2k u q in
  let upper, lower =
    match dtype d with
    | Float64 -> (1024, -2000)
    | Float32 -> (128, -150)
    | _ -> (23, -22)
  in
  let u = where O.(d >= int upper) (float_like d infinity) u in
  let u = where O.(d < int lower) (float_like d 0.) u in
  where O.(d <> d) (float_like d Dtype.nan) u

let xlog2 d =
  check d;
  let dt = dtype d in
  (* float16 scales subnormals by 2^10, since 2^64 overflows it. *)
  let denormal_exp = if dt = Float16 then 10 else 64 in
  let flt_min = float_like d (if dt = Float16 then 6.1e-5 else 1e-4) in
  let is_denormal = O.(d < flt_min) in
  let a = where is_denormal O.(d * float (Float.ldexp 1. denormal_exp)) d in
  let e = cast (ilogb2k O.(a * float (1.0 /. 0.75))) (dtype a) in
  let m = ldexp3k a O.(-e) in
  let e = where is_denormal O.(e - int denormal_exp) e in
  let x = O.((m - float 1.0) / (m + float 1.0)) in
  let x2 = O.(x * x) in
  let r =
    if dt = Float64 then
      let t =
        poly_n x2
          [
            0.2211941750456081490e+0;
            0.2200768693152277689e+0;
            0.2623708057488514656e+0;
            0.3205977477944495502e+0;
            0.4121985945485324709e+0;
            0.5770780162997058982e+0;
            0.96179669392608091449;
          ]
      in
      O.((t * (x * x2)) + e + (x * float 2.885390081777926774))
    else
      let t = poly_n x2 [ 0.4374550283e+0; 0.5764790177e+0; 0.9618012905120 ] in
      let hi = O.((t * (x * x2)) + e + (x * float 2.8853900432586669922)) in
      (* The low part of the constant underflows in float16. *)
      if dt = Float32 then O.(hi + (x * float 3.2734474483568488616e-08))
      else hi
  in
  let r = where O.(d <> float infinity) r (float_like r infinity) in
  let r = where O.(d <> float 0.0) r (float_like r neg_infinity) in
  let r = where O.(d < float (-0.0)) (float_like r Dtype.nan) r in
  let r = where O.(d <> d) (float_like r Dtype.nan) r in
  (* Some targets do not find -0.0 equal to 0.0; its reciprocal is -inf. *)
  where O.(reciprocal d <> float neg_infinity) r (float_like r neg_infinity)

let xpow base exponent =
  let ret =
    exp2 (mul (log2 (where O.(base < int 0) O.(-base) base)) exponent)
  in
  (* A negative base gives NaN for an exponent that is not an integer, and the
     sign of the exponent's parity otherwise; -inf is never NaN, its power stays
     |base| ** exponent. *)
  let non_int = O.(exponent <> cast (cast exponent Int32) (dtype exponent)) in
  let magnitude = where O.(exponent < int 0) O.(-exponent) exponent in
  let is_odd = cast (mod_ (cast magnitude Int32) (int 2)) Bool in
  let neg_base =
    where non_int
      (where O.(base <> float neg_infinity) (float_like ret Dtype.nan) ret)
      (where is_odd O.(-ret) ret)
  in
  (* x ** 0 is 1, 0 ** 0 and inf ** 0 included. *)
  where
    (eq exponent (int 0))
    (int_like ret 1)
    (where O.(base < int 0) neg_base ret)

let patterns ~force ops =
  let other_floats =
    List.filter (fun dt -> not (List.mem dt transcendental_dtypes)) Dtype.floats
  in
  let lowered op = force || not (Op.Set.mem op ops) in
  let d = Upat.var "d" in
  let decompose (op, f) =
    if not (lowered op) then []
    else
      Pattern_matcher.
        [
          rule (Upat.op ~dtype:transcendental_dtypes ~src:[ d ] op) (fun m ->
              Some (f (m "d")));
          rule (Upat.op ~dtype:other_floats ~src:[ d ] ~name:"x" op) (fun m ->
              let x = m "x" in
              Some (cast (alu (cast (m "d") Float32) (Ops.op x) []) (dtype x)));
        ]
  in
  let sqrt =
    if not (lowered Sqrt) then []
    else
      [
        Pattern_matcher.rule (Upat.op ~each:d Sqrt) (fun m ->
            Some (xpow (m "d") (float_like (m "d") 0.5)));
      ]
  in
  let transcendentals =
    [ (Op.Exp2, xexp2); (Log2, xlog2); (Sin, fun d -> xsin d) ]
  in
  Pattern_matcher.v (List.concat_map decompose transcendentals @ sqrt)
