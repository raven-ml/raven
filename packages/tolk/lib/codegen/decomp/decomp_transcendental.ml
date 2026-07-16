(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Transcendentals *)

open Tolk_uop


let transcendental_scalars = [ Dtype.Float16; Dtype.Float32; Dtype.Float64 ]

let is_transcendental dt = List.mem dt transcendental_scalars

let require_transcendental name node =
  if not (is_transcendental (Uop.dtype node)) then
    invalid_arg
      (Printf.sprintf
         "Decomp_transcendental.%s: expected float16, float32, or float64"
         name)

let const_float_v v x = Uop.const (Const.float v x)
let const_float_dt dt x = const_float_v dt x
let const_int64_v v n = Uop.const (Const.int64 v n)

let float_div lhs rhs =
  Uop.alu_binary ~op:Ops.Mul ~lhs
    ~rhs:(Uop.alu_unary ~op:Ops.Reciprocal ~src:rhs)

(* Shift by a simplified constant [Uop.t] amount [y]. Power-of-two
   multiply/divide encodes the shift on [x]'s own integer dtype, matching
   tinygrad's helper-level expectation that [y.simplify().arg] exists. *)
let shl x y =
  let v = Uop.dtype x in
  match Uop.const_int_value y with
  | Some n ->
      Uop.alu_binary ~op:Ops.Mul ~lhs:x
        ~rhs:(const_int64_v v (Int64.shift_left 1L n))
  | None ->
      invalid_arg "Decomp_transcendental.shl: shift amount must be a constant"

let shr x y =
  let v = Uop.dtype x in
  match Uop.const_int_value y with
  | Some n ->
      Uop.alu_binary ~op:Ops.Floordiv ~lhs:x
        ~rhs:(const_int64_v v (Int64.shift_left 1L n))
  | None ->
      invalid_arg "Decomp_transcendental.shr: shift amount must be a constant"

(* IEEE 754 bit-layout helpers. All operate on a [Uop.t] with a
   floating-point dtype; sizes are taken from [Dtype.finfo]. *)

let mantissa_bits dt = snd (Dtype.finfo dt)
let exponent_bias dt =
  let e, _ = Dtype.finfo dt in
  let bias = (1 lsl (e - 1)) - 1 in
  (* fp8 fnuz variants omit the usual [-1] offset. *)
  match dt with
  | Dtype.Fp8e4m3fnuz | Dtype.Fp8e5m2fnuz -> bias + 1
  | _ -> bias
let exponent_mask dt =
  let e, _ = Dtype.finfo dt in (1 lsl e) - 1

(* Integer dtype paired with each supported float width. *)
let int_for_float = function
  | Dtype.Float64 -> Dtype.int64
  | Dtype.Float32 -> Dtype.int32
  | Dtype.Float16 -> Dtype.int16
  | _ -> Dtype.int32

let uint_for_float = function
  | Dtype.Float64 -> Dtype.uint64
  | Dtype.Float32 -> Dtype.uint32
  | Dtype.Float16 -> Dtype.uint16
  | _ -> Dtype.uint32

let float_for_int = function
  | Dtype.Int64 -> Dtype.float64
  | Dtype.Int32 -> Dtype.float32
  | Dtype.Int16 -> Dtype.float16
  | _ -> Dtype.float32

(* A [const_like] that works for floats. [Uop.const_like] only accepts
   integer dtypes. *)
let fconst_like node x =
  let v = Uop.dtype node in
  Uop.const (Const.float v x)

(* [rintk d] rounds [d] to the nearest integer away from zero, returning a
   value in the integer dtype matching [d]'s float width. *)
let rintk d =
  let fdt = Uop.dtype d in
  let out = int_for_float fdt in
  let zero = fconst_like d 0.0 in
  let open Uop.O in
  let bias = where (d < zero) (fconst_like d (-0.5)) (fconst_like d 0.5) in
  cast out (d + bias)

(* [pow2if q float_dtype] is [(float)(2^q)] for integer [q] whose range
   fits a biased exponent of [float_dtype]. *)
let pow2if q float_dtype =
  let qdt = Uop.dtype q in
  (* int16 pairs with the caller's float width; int32/int64 with f32/f64. *)
  let out = match qdt with Dtype.Int16 -> float_dtype | _ -> float_for_int qdt in
  let qv = qdt in
  let q_biased =
    Uop.alu_binary ~op:Ops.Add ~lhs:q
      ~rhs:(const_int64_v qv (Int64.of_int (exponent_bias out)))
  in
  let shifted = shl q_biased (Uop.const (Const.int qv (mantissa_bits out))) in
  Uop.bitcast ~src:shifted ~dtype:out

(* [ilogb2k d] is the integer part of [log2 d] for [d] in [\[0, +inf)].
   Bit-extracts the exponent field. *)
let ilogb2k d =
  let fdt = Uop.dtype d in
  let int_dt = int_for_float fdt in
  let iv = int_dt in
  let dint = Uop.bitcast ~src:d ~dtype:int_dt in
  let mb = Uop.const (Const.int iv (mantissa_bits fdt)) in
  let mask = const_int64_v iv (Int64.of_int (exponent_mask fdt)) in
  let masked = Uop.alu_binary ~op:Ops.And ~lhs:(shr dint mb) ~rhs:mask in
  Uop.alu_binary ~op:Ops.Add ~lhs:masked
    ~rhs:(const_int64_v iv (Int64.of_int (-exponent_bias fdt)))

(* [ldexp3k d e] is [d * 2^e] via direct manipulation of the exponent
   field. Safe for any [d] (including denormals). *)
let ldexp3k d e =
  let fdt = Uop.dtype d in
  let int_dt = int_for_float fdt in
  let iv = int_dt in
  let m1 = Uop.bitcast ~src:d ~dtype:int_dt in
  let e_int = Uop.cast ~src:e ~dtype:int_dt in
  let mb = Uop.const (Const.int iv (mantissa_bits fdt)) in
  let m2 = shl e_int mb in
  Uop.bitcast ~src:(Uop.alu_binary ~op:Ops.Add ~lhs:m1 ~rhs:m2) ~dtype:fdt

(* [ldexp2k d e] is [d * 2^e] via two fp multiplies. Faster than
   [ldexp3k] but requires [d > 0] and non-denormal. *)
let ldexp2k d e =
  let fdt = Uop.dtype d in
  let one = Uop.const (Const.int (Uop.dtype e) 1) in
  let half = shr e one in
  let other = Uop.alu_binary ~op:Ops.Sub ~lhs:e ~rhs:half in
  let mul = Uop.alu_binary ~op:Ops.Mul in
  mul ~lhs:(mul ~lhs:d ~rhs:(pow2if half fdt)) ~rhs:(pow2if other fdt)

(* [frexp v] returns [(mantissa, exponent)] assuming [v <> 0]. The
   mantissa is normalized into [\[0.5, 1.0)]. *)
let frexp v =
  let fdt = Uop.dtype v in
  let m1_raw, m2_raw = match fdt with
    | Dtype.Float64 -> 0x000FFFFFFFFFFFFFL, 0x3FE0000000000000L
    | Dtype.Float32 -> 0x807FFFFFL, 0x3F000000L
    | Dtype.Float16 -> 0x83FFL, 0x3800L
    | _ -> 0x807FFFFFL, 0x3F000000L
  in
  let uint_dt = uint_for_float fdt in
  let uv = uint_dt in
  let bits = Uop.bitcast ~src:v ~dtype:uint_dt in
  let mb = Uop.const (Const.int uv (mantissa_bits fdt)) in
  let mask_exp = const_int64_v uv (Int64.of_int (exponent_mask fdt)) in
  let exponent =
    Uop.alu_binary ~op:Ops.And ~lhs:(shr bits mb) ~rhs:mask_exp
  in
  let mantissa =
    Uop.bitcast ~dtype:fdt
      ~src:(Uop.alu_binary ~op:Ops.Or
              ~lhs:(Uop.alu_binary ~op:Ops.And ~lhs:bits
                      ~rhs:(const_int64_v uv m1_raw))
              ~rhs:(const_int64_v uv m2_raw))
  in
  let exp =
    Uop.alu_binary ~op:Ops.Add
      ~lhs:(Uop.alu_binary ~op:Ops.Add ~lhs:exponent
              ~rhs:(const_int64_v uv
                      (Int64.neg (Int64.of_int (exponent_bias fdt)))))
      ~rhs:(const_int64_v uv 1L)
  in
  mantissa, exp

(* [_lazy_map_numbers x inf ninf nan ratio] expresses
   [match x with inf -> inf | -inf -> _inf | nan -> nan | _ -> ratio]
   as nested [where]s. *)
let lazy_map_numbers x ~inf ~ninf ~nan ~ratio =
  let v = Uop.dtype x in
  let pos_inf = const_float_v v Float.infinity in
  let neg_inf = const_float_v v Float.neg_infinity in
  let open Uop.O in
  where (ne x pos_inf)
    (where (ne x x) nan (where (ne x neg_inf) ratio ninf))
    inf

(* Horner-form polynomial evaluation: [polyN x \[c0; c1; ...; cn\]] is
   [((c0*x + c1)*x + c2)*x + ... + cn]. *)
let polyN x coeffs =
  let v = Uop.dtype x in
  let c y = const_float_v v y in
  match coeffs with
  | [] -> c 0.0
  | first :: rest ->
      List.fold_left (fun acc ci ->
        Uop.alu_binary ~op:Ops.Add
          ~lhs:(Uop.alu_binary ~op:Ops.Mul ~lhs:acc ~rhs:x)
          ~rhs:(c ci)) (c first) rest

(* Payne-Hanek reduction: reduce an arbitrary angle [d] modulo pi/2 using
   a 190-bit table of 2/pi. Returns [(r, q)] with [r] the reduced angle
   and [q mod 4] the quadrant. Accurate for [|d| >= ~39800]. *)
let payne_hanek_reduction d =
  let fdt = Uop.dtype d in
  let two_over_pi_f =
    [| 0x00000000; 0x28be60db; 0x9391054a; 0x7f09d5f4; 0x7d4d3770;
       0x36d8a566; 0x4f10e410 |]
  in
  let intermediate_dtype =
    if fdt = Dtype.Float16 then Dtype.float32 else fdt
  in
  let uint64_dt = Dtype.uint64 in
  let int32_dt = Dtype.int32 in
  let uint32_dt = Dtype.uint32 in
  let u64v = uint64_dt in
  let u32v = uint32_dt in
  let iv = int32_dt in
  let f, e_raw = frexp d in
  let ia =
    Uop.cast ~dtype:uint64_dt
      ~src:(Uop.alu_binary ~op:Ops.Mul
              ~lhs:(Uop.cast ~src:f ~dtype:intermediate_dtype)
              ~rhs:(const_float_dt intermediate_dtype 4.294967296e9))
  in
  let i_5 = Uop.const (Const.int u64v 5) in
  let i = shr (Uop.cast ~src:e_raw ~dtype:uint64_dt) i_5 in
  let e =
    Uop.alu_binary ~op:Ops.And
      ~lhs:(Uop.cast ~src:e_raw ~dtype:int32_dt)
      ~rhs:(Uop.const (Const.int iv 31))
  in
  let offset =
    Uop.alu_binary ~op:Ops.Sub ~lhs:(Uop.const (Const.int iv 32)) ~rhs:e
  in
  let rec take an off count =
    if count + off < Array.length two_over_pi_f - 1 then
      let inner = take an off (count + 1) in
      Uop.alu_ternary ~op:Ops.Where
        ~a:(Uop.alu_binary ~op:Ops.Cmpne ~lhs:i
              ~rhs:(const_int64_v u64v (Int64.of_int count)))
        ~b:inner
        ~c:(const_int64_v u32v (Int64.of_int two_over_pi_f.(count + off)))
    else an
  in
  let zero_u32 = const_int64_v u32v 0L in
  let a = Array.init 4 (fun off -> take zero_u32 off 0) in
  let shl_lazy x y =
    Uop.cast ~dtype:uint32_dt
      ~src:(Uop.alu_binary ~op:Ops.Mul
              ~lhs:(Uop.cast ~src:x ~dtype:uint64_dt)
              ~rhs:(Uop.cast ~src:(pow2if y fdt) ~dtype:uint64_dt))
  in
  let shr_lazy x y =
    Uop.cast ~dtype:uint32_dt
      ~src:(Uop.alu_binary ~op:Ops.Floordiv
              ~lhs:(Uop.cast ~src:x ~dtype:uint64_dt)
              ~rhs:(Uop.cast ~src:(pow2if y fdt) ~dtype:uint64_dt))
  in
  let hi = Uop.alu_binary ~op:Ops.Or
    ~lhs:(shl_lazy a.(0) e) ~rhs:(shr_lazy a.(1) offset) in
  let mi = Uop.alu_binary ~op:Ops.Or
    ~lhs:(shl_lazy a.(1) e) ~rhs:(shr_lazy a.(2) offset) in
  let lo = Uop.alu_binary ~op:Ops.Or
    ~lhs:(shl_lazy a.(2) e) ~rhs:(shr_lazy a.(3) offset) in
  let hp_mul x y =
    Uop.alu_binary ~op:Ops.Mul
      ~lhs:(Uop.cast ~src:x ~dtype:uint64_dt)
      ~rhs:(Uop.cast ~src:y ~dtype:uint64_dt)
  in
  let c32 = Uop.const (Const.int u64v 32) in
  let c62 = Uop.const (Const.int u64v 62) in
  let p =
    Uop.alu_binary ~op:Ops.Add
      ~lhs:(Uop.alu_binary ~op:Ops.Add
              ~lhs:(shl (hp_mul ia hi) c32)
              ~rhs:(hp_mul ia mi))
      ~rhs:(shr (hp_mul ia lo) c32)
  in
  let q = Uop.cast ~src:(shr p c62) ~dtype:int32_dt in
  let p_masked =
    Uop.alu_binary ~op:Ops.And ~lhs:p
      ~rhs:(const_int64_v u64v 0x3fffffffffffffffL)
  in
  let r =
    Uop.cast ~dtype:fdt
      ~src:(Uop.alu_binary ~op:Ops.Mul
              ~lhs:(Uop.cast ~src:p_masked ~dtype:intermediate_dtype)
              ~rhs:(const_float_dt intermediate_dtype 3.4061215800865545e-19))
  in
  let f_lt_half = Uop.alu_binary ~op:Ops.Cmplt ~lhs:f ~rhs:(fconst_like f 0.5) in
  let r_adj =
    Uop.alu_binary ~op:Ops.Add ~lhs:r
      ~rhs:(fconst_like r (-.Float.pi /. 2.0))
  in
  let q_adj = Uop.alu_binary ~op:Ops.Add ~lhs:q ~rhs:(Uop.const (Const.int iv 1)) in
  (Uop.alu_ternary ~op:Ops.Where ~a:f_lt_half ~b:r ~c:r_adj,
   Uop.alu_ternary ~op:Ops.Where ~a:f_lt_half ~b:q ~c:q_adj)

(* Cody-Waite reduction for [|d| <= ~39800]: subtracts multiples of pi/2
   using extended-precision decomposition [pi/2 = PI_A + PI_B + ...].
   Float16 promotes internally to Float32 for the reduction arithmetic. *)
let cody_waite_reduction d =
  let fdt = Uop.dtype d in
  let m_1_pi = 0.318309886183790671537767526745028724 in
  let muladd q c r =
    Uop.alu_binary ~op:Ops.Add
      ~lhs:(Uop.alu_binary ~op:Ops.Mul ~lhs:q ~rhs:c) ~rhs:r
  in
  let qdh =
    if fdt = Dtype.Float64 then
      let int64_dt = Dtype.int64 in
      Uop.alu_binary ~op:Ops.Mul
        ~lhs:(Uop.cast ~dtype:fdt
                ~src:(Uop.cast ~dtype:int64_dt
                        ~src:(Uop.alu_binary ~op:Ops.Mul ~lhs:d
                                ~rhs:(fconst_like d
                                        (m_1_pi /. Float.of_int (1 lsl 24))))))
        ~rhs:(fconst_like d (Float.of_int (1 lsl 24)))
    else fconst_like d 0.0
  in
  let quadrant =
    if fdt = Dtype.Float64 then
      rintk (Uop.alu_binary ~op:Ops.Sub
               ~lhs:(Uop.alu_binary ~op:Ops.Mul ~lhs:d ~rhs:(fconst_like d m_1_pi))
               ~rhs:qdh)
    else rintk (Uop.alu_binary ~op:Ops.Mul ~lhs:d ~rhs:(fconst_like d m_1_pi))
  in
  let q_float = Uop.cast ~src:quadrant ~dtype:fdt in
  let r = match fdt with
    | Dtype.Float64 ->
        let pi_a = 3.1415926218032836914 in
        let pi_b = 3.1786509424591713469e-08 in
        let pi_c = 1.2246467864107188502e-16 in
        let pi_d = 1.2736634327021899816e-24 in
        let r = muladd qdh (fconst_like d (-. pi_a)) d in
        let r = muladd q_float (fconst_like d (-. pi_a)) r in
        let r = muladd qdh (fconst_like d (-. pi_b)) r in
        let r = muladd q_float (fconst_like d (-. pi_b)) r in
        let r = muladd qdh (fconst_like d (-. pi_c)) r in
        let r = muladd q_float (fconst_like d (-. pi_c)) r in
        muladd (Uop.alu_binary ~op:Ops.Add ~lhs:qdh ~rhs:q_float)
               (fconst_like d (-. pi_d)) r
    | Dtype.Float16 ->
        let f32_dt = Dtype.float32 in
        let q32 = Uop.cast ~src:quadrant ~dtype:f32_dt in
        let c y = const_float_dt f32_dt y in
        let r = muladd q32 (c (-3.1414794921875))
                  (Uop.cast ~src:d ~dtype:f32_dt) in
        let r = muladd q32 (c (-0.00011315941810607910156)) r in
        let r = muladd q32 (c (-1.9841872589410058936e-09)) r in
        Uop.cast ~src:(muladd q32 (c (-1.2154201256553420762e-10)) r) ~dtype:fdt
    | _ ->
        let r = muladd q_float (fconst_like d (-3.1414794921875)) d in
        let r = muladd q_float (fconst_like d (-0.00011315941810607910156)) r in
        let r = muladd q_float (fconst_like d (-1.9841872589410058936e-09)) r in
        muladd q_float (fconst_like d (-1.2154201256553420762e-10)) r
  in
  let int32_dt = Dtype.int32 in
  r, Uop.cast ~src:quadrant ~dtype:int32_dt

(* Approximate [sin(d)] on a reduced angle. *)
let trig_poly d coeff32 coeff64 =
  let d2 = Uop.alu_binary ~op:Ops.Mul ~lhs:d ~rhs:d in
  let coeffs =
    if Uop.dtype d = Dtype.Float64 then coeff64 else coeff32
  in
  Uop.alu_binary ~op:Ops.Mul ~lhs:d ~rhs:(polyN d2 coeffs)

let sin_poly d =
  trig_poly d
    [ 2.6083159809786593541503e-06; -0.0001981069071916863322258;
      0.00833307858556509017944336; -0.166666597127914428710938; 1.0 ]
    [ -7.97255955009037868891952e-18; 2.81009972710863200091251e-15;
      -7.64712219118158833288484e-13; 1.60590430605664501629054e-10;
      -2.50521083763502045810755e-08; 2.75573192239198747630416e-06;
      -0.000198412698412696162806809; 0.00833333333333332974823815;
      -0.166666666666666657414808; 1.0 ]

let ifand q n =
  let v = Uop.dtype q in
  Uop.alu_binary ~op:Ops.Cmpne
    ~lhs:(Uop.alu_binary ~op:Ops.And ~lhs:q
            ~rhs:(const_int64_v v (Int64.of_int n)))
    ~rhs:(const_int64_v v 0L)

let sin_poly_small d q =
  let r = sin_poly d in
  let v = Uop.dtype r in
  let sign = Uop.alu_ternary ~op:Ops.Where
    ~a:(ifand q 1) ~b:(const_float_v v (-1.0)) ~c:(const_float_v v 1.0)
  in
  Uop.alu_binary ~op:Ops.Mul ~lhs:r ~rhs:sign

let sin_poly_large d q =
  let v = Uop.dtype d in
  let shifted = Uop.alu_binary ~op:Ops.Add ~lhs:d
    ~rhs:(Uop.alu_ternary ~op:Ops.Where ~a:(ifand q 1)
            ~b:(const_float_v v (Float.pi /. 2.0))
            ~c:(const_float_v v 0.0))
  in
  let r = sin_poly shifted in
  let sign = Uop.alu_ternary ~op:Ops.Where
    ~a:(ifand q 2) ~b:(const_float_v v (-1.0)) ~c:(const_float_v v 1.0)
  in
  Uop.alu_binary ~op:Ops.Mul ~lhs:r ~rhs:sign

(* [xsin ?fast ?switch_over d] is a 1.0 ULP approximation of [sin d].
   [switch_over] selects Cody-Waite below the threshold and Payne-Hanek
   above; [~fast:true] assumes [|d| <= switch_over]. *)
let xsin ?(fast = false) ?(switch_over = 30.0) d =
  require_transcendental "xsin" d;
  let v = Uop.dtype d in
  let nan_c = const_float_v v Float.nan in
  let zero = fconst_like d 0.0 in
  let x = lazy_map_numbers d ~inf:zero ~ninf:zero ~nan:zero ~ratio:d in
  let open Uop.O in
  let x_sign =
    where (ne x zero)
      (where (x < zero) (fconst_like x (-1.0)) (fconst_like x 1.0))
      zero
  in
  let x_abs = x * x_sign in
  let result =
    if fast then
      let r, q = cody_waite_reduction x_abs in
      sin_poly_small r q
    else
      let r_large, q_large = payne_hanek_reduction x_abs in
      let r_small, q_small = cody_waite_reduction x_abs in
      where (x_abs < fconst_like x_abs switch_over)
        (sin_poly_small r_small q_small)
        (sin_poly_large r_large q_large)
  in
  let result = result * x_sign in
  lazy_map_numbers d ~inf:nan_c ~ninf:nan_c ~nan:nan_c ~ratio:result

(* [xexp2 d] is a 1.0 ULP approximation of [2^d]. Follows Sleef's
   polynomial decomposition on [s = d - round(d)]. *)
let xexp2 d =
  require_transcendental "xexp2" d;
  let fdt = Uop.dtype d in
  let v = fdt in
  let zero = fconst_like d 0.0 in
  let x = lazy_map_numbers d ~inf:zero ~ninf:zero ~nan:zero ~ratio:d in
  let q = rintk x in
  let s = Uop.alu_binary ~op:Ops.Sub ~lhs:x ~rhs:(Uop.cast ~src:q ~dtype:fdt) in
  let u =
    if fdt = Dtype.Float64 then
      polyN s
        [ 0.4434359082926529454e-9; 0.7073164598085707425e-8;
          0.1017819260921760451e-6; 0.1321543872511327615e-5;
          0.1525273353517584730e-4; 0.1540353045101147808e-3;
          0.1333355814670499073e-2; 0.9618129107597600536e-2;
          0.5550410866482046596e-1; 0.2402265069591012214e+0;
          0.6931471805599452862e+0; 0.1000000000000000000e+1 ]
    else
      polyN s
        [ 0.1535920892e-3; 0.1339262701e-2; 0.9618384764e-2;
          0.5550347269e-1; 0.2402264476e+0; 0.6931471825e+0; 1.0 ]
  in
  let u = ldexp2k u q in
  let upper, lower = match fdt with
    | Dtype.Float64 -> 1024.0, -2000.0
    | Dtype.Float16 -> 23.0, -22.0
    | _ -> 128.0, -150.0
  in
  (* d >= upper -> +inf. Encoded as ¬(d < upper). *)
  let upper_c = fconst_like d upper in
  let lower_c = fconst_like d lower in
  let open Uop.O in
  let u =
    where (ne (d < upper_c) (Uop.const_bool true))
      (const_float_v v Float.infinity) u
  in
  let u = where (d < lower_c) zero u in
  let nan_c = const_float_v v Float.nan in
  where (ne d d) nan_c u

(* [xlog2 d] is a 1.0 ULP approximation of [log2 d] with denormal and
   edge-case handling. *)
let xlog2 d =
  require_transcendental "xlog2" d;
  let fdt = Uop.dtype d in
  let v = fdt in
  let denormal_exp = if fdt = Dtype.Float16 then 10 else 64 in
  let flt_min_val = match fdt with
    | Dtype.Float16 -> 6.1e-5 | _ -> 1e-4
  in
  let is_denormal = Uop.alu_binary ~op:Ops.Cmplt ~lhs:d
    ~rhs:(fconst_like d flt_min_val) in
  let a = Uop.alu_ternary ~op:Ops.Where ~a:is_denormal
    ~b:(Uop.alu_binary ~op:Ops.Mul ~lhs:d
          ~rhs:(fconst_like d (2.0 ** Float.of_int denormal_exp)))
    ~c:d
  in
  let e = Uop.cast ~src:(ilogb2k
      (Uop.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:(fconst_like a (1.0 /. 0.75))))
    ~dtype:fdt in
  let m = ldexp3k a (Uop.alu_unary ~op:Ops.Neg ~src:e) in
  let e = Uop.alu_ternary ~op:Ops.Where ~a:is_denormal
    ~b:(Uop.alu_binary ~op:Ops.Add ~lhs:e
          ~rhs:(fconst_like e (Float.of_int (-denormal_exp))))
    ~c:e
  in
  let one = fconst_like m 1.0 in
  let x =
    float_div
      (Uop.alu_binary ~op:Ops.Add ~lhs:m ~rhs:(fconst_like m (-1.0)))
      (Uop.alu_binary ~op:Ops.Add ~lhs:m ~rhs:one)
  in
  let x2 = Uop.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:x in
  let x_x2 = Uop.alu_binary ~op:Ops.Mul ~lhs:x ~rhs:x2 in
  let r =
    if fdt = Dtype.Float64 then
      let t = polyN x2
        [ 0.2211941750456081490e+0; 0.2200768693152277689e+0;
          0.2623708057488514656e+0; 0.3205977477944495502e+0;
          0.4121985945485324709e+0; 0.5770780162997058982e+0;
          0.96179669392608091449 ]
      in
      Uop.alu_binary ~op:Ops.Add
        ~lhs:(Uop.alu_binary ~op:Ops.Add
                ~lhs:(Uop.alu_binary ~op:Ops.Mul ~lhs:t ~rhs:x_x2) ~rhs:e)
        ~rhs:(Uop.alu_binary ~op:Ops.Mul ~lhs:x
                ~rhs:(fconst_like x 2.885390081777926774))
    else
      let t = polyN x2
        [ 0.4374550283e+0; 0.5764790177e+0; 0.9618012905120 ]
      in
      let base = Uop.alu_binary ~op:Ops.Add
        ~lhs:(Uop.alu_binary ~op:Ops.Add
                ~lhs:(Uop.alu_binary ~op:Ops.Mul ~lhs:t ~rhs:x_x2) ~rhs:e)
        ~rhs:(Uop.alu_binary ~op:Ops.Mul ~lhs:x
                ~rhs:(fconst_like x 2.8853900432586669922))
      in
      if fdt = Dtype.Float32 then
        Uop.alu_binary ~op:Ops.Add ~lhs:base
          ~rhs:(Uop.alu_binary ~op:Ops.Mul ~lhs:x
                  ~rhs:(fconst_like x 3.2734474483568488616e-08))
      else base
  in
  let inf = const_float_v v Float.infinity in
  let neg_inf = const_float_v v Float.neg_infinity in
  let nan_c = const_float_v v Float.nan in
  let open Uop.O in
  let r = where (ne d inf) r inf in
  let r = where (ne d (fconst_like d 0.0)) r neg_inf in
  let r = where (d < fconst_like d (-.0.0)) nan_c r in
  let r = where (ne d d) nan_c r in
  where (ne (Uop.alu_unary ~op:Ops.Reciprocal ~src:d) neg_inf) r neg_inf

(* [xpow base exponent] is [base ** exponent], expressed as
   [exp2(exponent * log2(|base|))] with sign and [0 ** 0] fixups. *)
let xpow base exponent =
  let dt = Uop.dtype base in
  let exp_dt = Uop.dtype exponent in
  let v = dt in
  let exp_v = exp_dt in
  let zero = const_float_v v 0.0 in
  let exp_zero = const_float_v exp_v 0.0 in
  let one = const_float_v v 1.0 in
  let nan_c = const_float_v v Float.nan in
  let two_i = Uop.const (Const.int Dtype.int32 2) in
  let open Uop.O in
  let is_neg = base < zero in
  let abs_base = where is_neg (neg base) base in
  let log_abs = Uop.alu_unary ~op:Ops.Log2 ~src:abs_base in
  let ret = Uop.alu_unary ~op:Ops.Exp2 ~src:(exponent * log_abs) in
  let int_exp = cast exp_dt (cast Dtype.int32 exponent) in
  let non_int = ne exponent int_exp in
  let abs_exp = where (exponent < exp_zero) (neg exponent) exponent in
  let is_odd =
    cast Dtype.bool (cast Dtype.int32 abs_exp mod two_i)
  in
  let neg_base = where non_int nan_c (where is_odd (neg ret) ret) in
  let zero_zero = Uop.alu_binary ~op:Ops.And
    ~lhs:(Uop.alu_binary ~op:Ops.Cmpeq ~lhs:base ~rhs:zero)
    ~rhs:(Uop.alu_binary ~op:Ops.Cmpeq ~lhs:exponent ~rhs:exp_zero)
  in
  where zero_zero one (where is_neg neg_base ret)
(* [via_f32 f d dtype] applies [f] directly when [dtype] is one of the
   three full-precision float kinds, and otherwise lifts [d] through
   float32 around [f] for narrower float types. *)
let via_f32 f d dtype =
  if is_transcendental dtype then Some (f d)
  else if Dtype.is_float dtype then
    let d32 = Uop.cast ~src:d ~dtype:Dtype.float32 in
    Some (Uop.cast ~src:(f d32) ~dtype)
  else None

(* Transcendental rewrite: lowers [Exp2]/[Log2]/[Sin] to their polynomial
   decompositions, [Sqrt] to [xpow(d, 0.5)], and upcasts narrow float
   dtypes through float32. *)
let get_transcendental_patterns (ops : Decomp_op.supported_ops) (node : Uop.t) =
  let src0 () =
    let s = Uop.src node in
    if Array.length s >= 1 then Some s.(0) else None
  in
  let dt = Uop.dtype node in
  match Uop.op node, src0 () with
  | Ops.Exp2, Some d
    when not ops.has_exp2 || ops.force_transcendental -> via_f32 xexp2 d dt
  | Ops.Log2, Some d
    when not ops.has_log2 || ops.force_transcendental -> via_f32 xlog2 d dt
  | Ops.Sin, Some d
    when not ops.has_sin || ops.force_transcendental -> via_f32 xsin d dt
  | Ops.Sqrt, Some d
    when not ops.has_sqrt || ops.force_transcendental ->
      let v = Uop.dtype d in
      Some (xpow d (const_float_v v 0.5))
  | _ -> None
