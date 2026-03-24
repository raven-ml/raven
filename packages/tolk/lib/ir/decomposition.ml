(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module K = Kernel

(* Helpers *)

let powers_of_two : (int64, int) Hashtbl.t =
  let tbl = Hashtbl.create 64 in
  for i = 0 to 62 do Hashtbl.replace tbl (Int64.shift_left 1L i) i done;
  tbl

let log2_of_power n = Hashtbl.find_opt powers_of_two n

let const_int_val node =
  match K.view node with
  | Const { value; _ } ->
    (match Const.view value with Int v -> Some v | _ -> None)
  | _ -> None

let iconst v = K.const (Const.int64 Dtype.index v)

(* Transcendentals *)

let xpow ~base ~exponent =
  let is_neg = K.binary ~op:`Cmplt ~lhs:base ~rhs:(K.const_float 0.0) in
  let abs_base = K.ternary ~op:`Where ~a:is_neg
    ~b:(K.unary ~op:`Neg ~src:base) ~c:base in
  let ret = K.unary ~op:`Exp2
    ~src:(K.binary ~op:`Mul ~lhs:exponent
            ~rhs:(K.unary ~op:`Log2 ~src:abs_base)) in
  let fdt = K.dtype_or Dtype.float32 ret in
  let int_exp = K.cast ~src:(K.cast ~src:exponent ~dtype:(Dtype.to_any Dtype.int32))
    ~dtype:(Dtype.to_any (K.dtype_or Dtype.float32 exponent)) in
  let non_int = K.binary ~op:`Cmpne ~lhs:exponent ~rhs:int_exp in
  let abs_exp = K.ternary ~op:`Where
    ~a:(K.binary ~op:`Cmplt ~lhs:exponent ~rhs:(K.const_float 0.0))
    ~b:(K.unary ~op:`Neg ~src:exponent) ~c:exponent in
  let is_odd = K.cast
    ~src:(K.binary ~op:`Mod
            ~lhs:(K.cast ~src:abs_exp ~dtype:(Dtype.to_any Dtype.int32))
            ~rhs:(K.const (Const.int Dtype.int32 2)))
    ~dtype:(Dtype.to_any Dtype.bool) in
  let nan_c = K.const (Const.float fdt Float.nan) in
  let neg_base = K.ternary ~op:`Where ~a:non_int ~b:nan_c
    ~c:(K.ternary ~op:`Where ~a:is_odd
          ~b:(K.unary ~op:`Neg ~src:ret) ~c:ret) in
  let zero_zero = K.binary ~op:`And
    ~lhs:(K.binary ~op:`Cmpeq ~lhs:base ~rhs:(K.const_float 0.0))
    ~rhs:(K.binary ~op:`Cmpeq ~lhs:exponent ~rhs:(K.const_float 0.0)) in
  K.ternary ~op:`Where ~a:zero_zero
    ~b:(K.const (Const.float fdt 1.0))
    ~c:(K.ternary ~op:`Where ~a:is_neg ~b:neg_base ~c:ret)

(* IEEE 754 helpers *)

let mantissa_bits dt = snd (Dtype.finfo (Dtype.scalar_of dt))

let exponent_bias dt =
  let e, _ = Dtype.finfo (Dtype.scalar_of dt) in (1 lsl (e - 1)) - 1

let exponent_mask dt =
  let e, _ = Dtype.finfo (Dtype.scalar_of dt) in (1 lsl e) - 1

(* Shift by constant via mul/div by power of 2. *)

let shr_const x n =
  let dt = K.dtype_or Dtype.int32 x in
  K.binary ~op:`Idiv ~lhs:x
    ~rhs:(K.const (Const.int64 dt (Int64.shift_left 1L n)))

let shl_const x n =
  let dt = K.dtype_or Dtype.int32 x in
  K.binary ~op:`Mul ~lhs:x
    ~rhs:(K.const (Const.int64 dt (Int64.shift_left 1L n)))

let const_of_node_int node =
  match K.view node with
  | Const { value; _ } ->
    (match Const.view value with Int v -> Some (Int64.to_int v) | _ -> None)
  | _ -> None

let expr_shr x y =
  match const_of_node_int y with
  | Some n -> shr_const x n
  | None -> failwith "expr_shr: non-constant shift amount"

let expr_shl x y =
  match const_of_node_int y with
  | Some n -> shl_const x n
  | None -> failwith "expr_shl: non-constant shift amount"

let lazy_map_numbers x ~inf:inf_val ~ninf:ninf_val ~nan:nan_val ~ratio =
  let fdt = K.dtype_or Dtype.float32 x in
  let pos_inf = K.const (Const.float fdt Float.infinity) in
  let neg_inf = K.const (Const.float fdt Float.neg_infinity) in
  K.ternary ~op:`Where
    ~a:(K.binary ~op:`Cmpne ~lhs:x ~rhs:pos_inf)
    ~b:(K.ternary ~op:`Where
          ~a:(K.binary ~op:`Cmpne ~lhs:x ~rhs:x) ~b:nan_val
          ~c:(K.ternary ~op:`Where
                ~a:(K.binary ~op:`Cmpne ~lhs:x ~rhs:neg_inf)
                ~b:ratio ~c:ninf_val))
    ~c:inf_val

let polyN x coeffs =
  let fdt = K.dtype_or Dtype.float32 x in
  let c v = K.const (Const.float fdt v) in
  match coeffs with
  | [] -> c 0.0
  | first :: rest ->
    List.fold_left
      (fun acc ci -> K.binary ~op:`Add ~lhs:(K.binary ~op:`Mul ~lhs:acc ~rhs:x) ~rhs:(c ci))
      (c first) rest

let const_like node v = K.const (Const.float (Dtype.scalar_of (K.dtype_or Dtype.float32 node)) v)
let int_const_like node v = K.const (Const.int64 (K.dtype_or Dtype.int32 node) v)

let int_for_float = function
  | Dtype.Float64 -> Dtype.int64
  | Float32 -> Dtype.int32
  | Float16 -> Dtype.int16
  | _ -> Dtype.int32

let uint_for_float = function
  | Dtype.Float64 -> Dtype.uint64
  | Float32 -> Dtype.uint32
  | Float16 -> Dtype.uint16
  | _ -> Dtype.uint32

let float_for_int = function
  | Dtype.Int64 -> Dtype.float64
  | Int32 -> Dtype.float32
  | Int16 -> Dtype.float16
  | _ -> Dtype.float32

let rintk d =
  let fdt = K.dtype_or Dtype.float32 d in
  let out_dtype = Dtype.vec (int_for_float (Dtype.scalar (Dtype.scalar_of fdt))) (Dtype.count fdt) in
  let zero = const_like d 0.0 in
  let rounded = K.binary ~op:`Add ~lhs:d
    ~rhs:(K.ternary ~op:`Where
            ~a:(K.binary ~op:`Cmplt ~lhs:d ~rhs:zero)
            ~b:(const_like d (-0.5)) ~c:(const_like d 0.5)) in
  K.cast ~src:rounded ~dtype:(Dtype.to_any out_dtype)

let pow2if q float_dtype =
  let qdt = K.dtype_or Dtype.int32 q in
  let scalar = Dtype.scalar (Dtype.scalar_of qdt) in
  let out_scalar = match scalar with
    | Int16 -> Dtype.scalar_of float_dtype
    | _ -> float_for_int scalar
  in
  let out_dtype = Dtype.vec out_scalar (Dtype.count qdt) in
  let q_biased = K.binary ~op:`Add ~lhs:q
    ~rhs:(int_const_like q (Int64.of_int (exponent_bias out_dtype))) in
  K.bitcast ~src:(shl_const q_biased (mantissa_bits out_dtype)) ~dtype:out_dtype

let ilogb2k d =
  let fdt = K.dtype_or Dtype.float32 d in
  let int_dtype = Dtype.vec (int_for_float (Dtype.scalar (Dtype.scalar_of fdt))) (Dtype.count fdt) in
  let dint = K.bitcast ~src:d ~dtype:int_dtype in
  let masked = K.binary ~op:`And ~lhs:(shr_const dint (mantissa_bits fdt))
    ~rhs:(K.const (Const.int64 int_dtype (Int64.of_int (exponent_mask fdt)))) in
  K.binary ~op:`Sub ~lhs:masked
    ~rhs:(K.const (Const.int64 int_dtype (Int64.of_int (exponent_bias fdt))))

let ldexp3k d e =
  let fdt = K.dtype_or Dtype.float32 d in
  let int_dtype = Dtype.vec (int_for_float (Dtype.scalar (Dtype.scalar_of fdt))) (Dtype.count fdt) in
  let m1 = K.bitcast ~src:d ~dtype:int_dtype in
  let m2 = shl_const (K.cast ~src:e ~dtype:(Dtype.to_any int_dtype)) (mantissa_bits fdt) in
  K.bitcast ~src:(K.binary ~op:`Add ~lhs:m1 ~rhs:m2) ~dtype:fdt

let ldexp2k d e =
  let fdt = K.dtype_or Dtype.float32 d in
  let half = shr_const e 1 in
  let other = K.binary ~op:`Sub ~lhs:e ~rhs:half in
  K.binary ~op:`Mul
    ~lhs:(K.binary ~op:`Mul ~lhs:d ~rhs:(pow2if half fdt))
    ~rhs:(pow2if other fdt)

let frexp_decomp v =
  let fdt = K.dtype_or Dtype.float32 v in
  let scalar = Dtype.scalar (Dtype.scalar_of fdt) in
  let mantissa_mask, half_exp_bits = match scalar with
    | Dtype.Float64 -> (0x000FFFFFFFFFFFFFL, 0x3FE0000000000000L)
    | Float32 -> (0x807FFFFFL, 0x3F000000L)
    | Float16 -> (0x83FFL, 0x3800L)
    | _ -> (0x807FFFFFL, 0x3F000000L)
  in
  let uint_dtype = Dtype.vec (uint_for_float scalar) (Dtype.count fdt) in
  let bits = K.bitcast ~src:v ~dtype:uint_dtype in
  let exponent = K.binary ~op:`And ~lhs:(shr_const bits (mantissa_bits fdt))
    ~rhs:(K.const (Const.int64 uint_dtype (Int64.of_int (exponent_mask fdt)))) in
  let mantissa = K.bitcast ~dtype:fdt
    ~src:(K.binary ~op:`Or
            ~lhs:(K.binary ~op:`And ~lhs:bits
                    ~rhs:(K.const (Const.int64 uint_dtype mantissa_mask)))
            ~rhs:(K.const (Const.int64 uint_dtype half_exp_bits))) in
  let exp = K.binary ~op:`Add
    ~lhs:(K.binary ~op:`Sub ~lhs:exponent
            ~rhs:(K.const (Const.int64 uint_dtype (Int64.of_int (exponent_bias fdt)))))
    ~rhs:(K.const (Const.int64 uint_dtype 1L)) in
  (mantissa, exp)

(* Payne-Hanek range reduction: reduce an arbitrary floating-point angle d
   to the interval [-pi/4, pi/4] with a quadrant indicator. Uses a table of
   2/pi digits and 64-bit integer arithmetic to maintain precision far beyond
   what Cody-Waite can handle (needed when |d| >> 1). Returns (r, q) where
   r is the reduced angle and q mod 4 selects the trig quadrant. *)

let payne_hanek_reduction d =
  let fdt = K.dtype_or Dtype.float32 d in
  let two_over_pi_f =
    [| 0x00000000; 0x28be60db; 0x9391054a; 0x7f09d5f4; 0x7d4d3770;
       0x36d8a566; 0x4f10e410 |] in
  let intermediate_dtype =
    if Dtype.scalar (Dtype.scalar_of fdt) = Dtype.Float16 then
      Dtype.vec Dtype.float32 (Dtype.count fdt)
    else fdt in
  let f, e_raw = frexp_decomp d in
  let uint64_dt = Dtype.vec Dtype.uint64 (Dtype.count fdt) in
  let int32_dt = Dtype.vec Dtype.int32 (Dtype.count fdt) in
  let uint32_dt = Dtype.vec Dtype.uint32 (Dtype.count fdt) in
  let ia = K.cast ~dtype:(Dtype.to_any uint64_dt)
    ~src:(K.binary ~op:`Mul
            ~lhs:(K.cast ~src:f ~dtype:(Dtype.to_any intermediate_dtype))
            ~rhs:(K.const (Const.float (Dtype.scalar_of intermediate_dtype) 4.294967296e9))) in
  let i = shr_const (K.cast ~src:e_raw ~dtype:(Dtype.to_any uint64_dt)) 5 in
  let e = K.binary ~op:`And
    ~lhs:(K.cast ~src:e_raw ~dtype:(Dtype.to_any int32_dt))
    ~rhs:(K.const (Const.int Dtype.int32 31)) in
  let offset = K.binary ~op:`Sub
    ~lhs:(K.const (Const.int Dtype.int32 32)) ~rhs:e in
  let rec take an off count =
    if count + off < Array.length two_over_pi_f - 1 then
      let inner = take an off (count + 1) in
      K.ternary ~op:`Where
        ~a:(K.binary ~op:`Cmpne ~lhs:i
              ~rhs:(K.const (Const.int64 uint64_dt (Int64.of_int count))))
        ~b:inner
        ~c:(K.const (Const.int64 uint32_dt (Int64.of_int two_over_pi_f.(count + off))))
    else an in
  let shift_lazy op x y =
    K.cast ~dtype:(Dtype.to_any uint32_dt)
      ~src:(K.binary ~op
              ~lhs:(K.cast ~src:x ~dtype:(Dtype.to_any uint64_dt))
              ~rhs:(K.cast ~src:(pow2if y fdt) ~dtype:(Dtype.to_any uint64_dt))) in
  let zero_u32 = K.const (Const.int64 uint32_dt 0L) in
  let a = Array.init 4 (fun off -> take zero_u32 off 0) in
  let combine ai aj =
    K.binary ~op:`Or
      ~lhs:(shift_lazy `Mul a.(ai) e)
      ~rhs:(shift_lazy `Idiv a.(aj) offset) in
  let hi = combine 0 1 in
  let mi = combine 1 2 in
  let lo = combine 2 3 in
  let hp_mul x y =
    K.binary ~op:`Mul
      ~lhs:(K.cast ~src:x ~dtype:(Dtype.to_any uint64_dt))
      ~rhs:(K.cast ~src:y ~dtype:(Dtype.to_any uint64_dt)) in
  let p = K.binary ~op:`Add
    ~lhs:(K.binary ~op:`Add
            ~lhs:(shl_const (hp_mul ia hi) 32)
            ~rhs:(hp_mul ia mi))
    ~rhs:(shr_const (hp_mul ia lo) 32) in
  let q = K.cast ~src:(shr_const p 62) ~dtype:(Dtype.to_any int32_dt) in
  let p_masked = K.binary ~op:`And ~lhs:p
    ~rhs:(K.const (Const.int64 Dtype.uint64 0x3ffffffffffffffFL)) in
  let r = K.cast ~dtype:(Dtype.to_any fdt)
    ~src:(K.binary ~op:`Mul
            ~lhs:(K.cast ~src:p_masked ~dtype:(Dtype.to_any intermediate_dtype))
            ~rhs:(K.const (Const.float (Dtype.scalar_of intermediate_dtype) 3.4061215800865545e-19))) in
  let f_lt_half = K.binary ~op:`Cmplt ~lhs:f ~rhs:(const_like f 0.5) in
  let r_adj = K.binary ~op:`Sub ~lhs:r ~rhs:(const_like r (Float.pi /. 2.0)) in
  let q_adj = K.binary ~op:`Add ~lhs:q ~rhs:(K.const (Const.int Dtype.int32 1)) in
  (K.ternary ~op:`Where ~a:f_lt_half ~b:r ~c:r_adj,
   K.ternary ~op:`Where ~a:f_lt_half ~b:q ~c:q_adj)

let cody_waite_reduction d =
  let fdt = K.dtype_or Dtype.float32 d in
  let scalar = Dtype.scalar (Dtype.scalar_of fdt) in
  let m_1_pi = 0.318309886183790671537767526745028724 in
  let muladd q c r = K.binary ~op:`Add
    ~lhs:(K.binary ~op:`Mul ~lhs:q ~rhs:c) ~rhs:r in
  let qdh =
    if scalar = Dtype.Float64 then
      K.binary ~op:`Mul
        ~lhs:(K.cast ~dtype:(Dtype.to_any fdt)
                ~src:(K.cast ~dtype:(Dtype.to_any (Dtype.vec Dtype.int64 (Dtype.count fdt)))
                        ~src:(K.binary ~op:`Mul ~lhs:d
                                ~rhs:(const_like d (m_1_pi /. Float.of_int (1 lsl 24))))))
        ~rhs:(const_like d (Float.of_int (1 lsl 24)))
    else const_like d 0.0 in
  let quadrant =
    if scalar = Dtype.Float64 then
      rintk (K.binary ~op:`Sub
               ~lhs:(K.binary ~op:`Mul ~lhs:d ~rhs:(const_like d m_1_pi))
               ~rhs:qdh)
    else rintk (K.binary ~op:`Mul ~lhs:d ~rhs:(const_like d m_1_pi)) in
  let q_float = K.cast ~src:quadrant ~dtype:(Dtype.to_any fdt) in
  let r = match scalar with
    | Dtype.Float64 ->
      let pi_a = 3.1415926218032836914 and pi_b = 3.1786509424591713469e-08 in
      let pi_c = 1.2246467864107188502e-16 and pi_d = 1.2736634327021899816e-24 in
      let r = muladd qdh (const_like d (-.pi_a)) d in
      let r = muladd q_float (const_like d (-.pi_a)) r in
      let r = muladd qdh (const_like d (-.pi_b)) r in
      let r = muladd q_float (const_like d (-.pi_b)) r in
      let r = muladd qdh (const_like d (-.pi_c)) r in
      let r = muladd q_float (const_like d (-.pi_c)) r in
      muladd (K.binary ~op:`Add ~lhs:qdh ~rhs:q_float) (const_like d (-.pi_d)) r
    | Dtype.Float16 ->
      let f32_dt = Dtype.vec Dtype.float32 (Dtype.count fdt) in
      let q32 = K.cast ~src:q_float ~dtype:(Dtype.to_any f32_dt) in
      let c v = K.const (Const.float Dtype.float32 v) in
      let r = muladd q32 (c (-3.1414794921875)) (K.cast ~src:d ~dtype:(Dtype.to_any f32_dt)) in
      let r = muladd q32 (c (-0.00011315941810607910156)) r in
      let r = muladd q32 (c (-1.9841872589410058936e-09)) r in
      K.cast ~src:(muladd q32 (c (-1.2154201256553420762e-10)) r) ~dtype:(Dtype.to_any fdt)
    | _ ->
      let r = muladd q_float (const_like d (-3.1414794921875)) d in
      let r = muladd q_float (const_like d (-0.00011315941810607910156)) r in
      let r = muladd q_float (const_like d (-1.9841872589410058936e-09)) r in
      muladd q_float (const_like d (-1.2154201256553420762e-10)) r
  in
  (r, K.cast ~src:quadrant ~dtype:(Dtype.to_any (Dtype.vec Dtype.int32 (Dtype.count fdt))))

(* Sine polynomial *)

let trig_poly d coeff32 coeff64 =
  let fdt = K.dtype_or Dtype.float32 d in
  let d2 = K.binary ~op:`Mul ~lhs:d ~rhs:d in
  let coeffs =
    if Dtype.scalar (Dtype.scalar_of fdt) = Dtype.Float64 then coeff64
    else coeff32
  in
  K.binary ~op:`Mul ~lhs:d ~rhs:(polyN d2 coeffs)

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
  let dt = K.dtype_or Dtype.int32 q in
  K.binary ~op:`Cmpne
    ~lhs:(K.binary ~op:`And ~lhs:q ~rhs:(K.const (Const.int64 dt (Int64.of_int n))))
    ~rhs:(K.const (Const.int64 dt 0L))

let sign_flip r q n =
  K.binary ~op:`Mul ~lhs:r
    ~rhs:(K.ternary ~op:`Where ~a:(ifand q n)
            ~b:(const_like r (-1.0)) ~c:(const_like r 1.0))

let sin_poly_small d q = sign_flip (sin_poly d) q 1

let sin_poly_large d q =
  let d_adj = K.binary ~op:`Add ~lhs:d
    ~rhs:(K.ternary ~op:`Where ~a:(ifand q 1)
            ~b:(const_like d (Float.pi /. 2.0)) ~c:(const_like d 0.0)) in
  sign_flip (sin_poly d_adj) q 2

(* Toplevel transcendentals *)

let xsin ?(fast = false) ?(switch_over = 30.0) d =
  let fdt = K.dtype_or Dtype.float32 d in
  let nan_c = K.const (Const.float (Dtype.scalar_of fdt) Float.nan) in
  let zero = const_like d 0.0 in
  let x = lazy_map_numbers d ~inf:zero ~ninf:zero ~nan:zero ~ratio:d in
  let x_sign = K.ternary ~op:`Where
    ~a:(K.binary ~op:`Cmpne ~lhs:x ~rhs:zero)
    ~b:(K.ternary ~op:`Where
          ~a:(K.binary ~op:`Cmplt ~lhs:x ~rhs:zero)
          ~b:(const_like x (-1.0)) ~c:(const_like x 1.0))
    ~c:zero in
  let x_abs = K.binary ~op:`Mul ~lhs:x ~rhs:x_sign in
  let result =
    if fast then
      let r, q = cody_waite_reduction x_abs in
      sin_poly_small r q
    else
      let r_large, q_large = payne_hanek_reduction x_abs in
      let r_small, q_small = cody_waite_reduction x_abs in
      K.ternary ~op:`Where
        ~a:(K.binary ~op:`Cmplt ~lhs:x_abs ~rhs:(const_like x_abs switch_over))
        ~b:(sin_poly_small r_small q_small)
        ~c:(sin_poly_large r_large q_large) in
  let result = K.binary ~op:`Mul ~lhs:result ~rhs:x_sign in
  lazy_map_numbers d ~inf:nan_c ~ninf:nan_c ~nan:nan_c ~ratio:result

let xexp2 d =
  let fdt = K.dtype_or Dtype.float32 d in
  let scalar = Dtype.scalar (Dtype.scalar_of fdt) in
  let zero = const_like d 0.0 in
  let x = lazy_map_numbers d ~inf:zero ~ninf:zero ~nan:zero ~ratio:d in
  let q = rintk x in
  let s = K.binary ~op:`Sub ~lhs:x ~rhs:(K.cast ~src:q ~dtype:(Dtype.to_any fdt)) in
  let u =
    if scalar = Dtype.Float64 then
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
          0.5550347269e-1; 0.2402264476e+0; 0.6931471825e+0; 1.0 ] in
  let u = ldexp2k u q in
  let upper, lower = match scalar with
    | Dtype.Float64 -> (1024.0, -2000.0)
    | Dtype.Float16 -> (23.0, -22.0)
    | _ -> (128.0, -150.0) in
  let inf = const_like d Float.infinity in
  let u = K.ternary ~op:`Where
    ~a:(K.binary ~op:`Cmplt ~lhs:d ~rhs:(const_like d upper)) ~b:u ~c:inf in
  let u = K.ternary ~op:`Where
    ~a:(K.binary ~op:`Cmplt ~lhs:d ~rhs:(const_like d lower)) ~b:zero ~c:u in
  let nan_c = K.const (Const.float (Dtype.scalar_of fdt) Float.nan) in
  K.ternary ~op:`Where
    ~a:(K.binary ~op:`Cmpne ~lhs:d ~rhs:d) ~b:nan_c ~c:u

let xlog2 d =
  let fdt = K.dtype_or Dtype.float32 d in
  let scalar = Dtype.scalar (Dtype.scalar_of fdt) in
  let denormal_exp = if scalar = Dtype.Float16 then 10 else 64 in
  let flt_min_val = if scalar = Dtype.Float16 then 6.1e-5 else 1e-4 in
  let is_denormal = K.binary ~op:`Cmplt ~lhs:d ~rhs:(const_like d flt_min_val) in
  let a = K.ternary ~op:`Where ~a:is_denormal
    ~b:(K.binary ~op:`Mul ~lhs:d ~rhs:(const_like d (Float.of_int (1 lsl denormal_exp))))
    ~c:d in
  let e = K.cast ~src:(ilogb2k (K.binary ~op:`Mul ~lhs:a ~rhs:(const_like a (1.0 /. 0.75))))
    ~dtype:(Dtype.to_any fdt) in
  let m = ldexp3k a (K.unary ~op:`Neg ~src:e) in
  let e = K.ternary ~op:`Where ~a:is_denormal
    ~b:(K.binary ~op:`Sub ~lhs:e ~rhs:(const_like e (Float.of_int denormal_exp)))
    ~c:e in
  let one = const_like m 1.0 in
  let x = K.binary ~op:`Fdiv
    ~lhs:(K.binary ~op:`Sub ~lhs:m ~rhs:one)
    ~rhs:(K.binary ~op:`Add ~lhs:m ~rhs:one) in
  let x2 = K.binary ~op:`Mul ~lhs:x ~rhs:x in
  let x_x2 = K.binary ~op:`Mul ~lhs:x ~rhs:x2 in
  let r =
    if scalar = Dtype.Float64 then
      let t = polyN x2
        [ 0.2211941750456081490e+0; 0.2200768693152277689e+0;
          0.2623708057488514656e+0; 0.3205977477944495502e+0;
          0.4121985945485324709e+0; 0.5770780162997058982e+0;
          0.96179669392608091449 ] in
      K.binary ~op:`Add
        ~lhs:(K.binary ~op:`Add ~lhs:(K.binary ~op:`Mul ~lhs:t ~rhs:x_x2) ~rhs:e)
        ~rhs:(K.binary ~op:`Mul ~lhs:x ~rhs:(const_like x 2.885390081777926774))
    else
      let t = polyN x2
        [ 0.4374550283e+0; 0.5764790177e+0; 0.9618012905120 ] in
      let base = K.binary ~op:`Add
        ~lhs:(K.binary ~op:`Add ~lhs:(K.binary ~op:`Mul ~lhs:t ~rhs:x_x2) ~rhs:e)
        ~rhs:(K.binary ~op:`Mul ~lhs:x ~rhs:(const_like x 2.8853900432586669922)) in
      if scalar = Dtype.Float32 then
        K.binary ~op:`Add ~lhs:base
          ~rhs:(K.binary ~op:`Mul ~lhs:x ~rhs:(const_like x 3.2734474483568488616e-08))
      else base in
  let inf = const_like d Float.infinity in
  let neg_inf = const_like d Float.neg_infinity in
  let nan_c = K.const (Const.float (Dtype.scalar_of fdt) Float.nan) in
  let r = K.ternary ~op:`Where
    ~a:(K.binary ~op:`Cmpne ~lhs:d ~rhs:inf) ~b:r ~c:inf in
  let r = K.ternary ~op:`Where
    ~a:(K.binary ~op:`Cmpne ~lhs:d ~rhs:(const_like d 0.0)) ~b:r ~c:neg_inf in
  let r = K.ternary ~op:`Where
    ~a:(K.binary ~op:`Cmplt ~lhs:d ~rhs:(const_like d (-0.0))) ~b:nan_c ~c:r in
  let r = K.ternary ~op:`Where
    ~a:(K.binary ~op:`Cmpne ~lhs:d ~rhs:d) ~b:nan_c ~c:r in
  (* reciprocal trick: devices where x == -0.0 fails *)
  K.ternary ~op:`Where
    ~a:(K.binary ~op:`Cmpne
          ~lhs:(K.unary ~op:`Recip ~src:d)
          ~rhs:(const_like d Float.neg_infinity))
    ~b:r ~c:neg_inf

(* Threefry *)

let threefry2x32 x key =
  let u64 = K.dtype_or Dtype.uint64 x in
  let u32 = Dtype.uint32 in
  let mask32 = K.const (Const.int64 u64 0xFFFFFFFFL) in
  let lo v = K.cast ~src:(K.binary ~op:`And ~lhs:v ~rhs:mask32) ~dtype:(Dtype.to_any u32) in
  let hi v = K.cast ~src:(K.binary ~op:`And ~lhs:(shr_const v 32) ~rhs:mask32) ~dtype:(Dtype.to_any u32) in
  let x0 = lo x and x1 = hi x in
  let key0 = lo key and key1 = hi key in
  let rotations = [| [| 13; 15; 26; 6 |]; [| 17; 29; 16; 24 |] |] in
  let ks = [| key1;
    K.binary ~op:`Xor
      ~lhs:(K.binary ~op:`Xor ~lhs:key0 ~rhs:key1)
      ~rhs:(K.const (Const.int64 u32 0x1BD11BDAL));
    key0 |] in
  let xr0 = ref (K.binary ~op:`Add ~lhs:x0 ~rhs:ks.(2)) in
  let xr1 = ref (K.binary ~op:`Add ~lhs:x1 ~rhs:ks.(0)) in
  for i = 0 to 4 do
    let rots = rotations.(i mod 2) in
    for j = 0 to 3 do
      let r = rots.(j) in
      let x0_new = K.binary ~op:`Add ~lhs:!xr0 ~rhs:!xr1 in
      let rotated = K.binary ~op:`Add
        ~lhs:(shl_const !xr1 r) ~rhs:(shr_const !xr1 (32 - r)) in
      xr1 := K.binary ~op:`Xor ~lhs:x0_new ~rhs:rotated;
      xr0 := x0_new
    done;
    xr0 := K.binary ~op:`Add ~lhs:!xr0 ~rhs:ks.(i mod 3);
    xr1 := K.binary ~op:`Add ~lhs:!xr1
      ~rhs:(K.binary ~op:`Add ~lhs:ks.((i + 1) mod 3)
              ~rhs:(K.const (Const.int64 u32 (Int64.of_int (i + 1)))))
  done;
  K.binary ~op:`Or
    ~lhs:(shl_const (K.cast ~src:!xr1 ~dtype:(Dtype.to_any u64)) 32)
    ~rhs:(K.cast ~src:!xr0 ~dtype:(Dtype.to_any u64))

(* Pattern matching *)

type supported_ops = {
  has_exp2 : bool;
  has_log2 : bool;
  has_sin : bool;
  has_sqrt : bool;
  has_recip : bool;
  has_neg : bool;
  has_sub : bool;
  has_max : bool;
  has_shl : bool;
  has_shr : bool;
  has_and : bool;
  has_or : bool;
  has_cmplt : bool;
  has_cmpeq : bool;
  has_fdiv : bool;
  has_threefry : bool;
  has_mulacc : bool;
  disable_fast_idiv : bool;
  force_transcendental : bool;
}

let transcendental_dtypes dt =
  let s = Dtype.scalar (Dtype.scalar_of dt) in
  s = Dtype.Float16 || s = Dtype.Float32 || s = Dtype.Float64

let get_transcendental_patterns (ops : supported_ops) node =
  let via_f32 f d dtype =
    if transcendental_dtypes dtype then Some (f d)
    else if Dtype.is_float dtype then
      Some (K.cast ~src:(f (K.cast ~src:d ~dtype:(Dtype.to_any Dtype.float32))) ~dtype:(Dtype.to_any dtype))
    else None in
  match K.view node with
  | Unary { op = `Exp2; src = d; dtype }
    when not ops.has_exp2 || ops.force_transcendental -> via_f32 xexp2 d dtype
  | Unary { op = `Log2; src = d; dtype }
    when not ops.has_log2 || ops.force_transcendental -> via_f32 xlog2 d dtype
  | Unary { op = `Sin; src = d; dtype }
    when not ops.has_sin || ops.force_transcendental -> via_f32 xsin d dtype
  | Unary { op = `Sqrt; src = d; _ }
    when not ops.has_sqrt || ops.force_transcendental ->
      Some (xpow ~base:d ~exponent:(const_like d 0.5))
  | _ -> None

(* Integer division *)

let magicgu ~vmax ~d =
  assert (d > 0);
  let nc = (vmax + 1) / d * d - 1 in
  let nbits =
    let rec bits v = if v <= 0 then 0 else 1 + bits (v lsr 1) in
    bits vmax in
  let rec find_s s =
    if s > 2 * nbits then failwith "magicgu: no solution found"
    else
      let two_s = 1 lsl s in
      if two_s > nc * (d - 1 - (two_s - 1) mod d) then
        ((two_s + d - 1 - (two_s - 1) mod d) / d, s)
      else find_s (s + 1) in
  find_s 0

let fast_idiv x d =
  assert (d > 0L);
  let d = Int64.to_int d in
  let bound_of = function `SInt n -> Int64.to_int n | _ -> 0 in
  let bound_of_max = function `SInt n -> Int64.to_int n | _ -> Int.max_int in
  let xmin = max (Divandmod.vmin x) (Int64.of_int (bound_of (Dtype.min Dtype.index))) in
  let xmax = min (Divandmod.vmax x) (Int64.of_int (bound_of_max (Dtype.max Dtype.index))) in
  let vmin_i = Int64.to_int xmin and vmax_i = Int64.to_int xmax in
  let m, s = magicgu ~vmax:(max (abs vmax_i) (abs vmin_i)) ~d in
  let m64 = Int64.of_int m in
  let fits =
    Int64.mul m64 (Int64.of_int vmin_i) >= Int64.of_int Int.min_int
    && Int64.mul m64 (Int64.of_int vmax_i) <= Int64.of_int Int.max_int in
  if fits then
    let shifted = K.binary ~op:`Shr
      ~lhs:(K.binary ~op:`Mul ~lhs:x ~rhs:(iconst m64))
      ~rhs:(iconst (Int64.of_int s)) in
    if xmin >= 0L then Some shifted
    else
      let correction = K.ternary ~op:`Where
        ~a:(K.binary ~op:`Cmplt ~lhs:x ~rhs:(iconst 0L))
        ~b:(iconst 1L) ~c:(iconst 0L) in
      Some (K.binary ~op:`Add ~lhs:shifted ~rhs:correction)
  else None

(* Long decomposition: int64 -> int32 pairs *)

let is_long_dtype (dt : Dtype.t) =
  Dtype.scalar dt = Dtype.Int64 || Dtype.scalar dt = Dtype.Uint64

let long_to_int_dtype (dt : Dtype.t) = match Dtype.scalar dt with
  | Int64 -> Dtype.int32 | Uint64 -> Dtype.uint32 | _ -> dt

let reindex_long (idx : K.t) off mul =
  match K.view idx with
  | Index { ptr; idxs = [ i ]; gate; _ } ->
    let open K.O in
    K.index ~ptr ~idxs:[ i * int_ mul + int_ off ] ?gate ()
  | _ -> idx

type l2i_op =
  [ `Neg | `Shl | `Shr | `Add | `Sub | `Mul | `Cmplt | `Cmpeq | `Cmpne
  | `Xor | `Or | `And | `Where | `Max | `Cast | `Bitcast ]

let rec l2i (op : l2i_op) (dt : Dtype.t) (uops : K.t list) : K.t * K.t =
  let zero = K.const (Const.int dt 0) in
  let a0, a1 = match uops with
    | [a0; a1] -> (a0, a1)
    | [a0; a1; _; _] -> (a0, a1)
    | _ -> failwith "l2i: unexpected operand count"
  in
  let b0, b1 = match uops with
    | [_; _; b0; b1] -> (b0, b1)
    | _ -> (zero, zero)
  in
  match op with
  | `Neg -> l2i `Sub dt [zero; zero; a0; a1]
  | `Shl ->
      let b0_mod = K.binary ~op:`And ~lhs:b0 ~rhs:(K.const (Const.int dt 31)) in
      let lo = expr_shl a0 b0_mod in
      let hi = K.binary ~op:`Or
        ~lhs:(expr_shl a1 b0_mod)
        ~rhs:(expr_shr (expr_shr a0 (K.const (Const.int dt 1)))
                (K.binary ~op:`Sub ~lhs:(K.const (Const.int dt 31)) ~rhs:b0_mod)) in
      let ge32 = K.binary ~op:`Cmplt ~lhs:(K.const (Const.int dt 31)) ~rhs:b0 in
      (K.ternary ~op:`Where ~a:ge32 ~b:zero ~c:lo,
       K.ternary ~op:`Where ~a:ge32 ~b:lo ~c:hi)
  | `Shr ->
      let b0_mod = K.binary ~op:`And ~lhs:b0 ~rhs:(K.const (Const.int dt 31)) in
      let lo = K.binary ~op:`Or
        ~lhs:(expr_shr a0 b0_mod)
        ~rhs:(expr_shl (expr_shl a1 (K.const (Const.int dt 1)))
                (K.binary ~op:`Sub ~lhs:(K.const (Const.int dt 31)) ~rhs:b0_mod)) in
      let hi = expr_shr a1 b0_mod in
      let ge32 = K.binary ~op:`Cmplt ~lhs:(K.const (Const.int dt 31)) ~rhs:b0 in
      (K.ternary ~op:`Where ~a:ge32 ~b:hi ~c:lo,
       K.ternary ~op:`Where ~a:ge32 ~b:zero ~c:hi)
  | `Add ->
      let low = K.binary ~op:`Add ~lhs:a0 ~rhs:b0 in
      let carry =
        K.cast
          ~src:(K.binary ~op:`Cmplt
            ~lhs:(K.bitcast ~src:low ~dtype:Dtype.uint32)
            ~rhs:(K.bitcast ~src:a0 ~dtype:Dtype.uint32))
          ~dtype:(Dtype.to_any dt)
      in
      (low, K.binary ~op:`Add ~lhs:(K.binary ~op:`Add ~lhs:a1 ~rhs:b1) ~rhs:carry)
  | `Sub ->
      let borrow =
        K.cast
          ~src:(K.binary ~op:`Cmplt
            ~lhs:(K.bitcast ~src:a0 ~dtype:Dtype.uint32)
            ~rhs:(K.bitcast ~src:b0 ~dtype:Dtype.uint32))
          ~dtype:(Dtype.to_any dt)
      in
      (K.binary ~op:`Sub ~lhs:a0 ~rhs:b0,
       K.binary ~op:`Sub ~lhs:(K.binary ~op:`Sub ~lhs:a1 ~rhs:b1) ~rhs:borrow)
  | `Cmplt ->
      let hi_lt = K.binary ~op:`Cmplt ~lhs:a1 ~rhs:b1 in
      let hi_eq = K.binary ~op:`Cmpeq ~lhs:a1 ~rhs:b1 in
      let lo_lt = K.binary ~op:`Cmplt
        ~lhs:(K.bitcast ~src:a0 ~dtype:Dtype.uint32)
        ~rhs:(K.bitcast ~src:b0 ~dtype:Dtype.uint32) in
      (K.binary ~op:`Or ~lhs:hi_lt ~rhs:(K.binary ~op:`And ~lhs:hi_eq ~rhs:lo_lt), zero)
  | `Cmpeq ->
      (K.binary ~op:`And
        ~lhs:(K.binary ~op:`Cmpeq ~lhs:a0 ~rhs:b0)
        ~rhs:(K.binary ~op:`Cmpeq ~lhs:a1 ~rhs:b1), zero)
  | `Cmpne ->
      (K.binary ~op:`Or
        ~lhs:(K.binary ~op:`Cmpne ~lhs:a0 ~rhs:b0)
        ~rhs:(K.binary ~op:`Cmpne ~lhs:a1 ~rhs:b1), zero)
  | `Xor -> (K.binary ~op:`Xor ~lhs:a0 ~rhs:b0, K.binary ~op:`Xor ~lhs:a1 ~rhs:b1)
  | `Or -> (K.binary ~op:`Or ~lhs:a0 ~rhs:b0, K.binary ~op:`Or ~lhs:a1 ~rhs:b1)
  | `And -> (K.binary ~op:`And ~lhs:a0 ~rhs:b0, K.binary ~op:`And ~lhs:a1 ~rhs:b1)
  | `Where ->
    (match uops with
    | [cond; t_lo; t_hi; f_lo; f_hi] ->
      (K.ternary ~op:`Where ~a:cond ~b:t_lo ~c:f_lo,
       K.ternary ~op:`Where ~a:cond ~b:t_hi ~c:f_hi)
    | _ -> failwith "l2i Where: need 5 operands")
  | `Max -> l2i `Where dt (fst (l2i `Cmplt dt uops) :: b0 :: b1 :: a0 :: [a1])
  | _ -> failwith "l2i: unsupported op"

let widen_long_ptr (dtype : Dtype.ptr) size =
  let new_base = long_to_int_dtype (Dtype.scalar_of (Dtype.base dtype)) in
  Dtype.ptr_of new_base ~addrspace:(Dtype.addrspace dtype) ~size:(size * 2)

let pm_long_decomp (node : K.t) : K.t option =
  match K.view node with
  | Param { idx; dtype } when is_long_dtype (Dtype.base dtype) ->
    Some (K.param ~idx ~dtype:(widen_long_ptr dtype (Dtype.ptr_size dtype)))
  | Define_local { size; dtype } when is_long_dtype (Dtype.base dtype) ->
    Some (K.define_local ~size:(size * 2) ~dtype:(widen_long_ptr dtype size))
  | Define_reg { size; dtype; slot } when is_long_dtype (Dtype.base dtype) ->
    Some (K.define_reg ~size:(size * 2) ~dtype:(widen_long_ptr dtype size) ~slot)
  | Index { dtype = Dtype.P pty; _ }
    when is_long_dtype (Dtype.scalar_of (Dtype.base pty)) ->
    let off = match K.tag node with Some "1" -> 1 | _ -> 0 in
    Some (K.replace (reindex_long node off 2)
            ~dtype:(long_to_int_dtype (Dtype.scalar_of (Dtype.base pty))) ())
  | Store { dst; value; ranges } when K.tag node = None ->
    (match K.dtype value with
    | Some dt when is_long_dtype dt ->
      Some (K.group [
        K.with_tag "0" (K.store ~dst:(reindex_long dst 0 2)
          ~value:(K.with_tag "0" value) ~ranges);
        K.with_tag "1" (K.store ~dst:(reindex_long dst 1 2)
          ~value:(K.with_tag "1" value) ~ranges)])
    | _ -> None)
  | Load { src; dtype; _ } when is_long_dtype dtype ->
    (match K.tag node with
    | Some tag_str ->
      Some (K.load ~src:(reindex_long src (if tag_str = "1" then 1 else 0) 2) ())
    | None -> None)
  | Const { value; dtype } when is_long_dtype dtype ->
    (match K.tag node, Const.view value with
    | Some "1", Int n ->
      Some (K.const (Const.int (long_to_int_dtype dtype)
              (Int64.to_int (Int64.shift_right_logical n 32))))
    | Some _, Int n ->
      Some (K.const (Const.int (long_to_int_dtype dtype)
              (Int64.to_int (Int64.logand n 0xFFFFFFFFL))))
    | _ -> None)
  | Binary { op = (`Cmplt | `Cmpeq | `Cmpne) as op; lhs; rhs; _ }
    when (match K.dtype lhs with Some dt -> is_long_dtype dt | None -> false) ->
    let dt = long_to_int_dtype (K.dtype_or Dtype.int32 lhs) in
    Some (fst (l2i op dt [
      K.with_tag "0" lhs; K.with_tag "1" lhs;
      K.with_tag "0" rhs; K.with_tag "1" rhs]))
  | (Binary { dtype; _ } | Unary { dtype; _ } | Ternary { dtype; _ })
    when is_long_dtype dtype && K.tag node <> None ->
    let dt = long_to_int_dtype dtype in
    let expanded = List.concat_map (fun c ->
      match K.dtype c with
      | Some cdt when is_long_dtype cdt ->
        [K.cast ~src:(K.with_tag "0" c) ~dtype:(Dtype.to_any dt);
         K.cast ~src:(K.with_tag "1" c) ~dtype:(Dtype.to_any dt)]
      | _ -> [c]) (K.children node) in
    let to_l2i_op : K.view -> l2i_op = function
      | Binary { op = `Add; _ } -> `Add | Binary { op = `Sub; _ } -> `Sub
      | Binary { op = `Mul; _ } -> `Mul | Binary { op = `Shl; _ } -> `Shl
      | Binary { op = `Shr; _ } -> `Shr | Binary { op = `And; _ } -> `And
      | Binary { op = `Or; _ } -> `Or | Binary { op = `Xor; _ } -> `Xor
      | Binary { op = `Cmplt; _ } -> `Cmplt | Binary { op = `Cmpeq; _ } -> `Cmpeq
      | Binary { op = `Cmpne; _ } -> `Cmpne | Binary { op = `Max; _ } -> `Max
      | Unary { op = `Neg; _ } -> `Neg
      | _ -> failwith "l2i: unsupported op" in
    let lo, hi = l2i (to_l2i_op (K.view node)) dt expanded in
    (match K.tag node with
    | Some "0" -> Some lo | Some "1" -> Some hi | _ -> None)
  | _ -> None

(* Float decomposition *)

let f2f_dt : Dtype.scalar -> Dtype.scalar = function
  | Float16 | Bfloat16 -> Uint16
  | Float32 -> Uint32 | Float64 -> Uint64
  | s -> s

let f2f (v : K.t) ~(fr : Dtype.scalar) ~(to_ : Dtype.scalar) : K.t =
  let dt_of s = Dtype.of_scalar s in
  let fs = Dtype.bitsize (dt_of fr) and fb = exponent_bias (dt_of fr) in
  let fe, fm = Dtype.finfo (dt_of fr) in
  let ts = Dtype.bitsize (dt_of to_) and tb = exponent_bias (dt_of to_) in
  let te, tm = Dtype.finfo (dt_of to_) in
  let to_uint = Dtype.of_scalar (f2f_dt to_) in
  let fr_uint = Dtype.of_scalar (f2f_dt fr) in
  (* Use Int64 for all mask/shift computations to avoid overflow on wide floats *)
  let i64_const dt n = K.const (Const.int64 dt n) in
  if fe <= te && fm < tm then begin
    let sign = shl_const
      (K.cast ~src:(K.binary ~op:`And ~lhs:v
        ~rhs:(i64_const fr_uint (Int64.shift_left 1L (fs - 1))))
        ~dtype:(Dtype.to_any to_uint))
      (ts - fs) in
    let nosign = K.cast
      ~src:(K.binary ~op:`And ~lhs:v
        ~rhs:(i64_const fr_uint (Int64.sub (Int64.shift_left 1L (fs - 1)) 1L)))
      ~dtype:(Dtype.to_any to_uint) in
    let exp = shr_const nosign fm in
    let norm = K.binary ~op:`Add
      ~lhs:(shl_const nosign (tm - fm))
      ~rhs:(i64_const to_uint (Int64.shift_left (Int64.of_int (tb - fb)) tm)) in
    let nan = K.binary ~op:`Or
      ~lhs:(shl_const nosign (tm - fm))
      ~rhs:(i64_const to_uint (Int64.shift_left (Int64.sub (Int64.shift_left 1L te) 1L) tm)) in
    let is_nan = K.binary ~op:`Cmpeq ~lhs:exp
      ~rhs:(i64_const to_uint (Int64.sub (Int64.shift_left 1L fe) 1L)) in
    let is_zero = K.binary ~op:`Cmpeq ~lhs:exp
      ~rhs:(i64_const to_uint 0L) in
    K.bitcast
      ~src:(K.binary ~op:`Or ~lhs:sign
        ~rhs:(K.ternary ~op:`Where ~a:is_zero
          ~b:(K.const (Const.int to_uint 0))
          ~c:(K.ternary ~op:`Where ~a:is_nan ~b:nan ~c:norm)))
      ~dtype:(Dtype.of_scalar to_)
  end else
    K.cast ~src:v ~dtype:(Dtype.to_any (Dtype.of_scalar to_))

let f2f_clamp (v : K.t) ~(dt_scalar : Dtype.scalar) : K.t =
  let dt = Dtype.of_scalar dt_scalar in
  let e, m = Dtype.finfo dt in
  let max_exp, max_man = (1 lsl e) - 2, (1 lsl m) - 1 in
  let max_val = 2.0 ** Float.of_int (max_exp - exponent_bias dt) *.
    (1.0 +. Float.of_int max_man /. Float.of_int (1 lsl m)) in
  let mx = K.const (Const.float (K.dtype_or Dtype.float32 v) max_val) in
  let neg_mx = K.unary ~op:`Neg ~src:mx in
  let inf = K.const (Const.float (K.dtype_or Dtype.float32 v) infinity) in
  let is_nan = K.binary ~op:`Cmpne ~lhs:v ~rhs:v in
  let lt_neg = K.binary ~op:`Cmplt ~lhs:v ~rhs:neg_mx in
  let gt_pos = K.binary ~op:`Cmplt ~lhs:mx ~rhs:v in
  K.ternary ~op:`Where ~a:is_nan ~b:v
    ~c:(K.ternary ~op:`Where ~a:lt_neg ~b:(K.unary ~op:`Neg ~src:inf)
      ~c:(K.ternary ~op:`Where ~a:gt_pos ~b:inf ~c:v))

type float_decomp_ctx = {
  from_dtype : Dtype.scalar;
  to_dtype : Dtype.scalar;
}

let pm_float_decomp (ctx : float_decomp_ctx) (node : K.t) : K.t option =
  let fr = ctx.from_dtype and to_ = ctx.to_dtype in
  let rebase_ptr (dtype : Dtype.ptr) =
    let new_base = Dtype.vec (Dtype.of_scalar (f2f_dt fr)) (Dtype.count (Dtype.base dtype)) in
    Dtype.ptr_of new_base ~addrspace:(Dtype.addrspace dtype) ~size:(Dtype.ptr_size dtype) in
  let tag n = K.with_tag (Dtype.scalar_to_string fr) n in
  match K.view node with
  | Param { idx; dtype } when Dtype.scalar (Dtype.base dtype) = fr ->
    Some (tag (K.param ~idx ~dtype:(rebase_ptr dtype)))
  | Define_local { size; dtype } when Dtype.scalar (Dtype.base dtype) = fr ->
    Some (tag (K.define_local ~size ~dtype:(rebase_ptr dtype)))
  | Define_reg { size; dtype; slot } when Dtype.scalar (Dtype.base dtype) = fr ->
    Some (tag (K.define_reg ~size ~dtype:(rebase_ptr dtype) ~slot))
  | Load { src; dtype; _ } when Dtype.scalar_of dtype = Dtype.of_scalar fr ->
    let storage_dt = Dtype.vec (Dtype.of_scalar (f2f_dt fr)) (Dtype.count dtype) in
    Some (f2f (K.replace (K.load ~src ()) ~dtype:storage_dt ()) ~fr ~to_)
  | Cast { src; dtype } when Dtype.scalar_of (Dtype.any_to_val dtype) = Dtype.of_scalar fr ->
    Some (f2f_clamp (K.cast ~src ~dtype:(Dtype.to_any (Dtype.vec (Dtype.of_scalar to_) (Dtype.count (Dtype.any_to_val dtype))))) ~dt_scalar:fr)
  | (Binary { dtype; _ } | Unary { dtype; _ } | Ternary { dtype; _ })
    when Dtype.scalar_of dtype = Dtype.of_scalar fr ->
    let new_children = List.map (fun c ->
      match K.dtype c with
      | Some cdt when Dtype.scalar cdt = fr ->
        K.cast ~src:c ~dtype:(Dtype.to_any (Dtype.vec (Dtype.of_scalar to_) (Dtype.count cdt)))
      | _ -> c) (K.children node) in
    Some (K.replace node ~children:new_children
            ~dtype:(Dtype.vec (Dtype.of_scalar to_) (Dtype.count dtype)) ())
  | _ -> None

(* Late rewrite patterns *)

let get_late_rewrite_patterns (ops : supported_ops) node =
  match K.view node with
  | Binary { op = `Max; lhs; rhs; _ }
    when not ops.has_max && ops.has_cmplt ->
    Some (K.ternary ~op:`Where ~a:(K.binary ~op:`Cmplt ~lhs ~rhs) ~b:rhs ~c:lhs)
  | Binary { op = `Mod; lhs = x; rhs; dtype }
    when ops.has_and && Dtype.is_int dtype
         && (Dtype.is_unsigned dtype || Divandmod.vmin x >= 0L) ->
    (match const_int_val rhs with
    | Some c when c > 0L && Option.is_some (log2_of_power c) ->
      Some (K.binary ~op:`And ~lhs:x ~rhs:(K.const (Const.int64 dtype (Int64.sub c 1L))))
    | _ -> None)
  | Binary { op = `Mul; lhs; rhs; dtype }
    when ops.has_shl && Dtype.is_int dtype ->
    let try_shift base c_node = match const_int_val c_node with
      | Some c when c > 0L ->
        Option.map (fun n ->
          K.binary ~op:`Shl ~lhs:base
            ~rhs:(K.const (Const.int64 dtype (Int64.of_int n))))
          (log2_of_power c)
      | _ -> None in
    (match try_shift lhs rhs with Some _ as r -> r | None -> try_shift rhs lhs)
  | Binary { op = `Idiv; lhs = x; rhs; dtype }
    when ops.has_shr && Dtype.is_int dtype && Dtype.is_unsigned dtype ->
    (match const_int_val rhs with
    | Some c when c > 0L ->
      Option.map (fun n ->
        K.binary ~op:`Shr ~lhs:x
          ~rhs:(K.const (Const.int64 dtype (Int64.of_int n))))
        (log2_of_power c)
    | _ -> None)
  | Binary { op = `Idiv; lhs = x; rhs; dtype }
    when ops.has_shr && Dtype.is_int dtype && not (Dtype.is_unsigned dtype) ->
    (match const_int_val rhs with
    | Some c when c > 0L ->
      (match log2_of_power c with
      | Some n ->
        let correction = K.ternary ~op:`Where
          ~a:(K.binary ~op:`Cmplt ~lhs:x ~rhs:(K.const (Const.int64 dtype 0L)))
          ~b:(K.const (Const.int64 dtype (Int64.sub c 1L)))
          ~c:(K.const (Const.int64 dtype 0L)) in
        Some (K.binary ~op:`Shr
                ~lhs:(K.binary ~op:`Add ~lhs:x ~rhs:correction)
                ~rhs:(K.const (Const.int64 dtype (Int64.of_int n))))
      | None -> if not ops.disable_fast_idiv then fast_idiv x c else None)
    | _ -> None)
  | Binary { op = `Idiv; lhs = x; rhs; dtype }
    when ops.has_shr && Dtype.is_int dtype && not ops.disable_fast_idiv ->
    (match const_int_val rhs with
    | Some d when d > 0L && Option.is_none (log2_of_power d) -> fast_idiv x d
    | _ -> None)
  | Binary { op = `Mul; lhs = x; rhs; _ } when ops.has_neg ->
    (match const_int_val rhs with
    | Some (-1L) -> Some (K.unary ~op:`Neg ~src:x)
    | _ -> match const_int_val x with
      | Some (-1L) -> Some (K.unary ~op:`Neg ~src:rhs)
      | _ -> None)
  | Binary { op = `Add; lhs; rhs = c; _ } when ops.has_mulacc ->
    (match K.view lhs with
    | Binary { op = `Mul; lhs = a; rhs = b; _ } ->
      Some (K.ternary ~op:`Mulacc ~a ~b ~c)
    | _ -> match K.view c with
      | Binary { op = `Mul; lhs = a; rhs = b; _ } ->
        Some (K.ternary ~op:`Mulacc ~a ~b ~c:lhs)
      | _ -> None)
  | Unary { op = `Recip; src = x; _ } when ops.has_fdiv ->
    Some (K.binary ~op:`Fdiv
            ~lhs:(K.const (Const.float (K.dtype_or Dtype.float32 x) 1.0)) ~rhs:x)
  | _ -> None
