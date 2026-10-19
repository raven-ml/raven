(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Each kind is nx's C ([nx_kinds_real.h], [nx_kinds.h]) written over nodes: the
   same operations in the same order, an fma ([Mulacc], rounded once) where the
   C writes one. A C branch is a selection, so both sides compute: where a
   side's value would be undefined in C on lanes it does not take, its operands
   are made safe first. *)

open Tolk
open Lower_bits

(* Types

   A node is a [float32] or a [float64]; its bits are the unsigned integer of
   its width. A pair [(a, b)] is a constant's value at each: [a] at float32, [b]
   at float64. *)

let wide x = Dtype.equal (Ops.dtype x) Float64
let pick x (a, b) = if wide x then b else a
let width x = if wide x then 64 else 32
let num u v = Shape.const_like u (`Float v)
let int u n = Shape.const_like u (`Int (Bigint.of_int n))
let word u n = Shape.const_like u (`Int (Bigint.of_int64_unsigned n))

let u64 u n =
  Shape.const_like ~dtype:Uint64 u (`Int (Bigint.of_int64_unsigned n))

let con x p = num x (pick x p)

let of_bits u =
  Ops.bitcast u (if Dtype.equal (Ops.dtype u) Uint64 then Float64 else Float32)

(* Operations *)

let ( +: ) = Ops.add
let ( -: ) = Ops.sub
let ( *: ) = Ops.mul
let ( /: ) x y = Ops.alu x Op.Fdiv [ y ]
let where = Ops.where
let lt = Ops.lt
let gt x y = Ops.lt y x
let eq = Ops.eq
let ne = Ops.ne
let ( ||: ) = Ops.bitwise_or
let ( &&: ) = Ops.bitwise_and

(* [x <= y] and [x >= y], false where an operand is NaN. *)
let le x y = lt x y ||: eq x y
let ge x y = le y x
let band = Ops.bitwise_and
let bor = Ops.bitwise_or
let bxor = Ops.bitwise_xor
let shl u n = Ops.shl u (int u n)
let shr u n = Ops.shr u (int u n)
let set u = ne u (int u 0)

(* Constants *)

let one x = num x 1.
let zero x = num x 0.
let half x = num x 0.5
let two x = num x 2.
let inf x = num x Float.infinity
let sign_word x = word (bits x) (Int64.shift_left 1L (width x - 1))
let ones x = word (bits x) (if wide x then -1L else 0xFFFFFFFFL)
let mag_word x = word (bits x) (Int64.pred (Int64.shift_left 1L (width x - 1)))
let bias = (127, 1023)
let mbits = (23, 52)
let mmask = (0x007FFFFFL, 0x000FFFFFFFFFFFFFL)
let one_bits = (0x3F800000L, 0x3FF0000000000000L)
let sqrt_half_bits = (0x3F3504F3L, 0x3FE6A09E667F3BCDL)
let min_normal_bits = (0x00800000L, 0x0010000000000000L)
let inf_bits = (0x7F800000L, 0x7FF0000000000000L)
let sub_scale = (0x1p23, 0x1p52)

(* Adding the shifter rounds a float below half of it to the nearest integer,
   ties to even; from [integral] on every float is an integer. *)
let shifter = (0x1.8p23, 0x1.8p52)
let integral = (0x1p23, 0x1p52)
let log2e = (0x1.715476p+0, 0x1.71547652b82fep+0)
let log2e_lo = (0x1.4ae0cp-26, 0x1.777d0ffda0d24p-56)
let ln2_hi = (0x1.62e4p-1, 0x1.62e42fefa38p-1)
let ln2_lo = (0x1.7f7d1cp-20, 0x1.ef35793c7673p-45)
let ivln2_hi = (0x1.716p+0, 0x1.71547652p+0)
let ivln2_lo = (-0x1.7135a8p-13, 0x1.705fc2eefa2p-33)
let log2_hi_mask = (0xFFFFF000L, 0xFFFFFFFF00000000L)
let two_thirds_hi = (0x1.555556p-1, 0x1.5555555555555p-1)
let two_thirds_lo = (-0x1.555556p-26, 0x1.5555555555555p-55)
let two_over_pi = (0x1.45f306p-1, 0x1.45f306dc9c883p-1)
let pio2_hi = (0x1.921fb6p+0, 0x1.921fb54442d18p+0)
let pio2_lo = (-0x1.777a5cp-25, 0x1.1a62633145c07p-54)
let pio2_3 = (-0x1.ee59dap-50, -0x1.f1976b7ed8fbcp-110)
let pi_hi = (0x1.921fb6p+1, 0x1.921fb54442d18p+1)
let pi_lo = (-0x1.777a5cp-24, 0x1.1a62633145c07p-53)
let exp_hi = (89., 710.)
let exp_lo = (-104., -746.)
let exp2_hi = (129., 1025.)
let exp2_lo = (-151., -1076.)
let expm1_lo = (-26., -40.)
let expm1_kcap = (64, 128)
let expm1_tiny = (0x1p-25, 0x1p-54)
let hyp_hi = (90., 712.)
let hyp_kcap = (32, 64)
let tanh_small = (0.625, 0.625)
let tanh_hi = (9.5, 19.5)
let pow_hi = (130., 1030.)
let pow_lo = (-160., -1080.)
let pio2_big = (0x47000000L, 0x41B0000000000000L)
let trig_tiny = (0x1p-12, 0x1p-27)

(* Polynomials, each from its leading coefficient. *)

let exp_q =
  ( [
      0x1.6ac6e2p-10; 0x1.123ddcp-7; 0x1.555858p-5; 0x1.55548cp-3; 0x1.fffffcp-2;
    ],
    [
      0x1.ad7f55323502cp-26;
      0x1.28ad86039e73p-22;
      0x1.71df2581c2cafp-19;
      0x1.a01999f89494ap-16;
      0x1.a01a012a34bddp-13;
      0x1.6c16c18430f75p-10;
      0x1.1111111127c73p-7;
      0x1.555555555085fp-5;
      0x1.55555555554fap-3;
      0x1.000000000000ap-1;
    ] )

let exp2_p =
  ( [
      0x1.177586p-13;
      0x1.5ef2ep-10;
      0x1.3b6816p-7;
      0x1.c6af9ep-5;
      0x1.ebfb94p-3;
      0x1.62e43p-1;
      1.;
    ],
    [
      0x1.c2cea03f4f3ap-36;
      0x1.ea014ee31fd31p-32;
      0x1.e4d06c7074e68p-28;
      0x1.b524dd55482cfp-24;
      0x1.62c021e26f2c5p-20;
      0x1.ffcbfc74d611dp-17;
      0x1.430912f883cf5p-13;
      0x1.5d87fe78a26c9p-10;
      0x1.3b2ab6fba4e22p-7;
      0x1.c6b08d704a0cfp-5;
      0x1.ebfbdff82c58fp-3;
      0x1.62e42fefa39efp-1;
      1.;
    ] )

let log_t =
  ( [ 0x1.ab4e0cp-3; 0x1.257d1cp-2; 0x1.9996acp-2; 0x1.555556p-1 ],
    [
      0x1.0c1994438ce53p-3;
      0x1.0fbc6f8b23776p-3;
      0x1.3b1c4a71fa6cep-3;
      0x1.745cf89e46188p-3;
      0x1.c71c720256c1fp-3;
      0x1.2492492476379p-2;
      0x1.9999999999a3cp-2;
      0x1.5555555555555p-1;
    ] )

let log_tp =
  ( [ 0x1.d8279ap-3; 0x1.2479cap-2; 0x1.9999a4p-2 ],
    [
      0x1.e0550027cf2a4p-4;
      0x1.df790206d23b4p-4;
      0x1.1118da70f5042p-3;
      0x1.3b1395780c99fp-3;
      0x1.745d177b75a35p-3;
      0x1.c71c71c6e999ap-3;
      0x1.2492492492525p-2;
      0x1.999999999999ap-2;
    ] )

let sin_s =
  ( [ -0x1.9952fap-13; 0x1.110776p-7; -0x1.555546p-3 ],
    [
      0x1.5d8ee79c0623bp-33;
      -0x1.ae5e57812c7p-26;
      0x1.71de356408ad5p-19;
      -0x1.a01a019bf9b6p-13;
      0x1.111111110f7bp-7;
      -0x1.5555555555548p-3;
    ] )

let cos_c =
  ( [ 0x1.99e80cp-16; -0x1.6c0c28p-10; 0x1.55554ap-5 ],
    [
      -0x1.8fa3958a5c52ap-37;
      0x1.1ee9d5c91b1dfp-29;
      -0x1.27e4f7e80273p-22;
      0x1.a01a019c80966p-16;
      -0x1.6c16c16c14f6cp-10;
      0x1.555555555554bp-5;
    ] )

let asin_b =
  ( [ 0x1.595c8cp-5; 0x1.8c3e2ap-6; 0x1.747bbap-5; 0x1.330204p-4; 0x1.5555c8p-3 ],
    [
      0x1.056cbe7839745p-5;
      -0x1.09d10e2ce02f9p-6;
      0x1.3ff33c58c8071p-6;
      0x1.abd19915948edp-8;
      0x1.8ec2b52f17c83p-7;
      0x1.c6fdcd892b624p-7;
      0x1.1c6be0fccac8fp-6;
      0x1.6e89f44cb6e5ap-6;
      0x1.f1c72c38610a7p-6;
      0x1.6db6db427c841p-5;
      0x1.333333336e7cbp-4;
      0x1.555555555539p-3;
    ] )

let tanh_t =
  ( [
      -0x1.750ce4p-8;
      0x1.51f8cep-6;
      -0x1.b835a4p-5;
      0x1.1106f6p-3;
      -0x1.555532p-2;
    ],
    [
      -0x1.0da727a627179p-16;
      0x1.4391214300e67p-14;
      -0x1.df5579a8b4a07p-13;
      0x1.336612e405191p-11;
      -0x1.7d68a32fdd9b9p-10;
      0x1.d6cf280ecda0fp-9;
      -0x1.226e162bf7b5p-7;
      0x1.664f477adac7ap-6;
      -0x1.ba1ba1b5064afp-5;
      0x1.111111110accdp-3;
      -0x1.55555555554ffp-2;
    ] )

let horner x = function
  | [] -> invalid_arg "Lower_real.horner: no coefficient"
  | c :: cs -> List.fold_left (fun p c -> mulacc p x (num x c)) (num x c) cs

let poly x p = horner x (pick x p)

(* Helpers *)

let copysign m s =
  of_bits (bor (band (bits m) (mag_word m)) (band (bits s) (sign_word s)))

(* 2^k for a normal power, [k] an int32 node. *)
let pow2 x k =
  let e = Ops.cast (Ops.add k (int k (pick x bias))) (unsigned (Ops.dtype x)) in
  of_bits (shl e (pick x mbits))

(* y 2^k rounded once: the first product is exact. *)
let scale y k =
  let k1 = shr k 1 in
  y *: pow2 y k1 *: pow2 y (Ops.sub k k1)

let two_sum a b =
  let hi = a +: b in
  let bb = hi -: a in
  (hi, a -: (hi -: bb) +: (b -: bb))

let fast_two_sum hi lo =
  let h = hi +: lo in
  (h, lo -: (h -: hi))

let div_pair (nh, nl) (dh, dl) =
  let nh, nl = fast_two_sum nh nl in
  let dh, dl = fast_two_sum dh dl in
  let rd = one dh /: dh in
  let qh = nh *: rd in
  (qh, (mulacc (negated qh) dh nh +: mulacc (negated qh) dl nl) *: rd)

(* From t = v + shifter, the integer nearest v: as a float, and as an int32 from
   the difference of the bits. *)
let nearest t =
  let s = con t shifter in
  let d =
    Ops.sub
      (Ops.bitcast t (signed (Ops.dtype t)))
      (Ops.bitcast s (signed (Ops.dtype t)))
  in
  (t -: s, Ops.cast d Int32)

(* NaN results: the first NaN operand, else the quiet NaN of a clear sign where
   r is a NaN. *)

let nan1 a r = where (is_nan a) a (where (is_nan r) (nan r) r)
let nan2 a b r = where (is_nan a) a (nan1 b r)

(* Exact kinds *)

let sign a =
  let s =
    where (gt a (zero a)) (one a) (where (lt a (zero a)) (num a (-1.)) (zero a))
  in
  where (is_nan a) a s

let trunc a =
  let m = magnitude a in
  let c = con a integral in
  let t = m +: c -: c in
  let t = where (gt t m) (t -: one a) t in
  let r = of_bits (bor (bits t) (band (bits a) (sign_word a))) in
  where (lt m c) r a

let floor a =
  let t = trunc a in
  where (gt t a) (t -: one a) t

let ceil a =
  let t = trunc a in
  where (lt t a) (t +: one a) t

let round a =
  let t = trunc a in
  let d = magnitude (a -: t) in
  where (ge d (half a)) (where (lt a (zero a)) (t -: one a) (t +: one a)) t

(* The greater (lesser) operand, [a] where [a] is NaN, and of equal operands the
   AND (OR) of their bits. *)
let maximum a b =
  let g = where (is_nan a) a (where (gt a b) a b) in
  let ne = where (ne a b) (ones a) (word (bits a) 0L) in
  of_bits (band (bits g) (bor (bits a) ne))

let minimum a b =
  let g = where (is_nan a) a (where (lt a b) a b) in
  let eq = where (ne a b) (word (bits a) 0L) (ones a) in
  of_bits (bor (bits g) (band (bits a) eq))

(* Exponentials *)

(* x = k ln2 + r, |r| <= ln2/2, and rc r's rounding error. *)
let exp_reduce x =
  let kf, k = nearest (mulacc x (con x log2e) (con x shifter)) in
  let m = negated kf in
  let rh = mulacc m (con x ln2_hi) x in
  let r = mulacc m (con x ln2_lo) rh in
  (r, mulacc m (con x ln2_lo) (rh -: r), k)

let exp x =
  let c = where (lt x (con x exp_hi)) x (con x exp_hi) in
  let c = where (gt c (con x exp_lo)) c (con x exp_lo) in
  let r, _, k = exp_reduce c in
  let em = mulacc (r *: r) (poly r exp_q) r in
  where (is_nan x) x (scale (one x +: em) k)

(* exp where 2^k (1 + em) is normal: k added to the exponent. *)
let exp_near x =
  let r, _, k = exp_reduce x in
  let em = mulacc (r *: r) (poly r exp_q) r in
  let b = bits (one x +: em) in
  of_bits (Ops.add b (shl (Ops.cast k (unsigned (Ops.dtype x))) (pick x mbits)))

let exp2 x =
  let x = where (gt x (con x exp2_hi)) (con x exp2_hi) x in
  let x = where (lt x (con x exp2_lo)) (con x exp2_lo) x in
  let kf, k = nearest (x +: con x shifter) in
  nan1 x (scale (poly (x -: kf) exp2_p) k)

let expm1 x =
  let c = where (gt x (con x exp_hi)) (con x exp_hi) x in
  let c = where (lt c (con x expm1_lo)) (con x expm1_lo) c in
  let r, rc, k = exp_reduce c in
  let h = mulacc (r *: r) (poly r exp_q) (mulacc rc r rc) in
  let cap = int k (pick x expm1_kcap) in
  let m = pow2 x (Ops.sub (int k 0) (where (gt k cap) cap k)) in
  let uh, ul = two_sum (one x) (negated m) in
  let sh, sl = two_sum uh r in
  let y = scale (sh +: (sl +: (ul +: h))) k in
  nan1 x (where (lt (magnitude x) (con x expm1_tiny)) x y)

(* Logarithms *)

(* u = 2^k (1 + f), f in [sqrt(1/2) - 1, sqrt 2 - 1); a subnormal scaled up
   first. *)
let log_reduce u =
  let b = bits u in
  let sub = lt b (word b (pick u min_normal_bits)) in
  let u = where sub (u *: con u sub_scale) u in
  let ix =
    Ops.add (bits u)
      (word b (Int64.sub (pick u one_bits) (pick u sqrt_half_bits)))
  in
  let e = Ops.cast (shr ix (pick u mbits)) Int32 in
  let e = Ops.sub e (int e (pick u bias)) in
  let k = Ops.sub e (where sub (int e (pick u mbits)) (int e 0)) in
  let m =
    Ops.add (band ix (word b (pick u mmask))) (word b (pick u sqrt_half_bits))
  in
  (of_bits m -: one u, Ops.cast k (Ops.dtype u))

(* k ln2 + log(1 + f) + c. *)
let log_core (f, k) c =
  let s = f /: (two f +: f) in
  let z = s *: s in
  let hfsq = half f *: f *: f in
  let sr =
    mulacc s (mulacc z (poly z log_t) hfsq) (mulacc k (con f ln2_lo) c)
  in
  mulacc k (con f ln2_hi) (negated (hfsq -: sr -: f))

let log_special u x y =
  let y = where (eq u (inf u)) u y in
  let y = where (eq u (zero u)) (negated (inf u)) y in
  let y = where (lt u (zero u)) (nan u) y in
  where (is_nan x) x y

let log x = log_special x x (log_core (log_reduce x) (zero x))

let log2 x =
  let f, k = log_reduce x in
  let s = f /: (two f +: f) in
  let z = s *: s in
  let hfsq = half f *: f *: f in
  let hi =
    of_bits (band (bits (f -: hfsq)) (word (bits f) (pick x log2_hi_mask)))
  in
  let lo = mulacc s (mulacc z (poly z log_t) hfsq) (f -: hi -: hfsq) in
  let v = mulacc lo (con x ivln2_hi) ((lo +: hi) *: con x ivln2_lo) in
  log_special x x (mulacc hi (con x ivln2_hi) v +: k)

let log1p x =
  let uh, ul = two_sum (one x) x in
  let y = log_core (log_reduce uh) (ul /: uh) in
  log_special uh x (where (eq x (zero x)) x y)

(* Trigonometric reductions: x = q pi/2 + hi + lo, q an int32. *)

(* Cody and Waite: at float32, pi/2 in three parts each subtracted by an fma; at
   float64, the second product and its subtraction kept as pairs. *)
let rem_cw x =
  let qf, q = nearest (mulacc x (con x two_over_pi) (con x shifter)) in
  let mq = negated qf in
  if not (wide x) then
    let a = mulacc mq (con x pio2_hi) x in
    let b = mulacc mq (con x pio2_lo) a in
    let hi = mulacc mq (con x pio2_3) b in
    (hi, mulacc mq (con x pio2_3) (b -: hi), q)
  else
    let a = mulacc mq (con x pio2_hi) x in
    let p = qf *: con x pio2_lo in
    let pe = mulacc qf (con x pio2_lo) (negated p) in
    let bh, bl = two_sum a (negated p) in
    let lo = mulacc mq (con x pio2_3) (bl -: pe) in
    let hi, lo = fast_two_sum bh lo in
    (hi, lo, q)

(* The bits of 2/pi after the point, behind one zero word. *)
let two_over_pi_words =
  [|
    0x00000000L;
    0xA2F9836EL;
    0x4E441529L;
    0xFC2757D1L;
    0xF534DDC0L;
    0xDB629599L;
    0x3C439041L;
    0xFE5163ABL;
    0xDEBBC561L;
    0xB7246E3AL;
    0x424DD2E0L;
    0x06492EEAL;
    0x09D1921CL;
    0xFE1DEB1CL;
    0xB129A73EL;
    0xE88235F5L;
    0x2EBB4484L;
    0xE99C7026L;
    0xB45F7E41L;
    0x3991D639L;
    0x835339F4L;
    0x9C845F8BL;
    0xBDF9283BL;
    0x1FF897FFL;
    0xDE05980FL;
    0xEF2F118BL;
    0x5A0A6D1FL;
    0x6D367ECFL;
    0x27CB09B7L;
    0x4F463F66L;
    0x9E5FEA2DL;
    0x7527BAC7L;
    0xEBE5F17BL;
    0x3D0739F7L;
    0x8A5292EAL;
    0x6BFB5FB1L;
    0x1F8D5D08L;
    0x56033046L;
    0xFC7B6BABL;
    0xF0CFBC20L;
    0x9AF4361DL;
  |]

(* The bits of 2 ln2 / pi after the point, behind one zero word, for the complex
   exp2's phase y ln2, reduced as the trigonometric kinds reduce theirs. *)
let two_ln2_over_pi_words =
  [|
    0x00000000L;
    0x70F7263DL;
    0xFA5A6EECL;
    0xB608582EL;
    0x55FEF3DDL;
    0x8E13BECDL;
    0x1D495C36L;
    0x1D72013AL;
    0xEA9F8FCAL;
    0x76E866FCL;
    0x9B48BE48L;
    0x15F5CB6FL;
    0xD8BEE211L;
    0x0394DB30L;
    0x20D6EB52L;
    0x30FFA9AEL;
    0x90CBC735L;
    0x51358CEDL;
    0x67C0DB3CL;
    0x490D15BDL;
    0xD91A29F5L;
    0x682402AFL;
    0x5E8F9C23L;
    0xBFF15CBCL;
    0x71F05FEDL;
    0xFE0BB062L;
    0x2A8C8F67L;
    0x7309C45AL;
    0x7502282AL;
    0xEAD0C243L;
    0x5C5527BBL;
    0x759F345EL;
    0x2D0628E8L;
    0xB1A563A2L;
    0x53E62918L;
    0xEA5FFDABL;
    0xEAC0D479L;
    0xCFEDB1B6L;
    0xFC78B439L;
    0xA333A25FL;
    0xDC77BE31L;
  |]

type table = Two_over_pi | Two_ln2_over_pi

let table_words = function
  | Two_over_pi -> two_over_pi_words
  | Two_ln2_over_pi -> two_ln2_over_pi_words

(* [window t g n nb] reads [n] runs of 32 bits of table [t]'s number, from bits
   [g], [g + 32] and so on, bit 0 the first of the zero word: the table's words
   [w] to [w + n], [w = g >> 5] of [nb] bits, are moved into place by one
   selection per bit of [w], as a barrel shifter does. *)
let window t g n nb =
  let t = table_words t in
  let w = shr g 5 in
  let sh = Ops.cast (band g (int g 31)) Uint64 in
  let at m = u64 g (if m < Array.length t then t.(m) else 0L) in
  let words = ref (Array.init (n + (1 lsl nb)) at) in
  for b = nb - 1 downto 0 do
    let step = 1 lsl b and prev = !words in
    let on = set (band w (int w step)) in
    words :=
      Array.init
        (Array.length prev - step)
        (fun m -> where on prev.(m + step) prev.(m))
  done;
  let ws = !words in
  Array.init n (fun j ->
      let pair = bor (shl ws.(j) 32) ws.(j + 1) in
      band (shr (Ops.shl pair sh) 32) (u64 g 0xFFFFFFFFL))

(* Payne and Hanek at float32: x k reduced by pi/2, for finite |x| >= 2^15 and
   [t] the bits of k 2/pi, below 1. x = m 2^e, and a window W of 96 bits of [t]
   gives x t = m W 2^-94 mod 4. Bits 94 and 95 of m W are q, the 64 below the
   fraction, rounded to nearest so that it is signed. Other lanes read a safe
   exponent. *)
let rem_big32 t x =
  let ix = band (bits x) (mag_word x) in
  let big = word ix (pick x pio2_big) in
  let ix = where (lt ix big) big ix in
  let e = Ops.cast (shr ix 23) Int32 in
  let e = Ops.sub e (int e 150) in
  let m =
    Ops.cast (bor (band ix (word ix 0x007FFFFFL)) (word ix 0x00800000L)) Uint64
  in
  let g = Ops.add e (int e 30) in
  let w = window t g 3 3 in
  let a0 = m *: w.(2) and a1 = m *: w.(1) and a2 = m *: w.(0) in
  let low u = band u (u64 u 0xFFFFFFFFL) in
  let t1 = Ops.add (shr a0 32) a1 in
  let t2 = Ops.add (shr t1 32) a2 in
  let f = bor (bor (shl t2 34) (shl (low t1) 2)) (shr (low a0) 30) in
  let top = shr f 63 in
  let q = Ops.add (Ops.cast (shr t2 30) Int32) (Ops.cast top Int32) in
  let part u s = Ops.cast u Float32 *: num x s in
  let c2 = part (shr f 40) 0x1p-24 in
  let c1 = part (band (shr f 16) (u64 f 0xFFFFFFL)) 0x1p-48 in
  let c0 = part (band f (u64 f 0xFFFFL)) 0x1p-64 in
  let c2 = where (set top) (c2 -: one x) c2 in
  let vh, vl = two_sum c2 c1 in
  let vlo = vl +: c0 in
  let ph = vh *: con x pio2_hi in
  let pl = mulacc vh (con x pio2_hi) (negated ph) in
  let pl = mulacc vh (con x pio2_lo) (mulacc vlo (con x pio2_hi) pl) in
  let hi = ph +: pl in
  let lo = pl -: (hi -: ph) in
  let n = negative x in
  ( where n (negated hi) hi,
    where n (negated lo) lo,
    where n (Ops.sub (int q 0) q) q )

(* Payne and Hanek at float64, for finite |x| >= 2^28: m of 53 bits and a window
   of 192 bits, the fraction's top 128 bits kept. *)
let rem_big64 t x =
  let ix = band (bits x) (mag_word x) in
  let big = word ix (pick x pio2_big) in
  let ix = where (lt ix big) big ix in
  let e = Ops.cast (shr ix 52) Int32 in
  let e = Ops.sub e (int e 1075) in
  let m =
    bor (band ix (word ix 0x000FFFFFFFFFFFFFL)) (word ix 0x0010000000000000L)
  in
  let g = Ops.add e (int e 30) in
  let win = window t g 6 5 in
  let w = Array.init 6 (fun j -> win.(5 - j)) in
  let low u = band u (u64 u 0xFFFFFFFFL) in
  let ml = low m and mh = shr m 32 in
  let p = Array.make 8 (u64 m 0L) in
  let c = ref (u64 m 0L) in
  for j = 0 to 5 do
    let t = Ops.add (ml *: w.(j)) !c in
    p.(j) <- low t;
    c := shr t 32
  done;
  p.(6) <- !c;
  c := u64 m 0L;
  for j = 0 to 5 do
    let t = Ops.add (Ops.add (mh *: w.(j)) p.(j + 1)) !c in
    p.(j + 1) <- low t;
    c := shr t 32
  done;
  p.(7) <- !c;
  let mask30 u = band u (u64 u 0x3FFFFFFFL) in
  let fh = bor (bor (shl (mask30 p.(5)) 34) (shl p.(4) 2)) (shr p.(3) 30) in
  let fl = bor (bor (shl (mask30 p.(3)) 34) (shl p.(2) 2)) (shr p.(1) 30) in
  let q =
    Ops.add (Ops.cast (shr p.(5) 30) Int32) (Ops.cast (shr fh 63) Int32)
  in
  let c2 = Ops.cast (Ops.cast (shr fh 32) Int32) Float64 *: num x 0x1p-32 in
  let c1 = Ops.cast (low fh) Float64 *: num x 0x1p-64 in
  let c0 = Ops.cast (shr fl 11) Float64 *: num x 0x1p-117 in
  let sh, sl = two_sum c2 c1 in
  let vh, vl = fast_two_sum sh (sl +: c0) in
  let ph = vh *: con x pio2_hi in
  let pl = mulacc vh (con x pio2_hi) (negated ph) in
  let pl = mulacc vh (con x pio2_lo) (mulacc vl (con x pio2_hi) pl) in
  let hi, lo = fast_two_sum ph pl in
  let n = negative x in
  ( where n (negated hi) hi,
    where n (negated lo) lo,
    where n (Ops.sub (int q 0) q) q )

(* Whether x takes the integer reduction: |x| finite and at or past the
   switch. *)
let trig_big x =
  let a = band (bits x) (mag_word x) in
  Ops.logical_not (lt a (word a (pick x pio2_big)))
  &&: lt a (word a (pick x inf_bits))

let rem_big t x = if wide x then rem_big64 t x else rem_big32 t x

let rem x =
  let big = trig_big x in
  let bh, bl, bq = rem_big Two_over_pi x in
  let nh, nl, nq = rem_cw x in
  (where big bh nh, where big bl nl, where big bq nq)

let sin_kernel hi lo =
  let z = hi *: hi in
  (hi, mulacc (hi *: z) (poly z sin_s) lo)

let cos_kernel hi lo =
  let z = hi *: hi in
  let zl = mulacc hi hi (negated z) in
  let hz = half hi *: z in
  let ch = one hi -: hz in
  let tail =
    mulacc (z *: z) (poly z cos_c)
      (mulacc (negated hi) lo (num hi (-0.5) *: zl))
  in
  (ch, one hi -: ch -: hz +: tail)

(* y with its sign flipped where bit 1 of k is set. *)
let flip2 y k =
  let b = Ops.cast (band k (int k 2)) (unsigned (Ops.dtype y)) in
  of_bits (bxor (bits y) (shl b (width y - 2)))

let odd q = set (band q (int q 1))
let small x = Ops.logical_not (ge (magnitude x) (con x trig_tiny))

(* sin x and cos x from x's reduction [(hi, lo, q)]. *)

let sin_at x (hi, lo, q) =
  let sh, sl = sin_kernel hi lo and ch, cl = cos_kernel hi lo in
  let y = flip2 (where (odd q) (ch +: cl) (sh +: sl)) q in
  let y = where (is_nan y) (nan y) y in
  where (small x) x y

let cos_at x (hi, lo, q) =
  let sh, sl = sin_kernel hi lo and ch, cl = cos_kernel hi lo in
  let y = flip2 (where (odd q) (sh +: sl) (ch +: cl)) (Ops.add q (int q 1)) in
  let y = where (is_nan y) (nan y) y in
  where (is_nan x) x y

let sin x = sin_at x (rem x)
let cos x = cos_at x (rem x)

let tan x =
  let hi, lo, q = rem x in
  let sh, sl = sin_kernel hi lo and ch, cl = cos_kernel hi lo in
  (* -cos / sin in odd quadrants, sin / cos in even *)
  let o = odd q in
  let n = (where o (negated ch) sh, where o (negated cl) sl) in
  let d = (where o sh ch, where o sl cl) in
  let qh, ql = div_pair n d in
  let y = qh +: ql in
  let y = where (is_nan y) (nan y) y in
  where (small x) x y

(* Inverse sine and cosine *)

(* asin a as s + tail: a itself up to 1/2, else s = sqrt((1 - a)/2), whose asin
   the caller doubles. *)
let asin_reduce a =
  let big = gt a (half a) in
  let z = where big ((one a -: a) *: half a) (a *: a) in
  let s = where big (Ops.sqrt z) a in
  let c =
    where (big &&: gt z (zero a)) (mulacc (negated s) s z /: (s +: s)) (zero a)
  in
  (s, mulacc (s *: z) (poly z asin_b) c, big)

let asin x =
  let a = magnitude x in
  let s, tail, big = asin_reduce a in
  let d = s +: s in
  let hi = con x pio2_hi -: d in
  let lo =
    con x pio2_hi -: hi -: d +: mulacc (num x (-2.)) tail (con x pio2_lo)
  in
  let y = where big (hi +: lo) (s +: tail) in
  let y = where (gt a (one x)) (nan x) y in
  nan1 x (copysign y x)

let acos x =
  let a = magnitude x in
  let s, tail, big = asin_reduce a in
  let n = negative x in
  let term = where big (s +: s) s in
  let tail = where big (tail +: tail) tail in
  let base_hi = where big (where n (con x pi_hi) (zero x)) (con x pio2_hi) in
  let base_lo = where big (where n (con x pi_lo) (zero x)) (con x pio2_lo) in
  let plus = ne big n in
  let term = where plus term (negated term) in
  let tail = where plus tail (negated tail) in
  let hh, hl = two_sum base_hi term in
  let y = hh +: (hl +: (base_lo +: tail)) in
  nan1 x (where (gt a (one x)) (nan x) y)

(* Hyperbolic functions: for a = k ln2 + r, em = expm1 r and w = 1 + em as
   pairs, and 2^-2k. *)

let hyp_reduce a =
  let a = where (gt a (con a hyp_hi)) (con a hyp_hi) a in
  let r, rc, k = exp_reduce a in
  let em_lo = mulacc (r *: r) (poly r exp_q) (mulacc rc r rc) in
  let wh, wl = two_sum (one a) r in
  let cap = int k (pick a hyp_kcap) in
  let m = pow2 a (Ops.mul (where (gt k cap) cap k) (int k (-2))) in
  ((r, em_lo), (wh, wl +: em_lo), m, k)

let sinh x =
  let (eh, el), w, m, k = hyp_reduce (magnitude x) in
  let nh, nl = two_sum (one x -: m) eh in
  let qh, ql = div_pair (nh, nl +: el) w in
  let vh, vl = two_sum eh qh in
  let y = vh +: (vl +: (el +: ql)) in
  nan1 x (copysign (scale y (Ops.sub k (int k 1))) x)

let cosh x =
  let _, (wh, wl), m, k = hyp_reduce (magnitude x) in
  let qh, ql = div_pair (m, zero x) (wh, wl) in
  let vh, vl = two_sum wh qh in
  let y = vh +: (vl +: (wl +: ql)) in
  nan1 x (scale y (Ops.sub k (int k 1)))

let tanh x =
  let a = magnitude x in
  let z = a *: a in
  let small = mulacc (a *: z) (poly z tanh_t) a in
  let hi = con x tanh_hi in
  let w = exp_near (two x *: where (lt a hi) a hi) in
  let dh, dl = fast_two_sum w (one x) in
  let qh = two x /: dh in
  let ql = (mulacc (negated qh) dh (two x) -: (qh *: dl)) *: (half x *: qh) in
  let s = one x -: qh in
  let y =
    where (lt a (con x tanh_small)) small (s +: (one x -: s -: qh -: ql))
  in
  let y = of_bits (bor (bits y) (band (bits x) (sign_word x))) in
  where (is_nan x) x y

(* pow: 2^(y log2|x|), log2|x| as a pair, the exponential exp2's. *)
let pow x y =
  let ax = magnitude x and ay = magnitude y in
  let f, k = log_reduce ax in
  let d = two x +: f in
  let dl = f -: (d -: two x) in
  let rd = one x /: d in
  let sh = f *: rd in
  let sl = mulacc (negated sh) dl (mulacc (negated sh) d f) *: rd in
  (* s^3 as c + cl, sl's share 3 s^2 sl included *)
  let z = sh *: sh in
  let zl = mulacc sh sh (negated z) in
  let c = sh *: z in
  let cl = mulacc (num x 3. *: z) sl (mulacc sh zl (mulacc sh z (negated c))) in
  (* (2/3) s^3 as a + al *)
  let a = c *: con x two_thirds_hi in
  let al =
    mulacc c (con x two_thirds_hi) (negated a)
    +: mulacc c (con x two_thirds_lo) (cl *: con x two_thirds_hi)
  in
  let tp = poly z log_tp in
  let lh = sh +: sh in
  let head = lh +: a in
  let ll = lh -: head +: a +: (sl +: sl +: mulacc (c *: z) tp al) in
  let lh = head in
  (* log2|x| = k + (lh + ll) / ln2 *)
  let ph = lh *: con x log2e in
  let pl = mulacc lh (con x log2e) (negated ph) in
  let pl = mulacc lh (con x log2e_lo) (mulacc ll (con x log2e) pl) in
  let g0, g1 = two_sum k ph in
  let gh = g0 +: (g1 +: pl) in
  let gl = g1 +: pl -: (gh -: g0) in
  let at_inf = eq ax (inf x) and at_zero = eq ax (zero x) in
  let gh = where at_inf ax (where at_zero (negated (inf x)) gh) in
  let gl = where (at_inf ||: at_zero) (zero x) gl in
  (* y log2|x|; past the range the result is 0 or inf and the tail may be NaN *)
  let hi = con x pow_hi and lo = con x pow_lo in
  let yh = y *: gh in
  let yl = mulacc y gl (mulacc y gh (negated yh)) in
  let yl = where (le yh hi &&: ge yh lo) yl (zero x) in
  let yh = where (gt yh hi) hi yh in
  let yh = where (lt yh lo) lo yh in
  let kf, ki = nearest (yh +: con x shifter) in
  let v = scale (poly (yh -: kf +: yl) exp2_p) ki in
  (* y's parity: ay + 2^(p-1) rounds ay to an integer, its last bit worth 1 *)
  let i = con x integral in
  let ty = ay +: i in
  let integer = ge ay i ||: eq (ty -: i) ay in
  let last = where (ge ay i) (bits ay) (bits ty) in
  let odd = lt ay (two x *: i) &&: integer &&: set (band last (int last 1)) in
  let n = negative x in
  let v = where (n &&: odd) (negated v) v in
  let finite = ne ax (zero x) &&: lt ax (inf x) in
  let v = where (n &&: Ops.logical_not integer &&: finite) (nan x) v in
  let v = where (eq ax (one x) &&: eq ay (inf x)) (one x) v in
  let v = nan2 x y v in
  where (eq x (one x) ||: eq y (zero x)) (one x) v

(* Type-specific kinds: atan, atan2, erf and mod. *)

let atan_k32 t =
  let z = t *: t in
  let a =
    horner z
      [
        0x1.7ed21ep-9;
        -0x1.0c2c0cp-6;
        0x1.61fdd4p-5;
        -0x1.3556b4p-4;
        0x1.b4e12ap-4;
        -0x1.230adcp-3;
        0x1.9978f4p-3;
        -0x1.5554dcp-2;
      ]
  in
  mulacc (t *: z) a t

let atan_k64 t =
  let z = t *: t in
  let a =
    horner z
      [
        -0x1.0a7ed459d86b8p-6;
        0x1.2b17be1f3a52cp-5;
        -0x1.97a089c9a7af5p-5;
        0x1.ddddadf2f06c5p-5;
        -0x1.10d6017d18c7ap-4;
        0x1.3b0f205706eb3p-4;
        -0x1.745cdba3cb3p-4;
        0x1.c71c6fdb0e4afp-4;
        -0x1.2492491ff2fd8p-3;
        0x1.999999998e7dp-3;
        -0x1.555555555550bp-2;
      ]
  in
  t *: z *: a

(* atan a, a >= 0, as a pair at float64: a = c + (a - c)/(1 + c a), c the
   nearest of 0, 1/2, 1, 3/2 and inf; the quotient's error and al kept in a
   tail. *)
let atan_pos a al =
  let n v = num a v in
  let b1 = ge a (n 0.4375) and b2 = ge a (n 0.6875) in
  let b3 = ge a (n 1.1875) and b4 = ge a (n 2.4375) in
  let c = where b3 (n 1.5) (where b2 (n 1.) (n 0.5)) in
  let num_ = where b4 (n (-1.)) (a -: c) in
  let den = where b4 a (mulacc c a (n 1.)) in
  let rden = n 1. /: den in
  let t = where b1 (num_ *: rden) a in
  let rest = mulacc (negated t) den num_ in
  let tl = where b1 (mulacc al (mulacc (negated t) c (n 1.)) rest *: rden) al in
  let tl = where b4 (rest *: rden) tl in
  let tl = where (lt a (inf a)) tl (n 0.) in
  let pick4 v4 v3 v2 v1 =
    where b4 (n v4) (where b3 (n v3) (where b2 (n v2) (where b1 (n v1) (n 0.))))
  in
  let ch =
    pick4 0x1.921fb54442d18p+0 0x1.f730bd281f69bp-1 0x1.921fb54442d18p-1
      0x1.dac670561bb4fp-2
  in
  let cl =
    pick4 0x1.1a62633145c07p-54 0x1.007887af0cbbdp-56 0x1.1a62633145c07p-55
      0x1.a2b7f222f65e2p-56
  in
  let p = atan_k64 t in
  let z = t *: t in
  let w = mulacc z (z -: n 1.) (n 1.) in
  let hh, hl = two_sum ch t in
  fast_two_sum hh (hl +: (cl +: mulacc tl w p))

let atan x =
  if wide x then nan1 x (copysign (fst (atan_pos (magnitude x) (zero x))) x)
  else
    let a = magnitude x in
    let big = gt a (one x) in
    let p = atan_k32 (where big (one x /: a) a) in
    let y = where big (con x pio2_hi +: (con x pio2_lo -: p)) p in
    let y = copysign y x in
    let y = where (is_nan y) (nan y) y in
    where (is_nan x) x y

(* The angle of (|x|, |y|) from atan of the smaller over the larger; by
   quadrant, added to or subtracted from 0, pi/2 or pi as a pair, then given y's
   sign. *)
let atan2 y x =
  let ax = magnitude x and ay = magnitude y in
  let swap = gt ay ax in
  let num_ = where swap ax ay and den = where swap ay ax in
  let both_inf = eq ax (inf x) &&: eq ay (inf x) in
  let xneg = negative x in
  let base_hi =
    where swap (con x pio2_hi) (where xneg (con x pi_hi) (zero x))
  in
  let base_lo =
    where swap (con x pio2_lo) (where xneg (con x pi_lo) (zero x))
  in
  let minus = ne swap xneg in
  let v =
    if not (wide x) then
      let t = num_ /: den in
      let t = where (eq den (zero x)) (zero x) t in
      let t = where both_inf (one x) t in
      let p = atan_k32 t in
      let p = where minus (negated p) p in
      let hh, hl = two_sum base_hi p in
      hh +: (hl +: base_lo)
    else
      (* 1/den overflows below 2^-1024: scaled, num/den is unchanged *)
      let s = where (lt den (num x 0x1p-960)) (num x 0x1p64) (one x) in
      let num_ = num_ *: s and den = den *: s in
      let rden = one x /: den in
      let t = num_ *: rden in
      let tl = mulacc (negated t) den num_ *: rden in
      let degenerate = eq den (zero x) ||: both_inf ||: eq den (inf x) in
      let t = where (eq den (zero x)) (zero x) t in
      let t = where both_inf (one x) t in
      let tl = where degenerate (zero x) tl in
      let ph, pl = atan_pos t tl in
      let ph = where minus (negated ph) ph
      and pl = where minus (negated pl) pl in
      let hh, hl = two_sum base_hi ph in
      hh +: (hl +: (base_lo +: pl))
  in
  nan2 y x (copysign v y)

let erf32 x =
  let a = magnitude x in
  let z = a *: a in
  let e =
    horner z
      [
        -0x1.252b1cp-17;
        0x1.cb3dccp-14;
        -0x1.ba9afap-11;
        0x1.55fc8cp-8;
        -0x1.b8260ep-6;
        0x1.ce2ee6p-4;
        -0x1.812746p-2;
      ]
  in
  let tail =
    where
      (lt a (num x 0x1p-100))
      (zero x)
      (a *: mulacc e z (num x (-0x1.f7ac92p-25)))
  in
  let small = mulacc a (num x 0x1.20dd76p+0) tail in
  let four = num x 4. in
  let u = where (lt four a) four a -: num x 2.5 in
  let q =
    horner u
      [
        -0x1.e11c0ep-21;
        -0x1.11733cp-21;
        0x1.088b0ep-16;
        -0x1.28a63ap-16;
        -0x1.54d0a6p-14;
        0x1.1b8868p-12;
        -0x1.0f0962p-12;
        -0x1.25a286p-11;
        0x1.6ea774p-9;
        -0x1.906118p-8;
        0x1.1a8c54p-7;
        -0x1.11ab44p-7;
        0x1.64e34ap-8;
        -0x1.1d7fe4p-9;
        0x1.aab886p-12;
      ]
  in
  let r = copysign (where (lt a (one x)) small (one x -: q)) x in
  where (is_nan x) x r

let erf_near64 x =
  let a = magnitude x in
  let z = a *: a in
  let e =
    horner z
      [
        0x1.3bd95b2fcc4c8p-42;
        -0x1.b3fe40ecb1d83p-38;
        0x1.9a0cfad36fb97p-34;
        -0x1.517c99ae9a999p-30;
        0x1.fcbb28cc88fd2p-27;
        -0x1.5f73c3747d06bp-23;
        0x1.b9e6c38c36437p-20;
        -0x1.f4d25bf9c9292p-17;
        0x1.f9a326f7afd31p-14;
        -0x1.c02db400361ccp-11;
        0x1.565bcd0e6a301p-8;
        -0x1.b82ce31288b48p-6;
        0x1.ce2f21a042be2p-4;
        -0x1.812746b0379e7p-2;
      ]
  in
  let small =
    mulacc a
      (num x 0x1.20dd750429b6dp+0)
      (a *: mulacc e z (num x 0x1.1ae3a914fed8p-56))
  in
  let u = a -: num x 1.75 in
  let q =
    horner u
      [
        0x1.1952d5a4cae0ep-27;
        -0x1.28e112f89c851p-26;
        -0x1.4fdfcb701a293p-24;
        0x1.5c6e3c15a59e3p-22;
        0x1.7545709555e68p-23;
        -0x1.b74c8ca4aa58fp-19;
        0x1.367dce5756fd5p-18;
        0x1.294cd0fe59fcfp-16;
        -0x1.20080dba8f773p-14;
        0x1.6d715d76f41ap-18;
        0x1.d041f3b63da76p-12;
        -0x1.d4cf0b04a217cp-11;
        -0x1.7ec1d228fe1f5p-11;
        0x1.8d10fb6c23dd8p-8;
        -0x1.25b378c8cdfffp-7;
        -0x1.b22257dd0a3ddp-8;
        0x1.8a0da543053a6p-5;
        -0x1.7148c3d57c2d8p-4;
        0x1.7a4a8a2bdce2cp-4;
        -0x1.b055303221016p-5;
        0x1.b4be201caa4b3p-7;
      ]
  in
  let r = copysign (where (lt a (one x)) small (one x -: q)) x in
  where (is_nan x) x r

(* Past 2.5: erf a = 1 - exp(G(a) - a^2), a taken as 5.95 past it. *)
let erf_far64 x =
  let a = magnitude x in
  let cap = num x 5.95 in
  let b = where (lt cap a) cap a in
  let u = b -: num x 4.25 in
  let g =
    horner u
      [
        0x1.7742963aff69cp-27;
        0x1.54fd3a02fb6fcp-30;
        0x1.52f34766557c1p-22;
        -0x1.8a631a0b82da7p-20;
        0x1.71c1a68f670c6p-17;
        -0x1.2f3790e7befacp-14;
        0x1.f7e642e7cc5b2p-12;
        -0x1.abc91bf263532p-9;
        0x1.87d48af31b6e3p-6;
        -0x1.ca4aec3d2de83p-3;
        -0x1.05cb67754a622p+1;
      ]
  in
  copysign (one x -: exp_near (g -: (b *: b))) x

let erf x =
  if not (wide x) then erf32 x
  else where (gt (magnitude x) (num x 2.5)) (erf_far64 x) (erf_near64 x)

(* fmod, exact. With |x| = mx 2^ex and |y| = my 2^ey, mx and my integers of the
   significand's bits, |x| mod |y| is (mx 2^(ex - ey) mod my) 2^ey: the
   remainder of mx by my, shifted left by the exponents' difference a chunk at a
   time and reduced again, each shift small enough that the shifted remainder
   holds in 64 bits. Any exact algorithm gives the bits nx's long division
   gives. *)
let fmod x y =
  let ux = band (bits x) (mag_word x) and uy = band (bits y) (mag_word y) in
  let ib = word ux (pick x inf_bits) in
  let special =
    eq uy (word uy 0L) ||: Ops.logical_not (lt ux ib) ||: gt uy ib
  in
  let mb = pick x mbits and b = pick x bias in
  let field u = Ops.cast (shr u mb) Int32 in
  let normal e = where (lt e (int e 1)) (int e 1) e in
  let sig_ u =
    let f = band u (word u (pick x mmask)) in
    let lead = word u (Int64.shift_left 1L mb) in
    Ops.cast (where (eq (field u) (int (field u) 0)) f (bor f lead)) Uint64
  in
  let ex = normal (field ux) and ey = normal (field uy) in
  let mx = sig_ ux and my = sig_ uy in
  let my = where (eq my (u64 my 0L)) (u64 my 1L) my in
  let d = Ops.sub ex ey in
  let chunk = 63 - mb in
  let steps = ((2 * b) + chunk - 1) / chunk in
  let modu r = Ops.alu r Op.Cmod [ my ] in
  let r = ref (modu mx) in
  for k = 0 to steps - 1 do
    let left = Ops.sub d (int d (k * chunk)) in
    let left = where (lt left (int d 0)) (int d 0) left in
    let left = where (gt left (int d chunk)) (int d chunk) left in
    r := modu (Ops.shl !r (Ops.cast left Uint64))
  done;
  (* r 2^(ey - bias - mbits), in two exact products when the scale is below the
     normal range *)
  let e = Ops.sub ey (int ey (b + mb)) in
  let e1 = where (lt e (int e (1 - b))) (int e (1 - b)) e in
  let m = Ops.cast !r (Ops.dtype x) *: pow2 x e1 *: pow2 x (Ops.sub e e1) in
  let sx = band (bits x) (sign_word x) in
  let r = where (lt ux uy) x (of_bits (bor (bits m) sx)) in
  where special (where (is_nan x) x (where (is_nan y) y (nan x))) r

(* Kinds *)

let unary (k : Nx_kernel.Prog.unary) x =
  match k with
  | Neg -> negated x
  | Recip -> nan1 x (one x /: x)
  | Abs -> magnitude x
  | Sign -> sign x
  | Sqrt -> nan1 x (Ops.sqrt x)
  | Exp -> exp x
  | Exp2 -> exp2 x
  | Log -> log x
  | Log2 -> log2 x
  | Log1p -> log1p x
  | Expm1 -> expm1 x
  | Sin -> sin x
  | Cos -> cos x
  | Tan -> tan x
  | Asin -> asin x
  | Acos -> acos x
  | Atan -> atan x
  | Sinh -> sinh x
  | Cosh -> cosh x
  | Tanh -> tanh x
  | Erf -> erf x
  | Floor -> floor x
  | Ceil -> ceil x
  | Round -> round x
  | Trunc -> trunc x
  | Asinh -> invalid_arg "Lower_real.unary: floats do not take Asinh"
  | Acosh -> invalid_arg "Lower_real.unary: floats do not take Acosh"
  | Atanh -> invalid_arg "Lower_real.unary: floats do not take Atanh"

let binary (k : Nx_kernel.Prog.binary) x y =
  match k with
  | Add -> nan2 x y (x +: y)
  | Sub -> nan2 x y (x -: y)
  | Mul -> nan2 x y (x *: y)
  | Fdiv -> nan2 x y (x /: y)
  | Mod -> fmod x y
  | Pow -> pow x y
  | Atan2 -> atan2 x y
  | Maximum -> maximum x y
  | Minimum -> minimum x y
  | Idiv | And | Or | Xor | Threefry ->
      invalid_arg "Lower_real.binary: a kind floats do not take"

let compare (k : Nx_kernel.Prog.compare) x y =
  match k with
  | Equal -> eq x y
  | Not_equal -> ne x y
  | Less -> lt x y
  | Less_equal -> le x y

let fma a b c =
  let r = nan1 c (mulacc a b c) in
  where (is_nan a) a (where (is_nan b) b r)

(* Parts of the complex kinds *)

let exp_q r = poly r exp_q
let exp2_poly r = poly r exp2_p
