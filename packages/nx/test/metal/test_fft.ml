(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.metal's Fourier transforms without the GPU: their tables against
   references that share none of the tables' method. Roots are checked against
   the host's sin and cos where the double's error leaves a float32 rounding no
   doubt, against identities of roots in double-double, and at the angles whose
   values are known; rounding against constructed halfway pairs; the kernels
   against direct double-double sums, each term a product of two roots, at every
   bin or a large kernel's sampled ones, before and after rounding, at
   tolerances the stated errors imply. Steps against the rules fft.mli states,
   at every 7-smooth length to 4096 and at the values the design states; long
   lines' factors against the q bound; how a line runs, Bluestein's m against
   every length it may take. *)

open Windtrap
module T = Nx_metal.Fft_tables

(* Double-double arithmetic of the test's own: exact products by fma. *)

let two_sum a b =
  let s = a +. b in
  let bb = s -. a in
  (s, a -. (s -. bb) +. (b -. bb))

let fast_two_sum a b =
  let s = a +. b in
  (s, b -. (s -. a))

(* Both parts' sums, renormalised twice: within about 2^-104 of the sum. *)
let dd_add (a : T.dd) (b : T.dd) : T.dd =
  let s, e = two_sum a.hi b.hi in
  let t, f = two_sum a.lo b.lo in
  let s, e = fast_two_sum s (e +. t) in
  let hi, lo = fast_two_sum s (e +. f) in
  { hi; lo }

let dd_neg (a : T.dd) : T.dd = { hi = -.a.hi; lo = -.a.lo }

let dd_mul (a : T.dd) (b : T.dd) : T.dd =
  let p = a.hi *. b.hi in
  let e = Float.fma a.hi b.hi (-.p) in
  let e = Float.fma a.hi b.lo (Float.fma a.lo b.hi e) in
  let hi = p +. e in
  { hi; lo = e -. (hi -. p) }

let cmul (a : T.cdd) (b : T.cdd) : T.cdd =
  {
    re = dd_add (dd_mul a.re b.re) (dd_neg (dd_mul a.im b.im));
    im = dd_add (dd_mul a.re b.im) (dd_mul a.im b.re);
  }

let cadd (a : T.cdd) (b : T.cdd) : T.cdd =
  { re = dd_add a.re b.re; im = dd_add a.im b.im }

(* [a - b] of two close double-doubles, in a float. *)
let dd_diff (a : T.dd) (b : T.dd) = a.hi -. b.hi +. (a.lo -. b.lo)

(* Floats *)

let f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let least_normal = Float.ldexp 1. (-126)

(* [x] rounded to the float32 the GPU reads: a subnormal as a zero. *)
let gpu32 x =
  let f = f32 x in
  if Float.abs f < least_normal then Float.copy_sign 0. f else f

(* The float32 every value in [lo, hi] rounds to, if one. *)
let settled lo hi =
  let a = gpu32 (Float.pred lo) and b = gpu32 (Float.succ hi) in
  if Float.equal a b then Some a else None

let bits32 x = Int32.float_of_bits x

(* Number [i] of table [b]: its real and imaginary parts. *)
let entry (b : T.bigbytes) i =
  let word at =
    let byte k = Bigarray.Array1.get b (at + k) in
    Int32.(
      logor
        (of_int (byte 0 lor (byte 1 lsl 8) lor (byte 2 lsl 16)))
        (shift_left (of_int (byte 3)) 24))
  in
  (bits32 (word (8 * i)), bits32 (word ((8 * i) + 4)))

let int32_at (b : T.bigbytes) i =
  let byte k = Bigarray.Array1.get b ((4 * i) + k) in
  let v = byte 0 lor (byte 1 lsl 8) lor (byte 2 lsl 16) lor (byte 3 lsl 24) in
  if v >= 1 lsl 31 then v - (1 lsl 32) else v

let entries b = Bigarray.Array1.dim b / 8

(* Roots against the host's sin and cos *)

(* The double the host gives for cos and -sin of 2π k / l, and a bound on its
   error: the angle's three roundings, 3 · 2^-53 of 2π, and 4 ulps of the
   result. *)
let libm l k =
  let t = 2. *. Float.pi *. (Float.of_int k /. Float.of_int l) in
  let slack v =
    (Float.ldexp 3. (-53) *. 2. *. Float.pi)
    +. (4. *. Float.epsilon *. Float.abs v)
  in
  let c = cos t and s = -.sin t in
  ((c, slack c), (s, slack s))

(* [Some f] where [v ± e] leaves the float32 rounding no doubt. *)
let decided (v, e) = settled (v -. e) (v +. e)

type rcase = { l : int; start : int; step : int; count : int }

let pp_rcase ppf c =
  Format.fprintf ppf "roots %d ~start:%d ~step:%d ~count:%d" c.l c.start c.step
    c.count

(* Lengths up to 2^33, a quarter of them past 2^31, and starts at an octant's
   ends and next to them, at zero and a quarter turn, or anywhere. *)
let rcase =
  let open Gen in
  with_pp pp_rcase
    (let* bits = frequency [ (3, int_range 0 30); (1, int_range 31 33) ] in
     let* l = int_range (1 lsl bits) ((1 lsl bits) + (1 lsl bits) - 1) in
     let l = Int.min l (1 lsl 33) in
     let* where = int_range 0 3 in
     let* octant = int_range 0 7 in
     let* near = int_range (-1) 1 in
     let* any = int_range 0 (l - 1) in
     let start =
       match where with
       | 0 -> any
       | 1 -> Int.max 0 ((octant * l / 8) + near)
       | 2 -> if l mod 4 = 0 then l / 4 else 0
       | _ -> 0
     in
     let* step = int_range 0 (Int.min l 1_000_000) in
     let+ count = int_range 1 8 in
     { l; start; step; count })

let law_libm c =
  let b = T.roots c.l ~start:c.start ~step:c.step ~count:c.count in
  equal int c.count (entries b);
  let r = 8 * c.start mod c.l in
  cover "an octant's end or next to it" (r < 9 || r > c.l - 9);
  cover "a quarter turn" (c.l mod 4 = 0 && 4 * c.start = c.l);
  cover "past 2^32 points" (c.l > 1 lsl 32);
  let checked = ref 0 in
  for i = 0 to c.count - 1 do
    let k = (c.start + (i * c.step)) mod c.l in
    let re, im = entry b i in
    let lre, lim = libm c.l k in
    (match decided lre with
    | Some f ->
        incr checked;
        equal ~msg:(Printf.sprintf "re of w_%d^%d" c.l k) float_exact f re
    | None -> ());
    match decided lim with
    | Some f ->
        incr checked;
        equal ~msg:(Printf.sprintf "im of w_%d^%d" c.l k) float_exact f im
    | None -> ()
  done;
  cover ~share:0.9 "a part the host decides" (!checked > 0)

(* Roots whose values are known *)

type known = { name : string; l : int; k : int; re : int32; im : int32 }

let sqrt_half = 0x3F3504F3l
let sqrt3_half = 0x3F5DB3D7l
let one = 0x3F800000l
let half = 0x3F000000l
let neg x = Int32.logor x Int32.min_int

let known =
  [
    { name = "w_7^0"; l = 7; k = 0; re = one; im = 0l };
    { name = "w_8^2"; l = 8; k = 2; re = 0l; im = neg one };
    { name = "w_8^4"; l = 8; k = 4; re = neg one; im = 0l };
    { name = "w_8^6"; l = 8; k = 6; re = 0l; im = one };
    { name = "w_8^1"; l = 8; k = 1; re = sqrt_half; im = neg sqrt_half };
    { name = "w_8^3"; l = 8; k = 3; re = neg sqrt_half; im = neg sqrt_half };
    { name = "w_8^5"; l = 8; k = 5; re = neg sqrt_half; im = sqrt_half };
    { name = "w_8^7"; l = 8; k = 7; re = sqrt_half; im = sqrt_half };
    { name = "w_12^1"; l = 12; k = 1; re = sqrt3_half; im = neg half };
    { name = "w_12^2"; l = 12; k = 2; re = half; im = neg sqrt3_half };
    { name = "w_12^5"; l = 12; k = 5; re = neg sqrt3_half; im = neg half };
    { name = "w_2^32^2^30"; l = 1 lsl 32; k = 1 lsl 30; re = 0l; im = neg one };
    {
      name = "w_3·2^30^(3·2^27)";
      l = 3 lsl 30;
      k = 3 lsl 27;
      re = sqrt_half;
      im = neg sqrt_half;
    };
  ]

let test_known c =
  let re, im = entry (T.roots c.l ~start:c.k ~step:1 ~count:1) 0 in
  equal ~msg:"re" float_exact (bits32 c.re) re;
  equal ~msg:"im" float_exact (bits32 c.im) im

(* Identities of roots in double-double *)

type icase = { l : int; a : int; b : int }

let pp_icase ppf c = Format.fprintf ppf "w_%d^%d, w_%d^%d" c.l c.a c.l c.b

let icase =
  let open Gen in
  with_pp pp_icase
    (let* bits = int_range 0 34 in
     let* l = int_range (1 lsl bits) ((1 lsl (bits + 1)) - 1) in
     let* a = int_range 0 (l - 1) in
     let+ b = int_range 0 (l - 1) in
     { l; a; b })

(* The .mli's claim: a root within 2^-98 of its exact value. *)
let root_err = Float.ldexp 1. (-98)

(* w = w₀ + δ, |w₀| = 1, |δ| <= 2^-98: ||w|² - 1| <= 2 2^-98 + |δ|², and the
   test's two squares and sum add under 2^-101. *)
let law_unit c =
  let w = T.root c.l c.a in
  let sq (x : T.dd) = dd_mul x x in
  let n = dd_add (sq w.re) (sq w.im) in
  at_most float_exact
    ~than:((2. *. root_err) +. Float.ldexp 1. (-101))
    (Float.abs (dd_diff n { hi = 1.; lo = 0. }))

(* Three roots each within 2^-98: a part of w^a w^b - w^(a+b) within 3 2^-98 +
   2^-196, and the test's product adds under 2^-101. *)
let law_product c =
  let w = cmul (T.root c.l c.a) (T.root c.l c.b) in
  let s = T.root c.l ((c.a + c.b) mod c.l) in
  let bound = (3. *. root_err) +. Float.ldexp 1. (-101) in
  cover "past a whole turn" (c.a + c.b >= c.l);
  at_most ~msg:"re" float_exact ~than:bound (Float.abs (dd_diff w.re s.re));
  at_most ~msg:"im" float_exact ~than:bound (Float.abs (dd_diff w.im s.im))

(* Rounding *)

type parity = Even | Odd
type pair = { below : float; lo : float; parity : parity }

let pp_pair ppf c = Format.fprintf ppf "halfway above %h, lo %h" c.below c.lo

(* A float32 [below] of either parity and the halfway point to the next toward
   +∞, with a [lo] of each sign or zero, within the halfway double's half ulp or
   past it. *)
let pair =
  let open Gen in
  with_pp pp_pair
    (let* bits = int32_range 0x00000001l 0x7F7FFFFEl in
     let* sign = bool in
     let* lo_sign = int_range (-1) 1 in
     let+ tiny = int_range 1 60 in
     let below = bits32 (if sign then neg bits else bits) in
     let parity = if Int32.logand bits 1l = 0l then Even else Odd in
     let lo =
       Float.of_int lo_sign *. Float.ldexp (Float.abs below) (-30 - tiny)
     in
     { below; lo; parity })

let next32 x =
  let b = Int32.bits_of_float x in
  bits32 (if x >= 0. then Int32.succ b else Int32.pred b)

let law_rounding c =
  let above = next32 c.below in
  let h = (c.below +. above) /. 2. in
  let expected =
    if c.lo = 0. then match c.parity with Even -> c.below | Odd -> above
    else if c.lo > 0. then above
    else c.below
  in
  let expected =
    if Float.abs expected < least_normal then Float.copy_sign 0. expected
    else expected
  in
  cover "lo toward the next" (c.lo > 0.);
  cover "lo away from the next" (c.lo < 0.);
  cover "lo within the halfway double's half ulp"
    (c.lo <> 0. && Float.abs c.lo <= Float.abs (Float.succ h -. h) /. 2.);
  cover "lo zero, below even" (c.lo = 0. && c.parity = Even);
  cover "lo zero, below odd" (c.lo = 0. && c.parity = Odd);
  equal float_exact expected (T.round { hi = h; lo = c.lo })

(* Pairs whose [hi] is not the double nearest [hi + lo], values that round below
   the least normal float32, and the tie between the largest subnormal,
   0x007FFFFF, and the least normal, where the halfway rule meets the flush. *)
let rounding_cases =
  let u = Float.ldexp 1. (-24) in
  let h = 1. +. u in
  let tie = least_normal -. Float.ldexp 1. (-150) in
  let tiny = Float.ldexp 1. (-200) in
  [
    ("hi halfway, lo past an ulp below", h, -.Float.ldexp 1. (-52), 1.);
    ( "hi past halfway, lo back below it",
      Float.succ h,
      -.Float.ldexp 1. (-51),
      1. );
    ( "hi below halfway, lo past it",
      Float.pred h,
      Float.ldexp 1. (-51),
      1. +. (2. *. u) );
    ("1 - 2^-25, lo zero, to the even 1", 1. -. Float.ldexp 1. (-25), 0., 1.);
    ( "1 - 2^-25, lo below",
      1. -. Float.ldexp 1. (-25),
      -.Float.ldexp 1. (-80),
      1. -. u );
    ("a subnormal", Float.ldexp 1. (-130), 0., 0.);
    ("a negative subnormal", -.Float.ldexp 1. (-130), 0., -0.);
    ("the least normal", least_normal, 0., least_normal);
    ( "just below the least normal, to it",
      least_normal -. Float.ldexp 1. (-151),
      0.,
      least_normal );
    ( "the subnormal tie, lo zero, to the even least normal",
      tie,
      0.,
      least_normal );
    ( "the subnormal tie, lo away from zero, to the least normal",
      tie,
      tiny,
      least_normal );
    ("the subnormal tie, lo toward zero, to a zero", tie, -.tiny, 0.);
    ("the negative subnormal tie, lo toward zero, to a zero", -.tie, tiny, -0.);
    ( "the negative subnormal tie, lo away from zero, to the least normal",
      -.tie,
      -.tiny,
      -.least_normal );
    ("zero", 0., 0., 0.);
    ("negative zero", -0., 0., -0.);
  ]

let test_rounding (_, hi, lo, expected) =
  equal float_exact expected (T.round { hi; lo })

(* Lengths *)

let smooth n =
  let rec strip n p = if n mod p = 0 then strip (n / p) p else n in
  n >= 1 && strip (strip (strip (strip n 2) 3) 5) 7 = 1

let is_prime n =
  let rec from d = d * d > n || (n mod d <> 0 && from (d + 1)) in
  n >= 2 && from 2

let raders =
  List.filter (fun n -> is_prime n && smooth (n - 1)) (List.init 4098 Fun.id)

(* The least 7-smooth length at or past [lo]. *)
let rec smooth_from lo = if smooth lo then lo else smooth_from (lo + 1)

(* Global tables *)

(* The first number at which [a] and [b] differ in a bit, or -1. *)
let bits (re, im) = (Int32.bits_of_float re, Int32.bits_of_float im)

let first_other a b =
  let rec from i =
    if i >= Int.min (entries a) (entries b) then -1
    else if bits (entry a i) <> bits (entry b i) then i
    else from (i + 1)
  in
  from 0

let test_global () =
  equal int 10 T.low_bits;
  List.iter
    (fun l ->
      let lo, hi = T.global l in
      let nh = (l + 1023) / 1024 in
      equal ~msg:"lo" int 1024 (entries lo);
      equal ~msg:"hi" int nh (entries hi);
      let lo' = T.roots l ~start:0 ~step:1 ~count:1024 in
      let hi' = T.roots l ~start:0 ~step:1024 ~count:nh in
      equal ~msg:"lo's first other number" int (-1) (first_other lo lo');
      equal ~msg:"hi's first other number" int (-1) (first_other hi hi'))
    [ 4097; 1 lsl 24; 3_828_125 ]

(* The chirp against the host *)

let law_chirp n =
  let b = T.chirp n in
  equal int n (entries b);
  let checked = ref 0 in
  for j = 0 to n - 1 do
    let k = j * j mod (2 * n) in
    let re, im = entry b j in
    let lre, lim = libm (2 * n) k in
    (match decided lre with
    | Some f ->
        incr checked;
        equal ~msg:(Printf.sprintf "re of c_%d" j) float_exact f re
    | None -> ());
    match decided lim with
    | Some f ->
        incr checked;
        equal ~msg:(Printf.sprintf "im of c_%d" j) float_exact f im
    | None -> ()
  done;
  cover "past j² = 2n" (n > 2);
  at_least ~msg:"parts the host decides" int ~than:n !checked

(* Kernels against direct sums *)

let conj (w : T.cdd) : T.cdd = { w with im = dd_neg w.im }

let scale (w : T.cdd) m : T.cdd =
  let d (x : T.dd) : T.dd =
    let q = x.hi /. m in
    let r = Float.fma (-.q) m x.hi in
    let q' = (r +. x.lo) /. m in
    let hi = q +. q' in
    { hi; lo = q' -. (hi -. q) }
  in
  { re = d w.re; im = d w.im }

let rec ceil_log2 ?(d = 0) n =
  if 1 lsl d >= n then d else ceil_log2 ~d:(d + 1) n

let rec pairwise (a : T.cdd array) lo hi =
  if hi - lo = 1 then a.(lo)
  else
    let mid = (lo + hi) / 2 in
    cadd (pairwise a lo mid) (pairwise a mid hi)

(* B_k = (1/m) Σ_j b_j w_m^(jk) at each of [bins], for [b] given at its nonzero
   [js]: each term the product of two roots the test takes one by one, the terms
   summed pairwise. *)
let direct m (js : (int * T.cdd) array) bins =
  let t = Array.init m (fun x -> T.root m x) in
  List.map
    (fun k ->
      let terms = Array.map (fun (j, bj) -> cmul bj t.(j * k mod m)) js in
      (k, scale (pairwise terms 0 (Array.length terms)) (Float.of_int m)))
    bins

(* The .mli's claim: a kernel within (1 + log₂ m) 2^-95 ‖B‖₂ in norm. *)
let kernel_err m = (1. +. Float.log2 (Float.of_int m)) *. Float.ldexp 1. (-95)

(* The direct sum's error in a part of a bin: [terms] terms of modulus one, each
   the product of two roots within 2^-98 and the test's [cmul], so within 2^-96;
   ⌈log₂ terms⌉ levels of additions, each within 2^-103 of sums whose moduli add
   to at most [terms]; and the division by [m]. *)
let ref_err ~terms m =
  let k = Float.of_int terms and levels = Float.of_int (ceil_log2 terms + 1) in
  ((k *. Float.ldexp 1. (-96)) +. (levels *. k *. Float.ldexp 1. (-103)))
  /. Float.of_int m

(* The kernel [got], made from [terms] numbers of modulus one, so of norm ‖B‖₂ =
   √(terms / m) by Parseval, against the direct sums [refs] at some of its bins:
   each part of each bin within the claim and the sum's error, each rounded part
   the float32 that every value within that bound rounds to, and where [refs]
   holds every bin, the whole within the claim in norm. *)
let check_kernel ~name ~terms ~m (got : T.cdd array) (bytes : T.bigbytes) refs =
  equal ~msg:"numbers" int m (Array.length got);
  equal ~msg:"bytes" int m (entries bytes);
  let claim = kernel_err m *. sqrt (Float.of_int terms /. Float.of_int m) in
  let r = ref_err ~terms m in
  let bound = claim +. r in
  let err = ref 0. in
  List.iter
    (fun (k, (w : T.cdd)) ->
      let dr = dd_diff got.(k).re w.re and di = dd_diff got.(k).im w.im in
      err := !err +. (dr *. dr) +. (di *. di);
      let msg part = Printf.sprintf "%s: %s %d" name part k in
      at_most ~msg:(msg "re") float_exact ~than:bound (Float.abs dr);
      at_most ~msg:(msg "im") float_exact ~than:bound (Float.abs di);
      let re, im = entry bytes k in
      let near (x : T.dd) =
        settled (x.hi +. (x.lo -. bound)) (x.hi +. (x.lo +. bound))
      in
      (match near w.re with
      | Some f -> equal ~msg:(msg "rounded re") float_exact f re
      | None -> ());
      match near w.im with
      | Some f -> equal ~msg:(msg "rounded im") float_exact f im
      | None -> ())
    refs;
  if List.length refs = m then
    at_most
      ~msg:(name ^ ": ‖x - y‖₂")
      float_exact
      ~than:(claim +. (sqrt (Float.of_int m) *. r))
      (sqrt !err)

(* Every bin, or a sample of a large kernel's: its ends, its middle and others
   spread by a multiplicative hash. A sampled bin is held only to the norm
   claim, which at m = 65536 is about 2^-83 of a typical entry, so a sample
   catches a wrong index or sign, not a loss of accuracy; the norm check of the
   drawn kernels, every bin, holds that. *)
let every m = List.init m Fun.id

let sample m =
  List.sort_uniq Int.compare
    ([ 0; 1; m / 2; m - 1 ]
    @ List.init 16 (fun i -> (i + 1) * 2654435761 land 0xFFFFFFFF mod m))

(* b_j = conj c_j = w_2n^-(j² mod 2n), at j and m - j. *)
let bluestein_terms ~n ~m =
  let b j = conj (T.root (2 * n) (j * j mod (2 * n))) in
  Array.of_list
    ((0, b 0)
    :: List.concat_map
         (fun j -> [ (j, b j); (m - j, b j) ])
         (List.init (n - 1) succ))

type bcase = { n : int; m : int }

let pp_bcase ppf c = Format.fprintf ppf "bluestein ~n:%d ~m:%d" c.n c.m

let bcase =
  let open Gen in
  with_pp pp_bcase
    (let* n = frequency [ (1, constant 1); (4, int_range 2 200) ] in
     let+ extra = int_range 0 (2 * n) in
     { n; m = smooth_from ((2 * n) - 1 + extra) })

let check_bluestein bins c =
  let js = bluestein_terms ~n:c.n ~m:c.m in
  check_kernel ~name:"bluestein" ~terms:(Array.length js) ~m:c.m
    (T.bluestein_dd ~n:c.n ~m:c.m)
    (T.bluestein ~n:c.n ~m:c.m)
    (direct c.m js (bins c.m))

let law_bluestein c =
  cover "m past 2n - 1" (c.m > (2 * c.n) - 1);
  cover "one point" (c.n = 1);
  (* The kernel's FFT orders its leaves by each prime of [m]. *)
  cover "odd m" (c.m mod 2 = 1);
  cover "a factor of 7" (c.m mod 7 = 0);
  check_bluestein every c

let bluestein_cases =
  [
    { n = 1031; m = 2304 };
    { n = 2048; m = 4096 };
    { n = 1500; m = 3000 };
    { n = 30000; m = 60000 };
    { n = 32768; m = 65536 };
  ]

let rec powmod b e n =
  if e = 0 then 1 mod n
  else
    let h = powmod b (e / 2) n in
    let h = h * h mod n in
    if e mod 2 = 1 then h * b mod n else h

(* The least k >= 1 with a^k = 1 mod n, by counting. *)
let order a n =
  let rec from k p = if p = 1 mod n then k else from (k + 1) (p * a mod n) in
  from 1 (a mod n)

(* Rader's logarithms are a group isomorphism from the units mod n onto the
   integers mod m, from the least primitive root's powers; the kernel is the
   direct sum over them. *)
let check_rader bins n =
  let m = n - 1 in
  let kernel, logs = T.rader n in
  equal ~msg:"logs" int (4 * n) (Bigarray.Array1.dim logs);
  equal ~msg:"log 0" int 0 (int32_at logs 0);
  let seen = Array.make m false in
  for a = 1 to n - 1 do
    let s = int32_at logs a in
    if s < 0 || s >= m || seen.(s) then
      failf "log %d = %d repeats or is out of range" a s;
    seen.(s) <- true
  done;
  let g = ref 0 in
  for a = 1 to n - 1 do
    if int32_at logs a = 1 mod m then g := a
  done;
  let g = !g in
  for a = 2 to g - 1 do
    if order a n = m then failf "%d is a primitive root below g = %d" a g
  done;
  equal ~msg:"g's order" int m (order g n);
  for a = 1 to n - 1 do
    equal
      ~msg:(Printf.sprintf "log g^log %d" a)
      int a
      (powmod g (int32_at logs a) n)
  done;
  let gi = powmod g (m - 1) n in
  let js = Array.init m (fun s -> (s, T.root n (powmod gi s n))) in
  check_kernel ~name:"rader" ~terms:m ~m (T.rader_dd n) kernel
    (direct m js (bins m))

(* Refusals *)

let refused =
  let roots l start = ignore (T.roots l ~start ~step:1 ~count:1) in
  [
    ("roots of no point", fun () -> roots 0 0);
    ("roots past 2^51 points", fun () -> roots ((1 lsl 51) + 1) 0);
    ("roots from a negative start", fun () -> roots 8 (-1));
    ("a chirp of no point", fun () -> ignore (T.chirp 0));
    ("Bluestein's m below 2n - 1", fun () -> ignore (T.bluestein ~n:10 ~m:18));
    ( "Bluestein's m of a prime past 7",
      fun () -> ignore (T.bluestein ~n:5 ~m:11) );
    ("Rader's tables of a composite", fun () -> ignore (T.rader 15));
    ("Rader's tables of 23, 22 = 2 · 11", fun () -> ignore (T.rader 23));
  ]

let test_refused (_, f) = raises_match Exn.invalid_arg f
let small_raders = List.filter (fun n -> n < 400) raders
let large_raders = [ 3889; 4001 ]

(* Steps *)

module F = Nx_metal.Fft

let int_array = array int

(* The radices of a step of [l] points with its 2s [k] at a time: a [2^k] for
   each [k] 2s, one power of two for the 2s left, then 7s, 5s and 3s. *)
let radices_by k l =
  let rec count n p e =
    if n mod p = 0 then count (n / p) p (e + 1) else (n, e)
  in
  let n, twos = count l 2 0 in
  let n, sevens = count n 7 0 in
  let n, fives = count n 5 0 in
  let _, threes = count n 3 0 in
  Array.concat
    [
      Array.make (twos / k) (1 lsl k);
      (if twos mod k > 0 then [| 1 lsl (twos mod k) |] else [||]);
      Array.make sevens 7;
      Array.make fives 5;
      Array.make threes 3;
    ]

let tl_of l radices p =
  Array.fold_left
    (fun t r -> Int.max t (((l / r) + (p / r) - 1) / (p / r)))
    1 radices

(* The .mli's rule: 8s at 8 points a thread where the line then takes at most
   256 threads, else 16s at 16. *)
let rule l =
  let eights = radices_by 3 l in
  if tl_of l eights 8 <= 256 then (eights, 8) else (radices_by 4 l, 16)

let radices l = fst (rule l)
let tl l = tl_of l (radices l) (snd (rule l))

(* Side by side, the same radices at 16 points a thread where 8 leave a group of
   512 threads fewer than 3 lines. *)
let side_p l = if 512 / tl l >= 3 then snd (rule l) else 16
let side_tl l = tl_of l (radices l) (side_p l)
let lengths = List.filter smooth (List.init 4096 succ)
let pow2 l = l land (l - 1) = 0

(* The .mli's rule restated, at every length: it holds the code to the rule, and
   the stated cases below hold the rule to the design. *)
let law_step l =
  let s = F.step l in
  let r, p = rule l in
  equal ~msg:"points" int l s.l;
  equal ~msg:"radices" int_array r s.radices;
  equal ~msg:"points a thread" int p s.p;
  equal ~msg:"every radix holds in its points a thread" bool true
    (Array.for_all (fun r -> r <= p) r);
  equal ~msg:"threads a line" int (tl l) s.tl;
  equal ~msg:"threads" int (Int.max 32 ((tl l + 31) / 32 * 32)) s.threads;
  at_most ~msg:"threads" int ~than:512 s.threads;
  let plane = 8 * if pow2 l && l <= 2048 then l + (l / 16) else l in
  equal ~msg:"plane" int plane (F.plane s);
  let g = F.rows s in
  at_least ~msg:"lines a group" int ~than:1 g;
  at_most ~msg:"their planes" int ~than:F.group_memory (g * plane);
  at_most ~msg:"their threads" int ~than:(Int.max 256 s.tl) (g * s.tl);
  equal ~msg:"no more lines fit" bool true
    ((g + 1) * plane > F.group_memory || (g + 1) * s.tl > 256);
  if l <= F.side_most then begin
    let t = F.side_step l in
    equal ~msg:"side radices" int_array r t.radices;
    equal ~msg:"side points a thread" int (side_p l) t.p;
    equal ~msg:"side threads a line" int (side_tl l) t.tl;
    let g = F.side l in
    equal ~msg:"lines side by side" int (Int.min (8192 / l) (512 / side_tl l)) g;
    at_least ~msg:"lines side by side" int ~than:3 g;
    at_most ~msg:"their float planes" int ~than:F.group_memory (4 * g * l);
    at_most ~msg:"their threads" int ~than:F.side_threads (g * side_tl l)
  end

(* The threads a line the design's table states, and the lines that fill 256
   threads while their planes fit 32 KiB: 64 points take 8 threads a line and
   544 bytes, so 32 lines; 1024 take 128 and 8704, so 2; 1728 take 116 and
   13,824, so 2; a line of 130 threads or more fills a group alone. *)
let step_cases =
  [
    (64, 32, 32);
    (1024, 128, 2);
    (1728, 128, 2);
    (1944, 160, 1);
    (2187, 160, 1);
    (4096, 256, 1);
    (3675, 288, 1);
    (3780, 288, 1);
    (3888, 288, 1);
    (3920, 288, 1);
    (3969, 288, 1);
    (4000, 288, 1);
    (4032, 288, 1);
    (4050, 288, 1);
  ]

(* The radices and threads a line the .mli and the design state. *)
let radix_cases =
  [
    (64, [| 8; 8 |], 8);
    (1000, [| 8; 5; 5; 5 |], 200);
    (1024, [| 8; 8; 8; 2 |], 128);
    (2048, [| 8; 8; 8; 4 |], 256);
    (2304, [| 16; 16; 3; 3 |], 154);
    (4096, [| 16; 16; 16 |], 256);
  ]

let test_radix_case (l, r, tl) =
  let s = F.step l in
  equal ~msg:"radices" int_array r s.radices;
  equal ~msg:"threads a line" int tl s.tl

let test_step_case (l, threads, g) =
  let s = F.step l in
  equal ~msg:"threads" int threads s.threads;
  equal ~msg:"lines a group" int g (F.rows s)

let test_most_threads () =
  let most =
    List.fold_left (fun t l -> Int.max t (F.step l).threads) 0 lengths
  in
  equal int 288 most

let side_cases =
  [
    (512, 8);
    (1000, 7);
    (1024, 4);
    (2048, 4);
    (1890, 3);
    (1944, 3);
    (1960, 3);
    (2000, 3);
    (2016, 3);
    (2025, 3);
  ]

let test_side_case (l, g) = equal int g (F.side l)

(* Long lines *)

(* The least q with 256^q >= n: ⌈log₂ n / 8⌉. *)
let q_bound n =
  let rec from q x = if x >= n then q else from (q + 1) (x * 256) in
  from 1 256

let rec smooth_below st most =
  let rec grow x =
    let p = [| 2; 3; 5; 7 |].(Random.State.int st 4) in
    if x * p > most then x
    else if Random.State.int st 8 = 0 then x * p
    else grow (x * p)
  in
  let x = grow 1 in
  if x > 4096 then x else smooth_below st most

type long = { n : int }

let long =
  let open Gen in
  with_pp
    (fun ppf c -> Format.fprintf ppf "factors %d" c.n)
    (let+ seed = int in
     { n = smooth_below (Random.State.make [| seed |]) (1 lsl 32) })

let law_factors c =
  let steps = F.factors c.n in
  let q = Array.length steps in
  cover "two steps" (q = 2);
  cover "three steps" (q = 3);
  cover "past 2^24 points" (c.n > 1 lsl 24);
  equal ~msg:"points" int c.n
    (Array.fold_left (fun x (s : F.step) -> x * s.l) 1 steps);
  at_least ~msg:"steps" int ~than:2 q;
  at_most ~msg:"steps" int ~than:(q_bound c.n) q;
  Array.iter
    (fun (s : F.step) ->
      at_most ~msg:"a step's points" int ~than:F.most_points s.l;
      equal ~msg:"a step's plan" int_array (F.step s.l).radices s.radices)
    steps

let factor_cases =
  [
    (1 lsl 24, [| 4096; 4096 |]);
    (3_828_125, [| 175; 175; 125 |]);
    (5_764_801, [| 2401; 2401 |]);
  ]

let test_factor_case (n, ls) =
  equal int_array ls (Array.map (fun (s : F.step) -> s.l) (F.factors n))

let test_most_long () =
  equal ~msg:"2^32" int 3 (Array.length (F.factors (1 lsl 32)))

(* Lines *)

(* Bluestein's cost of [m] points: its steps' radices weighed. *)
let weight = function
  | 2 -> 4.
  | 3 -> 7.3
  | 4 -> 7.
  | 5 -> 10.4
  | 7 -> 12.8
  | 8 -> 10.
  | _ -> 13.3

let cost m =
  let steps =
    if m <= 4096 then [| radices m |]
    else Array.map (fun (s : F.step) -> s.radices) (F.factors m)
  in
  Float.of_int m
  *. Array.fold_left
       (fun c r -> Array.fold_left (fun c r -> c +. weight r) c r)
       0. steps

let law_line (n, real) =
  let line = F.line n ~real in
  equal ~msg:"points" int n line.n;
  (match line.how with
  | Smooth ->
      equal ~msg:"7-smooth" bool true (smooth n);
      equal ~msg:"l1" int n line.l1
  | Rader ->
      equal ~msg:"a prime" bool true
        (is_prime n && smooth (n - 1) && n - 1 <= 4096 && not real);
      equal ~msg:"l1" int (n - 1) line.l1
  | Bluestein m ->
      equal ~msg:"neither 7-smooth nor Rader's" bool false
        (smooth n
        || ((not real) && is_prime n && smooth (n - 1) && n - 1 <= 4096));
      equal ~msg:"l1" int m line.l1;
      at_least ~msg:"m" int ~than:((2 * n) - 1) m;
      at_most ~msg:"m" int ~than:(2 * ((2 * n) - 1)) m;
      equal ~msg:"m 7-smooth" bool true (smooth m);
      for m' = (2 * n) - 1 to 2 * ((2 * n) - 1) do
        if smooth m' && m' <> m then
          equal
            ~msg:(Printf.sprintf "cheaper than %d" m')
            bool true
            (cost m < cost m' || (cost m = cost m' && m < m'))
      done);
  let ls = Array.map (fun (s : F.step) -> s.l) line.steps in
  if line.l1 <= 4096 then equal ~msg:"one step" int_array [| line.l1 |] ls
  else
    equal ~msg:"the long line's steps" int_array
      (Array.map (fun (s : F.step) -> s.l) (F.factors line.l1))
      ls

(* Lines of every length class. Rader's lines are one case in 12, so the law
   also runs 257 points, as it runs 8192 for a long 7-smooth line. *)
let line_case =
  let open Gen in
  with_pp
    (fun ppf (n, real) -> Format.fprintf ppf "line %d ~real:%b" n real)
    (let* real = bool in
     let+ n =
       frequency
         [
           (3, int_range 1 5000);
           (1, of_list raders);
           (1, int_range 5000 3_000_000);
           (* 7-smooth lengths past a group, which a range almost never
              draws. *)
           ( 1,
             map
               (fun (a, b) -> a * b)
               (pair (of_list lengths) (of_list lengths)) );
         ]
     in
     (n, real))

let law_line_case ((n, real) as c) =
  let line = F.line n ~real in
  cover "Rader's" (line.how = Rader);
  cover "Bluestein's in a group"
    (match line.how with Bluestein m -> m <= 4096 | _ -> false);
  cover "Bluestein's long"
    (match line.how with Bluestein m -> m > 4096 | _ -> false);
  cover "7-smooth long" (line.how = Smooth && n > 4096);
  law_line c

(* The flushing term's factor, [c = √(2n) Σ_ℓ f_ℓ ‖R_ℓ‖ / n] over the layers of
   the 4096-point step, is the one Fft states, rounded up within 0.01: its
   loads, then each pass of 16, past the first a product by its twiddles, two
   levels of sums, a product by roots of 16 and two levels of sums. A layer's
   part flushes once for a load or a sum, twice for a product; the transform
   after a layer of [s] remaining levels of sums has norm [2^(s/2)]. *)
(* A pass's layers as fft.mli states them: each [(f, ρ)], [f] the flushes a
   part takes and [ρ] the norm of the pass's DFT that remains after it. *)
let dft_layers =
  let s2 = sqrt 2. in
  function
  | 2 -> [ (1., 1.) ]
  | 4 -> [ (1., s2); (1., 1.) ]
  | 8 -> [ (1., 2.); (1., s2); (1., 1.); (1., 1.) ]
  | 16 -> [ (1., sqrt 8.); (1., 2.); (2., 2.); (1., s2); (1., 1.) ]
  | p ->
      (1., sqrt (Float.of_int p))
      :: List.init ((p - 1) / 2) (fun _ -> (1., s2))
      @ [ (1., 1.) ]

(* A pass of radix [r] after [s] points' share of the factor, over [√2]: its
   product by roots past the first pass, then its DFT's layers. *)
let pass_share r s =
  let sf = Float.of_int s and rf = Float.of_int r in
  (if s > 1 then 2. /. sqrt sf else 0.)
  +. List.fold_left
       (fun t (f, rho) -> t +. (f *. rho /. sqrt (sf *. rf)))
       0. (dft_layers r)

(* The flushing term's factor of a line whose passes have [radices], in order:
   [√(2n) Σ_ℓ f_ℓ ‖R_ℓ‖ / n], which is [√2] times the load's 1 and each pass's
   share. A line of no pass is its load and one product. *)
let flush_factor radices =
  if radices = [] then 2. *. sqrt 2.
  else
    let total, _ =
      List.fold_left
        (fun (t, s) r -> (t +. pass_share r s, s * r))
        (1., 1) radices
    in
    sqrt 2. *. total

(* The factor's least upper bound over every sequence of passes, below and
   above: the most shares from [s] points on, each pass's under [5 / √s], and
   [s] at least doubling a pass, so that the passes past [2^40] points add at
   most [5 / √s / (1 - 1/√2)]. *)
let flush_bound () =
  let radices = [ 2; 3; 4; 5; 7; 8; 16 ] in
  let cut = 1 lsl 40 in
  let most ~tail =
    let memo = Hashtbl.create 4096 in
    let rec best s =
      if s >= cut then
        if tail then 5. /. sqrt (Float.of_int s) /. (1. -. (1. /. sqrt 2.))
        else 0.
      else
        match Hashtbl.find_opt memo s with
        | Some v -> v
        | None ->
            let v =
              List.fold_left
                (fun m r -> Float.max m (pass_share r s +. best (s * r)))
                0. radices
            in
            Hashtbl.add memo s v;
            v
    in
    sqrt 2. *. (1. +. best 1)
  in
  (most ~tail:false, most ~tail:true)

(* A halved line's factor, of [n = 2 h] points, from its FFT's [c]: an r2c's
   bins carry the FFT's error times the split's norm, at most [√2], and its
   layers' own, each of [4 (h/2 + 1)] parts at most and of a remaining norm at
   most 1: its sums [E] and [D], its [U] and [V] of [2] flushes and half the
   parts, its last sums and its halves. A c2r's points carry the FFT's error and
   the pre-pass's layers', each times the FFT's norm [√h] and what remains of
   the pre-pass, at most [2] after the bins' loads and [√2] after [E] and [D].
   Each over [n 2^-126]. *)
let r2c_factor ~c h =
  let fh = Float.of_int h and parts = sqrt (Float.of_int ((2 * h) + 4)) in
  ((sqrt 2. *. c *. fh) +. (3. *. parts) +. (2. *. sqrt (fh +. 2.)))
  /. (2. *. fh)

let c2r_factor ~c h =
  let fh = Float.of_int h and parts = sqrt (Float.of_int ((2 * h) + 4)) in
  let loads = 2. *. sqrt (Float.of_int ((2 * h) + 2)) in
  let layers =
    loads +. (sqrt 2. *. parts) +. (2. *. sqrt (fh +. 2.)) +. parts
  in
  ((c *. fh) +. (sqrt fh *. layers)) /. (2. *. fh)

(* The stated factor is the least upper bound over every sequence of passes,
   rounded up, so it holds for every complex line, a long line's steps' passes
   in order included; 2187's steps nearly reach it and 4096's take 7.52. It
   bounds every halved line's too: those in a group from their steps' factor,
   longer ones from the bound. *)
(* A Rader line's factor, of [n] points by its FFT of [m = n - 1] points and
   factor [c], over [n 2^-126]: its loads' layer, of [n] points, carried by the
   DFT, of norm [√n]; the first FFT's error, carried to [y_0] and through the
   kernel and the second FFT, of norm [√m |B|], [|B| <= √n / m] by Gauss's sums;
   [y_0]'s sum; the kernel's product, of [2] flushes a part, carried by the
   second FFT; that FFT's error; and the sums with [x_0]. *)
let rader_factor ~c n =
  let fn = Float.of_int n and fm = Float.of_int (n - 1) in
  (sqrt 2. *. fn)
  +. (c *. fm *. (2. +. sqrt (fn /. fm)))
  +. sqrt 2.
  +. (2. *. sqrt 2. *. fm)
  +. sqrt (2. *. fm)
  |> fun t -> t /. fn

(* A Bluestein line's factor, of [n] points by its FFT of [m] points and factor
   [c], over [n 2^-126]: its loads' layer and its chirp's product, of [1] and
   [2] flushes a part on [n] points, carried by the DFT, of norm [√n]; the first
   FFT's error, carried through the kernel and the second FFT, of norm [√m |B|],
   [|B| <= √((2n - 1) / m)] since [Σ |B_k|² = (2n - 1) / m]; the kernel's
   product, carried by the second FFT; that FFT's error; and the chirp's last
   product, on [n] points. *)
let bluestein_factor ~c n m =
  let fn = Float.of_int n and fm = Float.of_int m in
  (3. *. sqrt 2. *. fn)
  +. (c *. fm *. sqrt ((2. *. fn) -. 1.))
  +. (2. *. sqrt 2. *. fm)
  +. (c *. fm)
  +. (2. *. sqrt (2. *. fn))
  |> fun t -> t /. fn

(* Rader's and Bluestein's lines, complex, odd real or halved, have factors of
   their own: Rader's stated as its class's greatest rounded up, Bluestein's a
   function of [n] at least each line's in a group, and at least the bound of
   its terms past a group, each FFT's factor at most the bound over every
   sequence of passes and [m] at most [4n]. *)
let test_conv_scales () =
  let at l = flush_factor (Array.to_list (F.step l).radices) in
  let _, hi = flush_bound () in
  let rader = ref 0. in
  let bluestein n c =
    at_most
      ~msg:(Printf.sprintf "Bluestein's of %d" n)
      float_exact
      ~than:(F.bluestein_flush_scale n)
      c
  in
  for n = 2 to F.most_points + 1 do
    match (F.line n ~real:false).how with
    | Rader -> rader := Float.max !rader (rader_factor ~c:(at (n - 1)) n)
    | Bluestein m when m <= F.most_points ->
        bluestein n (bluestein_factor ~c:(at m) n m)
    | Smooth | Bluestein _ -> ()
  done;
  for h = 2 to F.most_points / 2 do
    match (F.line h ~real:true).how with
    | Bluestein m when m <= F.most_points ->
        let c = bluestein_factor ~c:(at m) h m in
        bluestein h (r2c_factor ~c h);
        bluestein h (c2r_factor ~c h)
    | Smooth | Rader | Bluestein _ -> ()
  done;
  List.iter
    (fun n ->
      let c = bluestein_factor ~c:hi n (4 * n) in
      bluestein n c;
      bluestein n (r2c_factor ~c n);
      bluestein n (c2r_factor ~c n))
    [ 2049; 4099; 65537; 524_288; F.bluestein_points ];
  equal ~msg:"Rader's" float_exact (Float.ceil !rader) F.rader_flush_scale

let test_flush_scale () =
  let lo, hi = flush_bound () in
  List.iter
    (fun h ->
      let c = flush_factor (Array.to_list (F.step h).radices) in
      at_most ~msg:"an r2c's halved line" float_exact ~than:F.flush_scale
        (r2c_factor ~c h);
      at_most ~msg:"a c2r's halved line" float_exact ~than:F.flush_scale
        (c2r_factor ~c h))
    lengths;
  at_most ~msg:"a long r2c's halved line" float_exact ~than:F.flush_scale
    (r2c_factor ~c:hi 4097);
  at_most ~msg:"a long c2r's halved line" float_exact ~than:F.flush_scale
    (c2r_factor ~c:hi 4097);
  at_least ~msg:"stated against the bound" float_exact ~than:hi F.flush_scale;
  at_most ~msg:"stated against the bound" float_exact ~than:(lo +. 0.01)
    F.flush_scale;
  List.iter
    (fun l ->
      at_most ~msg:"a step's factor" float_exact ~than:lo
        (flush_factor (Array.to_list (F.step l).radices)))
    lengths;
  let at n = flush_factor (Array.to_list (F.step n).radices) in
  at_most ~msg:"4096 points" float_exact ~than:7.52 (at 4096);
  at_most ~msg:"2187 points" float_exact ~than:12.98 (at 2187);
  at_least ~msg:"2187 points" float_exact ~than:12.97 (at 2187)

let test_kept () =
  let first = F.line 1031 ~real:false in
  equal ~msg:"kept" bool true (first == F.line 1031 ~real:false);
  for n = 1 to F.kept_lines do
    ignore (F.line (100_000 + n) ~real:true)
  done;
  equal ~msg:"made again past the kept lines" bool false
    (first == F.line 1031 ~real:false)

(* Calls *)

module L = Nx_array.Layout
module P = Nx_metal.Plan

(* The transforms a call plans. *)
type kind = C2c | R2c | C2r

let kind_name = function C2c -> "c2c" | R2c -> "r2c" | C2r -> "c2r"

type ccase = {
  name : string;
  kind : kind;
  axes : int array;
  x : L.t;
  dst : L.t;
  bases : int * int;
}

let pp_ccase ppf c =
  Format.fprintf ppf "%s: %s ~bases:(%d, %d) ~axes:[%s] %a ~dst:%a" c.name
    (kind_name c.kind) (fst c.bases) (snd c.bases)
    (String.concat "; " (Array.to_list (Array.map string_of_int c.axes)))
    L.pp c.x L.pp c.dst

let plan_of c =
  match c.kind with
  | C2c -> F.c2c ~bases:c.bases ~axes:c.axes c.x ~dst:c.dst
  | R2c -> F.r2c ~bases:c.bases ~axes:c.axes c.x ~dst:c.dst
  | C2r -> F.c2r ~bases:c.bases ~axes:c.axes c.x ~dst:c.dst

(* A layout of [shape] whose axes lie in memory in [order], the innermost last,
   axis [a] [steps.(a)] times its packed stride, flipped where [flips],
   broadcast where [broadcast], from [base]. *)
let laid ?(broadcast = [||]) ?flips ?steps ~order ~base shape =
  let r = Array.length shape in
  let steps = Option.value steps ~default:(Array.make r 1) in
  let flips = Option.value flips ~default:(Array.make r false) in
  let strides = Array.make r 0 in
  let acc = ref 1 in
  for i = r - 1 downto 0 do
    let a = order.(i) in
    strides.(a) <- !acc * steps.(a);
    acc := !acc * steps.(a) * Int.max 1 shape.(a)
  done;
  Array.iter (fun a -> strides.(a) <- 0) broadcast;
  let offset = ref base in
  Array.iteri
    (fun a f ->
      if f && shape.(a) > 1 then begin
        offset := !offset + ((shape.(a) - 1) * strides.(a));
        strides.(a) <- -strides.(a)
      end)
    flips;
  L.v ~offset:!offset ~strides:(Iarray.of_array strides) (Iarray.of_array shape)

let packed shape =
  laid ~order:(Array.init (Array.length shape) Fun.id) ~base:0 shape

let permutation st r =
  let a = Array.init r Fun.id in
  for i = r - 1 downto 1 do
    let j = Random.State.int st (i + 1) in
    let t = a.(i) in
    a.(i) <- a.(j);
    a.(j) <- t
  done;
  a

let pick st a = a.(Random.State.int st (Array.length a))

(* Lengths by how their lines run: in a group of at most 2048 points, Rader's 17
   and 257 among them, and real lines halved to 17 and 1031, Bluestein's; in a
   group past 2048; long, 7-smooth or Bluestein's, odd or halved. *)
let short_lengths =
  [| 1; 2; 3; 7; 16; 17; 34; 60; 64; 257; 512; 1000; 1024; 1031; 2048 |]

let group_lengths = [| 2187; 2304; 3000; 3001; 4096; 2049; 2062; 8192 |]
let long_lengths = [| 8192 * 2; 6000; 1 lsl 16; 19683; 3_828_125 |]
let bluestein_lengths = [| 4097; 10007; 8194; 20014 |]

let scenarios =
  [|
    "any";
    "rows";
    "side by side";
    "transposed";
    "transposed operand";
    "transposed dst";
    "long";
    "long Bluestein";
    "wide";
    "in place";
    "columns of 4096";
    "no point";
  |]

(* The bins' shape of a real shape along [axis]. *)
let bins shape axis =
  Array.mapi (fun i d -> if i = axis then (d / 2) + 1 else d) shape

(* A case of [kind]: the real shape [shape], whose bins' shape the complex side
   takes, and the layouts [x] and [dst] make of either. *)
let any_most = 1 lsl 24

let ccase_of kind scenario seed =
  let st = Random.State.make [| seed |] in
  let base = pick st [| 0; 0; 3; (1 lsl 33) + 5 |] in
  let batch () = pick st [| 1; 2; 3; 7; 600 |] in
  let any () =
    match Random.State.int st 4 with
    | 0 -> pick st group_lengths
    | 1 -> pick st [| 0; 1; 5 |]
    | _ -> pick st short_lengths
  in
  let make name axes shape x dst =
    let a = axes.(Array.length axes - 1) in
    let xs, ds =
      match kind with
      | C2c -> (shape, shape)
      | R2c -> (shape, bins shape a)
      | C2r -> (bins shape a, shape)
    in
    let bases = pick st [| (0, 0); (0, 0); (120, 0); (8, 120) |] in
    { name; kind; axes; x = x xs; dst = dst ds; bases }
  in
  let laid' ?broadcast ?flips ?steps order base s =
    laid ?broadcast ?flips ?steps ~order ~base s
  in
  match scenarios.(scenario) with
  | "any" ->
      let r = 1 + Random.State.int st 3 in
      let shape = Array.init r (fun _ -> any ()) in
      (* Past [any_most] points, the last extents are drawn again from short
         ones: a plan split under a small budget has a launch for every few
         groups, and a larger call takes seconds to check. *)
      let rec fit i =
        if i > 0 && Array.fold_left ( * ) 1 shape > any_most then begin
          shape.(i) <- pick st [| 1; 2; 3; 7; 16; 17 |];
          fit (i - 1)
        end
      in
      fit (r - 1);
      let order = permutation st r in
      let steps = Array.init r (fun _ -> pick st [| 1; 1; 2; 3 |]) in
      let flips = Array.init r (fun _ -> Random.State.bool st) in
      let broadcast =
        Array.of_list
          (List.filter
             (fun _ -> Random.State.int st 6 = 0)
             (List.init r Fun.id))
      in
      let axes =
        let a =
          List.filter (fun _ -> Random.State.bool st) (List.init r Fun.id)
        in
        Array.of_list (if a = [] then [ Random.State.int st r ] else a)
      in
      let dst =
        if Random.State.int st 4 = 0 then laid' (permutation st r) 0 else packed
      in
      make "any" axes shape (laid' ~broadcast ~flips ~steps order base) dst
  | "rows" ->
      make "rows" [| 1 |] [| batch (); any () |] (laid' [| 0; 1 |] base) packed
  | "side by side" ->
      make "side by side" [| 0 |]
        [| pick st short_lengths; batch () |]
        (laid' [| 0; 1 |] base)
        packed
  | "transposed" ->
      make "transposed" [| 0 |]
        [| pick st group_lengths; pick st [| 2; 7 |] |]
        (laid' [| 0; 1 |] base)
        packed
  | "transposed operand" ->
      make "transposed operand" [| 1 |]
        [| pick st [| 2; 3 |]; any () |]
        (laid' ~steps:[| 1; 2 |] [| 0; 1 |] base)
        packed
  | "transposed dst" ->
      make "transposed dst" [| 1 |]
        [| pick st [| 2; 3 |]; any () |]
        (laid' [| 0; 1 |] base)
        (laid' ~steps:[| 1; 2 |] [| 0; 1 |] 0)
  | "long" | "long Bluestein" ->
      let n =
        pick st (if scenario = 6 then long_lengths else bluestein_lengths)
      in
      let order = permutation st 2 in
      let flips = [| Random.State.bool st; Random.State.bool st |] in
      make scenarios.(scenario) [| 1 |]
        [| pick st [| 1; 3 |]; n |]
        (laid' ~flips order base) packed
  | "wide" ->
      let n = pick st [| 2; 64; 4096 |] and b = pick st [| 1; 3 |] in
      let s0 = ((1 lsl 31) / (b * (n - 1))) + 1 in
      make "wide" [| 0 |] [| n; b |]
        (laid' ~steps:[| s0; 1 |] [| 0; 1 |] base)
        packed
  | "in place" ->
      make "in place" [| 0; 1 |]
        [| any (); any () |]
        (laid' (permutation st 2) base)
        packed
  | "columns of 4096" ->
      (* A c2c along 4096 points of 16 to 40 lines one element apart, a line's
         points [pitch] apart, 0 for packed, on either side of the column step's
         8 KiB, in a batch of one or three runs, the points or the lines
         flipped. *)
      let d = 16 + Random.State.int st 25 in
      let pitch = pick st [| 0; 512; 1024; 1025; 2048; 16384 |] in
      let pitch = if pitch = 0 then d else pitch in
      let runs = pick st [| 1; 1; 3 |] in
      let flip = Random.State.int st 4 in
      let ps = if flip = 1 then -pitch else pitch in
      let inner = if flip = 2 then -1 else 1 in
      let offset =
        (if ps < 0 then 4095 * pitch else 0) + if inner < 0 then d - 1 else 0
      in
      let shape, strides, axis =
        if runs = 1 then ([| 4096; d |], [| ps; inner |], 0)
        else ([| runs; 4096; d |], [| 4096 * pitch; ps; inner |], 1)
      in
      let x =
        L.v ~offset:(base + offset) ~strides:(Iarray.of_array strides)
          (Iarray.of_array shape)
      in
      {
        name = "columns of 4096";
        kind = C2c;
        axes = [| axis |];
        x;
        dst = packed shape;
        bases = pick st [| (0, 0); (120, 0); (8, 120) |];
      }
  | _ ->
      make "no point" [| 1 |] [| batch (); 0 |] (laid' [| 0; 1 |] base) packed

let ccase =
  let open Gen in
  with_pp pp_ccase
    (let* kind = of_list [ C2c; R2c; C2r ] in
     let* scenario = int_range 0 (Array.length scenarios - 1) in
     let+ seed = int in
     ccase_of kind scenario seed)

(* A side's lines and points, by the .mli's rule, from a layout alone. *)
type t_side = {
  t_ps : int;
  t_points : int;
  t_dims : int array;
  t_strides : int array;
}

let batch_of l axis =
  let keep a = Iarray.of_list (List.filteri (fun i _ -> i <> axis) a) in
  L.v ~offset:(L.offset l)
    ~strides:(keep (Iarray.to_list (L.strides l)))
    (keep (Iarray.to_list (L.shape l)))

let t_sides x dst axis =
  let m = L.coalesce [| batch_of x axis; batch_of dst axis |] in
  let side l b =
    {
      t_ps = L.stride l axis;
      t_points = L.dim l axis;
      t_dims = Iarray.to_array (L.shape b);
      t_strides = Iarray.to_array (L.strides b);
    }
  in
  (side x m.(0), side dst m.(1))

let t_wide s =
  s.t_points > 1 && Int.abs s.t_ps * (s.t_points - 1) > (1 lsl 31) - 1

(* A side's access by its strides alone, a batch of one line never columns;
   whether it is wide is apart. *)
let t_access s : F.access =
  let lines = Array.fold_left ( * ) 1 s.t_dims in
  if s.t_points <= 1 || Int.abs s.t_ps = 1 then Rows
  else if lines > 1 && Int.abs s.t_strides.(Array.length s.t_strides - 1) = 1
  then Columns
  else Scattered

let t_of (s : F.side) =
  { t_ps = s.ps; t_points = s.points; t_dims = s.dims; t_strides = s.strides }

let access =
  Testable.structural ~pp:(fun ppf (a : F.access) ->
      Format.pp_print_string ppf
        (match a with
        | Rows -> "rows"
        | Columns -> "columns"
        | Scattered -> "scattered"))

type token =
  | Step of F.path
  | Column of F.part
  | Move of F.input * F.scale * F.output
  | Split
  | Prepass

let pp_token ppf = function
  | Step Contiguous -> Format.pp_print_string ppf "contiguous step"
  | Step Side_by_side -> Format.pp_print_string ppf "side-by-side step"
  | Column First_pass -> Format.pp_print_string ppf "column step's first pass"
  | Column Last_passes -> Format.pp_print_string ppf "column step's last passes"
  | Move (i, sc, o) ->
      Format.fprintf ppf "transpose %s %s %s"
        (match i with Complex -> "complex" | Real -> "real" | Bins -> "bins")
        (match sc with
        | Plain -> "plain"
        | Twiddle -> "twiddle"
        | Chirp_in -> "chirp in"
        | Kernel -> "kernel"
        | Chirp_out -> "chirp out")
        (match o with Complex -> "complex" | Half -> "half" | Real -> "real")
  | Split -> Format.pp_print_string ppf "split"
  | Prepass -> Format.pp_print_string ppf "prepass"

let token = Testable.structural ~pp:pp_token

(* A side moved unchanged, its floats or its complex points. *)
let moved w : token =
  if w = 4 then Move (Real, Plain, Real) else Move (Complex, Plain, Complex)

(* A long line's stage, by 3.2's two levels, its first and last transposes
   [first] and [last]. *)
let long_stage (line : F.line) ~first ~last =
  let q = Array.length line.steps in
  let plain : token = Move (Complex, Plain, Complex) in
  let rec level i =
    if i = q - 1 then [ Step Contiguous ]
    else
      [ plain; Step Contiguous; Move (Complex, Twiddle, Complex) ]
      @ level (i + 1)
      @ [ plain ]
  in
  let tokens = level 0 in
  let t =
    List.length (List.filter (function Move _ -> true | _ -> false) tokens)
  in
  let k = ref 0 in
  List.map
    (function
      | Move _ as m ->
          incr k;
          if !k = 1 then first else if !k = t then last else m
      | x -> x)
    tokens

(* Whether a line of one step may take the column step by its sides, of [sw] and
   [dw] bytes an element: 4096 points, 7-smooth, both sides complex, columns and
   not wide, at least 16 lines one element apart along the batch's innermost
   axis, and at most [most] bytes between a line's consecutive points on either
   side and on the workspace, 8 bytes a line of that axis. *)
let column_open ?(most = 8192) ~sw ~dw (line : F.line) s d =
  let inner t = t.t_dims.(Array.length t.t_dims - 1) in
  let near t = 8 * Int.abs t.t_ps <= most in
  line.n = 4096 && line.how = Smooth && (not line.real) && sw = 8 && dw = 8
  && t_access s = Columns
  && t_access d = Columns
  && (not (t_wide s))
  && (not (t_wide d))
  && inner s >= 16
  && near s && near d
  && 8 * inner s <= most

(* Whether a line would take the column step but for the bytes between its
   points. *)
let column_past ~sw ~dw line s d =
  column_open ~most:max_int ~sw ~dw line s d
  && not (column_open ~sw ~dw line s d)

(* The launches 3.4's rule gives a line by its sides, of [sw] and [dw] bytes an
   element, a long one's first transpose reading [input] and its last writing
   [output]. A wide side's points pass a step's 32-bit offsets, so it moves
   through a transpose. *)
let pattern ?(sw = 8) ?(dw = 8) ?(input : F.input = Complex)
    ?(output : F.output = Complex) (line : F.line) s d =
  let src = t_access s and dst = t_access d in
  let apart s a = a = F.Scattered || t_wide s in
  let moves =
    (if src <> F.Rows then [ moved sw ] else [])
    @ [ Step Contiguous ]
    @ if dst <> F.Rows then [ moved dw ] else []
  in
  if Array.length line.steps = 1 then
    if src = F.Rows && dst = F.Rows then [ [ Step Contiguous ] ]
    else if line.l1 <= 2048 && (not (apart s src)) && not (apart d dst) then
      [ [ Step Side_by_side ]; moves ]
    else if column_open ~sw ~dw line s d then
      [ [ Column First_pass; Column Last_passes ] ]
    else [ moves ]
  else
    [
      (match line.how with
      | Bluestein _ ->
          long_stage line
            ~first:(Move (input, Chirp_in, Complex))
            ~last:(Move (Complex, Kernel, Complex))
          @ long_stage line
              ~first:(Move (Complex, Plain, Complex))
              ~last:(Move (Complex, Chirp_out, output))
      | Smooth | Rader ->
          long_stage line
            ~first:(Move (input, Plain, Complex))
            ~last:(Move (Complex, Plain, output)));
    ]

let token_of (l : F.launch) =
  match l.work with
  | Step s -> Step s.path
  | Column_step { part; _ } -> Column part
  | Transpose { input; scale; output; _ } -> Move (input, scale, output)
  | Split _ -> Split
  | Prepass _ -> Prepass

(* The launches of a call by work: each its launches of ranges of blocks. *)
let works (c : F.call) =
  let close cur acc = match cur with [] -> acc | _ -> List.rev cur :: acc in
  let acc, cur =
    Array.fold_left
      (fun (acc, cur) (l : F.launch) ->
        if l.first_group = 0 then (close cur acc, [ l ]) else (acc, l :: cur))
      ([], []) c.launches
  in
  List.rev (close cur acc)

let count dims = Array.fold_left ( * ) 1 dims

let rec take k = function
  | x :: rest when k > 0 ->
      let a, b = take (k - 1) rest in
      (x :: a, b)
  | l -> ([], l)

(* Line [k] of a side, in C order over its lines, and of a layout's lines along
   [axis]: the positions of their first points. *)
let unravel k dims strides =
  let pos = ref 0 and k = ref k in
  for i = Array.length dims - 1 downto 0 do
    pos := !pos + (!k mod dims.(i) * strides.(i));
    k := !k / dims.(i)
  done;
  !pos

let buffer =
  Testable.structural ~pp:(fun ppf (b : F.buffer) ->
      match b with
      | Operand -> Format.pp_print_string ppf "operand"
      | Dst -> Format.pp_print_string ppf "dst"
      | Workspace at -> Format.fprintf ppf "workspace from %d" at)

let check_side ~msg (s : F.side) ~buffer:b ~width l axis =
  let bt = batch_of l axis in
  let dims = Iarray.to_array (L.shape bt) in
  let strides = Iarray.to_array (L.strides bt) in
  let lines = count dims in
  equal ~msg:(msg ^ ": buffer") buffer b s.buffer;
  equal ~msg:(msg ^ ": width") int width s.width;
  equal ~msg:(msg ^ ": point stride") int (L.stride l axis) s.ps;
  equal ~msg:(msg ^ ": points") int (L.dim l axis) s.points;
  equal ~msg:(msg ^ ": lines") int lines (count s.dims);
  List.iter
    (fun k ->
      let at = s.first + unravel k s.dims s.strides in
      cover "a line past 2^32 elements" (at > 1 lsl 32);
      equal
        ~msg:(Printf.sprintf "%s: line %d" msg k)
        int
        (L.offset l + unravel k dims strides)
        at)
    (List.sort_uniq Int.compare [ 0; 1 mod lines; lines / 2; lines - 1 ])

(* A Split's or a Prepass's blocks: [fft_pair_block] pairs of bins a block, of
   one line. *)
let pair_block = Nx_metal.Kernels.fft_pair_block

let pair_blocks (s : F.side) =
  let h = s.points - 1 in
  count s.dims * (((h / 2) + pair_block) / pair_block)

(* Every launch: its groups, threads and threadgroup memory, its estimate within
   the budget, a step's lines by 3.1, a transpose's width by its sides. *)
let key =
  Testable.structural ~pp:(fun ppf -> function
    | None -> Format.pp_print_string ppf "no key"
    | Some (k, cs) -> Format.pp_print_string ppf (P.label k cs))

(* The key a step of [l] points takes planned alone: a packed line of [l]
   points, which takes the [Contiguous] step. *)
let lone_key l ~lines =
  let x = packed [| lines; l |] in
  match F.c2c ~bases:(0, 0) ~axes:[| 1 |] x ~dst:x with
  | Some { launches = [| l |]; _ } -> F.key l
  | Some _ | None -> fail "a call of a group's lines plans one launch"

(* The first key each step's points, path, lines a group, sides' kinds and
   convolution took. *)
let keys_seen = Hashtbl.create 64

(* A step's key is a function of its points, path, lines a group, what its sides
   hold and its line's convolution alone: the key a call of as many lines of its
   points takes where it is contiguous, complex and of no convolution, and the
   key every other step of its points, path, lines, kinds and convolution takes,
   a long line's included. *)
let check_key (l : F.launch) =
  let k = F.key l in
  match (k, l.work) with
  | None, _ -> ()
  | Some (_, cs), Step { step; path; lines; _ } -> (
      let open Nx_metal.Kernels in
      let kinds =
        ( cs.(fft_fc_halved),
          cs.(fft_fc_input),
          cs.(fft_fc_output),
          cs.(fft_fc_conv) )
      in
      if
        path = Contiguous
        && kinds
           = (fft_halved_none, fft_in_complex, fft_out_complex, fft_conv_none)
      then
        equal ~msg:"the key of its points and lines alone" key
          (lone_key step.l ~lines) k;
      match Hashtbl.find_opt keys_seen (step.l, path, lines, kinds) with
      | Some first -> equal ~msg:"the key of its points and path" key first k
      | None -> Hashtbl.add keys_seen (step.l, path, lines, kinds) k)
  | Some _, _ -> fail "a key off a step"

(* The least power of two at least [n]. *)
let least_pow2 n =
  let rec up p = if p >= n then p else up (2 * p) in
  up 1

let check_launch (l : F.launch) =
  check_key l;
  at_least ~msg:"groups" int ~than:1 l.groups;
  at_most ~msg:"groups" int ~than:P.most_groups l.groups;
  at_most ~msg:"threadgroup memory" int ~than:P.group_memory l.shared;
  equal ~msg:"seconds" float_exact
    (P.seconds ~threads:l.threads ~groups:(l.groups, 1, 1) l.cost)
    l.seconds;
  at_most ~msg:"seconds" float_exact ~than:(P.allowed ()) l.seconds;
  equal ~msg:"src's access" access (t_access (t_of l.src)) l.src.access;
  equal ~msg:"dst's access" access (t_access (t_of l.dst)) l.dst.access;
  match l.work with
  | Step { step; path = Contiguous; lines; _ } ->
      let n = Array.fold_left ( * ) 1 l.src.dims in
      equal ~msg:"lines" int (Int.min (F.rows step) (least_pow2 n)) lines;
      equal ~msg:"threads" int
        (Int.max 32 (((lines * step.tl) + 31) / 32 * 32))
        l.threads;
      equal ~msg:"planes" int (lines * F.plane step) l.shared;
      at_most ~msg:"lines' threads" int ~than:l.threads (lines * step.tl)
  | Step { step; path = Side_by_side; lines; line; _ } ->
      equal ~msg:"side points a thread" int (side_p step.l) step.p;
      let n = Array.fold_left ( * ) 1 l.src.dims in
      equal ~msg:"lines" int (Int.min (F.side step.l) (least_pow2 n)) lines;
      equal ~msg:"threads" int
        (if lines = F.side step.l then F.side_threads
         else Int.max 32 (((lines * step.tl) + 31) / 32 * 32))
        l.threads;
      equal ~msg:"float planes" int (4 * lines * step.l) l.shared;
      at_most ~msg:"lines' threads" int ~than:l.threads (lines * side_tl step.l);
      if line.how = Rader then
        at_most ~msg:"Rader's planes" int ~than:(32 * 1024)
          (lines * 4 * (line.n - 1))
  | Column_step { step; part; _ } ->
      equal ~msg:"threads" int 256 l.threads;
      equal ~msg:"the step's points" int 4096 step.l;
      equal ~msg:"threadgroup memory" int
        (match part with First_pass -> 0 | Last_passes -> 8 * 4096)
        l.shared
  | Transpose { wide; _ } ->
      cover "a wide transpose" wide;
      equal ~msg:"wide" bool (t_wide (t_of l.src) || t_wide (t_of l.dst)) wide
  | Split { wide } | Prepass { wide } ->
      equal ~msg:"wide" bool (t_wide (t_of l.src) || t_wide (t_of l.dst)) wide;
      equal ~msg:"threads" int 256 l.threads;
      equal ~msg:"threadgroup memory" int 0 l.shared

let steps_of (c : F.call) =
  List.filter_map
    (fun (l : F.launch) ->
      match l.work with
      | Step s when l.first_group = 0 -> Some (s.step.l, s.step.radices)
      | Column_step { step; part = First_pass; _ } when l.first_group = 0 ->
          Some (step.l, step.radices)
      | _ -> None)
    (Array.to_list c.launches)

let line_t =
  Testable.structural ~pp:(fun ppf (l : F.line) ->
      Format.fprintf ppf "%d%s by %s, steps of [%s]" l.n
        (if l.real then " halved" else "")
        (match l.how with
        | Smooth -> "its points"
        | Rader -> "Rader's"
        | Bluestein m -> Printf.sprintf "Bluestein's %d" m)
        (String.concat "; "
           (Array.to_list
              (Array.map (fun (s : F.step) -> string_of_int s.l) l.steps))))

(* An axis of a call: the lines it reads and writes, by layout, buffer and
   element bytes, the line, and the launches the .mli gives it. *)
type axis = {
  axis : int;
  src : L.t * F.buffer * int;
  dst : L.t * F.buffer * int;
  line : F.line;
  tokens : token list list;
      (** The paths it may take, the first where both are open: side by side
          takes the transposes where they cost less. *)
}

let complex_axis ~src ~dst axis =
  let n =
    L.dim
      (let l, _, _ = dst in
       l)
      axis
  in
  let line = F.line n ~real:false in
  let (s, _, _), (d, _, _) = (src, dst) in
  let s, d = t_sides s d axis in
  { axis; src; dst; line; tokens = pattern line s d }

(* A real line of [n] points: halved where [n] is even. *)
let real_line n =
  if n mod 2 = 0 then F.line (n / 2) ~real:true else F.line n ~real:false

let is_long (line : F.line) = Array.length line.steps > 1

(* The axes a case runs, in order, and the bytes of its copy. *)
let axes_of c =
  let last = Array.length c.axes - 1 in
  let a = c.axes.(last) in
  let earlier ~src ~dst =
    List.init last (fun k -> c.axes.(last - 1 - k))
    |> List.mapi (fun k axis ->
        complex_axis ~src:(if k = 0 then src else dst) ~dst axis)
  in
  match c.kind with
  | C2c ->
      let dst = (c.dst, F.Dst, 8) in
      (complex_axis ~src:(c.x, F.Operand, 8) ~dst a :: earlier ~src:dst ~dst, 0)
  | R2c ->
      let n = L.dim c.x a in
      let src = (c.x, F.Operand, 4) and dst = (c.dst, F.Dst, 8) in
      if n = 0 then
        ( [
            {
              axis = a;
              src;
              dst;
              line = F.line 1 ~real:false;
              tokens = [ [ Step Contiguous ] ];
            };
          ],
          0 )
      else
        let line = real_line n in
        let s, d = t_sides c.x c.dst a in
        let input, output =
          if line.real then ((Complex : F.input), (Complex : F.output))
          else (Real, Half)
        in
        let tokens =
          List.map
            (fun t -> t @ if is_long line && line.real then [ Split ] else [])
            (pattern ~sw:4 ~input ~output line s d)
        in
        ({ axis = a; src; dst; line; tokens } :: earlier ~src:dst ~dst, 0)
  | C2r ->
      let n = L.dim c.dst a in
      let copy = (packed (Iarray.to_array (L.shape c.x)), F.Workspace 0, 8) in
      let from, bytes =
        if last > 0 then (copy, 8 * L.numel c.x) else ((c.x, F.Operand, 8), 0)
      in
      let line = real_line n in
      let fl, _, _ = from in
      let s, d = t_sides fl c.dst a in
      let input, output =
        if line.real then ((Complex : F.input), (Complex : F.output))
        else (Bins, Real)
      in
      let tokens =
        List.map
          (fun t -> (if is_long line && line.real then [ Prepass ] else []) @ t)
          (pattern ~dw:4 ~input ~output line s d)
      in
      ( earlier ~src:(c.x, F.Operand, 8) ~dst:copy
        @ [ { axis = a; src = from; dst = (c.dst, F.Dst, 4); line; tokens } ],
        bytes )

(* A transpose's sides hold what its record reads and writes: [a b] points of 8
   bytes, of 4 for [Real], [⌊n / 2⌋ + 1] for [Bins] and [Half]; [n] where it
   chirps. *)
let check_record ~msg (line : F.line) (l : F.launch) =
  match l.work with
  | Transpose { split = a, b; input; scale; output; _ } -> (
      let pair = Windtrap.pair int int in
      let n_in = if scale = Chirp_in then line.n else a * b in
      let n_out = if scale = Chirp_out then line.n else a * b in
      let src = l.src and dst = l.dst in
      (match input with
      | Complex ->
          equal ~msg:(msg ^ ": bytes read") int (8 * n_in)
            (src.points * src.width)
      | Real ->
          equal ~msg:(msg ^ ": reals read") pair (n_in, 4)
            (src.points, src.width)
      | Bins ->
          equal ~msg:(msg ^ ": bins read") pair
            ((n_in / 2) + 1, 8)
            (src.points, src.width));
      match output with
      | Complex ->
          equal ~msg:(msg ^ ": bytes written") int (8 * n_out)
            (dst.points * dst.width)
      | Half ->
          equal ~msg:(msg ^ ": bins written") pair
            ((n_out / 2) + 1, 8)
            (dst.points, dst.width)
      | Real ->
          equal ~msg:(msg ^ ": reals written") pair (n_out, 4)
            (dst.points, dst.width))
  | _ -> ()

(* A side's memory: its buffer, its first byte, and its runs, the extents and
   byte strides of its lines and points, extents of 1 dropped and a run that
   continues the next merged with it. *)
let memory ?points (s : F.side) =
  let at, base =
    match s.buffer with
    | Operand -> (0, 0)
    | Dst -> (1, 0)
    | Workspace b -> (2, b)
  in
  let points = Option.value points ~default:s.points in
  let runs =
    List.map2
      (fun e st -> (e, s.width * st))
      (Array.to_list s.dims @ [ points ])
      (Array.to_list s.strides @ [ s.ps ])
    |> List.filter (fun (e, _) -> e <> 1)
  in
  let rec merge = function
    | (e1, s1) :: (e2, s2) :: rest when s1 = e2 * s2 ->
        merge ((e1 * e2, s2) :: rest)
    | r :: rest -> r :: merge rest
    | [] -> []
  in
  (at, base + (s.width * s.first), merge runs)

let memory_t =
  Testable.structural ~pp:(fun ppf (at, first, runs) ->
      Format.fprintf ppf "%s from byte %d, runs [%s]"
        (match at with 0 -> "operand" | 1 -> "dst" | _ -> "workspace")
        first
        (String.concat "; "
           (List.map (fun (e, st) -> Printf.sprintf "%d by %d" e st) runs)))

(* The bytes a side reaches, [lo, hi). *)
let reach (s : F.side) =
  let _, first, runs = memory s in
  List.fold_left
    (fun (lo, hi) (e, st) ->
      let d = (e - 1) * st in
      (lo + Int.min 0 d, hi + Int.max 0 d))
    (first, first + s.width)
    runs

(* Along an axis: each work reads the memory the one before wrote, a split the
   bins its FFT wrote and the one it adds; a transpose's sides lie apart; one
   before a long line's step lays the line out as rows of the step's points. *)
let check_flow ~msg (line : F.line) works =
  List.iter
    (fun w ->
      let l0 : F.launch = List.hd w in
      match l0.work with
      | Transpose _ ->
          let (a, _, _), (b, _, _) = (memory l0.src, memory l0.dst) in
          let (lo1, hi1), (lo2, hi2) = (reach l0.src, reach l0.dst) in
          equal
            ~msg:(msg ^ ": a transpose's sides apart")
            bool false
            (a = b && lo1 < hi2 && lo2 < hi1)
      | _ -> ())
    works;
  let rec pairs = function
    | w1 :: (w2 :: _ as rest) ->
        let prev : F.launch = List.hd w1 and next : F.launch = List.hd w2 in
        let read =
          match next.work with
          | Split _ -> memory ~points:(next.src.points - 1) next.src
          | _ -> memory next.src
        in
        equal
          ~msg:(msg ^ ": read where written")
          memory_t (memory prev.dst) read;
        (match (prev.work, next.work) with
        | Transpose { split = a, _; _ }, Step s when Array.length line.steps > 1
          ->
            equal ~msg:(msg ^ ": rows of the step's points") int s.step.l a
        | _ -> ());
        pairs rest
    | _ -> ()
  in
  pairs works

(* The case of the same extents, C-contiguous. *)
let packed_case c =
  let shape l = packed (Iarray.to_array (L.shape l)) in
  { c with x = shape c.x; dst = shape c.dst }

(* A long line's launches run on its digits. Its points are a mixed-radix number
   of its steps' points, most significant first: input digit [In j] of
   [steps.(j).l] points, so that point [n] is [n_0 s_1 … s_(q-1) + …]. A step
   turns the innermost digit, the next input digit in order, into its output
   digit [Out j]. A transpose [(a, b)] of runs of [a b] points, the innermost
   digits of [a b], moves its leading digits of [a] to the end. After each step
   but the last, a [Twiddle] transpose takes the run whose digits are the input
   digits left and the step's output, [w_(ab)^(rc)] the twiddle of Cooley and
   Tukey's factoring, and no other transpose scales so. A stage ends in natural
   order, its digits [Out (q - 1)], …, [Out 0], bin [k_0 + s_0 k_1 + …];
   Bluestein's first ends at its [Kernel], and its second starts on those points
   as natural input. *)
type digit = In of int | Out of int

let digits_text s d =
  String.concat " "
    (List.map
       (function
         | In j -> Printf.sprintf "n%d/%d" j s.(j)
         | Out j -> Printf.sprintf "k%d/%d" j s.(j))
       d)

(* The innermost digits of [d] whose points multiply to [p], and the rest. *)
let inner s d p =
  let size = function In j | Out j -> s.(j) in
  let rec go acc prod = function
    | rest when prod = p -> Some (List.rev rest, acc)
    | x :: rest when prod < p -> go (x :: acc) (prod * size x) rest
    | _ -> None
  in
  go [] 1 (List.rev d)

(* Each stage's digits where it ends, or where a launch cannot run on them. *)
let run_digits (line : F.line) works =
  let s = Array.map (fun (st : F.step) -> st.l) line.steps in
  let q = Array.length s in
  let fresh = List.init q (fun j -> In j) in
  let natural = List.init q (fun i -> Out (q - 1 - i)) in
  let ends = ref [] and d = ref fresh and next = ref 0 and failed = ref false in
  let fail why =
    ends := Printf.sprintf "%s at %s" why (digits_text s !d) :: !ends;
    failed := true
  in
  let ended () = ends := digits_text s !d :: !ends in
  (* The run a twiddle takes after step [j]. *)
  let twiddled j =
    List.init (q - 1 - j) (fun i -> In (j + 1 + i)) @ [ Out j ]
  in
  let owed = ref None in
  List.iter
    (fun w ->
      let l : F.launch = List.hd w in
      if not !failed then
        match l.work with
        | Transpose { split = a, b; scale; _ } -> (
            match inner s !d (a * b) with
            | None -> fail (Printf.sprintf "no run of %d points" (a * b))
            | Some (outer, run) -> (
                match (inner s run b, scale, !owed) with
                | None, _, _ -> fail (Printf.sprintf "no split (%d, %d)" a b)
                | Some _, Twiddle, Some j when run <> twiddled j ->
                    fail (Printf.sprintf "a twiddle after step %d" j)
                | Some _, Twiddle, None -> fail "a twiddle after no step"
                | Some _, scale, Some j when scale <> Twiddle ->
                    fail (Printf.sprintf "no twiddle after step %d" j)
                | Some (lead, tail), _, _ ->
                    d := outer @ tail @ lead;
                    owed := None;
                    (* Bluestein's first stage ends in its kernel's product: the
                       next starts on the same points as natural input. *)
                    if scale = Kernel then begin
                      ended ();
                      if !d = natural then begin
                        d := fresh;
                        next := 0
                      end
                      else failed := true
                    end))
        | Step st -> (
            match (List.rev !d, !owed) with
            | _, Some j -> fail (Printf.sprintf "no twiddle after step %d" j)
            | In j :: rest, None when j = !next && s.(j) = st.step.l ->
                d := List.rev (Out j :: rest);
                incr next;
                if j < q - 1 then owed := Some j
            | _ -> fail (Printf.sprintf "a step of %d points" st.step.l))
        | Column_step _ -> fail "a column step"
        | Split _ | Prepass _ -> ())
    works;
  if not !failed then ended ();
  List.rev !ends

let stages (line : F.line) =
  match line.how with Bluestein _ -> 2 | Smooth | Rader -> 1

let check_digits ~msg (line : F.line) works =
  let s = Array.map (fun (st : F.step) -> st.l) line.steps in
  let natural =
    digits_text s
      (List.init (Array.length s) (fun i -> Out (Array.length s - 1 - i)))
  in
  equal
    ~msg:(msg ^ ": each stage ends in natural order")
    (list string)
    (List.init (stages line) (fun _ -> natural))
    (run_digits line works)

(* A twiddled transpose [(a, b)] moves each run of [a b] points in tiles of 32
   columns [c < b] by 32 rows [r < a] of the [b] rows of [a] it writes, a run's
   tiles by column, then by row. *)
let run_tiles (a, b) = (a + 31) / 32 * ((b + 31) / 32)

let tile_at (a, b) g =
  let t = g mod run_tiles (a, b) and row_tiles = (a + 31) / 32 in
  let c = t / row_tiles * 32 and r = t mod row_tiles * 32 in
  ((c, Int.min b (c + 32)), (r, Int.min a (r + 32)))

(* A transpose's grid: its runs' [(a, b)], or for a move [(1, n)] its lines by
   their points, one run. *)
let grid_of (l : F.launch) (a, b) =
  if a = 1 then (count l.src.dims, b) else (a, b)

let tiles_of (l : F.launch) ((a, _) as split) =
  let grid = grid_of l split in
  if a = 1 then run_tiles grid else count l.src.dims * run_tiles grid

(* The low table, [low] numbers, read by each launch. *)
let low = 1 lsl Nx_metal.Kernels.fft_low_bits
let low_table = 8 * low

(* The high table's 128-byte lines charged to the 32 lanes at column [c], which
   read its entries [(r c) / low], at most [⌊31 c / low⌋ + 2] of them. *)
let charged_lines c = (((8 * ((31 * c / low) + 2)) + 127) / 128) + 1

(* The lines lanes at rows [r0, r1) of column [c] read, the table from a line's
   first byte. *)
let touched c (r0, r1) =
  (8 * ((r1 - 1) * c / low) / 128) - (8 * (r0 * c / low) / 128) + 1

(* A tile's bytes as charged: its points read and written at 8 bytes, its lanes'
   lines of the high table. *)
let tile_bytes split g =
  let (c0, c1), (r0, r1) = tile_at split g in
  let lines = ref 0 in
  for c = c0 to c1 - 1 do
    lines := !lines + charged_lines c
  done;
  (16 * (c1 - c0) * (r1 - r0)) + (128 * !lines)

let twiddled (c : F.call) =
  List.filter_map
    (fun (l : F.launch) ->
      match l.work with
      | Transpose { scale = Twiddle; split; _ } -> Some (split, l)
      | _ -> None)
    (Array.to_list c.launches)

(* The roots a step of [l] points reads: the first half of an even step's, since
   [w_l^(e + l/2) = -w_l^e], all of an odd one's. *)
let roots l = if l mod 2 = 0 then l / 2 else l

(* The most seconds of one group of a launch's work, from the launch alone: a
   step's, a split's or a pre-pass's groups alike, each reading its step's
   roots; a transpose's points each at its sides' charges, so its fullest tile,
   the first, costs most; a twiddled one's at a full row's tile of its last or
   second-last column, since tiles repeat by run and a column's table lines
   never fall as it grows. *)
let work_group (l : F.launch) =
  let bytes =
    match l.work with
    | Step s ->
        let t = 8 * roots s.step.l in
        ((l.cost.bytes - t) / l.groups) + t
    | Column_step { part = First_pass; _ } -> l.cost.bytes / l.groups
    | Column_step { step; part = Last_passes; _ } ->
        let t = 8 * roots step.l in
        ((l.cost.bytes - t) / l.groups) + t
    | Transpose { scale = Twiddle; split = (a, b) as split; _ } ->
        let rows = (a + 31) / 32 and columns = (b + 31) / 32 in
        List.fold_left
          (fun m column ->
            if column < 0 then m
            else Int.max m (tile_bytes split (column * rows)))
          0
          [ columns - 1; columns - 2 ]
        + low_table
    | Transpose { split; input; scale; output; _ } ->
        let charge (s : F.side) w =
          if s.access = Rows then w else 128 * (w / s.width)
        in
        let in_w = match input with Real -> 4 | Complex | Bins -> 8 in
        let out_w = match output with Real -> 4 | Complex | Half -> 8 in
        let scaled = if scale = Plain then 0 else 8 in
        let ga, gb = grid_of l split in
        (charge l.src in_w + charge l.dst out_w + scaled)
        * Int.min ga 32 * Int.min gb 32
    | Split _ | Prepass _ -> l.cost.bytes / l.groups
  in
  let cost : P.cost =
    { bytes; ops = l.cost.ops / l.groups; chain = l.cost.chain }
  in
  P.seconds ~threads:l.threads ~groups:(1, 1, 1) cost

(* Paths a drawn call takes in under 5% of cases. The full-budget law draws
   enough calls to reach each on every seed; the small-budget law, which draws
   fewer, classifies them, demands the paths it reaches often and the split
   itself, and runs one stated call of each, [split_examples]. *)
let real_axis (e : axis) =
  let _, _, sw = e.src and _, _, dw = e.dst in
  sw = 4 || dw = 4

let is_step = function
  | Step _ -> true
  | Column _ | Move _ | Split | Prepass -> false

(* Each path by its label: whether an axis of a call, with the bytes of the
   call's copy, takes it along [tokens], and the tokens of the path's work. *)
let rare =
  [
    ( "a long line of three steps",
      (fun _ _ (e : axis) _ -> is_long e.line && Array.length e.line.steps = 3),
      is_step );
    ( "a c2r through a copy",
      (fun c copy e _ -> c.kind = C2r && real_axis e && copy > 0),
      is_step );
    ( "a long halved r2c's split",
      (fun _ _ _ tokens -> List.mem Split tokens),
      fun t -> t = Split );
    ( "a long halved c2r's prepass",
      (fun _ _ _ tokens -> List.mem Prepass tokens),
      fun t -> t = Prepass );
    ( "an odd real Rader line",
      (fun _ _ e _ -> real_axis e && e.line.how = Rader),
      is_step );
    ( "a column step",
      (fun _ _ _ tokens -> List.mem (Column First_pass) tokens),
      function Column _ -> true | _ -> false );
  ]

(* A planned call's axes in order, each with its path and its works, the works
   left past them, and the bytes of the call's copy. *)
let by_axis c call =
  let rest = ref (works call) in
  let axes, copy = axes_of c in
  let taken =
    List.map
      (fun e ->
        let next t =
          let mine, _ = take (List.length t) !rest in
          List.map (fun w -> token_of (List.hd w)) mine
        in
        let tokens =
          match List.find_opt (fun t -> next t = t) e.tokens with
          | Some t -> t
          | None -> List.hd e.tokens
        in
        let mine, others = take (List.length tokens) !rest in
        rest := others;
        (e, tokens, mine))
      axes
  in
  (taken, !rest, copy)

let check_call ~split c =
  let cover label b =
    if split && List.exists (fun (l, _, _) -> l = label) rare then
      classify label b
    else cover label b
  in
  cover "flipped"
    (Array.exists (fun s -> s < 0) (Iarray.to_array (L.strides c.x)));
  cover "in place" (Array.length c.axes > 1);
  let plan = plan_of c in
  match plan with
  | None -> classify "declined" true
  | Some call ->
      let keyed path =
        Array.exists
          (fun (l : F.launch) ->
            match l.work with
            | Step s -> s.path = path && Option.is_some (F.key l)
            | _ -> false)
          call.launches
      in
      cover "a specialised step, contiguous" (keyed Contiguous);
      cover "a specialised step side by side" (keyed Side_by_side);
      Array.iter check_launch call.launches;
      equal ~msg:"seconds" float_exact
        (Array.fold_left
           (fun t (l : F.launch) -> t +. l.seconds)
           0. call.launches)
        call.seconds;
      if L.numel c.dst = 0 then
        equal ~msg:"no launch" int 0 (Array.length call.launches)
      else begin
        let axes, rest, copy = by_axis c call in
        let most = ref 0 in
        List.iter
          (fun (e, tokens, mine) ->
            let sl, sb, sw = e.src and dl, db, dw = e.dst in
            let line = e.line and axis = e.axis in
            let msg = Printf.sprintf "axis %d" axis in
            equal ~msg:(msg ^ ": path") (list token) tokens
              (List.map (fun w -> token_of (List.hd w)) mine);
            let at kind = c.kind = kind && real_axis e in
            cover "contiguous" (tokens = [ Step Contiguous ]);
            cover "side by side" (tokens = [ Step Side_by_side ]);
            let ts, td = t_sides sl dl axis in
            cover "the column step"
              (tokens = [ Column First_pass; Column Last_passes ]);
            cover "columns past the column step's strides"
              (column_past ~sw ~dw line ts td);
            cover "the operand through a transpose"
              (match tokens with
              | [ Move _; Step Contiguous ] -> true
              | _ -> false);
            cover "dst through a transpose"
              (match tokens with
              | [ Step Contiguous; Move _ ] -> true
              | _ -> false);
            cover "both through transposes"
              (match tokens with
              | [ Move _; Step Contiguous; Move _ ] -> true
              | _ -> false);
            List.iter
              (fun (label, takes, _) -> cover label (takes c copy e tokens))
              rare;
            cover "long 7-smooth" (line.how = Smooth && is_long line);
            cover "long Bluestein"
              (match line.how with Bluestein _ -> is_long line | _ -> false);
            cover "an r2c's halved line" (at R2c && line.real);
            cover "an r2c's odd line" (at R2c && (not line.real) && line.n > 1);
            cover "a c2r's halved line" (at C2r && line.real);
            cover "a c2r's odd line" (at C2r && (not line.real) && line.n > 1);
            cover "a halved Bluestein line"
              (line.real
              && match line.how with Bluestein _ -> true | _ -> false);
            cover "an r2c of no point"
              (c.kind = R2c && sw = 4 && L.dim sl axis = 0);
            let first = List.hd (List.hd mine) in
            let lst = List.hd (List.rev (List.concat mine)) in
            check_side ~msg:(msg ^ ": operand") first.src ~buffer:sb ~width:sw
              sl axis;
            check_side ~msg:(msg ^ ": dst") lst.dst ~buffer:db ~width:dw dl axis;
            check_flow ~msg line mine;
            if is_long line then check_digits ~msg line mine;
            (* Each work's launches take its blocks in order: a step's of its
               lines, a transpose's (a, b) of the line's points, the operand's
               or dst's moved unchanged beside one step. *)
            List.iter
              (fun w ->
                let l0 = List.hd w in
                let next =
                  List.fold_left
                    (fun at (l : F.launch) ->
                      equal
                        ~msg:(msg ^ ": a launch's first block")
                        int at l.first_group;
                      at + l.groups)
                    0 w
                in
                if split then
                  cover "a work split into launches" (List.length w > 1);
                match l0.work with
                | Step s ->
                    equal ~msg:(msg ^ ": the step's line") line_t line s.line;
                    equal ~msg:(msg ^ ": blocks") int
                      ((count l0.src.dims + s.lines - 1) / s.lines)
                      next
                | Column_step s ->
                    equal ~msg:(msg ^ ": the step's line") line_t line s.line;
                    let d = l0.src.dims.(Array.length l0.src.dims - 1) in
                    equal ~msg:(msg ^ ": blocks") int
                      (count l0.src.dims / d * ((d + 15) / 16) * 16)
                      next
                | Transpose { split; _ } when is_long line ->
                    check_record ~msg line l0;
                    equal ~msg:(msg ^ ": tiles") int (tiles_of l0 split) next
                | Transpose { split; _ } ->
                    check_record ~msg line l0;
                    equal ~msg:(msg ^ ": tiles") int (tiles_of l0 split) next;
                    cover "a 4-byte transpose" (l0.src.width = 4);
                    equal ~msg:(msg ^ ": moved bytes") int l0.src.width
                      l0.dst.width;
                    equal ~msg:(msg ^ ": moved points") int l0.src.points
                      l0.dst.points;
                    equal ~msg:(msg ^ ": split") (Windtrap.pair int int)
                      (1, l0.src.points) split
                | Split _ ->
                    equal ~msg:(msg ^ ": split blocks") int (pair_blocks l0.dst)
                      next
                | Prepass _ ->
                    equal ~msg:(msg ^ ": prepass blocks") int
                      (pair_blocks l0.src) next)
              mine;
            let lines = count (Iarray.to_array (L.shape (batch_of sl axis))) in
            most := Int.max !most (3 * 8 * lines * line.l1))
          axes;
        equal ~msg:"launches past the axes" int 0 (List.length rest);
        cover "workspace" (call.workspace > 0);
        at_most ~msg:"workspace" int ~than:(copy + !most) call.workspace;
        (* Steps by n alone: the same extents C-contiguous plan the same. *)
        match plan_of (packed_case c) with
        | Some packed_call ->
            equal ~msg:"steps by n alone"
              (list (Windtrap.pair int int_array))
              (steps_of packed_call) (steps_of call)
        | None -> ()
      end

(* The most seconds of one group of a case's plan at the full budget. *)
let most_group c =
  match plan_of c with
  | Some call when Array.length call.launches > 0 ->
      Some
        (Array.fold_left
           (fun m l -> Float.max m (work_group l))
           0. call.launches)
  | _ -> None

(* Under the least launch budget a call's groups allow, the most one-group
   seconds of its plan at the full budget and of its C-contiguous twin's, which
   the laws plan too, the same laws with work split; just below its own, the
   plan is refused. *)
let law_split c =
  match (most_group c, most_group (packed_case c)) with
  | None, _ -> check_call ~split:true c
  | Some own, twin ->
      let s = Float.max own (Option.value twin ~default:own) in
      Fun.protect
        ~finally:(fun () -> P.bound_budget P.budget)
        (fun () ->
          P.bound_budget (Float.pred own);
          raises_match Exn.invalid_arg (fun () -> ignore (plan_of c));
          P.bound_budget s;
          check_call ~split:true c)

(* [c]'s plan under the least budget [law_split] plans it at. *)
let split_plan c =
  let own = Option.get (most_group c) in
  let twin = Option.value (most_group (packed_case c)) ~default:own in
  Fun.protect
    ~finally:(fun () -> P.bound_budget P.budget)
    (fun () ->
      P.bound_budget (Float.max own twin);
      Option.get (plan_of c))

let law_call c = check_call ~split:false c

(* For each of the [rare] paths, a call that takes it and splits under the least
   budget its groups allow, with the work of the path: an odd real Rader line,
   of 3001 points along the last axis of a broadcast operand; a long line of
   three steps, of 3,828,125 = 175 · 175 · 125 points; a c2r of two axes, whose
   last axis reads its copy in workspace; a long halved r2c's split and a long
   halved c2r's pre-pass, of 8194 points, Bluestein's; a column step of 4096 by
   17 columns, two blocks a run. *)
let split_examples =
  let v ?(offset = 0) strides shape =
    L.v ~offset ~strides:(Iarray.of_array strides) (Iarray.of_array shape)
  in
  let case name kind axes x dst bases = { name; kind; axes; x; dst; bases } in
  [
    ( case "odd real Rader" R2c [| 0; 1 |]
        (v ~offset:3 [| 3; 0 |] [| 1024; 3001 |])
        (packed [| 1024; 1501 |])
        (0, 0),
      "an odd real Rader line" );
    ( case "three-step long" C2c [| 1 |]
        (v [| 0; 1 |] [| 1; 3828125 |])
        (v [| 0; 1 |] [| 1; 3828125 |])
        (120, 0),
      "a long line of three steps" );
    ( case "c2r through a copy" C2r [| 0; 1 |]
        (packed [| 60; 501 |])
        (packed [| 60; 1000 |])
        (8, 120),
      "a c2r through a copy" );
    ( case "long halved split" R2c [| 1 |]
        (v ~offset:8589934597 [| 1; 3 |] [| 3; 8194 |])
        (packed [| 3; 4098 |])
        (0, 0),
      "a long halved r2c's split" );
    ( case "long halved pre-pass" C2r [| 1 |]
        (v ~offset:8589934599 [| -1; 3 |] [| 3; 4098 |])
        (packed [| 3; 8194 |])
        (0, 0),
      "a long halved c2r's prepass" );
    ( case "column step" C2c [| 0 |]
        (packed [| 4096; 17 |])
        (packed [| 4096; 17 |])
        (0, 0),
      "a column step" );
  ]

(* Calls of paths the small-budget law demands that its draws reach in 3 to 6%
   of cases, so that each run reaches them: an r2c of no point, a column step
   whose points lie 16384 elements apart, past its strides, and a c2c of 64
   points 2^31 / 63 + 1 elements apart, which moves through wide transposes. *)
let split_covers =
  let v strides shape =
    L.v ~strides:(Iarray.of_array strides) (Iarray.of_array shape)
  in
  let case name kind axes x dst =
    { name; kind; axes; x; dst; bases = (0, 0) }
  in
  [
    case "r2c of no point" R2c [| 1 |] (packed [| 3; 0 |]) (packed [| 3; 1 |]);
    case "columns past the strides" C2c [| 0 |]
      (v [| 16384; 1 |] [| 4096; 17 |])
      (packed [| 4096; 17 |]);
    case "wide" C2c [| 0 |]
      (v [| ((1 lsl 31) / 63) + 1; 1 |] [| 64; 1 |])
      (packed [| 64; 1 |]);
  ]

(* Each stated call takes its path, by the predicate [check_call] classifies,
   and the path's work under the least budget is cut into launches, the second
   past the work's first group. *)
let test_split_paths (c, label) =
  let _, takes, work = List.find (fun (l, _, _) -> l = label) rare in
  let axes, _, copy = by_axis c (split_plan c) in
  let on = List.filter (fun (e, tokens, _) -> takes c copy e tokens) axes in
  greater ~msg:"axes taking the path" int ~than:0 (List.length on);
  List.iter
    (fun (_, _, mine) ->
      let ws = List.filter (fun w -> work (token_of (List.hd w))) mine in
      greater ~msg:"works of the path" int ~than:0 (List.length ws);
      List.iter
        (fun w ->
          at_least ~msg:"launches of the path's work" int ~than:2
            (List.length w);
          greater ~msg:"the second launch's first group" int ~than:0
            (List.nth w 1 : F.launch).first_group)
        ws)
    on

(* A twiddled launch's bytes: its tiles' as charged, and at least the points
   they move and the high table's lines their lanes read, with the low table. *)
let check_twiddled (split, (l : F.launch)) =
  equal ~msg:"src's access" access Rows l.src.access;
  equal ~msg:"dst's access" access Rows l.dst.access;
  let charged = ref 0 and read = ref 0 in
  for g = l.first_group to l.first_group + l.groups - 1 do
    charged := !charged + tile_bytes split g;
    let (c0, c1), rows = tile_at split g in
    read := !read + (16 * (c1 - c0) * (snd rows - fst rows));
    for c = c0 to c1 - 1 do
      read := !read + (128 * touched c rows)
    done
  done;
  equal ~msg:"bytes" int (!charged + low_table) l.cost.bytes;
  at_least ~msg:"bytes against those read" int ~than:(!read + low_table)
    l.cost.bytes

type tcase = { n : int; lines : int; parts : int }

(* Long lines, those of two 4096-point steps among them, in launches of about a
   [parts]th of a twiddled transpose's. *)
let tcase =
  let open Gen in
  with_pp
    (fun ppf c -> Format.fprintf ppf "[%d; %d] in %d parts" c.lines c.n c.parts)
    (let+ seed = int in
     let st = Random.State.make [| seed |] in
     let n =
       if Random.State.bool st then smooth_below st (1 lsl 24)
       else pick st [| 1 lsl 24; 455_625 |]
     in
     { n; lines = pick st [| 1; 3 |]; parts = 2 + Random.State.int st 15 })

(* A case's twiddled launches under a budget of about a [parts]th of its
   largest. *)
let twiddled_in c =
  let shape = [| c.lines; c.n |] in
  let case =
    {
      name = "twiddle";
      kind = C2c;
      axes = [| 1 |];
      x = packed shape;
      dst = packed shape;
      bases = (0, 0);
    }
  in
  match (plan_of case, most_group case) with
  | Some call, Some own -> (
      let whole =
        List.fold_left
          (fun m (_, (l : F.launch)) -> Float.max m l.seconds)
          0. (twiddled call)
      in
      Fun.protect
        ~finally:(fun () -> P.bound_budget P.budget)
        (fun () ->
          P.bound_budget (Float.max own (whole /. Float.of_int c.parts));
          plan_of case)
      |> function
      | None -> fail "declined"
      | Some call -> twiddled call)
  | _ -> fail "declined"

let law_twiddle c =
  List.iter
    (fun (((a, b) as split), (l : F.launch)) ->
      let first = l.first_group and last = l.first_group + l.groups - 1 in
      let (c0, _), _ = tile_at split first in
      cover "a twiddled transpose split" (first > 0);
      cover "a launch of columns from 2048 alone" (c0 >= 2048 && b > 2048);
      cover "a launch across runs"
        (first / run_tiles (a, b) <> last / run_tiles (a, b));
      check_twiddled (split, l))
    (twiddled_in c)

(* At 4096 columns, in launches of at most 128 tiles, some launch holds the last
   32 columns' tiles alone, and each is charged at least what it reads. *)
let test_last_columns () =
  let launches = twiddled_in { n = 1 lsl 24; lines = 1; parts = 256 } in
  let last (((_, b) as split), (l : F.launch)) =
    let (c0, _), _ = tile_at split l.first_group in
    c0 = b - 32
  in
  equal ~msg:"a launch of the last columns alone" bool true
    (List.exists last launches);
  List.iter check_twiddled launches

(* Two-step lines' twiddled transposes: their tiles' points at 16 bytes, the low
   table's 8 KiB, the product's 4 links past a transpose's and two divisions' 3
   each to find its tile, and each column's lanes in each of its tiles charged 2
   lines of the high table, one more from each of columns 496, 1024, 1553, 2082,
   2610, 3139 and 3667: 2 at each of the 90 columns of [(90, 90)]'s 3 tiles a
   column; 2576 lines over 1024 columns, in 32 tiles; 22,293 over 4096, in
   128. *)
let twiddle_cases =
  [
    (8100, (90, 90), 9, (8100 * 16) + (128 * 3 * 90 * 2) + 8192);
    (1 lsl 20, (1024, 1024), 1024, (1 lsl 24) + (128 * 32 * 2576) + 8192);
    (1 lsl 24, (4096, 4096), 16384, (1 lsl 28) + (128 * 128 * 22_293) + 8192);
  ]

let test_twiddle (n, split, groups, bytes) =
  match
    F.c2c ~bases:(0, 0) ~axes:[| 0 |] (packed [| n |]) ~dst:(packed [| n |])
  with
  | None -> fail "declined"
  | Some c -> (
      match twiddled c with
      | [ (s, l) ] ->
          equal ~msg:"split" (Windtrap.pair int int) split s;
          equal ~msg:"groups" int groups l.groups;
          equal ~msg:"bytes" int bytes l.cost.bytes;
          equal ~msg:"chain" int
            (P.read_links + 4 + 2 + 4 + (3 * 2))
            l.cost.chain
      | _ -> fail "not one twiddled launch")

(* A full group's lines, in one group: a dependent read, then each pass its
   butterfly's depth and two barriers of 4 links, and past the first its
   twiddle's product, 2 links, and the issue of a thread's ⌈(l / r) / tl⌉ (r -
   1) loads at 0.15 links a SIMD group's load. - 1024 points, radices 8 8 8 2: 2
   lines of 128 threads, 8 SIMD groups, 7, 7 and 4 loads; 16 + ⌈12 + 2 (14 +
   8.4) + (11 + 4.8)⌉. - 90 points, radices 2 5 3 3: 14 lines of 18 threads, 8
   SIMD groups, 4 loads each pass, 1 · 4, 2 · 2 and 2 · 2; 16 + ⌈9 + (15 + 4.8)
   + 2 (13 + 4.8)⌉. *)
let chain_cases = [ (1024, 2, 256, 16 + 73); (90, 14, 256, 16 + 65) ]

let test_step_chain (n, lines, threads, chain) =
  let x = packed [| lines; n |] in
  match F.c2c ~bases:(0, 0) ~axes:[| 1 |] x ~dst:x with
  | Some { launches = [| l |]; _ } ->
      equal ~msg:"threads" int threads l.threads;
      equal ~msg:"chain" int chain l.cost.chain
  | _ -> fail "not one launch"

(* Transposes of runs, or lines, of 90 or 65 points take 3 by 3 tiles of 32: a
   line of 8100 = 90 · 90 points, each of its three transposes; 65 lines of 65
   points, Bluestein's, moved through the workspace from a scattered side into a
   columns one. *)
let tile_cases =
  [
    ("8100", packed [| 8100 |], [| 9; 9; 9 |]);
    ( "[65; 65] along axis 0",
      laid ~steps:[| 1; 2 |] ~order:[| 0; 1 |] ~base:0 [| 65; 65 |],
      [| 9; 9 |] );
  ]

let test_tiles (_, x, blocks) =
  match
    F.c2c ~bases:(0, 0) ~axes:[| 0 |] x
      ~dst:(packed (Iarray.to_array (L.shape x)))
  with
  | None -> fail "declined"
  | Some c ->
      let moves =
        List.filter_map
          (fun (l : F.launch) ->
            match l.work with Transpose _ -> Some l.groups | _ -> None)
          (Array.to_list c.launches)
      in
      equal ~msg:"tiles" int_array blocks (Array.of_list moves)

(* Rader's lines side by side take [side (n - 1)] lines, a float plane of [n -
   1] points each: 17 points, 16 at 2 threads a line, 256 lines; 257, 256 at 32,
   16 lines; both 16 KiB. *)
let test_rader_planes (n, g) =
  let shape = [| n; 600 |] in
  match
    F.c2c ~bases:(0, 0) ~axes:[| 0 |] (packed shape) ~dst:(packed shape)
  with
  | None -> fail "declined"
  | Some c -> (
      match c.launches with
      | [| { work = Step { path = Side_by_side; lines; _ }; shared; _ } |] ->
          equal ~msg:"lines" int g lines;
          equal ~msg:"planes" int (16 * 1024) shared
      | _ -> fail "not one side-by-side step")

(* A side's points reach past 32-bit offsets where [|ps| (points - 1)] passes
   [2^31 - 1] elements: a complex side of 2 points, a real one of 2, and a real
   one of 16384, whose long halved line reads its points as pairs of floats, at
   the most [ps] within and the least past. *)
let wide_cases =
  [
    ("c2c of 2", C2c, 2, (1 lsl 31) - 1, false);
    ("c2c of 2", C2c, 2, 1 lsl 31, true);
    ("r2c of 2", R2c, 2, (1 lsl 31) - 1, false);
    ("r2c of 2", R2c, 2, 1 lsl 31, true);
    ("r2c of 16384", R2c, 16384, ((1 lsl 31) - 1) / 16383, false);
    ("r2c of 16384", R2c, 16384, (((1 lsl 31) - 1) / 16383) + 1, true);
  ]

let test_wide (_, t, n, ps, wide) =
  let x =
    L.v ~strides:(Iarray.of_array [| ps; 0 |]) (Iarray.of_array [| n; 1 |])
  in
  let plan =
    match t with
    | C2c -> F.c2c ~bases:(0, 0) ~axes:[| 0 |] x ~dst:(packed [| n; 1 |])
    | R2c ->
        F.r2c ~bases:(0, 0) ~axes:[| 0 |] x ~dst:(packed [| (n / 2) + 1; 1 |])
    | C2r -> fail "no c2r case"
  in
  match plan with
  | None -> fail "declined"
  | Some c -> (
      match c.launches.(0).work with
      | Transpose t ->
          equal ~msg:"reads the operand" bool true
            (c.launches.(0).src.buffer = Operand);
          equal ~msg:"wide" bool wide t.wide
      | _ -> fail "a step reads the operand")

(* A long halved line's real side moves its points as pairs of floats, 8 bytes a
   point on rows and two cache lines off them; a workspace side, rows, 8. *)
let pair_cases =
  let n = 32768 and h = 16385 in
  let stride =
    L.v ~strides:(Iarray.of_array [| 2 |]) (Iarray.of_array [| n |])
  in
  [
    ("an r2c's operand on rows", R2c, packed [| n |], packed [| h |], 8 + 8);
    ("an r2c's operand off rows", R2c, stride, packed [| h |], 256 + 8);
    ("a c2r's dst on rows", C2r, packed [| h |], packed [| n |], 8 + 8);
  ]

let test_pairs (_, kind, x, dst, per_point) =
  let plan =
    plan_of { name = ""; kind; axes = [| 0 |]; x; dst; bases = (0, 0) }
  in
  match plan with
  | None -> fail "declined"
  | Some c -> (
      let l =
        if kind = R2c then c.launches.(0)
        else c.launches.(Array.length c.launches - 1)
      in
      match l.work with
      | Transpose _ ->
          equal ~msg:"bytes" int (l.groups * 1024 * per_point) l.cost.bytes
      | _ -> fail "not a transpose")

(* A c2r's pre-pass reading bins 2^12 elements apart, 2^20 + 1 of them, spans
   2^32 elements: past 32-bit offsets. *)
let test_wide_prepass () =
  let h = 1 lsl 20 in
  let x =
    L.v ~strides:(Iarray.of_array [| 1 lsl 12 |]) (Iarray.of_array [| h + 1 |])
  in
  match F.c2r ~bases:(0, 0) ~axes:[| 0 |] x ~dst:(packed [| 2 * h |]) with
  | None -> fail "declined"
  | Some c -> (
      match c.launches.(0).work with
      | Prepass { wide } -> equal ~msg:"wide" bool true wide
      | _ -> fail "not a pre-pass")

(* A line of two steps [s_0], [s_1] ends as [s_0] rows of [s_1] written as [s_1]
   rows of [s_0], natural order; Bluestein's at each of its two stages. *)
let last_split_cases =
  [ ("6000", 6000, (100, 60)); ("65536", 65536, (256, 256)) ]

let last_splits (c : F.call) =
  let splits =
    List.filter_map
      (fun (l : F.launch) ->
        match l.work with
        | Transpose { split; _ } when l.first_group = 0 -> Some split
        | _ -> None)
      (Array.to_list c.launches)
  in
  splits

let test_last_split (_, n, split) =
  let x = packed [| n |] in
  match F.c2c ~bases:(0, 0) ~axes:[| 0 |] x ~dst:x with
  | None -> fail "declined"
  | Some c ->
      equal ~msg:"the last transpose's split" (Windtrap.pair int int) split
        (List.hd (List.rev (last_splits c)))

let test_bluestein_last_split () =
  let n = 10007 in
  let line = F.line n ~real:false in
  let x = packed [| n |] in
  match (line.steps, F.c2c ~bases:(0, 0) ~axes:[| 0 |] x ~dst:x) with
  | [| s0; s1 |], Some c ->
      let splits = Array.of_list (last_splits c) in
      equal ~msg:"stage 1's last split" (Windtrap.pair int int) (s0.l, s1.l)
        splits.(2);
      equal ~msg:"stage 2's last split" (Windtrap.pair int int) (s0.l, s1.l)
        splits.(5)
  | _, None -> fail "declined"
  | _ -> fail "not two steps"

(* Long lines come out in natural order: 7-smooth ones of two and three steps,
   and Bluestein's of two, to 524,287 points, the most within n_B. *)
let order_cases =
  [
    (65536, 2);
    (6000, 2);
    (10007, 2);
    (3_828_125, 3);
    (1 lsl 25, 3);
    (524_287, 2);
  ]

let test_order (n, q) =
  let line = F.line n ~real:false in
  equal ~msg:"steps" int q (Array.length line.steps);
  let x = packed [| n |] in
  match F.c2c ~bases:(0, 0) ~axes:[| 0 |] x ~dst:x with
  | None -> fail "declined"
  | Some c -> check_digits ~msg:"line" line (works c)

let c2c_refused =
  let shape = [| 4; 4 |] in
  let call axes x () =
    ignore (F.c2c ~bases:(0, 0) ~axes x ~dst:(packed shape))
  in
  [
    ("no axis", call [||] (packed shape));
    ("axes out of order", call [| 1; 0 |] (packed shape));
    ("an axis past the rank", call [| 2 |] (packed shape));
    ("shapes differ", call [| 0 |] (packed [| 4; 5 |]));
  ]
  @ List.map
      (fun bases ->
        ( Printf.sprintf "bases (%d, %d)" (fst bases) (snd bases),
          fun () ->
            ignore
              (F.c2c ~bases ~axes:[| 0 |] (packed shape) ~dst:(packed shape)) ))
      [ (128, 0); (-8, 0); (0, 128) ]

let real_refused =
  let call f x dst () =
    ignore (f ~bases:(0, 0) ~axes:[| 0 |] (packed x) ~dst:(packed dst))
  in
  [
    ("an r2c into n bins", call F.r2c [| 8 |] [| 8 |]);
    ("an r2c into n / 2 bins", call F.r2c [| 8 |] [| 4 |]);
    ("an r2c with batches that differ", call F.r2c [| 8; 2 |] [| 5; 3 |]);
    ("a c2r from n bins", call F.c2r [| 8 |] [| 8 |]);
    ("a c2r from n / 2 + 2 bins", call F.c2r [| 6 |] [| 8 |]);
    ( "a c2r with no axis",
      fun () ->
        ignore
          (F.c2r ~bases:(0, 0) ~axes:[||] (packed [| 5 |]) ~dst:(packed [| 8 |]))
    );
  ]

(* A real line of 8192 points is a halved one of 4096, in one group: its plane
   fills 32 KiB, bin 4096 kept outside it. *)
let test_halved kind =
  let real = packed [| 8192 |] and bins = packed [| 4097 |] in
  let plan =
    match kind with
    | R2c -> F.r2c ~bases:(0, 0) ~axes:[| 0 |] real ~dst:bins
    | _ -> F.c2r ~bases:(0, 0) ~axes:[| 0 |] bins ~dst:real
  in
  match plan with
  | None -> fail "declined"
  | Some c -> (
      match c.launches with
      | [|
       {
         work = Step { path = Contiguous; step; line; lines; _ };
         src;
         dst;
         shared;
         _;
       };
      |] ->
          let reals, bins = if kind = R2c then (src, dst) else (dst, src) in
          equal ~msg:"line" line_t (F.line 4096 ~real:true) line;
          equal ~msg:"step's points" int 4096 step.l;
          equal ~msg:"lines a group" int 1 lines;
          equal ~msg:"plane" int (32 * 1024) shared;
          equal ~msg:"reals" (Windtrap.pair int int) (8192, 4)
            (reals.points, reals.width);
          equal ~msg:"bins" (Windtrap.pair int int) (4097, 8)
            (bins.points, bins.width)
      | _ -> fail "not one contiguous step")

(* An r2c of lines of no point stores a bin a line, from a step of one point. *)
let test_no_point () =
  match
    F.r2c ~bases:(0, 0) ~axes:[| 1 |]
      (packed [| 3; 0 |])
      ~dst:(packed [| 3; 1 |])
  with
  | None -> fail "declined"
  | Some c -> (
      match c.launches with
      | [| { work = Step { line; step; _ }; src; dst; _ } |] ->
          equal ~msg:"line" line_t (F.line 1 ~real:false) line;
          equal ~msg:"step's points" int 1 step.l;
          equal ~msg:"points read" int 0 src.points;
          equal ~msg:"bins stored" int 1 dst.points;
          equal ~msg:"lines" int 3 (count dst.dims)
      | _ -> fail "not one step")

(* A columns side's runs are charged the lines they touch from their worst
   start. A group's segment of a run starts at its buffer's byte [β] of a line,
   plus the run's lowest element, plus multiples of the point stride, the outer
   batch strides and [gcd(g, d)] lines, in bytes, mod 128. A group of [g] lines
   meets [⌈g / r⌉] runs of [r = min(g, d)], one more where neither of [g] and
   [d] divides the other. - x.(:, :, 0:2) of a complex [64; 1024; 3] along axis
   1: points 24 B apart, [g = 4] and [gcd(4, 2)] lines, 16 B, so runs start at
   multiples of 8, the worst 120, where its 16 B touch 2 lines: 2 runs a group,
   4 lines a point; dst, [64; 1024; 2], at multiples of 16, the worst 112, one
   line a run, 2; 32 groups of 4 lines, each 128 · 1024 · 6 bytes, and the 4 KiB
   roots, half the step's points. - [64; 1024; 2] over a buffer at byte 120 of
   its line: 120 again, 6 lines; at byte 0, 112, 4, 64 B a point a side. - x.(:,
   0:17) of [512; 32] along axis 0: [g = 8], [d = 17], one line apart, 8 B, the
   worst 120; 8 lines of 8 B from there touch 2 lines, 2 runs a group since
   neither of 8 and 17 divides the other, a side's 4 lines a point; dst's runs,
   17 lines of 8 B apart, alike; 3 groups, each 128 · 512 · 8 bytes. - [256; 17]
   of complex64 along axis 0 over a buffer at byte 4, a multiple of its
   storage's alignment but not of 8: [g = 16], [d = 17], 2 runs a group, which
   start at multiples of 8 past byte 4, the worst 124, where 128 B touch 2
   lines, 4 a point; dst on a line, the worst 120, 2 lines a run, 4; 2 groups,
   each 128 · 256 · 8 bytes, and the 1 KiB roots. *)
let start_cases =
  let v ?(offset = 0) strides shape =
    L.v ~offset ~strides:(Iarray.of_array strides) (Iarray.of_array shape)
  in
  [
    ( "x.(:, :, 0:2) of [64; 1024; 3]",
      v [| 3072; 3; 1 |] [| 64; 1024; 2 |],
      (0, 0),
      1,
      (32 * 128 * 1024 * 6) + 4096 );
    ( "[64; 1024; 2] at byte 120",
      packed [| 64; 1024; 2 |],
      (120, 0),
      1,
      (32 * 128 * 1024 * 6) + 4096 );
    ( "[64; 1024; 2] at byte 0",
      packed [| 64; 1024; 2 |],
      (0, 0),
      1,
      (32 * 128 * 1024 * 4) + 4096 );
    ( "x.(:, 0:17) of [512; 32]",
      v [| 32; 1 |] [| 512; 17 |],
      (0, 0),
      0,
      (3 * 128 * 512 * 8) + 2048 );
    ( "[256; 17] at byte 4",
      packed [| 256; 17 |],
      (4, 0),
      0,
      (2 * 128 * 256 * 8) + 1024 );
  ]

let side_step bases axis x =
  let dst = packed (Iarray.to_array (L.shape x)) in
  match F.c2c ~bases ~axes:[| axis |] x ~dst with
  | Some
      { launches = [| { work = Step { path = Side_by_side; _ }; _ } as l |]; _ }
    ->
      l
  | Some _ -> fail "not one side-by-side launch"
  | None -> fail "declined"

let test_starts (_, x, bases, axis, bytes) =
  equal ~msg:"bytes" int bytes (side_step bases axis x).cost.bytes

(* A view at byte 120 of a host buffer, which starts on a line, lies at byte 120
   of its line, and its columns are charged as the case above. *)
let test_view () =
  let b = Rig.Buffer.create Rig.host 4096 in
  List.iter
    (fun at ->
      equal
        ~msg:(Printf.sprintf "view at %d" at)
        int (at mod 128)
        (F.line_offset (Rig.Buffer.view b ~first:at ~length:256)))
    [ 0; 8; 120; 128; 248; 1000 ];
  let beta = F.line_offset (Rig.Buffer.view b ~first:120 ~length:256) in
  equal ~msg:"bytes" int
    ((32 * 128 * 1024 * 6) + 4096)
    (side_step (beta, 0) 1 (packed [| 64; 1024; 2 |])).cost.bytes

(* Side by side only where it costs at most the transposes, 288 B a point: x.(:,
   :, 15:18) of a complex [64; 2048; 32], [d = 3] against a 4-line group,
   neither dividing the other, starts its group's segments 8 B apart, the worst
   at byte 120, 3 runs of 2 lines a group, and so does its dst, 384 B a point;
   x.(:, :, 16:18), [d = 2] dividing the group, 2 runs of one line a group on
   each side, 128 B. *)
let route_cases =
  [
    ("x.(:, :, 15:18)", 15, 3, [ moved 8; Step Contiguous; moved 8 ]);
    ("x.(:, :, 16:18)", 16, 2, [ Step Side_by_side ]);
  ]

let test_route (name, offset, d, tokens) =
  let x =
    L.v ~offset
      ~strides:(Iarray.of_array [| 65536; 32; 1 |])
      (Iarray.of_array [| 64; 2048; d |])
  in
  let c =
    {
      name;
      kind = C2c;
      axes = [| 1 |];
      x;
      dst = packed [| 64; 2048; d |];
      bases = (0, 0);
    }
  in
  let plan = plan_of c in
  match plan with
  | None -> fail "declined"
  | Some call ->
      equal ~msg:"path" (list token) tokens
        (List.map (fun w -> token_of (List.hd w)) (works call))

(* The column step where a line's consecutive points lie at most 8 KiB apart on
   both sides, by the strides alone, a view's its buffer's: 4096 by 256 packed,
   as views of 4096 by 1024 (8 KiB), flipped too, of 4096 by 1025 (8 KiB and 8
   bytes) and of 4096 by 16384 (128 KiB), 4096 by 15, too few lines, and 16 runs
   of 256 lines along axis 1 of [16; 4096; 256], 2 KiB. *)
let column_cases =
  let v ?(offset = 0) strides shape =
    L.v ~offset ~strides:(Iarray.of_array strides) (Iarray.of_array shape)
  in
  let column = [ Column First_pass; Column Last_passes ] in
  let through = [ moved 8; Step Contiguous; moved 8 ] in
  [
    ("4096 x 256", v [| 256; 1 |] [| 4096; 256 |], 0, column);
    ("of 4096 x 1024", v [| 1024; 1 |] [| 4096; 256 |], 0, column);
    ( "of 4096 x 1024, flipped",
      v ~offset:(4095 * 1024) [| -1024; 1 |] [| 4096; 256 |],
      0,
      column );
    ("of 4096 x 1025", v [| 1025; 1 |] [| 4096; 256 |], 0, through);
    ("of 4096 x 16384", v [| 16384; 1 |] [| 4096; 256 |], 0, through);
    ("4096 x 15", v [| 15; 1 |] [| 4096; 15 |], 0, through);
    ("[16; 4096; 256] along 1", packed [| 16; 4096; 256 |], 1, column);
  ]

let column_call x axis bases =
  let shape = Iarray.to_array (L.shape x) in
  {
    name = "columns";
    kind = C2c;
    axes = [| axis |];
    x;
    dst = packed shape;
    bases;
  }

let test_column (_, x, axis, tokens) =
  let c = column_call x axis (0, 0) in
  let plan = plan_of c in
  match plan with
  | None -> fail "declined"
  | Some call ->
      equal ~msg:"path" (list token) tokens
        (List.map (fun w -> token_of (List.hd w)) (works call))

(* A column step's workspace (fft.mli): the axis's, line [c] of run [i] at
   element [4096 d i + c], its points [d] apart, the first pass's destination
   and the last passes' source; each group's bytes 256 segments of 128 bytes a
   side, two lines each from a start at byte 120, and the last passes the roots
   of 4096. *)
let column_bytes_cases =
  let mib = 1 lsl 20 in
  [
    ( "4096 x 256",
      [| 4096; 256 |],
      0,
      (0, 0),
      ([| 256 |], [| 1 |]),
      (16 * mib, 16 * mib) );
    ( "4096 x 256 at byte 120",
      [| 4096; 256 |],
      0,
      (120, 0),
      ([| 256 |], [| 1 |]),
      (24 * mib, 16 * mib) );
    ( "[16; 4096; 256] along 1",
      [| 16; 4096; 256 |],
      1,
      (0, 0),
      ([| 16; 256 |], [| 4096 * 256; 1 |]),
      (256 * mib, 256 * mib) );
  ]

let test_column_bytes (_, shape, axis, bases, (dims, strides), (first, last)) =
  let c = column_call (packed shape) axis bases in
  match plan_of c with
  | None -> fail "declined"
  | Some call -> (
      match works call with
      | [ ((f : F.launch) :: _ as fw); ((l : F.launch) :: _ as lw) ] ->
          let w = f.dst in
          equal ~msg:"workspace" buffer (Workspace 0) w.buffer;
          equal ~msg:"workspace's first" int 0 w.first;
          equal ~msg:"workspace's point stride" int 256 w.ps;
          equal ~msg:"workspace's lines" int_array dims w.dims;
          equal ~msg:"workspace's strides" int_array strides w.strides;
          equal ~msg:"read where written" bool true (l.src = w);
          let bytes ~table ws =
            List.fold_left
              (fun b (l : F.launch) -> b + l.cost.bytes - table)
              0 ws
          in
          equal ~msg:"the first pass's bytes" int first (bytes ~table:0 fw);
          equal ~msg:"the last passes' bytes" int last
            (bytes ~table:(8 * 2048) lw)
      | _ -> fail "not two works")

(* A column step's bound (fft.mli's c2c): where it fits, its seconds are at most
   0.51 of the transposes'. A layout that fits: 4096 points of [d] lines, [16]
   to [1024], each side's points [pitch] elements apart, [d] to [1024], so at
   most 8 KiB, in one run or three, its point axis or its line axis flipped,
   from any 4-byte base. *)
type fitting = {
  d : int;
  runs : int;
  sides : (int * int) array;
  fbases : int * int;
}

let pp_fitting ppf c =
  let side ppf (pitch, flip) =
    Format.fprintf ppf "pitch %d flip %d" pitch flip
  in
  Format.fprintf ppf "%d lines, %d runs, operand %a, dst %a, bases (%d, %d)" c.d
    c.runs side c.sides.(0) side c.sides.(1) (fst c.fbases) (snd c.fbases)

let fitting =
  let open Gen in
  with_pp pp_fitting
    (let* d =
       frequency
         [
           (1, constant 16);
           (1, constant 17);
           (1, constant 1024);
           (3, int_range 16 1024);
         ]
     in
     let side =
       let* pitch = int_range d 1024 in
       let+ flip = int_range 0 2 in
       (pitch, flip)
     in
     let* runs = frequency [ (1, constant 1); (1, constant 3) ] in
     let* x = side in
     let* y = side in
     let* bx = int_range 0 31 in
     let+ bd = int_range 0 31 in
     { d; runs; sides = [| x; y |]; fbases = (4 * bx, 4 * bd) })

let fitting_layout c (pitch, flip) =
  let ps = if flip = 1 then -pitch else pitch in
  let inner = if flip = 2 then -1 else 1 in
  let offset =
    (if ps < 0 then 4095 * pitch else 0) + if inner < 0 then c.d - 1 else 0
  in
  let shape, strides =
    if c.runs = 1 then ([| 4096; c.d |], [| ps; inner |])
    else ([| c.runs; 4096; c.d |], [| 4096 * pitch; ps; inner |])
  in
  L.v ~offset ~strides:(Iarray.of_array strides) (Iarray.of_array shape)

let column_bound = 0.51

let law_column_bound c =
  let x = fitting_layout c c.sides.(0) and dst = fitting_layout c c.sides.(1) in
  let axes = [| (if c.runs = 1 then 0 else 1) |] in
  cover "16 lines" (c.d = 16);
  cover "17 lines" (c.d = 17);
  cover "1024 lines" (c.d = 1024);
  cover "a flipped point axis" (Array.exists (fun (_, f) -> f = 1) c.sides);
  cover "a flipped line axis" (Array.exists (fun (_, f) -> f = 2) c.sides);
  cover "bases off a line" (c.fbases <> (0, 0));
  match
    ( F.c2c ~bases:c.fbases ~axes x ~dst,
      F.c2c_moved ~bases:c.fbases ~axes x ~dst )
  with
  | Some call, Some moved ->
      equal ~msg:"the column step" (list token)
        [ Column First_pass; Column Last_passes ]
        (List.map (fun w -> token_of (List.hd w)) (works call));
      at_most ~msg:"seconds" float_exact
        ~than:(column_bound *. moved.seconds)
        call.seconds
  | _ -> fail "declined"

let lone_cases =
  let v strides shape =
    L.v ~offset:0 ~strides:(Iarray.of_array strides) (Iarray.of_array shape)
  in
  [
    ("[1; 1024] of points 1 apart", v [| 1024; 1 |] [| 1; 1024 |], F.Rows);
    ("[1; 1024] of points 2 apart", v [| 1; 2 |] [| 1; 1024 |], F.Scattered);
    ( "[1; 1; 1024] of points 3 apart",
      v [| 1; 1; 3 |] [| 1; 1; 1024 |],
      Scattered );
  ]

let test_lone (_, x, expected) =
  let shape = Iarray.to_array (L.shape x) in
  match
    F.c2c ~bases:(0, 0) ~axes:[| Array.length shape - 1 |] x ~dst:(packed shape)
  with
  | None -> fail "declined"
  | Some call ->
      equal ~msg:"src's access" access expected call.launches.(0).src.access

(* nx_metal.mli: a line of at most 2^32 points plans, and one past it declines,
   at a 7-smooth length on either side. *)
let past_cases =
  [
    (C2c, 1 lsl 32, true);
    (C2c, 1 lsl 33, false);
    (R2c, 1 lsl 32, true);
    (R2c, 1 lsl 33, false);
    (C2r, 1 lsl 32, true);
    (C2r, 1 lsl 33, false);
  ]

let test_past (kind, n, planned) =
  let x, dst =
    match kind with
    | C2c -> ([| n |], [| n |])
    | R2c -> ([| n |], [| (n / 2) + 1 |])
    | C2r -> ([| (n / 2) + 1 |], [| n |])
  in
  let c =
    {
      name = "past";
      kind;
      axes = [| 0 |];
      x = packed x;
      dst = packed dst;
      bases = (0, 0);
    }
  in
  equal ~msg:"planned" bool planned (Option.is_some (plan_of c))

(* A call declines from its extents and its sides alone, whatever the launch
   budget: a c2c of 2^33 points, past the most a line takes, and of 524,289, a
   Bluestein line past the host's cut, under a budget below any group's
   seconds. *)
let test_declined_budget () =
  List.iter
    (fun n ->
      let c =
        {
          name = "declined";
          kind = C2c;
          axes = [| 0 |];
          x = packed [| n |];
          dst = packed [| n |];
          bases = (0, 0);
        }
      in
      let planned () = Option.is_some (plan_of c) in
      let msg = string_of_int n in
      equal ~msg:(msg ^ " at the budget") bool false (planned ());
      Fun.protect
        ~finally:(fun () -> P.bound_budget P.budget)
        (fun () ->
          P.bound_budget 1e-6;
          equal ~msg:(msg ^ " at 1 us") bool false (planned ())))
    [ 1 lsl 33; 524_289 ]

(* The host cut on Bluestein's m: the greatest m whose kernel the host makes
   within about 1 s, 2^20 (design §3.6). *)
let bluestein_cut = 1 lsl 20

(* n_B, from the plan's own m, which does not fall as n grows: m(n_B) is within
   the cut and m(n_B + 1) past it. *)
let test_cut () =
  equal ~msg:"cut" int bluestein_cut F.bluestein_most;
  let n = F.bluestein_points in
  at_most ~msg:"m(n_B)" int ~than:bluestein_cut (F.bluestein n);
  less ~msg:"cut" int ~than:(F.bluestein (n + 1)) bluestein_cut

let is_bluestein (line : F.line) =
  match line.how with F.Bluestein _ -> true | _ -> false

(* The Bluestein lengths of a transform nearest n_B, by their lines' points: the
   greatest within it and the least past it, [step] apart. *)
let around ~step ~from bluestein =
  let n_b = F.bluestein_points in
  let rec down n = if bluestein n then n else down (n - step) in
  let rec up n = if bluestein n then n else up (n + step) in
  (down (from n_b), up (from n_b + step))

let cut_cases =
  let c2c_lo, c2c_hi =
    around ~step:1 ~from:Fun.id (fun n -> is_bluestein (F.line n ~real:false))
  in
  let half_lo, half_hi =
    around ~step:2 ~from:(fun n -> 2 * n) (fun n -> is_bluestein (real_line n))
  in
  let odd n = if n mod 2 = 1 then n else n - 1 in
  let odd_lo, odd_hi =
    around ~step:2 ~from:odd (fun n -> is_bluestein (real_line n))
  in
  [
    (C2c, c2c_lo, true);
    (C2c, c2c_hi, false);
    (R2c, half_lo, true);
    (R2c, half_hi, false);
    (C2r, half_lo, true);
    (C2r, half_hi, false);
    (R2c, odd_lo, true);
    (R2c, odd_hi, false);
    (C2r, odd_lo, true);
    (C2r, odd_hi, false);
  ]

(* A Bluestein line within n_B plans, and one past it declines. *)
let test_cut_line (kind, n, planned) =
  let x, dst =
    match kind with
    | C2c -> ([| n |], [| n |])
    | R2c -> ([| n |], [| (n / 2) + 1 |])
    | C2r -> ([| (n / 2) + 1 |], [| n |])
  in
  let c =
    {
      name = "cut";
      kind;
      axes = [| 0 |];
      x = packed x;
      dst = packed dst;
      bases = (0, 0);
    }
  in
  equal ~msg:"planned" bool planned (Option.is_some (plan_of c))

(* Limits *)

(* Bluestein's m does not fall from one length to the next. *)
let law_m_rises n =
  at_most ~msg:"m(n)" int ~than:(F.bluestein (n + 1)) (F.bluestein n)

(* Each m past 4096 that a length to 2^32 takes, once, at its least length: that
   length takes it and the one before takes the m before. *)
(* The m of every length a group class weighs, from one sweep of their windows,
   is Bluestein's m of each. *)
let test_windows () =
  let ns = Array.init 8193 succ in
  equal ~msg:"m" (array int) (Array.map F.bluestein ns) (F.windows ns)

let test_least_takers () =
  let takers = F.least_takers () in
  let before = ref (F.most_points, 1) in
  List.iter
    (fun (m, n) ->
      let pm, _ = !before in
      less ~msg:"m in order" int ~than:m pm;
      equal ~msg:(Printf.sprintf "m(%d)" n) int m (F.bluestein n);
      equal
        ~msg:(Printf.sprintf "m(%d - 1)" n)
        int pm
        (if F.bluestein (n - 1) <= F.most_points then F.most_points
         else F.bluestein (n - 1));
      before := (m, n))
    takers;
  equal ~msg:"m(2^32)" int (fst !before) (F.bluestein (1 lsl 32))

(* The extents of a call of [kind] of [n] real points a line: its operand's,
   then its destination's. *)
let extents kind n =
  match kind with
  | C2c -> (n, n)
  | R2c -> (n, (n / 2) + 1)
  | C2r -> ((n / 2) + 1, n)

let laid_out strides shape =
  L.v ~offset:0 ~strides:(Iarray.of_array strides) (Iarray.of_array shape)

(* A call of [kind] along [lines] rows of [n] real points. *)
let rows_case kind n ~lines =
  let xs, ds = extents kind n in
  let lay s = laid_out [| s; 1 |] [| lines; s |] in
  {
    name = "rows";
    kind;
    axes = [| 1 |];
    x = lay xs;
    dst = lay ds;
    bases = (0, 0);
  }

(* Strides of a batch of [extents], the innermost [first], each axis one element
   past the span of those inside it, so that none merges. *)
let gapped ~first extents =
  let r = Array.length extents in
  let s = Array.make r first in
  for i = r - 2 downto 0 do
    s.(i) <- (s.(i + 1) * extents.(i + 1)) + 1
  done;
  s

(* A call of [kind] along rows of [n] real points over a batch of [rank] axes
   that do not merge: [l0] lines along the outermost and 2 along each other. *)
let ranked_case kind n ~l0 ~rank =
  let xs, ds = extents kind n in
  let batch = Array.init rank (fun i -> if i = 0 then l0 else 2) in
  let lay s =
    let st = gapped ~first:s batch in
    laid_out (Array.append st [| 1 |]) (Array.append batch [| s |])
  in
  {
    name = Printf.sprintf "rows at rank %d" rank;
    kind;
    axes = [| rank |];
    x = lay xs;
    dst = lay ds;
    bases = (0, 0);
  }

(* The high half of the unsigned product [a b], by 32-bit halves. *)
let mulhi a b =
  let lo x = Int64.logand x 0xFFFF_FFFFL in
  let hi x = Int64.shift_right_logical x 32 in
  let ll = Int64.mul (lo a) (lo b) and lh = Int64.mul (lo a) (hi b) in
  let hl = Int64.mul (hi a) (lo b) and hh = Int64.mul (hi a) (hi b) in
  let mid = Int64.add (hi ll) (Int64.add (lo lh) (lo hl)) in
  Int64.add hh (Int64.add (hi lh) (Int64.add (hi hl) (hi mid)))

(* A quotient by [d]'s divisor, as fft.mli states it. *)
let divided n d =
  if d = 1 then n
  else
    let rec bits x = if x = 0 then 0 else 1 + bits (x lsr 1) in
    let t = mulhi n (F.divisor d) in
    Int64.shift_right_logical
      (Int64.add t (Int64.shift_right_logical (Int64.sub n t) 1))
      (bits (d - 1) - 1)

let quotient =
  let open Gen in
  let* d =
    frequency
      [
        (2, int_range 1 1000);
        ( 2,
          let* k = int_range 1 61 in
          let+ e = int_range (-1) 1 in
          (1 lsl k) + e );
        (1, int_range 1 max_int);
      ]
  in
  let+ n =
    frequency
      [
        (3, int64);
        (1, of_list [ 0L; -1L; Int64.max_int; Int64.min_int ]);
        ( 2,
          let* q = int64 in
          let+ e = int_range (-1) 1 in
          Int64.add
            (Int64.mul (Int64.unsigned_div q (Int64.of_int d)) (Int64.of_int d))
            (Int64.of_int e) );
      ]
  in
  (d, n)

(* A division by a divisor is the unsigned quotient, for every 64-bit numerator
   and every divisor an extent or a count of tiles takes. *)
let law_divisor (d, n) =
  cover "a power of two" (d land (d - 1) = 0);
  cover "past 2^32" (d > 1 lsl 32);
  cover "a numerator past 2^63" (Int64.compare n 0L < 0);
  equal
    ~msg:(Printf.sprintf "%Lu / %d" n d)
    int64
    (Int64.unsigned_div n (Int64.of_int d))
    (divided n d)

(* A batch of five axes that do not merge plans. *)
let test_rank_five () =
  let c = ranked_case C2c 64 ~l0:1000 ~rank:5 in
  equal ~msg:"planned" bool true (Option.is_some (plan_of c))

(* nx_metal.mli: a call estimated past 10 s runs as several submissions, a
   Fourier transform as any other: a c2c of 2.2 G points along an axis off rows
   plans, each launch within the launch budget, though its launches take longer
   than a part together. Emitted as the door emits it, it finishes into parts,
   each within part_budget and part_launches, that hold its launches in order,
   its workspace a buffer of its own. Planned only. *)
let test_long_call () =
  let c =
    {
      name = "c2c of [240; 3000; 3000] along 0 and 1";
      kind = C2c;
      axes = [| 0; 1 |];
      x = packed [| 240; 3000; 3000 |];
      dst = packed [| 240; 3000; 3000 |];
      bases = (0, 0);
    }
  in
  match plan_of c with
  | None -> fail "declined"
  | Some call ->
      greater ~msg:"seconds" float_exact ~than:P.part_budget call.seconds;
      Array.iter
        (fun (l : F.launch) ->
          at_most ~msg:"a launch's seconds" float_exact ~than:(P.allowed ())
            l.seconds)
        call.launches;
      equal ~msg:"runs" bool true (F.runs call);
      let b = P.make () in
      let ws = if call.workspace > 0 then P.take b call.workspace else 0 in
      let place : F.buffer -> int * int = function
        | Operand -> (P.slot b 1, 0)
        | Dst -> (P.slot b 0, 0)
        | Workspace at -> (P.workspace b, ws + at)
      in
      Array.iter (F.emit b ~inverse:false ~place) call.launches;
      let planned = P.finish b in
      greater ~msg:"parts" int ~than:1 (Array.length planned.parts);
      let seconds (l : P.launch) =
        P.seconds
          ~threads:(Nx_metal.Kernels.threads l.kernel)
          ~groups:l.groups ~resident:l.resident l.cost
      in
      Array.iteri
        (fun i (p : P.part) ->
          let msg = Printf.sprintf "part %d" i in
          at_most ~msg:(msg ^ "'s launches") int ~than:P.part_launches
            (Array.length p.run);
          at_most ~msg:(msg ^ "'s seconds") float_exact ~than:P.part_budget
            (Array.fold_left (fun t l -> t +. seconds l) 0. p.run))
        planned.parts;
      (* A launch by its kernel, groups and words. *)
      let named (l : P.launch) =
        let x, y, z = l.groups in
        Printf.sprintf "%s %dx%dx%d %s"
          (Nx_metal.Kernels.name l.kernel)
          x y z
          (Bytes.to_string l.words |> String.escaped)
      in
      let runs = Array.map (fun (p : P.part) -> p.run) planned.parts in
      equal ~msg:"the parts' launches" (array string)
        (Array.map named planned.launches)
        (Array.map named (Array.concat (Array.to_list runs)));
      let kind : P.slot -> string = function
        | Fresh _ -> "fresh"
        | Workspace -> "workspace"
        | _ -> "other"
      in
      equal ~msg:"its workspace" string "fresh"
        (kind planned.slots.(P.workspace b))

(* At a launch budget just above a call's heaviest group, past half of it, each
   launch but a work's last holds more than [b - s_1], so the launches are at
   most the works and one for each [b - s_1] of the call's seconds. *)
let test_launches () =
  let c = rows_case C2c 8100 ~lines:3 in
  let s1 =
    match plan_of c with
    | Some call ->
        Array.fold_left (fun m l -> Float.max m (work_group l)) 0. call.launches
    | None -> fail "declined"
  in
  let b = 1.05 *. s1 in
  Fun.protect
    ~finally:(fun () -> P.bound_budget P.budget)
    (fun () ->
      P.bound_budget b;
      match plan_of c with
      | None -> fail "declined"
      | Some call ->
          less ~msg:"half the budget" float_exact ~than:s1 (b /. 2.);
          List.iter
            (fun w ->
              List.iteri
                (fun i (l : F.launch) ->
                  if i < List.length w - 1 then
                    less ~msg:"a launch's seconds" float_exact ~than:l.seconds
                      (b -. s1))
                w)
            (works call);
          at_most ~msg:"launches" int
            ~than:
              (List.length (works call)
              + Float.to_int (Float.ceil (call.seconds /. (b -. s1))))
            (Array.length call.launches))

let () =
  exit
    (run "nx_metal_fft"
       [
         group "tables"
           [
             prop "roots are the host's where its double decides the float32"
               rcase law_libm;
             cases "roots at known angles"
               ~id:(fun (c : known) -> c.name)
               known test_known;
             prop "a root's modulus is one" icase law_unit;
             prop "roots multiply as their exponents add" icase law_product;
             prop "a halfway hi rounds to lo's side" pair law_rounding;
             cases "rounding"
               ~id:(fun (n, _, _, _) -> n)
               rounding_cases test_rounding;
             test "a long line's tables are roots" test_global;
             prop "the chirp is the host's where its double decides"
               Gen.(int_range 1 5000)
               law_chirp;
             prop "Bluestein's kernel is the direct sum" bcase law_bluestein;
             cases "Bluestein's kernel at m up to 65536, at sampled bins"
               ~id:(Format.asprintf "%a" pp_bcase)
               bluestein_cases (check_bluestein sample);
             cases "Rader's tables" ~id:string_of_int small_raders
               (check_rader every);
             cases "Rader's tables at sampled bins" ~id:string_of_int
               large_raders (check_rader sample);
             cases "refusals" ~id:fst refused test_refused;
           ];
         group "steps"
           [
             cases "every 7-smooth step to 4096 points follows the rules"
               ~id:string_of_int lengths law_step;
             cases "radices the .mli states"
               ~id:(fun (l, _, _) -> string_of_int l)
               radix_cases test_radix_case;
             cases "steps the design states"
               ~id:(fun (l, _, _) -> string_of_int l)
               step_cases test_step_case;
             test "no step takes more than 288 threads" test_most_threads;
             cases "a step's chain counts its twiddles' issue"
               ~id:(fun (n, _, _, _) -> string_of_int n)
               chain_cases test_step_chain;
             cases "lines side by side"
               ~id:(fun (l, _) -> string_of_int l)
               side_cases test_side_case;
             prop "a long line's steps fit a group, at most ⌈log₂ n / 8⌉" long
               law_factors;
             cases "long lines' steps"
               ~id:(fun (n, _) -> string_of_int n)
               factor_cases test_factor_case;
             test "2^32 points take three steps" test_most_long;
             prop "a line runs as its points say"
               ~examples:[ (8192, false); (257, false) ]
               line_case law_line_case;
             test "the last 64 lines' plans are kept" test_kept;
             test "the flushing term's factor bounds every sequence of passes"
               test_flush_scale;
             test "Rader's and Bluestein's lines in a group have factors stated"
               test_conv_scales;
           ];
         group "calls"
           [
             prop ~count:2000 "a call's launches follow its sides" ccase
               law_call;
             cases "each rare path's work splits under the least budget"
               ~id:(fun (c, _) -> c.name)
               split_examples test_split_paths;
             prop ~count:300
               ~examples:(List.map fst split_examples @ split_covers)
               "under a small budget, work splits into launches" ccase law_split;
             prop
               "under a small budget, a twiddled launch's bytes are its tiles'"
               tcase law_twiddle;
             test "at 4096 columns, a launch of the last columns alone"
               test_last_columns;
             cases "a two-step line's twiddled transpose"
               ~id:(fun (n, _, _, _) -> string_of_int n)
               twiddle_cases test_twiddle;
             cases "transposes run in tiles of 32 by 32"
               ~id:(fun (name, _, _) -> name)
               tile_cases test_tiles;
             cases "Rader's lines side by side take their planes"
               ~id:(fun (n, _) -> string_of_int n)
               [ (17, 256); (257, 16) ]
               test_rader_planes;
             cases "points past 32-bit offsets take the wide transpose"
               ~id:(fun (name, _, _, ps, _) ->
                 Printf.sprintf "%s, ps %d" name ps)
               wide_cases test_wide;
             cases "columns runs are charged from their worst start"
               ~id:(fun (name, _, _, _, _) -> name)
               start_cases test_starts;
             test "a view's columns are charged from its byte of a line"
               test_view;
             cases "side by side where it costs at most the transposes"
               ~id:(fun (name, _, _, _) -> name)
               route_cases test_route;
             cases
               "the column step where a line's points lie at most 8 KiB apart"
               ~id:(fun (name, _, _, _) -> name)
               column_cases test_column;
             prop "a column step's seconds are at most 0.51 of the transposes'"
               fitting law_column_bound;
             cases "a column step's workspace and bytes"
               ~id:(fun (name, _, _, _, _, _) -> name)
               column_bytes_cases test_column_bytes;
             cases "a batch of one line is never columns"
               ~id:(fun (n, _, _) -> n)
               lone_cases test_lone;
             cases "a line plans to 2^32 points and declines past it"
               ~id:(fun (k, n, _) -> kind_name k ^ " " ^ string_of_int n)
               past_cases test_past;
             test "a declined call declines under any launch budget"
               test_declined_budget;
             test "a pre-pass of bins spanning 2^31 elements is wide"
               test_wide_prepass;
             cases "a two-step line's last transpose restores natural order"
               ~id:(fun (name, _, _) -> name)
               last_split_cases test_last_split;
             test "a two-step Bluestein line's stages end in natural order"
               test_bluestein_last_split;
             cases "a long halved line's real side moves pairs of floats"
               ~id:(fun (name, _, _, _, _) -> name)
               pair_cases test_pairs;
             cases "a long line's bins come out in natural order"
               ~id:(fun (n, _) -> string_of_int n)
               order_cases test_order;
             cases "refusals" ~id:fst c2c_refused test_refused;
             cases "a real line of 8192 points is halved in one group"
               ~id:kind_name [ R2c; C2r ] test_halved;
             test "an r2c of no point stores a bin a line" test_no_point;
             cases "real refusals" ~id:fst real_refused test_refused;
             test "a call past a part's seconds plans" test_long_call;
           ];
         group "limits"
           [
             prop "Bluestein's m does not fall as n grows"
               Gen.(int_range 1 ((1 lsl 32) - 1))
               law_m_rises;
             test "the least length taking each m" test_least_takers;
             test "n_B is the greatest length whose m is within the cut"
               test_cut;
             cases "a Bluestein line plans to n_B and declines past it"
               ~id:(fun (t, n, _) -> kind_name t ^ " " ^ string_of_int n)
               cut_cases test_cut_line;
             test "a sweep of windows gives each length's m" test_windows;
             test "a batch of five axes plans" test_rank_five;
             prop "a divisor's division is the quotient" quotient law_divisor;
             test "launches just above one group's budget" test_launches;
           ];
       ])
