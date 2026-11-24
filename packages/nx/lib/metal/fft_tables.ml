(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type bigbytes = Plan.bigbytes
type dd = { hi : float; lo : float }
type cdd = { re : dd; im : dd }

(* Double-double arithmetic: a value hi + lo, |lo| <= ulp(hi) / 2.

   Every product is an exact fma or a product bound once and used twice, never
   one an addition could absorb: ocamlopt fuses [a +. b *. c] into one rounding
   on arm64 and not on x86_64, and a table's bits must be the same on every
   host. *)

(* Error. Take each double-double addition and division to err by at most 2^-104
   relative to its result, and each product by 2^-103.

   A root: the angle t = (π/2) r / l <= π/4 errs by about 2^-102.6 absolute
   ([half_pi] within 2^-107.6, the ratio and the product), and sine and cosine
   move by no more than t does. Each series sums 15 terms into a partial sum of
   at most 1, 2^-104 each, 2^-100.1; term k comes from k products and k
   divisions, so the terms err by at most Σ_k 2k 2^-103 t^k / k! <= 2 t e^t
   2^-103 = 2^-101.2; the series stops below 2^-118. Each part errs by at most
   2^-99.3, a root by √2 2^-99.3 <= 2^-98.

   A product of two roots (the chirp, the FFT's twiddles) errs by at most 2
   2^-98 and its own arithmetic, about 2^-101: 2^-96.

   The FFT of m points, a level of radix p at a time: its twiddles, each within
   2^-96 of a number of modulus 1, err by 2^-96 relative; its p-point DFT, of
   entries from the same table, by √p 2^-96 in norm against the DFT's √p (radix
   2 adds and subtracts alone), and its additions and products by about p
   2^-102. A level of radix p thus errs by (1 + √p) 2^-96 + p 2^-102, and each
   level is a scaled unitary map, so the levels' relative errors add: per bit of
   m, at most 2^-95.2 (radix 3). With the operand's 2^-96 and the division by m,
   a kernel errs by at most (1 + log₂ m) 2^-95 relative in norm. *)

let dd x = { hi = x; lo = 0. }

let two_sum a b =
  let s = a +. b in
  let bb = s -. a in
  { hi = s; lo = a -. (s -. bb) +. (b -. bb) }

let fast_two_sum a b =
  let s = a +. b in
  { hi = s; lo = b -. (s -. a) }

let add a b =
  let s = two_sum a.hi b.hi and t = two_sum a.lo b.lo in
  let s = fast_two_sum s.hi (s.lo +. t.hi) in
  fast_two_sum s.hi (s.lo +. t.lo)

let neg a = { hi = -.a.hi; lo = -.a.lo }

(* p + e = a.hi b.hi exactly, then the cross terms. *)
let mul a b =
  let p = a.hi *. b.hi in
  let e = Float.fma a.hi b.hi (-.p) in
  fast_two_sum p (Float.fma a.hi b.lo (Float.fma a.lo b.hi e))

(* [a / d] for a float [d]: the remainder [a.hi - q d] is exact. *)
let div a d =
  let q = a.hi /. d in
  let r = Float.fma (-.q) d a.hi in
  fast_two_sum q ((r +. a.lo) /. d)

(* [a / b] of integers below 2^53. *)
let ratio a b = div (dd (Float.of_int a)) (Float.of_int b)

let cmul a b =
  {
    re = add (mul a.re b.re) (neg (mul a.im b.im));
    im = add (mul a.re b.im) (mul a.im b.re);
  }

let conj a = { a with im = neg a.im }

(* Roots of unity *)

let half_pi = { hi = 1.5707963267948966; lo = 6.123233995736766e-17 }

(* sin t and cos t for 0 <= t <= π/4 by their series: past 30 terms, t^k / k! is
   below 2^-118. *)
let sincos t =
  let rec go k term s c =
    if k = 30 then (s, c)
    else
      let s, c =
        match k mod 4 with
        | 0 -> (s, add c term)
        | 1 -> (add s term, c)
        | 2 -> (s, add c (neg term))
        | _ -> (add s (neg term), c)
      in
      go (k + 1) (div (mul term t) (Float.of_int (k + 1))) s c
  in
  go 0 (dd 1.) (dd 0.) (dd 0.)

(* w_l^k = e^(-2πi k / l), 4l <= 2^53: [4k = q l + r] by integers, the angle
   within its quarter turn taken from the nearer end, at most π/4, then turned
   by [q] quarters, so that the multiples of l / 4 are exact and the series
   converge within 30 terms. *)
let root l k =
  let k = k mod l in
  let q = 4 * k / l and r = 4 * k mod l in
  let far = 2 * r > l in
  let s, c = sincos (mul half_pi (ratio (if far then l - r else r) l)) in
  let s, c = if far then (c, s) else (s, c) in
  let cr, sr =
    match q with
    | 0 -> (c, s)
    | 1 -> (neg s, c)
    | 2 -> (neg c, neg s)
    | _ -> (s, neg c)
  in
  { re = cr; im = neg sr }

(* w_l^k for any k < l as low.(k mod 2^bits) high.(k lsr bits), the bits half of
   l's, so that a table of l roots takes about 2 √l of [root]. *)
type roots2 = { bits : int; low : cdd array; high : cdd array }

let roots2 l =
  let rec log2 b = if 1 lsl b < l then log2 (b + 1) else b in
  let bits = (log2 0 + 1) / 2 in
  let nl = 1 lsl bits in
  {
    bits;
    low = Array.init nl (root l);
    high = Array.init ((l + nl - 1) / nl) (fun i -> root l (i * nl));
  }

let at t k = cmul t.low.(k land ((1 lsl t.bits) - 1)) t.high.(k lsr t.bits)

(* Rounding *)

let least_normal = Float.ldexp 1. (-126)
let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* [hi + lo] renormalised, so that [hi] is the double nearest it, then [hi] to
   the nearest float32; where [hi] lies halfway, [g] is the other float32 and
   [lo] picks the side. A zero [lo] keeps the sign of a zero [hi], which a sum
   with [+0] would lose. *)
let round x =
  let x = if x.lo = 0. then x else two_sum x.hi x.lo in
  let f = to_f32 x.hi in
  let d = x.hi -. f in
  let f =
    if d = 0. || x.lo = 0. then f
    else
      let g = f +. (d +. d) in
      if to_f32 g = g && d > 0. = (x.lo > 0.) then g else f
  in
  if Float.abs f < least_normal then Float.copy_sign 0. f else f

let bigbytes n =
  Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n

let set32 (b : bigbytes) at v =
  for i = 0 to 3 do
    Bigarray.Array1.unsafe_set b (at + i) ((v lsr (8 * i)) land 0xff)
  done

let set_float32 b at f = set32 b at (Int32.to_int (Int32.bits_of_float f))

(* [count] numbers [number i], rounded. A part that rounds to zero, exactly or
   below the least normal float32, is stored [+0]: its sign is the arithmetic's,
   no number's, and a kernel's zero signs follow its tables'. *)
let table count number =
  let b = bigbytes (8 * count) in
  let part x =
    let f = round x in
    if f = 0. then 0. else f
  in
  for i = 0 to count - 1 do
    let w = number i in
    set_float32 b (8 * i) (part w.re);
    set_float32 b ((8 * i) + 4) (part w.im)
  done;
  b

(* Vectors of complex double-doubles, four floats a number. *)

let vec n = Float.Array.make (4 * n) 0.

let get v i =
  let a = 4 * i in
  {
    re = { hi = Float.Array.get v a; lo = Float.Array.get v (a + 1) };
    im = { hi = Float.Array.get v (a + 2); lo = Float.Array.get v (a + 3) };
  }

let set v i w =
  let a = 4 * i in
  Float.Array.set v a w.re.hi;
  Float.Array.set v (a + 1) w.re.lo;
  Float.Array.set v (a + 2) w.im.hi;
  Float.Array.set v (a + 3) w.im.lo

(* The forward DFT of a 7-smooth length, in place

   On vectors, four floats a number from [4 i]. Each operation below reads its
   operands and writes its result by index, so that no float is boxed whatever
   the compiler inlines, and computes as the operations above do, in the same
   order: the same bits. *)

module V = Float.Array

(* [d.(i)] gets [a.(j) + b.(k)], or [a.(j) - b.(k)] where [sub], as [add]. *)
let dd_add d i a j b k ~sub =
  let ah = V.unsafe_get a j and al = V.unsafe_get a (j + 1) in
  let bh = V.unsafe_get b k and bl = V.unsafe_get b (k + 1) in
  let bh = if sub then -.bh else bh and bl = if sub then -.bl else bl in
  let s = ah +. bh in
  let sb = s -. ah in
  let sl = ah -. (s -. sb) +. (bh -. sb) in
  let t = al +. bl in
  let tb = t -. al in
  let tl = al -. (t -. tb) +. (bl -. tb) in
  let u = sl +. t in
  let s2 = s +. u in
  let l2 = u -. (s2 -. s) in
  let u = l2 +. tl in
  let h = s2 +. u in
  V.unsafe_set d i h;
  V.unsafe_set d (i + 1) (u -. (h -. s2))

(* [d.(i)] gets [a.(j) b.(k)], as [mul]. *)
let dd_mul d i a j b k =
  let ah = V.unsafe_get a j and al = V.unsafe_get a (j + 1) in
  let bh = V.unsafe_get b k and bl = V.unsafe_get b (k + 1) in
  let p = ah *. bh in
  let e = Float.fma ah bh (-.p) in
  let u = Float.fma ah bl (Float.fma al bh e) in
  let h = p +. u in
  V.unsafe_set d i h;
  V.unsafe_set d (i + 1) (u -. (h -. p))

(* [d.(i)] gets [d.(i) / x], as [div]. *)
let dd_div d i x =
  let hi = V.unsafe_get d i and lo = V.unsafe_get d (i + 1) in
  let q = hi /. x in
  let r = Float.fma (-.q) x hi in
  let u = (r +. lo) /. x in
  let h = q +. u in
  V.unsafe_set d i h;
  V.unsafe_set d (i + 1) (u -. (h -. q))

(* [d.(i)] gets the complex [a.(j) b.(k)], as [cmul], through the products'
   eight floats [p], so [d] may be [a] or [b]. *)
let vmul p d i a j b k =
  dd_mul p 0 a j b k;
  dd_mul p 2 a (j + 2) b (k + 2);
  dd_mul p 4 a j b (k + 2);
  dd_mul p 6 a (j + 2) b k;
  dd_add d i p 0 p 2 ~sub:true;
  dd_add d (i + 2) p 4 p 6 ~sub:false

(* [d.(i)] gets the complex [a.(j) + b.(k)], or [-] where [sub]. *)
let vadd d i a j b k ~sub =
  dd_add d i a j b k ~sub;
  dd_add d (i + 2) a (j + 2) b (k + 2) ~sub

let vcopy d i a j =
  for c = 0 to 3 do
    V.unsafe_set d (i + c) (V.unsafe_get a (j + c))
  done

let least_factor n =
  if n mod 2 = 0 then 2
  else if n mod 3 = 0 then 3
  else if n mod 5 = 0 then 5
  else 7

(* [n]'s primes, least first, as [dft] takes them: [n = p_0 p_1 ⋯]. *)
let primes n =
  let rec go n acc =
    if n = 1 then Array.of_list (List.rev acc)
    else
      let p = least_factor n in
      go (n / p) (p :: acc)
  in
  go n []

(* [roots2 l]'s tables as vectors, and the scratch of their products. *)
type twiddles = { shift : int; lows : V.t; highs : V.t; prod : V.t }

let twiddles l =
  let t = roots2 l in
  let of_array a =
    let v = vec (Array.length a) in
    Array.iteri (fun i w -> set v i w) a;
    v
  in
  {
    shift = t.bits;
    lows = of_array t.low;
    highs = of_array t.high;
    prod = V.make 8 0.;
  }

(* [d.(i)] gets [w^k], as [at]. *)
let twiddle t d i k =
  vmul t.prod d i t.lows
    (4 * (k land ((1 lsl t.shift) - 1)))
    t.highs
    (4 * (k lsr t.shift))

(* Decimation in time by the least prime, as a recursion that takes the input [i
   + j is] for its outputs from [o]: its leaves put input [j = s_0 + s_1 p_0 +
   ⋯] at position [s_0 m_0 + s_1 m_1 + ⋯], [m_d = n / (p_0 ⋯ p_d)], each level
   then combines in place. [order] makes that order by following each cycle of
   it, a bit a number for those it has placed. *)
let order x n ps =
  let target j =
    let j = ref j and at = ref 0 and m = ref n in
    Array.iter
      (fun p ->
        m := !m / p;
        at := !at + (!j mod p * !m);
        j := !j / p)
      ps;
    !at
  in
  let placed = Bytes.make ((n + 7) / 8) '\000' in
  let mark j =
    let b = Char.code (Bytes.unsafe_get placed (j lsr 3)) in
    Bytes.unsafe_set placed (j lsr 3)
      (Char.unsafe_chr (b lor (1 lsl (j land 7))))
  in
  let is_placed j =
    Char.code (Bytes.unsafe_get placed (j lsr 3)) land (1 lsl (j land 7)) <> 0
  in
  let carry = V.make 8 0. in
  for j = 0 to n - 1 do
    if not (is_placed j) then begin
      mark j;
      vcopy carry 0 x (4 * j);
      let next = ref (target j) in
      while !next <> j do
        let k = !next in
        vcopy carry 4 x (4 * k);
        vcopy x (4 * k) carry 0;
        vcopy carry 0 carry 4;
        mark k;
        next := target k
      done;
      vcopy x (4 * j) carry 0
    end
  done

(* Level [d] of [dft] on the run of [n] from position [o]: its [p] parts, each
   transformed, combined by [p]-point DFTs whose twiddles are [w^(s k stride)]
   and whose entries are [w^(s q m stride)], [w] the root of order [nw]. *)
let rec combine x o n ps d t nw stride sc =
  if n > 1 then begin
    let p = ps.(d) in
    let m = n / p in
    for s = 0 to p - 1 do
      combine x (o + (s * m)) m ps (d + 1) t nw (stride * p) sc
    done;
    (* sc: the p terms from 0, a twiddle at 28, the sum at 32, an entry at
       36. *)
    let w = 28 and acc = 32 and e = 36 in
    for k = 0 to m - 1 do
      vcopy sc 0 x (4 * (o + k));
      for s = 1 to p - 1 do
        twiddle t sc w (s * k * stride mod nw);
        vmul t.prod sc (4 * s) x (4 * (o + (s * m) + k)) sc w
      done;
      if p = 2 then begin
        vadd x (4 * (o + k)) sc 0 sc 4 ~sub:false;
        vadd x (4 * (o + k + m)) sc 0 sc 4 ~sub:true
      end
      else
        for q = 0 to p - 1 do
          vcopy sc acc sc 0;
          for s = 1 to p - 1 do
            twiddle t sc e (s * q * m * stride mod nw);
            vmul t.prod sc e sc (4 * s) sc e;
            vadd sc acc sc acc sc e ~sub:false
          done;
          vcopy x (4 * (o + k + (m * q))) sc acc
        done
    done
  end

(* [a]'s [n] points in place by their forward DFT divided by [n]. *)
let fft_scaled a n =
  let ps = primes n in
  order a n ps;
  combine a 0 n ps 0 (twiddles n) n 1 (V.make 40 0.);
  let d = Float.of_int n in
  for i = 0 to (2 * n) - 1 do
    dd_div a (2 * i) d
  done

let to_array v n = Array.init n (get v)

(* Checks *)

let smooth n =
  let rec strip n p = if n mod p = 0 then strip (n / p) p else n in
  n >= 1 && strip (strip (strip (strip n 2) 3) 5) 7 = 1

let is_prime n =
  let rec from d = d * d > n || (n mod d <> 0 && from (d + 1)) in
  n >= 2 && from 2

let most = 1 lsl 53

let check_length name l =
  if l < 1 || l > most / 4 then
    invalid_arg (Printf.sprintf "Fft_tables.%s: length %d" name l)

(* Tables *)

let roots l ~start ~step ~count =
  check_length "roots" l;
  if start < 0 || step < 0 || count < 0 then
    invalid_arg
      (Printf.sprintf "Fft_tables.roots: start %d, step %d, count %d" start step
         count);
  let k = ref (start mod l) and dk = step mod l in
  table count (fun _ ->
      let w = root l !k in
      k := (!k + dk) mod l;
      w)

let low_bits = Kernels.fft_low_bits

let global l =
  let nl = 1 lsl low_bits in
  ( roots l ~start:0 ~step:1 ~count:nl,
    roots l ~start:0 ~step:nl ~count:((l + nl - 1) / nl) )

(* c_j = w_2n^(j² mod 2n), j < n, into [v]: j² mod 2n grows by 2j + 1 a step, so
   no product passes 64 bits. *)
let chirp_into v n =
  let t = roots2 (2 * n) in
  let q = ref 0 in
  for j = 0 to n - 1 do
    set v j (at t !q);
    q := (!q + (2 * j) + 1) mod (2 * n)
  done

let chirp n =
  if n < 1 || n > most / 8 then
    invalid_arg (Printf.sprintf "Fft_tables.chirp: %d points" n);
  let v = vec n in
  chirp_into v n;
  table n (get v)

let bluestein_vec ~n ~m =
  if n < 1 || m < (2 * n) - 1 || m > most / 4 || not (smooth m) then
    invalid_arg (Printf.sprintf "Fft_tables.bluestein: %d points at %d" n m);
  let t = roots2 (2 * n) in
  let a = vec m in
  let q = ref 0 in
  for j = 0 to n - 1 do
    let b = conj (at t !q) in
    set a j b;
    if j > 0 then set a (m - j) b;
    q := (!q + (2 * j) + 1) mod (2 * n)
  done;
  fft_scaled a m;
  a

let bluestein ~n ~m = table m (get (bluestein_vec ~n ~m))
let bluestein_dd ~n ~m = to_array (bluestein_vec ~n ~m) m

let rec powmod b e n =
  if e = 0 then 1 mod n
  else
    let h = powmod b (e / 2) n in
    let h = h * h mod n in
    if e mod 2 = 1 then h * b mod n else h

(* The least g whose powers reach every unit mod a prime [n]: g^(m/q) is not 1
   for any prime q of m = n - 1. *)
let primitive_root n =
  let m = n - 1 in
  let rec from g =
    if
      List.for_all
        (fun q -> m mod q <> 0 || powmod g (m / q) n <> 1)
        [ 2; 3; 5; 7 ]
    then g
    else from (g + 1)
  in
  from 1

let check_rader n =
  if n > 1 lsl 31 || not (is_prime n && smooth (n - 1)) then
    invalid_arg (Printf.sprintf "Fft_tables.rader: %d" n)

(* b_s = w_n^(g^-s mod n), s < m, and its transform divided by m. *)
let rader_vec n g =
  let m = n - 1 in
  let gi = powmod g (m - 1) n in
  let a = vec m in
  let p = ref 1 in
  for s = 0 to m - 1 do
    set a s (root n !p);
    p := !p * gi mod n
  done;
  fft_scaled a m;
  a

let rader n =
  check_rader n;
  let g = primitive_root n and m = n - 1 in
  let logs = bigbytes (4 * n) in
  Bigarray.Array1.fill logs 0;
  let p = ref 1 in
  for s = 0 to m - 1 do
    set32 logs (4 * !p) s;
    p := !p * g mod n
  done;
  (table m (get (rader_vec n g)), logs)

let rader_dd n =
  check_rader n;
  to_array (rader_vec n (primitive_root n)) (n - 1)
