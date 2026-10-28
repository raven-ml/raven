(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
module D = Nx_array.Dtype
module Bits = Lower_bits

(* A complex value as its two parts, nodes of its shape. *)
type z = { re : Ops.t; im : Ops.t }

let complex = D.Any D.Complex128
let map f z = { re = f z.re; im = f z.im }
let dims s = List.map (fun n -> Ops.Int n) s
let extents u = List.init (Shape.ndim u) (Bits.extent u)
let int u n = Shape.const_like u (`Int (Bigint.of_int n))
let zero u = Shape.const_like u (`Float 0.)

let parts u =
  match Lower_arith.parts complex u with
  | [ re; im ] -> { re; im }
  | _ -> assert false

let of_parts z = Lower_arith.of_parts complex [ z.re; z.im ]

let precision op u =
  match (Ops.dtype u : Dtype.t) with
  | Float32 | Float64 -> Ops.dtype u
  | _ ->
      invalid_arg
        (Printf.sprintf
           "Rune.Lower_fft: a %s of another precision is not lowered" op)

(* [only u axis b] is [b] at [axis] of [u]'s axes, [None] elsewhere. *)
let only u axis b =
  List.init (Shape.ndim u) (fun i -> if i = axis then Some b else None)

let cut u axis lo hi = Shape.shrink u (only u axis (Ops.Int lo, Ops.Int hi))

(* [lay u at target] is [u] with its axis [k] at axis [at.(k)] of the extents
   [target], expanded along the others. *)
let lay u at target =
  let order =
    List.sort
      (fun i j -> Int.compare (List.nth at i) (List.nth at j))
      (List.init (List.length at) Fun.id)
  in
  let s = List.mapi (fun i e -> if List.mem i at then e else 1) target in
  Shape.expand (Shape.reshape (Shape.permute u order) (dims s)) (dims target)

(* [index n] is an integer type that holds [4 n]. *)
let index n : Dtype.t = if n < 1 lsl 29 then Int32 else Int64

let rec least_factor n p =
  if p * p > n then n else if n mod p = 0 then p else least_factor n (p + 1)

(* Twiddles

   [twiddle t ~sign n m] is [e^(sign 2πi m / n)] at each element of the
   integer node [m], of values in [0, n), as floats of type [t]. The quarter
   turn [q] and the remainder [s] of [4m = q n + s] split the angle into [q
   π/2] and [φ = (π/2) s / n]. [sin φ] and [cos φ] are the sines of [(π/2) l /
   n] and [(π/2) (n - l) / n], [l = min s (n - s)]: arguments of at most
   [π/2], and sines, which {!Lower_real} holds closer than cosines. A whole
   quarter turn, [s = 0], is exact. *)

let twiddle t ~sign n m =
  let at = Ops.cast m (index n) in
  let n' = int at n in
  let four = Ops.mul at (int at 4) in
  let q = Ops.div ~rounding:`Floor four n' in
  let s = Ops.sub four (Ops.mul q n') in
  let l = Ops.minimum s (Ops.sub n' s) in
  let step = Float.pi /. 2. /. Float.of_int n in
  let sine k =
    let k = Ops.cast k t in
    let x = Ops.mul k (Shape.const_like k (`Float step)) in
    Lower_real.unary Nx_kernel.Prog.Sin x
  in
  let exact = Ops.eq l (int l 0) in
  let small = sine l and big = sine (Ops.sub n' l) in
  let small = Ops.where exact (zero small) small in
  let big = Ops.where exact (Shape.const_like big (`Float 1.)) big in
  let low = Ops.le (Ops.add s s) n' in
  let sn = Ops.where low small big and cs = Ops.where low big small in
  let quarter k = Ops.eq q (int q k) in
  let turn a b c d =
    Ops.where (quarter 0) a
      (Ops.where (quarter 1) b (Ops.where (quarter 2) c d))
  in
  let re = turn cs (Ops.neg sn) (Ops.neg cs) sn in
  let im = turn sn cs (Ops.neg sn) (Ops.neg cs) in
  { re; im = (if sign < 0 then Ops.neg im else im) }

(* [table t ~sign n p m] is the twiddles [w^(i j)] of [n] points for [i < p]
   and [j < m], [i j < n] or reduced modulo [n], computed once. *)
let table t ~sign n p m =
  let ty = index (Int.max n (p * m)) in
  let i = lay (Shape.arange ~dtype:ty p) [ 0 ] [ p; m ] in
  let j = lay (Shape.arange ~dtype:ty m) [ 1 ] [ p; m ] in
  let e = Ops.mul i j in
  let e = if (p - 1) * (m - 1) < n then e else Ops.mod_ e (int e n) in
  map Ops.contiguous (twiddle t ~sign n e)

let mul a b =
  {
    re = Ops.sub (Ops.mul a.re b.re) (Ops.mul a.im b.im);
    im = Ops.add (Ops.mul a.re b.im) (Ops.mul a.im b.re);
  }

(* Lines

   A line of [n = p m] points, [p] its least prime factor, is [p]-point
   transforms of the [m] interleaved lines [x[m j1 + j2]], a twiddle [w^(j2
   k1)], then [m]-point transforms along [j2] (Cooley and Tukey, "An
   algorithm for the machine calculation of complex Fourier series", 1965):

   {v
     y[k1 + p k2] = Σ_j2 w_m^(j2 k2) w_n^(j2 k1) Σ_j1 x[m j1 + j2] w_p^(j1 k1)
   v}

   A 2-point transform is [z0 + z1] and [z0 - z1]; another, up to
   [bluestein_prime], is a sum over the [p × p] table of [w_p^(j k)]. Each is
   contiguous, so that tolk realises it, as an inlined one would read every
   point of the line per result: one kernel per prime factor of [n], each of
   [O(p)] work per point.

   Above [bluestein_prime], whose table and sums grow as [p²], a [p]-point
   transform is a convolution (Bluestein, "A linear filtering approach to the
   computation of discrete Fourier transform", 1970). With [c[j] = e^(sign πi
   j² / p)], [j k = (j² + k² - (k - j)²) / 2] gives

   {v
     y[k] = c[k] Σ_j (x[j] c[j]) conj c[k - j]
   v}

   a convolution with [conj c] over [-p < k - j < p], computed cyclically over
   [m >= 2p - 1] points, a power of two: the inverse transform of the product
   of the transforms of [x c], zero past [p], and of [conj c] wrapped at [m],
   over [m]. Three transforms of [m] points, [O(log p)] work per point. *)

let bluestein_prime = 250

(* [blue_points p] is the least power of two at least [2p - 1]. *)
let blue_points p =
  let rec up m = if m >= (2 * p) - 1 then m else up (2 * m) in
  up 1

(* [chirp t ~sign p j] is [e^(sign πi j² / p)], [w_2p^(j² mod 2p)], at each
   element of the integer node [j], of values below [2p]. *)
let chirp t ~sign p j =
  let j = Ops.cast j (index (p * p)) in
  twiddle t ~sign (2 * p) (Ops.mod_ (Ops.mul j j) (int j (2 * p)))

let butterfly ax z =
  let s = extents z.re in
  let half i u = Shape.expand (cut u ax i (i + 1)) (dims s) in
  let first = lay (Shape.arange 2) [ ax ] s in
  let first = Ops.eq first (int first 0) in
  let f u =
    let x1 = half 1 u in
    Ops.add (half 0 u) (Ops.where first x1 (Ops.neg x1))
  in
  map f z

let dense t ~sign ax z =
  let s = extents z.re in
  let p = List.nth s ax in
  let target =
    List.concat (List.mapi (fun i e -> if i = ax then [ p; p ] else [ e ]) s)
  in
  let r = List.length s in
  let at = List.init r (fun i -> if i < ax then i else i + 1) in
  let z = map (fun u -> lay u at target) z in
  let w = map (fun u -> lay u [ ax; ax + 1 ] target) (table t ~sign p p p) in
  map (fun u -> Shape.rop u Op.Add [ ax + 1 ]) (mul z w)

let rec line t ~sign ax z =
  let s = extents z.re in
  let n = List.nth s ax in
  let p = least_factor n 2 in
  let stage z =
    map Ops.contiguous
      (if p = 2 then butterfly ax z
       else if p <= bluestein_prime then dense t ~sign ax z
       else bluestein t ~sign ax z)
  in
  if n <= 1 then z
  else if p = n then stage z
  else
    let m = n / p in
    let split =
      List.concat (List.mapi (fun i e -> if i = ax then [ p; m ] else [ e ]) s)
    in
    let z = stage (map (fun u -> Shape.reshape u (dims split)) z) in
    let w = map (fun u -> lay u [ ax; ax + 1 ] split) (table t ~sign n p m) in
    let z = line t ~sign (ax + 1) (mul z w) in
    let swap =
      List.init (List.length split) (fun i ->
          if i = ax then ax + 1 else if i = ax + 1 then ax else i)
    in
    map (fun u -> Shape.reshape (Shape.permute u swap) (dims s)) z

(* The kernel, [conj c[l]] at [i = l] and [i = m - l] for [l < p] and zero
   between, is [conj c] at [min i (m - i)] where that is below [p]: the two
   ranges meet nowhere as [m >= 2p - 1]. Its transform, over [m] exactly, has
   no batch axis; tolk computes it once for every line. *)
and bluestein t ~sign ax z =
  let s = extents z.re in
  let p = List.nth s ax in
  let m = blue_points p in
  let c = map Ops.contiguous (chirp t ~sign p (Shape.arange p)) in
  let at u = lay u [ ax ] s in
  let a = mul z (map at c) in
  let pad u =
    Shape.pad ~value:(`Float 0.) u (only u ax (Ops.Int 0, Ops.Int (m - p)))
  in
  let a = line t ~sign:(-1) ax (map pad a) in
  let i = Shape.arange m in
  let l = Ops.minimum i (Ops.sub (int i m) i) in
  let inside = Ops.lt l (int l p) in
  let b = chirp t ~sign p l in
  let b = { re = b.re; im = Ops.neg b.im } in
  let b = map (fun u -> Ops.where inside u (zero u)) b in
  let inv = 1. /. Float.of_int m in
  let over u = Ops.mul u (Shape.const_like u (`Float inv)) in
  let k = map over (line t ~sign:(-1) 0 b) in
  let sm = List.mapi (fun i e -> if i = ax then m else e) s in
  let k = map (fun u -> lay u [ ax ] sm) k in
  let v = line t ~sign:1 ax (mul a k) in
  mul (map (fun u -> cut u ax 0 p) v) (map at c)

(* Error bound

   The interface's error bound follows from each step's, in the unit roundoff
   [u] of the parts and [γ] as [Nx_kernel] defines it.

   Twiddles. A twiddle [w] is within [12 u] of [e^(±2πi k/n)], and exact at
   whole quarter turns. Its parts are the sines [Lower_real] states within 4
   ulps of arguments of at most [π/2], which carry at most [4 u] of relative
   error. A twiddle's product errs by at most [√2 γ(2) + 12 u <= 15 u] relative.

   Small primes. A [p]-point transform of the points [z] scales their norm by
   [√p] exactly, and is within [c(p) u] of its result's norm. A [2]-point
   transform rounds each part once, [u] relative to its result. One by the table
   is within [(√2 (p + 2) u + 12 u) Σ|z|] of each result.

   Bluestein. The chirp's product errs as a twiddle's does, by [15 u ‖z‖₂] in
   norm, and the forward transform, of norm [√m ‖z‖₂], adds [e(m) u] of it. The
   kernel [K], the transform of the [2p - 1] chirp values over [m], is at most
   [(2p - 1) / m <= 1] in magnitude, and within [(e(m) + 12) u √(2p - 1) / √m]
   of it in norm. That moves the product by [(e(m) + 12) u √p ‖z‖₂], since a
   transform's entries are at most [√p ‖z‖₂], and the product rounds within [3
   u]. The inverse transform multiplies these errors by [√m] and adds [e(m) u m
   ‖z‖₂], and the last chirp adds [15 u √p ‖z‖₂], against a result of norm [√p
   ‖z‖₂]. *)

(* Transforms *)

let sign : Nx_kernel.direction -> int = function Forward -> -1 | Inverse -> 1

let c2c d ~axes u =
  let t = precision "C2c" u in
  let along z ax = line t ~sign:(sign d) ax z in
  of_parts (List.fold_left along (parts u) axes)

let split axes =
  match List.rev axes with
  | last :: rest -> (List.rev rest, last)
  | [] -> invalid_arg "Rune.Lower_fft: a transform along no axis"

(* The last axis first, so that the others transform its [n/2 + 1] bins. *)
let r2c ~axes u =
  let t = precision "R2c" u in
  let others, last = split axes in
  let s = extents u in
  let n = List.nth s last in
  if n = 0 then
    let s = List.mapi (fun i e -> if i = last then 1 else e) s in
    let ones = List.map (fun _ -> Ops.Int 1) s in
    let zero = Shape.reshape (Ops.const ~dtype:t (`Float 0.)) ones in
    let zeros = Shape.expand zero (dims s) in
    of_parts { re = zeros; im = zeros }
  else
    let z = line t ~sign:(-1) last { re = u; im = zero u } in
    let z = map (fun u -> cut u last 0 ((n / 2) + 1)) z in
    of_parts (List.fold_left (fun z ax -> line t ~sign:(-1) ax z) z others)

(* The bins along the last axis completed to [n]: [X[k]] up to [n/2], bin [0]'s
   and an even [n]'s bin [n/2]'s imaginary parts zero, then [conj X[n - k]]. *)
let c2r ~axes ~n u =
  let t = precision "C2r" u in
  let others, last = split axes in
  let z = List.fold_left (fun z ax -> line t ~sign:1 ax z) (parts u) others in
  if n = 0 then cut z.re last 0 0
  else
    let s = extents z.re in
    let half = n / 2 in
    let k = lay (Shape.arange (half + 1)) [ last ] s in
    let real = Ops.eq k (int k 0) in
    let real =
      if n mod 2 = 0 then Ops.bitwise_or real (Ops.eq k (int k half)) else real
    in
    let z = { z with im = Ops.where real (zero z.im) z.im } in
    let tail = n - half - 1 in
    let z =
      if tail = 0 then z
      else
        let mirror u = Shape.flip (cut u last 1 (tail + 1)) [ last ] in
        {
          re = Shape.cat ~axis:last z.re [ mirror z.re ];
          im = Shape.cat ~axis:last z.im [ Ops.neg (mirror z.im) ];
        }
    in
    (line t ~sign:1 last z).re
