(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
module D = Nx_array.Dtype
module Arith = Lower_arith
module Bits = Lower_bits

type monoid = Nx_kernel.monoid = Sum | Prod | Max | Min

let refuse m ~scan (D.Any dt) =
  match (m, dt) with
  | Prod, (D.Complex64 | D.Complex128) ->
      raise
        (Arith.Unlowered "Rune.Lower_reduce: a complex Prod is not lowered")
  | (Max | Min), D.Complex128 when scan ->
      raise
        (Arith.Unlowered
           "Rune.Lower_reduce: a complex128 Max or Min scan is not lowered")
  | _ -> ()

let extents u = List.init (Shape.ndim u) (Bits.extent u)

let dims s = List.map (fun e -> Ops.Int e) s
let int u n = Shape.const_like u (`Int (Bigint.of_int n))
let bit k = `Int (Bigint.shift_left Bigint.one k)

(* [any p us] is the elementwise or of [p] over the nodes [us]. *)
let any p us =
  List.fold_left
    (fun a u -> Ops.bitwise_or a (p u))
    (p (List.hd us))
    (List.tl us)

(* Integers fold on unsigned patterns of 32 or 64 bits: C, Metal and CUDA leave
   signed overflow undefined and promote narrower types to [int], where [65535 *
   65535] overflows. The pattern wraps as nx's integers do, and [store] keeps
   its low bits by the unsigned type of the integer's width. *)
let folded t = if Dtype.itemsize t < 4 then Dtype.Uint32 else Bits.unsigned t
let lift u = Ops.cast u (folded (Ops.dtype u))
let store t r = Ops.bitcast (Ops.cast r (Bits.unsigned t)) t

(* [back u axes m] is the result [m] of a reduction of [u] along [axes],
   expanded to [u]'s shape. *)
let back u axes m =
  let s = extents u in
  let kept = List.mapi (fun i e -> if List.mem i axes then 1 else e) s in
  Shape.expand (Shape.reshape m (dims kept)) (dims s)

(* [along u axes v] is the vector [v] laid over [u]'s [axes], C order,
   expanded to [u]'s shape. *)
let along u axes v =
  let s = extents u in
  let over = List.mapi (fun i e -> if List.mem i axes then e else 1) s in
  Shape.expand (Shape.reshape v (dims over)) (dims s)

(* [only u axis b] is [b] at [axis] of [u]'s axes, [None] elsewhere. *)
let only u axis b =
  List.init (Shape.ndim u) (fun i -> if i = axis then Some b else None)

(* [padded ~value axis before after u] is [u] with [before] and [after]
   elements of [value] added along [axis], or removed where negative. *)
let padded ?value axis before after u =
  Shape.pad ?value u (only u axis (Ops.Int before, Ops.Int after))

(* [positions ~dtype u axes n] is [n - p] at each element of [u], [p] its
   position in C order of the indices along [axes], [n] their count: [n] down to
   [1] laid over [axes]. *)
let positions ~dtype u axes n =
  along u axes (Shape.arange ~start:n ~step:(-1) ~dtype 0)

(* Sums and products

   [arith f op u] folds the node [u] by [op]: a reduction or a scan. A float sum
   adds [+0] to its result: tolk starts a loop's accumulator from [+0], but sums
   the terms alone where no loop is left, so a sum of [-0] terms would stay
   [-0]. *)

let arith f op u =
  let t = Ops.dtype u in
  if not (Dtype.is_float t) then store t (f op (lift u))
  else
    let r = f op u in
    if Op.equal op Op.Add then Ops.add r (Shape.const_like r (`Float 0.)) else r

(* Extremes

   A float's order key is the signed integer of its bits, a negative float's
   magnitude bits flipped: the greater float has the greater key and [-0] is
   just below [+0]. The key is its own inverse. A minimum is the maximum of the
   complemented keys. NaN terms take the greatest key, which no number takes
   and which reads back as a NaN, so an extreme with a NaN term is a NaN.

   A complex number orders by its real part, then its imaginary part. With
   float32 parts, each key biased to unsigned, [k xor min_int], packs the real
   key over the imaginary one in a uint64, whose Max is the extreme: NaN terms
   take the greatest, both of whose parts read back as NaNs. With float64
   parts, a REDUCE Max of the real keys, then of the imaginary keys of the
   terms whose real key is it, NaN terms aside: a scan cannot take that second
   Max, and refuses. *)

let order k =
  let flipped = Shape.const_like k (Dtype.max (Ops.dtype k) :> Dtype.const) in
  Ops.where (Ops.lt k (int k 0)) (Ops.bitwise_xor k flipped) k

let least k = Shape.const_like k (Dtype.min (Ops.dtype k) :> Dtype.const)
let most k = Shape.const_like k (Dtype.max (Ops.dtype k) :> Dtype.const)

let extreme f axes ~greatest ~nan xs =
  let flip k = if greatest then k else Ops.bitwise_not k in
  let key u = flip (order (Ops.bitcast u (Bits.signed (Ops.dtype u)))) in
  let value t k = Ops.bitcast (order (flip k)) t in
  match xs with
  | [ x ] when not (Dtype.is_float (Ops.dtype x)) ->
      [ flip (f Op.Max (flip x)) ]
  | [ x ] ->
      let k = key x in
      [ value (Ops.dtype x) (f Op.Max (Ops.where nan (most k) k)) ]
  | [ re; im ] when Dtype.itemsize (Ops.dtype re) = 4 ->
      let biased k =
        Ops.cast (Ops.bitcast (Ops.bitwise_xor k (least k)) Uint32) Uint64
      in
      let kr = biased (key re) and ki = biased (key im) in
      let packed = Ops.bitwise_or (Ops.shl kr (int kr 32)) ki in
      let m = f Op.Max (Ops.where nan (most packed) packed) in
      let unbiased u =
        let k = Ops.bitcast (Ops.cast u Uint32) Int32 in
        Ops.bitwise_xor k (least k)
      in
      [
        value Float32 (unbiased (Ops.shr m (int m 32)));
        value Float32 (unbiased m);
      ]
  | [ re; im ] ->
      let kr = key re and ki = key im in
      let mr = f Op.Max (Ops.where nan (most kr) kr) in
      let top =
        Ops.bitwise_and (Ops.logical_not nan) (Ops.eq kr (back re axes mr))
      in
      let mi = f Op.Max (Ops.where top ki (least ki)) in
      [ value Float64 mr; value Float64 mi ]
  | _ -> invalid_arg "Lower_reduce: a value of another dtype"

(* [fold m f axes xs] folds the parts [xs] of a value of a carrier by [m] with
   [f] along [axes]. *)
let fold m f axes xs =
  match m with
  | Sum -> List.map (arith f Op.Add) xs
  | Prod -> List.map (arith f Op.Mul) xs
  | Max | Min -> extreme f axes ~greatest:(m = Max) ~nan:(any Bits.is_nan xs) xs

(* [carried ~src c x] is [x] in the carrier [c]; a boolean as the byte of its
   truth. *)
let carried ~src (D.Any dt as c) x =
  if D.is D.Boolean dt then Ops.cast (Ops.ne x (int x 0)) Uint8
  else Arith.cast ~src c x

let of_extreme : Nx_kernel.extreme -> monoid = function
  | Max -> Max
  | Min -> Min

(* The number of terms of [u] along [axes]. *)
let terms u axes = List.fold_left (fun n a -> n * Bits.extent u a) 1 axes

let num u v = Shape.const_like u (`Float v)
let finite u = Ops.lt (Bits.magnitude u) (num u Float.infinity)
let shift m = Ops.where (finite m) m (num m 0.)
let ( /: ) x y = Ops.alu x Op.Fdiv [ y ]
let exp = Lower_real.unary Exp
let log = Lower_real.unary Log

(* Logsumexp

   [c + log Σ exp (x - c)], [c] the maximum [m] where finite and [0] elsewhere,
   so that no term is shifted by an infinity. The sum is [0] for no term or
   terms all [-∞], [+∞] with a [+∞] term and a NaN with a NaN term, so the
   result is [-∞], [+∞] or a NaN there.

   For the bound in [lower_reduce.mli], an error of [E] ulps is a relative [(2E
   + 1) u]. [s] takes [(n - 1) u] from its sum, [(2E + 1) u] from [exp], and at
   most [(n - 1) u / e] from the rounding of the shifts [x - c], as [|x - c| exp
   (x - c) <= 1/e], so [1 + 1/e < 1.37]; its log, of a sum within [[1, n]], adds
   [(2L + 1) u ln n]. *)
let logsumexp f axes x =
  let c = shift (List.hd (fold Max f axes [ x ])) in
  let s = List.hd (fold Sum f axes [ exp (Ops.sub x (back x axes c)) ]) in
  Ops.add c (log s)

(* Moments

   The mean [Σ x / n], then the variance by the corrected two-pass algorithm
   (Chan, Golub and LeVeque, "Algorithms for computing the sample variance",
   1983): [(Σ d² - Σ d (Σ d / n)) / n], [d = x - mean]. [Σ (x - c)² - (Σ (x -
   c))² / n] is [Σ (x - μ)²] for any [c], so the mean's rounding reaches the
   variance at second order. [Σ d (Σ d / n)] is at most [Σ d²], so it overflows
   only where [Σ d²] does, which gives [+∞]. Where rounding makes the difference
   negative, the variance is [+0]. A NaN mean makes every [d] a NaN. Since the
   correction removes the mean's error to second order, the mean's bound [B]
   enters the variance's only through its products with [u]. *)
let moments f axes n x =
  let sum u = List.hd (fold Sum f axes [ u ]) in
  let s = sum x in
  let n = num s (Float.of_int n) in
  let mean = s /: n in
  let d = Ops.sub x (back x axes mean) in
  let s1 = sum d and s2 = sum (Ops.mul d d) in
  let q = Ops.sub s2 (Ops.mul s1 (s1 /: n)) /: n in
  let var =
    Ops.where
      (Ops.eq s2 (num s2 Float.infinity))
      s2
      (Ops.where (Ops.lt q (num q 0.)) (num q 0.) q)
  in
  (mean, var)

(* Arg

   A term matches the extreme where each of its parts has the extreme's bits,
   or where it has a NaN part and the extreme is a NaN. *)

let same x y =
  if Dtype.is_float (Ops.dtype x) then Ops.eq (Bits.bits x) (Bits.bits y)
  else Ops.eq x y

let all p us =
  List.fold_left
    (fun a u -> Ops.bitwise_and a (p u))
    (p (List.hd us))
    (List.tl us)

(* [position axes xs m] is the int64 position, in C order of the indices along
   [axes], of the first term of the parts [xs] matching the extreme [m]: [n - k]
   for the greatest key [k] of a match, keys [n] down to [1]. *)
let position axes xs m =
  let u = List.hd xs in
  let n = terms u axes in
  let m' = List.map (back u axes) m in
  let found = all (fun (x, y) -> same x y) (List.combine xs m') in
  let found =
    if not (Dtype.is_float (Ops.dtype u)) then found
    else
      Ops.bitwise_or found
        (Ops.bitwise_and (any Bits.is_nan xs) (any Bits.is_nan m'))
  in
  let key = positions ~dtype:Int64 u axes n in
  let k = Shape.rop (Ops.where found key (int key 0)) Op.Max axes in
  Ops.sub (int k n) k

(* [changes axis ys] is, at each position [j] along [axis] of the running
   extreme [ys], the position of its first term matching it: the running
   maximum of the positions where it changes its bits, as the extreme changes
   only at a term that matches no term before it. The extreme at [-1] is [0],
   whose change at [0] gives [0] anyway. *)
let changes axis ys =
  let u = List.hd ys in
  let n = Bits.extent u axis in
  let before y = padded axis 0 (-1) (padded axis 1 0 y) in
  let changed = any (fun y -> Ops.logical_not (same y (before y))) ys in
  let j = along u [ axis ] (Shape.arange ~dtype:Int64 n) in
  Shape.cumalu (Ops.where changed j (int j 0)) axis Op.Max

let reduce r axes ~src dt x =
  let c = Nx_kernel.carrier src in
  let f op u = Shape.rop u op axes in
  let xs = Arith.parts c (carried ~src c x) in
  let out ys = Arith.cast ~src:c dt (Arith.of_parts c ys) in
  match (r : Nx_kernel.reduction) with
  | Monoid m ->
      refuse m ~scan:false src;
      [ out (fold m f axes xs) ]
  | Logsumexp -> [ out [ logsumexp f axes (List.hd xs) ] ]
  | Moments ->
      let x = List.hd xs in
      let mean, var = moments f axes (terms x axes) x in
      [ out [ mean ]; out [ var ] ]
  | Arg e ->
      let m = fold (of_extreme e) f axes xs in
      [ out m; position axes xs m ]

(* Scans *)

(* [join axis k x] is [k] then [x] along [axis]. Shape.cat adds zero-padded
   pieces, which would turn a [-0] into [+0]: floats join as their bits. *)
let join axis k x =
  let cat k x = Shape.cat ~axis k [ x ] in
  match Ops.dtype x with
  | t when Dtype.is_float t ->
      let b = Bits.unsigned t in
      Ops.bitcast (cat (Ops.bitcast k b) (Ops.bitcast x b)) t
  | _ -> cat k x

(* [negative_zeros f x y] is the running float sum [y] of [x], [-0] where every
   term so far, the carry's included, is [-0], as a sum started from a [-0]
   carry gives. Cost: a running Max over bytes. *)
let negative_zeros f x y =
  let t = Ops.dtype x in
  if not (Dtype.is_float t) then y
  else
    let bits = Ops.bitcast x (Bits.unsigned t) in
    let sign = Shape.const_like bits (bit ((8 * Dtype.itemsize t) - 1)) in
    let other = f Op.Max (Ops.cast (Ops.ne bits sign) Uint8) in
    let zero = Shape.const_like y (`Float (-0.)) in
    Ops.where (Ops.eq other (int other 0)) zero y

(* A carry is the term before each slice's first: the scan runs over it, then
   drops its result. *)
let running m axis ~carry c xs =
  let f op u = Shape.cumalu u axis op in
  match carry with
  | None -> fold m f [ axis ] xs
  | Some k ->
      let joined =
        List.map2 (join axis) (Arith.parts c (carried ~src:c c k)) xs
      in
      let ys = fold m f [ axis ] joined in
      let ys = if m = Sum then List.map2 (negative_zeros f) joined ys else ys in
      List.map (padded axis (-1) 0) ys

(* [network axis combine xs] is the inclusive scan along [axis] of the states
   whose components are the nodes [xs], by Brent and Kung's network, as
   {!Nx_kernel.section-scans} states it: neighbours [2j] and [2j + 1] combine
   into pairs, the pairs scan, result [0] is state [0], result [2j + 1] is the
   pairs' result [j], and result [2j], for [j > 0], the pairs' result [j - 1]
   combined with state [2j]. [combine a b] combines the states [a] then [b].
   Each level is contiguous, so that tolk realises it, as an inlined level
   would read up to [n] states per result: [2 log₂ n] kernels of [O(n)] work
   in all. *)
let rec network axis combine xs =
  let m = Bits.extent (List.hd xs) axis in
  if m <= 1 then xs
  else
    let half = m / 2 and rows = (m + 1) / 2 in
    (* The states [2j + k], [j < n]. *)
    let pick k n u =
      let u = padded axis 0 ((2 * n) - m) u in
      let u = Shape.unflatten u axis [ Ops.Int n; Ops.Int 2 ] in
      let u = padded (axis + 1) (-k) (k - 1) u in
      Shape.squeeze ~axis:(axis + 1) u
    in
    let p = combine (List.map (pick 0 half) xs) (List.map (pick 1 half) xs) in
    let q = network axis combine (List.map Ops.contiguous p) in
    let before = List.map (padded axis 1 (rows - half - 1)) q in
    let states = List.map (pick 0 rows) xs in
    (* Result [0] is state [0] itself: a Moments state of count [0] combined
       first would multiply an overflowing [δ²] by [0]. *)
    let u = List.hd states in
    let j = along u [ axis ] (Shape.arange ~dtype:Int32 rows) in
    let first = Ops.eq j (int j 0) in
    let evens = List.map2 (Ops.where first) states (combine before states) in
    let interleave e o =
      let o = padded axis 0 (rows - half) o in
      let y = Shape.stack ~axis:(axis + 1) [ e; o ] in
      let y = Shape.flatten ~start:axis ~stop:(axis + 1) y in
      Ops.contiguous (padded axis 0 (m - (2 * rows)) y)
    in
    List.map2 interleave evens q

(* A Logsumexp state is a maximum [m] and a sum [s] of [exp (x - c)], [c] being
   [m] where finite and [0] elsewhere; a term [x] is [(x, exp (x - c))].

   For the scan's bound in [lower_reduce.mli], a combination rescales each sum
   by an [exp], within [(2E + 1) u], or by [exp 0 = 1] exactly, multiplies,
   within [u], and adds, within [u]. Along a term's path, the rescales'
   arguments round within [u (X - x)], where [X] is the terms' maximum, which
   moves [s] by a relative [u ln N] at most. [c + log s] adds the [log]'s [(2L +
   1) u ln N] and its own rounding. *)
let lse_terms x = [ x; exp (Ops.sub x (shift x)) ]

let lse_combine a b =
  match (a, b) with
  | [ ma; sa ], [ mb; sb ] ->
      let m = Ops.maximum ma mb in
      let c = shift m in
      let e u = exp (Ops.sub u c) in
      [ m; Ops.add (Ops.mul sa (e ma)) (Ops.mul sb (e mb)) ]
  | _ -> invalid_arg "Lower_reduce: a state of another arity"

(* A Moments state is a mean [μ] of terms shifted by [c], a sum [S] of squared
   deviations and a count [n], combined by Chan, Golub and LeVeque's (1.5b), the
   mean weighted; [c] is the slice's first term where finite, else [0], and a
   term [x] is [(x - c, 0, 1)].

   For the scan's bound in [lower_reduce.mli], a combination's mean errs by its
   states' largest error and [3 u R] at most, its sum by a relative [7 u], and
   the means' errors reach the variance through each [δ²], [δ] the difference of
   the states' means. At each depth the Cauchy-Schwarz inequality bounds their
   weighted sum by [N √V / 2]. *)
let moments_combine a b =
  match (a, b) with
  | [ ma; sa; na ], [ mb; sb; nb ] ->
      let n = Ops.add na nb in
      let wa = na /: n and wb = nb /: n in
      let d = Ops.sub mb ma in
      let between = Ops.mul (Ops.mul d d) (Ops.mul na wb) in
      let s = Ops.add (Ops.add sa sb) between in
      [ Ops.add (Ops.mul wa ma) (Ops.mul wb mb); s; n ]
  | _ -> invalid_arg "Lower_reduce: a state of another arity"

let scan r axis ~carry ~src dt x =
  let c = Nx_kernel.carrier src in
  let xs = Arith.parts c (carried ~src c x) in
  let out ys = Arith.cast ~src:c dt (Arith.of_parts c ys) in
  let x = List.hd xs in
  match (r : Nx_kernel.reduction) with
  | Monoid m ->
      refuse m ~scan:true src;
      [ out (running m axis ~carry c xs) ]
  | Arg e ->
      (* The running extreme by its Max or Min scan, and each position as the
         last position up to it where the extreme changes. *)
      let m = of_extreme e in
      refuse m ~scan:true src;
      let ys = running m axis ~carry:None c xs in
      [ out ys; changes axis ys ]
  | Logsumexp ->
      let ms = network axis lse_combine (lse_terms x) in
      let m = List.hd ms and s = List.nth ms 1 in
      [ out [ Ops.add (shift m) (log s) ] ]
  | Moments ->
      let first = padded axis 0 (1 - Bits.extent x axis) x in
      let c = shift (Shape.expand first (dims (extents x))) in
      let terms = [ Ops.sub x c; num x 0.; num x 1. ] in
      let ms = network axis moments_combine terms in
      let mean = List.hd ms and s = List.nth ms 1 and n = List.nth ms 2 in
      let var = Ops.where (finite mean) (s /: n) (num s Float.nan) in
      [ out [ Ops.add c mean ]; out [ var ] ]

(* Contractions

   Each operand is laid over the axes [B, I, J, K]: its batch axes in pair
   order, [a]'s free axes, [b]'s, its contracted axes in pair order; [a] of
   extent [1] along [J] and [b] along [I], expanded. A REDUCE Add over [K] sums
   the products; tolk lays out the reduce of a product itself. A complex product
   is [(ar br - ai bi) + i (ar bi + ai br)].

   For the bound in [lower_reduce.mli], tolk may fuse a product into the sum
   that takes it, and an element times its scale rounds once, so a float result
   takes [K + 1 + s] roundings, each within [2u] where a matrix unit truncates
   its sums. A complex part takes two products and a difference or a sum per
   term, so [γ(K + 2)]. *)

type scale = Exponent of Ops.t | Value of D.any * Ops.t
type operand = { dtype : D.any; node : Ops.t; scale : scale option }

let free r named =
  List.filter (fun i -> not (List.mem i named)) (List.init r Fun.id)

(* [laid u s order ~at ~ones target] is [u], of extents [s], with its axes in
   [order], [ones] axes of extent [1] inserted before axis [at], expanded to
   [target]. *)
let laid u s order ~at ~ones target =
  let s = List.map (List.nth s) order in
  let before = List.filteri (fun i _ -> i < at) s in
  let after = List.filteri (fun i _ -> i >= at) s in
  let s = before @ List.init ones (fun _ -> 1) @ after in
  Shape.expand (Shape.reshape (Shape.permute u order) (dims s)) (dims target)

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

(* A sub-byte accumulator's node holds a byte, which wraps to its width. *)
let wrap acc r =
  match acc with
  | D.Any D.Int4 -> Arith.cast ~src:(D.Any D.Int8) acc r
  | D.Any D.Uint4 -> Arith.cast ~src:(D.Any D.Uint8) acc r
  | _ -> r

let zero u =
  Shape.const_like u
    (if Dtype.is_float (Ops.dtype u) then `Float 0. else `Int Bigint.zero)

(* [products ~batch ~contracting a b] is the sums of the products of the parts
   [a] and [b], values of one accumulator, held as {!close} takes them: a
   float as it is, an integer lifted. With it, whether the contracted extent
   is [0]. *)
let products ~batch ~contracting a b =
  let sa = extents (List.hd a) and sb = extents (List.hd b) in
  let ba = List.map fst batch and bb = List.map snd batch in
  let ka = List.map fst contracting and kb = List.map snd contracting in
  let fa = free (List.length sa) (ba @ ka) in
  let fb = free (List.length sb) (bb @ kb) in
  let pick s = List.map (List.nth s) in
  let target = pick sa ba @ pick sa fa @ pick sb fb @ pick sa ka in
  let nb = List.length ba and nfa = List.length fa and nfb = List.length fb in
  let float = Dtype.is_float (Ops.dtype (List.hd a)) in
  let held u = if float then u else lift u in
  let lay s order ~at ~ones u = held (laid u s order ~at ~ones target) in
  let a = List.map (lay sa (ba @ fa @ ka) ~at:(nb + nfa) ~ones:nfb) a in
  let b = List.map (lay sb (bb @ fb @ kb) ~at:nb ~ones:nfa) b in
  let k = List.init (List.length ka) (fun i -> List.length target - 1 - i) in
  let sum u = Shape.rop u Op.Add k in
  let r =
    match (a, b) with
    | [ ar; ai ], [ br; bi ] ->
        [
          sum (Ops.sub (Ops.mul ar br) (Ops.mul ai bi));
          sum (Ops.add (Ops.mul ar bi) (Ops.mul ai br));
        ]
    | [ a ], [ b ] -> [ sum (Ops.mul a b) ]
    | _ -> invalid_arg "Lower_reduce.contract: operands of another dtype"
  in
  (r, List.exists (Int.equal 0) (pick sa ka))

(* [close ~acc ~out ~empty t init r] is the sums [r] of {!products}, of
   operands of node type [t], started from [init], or from [+0] added last,
   rounded once to [out]. Where [empty] the result is [init] itself: adding
   the empty sum's [+0] would turn an [init] of [-0] into [+0]. *)
let close ~acc ~out ~empty t init r =
  let float = Dtype.is_float t in
  let held u = if float then u else lift u in
  let r =
    match init with
    | Some i ->
        let i = List.map held (Arith.parts acc (Arith.cast ~src:out acc i)) in
        if empty then i else List.map2 Ops.add r i
    | None -> List.map (fun r -> Ops.add r (zero r)) r
  in
  let r = if float then r else List.map (store t) r in
  Arith.cast ~src:acc out (wrap acc (Arith.of_parts acc r))

(* [sum ~batch ~contracting ~acc ~out a b init] contracts the parts [a] and [b],
   values of [acc]. *)
let sum ~batch ~contracting ~acc ~out a b init =
  let r, empty = products ~batch ~contracting a b in
  close ~acc ~out ~empty (Ops.dtype (List.hd a)) init r

(* Scales

   A scale is read at each element of its block: the scale's axis of [k]
   repeated [n / k] times in place, for an operand's axis of [n]. A code [e]
   is the float of bits [e << 23] in [float32], [2^-127] being the subnormal
   of bits [1 << 22], and of bits [(e + 896) << 52] in [float64]: [2^(e -
   127)] exactly. *)

let blocks u s =
  let e = extents u in
  let ratio = List.map2 (fun n k -> if k = 0 then 1 else n / k) s e in
  let spread = List.concat (List.map2 (fun k r -> [ k; r ]) e ratio) in
  let u = Shape.reshape u (dims (List.concat_map (fun k -> [ k; 1 ]) e)) in
  Shape.reshape (Shape.expand u (dims spread)) (dims s)

let power (t : Dtype.t) code =
  let e = Ops.cast code (Bits.unsigned t) in
  let k n = Shape.const_like e (`Int (Bigint.of_int n)) in
  let bits =
    match t with
    | Float32 -> Ops.where (Ops.eq e (k 0)) (k 0x400000) (Ops.shl e (k 23))
    | _ -> Ops.shl (Ops.add e (k 896)) (k 52)
  in
  let x = Ops.bitcast bits t in
  Ops.where (Ops.eq e (k 255)) (Bits.nan x) x

(* Whether every element of [dt] times a scale of [scale] in [acc] is a
   bfloat16: an e2m1 value, of two significant bits, times a power of two of
   float32's range is one, an overflow's infinity and a NaN included. *)
let bfloat16_values dt scale acc =
  match (dt, scale, acc) with
  | D.Any D.Float4_e2m1fn, Exponent _, D.Any D.Float32 -> true
  | _ -> false

(* [entered acc ~other x] is [x]'s parts stored into [acc], each element times
   its block's scale read into [acc], the product rounded once. Where [other],
   the other operand's dtype, is bfloat16 and [x]'s products are all
   bfloat16s, a product enters as a float32 cast of its bfloat16, the same
   value: tolk's tensor cores multiply a product of two float32 casts of their
   input type, so the two operands then share the bf16 core. Against another
   dtype no core takes both, and the round trip would cost a cast per element
   for nothing. *)
let entered acc ~other { dtype; node; scale } =
  let x = Arith.cast ~src:dtype acc node in
  match (scale, acc) with
  | None, _ -> Arith.parts acc x
  | Some sc, D.Any (D.Float32 | D.Float64) ->
      let t = Arith.dtype acc and s = extents x in
      let v =
        match sc with
        | Exponent c -> Ops.mul x (power t (blocks c s))
        | Value (d, v) -> Ops.mul x (blocks (Arith.cast ~src:d acc v) s)
      in
      let bf16 = D.Any D.Bfloat16 in
      let paired = match other with D.Any D.Bfloat16 -> true | _ -> false in
      if paired && bfloat16_values dtype sc acc then
        [ Arith.cast ~src:bf16 acc (Arith.cast ~src:acc bf16 v) ]
      else [ v ]
  | Some _, D.Any d ->
      raise
        (Arith.Unlowered
           (Printf.sprintf
              "Rune.Lower_reduce: a contraction scaled in %s is not lowered"
              (D.name d)))

(* Segments

   A segmented contraction's ids, [Bs; n] for the batch extents [Bs] and the
   segmented axis's [n], put index [r] in group [ids.{r}] where that lies in
   [\[0, g)]. *)

(* Whether each id lies in [\[0, g)]: an unsigned comparison, which puts the
   negative ids past [g]. *)
let in_range o g =
  Ops.lt (Ops.bitcast o Uint64)
    (Shape.const_like ~dtype:Uint64 o (`Int (Bigint.of_int g)))

(* [unless has ~acc ~out init r] is the result [r] where [has] holds, and
   elsewhere that of no products: [init] as [acc] holds it, rounded into
   [out], or [+0]. *)
let unless has ~acc ~out init r =
  let none =
    match init with
    | Some i -> Arith.cast ~src:acc out (Arith.cast ~src:out acc i)
    | None -> zero r
  in
  let keep r n = Ops.where has r n in
  Arith.of_parts out
    (List.map2 keep (Arith.parts out r) (Arith.parts out none))

(* [move u ~src ~dst] is [u] with its axis [src] moved to [dst]. *)
let move u ~src ~dst =
  let rest = List.filter (( <> ) src) (List.init (Shape.ndim u) Fun.id) in
  let before = List.filteri (fun i _ -> i < dst) rest in
  let after = List.filteri (fun i _ -> i >= dst) rest in
  Shape.permute u (before @ (src :: after))

(* [block_bound ~n ~g c] bounds the blocks of [c] indices that [g] groups of
   [n] indices cut into: a group of [1 + c m] indices takes [m + 1] blocks,
   and at most [min n g] groups hold an index, so the groups take at most
   [m + ⌊(n - m) / c⌋] for [m = min n g]. *)
let block_bound ~n ~g c =
  let m = min n g in
  m + ((n - m) / c)

(* Routing

   Blocks of [c] indices of one group, each group's indices in order from its
   first block, routed by counting: a one-hot of the ids, its running sum
   along the indices, each index's rank in its group, the groups' counts and
   the exclusive scan of their blocks. Each index takes a {e slot} of the
   [S = B c] slots, [c] a block, its group's first block's first slot plus
   its rank. Each slot finds its index by a binary search of its group's
   running count for its rank, in [⌈log2 (n + 1)⌉] gathers, not a scatter,
   which tolk runs on one thread. The arrays a product reads are stored, so
   that its operands' indices read no reduce. *)

type route = {
  blocks : int;  (** [B]. *)
  slot : Ops.t;  (** [Bs; n]: each index's slot, [S] for one in no group. *)
  row : Ops.t;  (** [Bs; S]: each slot's index, [n] for a slot of none. *)
  owner : Ops.t;  (** [Bs; B]: each block's group, [0] for a block of none. *)
  start : Ops.t;  (** [Bs; g]: each group's first block. *)
  spans : Ops.t;  (** [Bs; g]: each group's count of blocks. *)
  count : Ops.t;  (** [Bs; g]: each group's count of indices. *)
}

(* [per_block u bs blocks c] is [u], [Bs; B], at each slot's block,
   [Bs; S]: [u.{σ / c}]. *)
let per_block u bs blocks c =
  let nb = List.length bs in
  let wide = bs @ [ blocks; c ] in
  let x = lay u (List.init (nb + 1) Fun.id) wide in
  Shape.reshape x (dims (bs @ [ blocks * c ]))

(* [owned u e] is [u], [Bs; g], read at each group [e], [Bs; m]. *)
let owned u e = Bits.take u (Shape.ndim u - 1) e

(* [first_above cum ~n ~g e t] is, for each [e] and [t] of [Bs; m], the least
   index [r < n] whose running count [cum.{r, e}] of group [e]'s indices,
   [cum] of [Bs; n g], passes [t], or [n]: the index of group [e]'s rank
   [t]. A search of [\[lo, hi)] halves it while it holds an index. *)
let first_above cum ~n ~g e t =
  let axis = Shape.ndim cum - 1 in
  let rec go steps lo hi =
    if steps = 0 then lo
    else
      let mid = Ops.div ~rounding:`Trunc (Ops.add lo hi) (int lo 2) in
      let v = Bits.take cum axis (Ops.add (Ops.mul mid (int mid g)) e) in
      let open_ = Ops.lt lo hi in
      let right = Ops.bitwise_and open_ (Ops.le v t) in
      let left = Ops.bitwise_and open_ (Ops.lt t v) in
      let lo = Ops.where right (Ops.add mid (int mid 1)) lo in
      let hi = Ops.where left mid hi in
      go (steps - 1) lo hi
  in
  let rec bits k = if 1 lsl k > n then k else bits (k + 1) in
  go (bits 0) (int t 0) (int t n)

let route o ~g ~c =
  let s = extents o in
  let nb = List.length s - 1 in
  let bs = List.filteri (fun i _ -> i < nb) s and n = List.nth s nb in
  let blocks = block_bound ~n ~g c in
  let wide = bs @ [ n; g ] in
  let up = List.init (nb + 1) Fun.id in
  let hot =
    Ops.eq (lay o up wide) (lay (Shape.arange ~dtype:Int64 g) [ nb + 1 ] wide)
  in
  let one = Ops.cast hot Int64 in
  let cum = Ops.contiguous (Shape.cumalu one nb Op.Add) in
  let rank = Ops.sub cum (int cum 1) in
  let rank = Shape.rop (Ops.where hot rank (int rank 0)) Op.Add [ nb + 1 ] in
  let count = Ops.contiguous (Shape.rop one Op.Add [ nb ]) in
  let spans = Ops.add count (int count (c - 1)) in
  let spans = Ops.div ~rounding:`Trunc spans (int spans c) in
  let start = Ops.sub (Shape.cumalu spans nb Op.Add) spans in
  let spans = Ops.contiguous spans and start = Ops.contiguous start in
  let keep = List.init nb Fun.id @ [ nb + 1 ] in
  let first = lay start keep wide in
  let first = Shape.rop (Ops.where hot first (int first 0)) Op.Add [ nb + 1 ] in
  let last = blocks * c in
  let slot = Ops.add (Ops.mul first (int first c)) rank in
  let slot = Ops.contiguous (Ops.where (in_range o g) slot (int slot last)) in
  (* Whether block [β] is group [e]'s, [Bs; g; B]. *)
  let target = bs @ [ g; blocks ] in
  let per u = lay u up target in
  let beta = lay (Shape.arange ~dtype:Int64 blocks) [ nb + 1 ] target in
  let from = per start in
  let mine =
    Ops.bitwise_and (Ops.le from beta) (Ops.lt beta (Ops.add from (per spans)))
  in
  let e = lay (Shape.arange ~dtype:Int64 g) [ nb ] target in
  let owner = Shape.rop (Ops.where mine e (int e 0)) Op.Add [ nb ] in
  let owner = Ops.contiguous owner in
  (* Each slot's group, its rank in it, and its index where the rank is
     below the group's count. *)
  let e = per_block owner bs blocks c in
  let sigma = lay (Shape.arange ~dtype:Int64 last) [ nb ] (bs @ [ last ]) in
  let t = Ops.sub sigma (Ops.mul (owned start e) (int sigma c)) in
  let cum = Shape.reshape cum (dims (bs @ [ n * g ])) in
  let r = first_above cum ~n ~g e t in
  let row = Ops.where (Ops.lt t (owned count e)) r (int r n) in
  { blocks; slot; row = Ops.contiguous row; owner; start; spans; count }

(* [spread p xs sides axis] is the positions [p], [Bs; m], laid over the
   extents [xs] with [axis]'s replaced by [m]: [p]'s batch axes at [sides],
   its last at [axis], expanded along the others. *)
let spread p xs sides axis =
  let m = List.nth (extents p) (Shape.ndim p - 1) in
  let target = List.mapi (fun i e -> if i = axis then m else e) xs in
  lay p (sides @ [ axis ]) target

(* [picked x axis p] is the operand [x] read at the positions [p] along
   [axis], [p] laid over [x]'s shape ({!spread}): its parts and its scale,
   spread over its elements, each by {!Bits.take}. A position outside the
   axis reads zeros. *)
let picked ?clamped { dtype; node; scale } axis p =
  let parts = Arith.parts dtype node in
  let s = extents (List.hd parts) in
  let take u = Bits.take ?clamped u axis p in
  let node = Arith.of_parts dtype (List.map take parts) in
  let scale =
    Option.map
      (function
        | Exponent c -> Exponent (take (blocks c s))
        | Value (d, v) -> Value (d, take (blocks v s)))
      scale
  in
  { dtype; node; scale }

(* [splitted u axis parts] is the node [u] with its axis [axis] cut into
   axes of the extents [parts]. *)
let splitted u axis parts =
  let s = extents u in
  let cut x e = if x = axis then parts else [ e ] in
  Shape.reshape u (dims (List.concat (List.mapi cut s)))

(* [split x axis parts] is the operand [x] with its axis [axis] cut so, its
   scale, read at its own elements ({!picked}), alike. *)
let split { dtype; node; scale } axis parts =
  let cut u = splitted u axis parts in
  let scale =
    Option.map
      (function
        | Exponent c -> Exponent (cut c) | Value (d, v) -> Value (d, cut v))
      scale
  in
  { dtype; node = cut node; scale }

(* [gathered dt u axis p] is the value [u] of [dt] read at the positions [p]
   along [axis], each part by {!Bits.take}. *)
let gathered dt u axis p =
  Arith.of_parts dt (List.map (fun u -> Bits.take u axis p) (Arith.parts dt u))

(* The rows an expert holds on average below which a [Rows] contraction reads
   [b] per row, and the rows of a block above it. Per row, each row streams
   its group's slice, [n] slices read for [n] rows; in blocks, a block of 64
   rows of one group shares each slice it reads, as a tensor core's
   workgroup of 64 rows does, against routing kernels and padding rows.
   Measured on gpt-oss-20b's MXFP4 experts layer, 32 experts on an RTX 5000
   Ada: at 2 rows an expert per row takes 1.6 ms against 2.5 in blocks, at
   4 rows 3.1 against 2.5, at 8 rows 6.6 against 2.9. *)
let per_row = 4

(* [block_rows ~n ~g] is the rows of a block of [n] rows over [g] groups:
   64 from 16 rows a group on average, where 64 rows of one group fill a
   tensor core's workgroup (on the RTX 5000 Ada blocks of 64 beat blocks of
   16 from 16 rows an expert: 3.2 against 7.0 ms on gpt-oss-20b's experts
   layer at 128 tokens). With fewer, the most rows up to 16, a power of two,
   whose padding stays under half the rows, [2 (c - 1) g < n], as
   [Nx.map_segments] bounds its own: a target whose padding rows cost as
   much as real ones, as the host's do, then computes under one and a half
   times the rows, where 64 would compute up to 16 times them. Below 2, each
   row reads its own slice. *)
let block_rows ~n ~g =
  if n >= 16 * g then 64
  else
    let rec go c = if c < 2 || 2 * (c - 1) * g < n then c else go (c / 2) in
    go 16

(* [Rows]: row [r] of [a]'s [a_axis] reads [b] at its group along [b_axis],
   which then holds [a]'s rows, a batch pair whose axis moves back among [a]'s
   free axes in the result. Each row contracts with its own group's slice
   alone: the work of one slice's contraction, whatever [g].

   With fewer than {!per_row} rows a group on average, each row reads [b]'s
   slice at its id, clamped into [\[0, g)]. Otherwise the rows are routed
   ({!route}) into blocks of {!block_rows} rows: each slot reads [a]'s row at
   its index and [b]'s slice at its block's group, [owner[σ / c]], and [init] at
   its index, so that [init] enters the sum once; its output is stored, and
   each row reads its slot's. A slot of no index reads [a] as zeros against a
   real slice, so that it may compute [0 · ∞]; no row reads it. A row in no
   group takes the no-product rule. *)
let rows ~batch ~contracting ~acc ~out a b init o ~a_axis ~b_axis =
  let shape x = extents (List.hd (Arith.parts x.dtype x.node)) in
  let sa = shape a and sb = shape b in
  let n = List.nth sa a_axis and g = List.nth sb b_axis in
  let nb = List.length batch in
  let fa = free (List.length sa) (List.map fst (batch @ contracting)) in
  let j = nb + Option.get (List.find_index (Int.equal a_axis) fa) in
  let a_sides = List.map fst batch and b_sides = List.map snd batch in
  let r_sides = List.init nb Fun.id in
  let inside = in_range o g in
  let init_at i p =
    let s = extents (List.hd (Arith.parts out i)) in
    gathered out i j (spread p s r_sides j)
  in
  let r =
    let c = if g = 0 then 1 else block_rows ~n ~g in
    if g = 0 || n < per_row * g || c < 2 then
      let id = Ops.where inside o (int o 0) in
      let b' = picked ~clamped:true b b_axis (spread id sb b_sides b_axis) in
      let a' = entered acc ~other:b.dtype a in
      let b' = entered acc ~other:a.dtype b' in
      let laid = Option.map (move ~src:j ~dst:nb) init in
      let pairs = batch @ [ (a_axis, b_axis) ] in
      let r = sum ~batch:pairs ~contracting ~acc ~out a' b' laid in
      move ~src:nb ~dst:j r
    else
      let r = route o ~g ~c in
      let blocks = r.blocks in
      (* [a]'s rows at their slots, [a_axis] cut into the blocks and their
         rows, and [b]'s slices at the blocks' groups: a batch pair of the
         blocks. *)
      let a' = picked a a_axis (spread r.row sa a_sides a_axis) in
      let a' = split a' a_axis [ blocks; c ] in
      let owner = spread r.owner sb b_sides b_axis in
      let b' = picked ~clamped:true b b_axis owner in
      let a' = entered acc ~other:b.dtype a' in
      let b' = entered acc ~other:a.dtype b' in
      let up x = if x > a_axis then x + 1 else x in
      let shift = List.map (fun (x, y) -> (up x, y)) in
      let pairs = shift batch @ [ (a_axis, b_axis) ] in
      (* The result's slots: [blocks] at [nb], the rows of a block at [j + 1],
         read back at each row's slot. *)
      let cut u =
        let cut u = splitted u j [ blocks; c ] in
        Arith.of_parts out (List.map cut (Arith.parts out u))
      in
      let laid =
        Option.map (fun i -> move ~src:j ~dst:nb (cut (init_at i r.row))) init
      in
      let contracting = shift contracting in
      let y = sum ~batch:pairs ~contracting ~acc ~out a' b' laid in
      let y = move ~src:nb ~dst:j y in
      (* The groups take the first blocks, each at least one row: a block past
         them holds only padding, whose slots no row reads, so it holds no value
         and its products need not run. *)
      let used = Shape.rop r.spans Op.Add [ nb ] in
      let s = extents (List.hd (Arith.parts out y)) in
      let beta = lay (Shape.arange ~dtype:Int64 blocks) [ j ] s in
      let held = Ops.lt beta (lay used r_sides s) in
      let held u = Ops.where held u (Shape.const_like u `Invalid) in
      let y = Arith.of_parts out (List.map held (Arith.parts out y)) in
      let merge u =
        let s = List.mapi (fun x e -> if x = j then e * c else e) (extents u) in
        let s = List.filteri (fun x _ -> x <> j + 1) s in
        Ops.contiguous (Shape.reshape u (dims s))
      in
      let y = Arith.of_parts out (List.map merge (Arith.parts out y)) in
      init_at y r.slot
  in
  let value = Arith.parts out r |> List.hd |> extents in
  let has = lay inside (r_sides @ [ j ]) value in
  unless has ~acc ~out init r

(* [Contracting] in blocks of [c] indices, routed ({!route}): each group's
   indices in order, cut into [⌈k / c⌉] blocks from its first for a group of
   [k], so that no block holds indices of two groups, as many blocks as
   {!block_bound} bounds. Each block's products sum into its partial result,
   each position read at its slot's index by rune's gather, a slot of none
   reading [0] from each operand, so that its product adds [+0]. Each group
   then sums its own blocks' partial results, materialised, over [⌈n / c⌉]
   places, the most blocks a group takes: its [i]th block at place [i], [+0]
   past its last. tolk splits a sum by its places, so the association
   follows the group's count alone. Summed by block number or by index, it
   would follow the other groups' counts, which place the group's first
   block and first index. Blocks of [n] indices give each group that holds
   an index one block, its own, at one place. *)
let blocked ~batch ~contracting ~acc ~out a b init o ~groups:g ~pair ~c =
  let ka, kb = List.nth contracting pair in
  let n = Bits.extent (List.hd a) ka in
  let nb = List.length batch in
  let r = route o ~g ~c in
  let blocks = r.blocks in
  let ms = extents r.row in
  let bs = List.filteri (fun i _ -> i < nb) ms in
  (* An operand's part [u] with its axis [k] cut into the blocks and their
     positions, each position read at its slot's index. *)
  let read side k u =
    let s = extents u in
    let sides = List.map side batch in
    let x = Bits.take u k (spread r.row s sides k) in
    let cut x e = if x = k then [ blocks; c ] else [ e ] in
    Shape.reshape x (dims (List.concat (List.mapi cut s)))
  in
  let a = List.map (read fst ka) a and b = List.map (read snd kb) b in
  let shift ka kb (i, j) =
    ((if i < ka then i else i + 1), if j < kb then j else j + 1)
  in
  let batch' = (ka, kb) :: List.map (shift ka kb) batch in
  let contracting' =
    List.mapi
      (fun q p -> if q = pair then (ka + 1, kb + 1) else shift ka kb p)
      contracting
  in
  let parts, empty = products ~batch:batch' ~contracting:contracting' a b in
  let parts = List.map Ops.contiguous parts in
  (* Each group's sum of its blocks' partial results, [g; Bs; fa; fb]: at
     [i] below its count of blocks the partial result of block [start + i],
     past it position [blocks], which loads [+0]. *)
  let most = (n + c - 1) / c in
  let place = bs @ [ g; most ] in
  let i = lay (Shape.arange ~dtype:Int64 most) [ nb + 1 ] place in
  let own u = lay u (List.init (nb + 1) Fun.id) place in
  let at =
    Ops.where (Ops.lt i (own r.spans)) (Ops.add (own r.start) i) (int i blocks)
  in
  let gather u =
    let rest = List.tl (extents u) in
    let wide = g :: most :: rest in
    let at = lay at (List.init nb (fun i -> i + 2) @ [ 0; 1 ]) wide in
    let at = Shape.reshape at (dims ((g * most) :: rest)) in
    let x = Shape.reshape (Bits.take u 0 at) (dims wide) in
    Shape.rop x Op.Add [ 1 ]
  in
  let t = Ops.dtype (List.hd a) in
  let y = close ~acc ~out ~empty t init (List.map gather parts) in
  let has = Ops.lt (int r.count 0) r.count in
  let value = Arith.parts out y |> List.hd |> extents in
  let has = lay has (List.init nb succ @ [ 0 ]) value in
  unless has ~acc ~out init y

(* [ceil_sqrt m] is [⌈√m⌉] for [m >= 0]. *)
let ceil_sqrt m =
  let r = Float.to_int (Float.sqrt (Float.of_int m)) in
  let r = if r * r > m then r - 1 else r in
  let r = if (r + 1) * (r + 1) <= m then r + 1 else r in
  if r * r < m then r + 1 else r

(* For each index of its other axes, a [Contracting] contraction of [g] groups
   over [n] indices takes, in blocks of [c] indices, [B c] products and [g B]
   sums of partial results, [B] the bound of {!block_bound}. The block size is
   [⌈n / g⌉] or [⌈√(n - g)⌉], whichever takes fewer, the latter near the
   count's least, [n - g + 2 g √(n - g) + g²], where that takes fewer
   products and sums than [g n]. Otherwise it is [n]: one block a group. *)
let block_size ~n ~g =
  let whole = max 1 n in
  if g = 0 || n <= g then whole
  else
    let cost c =
      let b = block_bound ~n ~g c in
      (b * c) + (g * b)
    in
    let by_groups = (n + g - 1) / g and by_root = max 1 (ceil_sqrt (n - g)) in
    let c = if cost by_root < cost by_groups then by_root else by_groups in
    if cost c < g * n then c else whole

let contract ~batch ~contracting ~segments ~acc ~out a b init =
  let batch = Iarray.to_list batch in
  let contracting = Iarray.to_list contracting in
  let entered () =
    (entered acc ~other:b.dtype a, entered acc ~other:a.dtype b)
  in
  match (segments : Ops.t Nx_kernel.segments option) with
  | None ->
      let a, b = entered () in
      sum ~batch ~contracting ~acc ~out a b init
  | Some (Rows { ids; a_axis; b_axis }) ->
      rows ~batch ~contracting ~acc ~out a b init ids ~a_axis ~b_axis
  | Some (Contracting { ids; groups; pair }) -> (
      let a, b = entered () in
      let n = Bits.extent (List.hd a) (fst (List.nth contracting pair)) in
      let c = block_size ~n ~g:groups in
      blocked ~batch ~contracting ~acc ~out a b init ids ~groups ~pair ~c)
