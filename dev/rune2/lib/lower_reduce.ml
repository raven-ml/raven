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
      invalid_arg "Rune.Lower_reduce: a complex Prod is not lowered"
  | (Max | Min), D.Complex128 when scan ->
      invalid_arg
        "Rune.Lower_reduce: a complex128 Max or Min scan is not lowered"
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
   result is [-∞], [+∞] or a NaN there. *)
let logsumexp f axes x =
  let c = shift (List.hd (fold Max f axes [ x ])) in
  let s = List.hd (fold Sum f axes [ exp (Ops.sub x (back x axes c)) ]) in
  Ops.add c (log s)

(* Moments

   The mean [Σ x / n], then the variance by the corrected two-pass algorithm
   (Chan, Golub and LeVeque, "Algorithms for computing the sample variance",
   1983): [(Σ d² - Σ d (Σ d / n)) / n], [d = x - mean]. [Σ (x - c)² - (Σ (x -
   c))² / n] is [Σ (x - μ)²] for any [c], so the mean's rounding reaches the
   variance at second order. [Σ d (Σ d / n)] is at most [Σ d²], so it
   overflows only where [Σ d²] does, which gives [+∞]. Rounding may make the
   difference negative: [+0]. A NaN mean makes every [d] a NaN. *)
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
   [m] where finite and [0] elsewhere; a term [x] is [(x, exp (x - c))]. *)
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
   deviations and a count [n], combined by Chan, Golub and LeVeque's (1.5b),
   the mean weighted; [c] is the slice's first term where finite, else [0],
   and a term [x] is [(x - c, 0, 1)]. *)
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
   is [(ar br - ai bi) + i (ar bi + ai br)]. *)

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

(* A sub-byte accumulator's node holds a byte, which wraps to its width. *)
let wrap acc r =
  match acc with
  | D.Any D.Int4 -> Arith.cast ~src:(D.Any D.Int8) acc r
  | D.Any D.Uint4 -> Arith.cast ~src:(D.Any D.Uint8) acc r
  | _ -> r

let contract ~batch ~contracting ~acc ~out (da, a) (db, b) init =
  let a = Arith.parts acc (Arith.cast ~src:da acc a) in
  let b = Arith.parts acc (Arith.cast ~src:db acc b) in
  let sa = extents (List.hd a) and sb = extents (List.hd b) in
  let ba = List.map fst (Iarray.to_list batch) in
  let bb = List.map snd (Iarray.to_list batch) in
  let ka = List.map fst (Iarray.to_list contracting) in
  let kb = List.map snd (Iarray.to_list contracting) in
  let fa = free (List.length sa) (ba @ ka) in
  let fb = free (List.length sb) (bb @ kb) in
  let pick s = List.map (List.nth s) in
  let target = pick sa ba @ pick sa fa @ pick sb fb @ pick sa ka in
  let nb = List.length ba and nfa = List.length fa and nfb = List.length fb in
  let t = Ops.dtype (List.hd a) in
  let float = Dtype.is_float t in
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
  (* The sum starts from [init], or from [+0] added last. With no products
     the result is [init] itself: adding the empty sum's [+0] would turn an
     [init] of [-0] into [+0]. *)
  let empty = List.exists (Int.equal 0) (pick sa ka) in
  let r =
    match init with
    | Some i ->
        let i = List.map held (Arith.parts acc (Arith.cast ~src:out acc i)) in
        if empty then i else List.map2 Ops.add r i
    | None ->
        let zero = if float then `Float 0. else `Int Bigint.zero in
        List.map (fun r -> Ops.add r (Shape.const_like r zero)) r
  in
  let r = if float then r else List.map (store t) r in
  Arith.cast ~src:acc out (wrap acc (Arith.of_parts acc r))
