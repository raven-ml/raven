(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
module D = Nx_array.Dtype
module Arith = Lower_arith
module Bits = Lower_bits

type monoid = Sum | Prod | Max | Min

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

(* NaNs

   Where a fold takes a NaN term, its result is the first NaN term, in C order
   of the reduced indices, quieted; where it takes none but gives a NaN, made
   from numbers as [∞ - ∞], the quiet NaN of a clear sign bit. Both are selected
   by integer bit patterns: bitcasts, integer folds and [where]. So a target
   whose float adder canonicalises NaNs, as CUDA's does, still gives these bits:
   no float arithmetic produces them.

   [first f axes ~nan us] is, at each result of the fold [f] along [axes], the
   bits of each part of [us] at the first term where [nan] holds, from that term
   on, and [0] where no term so far is a NaN. A term at position [p] of [n] keys
   as [n - p] where [nan] holds, [0] elsewhere: the greatest key is the first
   NaN's.

   Cost: for parts of 32 bits and [n < 2^31], each part packs its key over its
   bits in an int64, [(n - p) lsl 32 lor bits], whose REDUCE Max, or running
   Max, is that NaN's bits: one fold per part, reading the terms the fold reads,
   so a float32 sum or extreme takes two REDUCEs. Otherwise, a REDUCE Max of the
   keys over the whole slice marks the first NaN, and a Max of each part's
   marked bits, others [0], reads the terms again. *)

(* [positions ~dtype u axes n] is [n - p] at each element of [u], [p] its
   position in C order of the indices along [axes], [n] their count: [n] down to
   [1] laid over [axes]. *)
let positions ~dtype u axes n =
  let s = extents u in
  let along = List.mapi (fun i e -> if List.mem i axes then e else 1) s in
  let keys = Shape.arange ~start:n ~step:(-1) ~dtype 0 in
  Shape.expand (Shape.reshape keys (dims along)) (dims s)

let first f axes ~nan us =
  let u = List.hd us in
  let s = extents u in
  let n = List.fold_left (fun n a -> n * List.nth s a) 1 axes in
  let packs =
    Dtype.itemsize (Ops.dtype u) = 4 && n < Int32.to_int Int32.max_int
  in
  if packs then
    let key = positions ~dtype:Int64 u axes n in
    let key = Ops.shl key (int key 32) in
    let part u =
      let packed = Ops.bitwise_or key (Ops.cast (Bits.bits u) Int64) in
      Ops.cast (f Op.Max (Ops.where nan packed (int key 0))) Uint32
    in
    List.map part us
  else
    let dtype : Dtype.t =
      if n < Int32.to_int Int32.max_int then Int32 else Int64
    in
    let key = positions ~dtype u axes n in
    let key = Ops.where nan key (int key 0) in
    let marked =
      Ops.bitwise_and nan (Ops.eq key (back u axes (Shape.rop key Op.Max axes)))
    in
    List.map
      (fun u ->
        let b = Bits.bits u in
        f Op.Max (Ops.where marked b (int b 0)))
      us

(* [nans f axes ~nan us rs] is each part of the fold's result [rs] of the terms
   [us], with the first term where [nan] holds, each NaN part quieted, where one
   does, else the quiet NaN of a clear sign bit where the part is a NaN. *)
let nans f axes ~nan us rs =
  let bs = first f axes ~nan us in
  let nonzero b = Ops.ne b (int b 0) in
  let has = any nonzero bs in
  let fix b r =
    let term = Ops.bitcast b (Ops.dtype r) in
    let term = Ops.where (Bits.is_nan term) (Bits.quieted term) term in
    Ops.where has term (Ops.where (Bits.is_nan r) (Bits.nan r) r)
  in
  List.map2 fix bs rs

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
   complemented keys. NaN terms take the least key, which no number takes, so
   the maximum is of the numbers alone.

   A complex number orders by its real part, then its imaginary part. With
   float32 parts, each key biased to unsigned, [k xor min_int], packs the real
   key over the imaginary one in a uint64, whose Max is the extreme: NaN terms
   take [0], below every number's. With float64 parts, a REDUCE Max of the real
   keys, then of the imaginary keys of the terms whose real key is it: a scan
   cannot take that second Max, and refuses. *)

let order k =
  let flipped = Shape.const_like k (Dtype.max (Ops.dtype k) :> Dtype.const) in
  Ops.where (Ops.lt k (int k 0)) (Ops.bitwise_xor k flipped) k

let least k = Shape.const_like k (Dtype.min (Ops.dtype k) :> Dtype.const)

let extreme f axes ~greatest ~nan xs =
  let flip k = if greatest then k else Ops.bitwise_not k in
  let key u = flip (order (Ops.bitcast u (Bits.signed (Ops.dtype u)))) in
  let value t k = Ops.bitcast (order (flip k)) t in
  match xs with
  | [ x ] when not (Dtype.is_float (Ops.dtype x)) ->
      [ flip (f Op.Max (flip x)) ]
  | [ x ] ->
      let k = key x in
      [ value (Ops.dtype x) (f Op.Max (Ops.where nan (least k) k)) ]
  | [ re; im ] when Dtype.itemsize (Ops.dtype re) = 4 ->
      let biased k =
        Ops.cast (Ops.bitcast (Ops.bitwise_xor k (least k)) Uint32) Uint64
      in
      let kr = biased (key re) and ki = biased (key im) in
      let packed = Ops.bitwise_or (Ops.shl kr (int kr 32)) ki in
      let m = f Op.Max (Ops.where nan (int packed 0) packed) in
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
      let mr = f Op.Max (Ops.where nan (least kr) kr) in
      let top =
        Ops.bitwise_and (Ops.logical_not nan) (Ops.eq kr (back re axes mr))
      in
      let mi = f Op.Max (Ops.where top ki (least ki)) in
      [ value Float64 mr; value Float64 mi ]
  | _ -> invalid_arg "Lower_reduce: a value of another dtype"

(* [fold m f axes xs] folds the parts [xs] of a value of a carrier by [m] with
   [f] along [axes], NaNs aside. *)
let fold m f axes xs =
  match m with
  | Sum -> List.map (arith f Op.Add) xs
  | Prod -> List.map (arith f Op.Mul) xs
  | Max | Min -> extreme f axes ~greatest:(m = Max) ~nan:(any Bits.is_nan xs) xs

(* [with_nans m f axes xs ys] is the fold [ys] of the parts [xs] with NaNs as
   {!Nx_kernel.reduction} states: a sum or a product takes each part's NaNs
   alone, an extreme a complex number with a NaN part as a NaN. *)
let with_nans m f axes xs ys =
  if not (Dtype.is_float (Ops.dtype (List.hd xs))) then ys
  else
    match m with
    | Sum | Prod ->
        List.map2
          (fun x y -> List.hd (nans f axes ~nan:(Bits.is_nan x) [ x ] [ y ]))
          xs ys
    | Max | Min -> nans f axes ~nan:(any Bits.is_nan xs) xs ys

(* [carried ~src c x] is [x] in the carrier [c]; a boolean as the byte of its
   truth. *)
let carried ~src (D.Any dt as c) x =
  if D.is D.Boolean dt then Ops.cast (Ops.ne x (int x 0)) Uint8
  else Arith.cast ~src c x

let reduce m axes ~src dt x =
  refuse m ~scan:false src;
  let c = Nx_kernel.carrier src in
  let f op u = Shape.rop u op axes in
  let xs = Arith.parts c (carried ~src c x) in
  let ys = with_nans m f axes xs (fold m f axes xs) in
  Arith.cast ~src:c dt (Arith.of_parts c ys)

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
   drops its result. An extreme's NaN rule takes the carry as a term; a sum's or
   a product's takes the slice's own terms, so a NaN carry counts as made from
   numbers. *)
let scan m axis ~carry ~src dt x =
  refuse m ~scan:true src;
  let c = Nx_kernel.carrier src in
  let xs = Arith.parts c (carried ~src c x) in
  let f op u = Shape.cumalu u axis op in
  let axes = [ axis ] in
  let out ys = Arith.cast ~src:c dt (Arith.of_parts c ys) in
  match carry with
  | None -> out (with_nans m f axes xs (fold m f axes xs))
  | Some k ->
      let joined =
        List.map2 (join axis) (Arith.parts c (carried ~src:c c k)) xs
      in
      let n = List.nth (extents (List.hd xs)) axis in
      let drop u =
        Shape.shrink u
          (List.init (Shape.ndim u) (fun i ->
               if i = axis then Some (Ops.Int 1, Ops.Int (n + 1)) else None))
      in
      let ys = fold m f axes joined in
      let ys =
        match m with
        | Max | Min -> List.map drop (with_nans m f axes joined ys)
        | Sum ->
            let ys = List.map2 (negative_zeros f) joined ys in
            with_nans m f axes xs (List.map drop ys)
        | Prod -> with_nans m f axes xs (List.map drop ys)
      in
      out ys

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
