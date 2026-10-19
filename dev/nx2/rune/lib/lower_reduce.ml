(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
module D = Nx_array.Dtype
module Arith = Lower_arith

type monoid = Sum | Prod | Max | Min

let refuse m (D.Any dt) =
  match m with
  | (Prod | Max | Min) when D.is D.Complex dt ->
      invalid_arg "Lower_reduce: a complex product or extreme"
  | _ -> ()

let extents u =
  List.map
    (function
      | Ops.Int n -> n
      | Ops.Sym _ -> invalid_arg "Lower_reduce: a symbolic extent")
    (Shape.shape u)

let dims s = List.map (fun e -> Ops.Int e) s
let int u n = Shape.const_like u (`Int (Bigint.of_int n))
let bit k = `Int (Bigint.shift_left Bigint.one k)

(* Integers fold on the unsigned pattern of their width: C, Metal and CUDA leave
   signed overflow undefined, and the pattern wraps as nx's integers do. *)
let unsigned : Dtype.t -> Dtype.t = function
  | Int8 -> Uint8
  | Int16 -> Uint16
  | Int32 -> Uint32
  | Int64 -> Uint64
  | t -> t

(* The signed integer of a carrier float's width. *)
let signed : Dtype.t -> Dtype.t = function
  | Float32 -> Int32
  | Float64 -> Int64
  | _ -> invalid_arg "Lower_reduce: a float narrower than its carrier"

(* Sums and products

   [f op u] folds the node [u] by [op]: a reduction or a scan. A float sum adds
   [+0] to its result: tolk starts a loop's accumulator from [+0], but sums the
   terms alone where no loop is left, so a sum of [-0] terms would stay [-0]. *)

let arith f op u =
  let t = Ops.dtype u in
  if not (Dtype.is_float t) then Ops.cast (f op (Ops.cast u (unsigned t))) t
  else
    let r = f op u in
    if Op.equal op Op.Add then Ops.add r (Shape.const_like r (`Float 0.)) else r

(* Extremes

   A float's order key is the signed integer of its bits, a negative float's
   magnitude bits flipped: the greater float has the greater key and [-0] is
   just below [+0]. The key is its own inverse. A minimum is the maximum of the
   complemented keys. NaN terms take the least key, which no number takes, so
   the maximum is of the numbers alone.

   Where a fold takes a NaN, its result is the first NaN's bits quieted. [first
   u axes nan] marks each fold's first NaN term: a REDUCE Max of [n - position],
   [n] the fold's term count, over NaN terms, and [0] over the others. The fold
   by Add of the marked term's bits, others [0], is that NaN's bits from its
   position on, [0] before: a NaN's bits are never [0].

   Cost: a reduction takes three REDUCEs: the keys' Max and the positions' Max,
   which read the same terms, then the marked bits' Add, which reads them again
   with the positions' result. A scan takes the keys' running Max, the
   positions' REDUCE Max and the marked bits' running Add. *)

let order k =
  let flipped = Shape.const_like k (Dtype.max (Ops.dtype k) :> Dtype.const) in
  Ops.where (Ops.lt k (int k 0)) (Ops.bitwise_xor k flipped) k

(* [positions u axes] is [n - p] at each element of [u], [p] its position in C
   order of the indices along [axes], [n] their count. *)
let positions u axes =
  let s = extents u in
  let r = List.length s in
  let n = List.fold_left (fun n a -> n * List.nth s a) 1 axes in
  let dtype : Dtype.t =
    if n < Int32.to_int Int32.max_int then Int32 else Int64
  in
  let index a =
    let e = List.nth s a in
    let along = List.init r (fun i -> Ops.Int (if i = a then e else 1)) in
    Shape.expand (Shape.reshape (Shape.arange ~dtype e) along) (dims s)
  in
  let step a (p, stride) =
    let i = index a in
    (Ops.sub p (Ops.mul i (int i stride)), stride * List.nth s a)
  in
  let start = Shape.const_like ~dtype u (`Int (Bigint.of_int n)) in
  fst (List.fold_right step axes (start, 1))

let first u axes nan =
  let key = positions u axes in
  let key = Ops.where nan key (int key 0) in
  let s = extents u in
  let kept = List.mapi (fun i e -> if List.mem i axes then 1 else e) s in
  let m = Shape.rop key Op.Max axes in
  let m = Shape.expand (Shape.reshape m (dims kept)) (dims s) in
  Ops.bitwise_and nan (Ops.eq key m)

let float_extreme f axes ~greatest u =
  let t = Ops.dtype u in
  let k = signed t in
  let nan = Ops.ne u u in
  let flip key = if greatest then key else Ops.bitwise_not key in
  let key = flip (order (Ops.bitcast u k)) in
  let least = Shape.const_like key (Dtype.min k :> Dtype.const) in
  let m = f Op.Max (Ops.where nan least key) in
  let value = Ops.bitcast (order (flip m)) t in
  let bits = Ops.bitcast u (unsigned k) in
  let nan = f Op.Add (Ops.where (first u axes nan) bits (int bits 0)) in
  let quiet = Shape.const_like nan (bit (snd (Dtype.finfo t) - 1)) in
  let quieted = Ops.bitcast (Ops.bitwise_or nan quiet) t in
  Ops.where (Ops.ne nan (int nan 0)) quieted value

(* An integer's minimum is the complement of the maximum of the complements.
   Booleans fold as their truths' bytes, 0 or 1: any and all. *)
let extreme f axes ~greatest u =
  let flip u = if greatest then u else Ops.bitwise_not u in
  match Ops.dtype u with
  | t when Dtype.is_float t -> float_extreme f axes ~greatest u
  | _ -> flip (f Op.Max (flip u))

(* [fold m f axes c x] folds [x], a value of the carrier [c], by [m] with [f]
   along [axes]. *)
let fold m f axes c x =
  match m with
  | Sum -> Arith.of_parts c (List.map (arith f Op.Add) (Arith.parts c x))
  | Prod -> arith f Op.Mul x
  | Max -> extreme f axes ~greatest:true x
  | Min -> extreme f axes ~greatest:false x

(* [carried ~src c x] is [x] in the carrier [c]; a boolean as the byte of its
   truth. *)
let carried ~src (D.Any dt as c) x =
  if D.is D.Boolean dt then Ops.cast (Ops.ne x (int x 0)) Uint8
  else Arith.cast ~src c x

let reduce m axes ~src dt x =
  refuse m src;
  let c = Nx_kernel.carrier src in
  let f op u = Shape.rop u op axes in
  Arith.cast ~src:c dt (fold m f axes c (carried ~src c x))

(* Scans *)

(* [join axis k x] is [k] then [x] along [axis]. Shape.cat adds zero-padded
   pieces, which would turn a [-0] into [+0]: floats join as their bits. *)
let join axis k x =
  let cat k x = Shape.cat ~axis k [ x ] in
  match Ops.dtype x with
  | t when Dtype.is_float t ->
      let b = unsigned (signed t) in
      Ops.bitcast (cat (Ops.bitcast k b) (Ops.bitcast x b)) t
  | _ -> cat k x

(* [negative_zeros f x y] is the running float sum [y] of [x], [-0] where every
   term so far, the carry's included, is [-0], as a sum started from a [-0]
   carry gives. Cost: a running Max over bytes. *)
let negative_zeros f x y =
  let t = Ops.dtype x in
  if not (Dtype.is_float t) then y
  else
    let bits = Ops.bitcast x (unsigned (signed t)) in
    let sign = Shape.const_like bits (bit ((8 * Dtype.itemsize t) - 1)) in
    let other = f Op.Max (Ops.cast (Ops.ne bits sign) Uint8) in
    let zero = Shape.const_like y (`Float (-0.)) in
    Ops.where (Ops.eq other (int other 0)) zero y

(* A carry is the term before each slice's first: the scan runs over it, then
   drops its result. *)
let scan m axis ~carry ~src dt x =
  refuse m src;
  let c = Nx_kernel.carrier src in
  let x = carried ~src c x in
  let f op u = Shape.cumalu u axis op in
  let parts g x y =
    Arith.of_parts c (List.map2 g (Arith.parts c x) (Arith.parts c y))
  in
  match carry with
  | None -> Arith.cast ~src:c dt (fold m f [ axis ] c x)
  | Some k ->
      let x = parts (join axis) (carried ~src:c c k) x in
      let y = fold m f [ axis ] c x in
      let y = if m = Sum then parts (negative_zeros f) x y else y in
      let n = List.nth (extents y) axis in
      let drop i = if i = axis then Some (Ops.Int 1, Ops.Int n) else None in
      Arith.cast ~src:c dt (Shape.shrink y (List.init (Shape.ndim y) drop))

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
  let lift u = if float then u else Ops.cast u (unsigned t) in
  let lay s order ~at ~ones u = lift (laid u s order ~at ~ones target) in
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
  (* The sum starts from [init], or from [+0] added last. *)
  let start =
    match init with
    | Some i -> List.map lift (Arith.parts acc (Arith.cast ~src:out acc i))
    | None ->
        let zero = if float then `Float 0. else `Int Bigint.zero in
        List.map (fun r -> Shape.const_like r zero) r
  in
  let r = List.map2 (fun r s -> Ops.cast (Ops.add r s) t) r start in
  Arith.cast ~src:acc out (wrap acc (Arith.of_parts acc r))
