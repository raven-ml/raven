(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
module D = Nx_array.Dtype
module Arith = Lower_arith
module Bits = Lower_bits

let unlowered what =
  invalid_arg (Printf.sprintf "Rune.Lower.op: %s is not lowered" what)

let extents u = List.init (Shape.ndim u) (Bits.extent u)
let dims s = List.map (fun n -> Ops.Int n) s
let numel s = List.fold_left ( * ) 1 s
let int u n = Shape.const_like u (`Int (Bigint.of_int n))

let broadcast c s =
  Shape.expand (Shape.reshape c (List.map (fun _ -> Ops.Int 1) s)) (dims s)

let zeros t s = broadcast (Ops.const ~dtype:t (`Int Bigint.zero)) s

(* [laid s axis v] is the vector [v] laid along [axis] of the shape [s]. *)
let laid s axis v =
  let r = List.length s in
  let along = List.init r (fun i -> if i = axis then List.nth s i else 1) in
  Shape.expand (Shape.reshape v (dims along)) (dims s)

(* [joined axis a b] is [a] then [b] along [axis]. Shape.cat adds zero-padded
   pieces, which would turn a [-0] into [+0]: floats join as their bits. *)
let joined axis a b =
  match Ops.dtype a with
  | t when Dtype.is_float t ->
      let u = Bits.unsigned t in
      Ops.bitcast (Shape.cat ~axis (Ops.bitcast a u) [ Ops.bitcast b u ]) t
  | _ -> Shape.cat ~axis a [ b ]

(* [negative_zero u] is whether the float [u] is [-0]: its sign bit alone. *)
let negative_zero u =
  let b = Bits.bits u in
  let w = 8 * Dtype.itemsize (Ops.dtype u) in
  Ops.eq b (Shape.const_like b (`Int (Bigint.shift_left Bigint.one (w - 1))))

(* Gathers

   A gather is tolk's index of the operand's elements, flattened, at each
   element's offset: its own index's along the other axes, its position's along
   [axis]. A position outside the axis is read at [0], so that every read is in
   bounds, and selected as zero. *)

let take u axis p =
  let s = extents u and t = extents p in
  let n = List.nth s axis in
  if n = 0 || numel t = 0 then zeros (Ops.dtype u) t
  else
    let inside =
      Ops.lt (Ops.bitcast p Uint64)
        (Shape.const_like ~dtype:Uint64 p (`Int (Bigint.of_int n)))
    in
    let at = Ops.cast (Ops.where inside p (int p 0)) Weak_int in
    let stride d = numel (List.filteri (fun i _ -> i > d) s) in
    let term d =
      let v =
        if d = axis then at
        else laid t d (Shape.arange ~dtype:Weak_int (List.nth t d))
      in
      match stride d with 1 -> v | k -> Ops.mul v (Ops.int k)
    in
    let offset =
      List.fold_left
        (fun o d -> Ops.add o (term d))
        (term 0)
        (List.init (List.length s - 1) succ)
    in
    let read = Ops.index (Shape.reshape u [ Ops.Int (numel s) ]) [ offset ] in
    Ops.where inside read (int read 0)

let gather d ~axis idx x =
  Arith.of_parts d (List.map (fun u -> take u axis idx) (Arith.parts d x))

(* Scatters

   Each element of [into] meets each update along [axis] in a mask along a new
   last axis, true where the update's position is the element's: n x k work for
   n positions and k updates along [axis]. Updates that land on one element
   differ only along [axis], so C order of updates is their order along it. *)

(* [summed ca ~landed xs us] is, in the carrier [ca], the sum of each element of
   [xs] and the updates [us] that land on it, [landed] their mask along a last
   axis. It is [-0] where every term is, which a sum from tolk's [+0]
   accumulator would make [+0]. *)
let summed ca ~landed xs us =
  let r = Shape.ndim landed - 1 in
  let terms x u = joined r x (Ops.where landed u (int u 0)) in
  let sums =
    Lower_reduce.reduce (Monoid Sum) [ r ] ~src:ca ca
      (Arith.of_parts ca (List.map2 terms xs us))
  in
  let signed (x, u) y =
    if not (Dtype.is_float (Ops.dtype y)) then y
    else
      let other =
        Ops.bitwise_and landed (Ops.logical_not (negative_zero u))
      in
      let other = Shape.rop (Ops.cast other Uint8) Op.Max [ r ] in
      let x = Shape.reshape x (Shape.shape other) in
      let none = Ops.eq other (int other 0) in
      let all = Ops.bitwise_and (negative_zero x) none in
      Ops.where all (Shape.const_like y (`Float (-0.))) y
  in
  Arith.of_parts ca
    (List.map2 signed (List.combine xs us) (Arith.parts ca (List.hd sums)))

let scatter (D.Any dt as d) (c : Nx_kernel.combine) ~axis ~idx ~updates into =
  let s = extents (List.hd (Arith.parts d into)) and t = extents idx in
  let r = List.length s in
  let n = List.nth s axis and k = List.nth t axis in
  if n = 0 || numel t = 0 then into
  else
    let wide = s @ [ k ] in
    (* A node of [idx]'s shape moved to [wide], its axis [axis] last. *)
    let spread u =
      let others = List.filter (( <> ) axis) (List.init r Fun.id) in
      let one = List.mapi (fun i e -> if i = axis then 1 else e) s @ [ k ] in
      Shape.expand
        (Shape.reshape (Shape.permute u (others @ [ axis ])) (dims one))
        (dims wide)
    in
    let own u = Shape.reshape u (dims (s @ [ 1 ])) in
    let mask =
      Ops.eq (spread idx) (laid wide axis (Shape.arange ~dtype:Int64 n))
    in
    let landed =
      let m = Shape.rop (Ops.cast mask Uint8) Op.Max [ r ] in
      Ops.ne m (int m 0)
    in
    let kept y =
      Arith.of_parts d
        (List.map2 (Ops.where landed) (Arith.parts d y) (Arith.parts d into))
    in
    let extreme m =
      (* [into]'s element, first, then each update, the element standing in
         for one that does not land. *)
      let terms x u =
        let x = own x in
        joined r x (Ops.where mask (spread u) (Shape.expand x (dims wide)))
      in
      let xs = List.map2 terms (Arith.parts d into) (Arith.parts d updates) in
      let y = Lower_reduce.reduce (Monoid m) [ r ] ~src:d d in
      kept (List.hd (y (Arith.of_parts d xs)))
    in
    match c with
    | Set ->
        (* The last update to land, of highest position along [axis], wins:
           its bits are picked among zeros by an unsigned maximum. *)
        let order = laid wide r (Shape.arange ~dtype:Int64 k) in
        let latest =
          Shape.rop (Ops.where mask order (int order (-1))) Op.Max [ r ]
        in
        let last = Ops.eq order (Shape.expand (own latest) (dims wide)) in
        let pick u =
          let b = Bits.bits (spread u) in
          let b = Shape.rop (Ops.where last b (int b 0)) Op.Max [ r ] in
          Ops.bitcast b (Ops.dtype u)
        in
        kept (Arith.of_parts d (List.map pick (Arith.parts d updates)))
    | Max -> extreme Max
    | Min -> extreme Min
    | Add ->
        if D.is D.Boolean dt then unlowered "a boolean Add Scatter";
        let ca = Nx_kernel.carrier d in
        let lift u = Arith.parts ca (Arith.cast ~src:d ca u) in
        let xs = List.map own (lift into) in
        let us = List.map spread (lift updates) in
        kept (Arith.cast ~src:ca d (summed ca ~landed:mask xs us))
