(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk

(* Types *)

let unsigned t : Dtype.t =
  match Dtype.itemsize t with
  | 8 -> Uint64
  | 4 -> Uint32
  | 2 -> Uint16
  | 1 -> Uint8
  | n -> invalid_arg (Printf.sprintf "Lower_bits.unsigned: a width of %d" n)

let signed t : Dtype.t =
  match Dtype.itemsize t with
  | 8 -> Int64
  | 4 -> Int32
  | 2 -> Int16
  | 1 -> Int8
  | n -> invalid_arg (Printf.sprintf "Lower_bits.signed: a width of %d" n)

let extent u i =
  match List.nth (Shape.shape u) i with
  | Ops.Int n -> n
  | Ops.Sym _ -> invalid_arg "Lower_bits.extent: a symbolic extent"

(* Bits *)

let bits x = Ops.bitcast x (unsigned (Ops.dtype x))
let width x = 8 * Dtype.itemsize (Ops.dtype x)
let word x n = Shape.const_like (bits x) (`Int n)
let bit x k = word x (Bigint.shift_left Bigint.one k)
let sign x = bit x (width x - 1)
let of_bits x u = Ops.bitcast u (Ops.dtype x)
let negative x = Ops.ne (Ops.bitwise_and (bits x) (sign x)) (word x Bigint.zero)
let negated x = of_bits x (Ops.bitwise_xor (bits x) (sign x))

let magnitude x =
  of_bits x
    (Ops.bitwise_and (bits x)
       (word x (Bigint.pred (Bigint.shift_left Bigint.one (width x - 1)))))

(* NaN *)

let is_nan x = Ops.ne x x

(* Sets the [e] exponent bits and the quiet bit, the fraction's top bit. These
   are [e + 1] ones shifted left past the [m - 1] fraction bits below. *)
let nan x =
  let e, m = Dtype.finfo (Ops.dtype x) in
  let ones = Bigint.pred (Bigint.shift_left Bigint.one (e + 1)) in
  of_bits x (word x (Bigint.shift_left ones (m - 1)))

(* Fused multiply-add *)

let mulacc a b c = Ops.alu a Op.Mulacc [ b; c ]

(* Indexed loads

   A load reads the operand's elements, flattened, at each element's offset:
   its own index's along the other axes, its position's along [axis]. A
   position outside the axis is read at [0], so that every read is in bounds,
   and selected as zero. *)

let take u axis p =
  let extents u = List.init (Shape.ndim u) (extent u) in
  let numel s = List.fold_left ( * ) 1 s in
  let int u n = Shape.const_like u (`Int (Bigint.of_int n)) in
  let dims s = List.map (fun n -> Ops.Int n) s in
  let s = extents u and t = extents p in
  let n = List.nth s axis in
  if n = 0 || numel t = 0 then
    let z = Ops.const ~dtype:(Ops.dtype u) (`Int Bigint.zero) in
    Shape.expand (Shape.reshape z (List.map (fun _ -> Ops.Int 1) t)) (dims t)
  else
    let inside =
      Ops.lt (Ops.bitcast p Uint64)
        (Shape.const_like ~dtype:Uint64 p (`Int (Bigint.of_int n)))
    in
    let at = Ops.cast (Ops.where inside p (int p 0)) Weak_int in
    let stride d = numel (List.filteri (fun i _ -> i > d) s) in
    (* [laid d] is the arange of [t]'s axis [d] laid along it. *)
    let laid d =
      let e = List.nth t d in
      let along = List.mapi (fun i x -> if i = d then x else 1) t in
      Shape.expand
        (Shape.reshape (Shape.arange ~dtype:Weak_int e) (dims along))
        (dims t)
    in
    let term d =
      let v = if d = axis then at else laid d in
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
