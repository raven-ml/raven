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

(* The quiet bit is the fraction's first, below the exponent's ones. *)
let quiet_bit x = snd (Dtype.finfo (Ops.dtype x)) - 1

let nan x =
  let e, m = Dtype.finfo (Ops.dtype x) in
  let ones = Bigint.pred (Bigint.shift_left Bigint.one (e + 1)) in
  of_bits x (word x (Bigint.shift_left ones (m - 1)))

let quieted x = of_bits x (Ops.bitwise_or (bits x) (bit x (quiet_bit x)))

(* Fused multiply-add *)

let mulacc a b c = Ops.alu a Op.Mulacc [ b; c ]
