(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let max_rank = 32
let max_operands = 18
let max_numel = 1 lsl 53
let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let pp ppf a =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_seq
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Format.pp_print_int)
    (Iarray.to_seq a)

(* Array.make and Array.copy are calls into the runtime's C, about 8 ns each on
   the M1 Max; an array literal allocates inline. A shape of up to four axes
   takes a literal. *)
let zeros = function
  | 0 -> [||]
  | 1 -> [| 0 |]
  | 2 -> [| 0; 0 |]
  | 3 -> [| 0; 0; 0 |]
  | 4 -> [| 0; 0; 0; 0 |]
  | r -> Array.make r 0

(* An array and an iarray of one element type have one representation. The
   standard library's Iarray converts between them by this primitive
   (iarray.ml's [unsafe_of_array]) and exports only conversions that copy. A
   layout's strides are formed in an array that nothing writes afterwards, so
   freezing it in place saves the copy, a call into the runtime's C. *)
external freeze : int array -> int iarray = "%opaque"

let check_rank fn r =
  if r > max_rank then invalid_argf "%s: rank %d exceeds %d" fn r max_rank

(* The product of [s]'s extents, or the first refusal its scan meets: [-1] at
   a negative extent, [-2] where the extents other than 0 multiply past
   [max_numel]. A shape with a zero extent has no element, but the product of
   its other extents is bounded all the same: every product of a layout's
   extents then fits, whichever of them a kernel multiplies. Each product is
   formed only once it is known to fit. *)
let size s =
  let r = Iarray.length s and n = ref 1 and zero = ref false and i = ref 0 in
  while !i < r && !n > 0 do
    let d = Iarray.unsafe_get s !i in
    if d < 0 then n := -1
    else if d = 0 then zero := true
    (* Two factors below 2^26 multiply to at most max_numel: the division,
       tens of cycles, runs only for larger ones. *)
    else if !n lor d >= 1 lsl 26 && d > max_numel / !n then n := -2
    else n := !n * d;
    incr i
  done;
  if !n > 0 && !zero then 0 else !n

let numel fn s =
  let n = size s in
  if n >= 0 then n
  else if n = -2 then
    invalid_argf "%s: the extents of %a other than 0 multiply past %d" fn pp s
      max_numel
  else
    let i = ref 0 in
    while Iarray.get s !i >= 0 do
      incr i
    done;
    invalid_argf "%s: axis %d of %a is %d, below 0" fn !i pp s
      (Iarray.get s !i)

let fits s = Iarray.length s <= max_rank && size s >= 0
