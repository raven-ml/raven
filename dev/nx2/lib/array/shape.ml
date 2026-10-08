(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let max_rank = 32
let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let pp ppf a =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_array
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Format.pp_print_int)
    a

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

let copy s =
  let r = Array.length s in
  let s' = zeros r in
  for i = 0 to r - 1 do
    Array.unsafe_set s' i (Array.unsafe_get s i)
  done;
  s'

let check_rank fn r =
  if r > max_rank then invalid_argf "%s: rank %d exceeds %d" fn r max_rank

(* A shape with a zero extent has no element, whatever its other extents: its
   product is 0 before any of them is multiplied. Each product is formed only
   once it is known to fit. *)
let numel fn s =
  let r = Array.length s and zero = ref false in
  for i = 0 to r - 1 do
    let d = Array.unsafe_get s i in
    if d < 0 then invalid_argf "%s: axis %d of %a is %d, below 0" fn i pp s d;
    if d = 0 then zero := true
  done;
  if !zero then 0
  else begin
    let n = ref 1 in
    for i = 0 to r - 1 do
      let d = Array.unsafe_get s i in
      if !n > max_int / d then
        invalid_argf "%s: the number of elements of %a overflows" fn pp s;
      n := !n * d
    done;
    !n
  end
