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

let check_rank fn r =
  if r > max_rank then invalid_argf "%s: rank %d exceeds %d" fn r max_rank

(* A shape with a zero extent has no element, whatever its other extents: its
   product is 0 before any of them is multiplied. Each product is formed only
   once it is known to fit. *)
let numel fn s =
  Array.iter
    (fun d ->
      if d < 0 then invalid_argf "%s: extent %d of %a is negative" fn d pp s)
    s;
  if Array.mem 0 s then 0
  else begin
    let n = ref 1 in
    for i = 0 to Array.length s - 1 do
      let d = s.(i) in
      if !n > max_int / d then
        invalid_argf "%s: the number of elements of %a overflows" fn pp s;
      n := !n * d
    done;
    !n
  end
