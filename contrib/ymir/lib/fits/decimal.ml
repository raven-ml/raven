(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* FITS numbers compared as the exact decimals they write: 32768, 3.2768E4
   and 32768.0 are one number, and 9223372036854775807 is not 2^63. *)

(* [normal t] is [(negative, digits, exponent)] with [t = digits × 10^exponent],
   [digits] without leading or trailing zeros, and zero as [(false, "", 0)]. *)
let normal t =
  if not (Value.real_grammar t) then None
  else
    match Exact.split t with
    | Some (_, "", _) -> Some (false, "", 0)
    | r -> r

let equal a b =
  match (normal a, normal b) with Some x, Some y -> x = y | _ -> false
