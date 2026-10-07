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
    let n = String.length t in
    let negative = n > 0 && t.[0] = '-' in
    let i = if n > 0 && (t.[0] = '-' || t.[0] = '+') then 1 else 0 in
    let mantissa_end =
      match
        ( String.index_from_opt t i 'E',
          String.index_from_opt t i 'D',
          String.index_from_opt t i 'e',
          String.index_from_opt t i 'd' )
      with
      | a, b, c, d ->
          List.fold_left
            (fun m o -> match o with Some j -> Int.min m j | None -> m)
            n [ a; b; c; d ]
    in
    let mantissa = String.sub t i (mantissa_end - i) in
    let exponent =
      if mantissa_end = n then Some 0
      else
        int_of_string_opt
          ( String.sub t (mantissa_end + 1) (n - mantissa_end - 1) |> fun s ->
            if s <> "" && s.[0] = '+' then String.sub s 1 (String.length s - 1)
            else s )
    in
    match exponent with
    | None -> None
    | Some e ->
        let whole, frac =
          match String.index_opt mantissa '.' with
          | None -> (mantissa, "")
          | Some j ->
              ( String.sub mantissa 0 j,
                String.sub mantissa (j + 1) (String.length mantissa - j - 1) )
        in
        let digits = whole ^ frac in
        let e = e - String.length frac in
        let l = ref 0 and r = ref (String.length digits) in
        while !l < !r && digits.[!l] = '0' do
          incr l
        done;
        while !r > !l && digits.[!r - 1] = '0' do
          decr r
        done;
        if !l = !r then Some (false, "", 0)
        else
          Some
            ( negative,
              String.sub digits !l (!r - !l),
              e + (String.length digits - !r) )

let equal a b =
  match (normal a, normal b) with Some x, Some y -> x = y | _ -> false
