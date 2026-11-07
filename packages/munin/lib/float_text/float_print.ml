(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let special x =
  if Float.is_nan x then Some "nan"
  else if x = Float.infinity then Some "inf"
  else if x = Float.neg_infinity then Some "-inf"
  else None

let sign x = if Float.sign_bit x then "-" else ""

(* [layout p (d, e)] writes the nonzero decimal [d 10^e] of at most [p]
   digits as [%.*g] does. *)
let layout p (d, e) =
  let n = String.length d in
  let x = n - 1 + e in
  if x < -4 || x >= p then
    let fraction = if n > 1 then "." ^ String.sub d 1 (n - 1) else "" in
    Printf.sprintf "%c%se%c%02d" d.[0] fraction
      (if x < 0 then '-' else '+')
      (Int.abs x)
  else if e >= 0 then d ^ String.make e '0'
  else if x >= 0 then String.sub d 0 (x + 1) ^ "." ^ String.sub d (x + 1) (-e)
  else "0." ^ String.make (-x - 1) '0' ^ d

let g ?(precision = 6) x =
  match special x with
  | Some s -> s
  | None ->
      if x = 0. then sign x ^ "0"
      else
        let p = Int.max 1 precision in
        sign x ^ layout p (Float_text.significant p x)

let f decimals x =
  match special x with
  | Some s -> s
  | None ->
      let decimals = Int.max 0 decimals in
      let d, e = Float_text.decimals decimals x in
      (* The value times [10^decimals], as an integer of at least [decimals +
         1] digits. [e] is at least [-decimals]. *)
      let scaled = d ^ String.make (e + decimals) '0' in
      let scaled =
        String.make (Int.max 0 (decimals + 1 - String.length scaled)) '0'
        ^ scaled
      in
      let int = String.length scaled - decimals in
      sign x ^ String.sub scaled 0 int
      ^ if decimals = 0 then "" else "." ^ String.sub scaled int decimals

let exact x =
  match special x with
  | Some s -> s
  | None ->
      if x = 0. then sign x ^ "0"
      else
        let ((d, _) as v) = Float_text.shortest Float_text.binary64 x in
        sign x ^ layout (Int.max 15 (String.length d)) v

let exact_float x =
  let t = exact x in
  if String.exists (fun c -> c = '.' || c = 'e') t then t else t ^ ".0"
