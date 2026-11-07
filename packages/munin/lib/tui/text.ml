(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Float_print = Munin_float_text.Float_print

(* Widths are matrix's, the terminal's cells. *)
let width_method = `Unicode
let width s = Matrix_text.measure ~width_method ~tab_width:2 s
let pad w s = if width s >= w then s else s ^ String.make (w - width s) ' '
let pad_left w s = if width s >= w then s else String.make (w - width s) ' ' ^ s

(* [take n s] is the longest prefix of [s] at most [n] cells wide. *)
let take n s =
  let p = Matrix_text.find_pos ~width_method ~tab_width:2 s ~columns:n in
  String.sub s 0 p.byte_offset

let cut w s =
  if width s <= w then s else if w <= 0 then "" else take (w - 1) s ^ "…"

let fit w s = pad w (cut w s)

let elide w s =
  if width s <= w then s
  else
    match String.rindex_opt s '/' with
    | Some k when width s - width (String.sub s 0 k) + 2 <= w ->
        let last = String.sub s k (String.length s - k) in
        take (w - 1 - width last) s ^ "…" ^ last
    | _ -> cut w s

let non_finite x =
  if Float.is_nan x then "NaN" else if x > 0. then "inf" else "-inf"

(* Four significant digits with trailing zeros kept, so that values in a column
   line up digit for digit; scientific below 1e-4 and from 1e6, so that no value
   that is not zero prints as [0]. *)
let value x =
  if not (Float.is_finite x) then non_finite x
  else if x = 0. then "0"
  else
    let a = Float.abs x in
    if a < 1e-4 || a >= 1e6 then
      (* [Float_print.g] drops trailing zeros: they are put back in the
         mantissa, so that it has four digits. *)
      let s = Float_print.g ~precision:4 x in
      match String.index_opt s 'e' with
      | None -> s
      | Some i ->
          let m = String.sub s 0 i
          and rest = String.sub s i (String.length s - i) in
          let digits =
            String.length m
            - (if m.[0] = '-' then 1 else 0)
            - if String.contains m '.' then 1 else 0
          in
          let m = if String.contains m '.' then m else m ^ "." in
          m ^ String.make (max 0 (4 - digits)) '0' ^ rest
    else
      let e = Float.to_int (Float.floor (Float.log10 a)) in
      Float_print.f (max 0 (3 - e)) x

let signed x = if x > 0. then "+" ^ value x else value x

let percent_of s t =
  if t <= 0 then ""
  else Printf.sprintf "%d%%" (if s >= t then 100 else max 0 s * 100 / t)

let given x = if Float.is_finite x then Float_print.exact x else non_finite x

let count n =
  let s = string_of_int (abs n) in
  let l = String.length s in
  let b = Buffer.create (l + (l / 3) + 1) in
  if n < 0 then Buffer.add_char b '-';
  String.iteri
    (fun i c ->
      if i > 0 && (l - i) mod 3 = 0 then Buffer.add_char b ',';
      Buffer.add_char b c)
    s;
  Buffer.contents b

let duration s =
  let s = Float.to_int (Float.max 0. s) in
  if s < 60 then Printf.sprintf "%ds" s
  else if s < 3600 then
    if s mod 60 = 0 || s >= 600 then Printf.sprintf "%dm" (s / 60)
    else Printf.sprintf "%dm%02ds" (s / 60) (s mod 60)
  else if s < 86400 then
    let m = s / 60 mod 60 in
    if m = 0 then Printf.sprintf "%dh" (s / 3600)
    else Printf.sprintf "%dh%02dm" (s / 3600) m
  else
    let h = s / 3600 mod 24 in
    if h = 0 then Printf.sprintf "%dd" (s / 86400)
    else Printf.sprintf "%dd%dh" (s / 86400) h

let months =
  [|
    "Jan";
    "Feb";
    "Mar";
    "Apr";
    "May";
    "Jun";
    "Jul";
    "Aug";
    "Sep";
    "Oct";
    "Nov";
    "Dec";
  |]

let clock ~now t =
  let tm = Unix.localtime t and tn = Unix.localtime now in
  if tm.tm_yday = tn.tm_yday && tm.tm_year = tn.tm_year then
    Printf.sprintf "%02d:%02d" tm.tm_hour tm.tm_min
  else
    Printf.sprintf "%s %d %02d:%02d" months.(tm.tm_mon) tm.tm_mday tm.tm_hour
      tm.tm_min

let date t =
  let tm = Unix.localtime t in
  Printf.sprintf "%s %d %02d:%02d:%02d" months.(tm.tm_mon) tm.tm_mday tm.tm_hour
    tm.tm_min tm.tm_sec

let gib b = Float_print.f 1 (b /. 1073741824.)

let percent x =
  if x >= 10. then Float_print.f 0 x ^ "%" else Float_print.f 1 x ^ "%"

let mbs b = Float_print.g ~precision:2 (b /. 1e6)
