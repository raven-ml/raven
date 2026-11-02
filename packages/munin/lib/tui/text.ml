(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Float_text = Munin_float_text.Float_text

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

let digits p x =
  if Float.is_finite x then Float_text.g ~precision:p x else non_finite x

let value x = digits 4 x

(* Rounding is monotonic, so two values that print alike enclose only values
   that print as they do: checking each sorted value against the next tells
   every pair apart. *)
let values xs =
  let finite = List.sort_uniq Float.compare (List.filter Float.is_finite xs) in
  let a = Array.of_list finite in
  let distinct p =
    let ok = ref true and i = ref 0 in
    while !ok && !i + 1 < Array.length a do
      if String.equal (digits p a.(!i)) (digits p a.(!i + 1)) then ok := false;
      incr i
    done;
    !ok
  in
  let rec go p = if p >= 17 || distinct p then p else go (p + 1) in
  let p = go 4 in
  List.map (digits p) xs

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

let short n =
  if abs n < 1000 then string_of_int n
  else if abs n < 1_000_000 then
    Float_text.g ~precision:3 (Float.of_int n /. 1e3) ^ "k"
  else Float_text.g ~precision:3 (Float.of_int n /. 1e6) ^ "M"

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

let age s =
  let s = Float.to_int (Float.max 0. s) in
  if s < 60 then Printf.sprintf "%ds" s
  else if s < 3600 then Printf.sprintf "%dm" (s / 60)
  else if s < 86400 then Printf.sprintf "%dh" (s / 3600)
  else Printf.sprintf "%dd" (s / 86400)

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

let gib b = Float_text.f 1 (b /. 1073741824.)

let percent x =
  if x >= 10. then Float_text.f 0 x ^ "%" else Float_text.f 1 x ^ "%"

let mbs b = Float_text.g ~precision:2 (b /. 1e6)
let eighths = [| ""; "▏"; "▎"; "▍"; "▌"; "▋"; "▊"; "▉" |]

let bar w p =
  let p = if Float.is_nan p then 0. else Float.max 0. (Float.min 1. p) in
  let n = Float.to_int (Float.round (p *. Float.of_int (8 * w))) in
  let full = n / 8 and part = n mod 8 in
  let b = Buffer.create (w * 3) in
  for _ = 1 to full do
    Buffer.add_string b "█"
  done;
  if full < w then begin
    Buffer.add_string b (if part = 0 then " " else eighths.(part));
    Buffer.add_string b (String.make (w - full - 1) ' ')
  end;
  "▕" ^ Buffer.contents b ^ "▏"
