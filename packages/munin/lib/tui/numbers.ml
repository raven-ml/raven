(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Float_print = Munin_float_text.Float_print
module Float_read = Munin_float_text.Float_read
module Float_text = Munin_float_text.Float_text

let width_method = `Unicode
let width s = Matrix_text.measure ~width_method ~tab_width:2 s

(* [take n s] is the longest prefix of [s] at most [n] cells wide. *)
let take n s =
  let p = Matrix_text.find_pos ~width_method ~tab_width:2 s ~columns:n in
  String.sub s 0 p.byte_offset

let cut w s =
  if width s <= w then s else if w <= 0 then "" else take (w - 1) s ^ "…"

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

(* Round numbers *)

let decimal m k =
  match Float_read.of_string (string_of_int m ^ "e" ^ string_of_int k) with
  | Some v -> v
  | None -> invalid_arg "Numbers.decimal"

let decade x =
  let d, e = Float_text.shortest Float_text.binary64 x in
  String.length d - 1 + e

(* [positional d e] is [d] × 10{^ [e]} without an exponent: [("15", -1)] is
   [1.5], [("12", 3)] is [12000]. *)
let positional d e =
  let n = String.length d in
  if e >= 0 then d ^ String.make e '0'
  else if -e < n then String.sub d 0 (n + e) ^ "." ^ String.sub d (n + e) (-e)
  else "0." ^ String.make (-e - n) '0' ^ d

(* [exponential d e] is [d] × 10{^ [e]} as C's [%e] writes it, one digit
   before the point and at least two in the exponent: [("15", 5)] is
   [1.5e+06]. *)
let exponential d e =
  let lead = String.length d - 1 + e in
  let n = String.length d in
  let m = if n = 1 then d else String.sub d 0 1 ^ "." ^ String.sub d 1 (n - 1) in
  let p = string_of_int (abs lead) in
  m ^ "e" ^ (if lead < 0 then "-" else "+") ^ (if String.length p < 2 then "0" ^ p else p)

(* Four significant digits, counted once [x] is rounded to them, with
   trailing zeros kept, so that values in a column line up digit for
   digit. *)
let value x =
  if not (Float.is_finite x) then non_finite x
  else if x = 0. then "0"
  else
    let d, e = Float_text.significant 4 x in
    let lead = String.length d - 1 + e in
    (* The four digits, trailing zeros put back. *)
    let d4 = d ^ String.make (4 - String.length d) '0' and e4 = lead - 3 in
    let sign = if Float.sign_bit x then "-" else "" in
    if lead < -4 || lead >= 6 then sign ^ exponential d4 e4
    else sign ^ positional d4 e4

let accuracy key =
  let k = String.lowercase_ascii key in
  let n = String.length k in
  let rec has i = i + 3 <= n && (String.sub k i 3 = "acc" || has (i + 1)) in
  has 0

let as_percent key x = accuracy key && x >= 0. && x <= 1.

let metric key x =
  if as_percent key x then Float_print.f 2 (x *. 100.) ^ "%" else value x

let given x = if Float.is_finite x then Float_print.exact x else non_finite x
let signed x = if x > 0. then "+" ^ value x else value x

(* [plus s] is the printed number [s] with its sign, as C's [%+] gives it. *)
let plus s = if String.starts_with ~prefix:"-" s then s else "+" ^ s

let difference key ~value:v ~other:o =
  if not (Float.is_finite v && Float.is_finite o) then None
  else if as_percent key v && as_percent key o then
    let s = plus (Float_print.f 2 ((v -. o) *. 100.)) ^ " pt" in
    if s = "+0.00 pt" || s = "-0.00 pt" then None else Some s
  else if o = 0. then if v = 0. then None else Some (signed v)
  else
    let r = 100. *. (v -. o) /. Float.abs o in
    let s =
      plus (Float_print.f (if Float.abs r >= 10. then 0 else 1) r) ^ "%"
    in
    if s = "+0.0%" || s = "-0.0%" then None else Some s

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

(* [x] in the fewest significant digits that read back as [x], at most
   [most], and never fewer than its integer part's, which an exponent would
   take: an axis label names its tick's value. *)
let exact ~most x =
  let a = Float.abs x in
  let whole = if a >= 1. then decade a + 1 else 1 in
  let rec go p =
    let s = Float_print.g ~precision:p x in
    if p >= most || Float_read.of_string s = Some x then s else go (p + 1)
  in
  go (max 1 (min most whole))

let compact x =
  if not (Float.is_finite x) then non_finite x
  else
    let a = Float.abs x in
    let short v unit = exact ~most:4 v ^ unit in
    if a < 1000. then exact ~most:4 x
    else if a < 1e6 then short (x /. 1e3) "k"
    else if a < 1e9 then short (x /. 1e6) "M"
    else short (x /. 1e9) "G"

(* The columns of a y tick. *)
let tick_width = 6

(* A y tick's notations. Each writes the digits [d] and exponent [e] of the
   shortest decimal that reads back as the tick ({!Float_text}), so every
   label gives its tick exactly. *)
type notation = Plain | Suffix | Exponent

(* The suffixes by the power of ten they stand for. *)
let suffixes = [ (9, "G"); (6, "M"); (3, "k") ]

let digits notation d e =
  match notation with
  | Plain -> positional d e
  | Suffix -> (
      (* The power of ten of the leading digit. *)
      let lead = String.length d - 1 + e in
      match List.find_opt (fun (k, _) -> lead >= k) suffixes with
      | Some (k, unit) -> positional d (e - k) ^ unit
      | None -> positional d e)
  | Exponent -> exponential d e

let label notation x =
  if x = 0. then "0"
  else
    let d, e = Float_text.shortest Float_text.binary64 x in
    (if Float.sign_bit x then "-" else "") ^ digits notation d e

let tick xs =
  let fits n = List.for_all (fun x -> Float.is_finite x && width (label n x) <= tick_width) xs in
  Option.map label (List.find_opt fits [ Plain; Suffix; Exponent ])

let percent_of s t =
  if t <= 0 then ""
  else Printf.sprintf "%d%%" (if s >= t then 100 else max 0 s * 100 / t)

let percent x =
  if x >= 10. then Float_print.f 0 x ^ "%" else Float_print.f 1 x ^ "%"

let gib b = Float_print.f 1 (b /. 1073741824.)

let span s =
  let s = Float.to_int (Float.max 0. s) in
  if s < 60 then Printf.sprintf "%ds" s
  else if s < 3600 then Printf.sprintf "%dm %02ds" (s / 60) (s mod 60)
  else if s < 86400 then Printf.sprintf "%dh %02dm" (s / 3600) (s / 60 mod 60)
  else Printf.sprintf "%dd %dh" (s / 86400) (s / 3600 mod 24)

let months =
  [| "Jan"; "Feb"; "Mar"; "Apr"; "May"; "Jun"; "Jul"; "Aug"; "Sep"; "Oct";
     "Nov"; "Dec" |]

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
