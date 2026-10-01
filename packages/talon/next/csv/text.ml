(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A1 = Bigarray.Array1

type int64s = (int64, Bigarray.int64_elt, Bigarray.c_layout) A1.t
type float64s = (float, Bigarray.float64_elt, Bigarray.c_layout) A1.t

exception Invalid of string

let invalid why = raise_notrace (Invalid why)
let is_digit c = '0' <= c && c <= '9'
let digit b i = Char.code (Bytes.unsafe_get b i) - 48

let rec equals_from ~caseless b pos s i =
  i = String.length s
  ||
  let c = Bytes.unsafe_get b (pos + i) in
  let c = if caseless then Char.lowercase_ascii c else c in
  c = String.unsafe_get s i && equals_from ~caseless b pos s (i + 1)

let equals ~caseless b pos len s =
  len = String.length s && equals_from ~caseless b pos s 0

let is_minus b pos len = len > 0 && Bytes.unsafe_get b pos = '-'

let after_sign b pos len =
  if len > 0 && (Bytes.unsafe_get b pos = '-' || Bytes.unsafe_get b pos = '+')
  then pos + 1
  else pos

let check_digits b first stop =
  if first = stop then invalid "not an integer";
  for i = first to stop - 1 do
    if not (is_digit (Bytes.unsafe_get b i)) then invalid "not an integer"
  done

(* Booleans and numbers *)

let bool b pos len =
  if equals ~caseless:false b pos len "true" then true
  else if equals ~caseless:false b pos len "false" then false
  else invalid "not true or false"

let int ~min ~max b pos len =
  let first = after_sign b pos len and stop = pos + len in
  check_digits b first stop;
  (* Past 2^40 the value is out of every range, so it stops growing. *)
  let v = ref 0 in
  for i = first to stop - 1 do
    if !v < 1 lsl 40 then v := (10 * !v) + digit b i
  done;
  let v = if is_minus b pos len then - !v else !v in
  if v < min || v > max then invalid "out of range";
  v

(* [greater b first s i] is [true] iff the digits at [first] are greater than
   the digits [s] from [i], of equal length. *)
let rec greater b first s i =
  i < String.length s
  &&
  let c = Bytes.unsafe_get b (first + i) and d = String.unsafe_get s i in
  c > d || (c = d && greater b first s (i + 1))

(* [wide ~max ~min b pos len a i] reads an integer whose magnitude is at most
   the digits [max], or [min] when negative, and stores it modulo 2^64. Digit
   strings of equal length compare as their values. *)
let wide ~max ~min b pos len a i =
  let first = after_sign b pos len and stop = pos + len in
  check_digits b first stop;
  let first = ref first in
  while !first < stop - 1 && Bytes.unsafe_get b !first = '0' do
    incr first
  done;
  let first = !first in
  let limit = if is_minus b pos len then min else max in
  let n = stop - first and m = String.length limit in
  if n > m || (n = m && greater b first limit 0) then invalid "out of range";
  let v = ref 0L in
  for k = first to stop - 1 do
    v := Int64.add (Int64.mul !v 10L) (Int64.of_int (digit b k))
  done;
  A1.unsafe_set a i (if is_minus b pos len then Int64.neg !v else !v)

let int64 b pos len a i =
  wide ~max:"9223372036854775807" ~min:"9223372036854775808" b pos len a i

let uint64 b pos len a i =
  wide ~max:"18446744073709551615" ~min:"0" b pos len a i

let pow10 = Array.init 23 (fun e -> float_of_string ("1e" ^ string_of_int e))

(* The fast path is exact: a significand of at most 2^53 and a power of ten of
   at most 10^22 are exact doubles, and the product or quotient rounds once.
   Past 2^53 the significand stops growing, which leaves the fast path. *)
let float b pos len a k =
  let stop = pos + len in
  let first = after_sign b pos len in
  let rest = stop - first in
  if
    equals ~caseless:true b first rest "inf"
    || equals ~caseless:true b first rest "infinity"
  then A1.unsafe_set a k (if is_minus b pos len then neg_infinity else infinity)
  else if equals ~caseless:true b first rest "nan" then A1.unsafe_set a k nan
  else begin
    let i = ref first and m = ref 0 and exp = ref 0 in
    let frac = ref false and any = ref false in
    while
      !i < stop
      &&
      let c = Bytes.unsafe_get b !i in
      is_digit c || (c = '.' && not !frac)
    do
      if Bytes.unsafe_get b !i = '.' then frac := true
      else begin
        any := true;
        if !m <= 1 lsl 53 then begin
          m := (10 * !m) + digit b !i;
          if !frac then decr exp
        end
      end;
      incr i
    done;
    if not !any then invalid "not a number";
    if !i < stop && (Bytes.unsafe_get b !i = 'e' || Bytes.unsafe_get b !i = 'E')
    then begin
      incr i;
      let negative = !i < stop && Bytes.unsafe_get b !i = '-' in
      if !i < stop && (negative || Bytes.unsafe_get b !i = '+') then incr i;
      if not (!i < stop && is_digit (Bytes.unsafe_get b !i)) then
        invalid "not a number";
      let e = ref 0 in
      while !i < stop && is_digit (Bytes.unsafe_get b !i) do
        if !e < 100_000 then e := (10 * !e) + digit b !i;
        incr i
      done;
      exp := if negative then !exp - !e else !exp + !e
    end;
    if !i <> stop then invalid "not a number";
    if !m <= 1 lsl 53 && -22 <= !exp && !exp <= 22 then begin
      let v = Float.of_int !m in
      let v = if !exp >= 0 then v *. pow10.(!exp) else v /. pow10.(- !exp) in
      A1.unsafe_set a k (if is_minus b pos len then -.v else v)
    end
    else A1.unsafe_set a k (float_of_string (Bytes.sub_string b pos len))
  end

let int_pow10 = Array.init 19 (fun e -> int_of_float pow10.(e))

let too_many precision =
  invalid (Printf.sprintf "more than %d digits" precision)

let decimal ~precision ~scale b pos len =
  let stop = pos + len in
  let limit = int_pow10.(precision) in
  let i = ref (after_sign b pos len) and v = ref 0 and frac = ref (-1) in
  let any = ref false in
  while !i < stop do
    let c = Bytes.unsafe_get b !i in
    if c = '.' && !frac < 0 then frac := 0
    else if is_digit c then begin
      any := true;
      if !frac >= 0 then incr frac;
      if !frac <= scale then begin
        v := (10 * !v) + digit b !i;
        if !v >= limit then too_many precision
      end
      else if c <> '0' then
        invalid (Printf.sprintf "more than %d digits after the point" scale)
    end
    else invalid "not a decimal number";
    incr i
  done;
  if not !any then invalid "not a decimal number";
  for _ = Int.max 0 !frac + 1 to scale do
    if !v >= limit / 10 then too_many precision;
    v := 10 * !v
  done;
  if is_minus b pos len then - !v else !v

(* Dates and datetimes *)

let two_digits b i =
  let c0 = Bytes.unsafe_get b i and c1 = Bytes.unsafe_get b (i + 1) in
  if is_digit c0 && is_digit c1 then (10 * digit b i) + digit b (i + 1) else -1

let is_leap y = (y mod 4 = 0 && y mod 100 <> 0) || y mod 400 = 0

let days_in_month y m =
  match m with
  | 2 -> if is_leap y then 29 else 28
  | 4 | 6 | 9 | 11 -> 30
  | _ -> 31

(* Howard Hinnant's days_from_civil, for years from 0. *)
let days_from_civil y m d =
  let y = if m <= 2 then y - 1 else y in
  let era = (if y >= 0 then y else y - 399) / 400 in
  let yoe = y - (era * 400) in
  let doy = (((153 * ((m + 9) mod 12)) + 2) / 5) + d - 1 in
  let doe = (yoe * 365) + (yoe / 4) - (yoe / 100) + doy in
  (era * 146097) + doe - 719468

(* [date_at b pos] reads [YYYY-MM-DD] at [pos], whose ten bytes exist. *)
let date_at b pos =
  let y0 = two_digits b pos and y1 = two_digits b (pos + 2) in
  let m = two_digits b (pos + 5) and d = two_digits b (pos + 8) in
  if
    y0 < 0 || y1 < 0 || m < 0 || d < 0
    || Bytes.unsafe_get b (pos + 4) <> '-'
    || Bytes.unsafe_get b (pos + 7) <> '-'
  then invalid "not YYYY-MM-DD";
  let y = (100 * y0) + y1 in
  if m < 1 || m > 12 || d < 1 || d > days_in_month y m then
    invalid "not a day of the calendar";
  days_from_civil y m d

let date b pos len =
  if len <> 10 then invalid "not YYYY-MM-DD";
  date_at b pos

let per_second : Talon_next.Type.unit_ -> int = function
  | S -> 1
  | Ms -> 1_000
  | Us -> 1_000_000
  | Ns -> 1_000_000_000

let not_whole : Talon_next.Type.unit_ -> string = function
  | S -> "not a whole number of seconds"
  | Ms -> "not a whole number of milliseconds"
  | Us -> "not a whole number of microseconds"
  | Ns -> "not a whole number of nanoseconds"

let form ~zoned =
  invalid
    (if zoned then "not YYYY-MM-DDThh:mm:ss with an offset"
     else "not YYYY-MM-DDThh:mm:ss without an offset")

let datetime u ~zoned b pos len a k =
  let stop = pos + len in
  if len < 19 then form ~zoned;
  let days = date_at b pos in
  let t = Bytes.unsafe_get b (pos + 10) in
  let hh = two_digits b (pos + 11) and mm = two_digits b (pos + 14) in
  let ss = two_digits b (pos + 17) in
  if
    (t <> 'T' && t <> ' ')
    || hh < 0 || mm < 0 || ss < 0
    || Bytes.unsafe_get b (pos + 13) <> ':'
    || Bytes.unsafe_get b (pos + 16) <> ':'
  then form ~zoned;
  if hh > 23 || mm > 59 || ss > 59 then invalid "not a time of day";
  let i = ref (pos + 19) and ns = ref 0 in
  if !i < stop && Bytes.unsafe_get b !i = '.' then begin
    incr i;
    let first = !i in
    while !i < stop && !i - first < 9 && is_digit (Bytes.unsafe_get b !i) do
      ns := (10 * !ns) + digit b !i;
      incr i
    done;
    if !i = first then form ~zoned;
    for _ = !i - first to 8 do
      ns := 10 * !ns
    done
  end;
  let offset =
    if not zoned then 0
    else if !i < stop && Bytes.unsafe_get b !i = 'Z' then begin
      incr i;
      0
    end
    else if !i + 6 <= stop then begin
      let s = Bytes.unsafe_get b !i in
      let oh = two_digits b (!i + 1) and om = two_digits b (!i + 4) in
      if
        (s <> '+' && s <> '-')
        || oh < 0 || om < 0
        || Bytes.unsafe_get b (!i + 3) <> ':'
      then form ~zoned;
      if oh > 23 || om > 59 then invalid "not an offset";
      i := !i + 6;
      let o = (3600 * oh) + (60 * om) in
      if s = '-' then -o else o
    end
    else form ~zoned
  in
  if !i <> stop then form ~zoned;
  let secs = (86400 * days) + (3600 * hh) + (60 * mm) + ss - offset in
  let per = per_second u in
  let div = 1_000_000_000 / per in
  if !ns mod div <> 0 then invalid (not_whole u);
  if per < 1_000_000_000 then
    A1.unsafe_set a k (Int64.of_int ((secs * per) + (!ns / div)))
  else begin
    if secs > 9223372036 || secs < -9223372037 then invalid "out of range";
    (* In this range the ticks wrap at most once, away from the sign of
       [secs]. *)
    let v =
      Int64.add
        (Int64.mul (Int64.of_int secs) 1_000_000_000L)
        (Int64.of_int !ns)
    in
    if secs >= 0 <> (Int64.compare v 0L >= 0) then invalid "out of range";
    A1.unsafe_set a k v
  end

(* Text *)

let utf_8 b pos len =
  let stop = pos + len and i = ref pos in
  while !i < stop do
    if Bytes.unsafe_get b !i < '\x80' then incr i
    else begin
      let d = Bytes.get_utf_8_uchar b !i in
      let n = Uchar.utf_decode_length d in
      if (not (Uchar.utf_decode_is_valid d)) || !i + n > stop then
        invalid "not valid UTF-8";
      i := !i + n
    end
  done

let leading_zero b pos len =
  let i = after_sign b pos len in
  i + 1 < pos + len
  && Bytes.unsafe_get b i = '0'
  && is_digit (Bytes.unsafe_get b (i + 1))
