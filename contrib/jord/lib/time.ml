(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let leap y = (y mod 4 = 0 && y mod 100 <> 0) || y mod 400 = 0

let last_day y m =
  match m with
  | 2 -> if leap y then 29 else 28
  | 4 | 6 | 9 | 11 -> 30
  | _ -> 31

(* Days from 1970-01-01 to a day of the proleptic Gregorian calendar
   (Hinnant, "chrono-Compatible Low-Level Date Algorithms", days_from_civil). *)
let days_from_civil y m d =
  let y = if m <= 2 then y - 1 else y in
  let era = (if y >= 0 then y else y - 399) / 400 in
  let yoe = y - (era * 400) in
  let doy = (((153 * if m > 2 then m - 3 else m + 9) + 2) / 5) + d - 1 in
  let doe = (yoe * 365) + (yoe / 4) - (yoe / 100) + doy in
  (era * 146097) + doe - 719468

(* Past 2^28 years a day's instant, some 2^53 seconds, no longer holds
   every second. *)
let of_date (y, m, d) =
  if m < 1 || m > 12 || d < 1 || d > last_day y m || abs y >= 1 lsl 28 then
    invalid_arg
      (Printf.sprintf "Jord.Time.of_date: %d-%02d-%02d is not a day within \
                       2^28 years"
         y m d);
  Float.of_int (days_from_civil y m d * 86400)

(* The day [z] days from 1970-01-01 (Hinnant, civil_from_days). *)
let civil_from_days z =
  let z = z + 719468 in
  let era = (if z >= 0 then z else z - 146096) / 146097 in
  let doe = z - (era * 146097) in
  let yoe = (doe - (doe / 1460) + (doe / 36524) - (doe / 146096)) / 365 in
  let doy = doe - ((365 * yoe) + (yoe / 4) - (yoe / 100)) in
  let mp = ((5 * doy) + 2) / 153 in
  let d = doy - (((153 * mp) + 2) / 5) + 1 in
  let m = if mp < 10 then mp + 3 else mp - 9 in
  ((yoe + (era * 400) + if m <= 2 then 1 else 0), m, d)

(* The day of [t], counted from 1970-01-01. *)
let day fn t =
  let z = Float.floor (t /. 86400.) in
  (* civil_from_days adds 719468 days before it divides. *)
  if not (Float.abs z < 0x1p61) then
    invalid_arg (Printf.sprintf "Jord.Time.%s: instant %h" fn t);
  Float.to_int z

let date t = civil_from_days (day "date" t)

let day_of_year t =
  let z = day "day_of_year" t in
  let y, _, _ = civil_from_days z in
  z - days_from_civil y 1 1 + 1

(* A week runs from Monday to Sunday, and belongs to the year of its
   Thursday: the year whose first week holds its first Thursday
   (ISO 8601:2004, 2.2.10, 3.2.2). 1970-01-01 is a Thursday. *)
let iso_week t =
  let z = day "iso_week" t in
  let weekday = (((z + 3) mod 7) + 7) mod 7 in
  let thursday = z - weekday + 3 in
  let y, _, _ = civil_from_days thursday in
  (y, ((thursday - days_from_civil y 1 1) / 7) + 1)

(* RFC 3339's date-time: full-date, a separator, partial-time and
   time-offset (§5.6). *)
let of_rfc3339 s =
  let n = String.length s in
  let digit i = i < n && s.[i] >= '0' && s.[i] <= '9' in
  let int i len =
    if i + len > n || not (List.for_all digit (List.init len (( + ) i))) then
      raise Exit
    else int_of_string (String.sub s i len)
  in
  let char i c = if i >= n || s.[i] <> c then raise Exit in
  match
    let y = int 0 4 in
    char 4 '-';
    let m = int 5 2 in
    char 7 '-';
    let d = int 8 2 in
    if n <= 10 || not (List.mem s.[10] [ 'T'; 't'; ' ' ]) then raise Exit;
    let hh = int 11 2 in
    char 13 ':';
    let mm = int 14 2 in
    char 16 ':';
    let ss = int 17 2 in
    (* The fraction's digits, then the offset. *)
    let i = ref 19 in
    if !i < n && s.[!i] = '.' then begin
      incr i;
      if not (digit !i) then raise Exit;
      while digit !i do
        incr i
      done
    end;
    let frac = float_of_string ("0" ^ String.sub s 19 (!i - 19)) in
    let off =
      if !i < n && (s.[!i] = 'Z' || s.[!i] = 'z') && !i + 1 = n then 0
      else if !i + 6 = n && (s.[!i] = '+' || s.[!i] = '-') then begin
        char (!i + 3) ':';
        let oh = int (!i + 1) 2 and om = int (!i + 4) 2 in
        if oh > 23 || om > 59 then raise Exit;
        (if s.[!i] = '-' then -1 else 1) * ((oh * 60) + om)
      end
      else raise Exit
    in
    if m < 1 || m > 12 || d < 1 || d > last_day y m then raise Exit;
    if hh > 23 || mm > 59 || ss > 60 then raise Exit;
    (* A leap second is the last second of a UTC day, the offset taken off
       (RFC 3339 §5.7), and counts as the next minute's 0. *)
    if ss = 60 && (((((hh * 60) + mm - off) mod 1440) + 1440) mod 1440) <> 1439
    then raise Exit;
    let whole =
      (days_from_civil y m d * 86400)
      + (hh * 3600) + (mm * 60) + ss - (off * 60)
    in
    Float.of_int whole +. frac
  with
  | t -> Ok t
  | exception (Exit | Failure _) ->
      Error (Printf.sprintf "%S is not an RFC 3339 date-time" s)
