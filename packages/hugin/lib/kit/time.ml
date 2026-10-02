(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Instants *)

type t = { sec : int64; nsec : int }
type resolution = S | Ms | Us | Ns

let ns_per_s = 1_000_000_000
let s_per_day = 86_400

let per_second = function
  | S -> 1
  | Ms -> 1_000
  | Us -> 1_000_000
  | Ns -> ns_per_s

let epoch = { sec = 0L; nsec = 0 }

exception Unrepresentable

(* Overflow-checked int64 arithmetic, raising [Unrepresentable]. *)

let add64 a b =
  let s = Int64.add a b in
  (* The sum overflowed iff [a] and [b] have one sign and [s] the other. *)
  if Int64.compare (Int64.logand (Int64.logxor a s) (Int64.logxor b s)) 0L < 0
  then raise_notrace Unrepresentable
  else s

let mul64 a k =
  (* [k > 0]. *)
  let k = Int64.of_int k in
  if
    Int64.compare a (Int64.div Int64.max_int k) > 0
    || Int64.compare a (Int64.div Int64.min_int k) < 0
  then raise_notrace Unrepresentable
  else Int64.mul a k

(* [split64 n k] is [(q, r)] with [n = q k + r] and [r] in \[[0];[k - 1]\]. *)
let split64 n k =
  let k = Int64.of_int k in
  let q = Int64.div n k and r = Int64.rem n k in
  if Int64.compare r 0L < 0 then (Int64.pred q, Int64.to_int (Int64.add r k))
  else (q, Int64.to_int r)

(* [mul_add64 a k b] is [a × k + b] for [k > 0]. With [b = q k + r], it is [(a +
   q) × k + r], or [(a + q + 1) × k + (r - k)] below zero, whose product fits
   whenever the result does. *)
let mul_add64 a k b =
  let q, r = split64 (Int64.of_int b) k in
  let a = add64 a q in
  if Int64.compare a 0L >= 0 then add64 (mul64 a k) (Int64.of_int r)
  else add64 (mul64 (Int64.succ a) k) (Int64.of_int (r - k))

(* [fdiv a b] and [fmod a b] are the floor division of [a] by [b > 0] and its
   remainder, in \[[0];[b - 1]\]. *)
let fdiv a b = if a >= 0 then a / b else -((b - 1 - a) / b)
let fmod a b = a - (b * fdiv a b)

let v r n =
  let k = per_second r in
  let sec, rem = split64 n k in
  { sec; nsec = rem * (ns_per_s / k) }

let to_int64 r t =
  let k = per_second r in
  match mul_add64 t.sec k (t.nsec / (ns_per_s / k)) with
  | n -> Some n
  | exception Unrepresentable -> None

(* [add_ns t s ns] is [t] moved by [s] seconds and [ns] nanoseconds, [ns] in
   \[[0];[10^9 - 1]\]. *)
let add_ns t s ns =
  let nsec = t.nsec + ns in
  if nsec >= ns_per_s then
    { sec = add64 (add64 t.sec s) 1L; nsec = nsec - ns_per_s }
  else { sec = add64 t.sec s; nsec }

(* Civil dates and times *)

type tz_offset_s = int
type date = int * int * int
type time = int * int * int

let check_tz fn tz =
  if tz <= -s_per_day || tz >= s_per_day then
    invalid_arg
      (Printf.sprintf "Time.%s: offset %d s not in ]-86400;86400[" fn tz)

let is_leap y = (y mod 4 = 0 && y mod 100 <> 0) || y mod 400 = 0

let days_in_month y m =
  match m with
  | 2 -> if is_leap y then 29 else 28
  | 4 | 6 | 9 | 11 -> 30
  | _ -> 31

(* Howard Hinnant's [days_from_civil] and [civil_from_days], days counted from
   1970-01-01. *)
let days_from_civil y m d =
  let y = if m <= 2 then y - 1 else y in
  let era = fdiv y 400 in
  let yoe = y - (era * 400) in
  let doy = (((153 * ((m + 9) mod 12)) + 2) / 5) + d - 1 in
  let doe = (yoe * 365) + (yoe / 4) - (yoe / 100) + doy in
  (era * 146_097) + doe - 719_468

let civil_from_days z =
  let z = z + 719_468 in
  let era = fdiv z 146_097 in
  let doe = z - (era * 146_097) in
  let yoe = (doe - (doe / 1460) + (doe / 36_524) - (doe / 146_096)) / 365 in
  let doy = doe - ((365 * yoe) + (yoe / 4) - (yoe / 100)) in
  let mp = ((5 * doy) + 2) / 153 in
  let d = doy - (((153 * mp) + 2) / 5) + 1 in
  let m = if mp < 10 then mp + 3 else mp - 9 in
  ((yoe + (era * 400) + if m <= 2 then 1 else 0), m, d)

(* Instants are representable within about 2.9 × 10^11 years of 1970. Years
   beyond [max_year] are not, which [check_year] tells before their day counts
   overflow an [int]; years closer to the bound are refused by the int64
   arithmetic of seconds. *)
let max_year = 300_000_000_000

let check_year y =
  if y > max_year || y < -max_year then raise_notrace Unrepresentable

(* [instant ~tz days s] is the instant [s] seconds into day [days] at offset
   [tz]. *)
let instant ~tz days s =
  { sec = mul_add64 (Int64.of_int days) s_per_day (s - tz); nsec = 0 }

(* [local ~tz t] is the day and the second of that day of [t] at [tz]. *)
let local ~tz t =
  let days, s = split64 t.sec s_per_day in
  let s = s + tz in
  (Int64.to_int days + fdiv s s_per_day, fmod s s_per_day)

let of_date_time ?(tz_offset_s = 0) ((y, m, d), (hh, mm, ss)) =
  check_tz "of_date_time" tz_offset_s;
  if m < 1 || m > 12 || d < 1 || d > days_in_month y m then
    invalid_arg
      (Printf.sprintf "Time.of_date_time: %d-%d-%d is not a date" y m d);
  if hh < 0 || hh > 23 || mm < 0 || mm > 59 || ss < 0 || ss > 59 then
    invalid_arg
      (Printf.sprintf "Time.of_date_time: %d:%d:%d is not a time" hh mm ss);
  match
    check_year y;
    instant ~tz:tz_offset_s (days_from_civil y m d)
      ((hh * 3600) + (mm * 60) + ss)
  with
  | t -> t
  | exception Unrepresentable ->
      invalid_arg "Time.of_date_time: the instant is not representable"

let of_date ?tz_offset_s d = of_date_time ?tz_offset_s (d, (0, 0, 0))

let to_date_time ?(tz_offset_s = 0) t =
  check_tz "to_date_time" tz_offset_s;
  let days, s = local ~tz:tz_offset_s t in
  (civil_from_days days, (s / 3600, s / 60 mod 60, s mod 60))

(* Intervals *)

(* An interval of fixed duration has the boundaries whose local time in
   nanoseconds is [phase] modulo [period], [phase] in \[[0];[period - 1]\];
   months and years are counted as their units' indices. [Months k] never has
   [k] a multiple of 12, which is [Years]. *)
type interval =
  | Fixed of { period : int; phase : int }
  | Months of int
  | Years of int

let ns_per_day = s_per_day * ns_per_s

let fixed fn unit k =
  if k < 1 then invalid_arg (Printf.sprintf "Time.%s: stride %d below 1" fn k);
  if k > max_int / unit then
    invalid_arg
      (Printf.sprintf "Time.%s: stride %d spans more than 2^62 ns" fn k);
  Fixed { period = k * unit; phase = 0 }

let nanoseconds k = fixed "nanoseconds" 1 k
let microseconds k = fixed "microseconds" 1_000 k
let milliseconds k = fixed "milliseconds" 1_000_000 k
let seconds k = fixed "seconds" ns_per_s k
let minutes k = fixed "minutes" (60 * ns_per_s) k
let hours k = fixed "hours" (3600 * ns_per_s) k
let days k = fixed "days" ns_per_day k

let weeks k =
  match fixed "weeks" (7 * ns_per_day) k with
  | Fixed { period; _ } -> Fixed { period; phase = period - (3 * ns_per_day) }
  | Months _ | Years _ -> assert false

let months k =
  if k < 1 then invalid_arg (Printf.sprintf "Time.months: stride %d below 1" k);
  if k mod 12 = 0 then Years (k / 12) else Months k

let years k =
  if k < 1 then invalid_arg (Printf.sprintf "Time.years: stride %d below 1" k);
  Years k

(* [addmod a b p] is [(a + b) mod p] for [a] and [b] in \[[0];[p - 1]\], [p <=
   2^62]: the sum may wrap around [max_int], and subtracting [p] wraps it
   back. *)
let addmod a b p =
  let s = a + b in
  if s < 0 || s >= p then s - p else s

let rec mulmod a b p =
  if b = 0 then 0
  else
    let h = mulmod (addmod a a p) (b lsr 1) p in
    if b land 1 = 1 then addmod h a p else h

(* [offset ~tz period phase t] is the nanoseconds by which [t] follows the
   latest boundary of [Fixed { period; phase }] at [tz], in \[[0];[period -
   1]\]. *)
let offset ~tz period phase t =
  let _, s = split64 t.sec period in
  let x = mulmod s (ns_per_s mod period) period in
  let c = fmod ((tz * ns_per_s) + t.nsec) period in
  addmod x (fmod (c - phase) period) period

(* [sub_ns t r] is [t] moved back by [r >= 0] nanoseconds. *)
let sub_ns t r =
  let s = r / ns_per_s and ns = r mod ns_per_s in
  if ns = 0 then { t with sec = add64 t.sec (Int64.of_int (-s)) }
  else add_ns t (Int64.of_int (-s - 1)) (ns_per_s - ns)

(* [month_index ~tz t] is [12 y + m - 1] for the month [m] of year [y] of [t] at
   [tz], and whether [t] starts that month. *)
let month_index ~tz t =
  let days, s = local ~tz t in
  let y, m, d = civil_from_days days in
  ((12 * y) + m - 1, d = 1 && s = 0 && t.nsec = 0)

let month_start ~tz i =
  let y = fdiv i 12 in
  check_year y;
  instant ~tz (days_from_civil y (fmod i 12 + 1) 1) 0

let year_start ~tz y =
  check_year y;
  instant ~tz (days_from_civil y 1 1) 0

let unrepresentable fn =
  invalid_arg (Printf.sprintf "Time.%s: the boundary is not representable" fn)

let floor ?(tz_offset_s = 0) i t =
  check_tz "floor" tz_offset_s;
  let tz = tz_offset_s in
  match
    match i with
    | Fixed { period; phase } -> sub_ns t (offset ~tz period phase t)
    | Months k ->
        let m, _ = month_index ~tz t in
        month_start ~tz (k * fdiv m k)
    | Years k ->
        let m, _ = month_index ~tz t in
        year_start ~tz (k * fdiv (fdiv m 12) k)
  with
  | b -> b
  | exception Unrepresentable -> unrepresentable "floor"

let ceil ?(tz_offset_s = 0) i t =
  check_tz "ceil" tz_offset_s;
  let tz = tz_offset_s in
  match
    match i with
    | Fixed { period; phase } ->
        let r = offset ~tz period phase t in
        if r = 0 then t
        else
          let r = period - r in
          add_ns t (Int64.of_int (r / ns_per_s)) (r mod ns_per_s)
    | Months k ->
        let m, start = month_index ~tz t in
        if start && fmod m k = 0 then t else month_start ~tz (k * (fdiv m k + 1))
    | Years k ->
        let m, start = month_index ~tz t in
        let y = fdiv m 12 in
        if start && fmod m 12 = 0 && fmod y k = 0 then t
        else year_start ~tz (k * (fdiv y k + 1))
  with
  | b -> b
  | exception Unrepresentable -> unrepresentable "ceil"

(* [mul_ns n period] is [n × period] nanoseconds as seconds and nanoseconds in
   \[[0];[10^9 - 1]\]. *)
let mul_ns n period =
  let ps = period / ns_per_s and pn = period mod ns_per_s in
  let q = n / ns_per_s and r = n mod ns_per_s in
  let rn = r * pn in
  let s = if ps = 0 then 0L else mul64 (Int64.of_int n) ps in
  (add64 s (Int64.of_int ((q * pn) + fdiv rn ns_per_s)), fmod rn ns_per_s)

(* [add_months ~tz months n t] moves the civil date of [t] at [tz] by [n]
   strides of [months] months, clamping the day. *)
let add_months ~tz months n t =
  let limit = 24 * max_year / months in
  if n > limit || n < -limit then raise_notrace Unrepresentable;
  let days, s = local ~tz t in
  let y, m, d = civil_from_days days in
  let i = (12 * y) + m - 1 + (n * months) in
  let y = fdiv i 12 and m = fmod i 12 + 1 in
  let d = Int.min d (days_in_month y m) in
  { (instant ~tz (days_from_civil y m d) s) with nsec = t.nsec }

let add ?(tz_offset_s = 0) i n t =
  check_tz "add" tz_offset_s;
  let tz = tz_offset_s in
  match
    match i with
    | Fixed { period; _ } ->
        let s, ns = mul_ns n period in
        add_ns t s ns
    | Months k -> add_months ~tz k n t
    | Years k ->
        if n = 0 then t
        else if k > 2 * max_year then raise_notrace Unrepresentable
        else add_months ~tz (12 * k) n t
  with
  | t -> t
  | exception Unrepresentable ->
      invalid_arg "Time.add: the result is not representable"

let compare t t' =
  match Int64.compare t.sec t'.sec with
  | 0 -> Int.compare t.nsec t'.nsec
  | c -> c

let equal t t' = Int64.equal t.sec t'.sec && t.nsec = t'.nsec

let range ?(tz_offset_s = 0) i t t' =
  check_tz "range" tz_offset_s;
  if compare t' t < 0 then [||]
  else
    match ceil ~tz_offset_s i t with
    | exception Invalid_argument _ -> [||]
    | first ->
        let rec loop acc b =
          if compare b t' > 0 then acc
          else
            match add ~tz_offset_s i 1 b with
            | next -> loop (b :: acc) next
            | exception Invalid_argument _ -> b :: acc
        in
        Array.of_list (List.rev (loop [] first))

let equal_interval i i' =
  match (i, i') with
  | Fixed f, Fixed f' -> f.period = f'.period && f.phase = f'.phase
  | Months k, Months k' | Years k, Years k' -> k = k'
  | (Fixed _ | Months _ | Years _), _ -> false

let pp_interval ppf i =
  let pp n unit =
    Format.fprintf ppf "%d %s%s" n unit (if n = 1 then "" else "s")
  in
  match i with
  | Months k -> pp k "month"
  | Years k -> pp k "year"
  | Fixed { period; phase } when phase <> 0 ->
      pp (period / (7 * ns_per_day)) "week"
  | Fixed { period; _ } ->
      let units =
        [
          (ns_per_day, "day");
          (3600 * ns_per_s, "hour");
          (60 * ns_per_s, "minute");
          (ns_per_s, "second");
          (1_000_000, "millisecond");
          (1_000, "microsecond");
        ]
      in
      let rec find = function
        | (u, name) :: us ->
            if period mod u = 0 then pp (period / u) name else find us
        | [] -> pp period "nanosecond"
      in
      find units

(* Formatting *)

let pp ppf t =
  let (y, m, d), (hh, mm, ss) = to_date_time t in
  if y >= 0 && y <= 9999 then Format.fprintf ppf "%04d" y
  else Format.fprintf ppf "%c%04d" (if y < 0 then '-' else '+') (Int.abs y);
  Format.fprintf ppf "-%02d-%02dT%02d:%02d:%02d" m d hh mm ss;
  if t.nsec <> 0 then begin
    let f = Printf.sprintf "%09d" t.nsec in
    let n = ref 9 in
    while f.[!n - 1] = '0' do
      decr n
    done;
    Format.fprintf ppf ".%s" (String.sub f 0 !n)
  end;
  Format.pp_print_char ppf 'Z'
