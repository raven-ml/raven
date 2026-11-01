(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir_units.Quantity
module Unit = Ymir_units.Unit

let strf = Printf.sprintf

type !'a uniform
type utc
type tai = [ `Tai ] uniform
type tt = [ `Tt ] uniform
type tdb = [ `Tdb ] uniform

type _ scale =
  | Utc : utc scale
  | Tai : tai scale
  | Tt : tt scale
  | Tdb : tdb scale

let utc = Utc
let tai = Tai
let tt = Tt
let tdb = Tdb

let name : type s. s scale -> string = function
  | Utc -> "utc"
  | Tai -> "tai"
  | Tt -> "tt"
  | Tdb -> "tdb"

type 'd days = (int64, Nx.int64_elt, 'd) Nx.t
type 'd seconds = (float, Nx.float64_elt, 'd) Nx.t
type ('s, 'd) t = { scale : 's scale; day : 'd days; second : 'd seconds }

let scale t = t.scale
let day t = t.day
let second t = Quantity.v Unit.second t.second

(* Constants *)

let day_s = 86400.
let mjd_j2000 = 51544.5
let mjd_jd = 2400000.5
let tt_minus_tai = 32.184
let millennium_days = 365250.

(* [jd_whole] is the whole part of JD = MJD + 2400000.5. *)
let jd_whole = 2400000.

(* Messages *)

(* [at i] places a message at the batch index [i]: [" at [3; 1]"], or nothing
   for a single instant. *)
let at i =
  if Array.length i = 0 then ""
  else
    strf " at [%s]"
      (String.concat "; " (Array.to_list (Array.map string_of_int i)))

(* [num x] is {!Transform.num}, which lies above Time: the shortest decimal
   that reads as [x], the same text on every system. *)
let num x = Format.asprintf "%a" (Nx.Dtype.pp_value Nx.float64) x

(* Calendar *)

(* Days from civil, after Hinnant's "chrono-Compatible Low-Level Date
   Algorithms": the proleptic Gregorian calendar counted in eras of 400 years,
   146097 days, from a year that begins in March so that the leap day ends it.
   1970-01-01 is MJD 40587. *)
let mjd_of_date (y, m, d) =
  let y = if m <= 2 then y - 1 else y in
  let era = (if y >= 0 then y else y - 399) / 400 in
  let yoe = y - (era * 400) in
  let doy = ((153 * (m + if m > 2 then -3 else 9)) + 2) / 5 + d - 1 in
  let doe = (yoe * 365) + (yoe / 4) - (yoe / 100) + doy in
  (era * 146097) + doe - 719468 + 40587

let date_of_mjd n =
  let z = n - 40587 + 719468 in
  let era = (if z >= 0 then z else z - 146096) / 146097 in
  let doe = z - (era * 146097) in
  let yoe = (doe - (doe / 1460) + (doe / 36524) - (doe / 146096)) / 365 in
  let doy = doe - ((365 * yoe) + (yoe / 4) - (yoe / 100)) in
  let mp = ((5 * doy) + 2) / 153 in
  let d = doy - (((153 * mp) + 2) / 5) + 1 in
  let m = if mp < 10 then mp + 3 else mp - 9 in
  ((yoe + (era * 400) + if m <= 2 then 1 else 0), m, d)

let days_in_month y m =
  match m with
  | 2 -> if (y mod 4 = 0 && y mod 100 <> 0) || y mod 400 = 0 then 29 else 28
  | 4 | 6 | 9 | 11 -> 30
  | _ -> 31

(* TAI - UTC *)

(* From 0h UTC of day [utc_starts.(i)] until the next start, TAI - UTC is
   [offset + (MJD - reference) × rate], MJD with its fraction of the day. A day
   [d] of interval [i] begins at TAI [d + dat0 d] and its UTC second lasts
   [1 + rate / 86400] SI seconds; [dat0 (d + 1) - dat0 d - rate] is the jump at
   its end, a leap second from 1972, so the UTC day lasts [86400 + jump] UTC
   seconds. This is the convention of the IAU SOFA library's UTC functions,
   which hold a jump before 1972 as they hold a leap second. *)

let utc_first = Time_tables.utc_starts.(0)

type 'd intervals = {
  starts : 'd days;
  offsets : 'd seconds;
  references : 'd seconds;
  rates : 'd seconds;
}

let intervals on =
  let n = Array.length Time_tables.utc_starts in
  let vector xs = Nx.place on (Nx.create Nx.float64 [| n |] xs) in
  {
    starts =
      Nx.place on
        (Nx.create Nx.int64 [| n |]
           (Array.map Int64.of_int Time_tables.utc_starts));
    offsets = vector Time_tables.utc_offsets;
    references = vector Time_tables.utc_references;
    rates = vector Time_tables.utc_rates;
  }

(* [interval tables d] is the index of [d]'s interval, [-1] before the first
   start. *)
let interval tables d =
  let after = Nx.greater_equal (Nx.unsqueeze ~axes:[ -1 ] d) tables.starts in
  Nx.sub_s (Nx.sum ~axes:[ -1 ] (Nx.cast Nx.int64 after)) 1L

(* [utc_day tables d] is [(dat0, rate, length)] of UTC day [d]: TAI - UTC at its
   0h, its rate in seconds a day, and its length in UTC seconds. *)
let utc_day tables d =
  let dat0 d =
    let i = interval tables d in
    let read v = Nx.take i v in
    let df = Nx.cast Nx.float64 d in
    (Nx.add (read tables.offsets)
       (Nx.mul (Nx.sub df (read tables.references)) (read tables.rates)),
     read tables.rates)
  in
  let today, rate = dat0 d in
  let tomorrow, _ = dat0 (Nx.add_s d 1L) in
  (today, rate, Nx.add_s (Nx.sub (Nx.sub tomorrow today) rate) day_s)

let utc_defined tables d = Nx.greater_equal_s (interval tables d) 0L

let leap_seconds_known_until =
  {
    scale = Utc;
    day = Nx.scalar Nx.int64 (Int64.of_int Time_tables.utc_expires);
    second = Nx.scalar Nx.float64 0.;
  }

(* Normalising *)

(* [two_sum a b] is [(h, l)] with [h = a + b] rounded and [h + l = a + b]
   exactly (Knuth). Its [l] carries no derivative. *)
let two_sum a b =
  let h = Nx.add a b in
  let b' = Nx.sub h a in
  (h, Nx.add (Nx.sub a (Nx.sub h b')) (Nx.sub b b'))

(* [carry d a b] is the instant [a + b] seconds after 0h of day [d], [a + b]
   within a few days of it, as a day and the seconds in [[0, 86400)]. The sum
   and the whole days taken off it are kept exact, so the seconds round once;
   a sum that rounds to the day's end is 0h of the next day. A NaN second keeps
   day [d]. *)
let carry d a b =
  let hi, lo = two_sum a b in
  let n = Nx.floor (Nx.div_s hi day_s) in
  let n = Nx.where (Nx.isnan n) (Nx.zeros_like n) n in
  let y, e = two_sum hi (Nx.mul_s n (-.day_s)) in
  let s = Nx.add y (Nx.add e lo) in
  let fix cond s' dn (s, n) = (Nx.where cond s' s, Nx.where cond (dn n) n) in
  let up (s, n) =
    fix (Nx.greater_equal_s s day_s) (Nx.sub_s s day_s)
      (fun n -> Nx.add_s n 1.)
      (s, n)
  in
  let down (s, n) =
    fix (Nx.less_s s 0.) (Nx.add_s s day_s) (fun n -> Nx.sub_s n 1.) (s, n)
  in
  (* [n] comes from a rounded quotient and may be one off; a negative second
     that gains a day may round up to it. *)
  let s, n = up (down (up (s, n))) in
  (Nx.add d (Nx.cast Nx.int64 n), s)

let shift scale t dx =
  let day, second = carry t.day t.second dx in
  { scale; day; second }

let shift_s scale t dx = shift scale t (Nx.scalar_like t.second dx)

(* Steps between scales *)

let tai_of_utc t =
  let tables = intervals (Nx.beside t.second) in
  let dat0, rate, _ = utc_day tables t.day in
  (* s (1 + rate / 86400) + dat0: SI seconds since the UTC day's 0h, which is
     TAI 0h of the day plus dat0. From 1972 the rate is 0 and the sum adds
     whole seconds. *)
  let si = Nx.fma t.second (Nx.div_s rate day_s) t.second in
  let day, second = carry t.day si dat0 in
  { scale = Tai; day; second }

(* A TAI computed from UTC's first instant, through TT or TDB, comes back
   within this many seconds of it, as {!Ymir.Time.convert} states. Such a TAI
   before UTC's start is UTC's first instant. *)
let utc_start_slack = 4e-11

let utc_of_tai t =
  let tables = intervals (Nx.beside t.second) in
  let yesterday = Nx.sub_s t.day 1L in
  let dat0, rate, _ = utc_day tables t.day in
  let dat0', rate', length' = utc_day tables yesterday in
  let per rate x = Nx.div x (Nx.add_s (Nx.div_s rate day_s) 1.) in
  (* The UTC day that holds the instant is the TAI day, or the one before when
     the instant precedes that day's UTC 0h. From 1972 each subtracts whole
     seconds and rounds at most once. *)
  let today = per rate (Nx.sub t.second dat0) in
  let before = per rate' (Nx.add t.second (Nx.rsub_s day_s dat0')) in
  let late = Nx.logical_or (Nx.greater_equal_s today 0.) (Nx.isnan today) in
  let defined = utc_defined tables t.day in
  let defined' = utc_defined tables yesterday in
  (* Before 1972 a jump that shortens a day leaves a gap of TAI before the
     next UTC 0h, of [jump × rate / 86400] s, and a seconds that rounds up to
     the day's length is no instant of it: both are the next day's 0h. *)
  let within = Nx.logical_and defined' (Nx.less before length') in
  let early = Nx.logical_and (Nx.logical_not late) within in
  let at_start =
    Nx.logical_and (Nx.logical_not late) (Nx.logical_not within)
  in
  let valid =
    Nx.logical_or
      (Nx.logical_and late defined)
      (Nx.logical_or early
         (Nx.logical_and at_start
            (Nx.logical_and defined
               (Nx.logical_or defined'
                  (Nx.greater_equal_s today (-.utc_start_slack))))))
  in
  Nx.check Nx.Ptree.unit valid () (fun i () ->
      Invalid_argument
        (strf "Time.convert: TAI%s precedes UTC, which begins at 1961-01-01"
           (at i)));
  let second =
    Nx.where late today (Nx.where early before (Nx.zeros_like today))
  in
  { scale = Utc; day = Nx.where early yesterday t.day; second }

let tt_of_tai t = shift_s Tt t tt_minus_tai
let tai_of_tt t = shift_s Tai t (-.tt_minus_tai)

(* TDB - TT *)

(* [millennia t] is Julian millennia since J2000.0 on [t]'s scale. *)
let millennia t =
  let days = Nx.sub_s (Nx.cast Nx.float64 t.day) mjd_j2000 in
  Nx.div_s (Nx.add days (Nx.div_s t.second day_s)) millennium_days

(* Terms per block of the series: each block evaluates [block] terms per
   instant at once, which bounds the intermediates at [block] values per
   instant. *)
let block = 32

(* [series tm terms] is [Σ a sin (f T + p)] over the triples [terms]. *)
let series tm terms =
  let n = Array.length terms / 3 in
  let on = Nx.beside tm in
  let tm = Nx.unsqueeze ~axes:[ -1 ] tm in
  let rec go acc j =
    if j >= n then acc
    else
      let k = min block (n - j) in
      let column c =
        Nx.place on
          (Nx.create Nx.float64 [| k |]
             (Array.init k (fun i -> terms.((3 * (j + i)) + c))))
      in
      let a = column 0 and f = column 1 and p = column 2 in
      let terms = Nx.mul a (Nx.sin (Nx.add (Nx.mul f tm) p)) in
      go (Nx.add acc (Nx.sum ~axes:[ -1 ] terms)) (j + k)
  in
  go (Nx.zeros_like (Nx.squeeze ~axes:[ -1 ] tm)) 0

(* [tdb_minus_tt tm] is TDB - TT at the geocentre in seconds, [tm] TDB in Julian
   millennia since J2000.0: the series of the IAU SOFA library's iauDtdb, as
   ERFA's eraDtdb holds it, without its topocentric terms. *)
let tdb_minus_tt tm =
  let powers = Array.map (series tm) Time_tables.fairhead in
  let horner =
    Array.fold_right (fun w acc -> Nx.add (Nx.mul acc tm) w) powers
      (Nx.zeros_like tm)
  in
  let jpl =
    Nx.add (series tm Time_tables.jpl)
      (Nx.mul_s (Nx.square tm) Time_tables.jpl_t2)
  in
  Nx.add horner jpl

let tt_of_tdb t = shift Tt t (Nx.neg (tdb_minus_tt (millennia t)))

(* The model's argument is TDB, so TT's TDB is the fixed point of
   [tdb = tt + f tdb]: [f]'s rate is below 4e-10, and the second step leaves
   an error below 1e-21 s. *)
let tdb_of_tt t =
  let tm = millennia t in
  let first = tdb_minus_tt tm in
  let tm' = Nx.add tm (Nx.div_s first (day_s *. millennium_days)) in
  shift Tdb t (tdb_minus_tt tm')

(* Converting *)

let rec convert : type a b d. b scale -> (a, d) t -> (b, d) t =
 fun target t ->
  match (t.scale, target) with
  | Utc, Utc -> t
  | Tai, Tai -> t
  | Tt, Tt -> t
  | Tdb, Tdb -> t
  | Utc, _ -> convert target (tai_of_utc t)
  | Tai, Utc -> utc_of_tai t
  | Tai, _ -> convert target (tt_of_tai t)
  | Tt, (Utc | Tai) -> convert target (tai_of_tt t)
  | Tt, Tdb -> tdb_of_tt t
  | Tdb, _ -> convert target (tt_of_tdb t)

(* Constructors *)

(* [length scale like d] is the length of day [d] on [scale], in its
   seconds. *)
let length : type s. s scale -> 'd seconds -> 'd days -> 'd seconds =
 fun scale like d ->
  match scale with
  | Utc ->
      let _, _, length = utc_day (intervals (Nx.beside like)) d in
      length
  | Tai | Tt | Tdb -> Nx.full_like like day_s

(* Days stay within 2^53 of MJD 0, about 2.5e13 years, where float64 holds
   each one and its sum with a fraction keeps its digits. *)
let day_bound = 0x1p53

(* [check_days fn within] raises where [within], whether a day is within the
   bound, is false. *)
let check_days fn within =
  Nx.check Nx.Ptree.unit within ()
    (fun i () ->
      Invalid_argument
        (strf "Time.%s: the day%s is more than 2^53 days from MJD 0" fn (at i)))

let check_within (type s) fn (scale : s scale) day second =
  let shape = Nx.broadcast_shapes [ Nx.shape day; Nx.shape second ] in
  let day = Nx.broadcast_to shape day
  and second = Nx.broadcast_to shape second in
  check_days fn
    (Nx.less_equal_s (Nx.abs day) (Int64.of_float day_bound));
  Nx.check Nx.Ptree.unit
    (Nx.logical_not (Nx.isinf second))
    ()
    (fun i () ->
      Invalid_argument (strf "Time.%s: second%s is infinite" fn (at i)));
  (match scale with
  | Utc ->
      let tables = intervals (Nx.beside second) in
      Nx.check Nx.Ptree.tensor (utc_defined tables day) day (fun i d ->
          Invalid_argument
            (strf
               "Time.%s: UTC day %Ld%s precedes UTC, which begins at MJD %d \
                (1961-01-01)"
               fn (Nx.item [] d) (at i) utc_first))
  | Tai | Tt | Tdb -> ());
  let length = length scale second day in
  let ok =
    Nx.logical_and (Nx.greater_equal_s second 0.) (Nx.less second length)
  in
  Nx.check
    Nx.Ptree.(pair tensor (pair tensor tensor))
    (Nx.logical_or ok (Nx.isnan second))
    (day, (second, length))
    (fun i (d, (s, l)) ->
      Invalid_argument
        (strf "Time.%s: second %s%s is outside [0, %s) of %s day %Ld" fn
           (num (Nx.item [] s)) (at i)
           (num (Nx.item [] l)) (name scale) (Nx.item [] d)));
  (day, second)

let v scale ~day ~second =
  let second = Quantity.value Unit.second second in
  let day, second = check_within "v" scale day second in
  { scale; day; second }

let check_finite fn what x =
  Nx.check Nx.Ptree.unit (Nx.isfinite x) () (fun i () ->
      Invalid_argument (strf "Time.%s: %s%s is not finite" fn what (at i)))

let of_jd scale j1 j2 =
  check_finite "of_jd" "the first part" j1;
  check_finite "of_jd" "the second part" j2;
  let d1 = Nx.floor j1 and d2 = Nx.floor j2 in
  let f1 = Nx.sub j1 d1 and f2 = Nx.sub j2 d2 in
  (* MJD = JD - 2400000.5 = (d1 + d2 - 2400000) + (f1 - 1/2 + f2): the days
     and [f1 - 1/2] are exact, and the seconds [(f1 - 1/2) 86400 + f2 86400],
     below 2^17 in magnitude, round twice. *)
  let days = Nx.sub_s (Nx.add d1 d2) jd_whole in
  check_days "of_jd" (Nx.less_equal_s (Nx.abs days) day_bound);
  let x =
    Nx.fma (Nx.sub_s f1 0.5) (Nx.scalar_like f1 day_s) (Nx.mul_s f2 day_s)
  in
  let day, second = carry (Nx.cast Nx.int64 days) x (Nx.zeros_like x) in
  { scale; day; second }

let jd t =
  ( Nx.add_s (Nx.cast Nx.float64 t.day) mjd_jd,
    Nx.div_s t.second day_s )

let clock_limit fn (h, mi, s) =
  if h < 0 || h > 23 || mi < 0 || mi > 59 || not (Float.is_finite s) || s < 0.
  then
    invalid_arg
      (strf "Time.%s: %02d:%02d:%s is not a time of day" fn h mi (num s))

let of_date (type s) (scale : s scale) (y, m, d) (h, mi, s) =
  if m < 1 || m > 12 || d < 1 || d > days_in_month y m then
    invalid_arg (strf "Time.of_date: %d-%02d-%02d is not a date" y m d);
  clock_limit "of_date" (h, mi, s);
  (* Only a UTC day's last minute may pass 60 s: its length decides. *)
  if s >= 60. && not (h = 23 && mi = 59) then
    invalid_arg
      (strf "Time.of_date: %02d:%02d:%s is not a time of day" h mi (num s));
  (* A year past 2^53 is past the day bound, and the day count of a smaller
     one does not overflow. *)
  let past =
    strf "Time.of_date: %d-%02d-%02d is more than 2^53 days from MJD 0"
  in
  if abs y > 1 lsl 53 then invalid_arg (past y m d);
  let n = mjd_of_date (y, m, d) in
  if Float.abs (float_of_int n) > day_bound then invalid_arg (past y m d);
  let second = (3600. *. float_of_int h) +. (60. *. float_of_int mi) +. s in
  let day = Nx.scalar Nx.int64 (Int64.of_int n) in
  (match scale with
  | Utc when n < utc_first ->
      invalid_arg
        (strf
           "Time.of_date: %d-%02d-%02d precedes UTC, which begins at \
            1961-01-01"
           y m d)
  | _ -> ());
  let length = Nx.item [] (length scale (Nx.scalar Nx.float64 0.) day) in
  (* The clock is judged by its last minute, [length - 86340] s long and exact;
     its seconds since 0h may then round up to the day's end, the next day's
     0h. *)
  if h = 23 && mi = 59 && s >= length -. 86340. then
    invalid_arg
      (strf
         "Time.of_date: %d-%02d-%02d %02d:%02d:%s is past the end of the %s \
          day, %s s long"
         y m d h mi (num s) (name scale) (num length));
  if second >= length then
    {
      scale;
      day = Nx.scalar Nx.int64 (Int64.of_int (n + 1));
      second = Nx.scalar Nx.float64 0.;
    }
  else { scale; day; second = Nx.scalar Nx.float64 second }

let date t =
  if Nx.ndim t.second <> 0 then
    invalid_arg
      (strf "Time.date: expected a single instant, got shape [%s]"
         (String.concat "; "
            (Array.to_list (Array.map string_of_int (Nx.shape t.second)))));
  let s = Nx.item [] t.second in
  if Float.is_nan s then invalid_arg "Time.date: the instant is NaN";
  (* [whole x u] is the whole number of [u] in [x], at most [top]: a quotient
     that rounds up to the next integer is taken back. Each remainder is exact,
     as [x] is below twice the part taken off. *)
  let whole x u top =
    let n = int_of_float (x /. u) in
    let n = if float_of_int n *. u > x then n - 1 else n in
    min top n
  in
  let h = whole s 3600. 23 in
  let rest = s -. (3600. *. float_of_int h) in
  let mi = whole rest 60. 59 in
  let sec = rest -. (60. *. float_of_int mi) in
  (date_of_mjd (Int64.to_int (Nx.item [] t.day)), (h, mi, sec))

(* Arithmetic *)

let diff a b =
  let days = Nx.cast Nx.float64 (Nx.sub a.day b.day) in
  Quantity.v Unit.second
    (Nx.fma days (Nx.scalar_like days day_s) (Nx.sub a.second b.second))

let add t span =
  let span = Quantity.value Unit.second span in
  check_finite "add" "the span" span;
  let n = Nx.trunc (Nx.div_s span day_s) in
  check_days "add"
    (Nx.less_equal_s (Nx.abs (Nx.add (Nx.cast Nx.float64 t.day) n)) day_bound);
  (* [span - n 86400] is exact: [n] truncates, so the remainder is no larger
     than the span and, within the day bound, a multiple of its ulp. *)
  let rest = Nx.fma (Nx.neg n) (Nx.scalar_like n day_s) span in
  let day = Nx.add t.day (Nx.cast Nx.int64 n) in
  let day, second = carry day t.second rest in
  { t with day; second }

(* Structures *)

let walk c t =
  let open Nx.Ptree.Walk in
  case c (name t.scale);
  let day = field c "day" tensor t.day in
  let second = field c "second" tensor t.second in
  { scale = t.scale; day; second }

let ptree () = Nx.Ptree.v walk
